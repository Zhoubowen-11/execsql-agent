"""Integration tests for the deterministic FSQ Shanghai SQLite build."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from execsql_agent.tools.schema_loader import SchemaLoader
from execsql_agent.tools.sql_executor import SQLExecutor
from scripts.build_fsq_shanghai_db import (
    CANONICAL_DISTRICTS,
    DEFAULT_CATEGORIES,
    DEFAULT_PLACES,
    EXPECTED_PLACES_COUNT,
    build_database,
    canonical_district,
    inspect_parquet,
)


@pytest.fixture(scope="session")
def fsq_database(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the complete 91,770-row database once for the test session."""

    assert DEFAULT_PLACES.is_file()
    assert DEFAULT_CATEGORIES.is_file()
    output = tmp_path_factory.mktemp("fsq") / "shanghai_places.db"
    report = build_database(DEFAULT_PLACES, DEFAULT_CATEGORIES, output)
    assert report["database_size_bytes"] == output.stat().st_size
    return output


def _connect_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def test_real_parquet_schemas_are_recognized() -> None:
    places = inspect_parquet(DEFAULT_PLACES, sample_size=1)
    categories = inspect_parquet(DEFAULT_CATEGORIES, sample_size=1)
    place_types = {
        str(field["name"]): str(field["type"])
        for field in places["schema"]  # type: ignore[union-attr]
    }
    category_names = {
        str(field["name"])
        for field in categories["schema"]  # type: ignore[union-attr]
    }
    assert place_types["fsq_category_ids"] == "VARCHAR[]"
    assert place_types["fsq_category_labels"] == "VARCHAR[]"
    assert {"category_id", "category_level", "category_label"} <= category_names


def test_build_creates_core_tables(fsq_database: Path) -> None:
    assert fsq_database.is_file()
    connection = _connect_read_only(fsq_database)
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    finally:
        connection.close()
    assert {"places", "categories", "place_categories", "data_metadata"} <= tables
    assert "place_unresolved_flags" in tables


def test_place_count_primary_key_and_coordinates(fsq_database: Path) -> None:
    connection = _connect_read_only(fsq_database)
    try:
        count = connection.execute("SELECT COUNT(*) FROM places").fetchone()[0]
        distinct_count = connection.execute(
            "SELECT COUNT(DISTINCT fsq_place_id) FROM places"
        ).fetchone()[0]
        null_coordinates = connection.execute(
            "SELECT COUNT(*) FROM places WHERE latitude IS NULL OR longitude IS NULL"
        ).fetchone()[0]
        non_cn = connection.execute(
            "SELECT COUNT(*) FROM places WHERE country <> 'CN'"
        ).fetchone()[0]
    finally:
        connection.close()
    assert count == EXPECTED_PLACES_COUNT
    assert distinct_count == EXPECTED_PLACES_COUNT
    assert null_coordinates == 0
    assert non_cn == 0


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("Pudong", "Pudong"),
        ("浦东新区", "Pudong"),
        ("Xuhui", "Xuhui"),
        ("徐汇区", "Xuhui"),
        ("Jing'an", "Jing'an"),
        ("Jing’an", "Jing'an"),
        ("Jìng'ān", "Jing'an"),
        ("静安区", "Jing'an"),
    ],
)
def test_district_aliases_map_to_canonical_names(alias: str, expected: str) -> None:
    assert canonical_district(alias, None, "Shanghai") == expected


def test_uncertain_or_conflicting_district_values_remain_null() -> None:
    assert canonical_district("Shanghai", "Shanghai", None) is None
    assert canonical_district("Jinqiao", "Shanghai", None) is None
    assert canonical_district("Pudong", "Shanghai", "Xuhui") is None


def test_database_contains_at_most_sixteen_canonical_districts(
    fsq_database: Path,
) -> None:
    connection = _connect_read_only(fsq_database)
    try:
        values = {
            str(row[0])
            for row in connection.execute(
                """
                SELECT DISTINCT canonical_district
                FROM places
                WHERE canonical_district IS NOT NULL
                """
            )
        }
    finally:
        connection.close()
    assert len(values) <= 16
    assert values <= CANONICAL_DISTRICTS


def test_foreign_keys_and_category_join(fsq_database: Path) -> None:
    connection = _connect_read_only(fsq_database)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        joined = connection.execute(
            """
            SELECT p.fsq_place_id, p.name, c.category_name, pc.is_primary
            FROM places p
            JOIN place_categories pc USING(fsq_place_id)
            JOIN categories c USING(category_id)
            WHERE p.name IS NOT NULL
            ORDER BY p.fsq_place_id, pc.is_primary DESC, c.category_name
            LIMIT 1
            """
        ).fetchone()
    finally:
        connection.close()
    assert foreign_key_errors == []
    assert joined is not None
    assert joined[0] and joined[1] and joined[2]
    assert joined[3] in {0, 1}


def test_database_integrity_and_indexes(fsq_database: Path) -> None:
    connection = _connect_read_only(fsq_database)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        index_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
    finally:
        connection.close()
    assert integrity == "ok"
    assert {
        "idx_places_name",
        "idx_places_locality",
        "idx_places_region",
        "idx_places_canonical_district",
        "idx_places_latitude_longitude",
        "idx_place_categories_category_id",
        "idx_categories_category_name",
    } <= index_names


def test_schema_loader_reads_standard_sqlite_database(fsq_database: Path) -> None:
    schema = SchemaLoader(fsq_database).load()
    tables = {table.name: table for table in schema.tables}
    assert {"places", "categories", "place_categories", "data_metadata"} <= tables.keys()
    assert tables["places"].primary_keys == ["fsq_place_id"]
    assert len(tables["place_categories"].foreign_keys) == 2


def test_sql_executor_reads_and_blocks_writes(fsq_database: Path) -> None:
    executor = SQLExecutor(fsq_database, max_rows=5)
    selected = executor.execute(
        """
        SELECT p.name, c.category_name
        FROM places p
        JOIN place_categories pc USING(fsq_place_id)
        JOIN categories c USING(category_id)
        ORDER BY p.fsq_place_id
        LIMIT 5
        """
    )
    assert selected.execution_success is True
    assert selected.returned_row_count == 5

    for unsafe_sql in (
        "UPDATE places SET name = 'blocked'",
        "DELETE FROM places",
        "DROP TABLE places",
    ):
        blocked = executor.execute(unsafe_sql)
        assert blocked.executed is False
        assert blocked.execution_success is False
