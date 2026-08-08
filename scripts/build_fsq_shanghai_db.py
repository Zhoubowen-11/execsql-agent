"""Build the FSQ Shanghai business SQLite database from cleaned Parquet inputs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import unicodedata
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import duckdb

DEFAULT_PLACES = Path("data/fsq/processed/shanghai_places_filtered.parquet")
DEFAULT_CATEGORIES = Path(
    "data/fsq/raw/release/dt=2026-07-09/categories/parquet/categories_000000.parquet"
)
DEFAULT_OUTPUT = Path("data/fsq/shanghai_places.db")
DEFAULT_REPORT_DIR = Path("data/fsq/reports")
EXPECTED_PLACES_COUNT = 91_770
SNAPSHOT_DATE = "2026-07-09"

DISTRICT_ALIASES: dict[str, tuple[str, ...]] = {
    "Pudong": (
        "Pudong",
        "Pudong District",
        "Pudong New Area",
        "Pudong New District",
        "Pudong Xinqu",
        "Pudongxinqu",
        "浦东",
        "浦东新区",
    ),
    "Huangpu": ("Huangpu", "Huangpu District", "Huangpu Qu", "黄浦", "黄浦区"),
    "Xuhui": ("Xuhui", "Xuhui District", "Xuhui Qu", "徐汇", "徐汇区"),
    "Changning": (
        "Changning",
        "Changning District",
        "Changning Qu",
        "长宁",
        "长宁区",
    ),
    "Jing'an": (
        "Jing'an",
        "Jing’an",
        "Jingan",
        "Jing An",
        "Jing'an District",
        "Jingan District",
        "Jing'an Qu",
        "Jingan Qu",
        "静安",
        "静安区",
    ),
    "Putuo": ("Putuo", "Putuo District", "Putuo Qu", "普陀", "普陀区"),
    "Hongkou": ("Hongkou", "Hongkou District", "Hongkou Qu", "虹口", "虹口区"),
    "Yangpu": ("Yangpu", "Yangpu District", "Yangpu Qu", "杨浦", "杨浦区"),
    "Minhang": ("Minhang", "Minhang District", "Minhang Qu", "闵行", "闵行区"),
    "Baoshan": ("Baoshan", "Baoshan District", "Baoshan Qu", "宝山", "宝山区"),
    "Jiading": ("Jiading", "Jiading District", "Jiading Qu", "嘉定", "嘉定区"),
    "Jinshan": ("Jinshan", "Jinshan District", "Jinshan Qu", "金山", "金山区"),
    "Songjiang": (
        "Songjiang",
        "Songjiang District",
        "Songjiang Qu",
        "松江",
        "松江区",
    ),
    "Qingpu": ("Qingpu", "Qingpu District", "Qingpu Qu", "青浦", "青浦区"),
    "Fengxian": (
        "Fengxian",
        "Fengxian District",
        "Fengxian Qu",
        "奉贤",
        "奉贤区",
    ),
    "Chongming": (
        "Chongming",
        "Chongming District",
        "Chongming Qu",
        "崇明",
        "崇明区",
    ),
}

SCHEMA_SQL = """
CREATE TABLE places (
    fsq_place_id TEXT PRIMARY KEY NOT NULL,
    name TEXT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    address TEXT,
    locality TEXT,
    region TEXT,
    admin_region TEXT,
    canonical_district TEXT CHECK (
        canonical_district IS NULL OR canonical_district IN (
            'Pudong', 'Huangpu', 'Xuhui', 'Changning', 'Jing''an', 'Putuo',
            'Hongkou', 'Yangpu', 'Minhang', 'Baoshan', 'Jiading', 'Jinshan',
            'Songjiang', 'Qingpu', 'Fengxian', 'Chongming'
        )
    ),
    postcode TEXT,
    post_town TEXT,
    po_box TEXT,
    country TEXT NOT NULL,
    date_created TEXT,
    date_refreshed TEXT,
    date_closed TEXT,
    website TEXT,
    tel TEXT,
    email TEXT,
    facebook_id INTEGER,
    instagram TEXT,
    twitter TEXT,
    placemaker_url TEXT,
    snapshot_date TEXT NOT NULL
);

CREATE TABLE categories (
    category_id TEXT PRIMARY KEY NOT NULL,
    category_level INTEGER NOT NULL CHECK (category_level BETWEEN 1 AND 6),
    category_name TEXT NOT NULL,
    category_label TEXT,
    parent_category_id TEXT,
    level1_category_id TEXT,
    level1_category_name TEXT,
    level2_category_id TEXT,
    level2_category_name TEXT,
    level3_category_id TEXT,
    level3_category_name TEXT,
    level4_category_id TEXT,
    level4_category_name TEXT,
    level5_category_id TEXT,
    level5_category_name TEXT,
    level6_category_id TEXT,
    level6_category_name TEXT,
    snapshot_date TEXT NOT NULL,
    FOREIGN KEY (parent_category_id) REFERENCES categories(category_id)
        DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE place_categories (
    fsq_place_id TEXT NOT NULL,
    category_id TEXT NOT NULL,
    is_primary INTEGER NOT NULL CHECK (is_primary IN (0, 1)),
    PRIMARY KEY (fsq_place_id, category_id),
    FOREIGN KEY (fsq_place_id) REFERENCES places(fsq_place_id) ON DELETE CASCADE,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);

CREATE TABLE place_unresolved_flags (
    fsq_place_id TEXT NOT NULL,
    flag TEXT NOT NULL,
    PRIMARY KEY (fsq_place_id, flag),
    FOREIGN KEY (fsq_place_id) REFERENCES places(fsq_place_id) ON DELETE CASCADE
);

CREATE TABLE data_metadata (
    metadata_id INTEGER PRIMARY KEY CHECK (metadata_id = 1),
    dataset_name TEXT NOT NULL,
    source_repository TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,
    region TEXT NOT NULL,
    source_places_file TEXT NOT NULL,
    source_categories_file TEXT NOT NULL,
    build_timestamp TEXT NOT NULL,
    places_count INTEGER NOT NULL,
    categories_count INTEGER NOT NULL,
    place_categories_count INTEGER NOT NULL,
    unresolved_flags_count INTEGER NOT NULL,
    filtering_note TEXT NOT NULL,
    license TEXT NOT NULL,
    notice TEXT NOT NULL
);
"""

INDEX_SQL = """
CREATE INDEX idx_places_name ON places(name);
CREATE INDEX idx_places_locality ON places(locality);
CREATE INDEX idx_places_region ON places(region);
CREATE INDEX idx_places_canonical_district ON places(canonical_district);
CREATE INDEX idx_places_latitude_longitude ON places(latitude, longitude);
CREATE INDEX idx_place_categories_category_id ON place_categories(category_id);
CREATE INDEX idx_categories_category_name ON categories(category_name);
CREATE INDEX idx_place_unresolved_flags_flag ON place_unresolved_flags(flag);
"""

PLACE_COLUMNS = (
    "fsq_place_id, name, latitude, longitude, address, locality, region, "
    "admin_region, canonical_district, postcode, post_town, po_box, country, date_created, "
    "date_refreshed, date_closed, website, tel, email, facebook_id, instagram, "
    "twitter, placemaker_url, snapshot_date"
)

FILTERING_NOTE = (
    "The supplied processed Parquet is the authoritative cleaned Shanghai artifact. "
    "This build verifies unique IDs, non-null coordinates, country=CN, and the observed "
    "coordinate extent, but the upstream candidate and polygon-filtering script was not "
    "included in this repository and is not reconstructed or invented here."
)


class BuildValidationError(RuntimeError):
    """Raised when the constructed database fails a required quality check."""


def _normalize_district_value(value: str) -> str:
    """Normalize spelling and punctuation without using fuzzy matching."""

    text = unicodedata.normalize("NFKC", value).replace("’", "'").replace("‘", "'")
    text = "".join(
        character
        for character in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(character)
    ).casefold()
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE).strip()
    text = re.sub(r"^(上海市|上海)\s*", "", text)
    text = re.sub(r"\s*(上海市|上海)$", "", text)
    tokens = text.split()
    if tokens and tokens[0] == "shanghai":
        tokens.pop(0)
        if tokens and tokens[0] in {"city", "shi"}:
            tokens.pop(0)
    if len(tokens) >= 2 and tokens[-2:] == ["shanghai", "shi"]:
        tokens = tokens[:-2]
    elif tokens and tokens[-1] == "shanghai":
        tokens.pop()
    return " ".join(tokens)


DISTRICT_ALIAS_LOOKUP = {
    _normalize_district_value(alias): district
    for district, aliases in DISTRICT_ALIASES.items()
    for alias in aliases
}
CANONICAL_DISTRICTS = frozenset(DISTRICT_ALIASES)


def canonical_district(*values: str | None) -> str | None:
    """Return one unambiguous standard district from explicit source aliases."""

    matches = {
        district
        for value in values
        if value and (district := DISTRICT_ALIAS_LOOKUP.get(_normalize_district_value(value)))
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _add_canonical_district(row: tuple[object, ...]) -> tuple[object, ...]:
    locality = row[5] if isinstance(row[5], str) else None
    region = row[6] if isinstance(row[6], str) else None
    admin_region = row[7] if isinstance(row[7], str) else None
    district = canonical_district(locality, region, admin_region)
    return (*row[:8], district, *row[8:])


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def inspect_parquet(path: str | Path, *, sample_size: int = 3) -> dict[str, object]:
    """Return the real DuckDB schema, row count, and JSON-compatible samples."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Parquet input does not exist: {source}")
    relation = f"read_parquet('{_sql_path(source)}')"
    connection = duckdb.connect()
    try:
        schema_rows = connection.execute(
            f"DESCRIBE SELECT * FROM {relation}"
        ).fetchall()
        cursor = connection.execute(f"SELECT * FROM {relation} LIMIT {sample_size}")
        column_names = [description[0] for description in cursor.description]
        samples = [
            dict(zip(column_names, row, strict=True)) for row in cursor.fetchall()
        ]
        row_count = int(connection.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0])
    finally:
        connection.close()
    return {
        "path": _portable_path(source),
        "row_count": row_count,
        "schema": [
            {
                "name": str(row[0]),
                "type": str(row[1]),
                "nullable": str(row[2]),
            }
            for row in schema_rows
        ],
        "samples": samples,
    }


def _copy_query(
    sqlite_connection: sqlite3.Connection,
    duckdb_connection: duckdb.DuckDBPyConnection,
    *,
    select_sql: str,
    insert_sql: str,
    batch_size: int = 5_000,
    row_transform: Callable[[tuple[object, ...]], tuple[object, ...]] | None = None,
) -> int:
    cursor = duckdb_connection.execute(select_sql)
    inserted = 0
    while batch := cursor.fetchmany(batch_size):
        rows = [row_transform(row) for row in batch] if row_transform else batch
        sqlite_connection.executemany(insert_sql, rows)
        inserted += len(batch)
    return inserted


def _create_database(
    temporary_output: Path,
    places_path: Path,
    categories_path: Path,
    *,
    expected_places_count: int,
) -> tuple[dict[str, int], str]:
    places_relation = f"read_parquet('{_sql_path(places_path)}')"
    categories_relation = f"read_parquet('{_sql_path(categories_path)}')"
    build_timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    duckdb_connection = duckdb.connect()
    sqlite_connection = sqlite3.connect(temporary_output)
    try:
        sqlite_connection.execute("PRAGMA foreign_keys = ON")
        sqlite_connection.execute("PRAGMA journal_mode = DELETE")
        sqlite_connection.execute("PRAGMA synchronous = FULL")
        sqlite_connection.executescript(SCHEMA_SQL)
        sqlite_connection.execute("BEGIN IMMEDIATE")

        categories_count = _copy_query(
            sqlite_connection,
            duckdb_connection,
            select_sql=f"""
                SELECT category_id, category_level, category_name, category_label,
                    CASE category_level
                        WHEN 1 THEN NULL
                        WHEN 2 THEN level1_category_id
                        WHEN 3 THEN level2_category_id
                        WHEN 4 THEN level3_category_id
                        WHEN 5 THEN level4_category_id
                        WHEN 6 THEN level5_category_id
                    END AS parent_category_id,
                    level1_category_id, level1_category_name,
                    level2_category_id, level2_category_name,
                    level3_category_id, level3_category_name,
                    level4_category_id, level4_category_name,
                    level5_category_id, level5_category_name,
                    level6_category_id, level6_category_name,
                    CAST(dt AS VARCHAR) AS snapshot_date
                FROM {categories_relation}
                ORDER BY category_id
            """,
            insert_sql="""
                INSERT INTO categories VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """,
        )

        places_count = _copy_query(
            sqlite_connection,
            duckdb_connection,
            select_sql=f"""
                SELECT fsq_place_id, name, CAST(latitude AS DOUBLE),
                    CAST(longitude AS DOUBLE), address, locality, region, admin_region,
                    postcode, post_town, po_box, country,
                    CAST(TRY_CAST(date_created AS DATE) AS VARCHAR),
                    CAST(TRY_CAST(date_refreshed AS DATE) AS VARCHAR),
                    CAST(TRY_CAST(date_closed AS DATE) AS VARCHAR),
                    website, tel, email, facebook_id, instagram, twitter, placemaker_url,
                    CAST(dt AS VARCHAR) AS snapshot_date
                FROM {places_relation}
                ORDER BY fsq_place_id
            """,
            insert_sql=(
                f"INSERT INTO places ({PLACE_COLUMNS}) "
                f"VALUES ({', '.join('?' for _ in range(24))})"
            ),
            row_transform=_add_canonical_district,
        )

        place_categories_count = _copy_query(
            sqlite_connection,
            duckdb_connection,
            select_sql=f"""
                SELECT fsq_place_id, category_id,
                    CAST(ordinality = 1 AS INTEGER) AS is_primary
                FROM {places_relation},
                    UNNEST(fsq_category_ids) WITH ORDINALITY u(category_id, ordinality)
                ORDER BY fsq_place_id, ordinality, category_id
            """,
            insert_sql="INSERT INTO place_categories VALUES (?, ?, ?)",
        )

        unresolved_flags_count = _copy_query(
            sqlite_connection,
            duckdb_connection,
            select_sql=f"""
                SELECT DISTINCT fsq_place_id, flag
                FROM {places_relation}, UNNEST(unresolved_flags) u(flag)
                ORDER BY fsq_place_id, flag
            """,
            insert_sql="INSERT INTO place_unresolved_flags VALUES (?, ?)",
        )

        if places_count != expected_places_count:
            raise BuildValidationError(
                f"Source place count is {places_count}, expected {expected_places_count}."
            )

        for statement in INDEX_SQL.split(";"):
            if statement.strip():
                sqlite_connection.execute(statement)
        sqlite_connection.execute(
            """
            INSERT INTO data_metadata VALUES (
                1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                "FSQ OS Places",
                "foursquare/fsq-os-places",
                SNAPSHOT_DATE,
                "Shanghai",
                _portable_path(places_path),
                _portable_path(categories_path),
                build_timestamp,
                places_count,
                categories_count,
                place_categories_count,
                unresolved_flags_count,
                FILTERING_NOTE,
                "Apache License 2.0",
                (
                    "Copyright Foursquare Labs, Inc. Retain upstream attribution and "
                    "review the FSQ OS Places license and notices before redistribution."
                ),
            ),
        )
        sqlite_connection.commit()
    except (duckdb.Error, sqlite3.Error, OSError, ValueError, BuildValidationError):
        sqlite_connection.rollback()
        raise
    finally:
        sqlite_connection.close()
        duckdb_connection.close()
    return (
        {
            "places": places_count,
            "categories": categories_count,
            "place_categories": place_categories_count,
            "place_unresolved_flags": unresolved_flags_count,
            "data_metadata": 1,
        },
        build_timestamp,
    )


def _index_inventory(connection: sqlite3.Connection) -> dict[str, list[dict[str, object]]]:
    inventory: dict[str, list[dict[str, object]]] = {}
    for table_name in (
        "places",
        "categories",
        "place_categories",
        "place_unresolved_flags",
        "data_metadata",
    ):
        indexes = []
        for row in connection.execute(f"PRAGMA index_list('{table_name}')"):
            index_name = str(row[1])
            columns = [
                str(info[2])
                for info in connection.execute(f"PRAGMA index_info('{index_name}')")
            ]
            indexes.append(
                {
                    "name": index_name,
                    "unique": bool(row[2]),
                    "origin": str(row[3]),
                    "columns": columns,
                }
            )
        inventory[table_name] = indexes
    return inventory


def validate_database(
    database_path: str | Path, *, expected_places_count: int = EXPECTED_PLACES_COUNT
) -> dict[str, object]:
    """Run required row, relationship, foreign-key, and integrity checks."""

    path = Path(database_path)
    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        table_counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "places",
                "categories",
                "place_categories",
                "place_unresolved_flags",
                "data_metadata",
            )
        }
        duplicate_place_ids = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT fsq_place_id FROM places GROUP BY fsq_place_id HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )
        null_coordinates = int(
            connection.execute(
                "SELECT COUNT(*) FROM places WHERE latitude IS NULL OR longitude IS NULL"
            ).fetchone()[0]
        )
        non_cn_places = int(
            connection.execute("SELECT COUNT(*) FROM places WHERE country <> 'CN'").fetchone()[0]
        )
        mapped_districts = int(
            connection.execute(
                "SELECT COUNT(*) FROM places WHERE canonical_district IS NOT NULL"
            ).fetchone()[0]
        )
        district_counts = {
            str(row[0]): int(row[1])
            for row in connection.execute(
                """
                SELECT canonical_district, COUNT(*)
                FROM places
                WHERE canonical_district IS NOT NULL
                GROUP BY canonical_district
                ORDER BY canonical_district
                """
            )
        }
        invalid_districts = sorted(set(district_counts) - CANONICAL_DISTRICTS)
        orphan_places = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM place_categories pc
                LEFT JOIN places p USING(fsq_place_id) WHERE p.fsq_place_id IS NULL
                """
            ).fetchone()[0]
        )
        orphan_categories = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM place_categories pc
                LEFT JOIN categories c USING(category_id) WHERE c.category_id IS NULL
                """
            ).fetchone()[0]
        )
        foreign_key_errors = [list(row) for row in connection.execute("PRAGMA foreign_key_check")]
        integrity_rows = [str(row[0]) for row in connection.execute("PRAGMA integrity_check")]
        sample_rows = [
            {
                "fsq_place_id": str(row[0]),
                "name": row[1],
                "locality": row[2],
                "region": row[3],
                "categories": row[4].split(" | ") if row[4] else [],
            }
            for row in connection.execute(
                """
                SELECT p.fsq_place_id, p.name, p.locality, p.region,
                    GROUP_CONCAT(c.category_name, ' | ') AS category_names
                FROM places p
                LEFT JOIN place_categories pc USING(fsq_place_id)
                LEFT JOIN categories c USING(category_id)
                GROUP BY p.fsq_place_id, p.name, p.locality, p.region
                ORDER BY p.fsq_place_id
                LIMIT 10
                """
            )
        ]
        indexes = _index_inventory(connection)
    finally:
        connection.close()

    checks = {
        "places_count": {
            "passed": table_counts["places"] == expected_places_count,
            "expected": expected_places_count,
            "actual": table_counts["places"],
        },
        "unique_fsq_place_id": {
            "passed": duplicate_place_ids == 0,
            "duplicate_groups": duplicate_place_ids,
        },
        "coordinates_not_null": {
            "passed": null_coordinates == 0,
            "invalid_rows": null_coordinates,
        },
        "country_is_cn": {
            "passed": non_cn_places == 0,
            "invalid_rows": non_cn_places,
        },
        "canonical_districts_valid": {
            "passed": len(district_counts) <= 16 and not invalid_districts,
            "distinct_count": len(district_counts),
            "invalid_values": invalid_districts,
        },
        "no_orphan_places": {
            "passed": orphan_places == 0,
            "orphan_rows": orphan_places,
        },
        "no_orphan_categories": {
            "passed": orphan_categories == 0,
            "orphan_rows": orphan_categories,
        },
        "foreign_key_check": {
            "passed": not foreign_key_errors,
            "errors": foreign_key_errors,
        },
        "integrity_check": {
            "passed": integrity_rows == ["ok"],
            "result": integrity_rows,
        },
    }
    failed = [name for name, result in checks.items() if not bool(result["passed"])]
    if failed:
        raise BuildValidationError(f"Database quality checks failed: {', '.join(failed)}")
    return {
        "table_counts": table_counts,
        "checks": checks,
        "indexes": indexes,
        "sample_places": sample_rows,
        "district_mapping": {
            "mapped_count": mapped_districts,
            "unmapped_count": table_counts["places"] - mapped_districts,
            "coverage": mapped_districts / table_counts["places"],
            "district_counts": district_counts,
        },
    }


def build_database(
    places_path: str | Path,
    categories_path: str | Path,
    output_path: str | Path,
    *,
    expected_places_count: int = EXPECTED_PLACES_COUNT,
) -> dict[str, object]:
    """Build a fresh validated database and atomically replace the target."""

    places = Path(places_path)
    categories = Path(categories_path)
    output = Path(output_path)
    if not places.is_file():
        raise FileNotFoundError(f"Places Parquet does not exist: {places}")
    if not categories.is_file():
        raise FileNotFoundError(f"Categories Parquet does not exist: {categories}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    started_at = perf_counter()
    completed = False
    try:
        source_counts, build_timestamp = _create_database(
            temporary,
            places,
            categories,
            expected_places_count=expected_places_count,
        )
        validation = validate_database(
            temporary, expected_places_count=expected_places_count
        )
        os.replace(temporary, output)
        completed = True
    finally:
        if not completed and temporary.exists():
            temporary.unlink()

    duration_ms = (perf_counter() - started_at) * 1000
    places_schema = inspect_parquet(places, sample_size=0)
    categories_schema = inspect_parquet(categories, sample_size=0)
    return {
        "schema_version": "1.0",
        "dataset_name": "FSQ OS Places",
        "snapshot_date": SNAPSHOT_DATE,
        "region": "Shanghai",
        "source_repository": "foursquare/fsq-os-places",
        "license": "Apache License 2.0",
        "notice": (
            "Copyright Foursquare Labs, Inc.; upstream attribution and notices must "
            "be retained when redistributing derived data."
        ),
        "inputs": {
            "places": _portable_path(places),
            "categories": _portable_path(categories),
        },
        "output": _portable_path(output),
        "build_timestamp": build_timestamp,
        "build_duration_ms": duration_ms,
        "database_size_bytes": output.stat().st_size,
        "source_counts": source_counts,
        "source_schemas": {
            "places": places_schema["schema"],
            "categories": categories_schema["schema"],
        },
        "category_association": {
            "source_field": "fsq_category_ids",
            "source_type": "VARCHAR[]",
            "is_primary_rule": "The first array element (ordinality=1) is marked primary.",
        },
        "filtering_note": FILTERING_NOTE,
        "quality": validation,
        "known_limitations": [
            "The upstream Shanghai candidate and fine-filter implementation was not supplied.",
            "Geometry and point bbox are not duplicated because latitude/longitude are retained.",
            (
                "Primary category is inferred from array order because no explicit source "
                "boolean exists."
            ),
            "The source contains 4,615 places without category IDs; they remain valid places.",
        ],
    }


def _json_default(value: object) -> object:
    if isinstance(value, (datetime, Path)):
        return str(value)
    return str(value)


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render_markdown_report(report: dict[str, object]) -> str:
    """Render the JSON report as a concise auditable Markdown build report."""

    quality = report["quality"]
    assert isinstance(quality, dict)
    table_counts = quality["table_counts"]
    checks = quality["checks"]
    indexes = quality["indexes"]
    samples = quality["sample_places"]
    district_mapping = quality["district_mapping"]
    lines = [
        "# FSQ 上海地点数据库构建报告",
        "",
        f"- 数据集：{report['dataset_name']}",
        f"- 快照：{report['snapshot_date']}",
        f"- 输出：`{report['output']}`",
        f"- 构建时间：{report['build_timestamp']}",
        f"- 构建耗时：{float(report['build_duration_ms']):.2f} ms",
        f"- 数据库大小：{int(report['database_size_bytes']):,} bytes",
        f"- 许可证：{report['license']}",
        "",
        "## 输入",
        "",
    ]
    inputs = report["inputs"]
    assert isinstance(inputs, dict)
    lines.extend(
        [
            f"- 地点：`{inputs['places']}`",
            f"- 分类：`{inputs['categories']}`",
            "",
            "## 表行数",
            "",
            "| 表 | 行数 |",
            "|---|---:|",
        ]
    )
    assert isinstance(table_counts, dict)
    for table_name, count in table_counts.items():
        lines.append(f"| `{table_name}` | {int(count):,} |")
    lines.extend(
        [
            "",
            "## 数据质量",
            "",
            "| 检查 | 结果 | 详情 |",
            "|---|---|---|",
        ]
    )
    assert isinstance(checks, dict)
    for check_name, raw_result in checks.items():
        assert isinstance(raw_result, dict)
        passed = bool(raw_result["passed"])
        details = json.dumps(raw_result, ensure_ascii=False, default=_json_default)
        lines.append(f"| `{check_name}` | {'PASS' if passed else 'FAIL'} | `{details}` |")
    assert isinstance(district_mapping, dict)
    lines.extend(
        [
            "",
            "## 行政区标准化",
            "",
            f"- 已映射：{int(district_mapping['mapped_count']):,}",
            f"- 未映射：{int(district_mapping['unmapped_count']):,}",
            f"- 覆盖率：{float(district_mapping['coverage']):.4%}",
        ]
    )
    lines.extend(["", "## 索引", ""])
    assert isinstance(indexes, dict)
    for table_name, raw_indexes in indexes.items():
        assert isinstance(raw_indexes, list)
        rendered = ", ".join(
            f"`{index['name']}({', '.join(index['columns'])})`"
            for index in raw_indexes
            if isinstance(index, dict)
        )
        lines.append(f"- `{table_name}`：{rendered or '无'}")
    lines.extend(["", "## 抽样地点及分类", ""])
    assert isinstance(samples, list)
    for sample in samples:
        assert isinstance(sample, dict)
        categories = ", ".join(str(item) for item in sample["categories"]) or "无分类"
        lines.append(
            f"- `{sample['fsq_place_id']}` {sample['name']}：{categories}"
        )
    lines.extend(["", "## 已知限制", ""])
    limitations = report["known_limitations"]
    assert isinstance(limitations, list)
    lines.extend(f"- {limitation}" for limitation in limitations)
    lines.append("")
    return "\n".join(lines)


def write_reports(report: dict[str, object], report_dir: str | Path) -> tuple[Path, Path]:
    """Write JSON and Markdown build reports atomically."""

    directory = Path(report_dir)
    json_path = directory / "build_report.json"
    markdown_path = directory / "build_report.md"
    _write_atomic(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
    )
    _write_atomic(markdown_path, render_markdown_report(report))
    return json_path.resolve(), markdown_path.resolve()


def _print_summary(report: dict[str, object], report_paths: Sequence[Path]) -> None:
    quality = report["quality"]
    assert isinstance(quality, dict)
    print("FSQ 上海地点数据库构建成功。")
    print(f"输出：{report['output']}")
    print(f"数据库大小：{int(report['database_size_bytes']):,} bytes")
    print(f"构建耗时：{float(report['build_duration_ms']):.2f} ms")
    print("表行数：")
    table_counts = quality["table_counts"]
    assert isinstance(table_counts, dict)
    for table_name, count in table_counts.items():
        print(f"  {table_name}: {int(count):,}")
    print("数据质量检查：")
    checks = quality["checks"]
    assert isinstance(checks, dict)
    for name, raw_result in checks.items():
        assert isinstance(raw_result, dict)
        print(f"  {name}: {'PASS' if raw_result['passed'] else 'FAIL'}")
    district_mapping = quality["district_mapping"]
    assert isinstance(district_mapping, dict)
    print(
        "行政区映射："
        f"{int(district_mapping['mapped_count']):,} mapped, "
        f"{int(district_mapping['unmapped_count']):,} unmapped, "
        f"{float(district_mapping['coverage']):.4%} coverage"
    )
    print("索引：")
    indexes = quality["indexes"]
    assert isinstance(indexes, dict)
    for table_name, raw_indexes in indexes.items():
        assert isinstance(raw_indexes, list)
        names = [str(index["name"]) for index in raw_indexes if isinstance(index, dict)]
        print(f"  {table_name}: {', '.join(names) if names else 'none'}")
    print("抽样地点及分类：")
    samples = quality["sample_places"]
    assert isinstance(samples, list)
    for sample in samples:
        assert isinstance(sample, dict)
        category_names = ", ".join(str(value) for value in sample["categories"])
        print(f"  {sample['fsq_place_id']} | {sample['name']} | {category_names or '无分类'}")
    for path in report_paths:
        print(f"报告：{path}")


def build_parser() -> argparse.ArgumentParser:
    """Build the FSQ database command parser."""

    parser = argparse.ArgumentParser(description="构建 FSQ 上海地点 SQLite 数据库。")
    parser.add_argument("--places", type=Path, default=DEFAULT_PLACES)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build, validate, report, and render clear failures."""

    args = build_parser().parse_args(argv)
    try:
        report = build_database(args.places, args.categories, args.output)
        report_paths = write_reports(report, args.report_dir)
    except (
        BuildValidationError,
        FileNotFoundError,
        OSError,
        ValueError,
        duckdb.Error,
        sqlite3.Error,
    ) as error:
        print(f"构建失败：{error}")
        return 1
    _print_summary(report, report_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
