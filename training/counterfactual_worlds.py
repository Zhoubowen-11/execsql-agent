"""Build deterministic multi-world SQLite tasks for execution-verifiable GRPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from execsql_agent.tools.registry import ToolRegistry
from execsql_agent.tools.sql_executor import SQLExecutor
from execsql_agent.tools.sql_validator import SQLValidator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/root/autodl-tmp/models/Qwen3-8B")
DEFAULT_OUTPUT = PROJECT_ROOT / "data/fsq/train/counterfactual/tasks_v1.jsonl"
DEFAULT_WORLD_DIR = PROJECT_ROOT / "data/fsq/train/counterfactual/worlds"

FIELDS = [
    ("tel", "联系电话"),
    ("website", "官网"),
    ("postcode", "邮政编码"),
    ("email", "电子邮箱"),
    ("locality", "城市标注"),
    ("address", "地址"),
]
DISTRICTS = ["Baoshan", "Huangpu", "Pudong", "Xuhui"]

PLACES_SCHEMA = """
CREATE TABLE places (
    fsq_place_id TEXT PRIMARY KEY NOT NULL,
    name TEXT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    address TEXT,
    locality TEXT,
    region TEXT,
    admin_region TEXT,
    canonical_district TEXT,
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
CREATE INDEX idx_places_district ON places(canonical_district);
CREATE INDEX idx_places_refreshed ON places(date_refreshed);
"""


@dataclass(frozen=True)
class CounterfactualCase:
    """One semantic task whose oracle is checked in four hidden worlds."""

    case_id: str
    template_family: str
    split: str
    question: str
    oracle_sql: str
    wrong_sql: str
    wrong_explanation: str
    parameter_signature: str


def _condition(field: str, present: bool, prefix: str = "") -> str:
    name = f"{prefix}{field}"
    if present:
        return f"{name} IS NOT NULL AND TRIM({name}) <> ''"
    return f"({name} IS NULL OR TRIM({name}) = '')"


def _split(index: int) -> str:
    if index < 16:
        return "train"
    if index < 20:
        return "dev"
    return "hard"


def _signature(family: str, values: dict[str, object]) -> str:
    payload = json.dumps(
        {"family": family, **values}, ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def generate_cases() -> list[CounterfactualCase]:
    """Create 24 disjoint parameter combinations for each of three families."""

    cases: list[CounterfactualCase] = []
    for index in range(24):
        field, label = FIELDS[index % len(FIELDS)]
        present = (index // len(FIELDS)) % 2 == 0
        date_field = "date_refreshed" if index < 12 else "date_created"
        status = "有效填写了" if present else "没有有效填写"
        condition = _condition(field, present)
        oracle = (
            f"SELECT {date_field} AS latest_date, COUNT(*) AS record_count FROM places "
            f"WHERE {condition} AND {date_field} = (SELECT MAX({date_field}) FROM places "
            f"WHERE {condition} AND {date_field} IS NOT NULL AND TRIM({date_field}) <> '')"
        )
        wrong = (
            f"SELECT MAX({date_field}) AS latest_date, COUNT(*) AS record_count FROM places "
            f"WHERE {condition} AND {date_field} IS NOT NULL AND TRIM({date_field}) <> ''"
        )
        cases.append(
            CounterfactualCase(
                case_id=f"ctf_a_{index + 1:03d}",
                template_family="aggregation_scope",
                split=_split(index),
                question=(
                    f"在{status}{label}的地点中，最近一次"
                    f"{'刷新' if date_field == 'date_refreshed' else '创建'}日期是哪一天？"
                    "当天共有多少个地点？"
                ),
                oracle_sql=oracle,
                wrong_sql=wrong,
                wrong_explanation="MAX 日期与全范围 COUNT 混用，只在所有合格记录恰好同日时正确。",
                parameter_signature=_signature(
                    "aggregation_scope",
                    {"field": field, "present": present, "date": date_field},
                ),
            )
        )

    for index in range(24):
        field, label = FIELDS[index % len(FIELDS)]
        present = (index // len(FIELDS)) % 2 == 0
        active_only = index >= 12
        k = 2 if index % 2 == 0 else 3
        condition = _condition(field, present)
        if active_only:
            condition += " AND (date_closed IS NULL OR TRIM(date_closed) = '')"
        status = "有效填写了" if present else "没有有效填写"
        scope = "且未填写关闭日期" if active_only else ""
        oracle = (
            "SELECT canonical_district, COUNT(*) AS place_count FROM places "
            f"WHERE canonical_district IS NOT NULL AND {condition} "
            "GROUP BY canonical_district "
            f"ORDER BY place_count DESC, canonical_district ASC LIMIT {k}"
        )
        wrong = (
            "SELECT canonical_district, COUNT(*) AS place_count FROM places "
            f"WHERE canonical_district IS NOT NULL AND {condition} "
            "GROUP BY canonical_district "
            f"ORDER BY canonical_district ASC LIMIT {k}"
        )
        cases.append(
            CounterfactualCase(
                case_id=f"ctf_r_{index + 1:03d}",
                template_family="ranking",
                split=_split(index),
                question=(
                    f"在可明确识别行政区、{status}{label}{scope}的地点中，"
                    f"数量最多的前 {k} 个区是哪些？数量相同时按英文名升序。"
                ),
                oracle_sql=oracle,
                wrong_sql=wrong,
                wrong_explanation="按行政区名称而非地点数量排名，只在频数恰好随名称递减时正确。",
                parameter_signature=_signature(
                    "ranking",
                    {
                        "field": field,
                        "present": present,
                        "active_only": active_only,
                        "k": k,
                    },
                ),
            )
        )

    for index in range(24):
        field, label = FIELDS[index % len(FIELDS)]
        present = (index // len(FIELDS)) % 2 == 0
        active_only = index >= 12
        condition = _condition(field, present)
        if active_only:
            condition += " AND (date_closed IS NULL OR TRIM(date_closed) = '')"
        status = "有效填写了" if present else "没有有效填写"
        scope = "且未填写关闭日期" if active_only else ""
        oracle = (
            "WITH district_counts AS ("
            "SELECT canonical_district, COUNT(*) AS place_count FROM places "
            f"WHERE canonical_district IS NOT NULL AND {condition} "
            "GROUP BY canonical_district) "
            "SELECT canonical_district, place_count FROM district_counts "
            "WHERE place_count = (SELECT MAX(place_count) FROM district_counts) "
            "ORDER BY canonical_district ASC"
        )
        wrong = (
            "SELECT canonical_district, COUNT(*) AS place_count FROM places "
            f"WHERE canonical_district IS NOT NULL AND {condition} "
            "GROUP BY canonical_district "
            "ORDER BY place_count DESC, canonical_district ASC LIMIT 1"
        )
        cases.append(
            CounterfactualCase(
                case_id=f"ctf_t_{index + 1:03d}",
                template_family="ties",
                split=_split(index),
                question=(
                    f"在可明确识别行政区、{status}{label}{scope}的地点中，哪个区数量最多？"
                    "如果第一名并列，请返回所有并列行政区，并按英文名升序。"
                ),
                oracle_sql=oracle,
                wrong_sql=wrong,
                wrong_explanation="LIMIT 1 丢失并列第一，只在最高值唯一时正确。",
                parameter_signature=_signature(
                    "ties",
                    {"field": field, "present": present, "active_only": active_only},
                ),
            )
        )
    if len(cases) != 72:
        raise ValueError(f"expected 72 cases, got {len(cases)}")
    return cases


def _create_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    connection.executescript(PLACES_SCHEMA)
    return connection


def _insert_place(
    connection: sqlite3.Connection,
    *,
    place_id: str,
    district: str,
    field: str,
    value: str | None,
    date_field: str | None = None,
    date_value: str | None = None,
    closed: bool = False,
    display_name: str | None = None,
) -> None:
    columns = {
        "fsq_place_id": place_id,
        "name": display_name or place_id,
        "latitude": 31.2,
        "longitude": 121.5,
        "canonical_district": district,
        "country": "CN",
        "snapshot_date": "2026-07-09",
        "date_closed": "2024-01-01" if closed else None,
        field: value,
    }
    if date_field is not None:
        columns[date_field] = date_value
    names = list(columns)
    placeholders = ",".join("?" for _ in names)
    connection.execute(
        f"INSERT INTO places ({','.join(names)}) VALUES ({placeholders})",
        [columns[name] for name in names],
    )


def _qualifying_value(present: bool, index: int) -> str | None:
    if present:
        return f"value-{index}"
    return None if index % 2 == 0 else ""


def _nonqualifying_value(present: bool, index: int) -> str | None:
    return _qualifying_value(not present, index)


def _build_aggregation_world(
    case: CounterfactualCase, path: Path, world_index: int
) -> None:
    params = case.parameter_signature
    del params
    case_index = int(case.case_id.rsplit("_", 1)[1]) - 1
    field, _ = FIELDS[case_index % len(FIELDS)]
    present = (case_index // len(FIELDS)) % 2 == 0
    date_field = "date_refreshed" if case_index < 12 else "date_created"
    dates = (
        ["2026-01-04"] * 4,
        ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-04", "2026-01-04"],
        ["2026-01-01", "2026-01-02", "2026-01-02", "2026-01-04", "2026-01-04", "2026-01-04"],
        ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
    )[world_index]
    connection = _create_database(path)
    try:
        with connection:
            for row_index, date in enumerate(dates):
                _insert_place(
                    connection,
                    place_id=f"{case.case_id}-w{world_index}-q{row_index}",
                    district=DISTRICTS[row_index % len(DISTRICTS)],
                    field=field,
                    value=_qualifying_value(present, row_index),
                    date_field=date_field,
                    date_value=date,
                )
            if world_index > 0:
                for row_index in range(world_index):
                    _insert_place(
                        connection,
                        place_id=f"{case.case_id}-w{world_index}-x{row_index}",
                        district=DISTRICTS[-1],
                        field=field,
                        value=_nonqualifying_value(present, row_index),
                        date_field=date_field,
                        date_value="2026-01-09",
                    )
    finally:
        connection.close()


def _count_patterns(case: CounterfactualCase, world_index: int) -> list[int]:
    """Return V1.1 counts that exercise ranking boundaries and tie cardinality."""

    if case.template_family == "ranking":
        case_index = int(case.case_id.rsplit("_", 1)[1]) - 1
        k = 2 if case_index % 2 == 0 else 3
        boundary = [8, 7, 7, 3] if k == 2 else [8, 7, 5, 5]
        return (
            [9, 7, 5, 3],  # clear ranking; alphabetical order is accidentally correct
            [5, 9, 8, 7],  # close ranking with a different winner
            boundary,  # tie exactly at the top-k boundary
            [6, 6, 4, 2],  # top tie requires stable secondary ordering
        )[world_index]
    return (
        [8, 5, 3, 1],  # unique top
        [6, 6, 0, 0],  # exactly two districts, tied
        [5, 5, 5, 0],  # exactly three districts, tied
        [7, 3, 7, 2],  # non-adjacent tie plus lower rows and duplicate names
    )[world_index]


def _build_group_world(case: CounterfactualCase, path: Path, world_index: int) -> None:
    case_index = int(case.case_id.rsplit("_", 1)[1]) - 1
    field, _ = FIELDS[case_index % len(FIELDS)]
    present = (case_index // len(FIELDS)) % 2 == 0
    active_only = case_index >= 12
    connection = _create_database(path)
    try:
        with connection:
            for district, count in zip(
                DISTRICTS, _count_patterns(case, world_index), strict=True
            ):
                for row_index in range(count):
                    _insert_place(
                        connection,
                        place_id=(
                            f"{case.case_id}-w{world_index}-{district}-{row_index}"
                        ),
                        district=district,
                        field=field,
                        value=_qualifying_value(present, row_index),
                        display_name=(
                            "Shared Display Name"
                            if case.template_family == "ties"
                            and world_index == 3
                            and district in {"Baoshan", "Pudong"}
                            else None
                        ),
                    )
            if world_index > 0:
                for row_index in range(world_index + 1):
                    _insert_place(
                        connection,
                        place_id=f"{case.case_id}-w{world_index}-x{row_index}",
                        district=DISTRICTS[(world_index + row_index) % len(DISTRICTS)],
                        field=field,
                        value=_nonqualifying_value(present, row_index),
                        closed=active_only,
                    )
    finally:
        connection.close()


def build_worlds(
    case: CounterfactualCase, world_dir: Path
) -> list[dict[str, object]]:
    """Create four hidden databases and execute the oracle in each."""

    validator = SQLValidator()
    safety = validator.validate(case.oracle_sql)
    if not safety.safe:
        raise ValueError(f"unsafe oracle for {case.case_id}")
    worlds: list[dict[str, object]] = []
    for world_index in range(4):
        path = world_dir / f"{case.case_id}_w{world_index}.db"
        if case.template_family == "aggregation_scope":
            _build_aggregation_world(case, path, world_index)
        else:
            _build_group_world(case, path, world_index)
        execution = SQLExecutor(path, max_rows=100).execute(case.oracle_sql)
        if not execution.execution_success or execution.truncated:
            raise ValueError(f"oracle failed for {case.case_id} world {world_index}")
        expected = {
            "columns": execution.columns,
            "rows": execution.rows,
            "ordered": True,
            "numeric_tolerance": 1.0e-6,
            "strict_columns": False,
        }
        worlds.append(
            {
                "world_id": f"{case.case_id}_w{world_index}",
                "database_path": str(path.resolve()),
                "expected_result": expected,
            }
        )
    return worlds


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def build_dataset(model_path: Path, output: Path, world_dir: Path) -> dict[str, Any]:
    """Materialize prompts and reward-side metadata without exposing hidden worlds."""

    from grpo_hard_dataset import HardCase, _inspect_history, _tool_schemas
    from transformers import AutoTokenizer

    template_db = world_dir / "schema_only.db"
    connection = _create_database(template_db)
    connection.close()
    registry = ToolRegistry(template_db)
    tools = _tool_schemas(registry)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    items: list[dict[str, Any]] = []
    lengths: list[int] = []
    for case in generate_cases():
        worlds = build_worlds(case, world_dir)
        hard_case = HardCase(
            case_id=case.case_id,
            template_family=case.template_family,
            difficulty="counterfactual",
            question=case.question,
            oracle_sql=case.oracle_sql,
            inspect_tables=["places"],
            parameter_signature=case.parameter_signature,
        )
        history = _inspect_history(hard_case, registry)
        prompt = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
            enable_thinking=False,
        )
        if not isinstance(prompt, str):
            raise TypeError("chat template did not return text")
        serialized_worlds = json.dumps(worlds, ensure_ascii=False, separators=(",", ":"))
        forbidden = (case.oracle_sql, serialized_worlds, "expected_result", "database_path")
        if any(value in prompt for value in forbidden):
            raise ValueError(f"reward metadata leaked into prompt: {case.case_id}")
        token_count = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        lengths.append(token_count)
        items.append(
            {
                "prompt": prompt,
                "case_id": case.case_id,
                "template_family": case.template_family,
                "split": case.split,
                "difficulty": "counterfactual",
                "parameter_signature": case.parameter_signature,
                "prompt_token_count": token_count,
                "question": case.question,
                "worlds_json": serialized_worlds,
                "oracle_sql": case.oracle_sql,
                "wrong_sql": case.wrong_sql,
                "wrong_explanation": case.wrong_explanation,
            }
        )
    _atomic_jsonl(output, items)
    return {
        "task_count": len(items),
        "family_counts": {
            family: sum(item["template_family"] == family for item in items)
            for family in ("aggregation_scope", "ranking", "ties")
        },
        "split_counts": {
            split: sum(item["split"] == split for item in items)
            for split in ("train", "dev", "hard")
        },
        "world_count": len(items) * 4,
        "prompt_tokens": {
            "min": min(lengths),
            "mean": statistics.fmean(lengths),
            "max": max(lengths),
        },
        "leakage_count": 0,
        "duplicate_question_count": len(items)
        - len({str(item["question"]) for item in items}),
        "duplicate_signature_count": len(items)
        - len({str(item["parameter_signature"]) for item in items}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--world-dir", type=Path, default=DEFAULT_WORLD_DIR)
    args = parser.parse_args()
    stats = build_dataset(args.model, args.output, args.world_dir)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
