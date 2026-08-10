"""Generate execution-verified, train-side RL-hard SQL-decision prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from execsql_agent.models import ToolCallRequest
from execsql_agent.tools.registry import ToolRegistry
from execsql_agent.tools.sql_executor import SQLExecutor
from execsql_agent.tools.sql_validator import SQLValidator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / "data/fsq/shanghai_places.db"
DEFAULT_MODEL = Path("/root/autodl-tmp/models/Qwen3-8B")
DEFAULT_OUTPUT = PROJECT_ROOT / "data/fsq/train/grpo/rl_train_hard_v1_candidates.jsonl"

DISTRICTS = [
    ("浦东新区", "Pudong"),
    ("徐汇区", "Xuhui"),
    ("黄浦区", "Huangpu"),
    ("静安区", "Jing'an"),
    ("长宁区", "Changning"),
    ("闵行区", "Minhang"),
    ("宝山区", "Baoshan"),
    ("嘉定区", "Jiading"),
    ("松江区", "Songjiang"),
    ("青浦区", "Qingpu"),
    ("普陀区", "Putuo"),
    ("虹口区", "Hongkou"),
    ("杨浦区", "Yangpu"),
    ("金山区", "Jinshan"),
    ("奉贤区", "Fengxian"),
    ("崇明区", "Chongming"),
]


@dataclass(frozen=True)
class HardCase:
    """One new train-side parameter combination and its generation oracle."""

    case_id: str
    template_family: str
    difficulty: str
    question: str
    oracle_sql: str
    inspect_tables: list[str]
    parameter_signature: str


@dataclass(frozen=True)
class HardDatasetStats:
    """Construction and leakage diagnostics."""

    candidate_count: int
    family_counts: dict[str, int]
    prompt_token_min: int
    prompt_token_mean: float
    prompt_token_max: int
    duplicate_questions: int
    duplicate_sql: int
    leakage_count: int


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _present(field: str) -> str:
    return f"{field} IS NOT NULL AND TRIM({field}) <> ''"


def _missing(field: str) -> str:
    return f"({field} IS NULL OR TRIM({field}) = '')"


def _signature(family: str, parameters: dict[str, object]) -> str:
    raw = json.dumps(
        {"family": family, "parameters": parameters},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _case(
    family: str,
    index: int,
    *,
    question: str,
    sql: str,
    tables: list[str],
    parameters: dict[str, object],
) -> HardCase:
    code = {
        "aggregation_scope": "a",
        "filtering_semantics": "f",
        "ranking": "r",
        "ties": "t",
        "multi_category_semantics": "m",
        "output_shape": "o",
    }[family]
    return HardCase(
        case_id=f"rlh_v1_{code}_{index:03d}",
        template_family=family,
        difficulty="hard",
        question=question,
        oracle_sql=sql,
        inspect_tables=tables,
        parameter_signature=_signature(family, parameters),
    )


def _readonly_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=ro"


def _category_names(database: Path, limit: int = 20) -> list[str]:
    sql = (
        "SELECT c.category_name FROM categories c "
        "JOIN place_categories pc ON pc.category_id = c.category_id "
        "GROUP BY c.category_id, c.category_name "
        "HAVING COUNT(DISTINCT pc.fsq_place_id) >= 40 "
        "ORDER BY COUNT(DISTINCT pc.fsq_place_id) DESC, c.category_name ASC LIMIT ?"
    )
    with sqlite3.connect(_readonly_uri(database), uri=True) as connection:
        names = [str(row[0]) for row in connection.execute(sql, (limit,))]
    if len(names) != limit:
        raise ValueError(f"expected {limit} train-side category values, found {len(names)}")
    return names


def generate_hard_cases(database: Path) -> list[HardCase]:
    """Create 120 new combinations across six abstract semantic families."""

    categories = _category_names(database)
    cases: list[HardCase] = []

    for index in range(20):
        district_cn, district_en = DISTRICTS[index % len(DISTRICTS)]
        category = categories[index]
        contact_field = "tel" if index % 2 == 0 else "website"
        contact_label = "联系电话" if contact_field == "tel" else "官网"
        sql = (
            "SELECT COUNT(DISTINCT p.fsq_place_id) AS place_count "
            "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
            "JOIN categories c ON c.category_id = pc.category_id "
            f"WHERE p.canonical_district = {_literal(district_en)} "
            f"AND c.category_name = {_literal(category)} AND {_present('p.' + contact_field)}"
        )
        cases.append(
            _case(
                "aggregation_scope",
                index + 1,
                question=(
                    f"在能够明确识别为{district_cn}的地点中，属于 {category} "
                    f"且留有{contact_label}的"
                    "独立地点有多少个？同一地点出现多个分类关联时只算一次。"
                ),
                sql=sql,
                tables=["places", "place_categories", "categories"],
                parameters={
                    "district": district_en,
                    "category": category,
                    "contact": contact_field,
                },
            )
        )

    fields = [
        ("tel", "联系电话"),
        ("website", "官网"),
        ("postcode", "邮政编码"),
        ("email", "电子邮箱"),
        ("locality", "原始城市标注"),
    ]
    for index in range(20):
        district_cn, district_en = DISTRICTS[(index * 3) % len(DISTRICTS)]
        category = categories[(index * 7) % len(categories)]
        field, field_label = fields[index % len(fields)]
        want_present = index % 2 == 0
        condition = _present("p." + field) if want_present else _missing("p." + field)
        status = "填写了" if want_present else "没有有效填写"
        sql = (
            "SELECT COUNT(DISTINCT p.fsq_place_id) AS place_count "
            "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
            "JOIN categories c ON c.category_id = pc.category_id "
            f"WHERE p.canonical_district = {_literal(district_en)} "
            f"AND p.country = 'CN' AND c.category_name = {_literal(category)} "
            f"AND {condition}"
        )
        cases.append(
            _case(
                "filtering_semantics",
                index + 1,
                question=(
                    f"{district_cn}国家标注为中国的 {category} 地点里，"
                    f"{status}{field_label}的有多少个？"
                    "空白内容也按未填写处理，并按地点去重。"
                ),
                sql=sql,
                tables=["places", "place_categories", "categories"],
                parameters={
                    "district": district_en,
                    "category": category,
                    "field": field,
                    "present": want_present,
                },
            )
        )

    for index in range(20):
        category = categories[index]
        k = (3, 5, 7, 10)[index % 4]
        field, field_label = (("tel", "联系电话"), ("website", "官网"))[index % 2]
        sql = (
            "SELECT p.canonical_district, COUNT(DISTINCT p.fsq_place_id) AS place_count "
            "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
            "JOIN categories c ON c.category_id = pc.category_id "
            "WHERE p.canonical_district IS NOT NULL "
            f"AND c.category_name = {_literal(category)} AND {_present('p.' + field)} "
            "GROUP BY p.canonical_district "
            "ORDER BY place_count DESC, p.canonical_district ASC "
            f"LIMIT {k}"
        )
        cases.append(
            _case(
                "ranking",
                index + 1,
                question=(
                    f"只看能够明确映射行政区、并留有{field_label}的 {category} 地点，数量最多的"
                    f"前 {k} 个区是哪些？数量相同时按行政区英文名升序确定顺序。"
                ),
                sql=sql,
                tables=["places", "place_categories", "categories"],
                parameters={"category": category, "k": k, "contact": field},
            )
        )

    for index in range(20):
        category = categories[index]
        contact_field = "tel" if index % 2 == 0 else "website"
        contact_label = "联系电话" if contact_field == "tel" else "官网"
        sql = (
            "WITH district_counts AS ("
            "SELECT p.canonical_district, COUNT(DISTINCT p.fsq_place_id) AS place_count "
            "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
            "JOIN categories c ON c.category_id = pc.category_id "
            "WHERE p.canonical_district IS NOT NULL "
            f"AND c.category_name = {_literal(category)} AND {_present('p.' + contact_field)} "
            "GROUP BY p.canonical_district) "
            "SELECT canonical_district, place_count FROM district_counts "
            "WHERE place_count = (SELECT MAX(place_count) FROM district_counts) "
            "ORDER BY canonical_district ASC"
        )
        cases.append(
            _case(
                "ties",
                index + 1,
                question=(
                    f"在可明确识别行政区且留有{contact_label}的 {category} 地点中，哪个区数量最多？"
                    "如果第一名并列，请返回所有并列行政区，并按英文名升序排列。"
                ),
                sql=sql,
                tables=["places", "place_categories", "categories"],
                parameters={"category": category, "contact": contact_field},
            )
        )

    for index in range(20):
        district_cn, district_en = DISTRICTS[(index * 5) % len(DISTRICTS)]
        threshold = (1, 2, 3, 4)[(index + index // len(DISTRICTS)) % 4]
        contact_field = "tel" if index % 2 == 0 else "website"
        contact_label = "联系电话" if contact_field == "tel" else "官网"
        sql = (
            "SELECT COUNT(*) AS place_count FROM ("
            "SELECT p.fsq_place_id FROM places p "
            "JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
            f"WHERE p.canonical_district = {_literal(district_en)} "
            f"AND {_present('p.' + contact_field)} GROUP BY p.fsq_place_id "
            f"HAVING COUNT(DISTINCT pc.category_id) > {threshold}) AS qualified_places"
        )
        cases.append(
            _case(
                "multi_category_semantics",
                index + 1,
                question=(
                    f"{district_cn}留有{contact_label}的地点中，同时标注至少 "
                    f"{threshold + 1} 种不同类型的"
                    "独立地点有多少个？重复的同类标注不能重复计数。"
                ),
                sql=sql,
                tables=["places", "place_categories"],
                parameters={
                    "district": district_en,
                    "threshold": threshold,
                    "contact": contact_field,
                },
            )
        )

    for index in range(20):
        district_cn, district_en = DISTRICTS[(index * 7) % len(DISTRICTS)]
        category = categories[(index * 3) % len(categories)]
        sql = (
            "SELECT COUNT(DISTINCT p.fsq_place_id) AS total_places, "
            "COUNT(DISTINCT CASE WHEN p.tel IS NOT NULL AND TRIM(p.tel) <> '' "
            "THEN p.fsq_place_id END) AS with_phone, "
            "COUNT(DISTINCT CASE WHEN p.website IS NOT NULL AND TRIM(p.website) <> '' "
            "THEN p.fsq_place_id END) AS with_website, "
            "ROUND(100.0 * COUNT(DISTINCT CASE WHEN p.tel IS NOT NULL AND TRIM(p.tel) <> '' "
            "THEN p.fsq_place_id END) / NULLIF(COUNT(DISTINCT p.fsq_place_id), 0), 2) "
            "AS phone_percentage "
            "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
            "JOIN categories c ON c.category_id = pc.category_id "
            f"WHERE p.canonical_district = {_literal(district_en)} "
            f"AND c.category_name = {_literal(category)}"
        )
        cases.append(
            _case(
                "output_shape",
                index + 1,
                question=(
                    f"请一次给出{district_cn} {category} 地点的四项数据：地点总数、留有电话的数量、"
                    "留有官网的数量，以及电话覆盖率（百分比保留两位小数）。各项都按独立地点计算。"
                ),
                sql=sql,
                tables=["places", "place_categories", "categories"],
                parameters={"district": district_en, "category": category},
            )
        )

    if len(cases) != 120:
        raise ValueError(f"expected 120 RL-hard cases, found {len(cases)}")
    return cases


def _tool_schemas(registry: ToolRegistry) -> list[dict[str, object]]:
    return [
        {"type": "function", "function": definition.model_dump(mode="json")}
        for definition in registry.definitions
    ]


def _inspect_history(case: HardCase, registry: ToolRegistry) -> list[dict[str, object]]:
    call = ToolCallRequest(
        id=f"{case.case_id}_inspect",
        name="inspect_schema",
        arguments={"table_names": case.inspect_tables},
    )
    validation, result = registry.dispatch(call)
    if not validation.valid or not result.success:
        raise ValueError(f"{case.case_id}: inspect_schema failed")
    return [
        {"role": "user", "content": case.question},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            call.arguments,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.name,
            "content": result.model_dump_json(),
        },
    ]


def build_hard_items(
    *,
    database: Path,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[dict[str, Any]], HardDatasetStats]:
    """Execute every oracle and serialize Qwen prompts with reward metadata outside it."""

    registry = ToolRegistry(database)
    tools = _tool_schemas(registry)
    validator = SQLValidator()
    executor = SQLExecutor(database, max_rows=100)
    cases = generate_hard_cases(database)
    items: list[dict[str, Any]] = []
    lengths: list[int] = []
    leaks = 0
    for case in cases:
        safety = validator.validate(case.oracle_sql, database_path=database)
        if not safety.safe or safety.syntax_valid is not True:
            raise ValueError(f"{case.case_id}: unsafe or invalid oracle SQL")
        execution = executor.execute(case.oracle_sql)
        if not execution.execution_success or execution.truncated:
            raise ValueError(f"{case.case_id}: oracle execution failed or truncated")
        history = _inspect_history(case, registry)
        prompt = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
            enable_thinking=False,
        )
        if not isinstance(prompt, str):
            raise TypeError("chat template did not return text")
        if case.oracle_sql in prompt or "expected_result" in prompt.casefold():
            leaks += 1
            raise ValueError(f"{case.case_id}: reward-side metadata leaked into prompt")
        token_count = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        lengths.append(token_count)
        expected = {
            "columns": execution.columns,
            "rows": execution.rows,
            "ordered": True,
            "numeric_tolerance": 1.0e-6,
            "strict_columns": False,
        }
        items.append(
            {
                "prompt": prompt,
                "case_id": case.case_id,
                "database_path": str(database.resolve()),
                "expected_result_json": json.dumps(
                    expected, ensure_ascii=False, separators=(",", ":")
                ),
                "template_family": case.template_family,
                "difficulty": case.difficulty,
                "parameter_signature": case.parameter_signature,
                "prompt_token_count": token_count,
                "question": case.question,
                "oracle_sql": case.oracle_sql,
                "oracle_execution": execution.model_dump(mode="json"),
            }
        )
    questions = [case.question for case in cases]
    normalized_sql = [validator.validate(case.oracle_sql).normalized_sql for case in cases]
    family_counts: dict[str, int] = {}
    for case in cases:
        family_counts[case.template_family] = family_counts.get(case.template_family, 0) + 1
    stats = HardDatasetStats(
        candidate_count=len(items),
        family_counts=dict(sorted(family_counts.items())),
        prompt_token_min=min(lengths),
        prompt_token_mean=statistics.fmean(lengths),
        prompt_token_max=max(lengths),
        duplicate_questions=len(questions) - len(set(questions)),
        duplicate_sql=len(normalized_sql) - len(set(normalized_sql)),
        leakage_count=leaks,
    )
    if stats.duplicate_questions or stats.duplicate_sql:
        raise ValueError(f"RL-hard candidates are not unique: {asdict(stats)}")
    return items, stats


def _atomic_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    items, stats = build_hard_items(database=args.database, tokenizer=tokenizer)
    _atomic_jsonl(args.output, items)
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
