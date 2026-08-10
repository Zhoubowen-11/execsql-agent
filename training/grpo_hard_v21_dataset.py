"""Generate boundary-focused RL-hard V2.1 candidates from train-side data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from grpo_hard_dataset import DISTRICTS, HardCase, _category_names, _literal, _missing, _present
from grpo_hard_v2_dataset import (
    DEFAULT_DATABASE,
    DEFAULT_MODEL,
    _atomic_jsonl,
    _signature,
    build_items,
)
from transformers import AutoTokenizer

from execsql_agent.tools.sql_executor import SQLExecutor
from execsql_agent.tools.sql_validator import SQLValidator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXISTING = PROJECT_ROOT / "data/fsq/train/grpo/rl_train_hard_v2_candidates.jsonl"
DEFAULT_SUPPLEMENT = PROJECT_ROOT / "data/fsq/train/grpo/rl_train_hard_v2_supplement.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/fsq/train/grpo/rl_train_hard_v21_candidates.jsonl"


def make_case(
    family: str,
    index: int,
    *,
    question: str,
    sql: str,
    tables: list[str],
    parameters: dict[str, object],
) -> HardCase:
    payload = {"difficulty": "boundary", **parameters}
    return HardCase(
        case_id=f"rlh_v21_{'m' if family == 'multi_category_semantics' else 'o'}_{index + 1:03d}",
        template_family=family,
        difficulty="boundary",
        question=question,
        oracle_sql=sql,
        inspect_tables=tables,
        parameter_signature=_signature(family, payload),
    )


def cooccurring_pairs(database: Path) -> list[tuple[str, str]]:
    sql = (
        "SELECT c1.category_name, c2.category_name "
        "FROM place_categories pc1 "
        "JOIN place_categories pc2 ON pc2.fsq_place_id = pc1.fsq_place_id "
        "AND pc2.category_id > pc1.category_id "
        "JOIN categories c1 ON c1.category_id = pc1.category_id "
        "JOIN categories c2 ON c2.category_id = pc2.category_id "
        "GROUP BY c1.category_name, c2.category_name "
        "HAVING COUNT(DISTINCT pc1.fsq_place_id) >= 5 "
        "ORDER BY COUNT(DISTINCT pc1.fsq_place_id) DESC, c1.category_name, c2.category_name "
        "LIMIT 100"
    )
    result = SQLExecutor(database, max_rows=100).execute(sql)
    if not result.execution_success or result.truncated or len(result.rows) < 30:
        raise ValueError("could not derive enough train-side category intersections")
    return [(str(row[0]), str(row[1])) for row in result.rows]


def multi_category_cases(database: Path) -> list[HardCase]:
    categories = _category_names(database, limit=60)
    pairs = cooccurring_pairs(database)
    cases: list[HardCase] = []
    for index in range(120):
        district_cn, district_en = DISTRICTS[(index * 7 + 3) % len(DISTRICTS)]
        n = (2, 3, 4)[index % 3]
        k = (3, 5, 8, 10)[(index // 3) % 4]
        if index < 30:
            sql = (
                "SELECT p.fsq_place_id, p.name, COUNT(DISTINCT pc.category_id) AS category_count "
                "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
                f"WHERE p.canonical_district = {_literal(district_en)} "
                "GROUP BY p.fsq_place_id, p.name "
                f"HAVING COUNT(DISTINCT pc.category_id) = {n} "
                "ORDER BY p.fsq_place_id ASC LIMIT " + str(k)
            )
            question = (
                f"在可明确归入{district_cn}的地点中，列出恰好关联 {n} 种不同类型的前 {k} 个地点，"
                "依次给出地点编号、名称和类型数，并按地点编号升序。"
            )
            parameters = {"shape": "exact_n_rows", "district": district_en, "n": n, "k": k}
            tables = ["places", "place_categories"]
        elif index < 60:
            field, label = (("tel", "联系电话"), ("website", "官网"), ("email", "电子邮箱"))[
                index % 3
            ]
            sql = (
                "SELECT COUNT(*) AS place_count FROM ("
                "SELECT p.fsq_place_id FROM places p "
                "JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
                f"WHERE p.canonical_district = {_literal(district_en)} "
                f"AND {_present('p.' + field)} "
                "GROUP BY p.fsq_place_id "
                f"HAVING COUNT(DISTINCT pc.category_id) >= {n}) AS qualified"
            )
            question = (
                f"{district_cn}有效填写{label}且至少关联 {n} 种不同类型的独立地点共有多少个？"
                "只返回一个总数，不要逐个列出地点。"
            )
            parameters = {"shape": "entity_scalar", "district": district_en, "field": field, "n": n}
            tables = ["places", "place_categories"]
        elif index < 90:
            first, second = pairs[index - 60]
            sql = (
                "SELECT p.fsq_place_id, p.name FROM places p "
                "JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
                "JOIN categories c ON c.category_id = pc.category_id "
                f"WHERE p.canonical_district = {_literal(district_en)} "
                f"AND c.category_name IN ({_literal(first)}, {_literal(second)}) "
                "GROUP BY p.fsq_place_id, p.name "
                "HAVING COUNT(DISTINCT c.category_name) = 2 "
                "ORDER BY p.fsq_place_id ASC LIMIT " + str(k)
            )
            question = (
                f"在可明确归入{district_cn}的地点中，"
                f"列出同时包含 {first} 和 {second} 两种类型的前 {k} 个地点，"
                "给出地点编号和名称，并按地点编号升序。"
            )
            parameters = {
                "shape": "intersection",
                "district": district_en,
                "a": first,
                "b": second,
                "k": k,
            }
            tables = ["places", "place_categories", "categories"]
        else:
            category = categories[(index * 11) % len(categories)]
            sql = (
                "SELECT p.fsq_place_id, p.name, COUNT(DISTINCT pc.category_id) AS category_count "
                "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
                "JOIN categories c ON c.category_id = pc.category_id "
                f"WHERE p.canonical_district = {_literal(district_en)} "
                "GROUP BY p.fsq_place_id, p.name "
                f"HAVING COUNT(DISTINCT pc.category_id) >= {n} AND "
                "SUM(CASE WHEN pc.is_primary = 1 AND c.category_name = "
                f"{_literal(category)} THEN 1 ELSE 0 END) > 0 "
                "ORDER BY category_count DESC, p.fsq_place_id ASC LIMIT " + str(k)
            )
            question = (
                f"在可明确归入{district_cn}的地点中，列出首项推断类型为 {category}、"
                f"并且总共至少关联 {n} 种类型的前 {k} 个地点；"
                "给出地点编号、名称和不同类型数，先按类型数降序，同数按地点编号升序。"
            )
            parameters = {
                "shape": "primary_membership",
                "district": district_en,
                "category": category,
                "n": n,
                "k": k,
            }
            tables = ["places", "place_categories", "categories"]
        cases.append(
            make_case(
                "multi_category_semantics",
                index,
                question=question,
                sql=sql,
                tables=tables,
                parameters=parameters,
            )
        )
    return cases


def output_shape_cases(database: Path) -> list[HardCase]:
    categories = _category_names(database, limit=60)
    fields = (
        ("tel", "联系电话"),
        ("website", "官网"),
        ("email", "电子邮箱"),
        ("postcode", "邮政编码"),
    )
    cases: list[HardCase] = []
    for index in range(100):
        district_cn, district_en = DISTRICTS[(index * 5 + 1) % len(DISTRICTS)]
        category = categories[(index * 13 + 2) % len(categories)]
        k = (3, 5, 8, 10)[index % 4]
        field, label = fields[index % len(fields)]
        if index < 25:
            sql = (
                "SELECT p.fsq_place_id, p.name FROM places p "
                "JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
                "JOIN categories c ON c.category_id = pc.category_id "
                f"WHERE p.canonical_district = {_literal(district_en)} "
                f"AND c.category_name = {_literal(category)} AND {_present('p.' + field)} "
                "ORDER BY p.name ASC, p.fsq_place_id ASC LIMIT " + str(k)
            )
            question = (
                f"列出{district_cn}有效填写{label}的 {category} 地点中，按名称排序最靠前的 {k} 个；"
                "必须同时返回地点编号和名称，名称相同时按地点编号升序。"
            )
            parameters = {
                "shape": "id_and_name",
                "district": district_en,
                "category": category,
                "field": field,
                "k": k,
            }
            tables = ["places", "place_categories", "categories"]
        elif index < 50:
            sql = (
                "SELECT p.name, p.fsq_place_id, p.canonical_district FROM places p "
                "JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
                "JOIN categories c ON c.category_id = pc.category_id "
                f"WHERE p.canonical_district = {_literal(district_en)} "
                f"AND c.category_name = {_literal(category)} "
                "ORDER BY p.name ASC, p.fsq_place_id ASC LIMIT " + str(k)
            )
            question = (
                "请按名称、地点编号、所属标准行政区这一顺序，"
                f"返回{district_cn}名称排序最靠前的 {k} 个 {category} 地点；"
                "名称相同时按地点编号升序。"
            )
            parameters = {
                "shape": "ordered_columns",
                "district": district_en,
                "category": category,
                "k": k,
            }
            tables = ["places", "place_categories", "categories"]
        elif index < 75:
            sql = (
                "SELECT COUNT(DISTINCT p.fsq_place_id) AS total_places, "
                f"COUNT(DISTINCT CASE WHEN {_present('p.' + field)} "
                "THEN p.fsq_place_id END) AS populated_places "
                "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
                "JOIN categories c ON c.category_id = pc.category_id "
                f"WHERE p.canonical_district = {_literal(district_en)} "
                f"AND c.category_name = {_literal(category)} AND {_missing('p.date_closed')}"
            )
            question = (
                f"请在同一行依次给出{district_cn}未填写关闭日期的 {category} 独立地点总数，"
                f"以及其中有效填写{label}的独立地点数；不要只返回后一个指标。"
            )
            parameters = {
                "shape": "two_metrics",
                "district": district_en,
                "category": category,
                "field": field,
            }
            tables = ["places", "place_categories", "categories"]
        else:
            sql = (
                "WITH counts AS (SELECT p.canonical_district AS district, "
                "COUNT(DISTINCT p.fsq_place_id) AS place_count FROM places p "
                "JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
                "JOIN categories c ON c.category_id = pc.category_id "
                "WHERE p.canonical_district IS NOT NULL "
                f"AND c.category_name = {_literal(category)} GROUP BY p.canonical_district), "
                "cutoff AS (SELECT place_count FROM counts "
                f"ORDER BY place_count DESC LIMIT 1 OFFSET {k - 1}) "
                "SELECT district, place_count FROM counts "
                "WHERE place_count >= (SELECT place_count FROM cutoff) "
                "ORDER BY place_count DESC, district ASC"
            )
            question = (
                f"只基于可明确识别行政区的数据，找出 {category} 地点数量排名前 {k} 的区；"
                "如果第 K 名有并列，请把所有达到该数量的区都返回，并给出区名和地点数。"
            )
            parameters = {"shape": "top_k_with_ties", "category": category, "k": k}
            tables = ["places", "place_categories", "categories"]
        cases.append(
            make_case(
                "output_shape",
                index,
                question=question,
                sql=sql,
                tables=tables,
                parameters=parameters,
            )
        )
    return cases


def collision_audit(items: list[dict[str, Any]], paths: list[Path]) -> dict[str, int]:
    validator = SQLValidator()
    questions = {"".join(str(item["question"]).split()).casefold() for item in items}
    sql = {validator.validate(str(item["oracle_sql"])).normalized_sql for item in items}
    signatures = {str(item["parameter_signature"]) for item in items}
    collisions = {"question": 0, "oracle_sql": 0, "parameter_signature": 0}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                other = json.loads(line)
                collisions["question"] += (
                    "".join(str(other["question"]).split()).casefold() in questions
                )
                collisions["oracle_sql"] += (
                    validator.validate(str(other["oracle_sql"])).normalized_sql in sql
                )
                collisions["parameter_signature"] += str(other["parameter_signature"]) in signatures
    if any(collisions.values()):
        raise ValueError(f"V2.1 collision audit failed: {collisions}")
    return collisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--existing", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    existing = args.existing or [DEFAULT_EXISTING, DEFAULT_SUPPLEMENT]
    cases = multi_category_cases(args.database) + output_shape_cases(args.database)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    items, stats = build_items(
        database=args.database, tokenizer=tokenizer, existing_path=existing[0], cases=cases
    )
    stats["existing_collision_audit"] = collision_audit(items, existing)
    _atomic_jsonl(args.output, items)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
