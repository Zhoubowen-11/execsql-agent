"""Build the independent FSQ Shanghai holdout set from real SQLite results."""

# ruff: noqa: E501 -- gold SQL stays beside each auditable holdout definition.

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from execsql_agent.tools.sql_executor import SQLExecutor
from execsql_agent.tools.sql_validator import SQLValidator

DEFAULT_DATABASE = Path("data/fsq/shanghai_places.db")
DEFAULT_OUTPUT = Path("data/fsq/eval/questions_holdout_v1.json")
REFERENCE_DATASETS = (
    Path("data/fsq/eval/questions_v1.json"),
    Path("data/fsq/eval/questions_v2.json"),
)


@dataclass(frozen=True)
class HoldoutSpec:
    """One deterministic holdout question and its gold query."""

    case_id: str
    question: str
    group: str
    gold_sql: str
    ordered: bool = True


SPECS = (
    HoldoutSpec("fsq_h01", "有多少地点留下了电子邮箱，却没有留下联系电话？", "single_filter_quality", "SELECT COUNT(*) AS place_count FROM places WHERE email IS NOT NULL AND TRIM(email) <> '' AND (tel IS NULL OR TRIM(tel) = '')"),
    HoldoutSpec("fsq_h02", "有多少地点的创建日期和最近更新时间恰好是同一天？", "single_filter_quality", "SELECT COUNT(*) AS place_count FROM places WHERE date_created IS NOT NULL AND date_refreshed IS NOT NULL AND date_created = date_refreshed"),
    HoldoutSpec("fsq_h03", "已经记录为关闭、但仍保留官网链接的地点有多少个？", "single_filter_quality", "SELECT COUNT(*) AS place_count FROM places WHERE date_closed IS NOT NULL AND website IS NOT NULL AND TRIM(website) <> ''"),
    HoldoutSpec("fsq_h04", "数据质量检查中，被标记了至少两类待处理问题的地点有多少个？", "single_filter_quality", "SELECT COUNT(*) AS place_count FROM (SELECT fsq_place_id FROM place_unresolved_flags GROUP BY fsq_place_id HAVING COUNT(DISTINCT flag) >= 2)"),
    HoldoutSpec("fsq_h05", "全部地点中，留有官网链接的占比是多少？请同时给出地点总数、留有官网的数量和百分比。", "aggregate_ratio_topk", "SELECT COUNT(*) AS total_places, SUM(CASE WHEN website IS NOT NULL AND TRIM(website) <> '' THEN 1 ELSE 0 END) AS places_with_website, ROUND(100.0 * SUM(CASE WHEN website IS NOT NULL AND TRIM(website) <> '' THEN 1 ELSE 0 END) / COUNT(*), 4) AS website_percentage FROM places"),
    HoldoutSpec("fsq_h06", "只留联系电话和只留官网链接的地点分别有多少？哪一组更多，多多少？", "aggregate_ratio_topk", "SELECT SUM(CASE WHEN tel IS NOT NULL AND TRIM(tel) <> '' AND (website IS NULL OR TRIM(website) = '') THEN 1 ELSE 0 END) AS phone_only_count, SUM(CASE WHEN website IS NOT NULL AND TRIM(website) <> '' AND (tel IS NULL OR TRIM(tel) = '') THEN 1 ELSE 0 END) AS website_only_count, SUM(CASE WHEN tel IS NOT NULL AND TRIM(tel) <> '' AND (website IS NULL OR TRIM(website) = '') THEN 1 ELSE 0 END) - SUM(CASE WHEN website IS NOT NULL AND TRIM(website) <> '' AND (tel IS NULL OR TRIM(tel) = '') THEN 1 ELSE 0 END) AS phone_only_lead FROM places"),
    HoldoutSpec("fsq_h07", "按年月看，新增地点最多的五个月是哪几个月？如果数量相同，较早的月份排在前面。", "aggregate_ratio_topk", "SELECT SUBSTR(date_created, 1, 7) AS created_month, COUNT(*) AS place_count FROM places WHERE date_created IS NOT NULL GROUP BY created_month ORDER BY place_count DESC, created_month ASC LIMIT 5"),
    HoldoutSpec("fsq_h08", "在至少收录五百个地点的原始城市或区县标注中，官网覆盖率最高的前五个是哪些？比例相同时按名称排列。", "aggregate_ratio_topk", "SELECT locality, COUNT(*) AS total_places, SUM(CASE WHEN website IS NOT NULL AND TRIM(website) <> '' THEN 1 ELSE 0 END) AS places_with_website, ROUND(100.0 * SUM(CASE WHEN website IS NOT NULL AND TRIM(website) <> '' THEN 1 ELSE 0 END) / COUNT(*), 4) AS website_percentage FROM places WHERE locality IS NOT NULL AND TRIM(locality) <> '' GROUP BY locality HAVING COUNT(*) >= 500 ORDER BY website_percentage DESC, locality ASC LIMIT 5"),
    HoldoutSpec("fsq_h09", "在至少收录一千个地点的创建年份中，已关闭地点占比最高的五个年份是哪些？占比相同时较早年份排在前面。", "aggregate_ratio_topk", "SELECT SUBSTR(date_created, 1, 4) AS created_year, COUNT(*) AS total_places, SUM(CASE WHEN date_closed IS NOT NULL THEN 1 ELSE 0 END) AS closed_places, ROUND(100.0 * SUM(CASE WHEN date_closed IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 4) AS closed_percentage FROM places WHERE date_created IS NOT NULL GROUP BY created_year HAVING COUNT(*) >= 1000 ORDER BY closed_percentage DESC, created_year ASC LIMIT 5"),
    HoldoutSpec("fsq_h10", "2025年创建的地点和2025年更新的地点各有多少个？其中两项都满足的有多少个？", "aggregate_ratio_topk", "SELECT COUNT(DISTINCT CASE WHEN date_created LIKE '2025-%' THEN fsq_place_id END) AS created_in_2025, COUNT(DISTINCT CASE WHEN date_refreshed LIKE '2025-%' THEN fsq_place_id END) AS refreshed_in_2025, COUNT(DISTINCT CASE WHEN date_created LIKE '2025-%' AND date_refreshed LIKE '2025-%' THEN fsq_place_id END) AS both_in_2025 FROM places"),
    HoldoutSpec("fsq_h11", "在至少覆盖一百个地点的类型中，留下联系电话的比例最高的是哪五种？比例相同时按类型名称排列。", "category_join", "SELECT c.category_name, COUNT(DISTINCT p.fsq_place_id) AS total_places, COUNT(DISTINCT CASE WHEN p.tel IS NOT NULL AND TRIM(p.tel) <> '' THEN p.fsq_place_id END) AS places_with_phone, ROUND(100.0 * COUNT(DISTINCT CASE WHEN p.tel IS NOT NULL AND TRIM(p.tel) <> '' THEN p.fsq_place_id END) / COUNT(DISTINCT p.fsq_place_id), 4) AS phone_percentage FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id JOIN categories c ON c.category_id = pc.category_id GROUP BY c.category_id, c.category_name HAVING COUNT(DISTINCT p.fsq_place_id) >= 100 ORDER BY phone_percentage DESC, c.category_name ASC LIMIT 5"),
    HoldoutSpec("fsq_h12", "把数据中排在第一项的类型作为推断口径，在拥有多种类型的地点里，作为第一项出现最多的五种类型是什么？数量相同时按名称排列。", "category_join", "WITH multi AS (SELECT fsq_place_id FROM place_categories GROUP BY fsq_place_id HAVING COUNT(*) > 1) SELECT c.category_name, COUNT(DISTINCT pc.fsq_place_id) AS place_count FROM multi m JOIN place_categories pc ON pc.fsq_place_id = m.fsq_place_id AND pc.is_primary = 1 JOIN categories c ON c.category_id = pc.category_id GROUP BY c.category_id, c.category_name ORDER BY place_count DESC, c.category_name ASC LIMIT 5"),
    HoldoutSpec("fsq_h13", "同时被标注为 Coffee Shop 和 Bakery 的地点有多少个？", "category_join", "SELECT COUNT(DISTINCT coffee.fsq_place_id) AS place_count FROM place_categories coffee JOIN categories cc ON cc.category_id = coffee.category_id AND cc.category_name = 'Coffee Shop' JOIN place_categories bakery ON bakery.fsq_place_id = coffee.fsq_place_id JOIN categories bc ON bc.category_id = bakery.category_id AND bc.category_name = 'Bakery'"),
    HoldoutSpec("fsq_h14", "分类目录的各个层级中，分别还有多少种类型没有关联到任何上海地点？", "category_join", "SELECT c.category_level, COUNT(*) AS unused_category_count FROM categories c LEFT JOIN place_categories pc ON pc.category_id = c.category_id WHERE pc.category_id IS NULL GROUP BY c.category_level ORDER BY c.category_level ASC"),
    HoldoutSpec("fsq_h15", "按分类目录的一级大类汇总，覆盖地点最多的五个大类是什么？数量相同时按大类名称排列。", "category_join", "SELECT c.level1_category_name, COUNT(DISTINCT pc.fsq_place_id) AS place_count FROM place_categories pc JOIN categories c ON c.category_id = pc.category_id WHERE c.level1_category_name IS NOT NULL GROUP BY c.level1_category_name ORDER BY place_count DESC, c.level1_category_name ASC LIMIT 5"),
    HoldoutSpec("fsq_h16", "已经关闭的地点最常见的五种类型是什么？数量相同时按类型名称排列。", "category_join", "SELECT c.category_name, COUNT(DISTINCT p.fsq_place_id) AS closed_place_count FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id JOIN categories c ON c.category_id = pc.category_id WHERE p.date_closed IS NOT NULL GROUP BY c.category_id, c.category_name ORDER BY closed_place_count DESC, c.category_name ASC LIMIT 5"),
    HoldoutSpec("fsq_h17", "只看能够明确映射到行政区的地点，在地点总数不少于五百的区中，联系电话覆盖率最高的五个区是哪些？比例相同时按区名排列。", "district_analysis", "SELECT canonical_district, COUNT(*) AS total_places, SUM(CASE WHEN tel IS NOT NULL AND TRIM(tel) <> '' THEN 1 ELSE 0 END) AS places_with_phone, ROUND(100.0 * SUM(CASE WHEN tel IS NOT NULL AND TRIM(tel) <> '' THEN 1 ELSE 0 END) / COUNT(*), 4) AS phone_percentage FROM places WHERE canonical_district IS NOT NULL GROUP BY canonical_district HAVING COUNT(*) >= 500 ORDER BY phone_percentage DESC, canonical_district ASC LIMIT 5"),
    HoldoutSpec("fsq_h18", "只看能够明确映射到行政区的地点，没有任何类型标注的地点最多的是哪五个区？数量相同时按区名排列。", "district_analysis", "SELECT p.canonical_district, COUNT(DISTINCT p.fsq_place_id) AS uncategorized_place_count FROM places p LEFT JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id WHERE p.canonical_district IS NOT NULL AND pc.fsq_place_id IS NULL GROUP BY p.canonical_district ORDER BY uncategorized_place_count DESC, p.canonical_district ASC LIMIT 5"),
    HoldoutSpec("fsq_h19", "只看能够明确映射到行政区且至少收录一百个地点的区，多类型地点占比最高的五个区是哪些？比例相同时按区名排列。", "district_analysis", "WITH district_totals AS (SELECT canonical_district, COUNT(*) AS total_places FROM places WHERE canonical_district IS NOT NULL GROUP BY canonical_district), multi AS (SELECT fsq_place_id FROM place_categories GROUP BY fsq_place_id HAVING COUNT(*) > 1), district_multi AS (SELECT p.canonical_district, COUNT(DISTINCT p.fsq_place_id) AS multi_category_places FROM places p JOIN multi m ON m.fsq_place_id = p.fsq_place_id WHERE p.canonical_district IS NOT NULL GROUP BY p.canonical_district) SELECT dt.canonical_district, dt.total_places, COALESCE(dm.multi_category_places, 0) AS multi_category_places, ROUND(100.0 * COALESCE(dm.multi_category_places, 0) / dt.total_places, 4) AS multi_category_percentage FROM district_totals dt LEFT JOIN district_multi dm ON dm.canonical_district = dt.canonical_district WHERE dt.total_places >= 100 ORDER BY multi_category_percentage DESC, dt.canonical_district ASC LIMIT 5"),
    HoldoutSpec("fsq_h20", "只看能够明确映射到行政区的地点，浦东和徐汇分别有多少地点在2026年更新？各占本区地点的百分之多少？", "district_analysis", "SELECT canonical_district, COUNT(*) AS total_places, SUM(CASE WHEN date_refreshed LIKE '2026-%' THEN 1 ELSE 0 END) AS refreshed_in_2026, ROUND(100.0 * SUM(CASE WHEN date_refreshed LIKE '2026-%' THEN 1 ELSE 0 END) / COUNT(*), 4) AS refreshed_percentage FROM places WHERE canonical_district IS NOT NULL AND canonical_district IN ('Pudong', 'Xuhui') GROUP BY canonical_district ORDER BY canonical_district ASC"),
)


def _normalized_question(question: str) -> str:
    return re.sub(r"\W+", "", question.casefold(), flags=re.UNICODE)


def _reference_values(paths: tuple[Path, ...]) -> tuple[set[str], set[str]]:
    questions: set[str] = set()
    sql: set[str] = set()
    validator = SQLValidator()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Reference evaluation dataset does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            questions.add(_normalized_question(case["question"]))
            safety = validator.validate(case["gold_sql"])
            sql.add(safety.normalized_sql)
    return questions, sql


def build_holdout(
    database: Path,
    output: Path,
    reference_datasets: tuple[Path, ...] = REFERENCE_DATASETS,
) -> Path:
    """Validate and execute all holdout SQL before atomically writing the dataset."""

    if not database.is_file():
        raise FileNotFoundError(f"FSQ database does not exist: {database}")
    reference_questions, reference_sql = _reference_values(reference_datasets)
    validator = SQLValidator()
    executor = SQLExecutor(database, max_rows=100)
    seen_questions: set[str] = set()
    seen_sql: set[str] = set()
    cases: list[dict[str, object]] = []

    for spec in SPECS:
        normalized_question = _normalized_question(spec.question)
        if normalized_question in reference_questions or normalized_question in seen_questions:
            raise ValueError(f"Duplicate holdout question: {spec.case_id}")
        safety = validator.validate(spec.gold_sql, database_path=database)
        if not safety.safe or safety.syntax_valid is not True:
            raise ValueError(
                f"Gold SQL validation failed for {spec.case_id}: "
                f"{safety.model_dump_json()}"
            )
        if safety.normalized_sql in reference_sql or safety.normalized_sql in seen_sql:
            raise ValueError(f"Duplicate holdout gold SQL: {spec.case_id}")
        result = executor.execute(spec.gold_sql)
        if not result.execution_success or result.truncated:
            raise ValueError(
                f"Gold SQL execution failed for {spec.case_id}: "
                f"{result.model_dump_json()}"
            )
        seen_questions.add(normalized_question)
        seen_sql.add(safety.normalized_sql)
        cases.append(
            {
                "case_id": spec.case_id,
                "question": spec.question,
                "tags": ["holdout", spec.group],
                "gold_sql": spec.gold_sql,
                "expected_result": {
                    "columns": result.columns,
                    "rows": result.rows,
                },
                "ordered": spec.ordered,
                "strict_columns": False,
            }
        )

    payload = {
        "dataset_name": "fsq_shanghai_holdout_v1",
        "database_id": database.stem,
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(output)
    return output.resolve()


def main() -> None:
    """Parse paths and build the deterministic holdout dataset."""

    parser = argparse.ArgumentParser(
        description="从真实 FSQ SQLite 数据生成独立 holdout 评测集。"
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    created = build_holdout(args.database, args.output)
    print(f"FSQ holdout evaluation dataset created: {created}")


if __name__ == "__main__":
    main()
