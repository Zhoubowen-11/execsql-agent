"""Generate a 480-case, difficulty-layered RL-hard V2 train-side dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from grpo_hard_dataset import (
    DISTRICTS,
    HardCase,
    _category_names,
    _inspect_history,
    _literal,
    _missing,
    _present,
    _tool_schemas,
)
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from execsql_agent.tools.registry import ToolRegistry
from execsql_agent.tools.sql_executor import SQLExecutor
from execsql_agent.tools.sql_validator import SQLValidator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / "data/fsq/shanghai_places.db"
DEFAULT_MODEL = Path("/root/autodl-tmp/models/Qwen3-8B")
DEFAULT_V1 = PROJECT_ROOT / "data/fsq/train/grpo/rl_train_hard_v1_candidates.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/fsq/train/grpo/rl_train_hard_v2_candidates.jsonl"
DEFAULT_SUPPLEMENT_OUTPUT = PROJECT_ROOT / "data/fsq/train/grpo/rl_train_hard_v2_supplement.jsonl"

FAMILY_CODES = {
    "aggregation_scope": "a",
    "filtering_semantics": "f",
    "ranking": "r",
    "ties": "t",
    "multi_category_semantics": "m",
    "output_shape": "o",
}


def _signature(family: str, parameters: dict[str, object]) -> str:
    raw = json.dumps(
        {"family": family, "parameters": parameters},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _difficulty(index: int) -> str:
    if index < 20:
        return "easy"
    if index < 60:
        return "boundary"
    return "hard"


def _make(
    family: str,
    index: int,
    *,
    question: str,
    sql: str,
    tables: list[str],
    parameters: dict[str, object],
    difficulty_override: str | None = None,
) -> HardCase:
    difficulty = difficulty_override or _difficulty(index)
    parameter_payload = {"difficulty": difficulty, **parameters}
    return HardCase(
        case_id=f"rlh_v2_{FAMILY_CODES[family]}_{index + 1:03d}",
        template_family=family,
        difficulty=difficulty,
        question=question,
        oracle_sql=sql,
        inspect_tables=tables,
        parameter_signature=_signature(family, parameter_payload),
    )


def aggregation_scope(categories: list[str]) -> list[HardCase]:
    cases: list[HardCase] = []
    for index in range(80):
        district_cn, district_en = DISTRICTS[(index * 5) % len(DISTRICTS)]
        category = categories[(index * 7) % len(categories)]
        difficulty = _difficulty(index)
        if difficulty == "easy":
            condition = ""
            condition_text = ""
            parameters: dict[str, object] = {
                "district": district_en,
                "category": category,
                "scope": "distinct_base",
            }
        elif difficulty == "boundary":
            variant = index % 4
            if variant == 0:
                condition = f" AND {_present('p.tel')}"
                condition_text = "且留有联系电话"
            elif variant == 1:
                condition = f" AND {_present('p.website')}"
                condition_text = "且留有官网"
            elif variant == 2:
                condition = f" AND {_present('p.tel')} AND {_present('p.website')}"
                condition_text = "且电话和官网都完整"
            else:
                condition = " AND pc.is_primary = 1"
                condition_text = "且该类型是源分类数组第一项"
            parameters = {
                "district": district_en,
                "category": category,
                "scope": "boundary",
                "variant": variant,
            }
        else:
            second_category = categories[(index * 7 + 11) % len(categories)]
            condition = (
                f" AND c.category_name IN ({_literal(category)}, "
                f"{_literal(second_category)}) AND {_present('p.tel')}"
            )
            condition_text = f"、属于 {category} 或 {second_category}，并留有联系电话"
            parameters = {
                "district": district_en,
                "category_a": category,
                "category_b": second_category,
                "scope": "union_distinct",
            }
        category_filter = (
            f"c.category_name = {_literal(category)}" if difficulty != "hard" else "1 = 1"
        )
        sql = (
            "SELECT COUNT(DISTINCT p.fsq_place_id) AS place_count "
            "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
            "JOIN categories c ON c.category_id = pc.category_id "
            f"WHERE p.canonical_district = {_literal(district_en)} "
            f"AND {_missing('p.date_closed')} AND {category_filter}{condition}"
        )
        base_text = (
            f"在能够明确归入{district_cn}且未填写关闭日期的地点中，属于 {category} "
            "的独立地点有多少个"
            if difficulty != "hard"
            else f"在能够明确归入{district_cn}且未填写关闭日期的地点中"
            f"{condition_text}的独立地点有多少个"
        )
        question = (
            f"{base_text}{condition_text if difficulty != 'hard' else ''}？"
            "同一地点即使有多条分类关联也只计算一次。"
        )
        cases.append(
            _make(
                "aggregation_scope",
                index,
                question=question,
                sql=sql,
                tables=["places", "place_categories", "categories"],
                parameters=parameters,
            )
        )
    return cases


def filtering_semantics(categories: list[str]) -> list[HardCase]:
    fields = [
        ("tel", "联系电话"),
        ("website", "官网"),
        ("postcode", "邮政编码"),
        ("email", "电子邮箱"),
        ("locality", "原始城市标注"),
    ]
    cases: list[HardCase] = []
    for index in range(80):
        district_cn, district_en = DISTRICTS[(index * 5) % len(DISTRICTS)]
        category = categories[(index * 7) % len(categories)]
        field, label = fields[index % len(fields)]
        present = index % 2 == 0
        status = "有效填写了" if present else "没有有效填写"
        main_condition = _present("p." + field) if present else _missing("p." + field)
        difficulty = _difficulty(index)
        if difficulty == "easy":
            join = ""
            category_condition = ""
            question = (
                f"{district_cn}未填写关闭日期的地点中，{status}{label}的有多少个？"
                "空白内容按未填写处理。"
            )
            tables = ["places"]
        elif difficulty == "boundary":
            join = (
                " JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
                "JOIN categories c ON c.category_id = pc.category_id"
            )
            category_condition = f" AND c.category_name = {_literal(category)}"
            question = (
                f"{district_cn}未填写关闭日期的 {category} 地点中，{status}{label}的"
                "独立地点有多少个？"
                "空字符串也按未填写处理。"
            )
            tables = ["places", "place_categories", "categories"]
        else:
            join = (
                " JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
                "JOIN categories c ON c.category_id = pc.category_id"
            )
            if index % 2 == 0:
                extra_condition = f"{_present('p.tel')} AND {_missing('p.website')}"
                extra_text = "有电话但没有有效官网"
            else:
                extra_condition = f"{_missing('p.tel')} AND {_present('p.website')}"
                extra_text = "没有有效电话但有官网"
            main_condition = extra_condition
            category_condition = (
                f" AND c.category_name = {_literal(category)} AND pc.is_primary = 1"
            )
            question = (
                f"{district_cn}未填写关闭日期、并把 {category} 作为源分类数组第一项的"
                f"地点中，{extra_text}的"
                "独立地点有多少个？NULL 和空字符串都视为缺失。"
            )
            tables = ["places", "place_categories", "categories"]
        sql = (
            "SELECT COUNT(DISTINCT p.fsq_place_id) AS place_count FROM places p"
            f"{join} WHERE p.canonical_district = {_literal(district_en)} "
            f"AND {_missing('p.date_closed')} AND {main_condition}{category_condition}"
        )
        cases.append(
            _make(
                "filtering_semantics",
                index,
                question=question,
                sql=sql,
                tables=tables,
                parameters={
                    "district": district_en,
                    "category": category if difficulty != "easy" else None,
                    "field": field,
                    "present": present,
                    "shape": difficulty,
                },
            )
        )
    return cases


def ranking(categories: list[str]) -> list[HardCase]:
    cases: list[HardCase] = []
    for index in range(80):
        category = categories[(index * 7) % len(categories)]
        district_cn, district_en = DISTRICTS[(index * 5) % len(DISTRICTS)]
        k = (3, 5, 7, 10)[index % 4]
        difficulty = _difficulty(index)
        if difficulty == "easy":
            easy_fields = [
                ("tel", "联系电话"),
                ("website", "官网"),
                ("postcode", "邮政编码"),
                ("email", "电子邮箱"),
                ("locality", "原始城市标注"),
            ]
            field, label = easy_fields[index % len(easy_fields)]
            field_present = (index // len(easy_fields)) % 2 == 0
            closed_missing = index // (len(easy_fields) * 2) == 0
            field_condition = _present(field) if field_present else _missing(field)
            closed_condition = (
                _missing("date_closed") if closed_missing else _present("date_closed")
            )
            field_status = "有效填写了" if field_present else "没有有效填写"
            closed_status = "未填写关闭日期" if closed_missing else "填写了关闭日期"
            sql = (
                "SELECT canonical_district, COUNT(*) AS place_count FROM places "
                "WHERE canonical_district IS NOT NULL "
                f"AND {field_condition} AND {closed_condition} "
                "GROUP BY canonical_district "
                "ORDER BY place_count DESC, canonical_district ASC "
                f"LIMIT {k}"
            )
            question = (
                f"在可明确识别行政区、{closed_status}且{field_status}{label}的地点中，"
                f"数量最多的前 {k} 个区是哪些？"
                "同数时按行政区英文名升序排列。"
            )
            tables = ["places"]
            parameters = {
                "scope": "district",
                "field": field,
                "field_present": field_present,
                "closed_missing": closed_missing,
                "k": k,
            }
        elif difficulty == "boundary":
            condition = "" if index % 2 == 0 else f" AND {_present('p.tel')}"
            condition_text = "" if index % 2 == 0 else "且留有联系电话"
            sql = (
                "SELECT p.canonical_district, COUNT(DISTINCT p.fsq_place_id) AS place_count "
                "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
                "JOIN categories c ON c.category_id = pc.category_id "
                "WHERE p.canonical_district IS NOT NULL "
                f"AND c.category_name = {_literal(category)}{condition} "
                "GROUP BY p.canonical_district "
                "ORDER BY place_count DESC, p.canonical_district ASC "
                f"LIMIT {k}"
            )
            question = (
                f"只看行政区可明确识别{condition_text}的 {category} 地点，数量最多的前 {k} 个区"
                "是哪些？数量相同时按行政区英文名升序。"
            )
            tables = ["places", "place_categories", "categories"]
            parameters = {"scope": "category_district", "category": category, "k": k}
        else:
            closed_missing = (index // len(DISTRICTS)) % 2 == 0
            closed_condition = (
                _missing("p.date_closed") if closed_missing else _present("p.date_closed")
            )
            closed_text = "未填写关闭日期" if closed_missing else "填写了关闭日期"
            condition = (
                f"{_present('p.tel')} AND {_present('p.website')} "
                f"AND {closed_condition} AND pc.is_primary = 1"
            )
            sql = (
                "SELECT c.category_name, COUNT(DISTINCT p.fsq_place_id) AS place_count "
                "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
                "JOIN categories c ON c.category_id = pc.category_id "
                f"WHERE p.canonical_district = {_literal(district_en)} AND {condition} "
                "GROUP BY c.category_id, c.category_name "
                "ORDER BY place_count DESC, c.category_name ASC "
                f"LIMIT {k}"
            )
            question = (
                f"{district_cn}电话和官网都完整、{closed_text}的地点中，按源分类数组第一项"
                "统计，覆盖独立地点"
                f"最多的前 {k} 种类型是什么？同数时按类型英文名升序。"
            )
            tables = ["places", "place_categories", "categories"]
            parameters = {
                "scope": "primary_category",
                "district": district_en,
                "k": k,
                "closed_missing": closed_missing,
            }
        cases.append(
            _make(
                "ranking",
                index,
                question=question,
                sql=sql,
                tables=tables,
                parameters=parameters,
            )
        )
    return cases


def ties(categories: list[str]) -> list[HardCase]:
    cases: list[HardCase] = []
    for index in range(80):
        category = categories[(index * 7) % len(categories)]
        k = (1, 3, 5, 7)[index % 4]
        difficulty = _difficulty(index)
        contact = "" if index % 2 == 0 else f" AND {_present('p.tel')}"
        contact_text = "" if index % 2 == 0 else "且留有联系电话"
        boundary = f" AND {_missing('p.date_closed')}" if difficulty == "boundary" else ""
        boundary_text = "且未填写关闭日期" if difficulty == "boundary" else ""
        base = (
            "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
            "JOIN categories c ON c.category_id = pc.category_id "
            "WHERE p.canonical_district IS NOT NULL "
            f"AND c.category_name = {_literal(category)}{contact}{boundary} "
        )
        if difficulty in {"easy", "boundary"}:
            sql = (
                "SELECT p.canonical_district, COUNT(DISTINCT p.fsq_place_id) AS place_count "
                + base
                + "GROUP BY p.canonical_district "
                "ORDER BY place_count DESC, p.canonical_district ASC "
                f"LIMIT {k}"
            )
            count_text = "一个" if k == 1 else f"前 {k} 个"
            question = (
                f"可明确识别行政区的 {category} 地点中，{contact_text}{boundary_text}"
                f"数量最多的{count_text}区"
                "是哪些？如数量相同，按行政区英文名升序决定取舍，不额外扩展结果。"
            )
            parameters = {
                "tie_policy": "stable_cutoff",
                "category": category,
                "contact": bool(contact),
                "k": k,
            }
        else:
            sql = (
                "WITH district_counts AS (SELECT p.canonical_district, "
                "COUNT(DISTINCT p.fsq_place_id) AS place_count "
                + base
                + "GROUP BY p.canonical_district), ranked AS ("
                "SELECT canonical_district, place_count, "
                "DENSE_RANK() OVER (ORDER BY place_count DESC) AS place_rank "
                "FROM district_counts) SELECT canonical_district, place_count "
                "FROM ranked WHERE place_rank <= 2 "
                "ORDER BY place_rank ASC, canonical_district ASC"
            )
            question = (
                f"可明确识别行政区的 {category} 地点中，请返回数量排名前两档的所有行政区；"
                "处在同一数量档的区都要保留，并按英文名升序排列。"
            )
            parameters = {
                "tie_policy": "dense_rank_two_levels",
                "category": category,
                "contact": bool(contact),
            }
        cases.append(
            _make(
                "ties",
                index,
                question=question,
                sql=sql,
                tables=["places", "place_categories", "categories"],
                parameters=parameters,
            )
        )
    return cases


def multi_category_semantics(categories: list[str]) -> list[HardCase]:
    cases: list[HardCase] = []
    for index in range(80):
        district_cn, district_en = DISTRICTS[(index * 5) % len(DISTRICTS)]
        category = categories[(index * 7) % len(categories)]
        threshold = (1, 2, 3, 4)[(index + index // len(DISTRICTS)) % 4]
        difficulty = _difficulty(index)
        if difficulty == "easy":
            where = (
                f"WHERE p.canonical_district = {_literal(district_en)} "
                f"AND {_missing('p.date_closed')} "
            )
            question = (
                f"{district_cn}未填写关闭日期且至少标注 {threshold + 1} 种不同类型的"
                "独立地点有多少个？"
                "同一种类型的重复关联只能算一次。"
            )
            parameters = {"scope": "district", "district": district_en, "n": threshold + 1}
        elif difficulty == "boundary":
            contact_fields = [
                ("tel", "联系电话"),
                ("website", "官网"),
                ("postcode", "邮政编码"),
                ("email", "电子邮箱"),
                ("locality", "原始城市标注"),
            ]
            field, label = contact_fields[index % len(contact_fields)]
            where = (
                f"WHERE p.canonical_district = {_literal(district_en)} "
                f"AND {_missing('p.date_closed')} AND {_present('p.' + field)} "
            )
            question = (
                f"{district_cn}未填写关闭日期、留有{label}，并至少标注 "
                f"{threshold + 1} 种不同类型的独立地点"
                "有多少个？分类按不同编号去重。"
            )
            parameters = {
                "scope": "district_contact",
                "district": district_en,
                "field": field,
                "n": threshold + 1,
            }
        else:
            where = (
                f"WHERE p.canonical_district = {_literal(district_en)} "
                f"AND {_missing('p.date_closed')} "
            )
            question = (
                f"{district_cn}未填写关闭日期、包含 {category} 类型，且总共至少标注 "
                f"{threshold + 1} 种不同类型"
                "的独立地点有多少个？"
            )
            parameters = {
                "scope": "contains_category",
                "district": district_en,
                "category": category,
                "n": threshold + 1,
            }
        having = f"HAVING COUNT(DISTINCT pc.category_id) > {threshold}"
        if difficulty == "hard":
            having += (
                f" AND SUM(CASE WHEN c.category_name = {_literal(category)} THEN 1 ELSE 0 END) > 0"
            )
            join = (
                "JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
                "JOIN categories c ON c.category_id = pc.category_id "
            )
            tables = ["places", "place_categories", "categories"]
        else:
            join = "JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
            tables = ["places", "place_categories"]
        sql = (
            "SELECT COUNT(*) AS place_count FROM (SELECT p.fsq_place_id FROM places p "
            + join
            + where
            + "GROUP BY p.fsq_place_id "
            + having
            + ") AS qualified_places"
        )
        cases.append(
            _make(
                "multi_category_semantics",
                index,
                question=question,
                sql=sql,
                tables=tables,
                parameters=parameters,
            )
        )
    return cases


def output_shape(categories: list[str]) -> list[HardCase]:
    cases: list[HardCase] = []
    for index in range(80):
        district_cn, district_en = DISTRICTS[(index * 5) % len(DISTRICTS)]
        category = categories[(index * 7) % len(categories)]
        difficulty = _difficulty(index)
        if difficulty == "easy":
            metric_fields = [
                ("tel", "联系电话", "with_phone"),
                ("website", "官网", "with_website"),
                ("postcode", "邮政编码", "with_postcode"),
                ("email", "电子邮箱", "with_email"),
                ("locality", "原始城市标注", "with_locality"),
            ]
            metric_field, metric_label, metric_alias = metric_fields[index % len(metric_fields)]
            sql = (
                "SELECT COUNT(*) AS total_places, "
                f"SUM(CASE WHEN {_present(metric_field)} THEN 1 ELSE 0 END) "
                f"AS {metric_alias} FROM places "
                f"WHERE canonical_district = {_literal(district_en)} "
                f"AND {_missing('date_closed')}"
            )
            question = (
                f"请在一行中给出{district_cn}未填写关闭日期的地点总数和有效填写"
                f"{metric_label}的地点数。"
            )
            tables = ["places"]
            parameters = {
                "shape": "two_scalars",
                "district": district_en,
                "metric_field": metric_field,
            }
        elif difficulty == "boundary":
            sql = (
                "SELECT COUNT(DISTINCT p.fsq_place_id) AS total_places, "
                "COUNT(DISTINCT CASE WHEN p.tel IS NOT NULL AND TRIM(p.tel) <> '' "
                "THEN p.fsq_place_id END) AS with_phone, "
                "COUNT(DISTINCT CASE WHEN p.website IS NOT NULL AND TRIM(p.website) <> '' "
                "THEN p.fsq_place_id END) AS with_website, "
                "ROUND(100.0 * COUNT(DISTINCT CASE WHEN p.tel IS NOT NULL "
                "AND TRIM(p.tel) <> '' THEN p.fsq_place_id END) / "
                "NULLIF(COUNT(DISTINCT p.fsq_place_id), 0), 2) AS phone_percentage "
                "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
                "JOIN categories c ON c.category_id = pc.category_id "
                f"WHERE p.canonical_district = {_literal(district_en)} "
                f"AND {_missing('p.date_closed')} "
                f"AND c.category_name = {_literal(category)}"
            )
            question = (
                f"请在一行中给出{district_cn}未填写关闭日期的 {category} 地点总数、"
                "有电话的数量、有官网的数量，"
                "以及电话覆盖率；按独立地点计数，百分比保留两位小数。"
            )
            tables = ["places", "place_categories", "categories"]
            parameters = {"shape": "four_metrics", "district": district_en, "category": category}
        else:
            k = (3, 5, 7, 10)[index % 4]
            sql = (
                "SELECT p.canonical_district, "
                "COUNT(DISTINCT p.fsq_place_id) AS total_places, "
                "COUNT(DISTINCT CASE WHEN p.tel IS NOT NULL AND TRIM(p.tel) <> '' "
                "THEN p.fsq_place_id END) AS with_phone, "
                "ROUND(100.0 * COUNT(DISTINCT CASE WHEN p.tel IS NOT NULL "
                "AND TRIM(p.tel) <> '' THEN p.fsq_place_id END) / "
                "NULLIF(COUNT(DISTINCT p.fsq_place_id), 0), 2) AS phone_percentage "
                "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
                "JOIN categories c ON c.category_id = pc.category_id "
                "WHERE p.canonical_district IS NOT NULL "
                f"AND {_missing('p.date_closed')} "
                f"AND c.category_name = {_literal(category)} "
                "GROUP BY p.canonical_district "
                "ORDER BY total_places DESC, p.canonical_district ASC "
                f"LIMIT {k}"
            )
            question = (
                f"对未填写关闭日期的 {category} 地点总数最多的前 {k} 个可识别行政区，"
                "请同时给出区名、地点总数、"
                "有电话的数量和电话覆盖率；同数按英文区名升序，百分比保留两位小数。"
            )
            tables = ["places", "place_categories", "categories"]
            parameters = {"shape": "grouped_metrics", "category": category, "k": k}
        cases.append(
            _make(
                "output_shape",
                index,
                question=question,
                sql=sql,
                tables=tables,
                parameters=parameters,
            )
        )
    return cases


def generate_v2_cases(database: Path) -> list[HardCase]:
    categories = _category_names(database, limit=40)
    factories = [
        aggregation_scope,
        filtering_semantics,
        ranking,
        ties,
        multi_category_semantics,
        output_shape,
    ]
    cases = [case for factory in factories for case in factory(categories)]
    if len(cases) != 480:
        raise ValueError(f"expected 480 V2 cases, found {len(cases)}")
    return cases


def generate_supplement_cases(database: Path) -> list[HardCase]:
    """Generate 120 boundary-focused cases for the three under-mixed families."""
    categories = _category_names(database, limit=40)
    cases: list[HardCase] = []
    fields = [("tel", "联系电话"), ("website", "官网"), ("email", "电子邮箱")]
    for offset in range(20):
        district_cn, district_en = DISTRICTS[(offset * 3 + 1) % len(DISTRICTS)]
        category = categories[(offset * 9 + 3) % len(categories)]
        field, label = fields[offset % len(fields)]
        sql = (
            "SELECT COUNT(DISTINCT p.fsq_place_id) AS place_count FROM places p "
            "JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
            "JOIN categories c ON c.category_id = pc.category_id "
            f"WHERE p.canonical_district = {_literal(district_en)} "
            f"AND c.category_name = {_literal(category)} AND {_present('p.' + field)}"
        )
        cases.append(
            _make(
                "aggregation_scope",
                80 + offset,
                question=(
                    f"在可明确归入{district_cn}的 {category} 地点中，"
                    f"有效填写{label}的独立地点有多少个？"
                    "同一地点即使有重复分类关联也只计算一次。"
                ),
                sql=sql,
                tables=["places", "place_categories", "categories"],
                difficulty_override="boundary",
                parameters={
                    "supplement": True,
                    "district": district_en,
                    "category": category,
                    "field": field,
                },
            )
        )
    for offset in range(60):
        district_cn, district_en = DISTRICTS[(offset * 5 + 2) % len(DISTRICTS)]
        minimum = (2, 3, 4)[offset % 3]
        k = (3, 5, 8)[(offset // 3) % 3]
        sql = (
            "SELECT p.fsq_place_id, p.name, COUNT(DISTINCT pc.category_id) AS category_count "
            "FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id "
            f"WHERE p.canonical_district = {_literal(district_en)} "
            "GROUP BY p.fsq_place_id, p.name "
            f"HAVING COUNT(DISTINCT pc.category_id) >= {minimum} "
            "ORDER BY category_count DESC, p.fsq_place_id ASC "
            f"LIMIT {k}"
        )
        cases.append(
            _make(
                "multi_category_semantics",
                80 + offset,
                question=(
                    f"在可明确归入{district_cn}的地点中，"
                    f"列出分类标注不少于 {minimum} 种的前 {k} 个地点，"
                    "给出地点编号、名称和不同分类数；先按分类数从多到少，同数按地点编号升序。"
                ),
                sql=sql,
                tables=["places", "place_categories"],
                difficulty_override="boundary",
                parameters={
                    "supplement": True,
                    "district": district_en,
                    "minimum": minimum,
                    "k": k,
                },
            )
        )
    for offset in range(40):
        district_cn, district_en = DISTRICTS[(offset * 7 + 4) % len(DISTRICTS)]
        field, label = fields[offset % len(fields)]
        closed = offset % 2 == 1
        date_condition = _present("date_closed") if closed else _missing("date_closed")
        date_text = "已填写关闭日期" if closed else "未填写关闭日期"
        sql = (
            "SELECT COUNT(*) AS total_places, "
            f"SUM(CASE WHEN {_present(field)} THEN 1 ELSE 0 END) AS populated_places "
            "FROM places "
            f"WHERE canonical_district = {_literal(district_en)} AND {date_condition}"
        )
        cases.append(
            _make(
                "output_shape",
                80 + offset,
                question=(
                    f"请在同一行完整给出{district_cn}{date_text}的地点总数，以及其中有效填写{label}的地点数。"
                ),
                sql=sql,
                tables=["places"],
                difficulty_override="boundary",
                parameters={
                    "supplement": True,
                    "district": district_en,
                    "field": field,
                    "closed": closed,
                },
            )
        )
    if len(cases) != 120:
        raise ValueError(f"expected 120 supplement cases, found {len(cases)}")
    return cases


def _load_existing(path: Path, validator: SQLValidator) -> tuple[set[str], set[str]]:
    if not path.exists():
        return set(), set()
    questions: set[str] = set()
    sql: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            questions.add("".join(str(item["question"]).split()).casefold())
            sql.add(validator.validate(str(item["oracle_sql"])).normalized_sql)
    return questions, sql


def build_items(
    *,
    database: Path,
    tokenizer: PreTrainedTokenizerBase,
    existing_path: Path,
    cases: list[HardCase] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry = ToolRegistry(database)
    tools = _tool_schemas(registry)
    validator = SQLValidator()
    executor = SQLExecutor(database, max_rows=100)
    existing_questions, existing_sql = _load_existing(existing_path, validator)
    cases = generate_v2_cases(database) if cases is None else cases
    items: list[dict[str, Any]] = []
    lengths: list[int] = []
    normalized_questions: list[str] = []
    normalized_sql: list[str] = []
    signatures: list[str] = []
    for case in cases:
        question_key = "".join(case.question.split()).casefold()
        safety = validator.validate(case.oracle_sql, database_path=database)
        if not safety.safe or safety.syntax_valid is not True:
            raise ValueError(f"{case.case_id}: oracle SQL failed validation")
        if question_key in existing_questions or safety.normalized_sql in existing_sql:
            raise ValueError(f"{case.case_id}: collides with RL-hard V1")
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
            raise ValueError(f"{case.case_id}: reward metadata leaked into prompt")
        token_count = len(tokenizer(prompt, add_special_tokens=False).input_ids)
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
        lengths.append(token_count)
        normalized_questions.append(question_key)
        normalized_sql.append(safety.normalized_sql)
        signatures.append(case.parameter_signature)
    duplicates = {
        "question": len(normalized_questions) - len(set(normalized_questions)),
        "oracle_sql": len(normalized_sql) - len(set(normalized_sql)),
        "parameter_signature": len(signatures) - len(set(signatures)),
    }
    if any(duplicates.values()):
        duplicate_details: dict[str, list[list[str]]] = {}
        for name, values in (
            ("question", normalized_questions),
            ("oracle_sql", normalized_sql),
            ("parameter_signature", signatures),
        ):
            grouped: dict[str, list[str]] = {}
            for case, value in zip(cases, values, strict=True):
                grouped.setdefault(value, []).append(case.case_id)
            duplicate_details[name] = [ids for ids in grouped.values() if len(ids) > 1]
        raise ValueError(f"V2 duplicate audit failed: {duplicates}; details={duplicate_details}")
    stats = {
        "candidate_count": len(items),
        "family_counts": dict(Counter(case.template_family for case in cases)),
        "difficulty_counts": dict(Counter(case.difficulty for case in cases)),
        "family_difficulty_counts": {
            family: dict(
                Counter(case.difficulty for case in cases if case.template_family == family)
            )
            for family in FAMILY_CODES
            if any(case.template_family == family for case in cases)
        },
        "prompt_token_min": min(lengths),
        "prompt_token_mean": statistics.fmean(lengths),
        "prompt_token_max": max(lengths),
        "duplicates": duplicates,
        "v1_question_collisions": 0,
        "v1_sql_collisions": 0,
        "leakage_count": 0,
    }
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
    parser.add_argument("--existing-v1", type=Path, default=DEFAULT_V1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--supplement", action="store_true")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    items, stats = build_items(
        database=args.database,
        tokenizer=tokenizer,
        existing_path=args.existing_v1,
        cases=generate_supplement_cases(args.database) if args.supplement else None,
    )
    _atomic_jsonl(args.output, items)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
