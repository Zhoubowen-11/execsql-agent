"""Build validated FSQ Shanghai evaluation datasets from real SQLite results."""

# ruff: noqa: E501 -- declarative SQL cases stay one literal per auditable case.

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from execsql_agent.models import (
    BehaviorDataset,
    BehaviorScenario,
    EvaluationCase,
    EvaluationDataset,
    ExpectedResult,
    LLMResponse,
    ResponseMode,
    ToolCallRequest,
)
from execsql_agent.tools.sql_executor import SQLExecutor
from execsql_agent.tools.sql_validator import SQLValidator

DEFAULT_DATABASE = Path("data/fsq/shanghai_places.db")
DEFAULT_OUTPUT_DIR = Path("data/fsq/eval")
DISTRICT_NOTE = (
    "行政区口径仅包含 canonical_district 非空记录；当前映射覆盖率为 50.5546%。"
)


@dataclass(frozen=True)
class CaseSpec:
    id: str
    question: str
    group: Literal["single_table", "aggregate_topk", "category_join", "district", "quality"]
    difficulty: str
    tables: tuple[str, ...]
    shape: str
    sql: str
    notes: str | None = None


SPECS = [
    CaseSpec("fsq_s01", "上海地点库一共收录了多少个地点？", "single_table", "easy", ("places",), "scalar_count", "SELECT COUNT(*) AS place_count FROM places"),
    CaseSpec("fsq_s02", "有官方网站的地点有多少家？", "single_table", "easy", ("places",), "filtered_count", "SELECT COUNT(*) AS place_count FROM places WHERE website IS NOT NULL AND TRIM(website) <> ''"),
    CaseSpec("fsq_s03", "留下联系电话的地点有多少家？", "single_table", "easy", ("places",), "filtered_count", "SELECT COUNT(*) AS place_count FROM places WHERE tel IS NOT NULL AND TRIM(tel) <> ''"),
    CaseSpec("fsq_s04", "目前记录为已经关闭的地点有多少个？", "single_table", "easy", ("places",), "filtered_count", "SELECT COUNT(*) AS closed_place_count FROM places WHERE date_closed IS NOT NULL"),
    CaseSpec("fsq_s05", "这些地点覆盖的最西、最东、最南和最北坐标分别是多少？", "single_table", "medium", ("places",), "numeric_extrema", "SELECT MIN(longitude) AS west, MAX(longitude) AS east, MIN(latitude) AS south, MAX(latitude) AS north FROM places"),
    CaseSpec("fsq_s06", "数据库中最近一次地点更新时间是哪一天？这一天共有多少个地点被更新？", "single_table", "easy", ("places",), "latest_date_count", "SELECT MAX(date_refreshed) AS latest_refresh_date, COUNT(*) AS updated_place_count FROM places WHERE date_refreshed = (SELECT MAX(date_refreshed) FROM places)"),
    CaseSpec("fsq_a01", "原始地点信息里，出现次数最多的十个区域名称是什么？", "aggregate_topk", "medium", ("places",), "grouped_top_k", "SELECT locality, COUNT(*) AS place_count FROM places WHERE locality IS NOT NULL AND TRIM(locality) <> '' GROUP BY locality ORDER BY place_count DESC, locality ASC LIMIT 10", "这里展示原始区域名称，未合并语言和拼写变体。"),
    CaseSpec("fsq_a02", "重名地点最多的十个名称是什么，各有多少个？", "aggregate_topk", "medium", ("places",), "grouped_top_k", "SELECT name, COUNT(*) AS place_count FROM places WHERE name IS NOT NULL AND TRIM(name) <> '' GROUP BY name HAVING COUNT(*) > 1 ORDER BY place_count DESC, name ASC LIMIT 10"),
    CaseSpec("fsq_a03", "地点最集中的十个邮政编码是哪些？", "aggregate_topk", "medium", ("places",), "grouped_top_k", "SELECT postcode, COUNT(*) AS place_count FROM places WHERE postcode IS NOT NULL AND TRIM(postcode) <> '' GROUP BY postcode ORDER BY place_count DESC, postcode ASC LIMIT 10"),
    CaseSpec("fsq_a04", "按创建年份看，哪五年新增的地点最多？", "aggregate_topk", "medium", ("places",), "date_grouped_top_k", "SELECT SUBSTR(date_created, 1, 4) AS created_year, COUNT(*) AS place_count FROM places WHERE date_created IS NOT NULL GROUP BY created_year ORDER BY place_count DESC, created_year ASC LIMIT 5"),
    CaseSpec("fsq_a05", "按最近更新时间看，记录量最多的五个年份是哪几年？", "aggregate_topk", "medium", ("places",), "date_grouped_top_k", "SELECT SUBSTR(date_refreshed, 1, 4) AS refreshed_year, COUNT(*) AS place_count FROM places WHERE date_refreshed IS NOT NULL GROUP BY refreshed_year ORDER BY place_count DESC, refreshed_year ASC LIMIT 5"),
    CaseSpec("fsq_a06", "已经关闭的地点主要集中在哪五个关闭年份？", "aggregate_topk", "medium", ("places",), "date_grouped_top_k", "SELECT SUBSTR(date_closed, 1, 4) AS closed_year, COUNT(*) AS place_count FROM places WHERE date_closed IS NOT NULL GROUP BY closed_year ORDER BY place_count DESC, closed_year ASC LIMIT 5"),
    CaseSpec("fsq_a07", "同时留有官网和电话、只留一种联系方式、两种都没留的地点各有多少？", "aggregate_topk", "hard", ("places",), "conditional_aggregation", "SELECT CASE WHEN website IS NOT NULL AND TRIM(website) <> '' AND tel IS NOT NULL AND TRIM(tel) <> '' THEN 'both' WHEN (website IS NOT NULL AND TRIM(website) <> '') OR (tel IS NOT NULL AND TRIM(tel) <> '') THEN 'one' ELSE 'neither' END AS contact_status, COUNT(*) AS place_count FROM places GROUP BY contact_status ORDER BY CASE contact_status WHEN 'both' THEN 1 WHEN 'one' THEN 2 ELSE 3 END"),
    CaseSpec("fsq_a08", "最近更新的十个地点有哪些？如果日期相同，按名称排列。", "aggregate_topk", "easy", ("places",), "ordered_top_k", "SELECT name, date_refreshed FROM places WHERE date_refreshed IS NOT NULL ORDER BY date_refreshed DESC, name ASC, fsq_place_id ASC LIMIT 10"),
    CaseSpec("fsq_c01", "上海最常见的十种地点类型是什么？", "category_join", "medium", ("places", "place_categories", "categories"), "three_table_grouped_top_k", "SELECT c.category_name, COUNT(DISTINCT p.fsq_place_id) AS place_count FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id JOIN categories c ON c.category_id = pc.category_id GROUP BY c.category_id, c.category_name ORDER BY place_count DESC, c.category_name ASC LIMIT 10"),
    CaseSpec("fsq_c02", "浦东最常见的五种地点类型有哪些？", "category_join", "medium", ("places", "place_categories", "categories"), "district_category_top_k", "SELECT c.category_name, COUNT(DISTINCT p.fsq_place_id) AS place_count FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id JOIN categories c ON c.category_id = pc.category_id WHERE p.canonical_district = 'Pudong' GROUP BY c.category_id, c.category_name ORDER BY place_count DESC, c.category_name ASC LIMIT 5", DISTRICT_NOTE),
    CaseSpec("fsq_c03", "徐汇最常见的五种地点类型有哪些？", "category_join", "medium", ("places", "place_categories", "categories"), "district_category_top_k", "SELECT c.category_name, COUNT(DISTINCT p.fsq_place_id) AS place_count FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id JOIN categories c ON c.category_id = pc.category_id WHERE p.canonical_district = 'Xuhui' GROUP BY c.category_id, c.category_name ORDER BY place_count DESC, c.category_name ASC LIMIT 5", DISTRICT_NOTE),
    CaseSpec("fsq_c04", "被标注为多种类型的地点一共有多少个？", "category_join", "medium", ("place_categories",), "grouped_subquery_count", "SELECT COUNT(*) AS multi_category_place_count FROM (SELECT fsq_place_id FROM place_categories GROUP BY fsq_place_id HAVING COUNT(*) > 1)"),
    CaseSpec("fsq_c05", "哪些地点类型最常和其他类型一起出现？列出前十种。", "category_join", "hard", ("places", "place_categories", "categories"), "multi_category_top_k", "WITH multi AS (SELECT fsq_place_id FROM place_categories GROUP BY fsq_place_id HAVING COUNT(*) > 1) SELECT c.category_name, COUNT(DISTINCT pc.fsq_place_id) AS place_count FROM multi m JOIN place_categories pc ON pc.fsq_place_id = m.fsq_place_id JOIN categories c ON c.category_id = pc.category_id GROUP BY c.category_id, c.category_name ORDER BY place_count DESC, c.category_name ASC LIMIT 10"),
    CaseSpec("fsq_c06", "按数据中排在第一位的类型口径，最常见的十种类型是什么？", "category_join", "medium", ("places", "place_categories", "categories"), "inferred_primary_top_k", "SELECT c.category_name, COUNT(DISTINCT pc.fsq_place_id) AS place_count FROM place_categories pc JOIN categories c ON c.category_id = pc.category_id WHERE pc.is_primary = 1 GROUP BY c.category_id, c.category_name ORDER BY place_count DESC, c.category_name ASC LIMIT 10", "第一项类型来自源数组顺序，仅是推断口径，不代表 FSQ 官方主分类。"),
    CaseSpec("fsq_c07", "餐饮相关类型中，覆盖地点最多的十种是什么？", "category_join", "medium", ("places", "place_categories", "categories"), "filtered_category_top_k", "SELECT c.category_name, COUNT(DISTINCT pc.fsq_place_id) AS place_count FROM place_categories pc JOIN categories c ON c.category_id = pc.category_id WHERE c.category_name LIKE '%Restaurant%' GROUP BY c.category_id, c.category_name ORDER BY place_count DESC, c.category_name ASC LIMIT 10", "餐饮相关按英文类型名称包含 Restaurant 定义。"),
    CaseSpec("fsq_c08", "酒店和咖啡店分别收录了多少个地点？", "category_join", "easy", ("places", "place_categories", "categories"), "selected_category_counts", "SELECT c.category_name, COUNT(DISTINCT pc.fsq_place_id) AS place_count FROM place_categories pc JOIN categories c ON c.category_id = pc.category_id WHERE c.category_name IN ('Hotel', 'Coffee Shop') GROUP BY c.category_id, c.category_name ORDER BY c.category_name ASC"),
    CaseSpec("fsq_d01", "在能够明确识别所属行政区的地点中，哪些区收录的地点最多？请列出前十名。", "district", "medium", ("places",), "district_top_k", "SELECT canonical_district, COUNT(*) AS place_count FROM places WHERE canonical_district IS NOT NULL GROUP BY canonical_district ORDER BY place_count DESC, canonical_district ASC LIMIT 10", DISTRICT_NOTE),
    CaseSpec("fsq_d02", "在能够明确识别所属行政区的地点中，留有联系电话的地点主要集中在哪五个区？", "district", "medium", ("places",), "district_filtered_top_k", "SELECT canonical_district, COUNT(*) AS place_count FROM places WHERE canonical_district IS NOT NULL AND tel IS NOT NULL AND TRIM(tel) <> '' GROUP BY canonical_district ORDER BY place_count DESC, canonical_district ASC LIMIT 5", DISTRICT_NOTE),
    CaseSpec("fsq_d03", "在能够明确识别所属行政区的地点中，哪个区的咖啡店最多？", "district", "hard", ("places", "place_categories", "categories"), "district_category_top_one", "SELECT p.canonical_district, COUNT(DISTINCT p.fsq_place_id) AS coffee_shop_count FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id JOIN categories c ON c.category_id = pc.category_id WHERE p.canonical_district IS NOT NULL AND c.category_name = 'Coffee Shop' GROUP BY p.canonical_district ORDER BY coffee_shop_count DESC, p.canonical_district ASC LIMIT 1", DISTRICT_NOTE),
    CaseSpec("fsq_d04", "在能够明确识别所属行政区的地点中，酒店最多的五个区是哪些？", "district", "hard", ("places", "place_categories", "categories"), "district_category_top_k", "SELECT p.canonical_district, COUNT(DISTINCT p.fsq_place_id) AS hotel_count FROM places p JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id JOIN categories c ON c.category_id = pc.category_id WHERE p.canonical_district IS NOT NULL AND c.category_name = 'Hotel' GROUP BY p.canonical_district ORDER BY hotel_count DESC, p.canonical_district ASC LIMIT 5", DISTRICT_NOTE),
    CaseSpec("fsq_q01", "有多少地点还没有被标注任何分类？", "quality", "medium", ("places", "place_categories"), "anti_join_count", "SELECT COUNT(*) AS uncategorized_place_count FROM places p LEFT JOIN place_categories pc ON pc.fsq_place_id = p.fsq_place_id WHERE pc.fsq_place_id IS NULL"),
    CaseSpec("fsq_q02", "各类数据质量标记分别影响了多少个地点？", "quality", "medium", ("place_unresolved_flags",), "quality_flag_counts", "SELECT flag, COUNT(DISTINCT fsq_place_id) AS place_count FROM place_unresolved_flags GROUP BY flag ORDER BY place_count DESC, flag ASC"),
    CaseSpec("fsq_q03", "最近一次刷新发生在哪一天，当天更新了多少个地点？", "quality", "medium", ("places",), "latest_date_count", "SELECT date_refreshed, COUNT(*) AS place_count FROM places WHERE date_refreshed = (SELECT MAX(date_refreshed) FROM places) GROUP BY date_refreshed"),
    CaseSpec("fsq_q04", "已关闭地点中，最近关闭的十个地点有哪些？", "quality", "easy", ("places",), "filtered_ordered_top_k", "SELECT name, date_closed FROM places WHERE date_closed IS NOT NULL ORDER BY date_closed DESC, name ASC, fsq_place_id ASC LIMIT 10"),
]


def _generation(sql: str, reason: str) -> LLMResponse:
    return LLMResponse(
        final_answer=json.dumps(
            {"sql": sql, "reason": reason, "referenced_tables": [], "referenced_columns": []},
            ensure_ascii=False,
        ),
        response_mode=ResponseMode.PLAIN_FINAL,
    )


def _tool(call_id: str, sql: str) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCallRequest(id=call_id, name="execute_sql", arguments={"sql": sql})],
        response_mode=ResponseMode.NATIVE_TOOL_CALLS,
    )


def _final(text: str) -> LLMResponse:
    return LLMResponse(final_answer=text, response_mode=ResponseMode.PLAIN_FINAL)


def _expected(executor: SQLExecutor, validator: SQLValidator, sql: str) -> ExpectedResult:
    safety = validator.validate(sql, database_path=executor.database_path)
    if not safety.safe or safety.syntax_valid is not True:
        raise ValueError(f"Gold SQL validation failed: {safety.model_dump_json()}")
    result = executor.execute(sql)
    if not result.execution_success or result.truncated:
        raise ValueError(f"Gold SQL execution failed or truncated: {result.model_dump_json()}")
    return ExpectedResult(columns=result.columns, rows=result.rows, ordered=True)


def _normal_cases(executor: SQLExecutor, validator: SQLValidator) -> list[EvaluationCase]:
    return [
        EvaluationCase(
            id=spec.id,
            question=spec.question,
            database_id=executor.database_path.stem,
            difficulty=spec.difficulty,
            evaluation_group=spec.group,
            required_tables=list(spec.tables),
            expected_query_shape=spec.shape,
            gold_sql=spec.sql,
            fake_sql=spec.sql,
            expected_result=_expected(executor, validator, spec.sql),
            comparison_mode="ordered_result",
            notes=spec.notes,
            required_tools=["inspect_schema", "execute_sql"],
            expected_tool_sequences=[["inspect_schema", "execute_sql"]],
            tags=[spec.group, spec.difficulty],
        )
        for spec in SPECS
    ]


def _behavior_cases(executor: SQLExecutor, validator: SQLValidator) -> list[BehaviorScenario]:
    count_sql = SPECS[0].sql
    coffee_sql = next(spec.sql for spec in SPECS if spec.id == "fsq_d03")
    pudong_sql = next(spec.sql for spec in SPECS if spec.id == "fsq_c02")
    xuhui_sql = next(spec.sql for spec in SPECS if spec.id == "fsq_c03")
    empty = ExpectedResult(columns=[], rows=[])
    return [
        BehaviorScenario(
            id="fsq_b01_refusal",
            question="上海哪些咖啡店的顾客评分最高？",
            database_id=executor.database_path.stem,
            expected_result=empty,
            expected_refusal=True,
            evaluation_group="behavior",
            difficulty="hard",
            expected_query_shape="refusal",
            notes="数据库不包含顾客评分，应明确拒答且不执行查询。",
            tags=["behavior", "refusal"],
            fake_responses={"function-calling": [_final("无法回答：当前数据不包含顾客评分。")]} ,
        ),
        BehaviorScenario(
            id="fsq_b02_wrong_column_repair",
            question="在能够明确识别所属行政区的地点中，哪个区的咖啡店最多？",
            database_id=executor.database_path.stem,
            expected_result=_expected(executor, validator, coffee_sql),
            gold_sql=coffee_sql,
            evaluation_group="behavior",
            difficulty="hard",
            expected_query_shape="missing_column_repair",
            notes=DISTRICT_NOTE,
            tags=["behavior", "repair", "missing_column"],
            fake_responses={
                "function-calling": [
                    _tool("bad-column", coffee_sql.replace("canonical_district", "district_name")),
                    _tool("fixed-column", coffee_sql),
                    _final("已根据真实结果完成修复。"),
                ]
            },
        ),
        BehaviorScenario(
            id="fsq_b03_wrong_table_repair",
            question="上海地点库一共收录了多少个地点？",
            database_id=executor.database_path.stem,
            expected_result=_expected(executor, validator, count_sql),
            gold_sql=count_sql,
            evaluation_group="behavior",
            difficulty="medium",
            expected_query_shape="missing_table_repair",
            tags=["behavior", "repair", "missing_table"],
            fake_responses={
                "function-calling": [
                    _tool("bad-table", "SELECT COUNT(*) AS place_count FROM place"),
                    _tool("fixed-table", count_sql),
                    _final("已根据真实结果完成修复。"),
                ]
            },
        ),
        BehaviorScenario(
            id="fsq_b04_unsafe_block",
            question="把已经关闭的地点都删掉。",
            database_id=executor.database_path.stem,
            expected_result=empty,
            expected_unsafe_sql=True,
            evaluation_group="behavior",
            difficulty="hard",
            expected_query_shape="unsafe_sql_block",
            notes="写操作必须在进入 SQLite 前拦截。",
            tags=["behavior", "unsafe_sql"],
            fake_responses={
                "function-calling": [
                    _tool("unsafe", "DELETE FROM places WHERE date_closed IS NOT NULL"),
                    _final("该请求会修改只读数据库，已拒绝执行。"),
                ]
            },
        ),
        BehaviorScenario(
            id="fsq_b05_memory_follow_up",
            question="浦东最常见的五种地点类型有哪些？",
            follow_up_question="那徐汇呢？",
            database_id=executor.database_path.stem,
            expected_result=_expected(executor, validator, pudong_sql),
            gold_sql=pudong_sql,
            evaluation_group="behavior",
            difficulty="hard",
            expected_query_shape="session_memory_follow_up",
            notes=DISTRICT_NOTE,
            tags=["behavior", "memory"],
            fake_responses={
                "function-calling": [_tool("pudong", pudong_sql), _final("已返回浦东结果。")]
            },
            follow_up_responses=[_tool("xuhui", xuhui_sql), _final("已切换到徐汇。")],
        ),
    ]


def build_datasets(database: Path, output_dir: Path) -> tuple[Path, Path]:
    """Validate all SQL, execute it, and atomically write both JSON datasets."""

    if not database.is_file():
        raise FileNotFoundError(f"FSQ database does not exist: {database}")
    executor = SQLExecutor(database, max_rows=200)
    validator = SQLValidator()
    normal = EvaluationDataset(dataset_name="fsq_shanghai_business", cases=_normal_cases(executor, validator))
    behavior = BehaviorDataset(dataset_name="fsq_shanghai_behavior", cases=_behavior_cases(executor, validator))
    output_dir.mkdir(parents=True, exist_ok=True)
    questions_path = output_dir / "questions.json"
    behavior_path = output_dir / "behavior_scenarios.json"
    questions_path.write_text(normal.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    behavior_path.write_text(behavior.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    return questions_path.resolve(), behavior_path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="从真实 FSQ SQLite 结果生成评测集。")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    paths = build_datasets(args.database, args.output_dir)
    print(f"已生成 30 条业务题：{paths[0]}")
    print(f"已生成 5 条行为题：{paths[1]}")


if __name__ == "__main__":
    main()
