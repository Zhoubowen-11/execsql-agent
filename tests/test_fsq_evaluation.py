"""Validation for the real-result FSQ business and behavior datasets."""

import hashlib
from pathlib import Path

from execsql_agent.agents.function_calling import FunctionCallingAgent
from execsql_agent.config import load_domain_config
from execsql_agent.evaluation.comparator import compare_execution_result
from execsql_agent.evaluation.evaluator import Evaluator
from execsql_agent.evaluation.synthetic import (
    build_scripted_client,
    load_behavior_dataset,
    load_evaluation_dataset,
)
from execsql_agent.llm.fake import FakeLLMClient
from execsql_agent.tools.sql_executor import SQLExecutor
from execsql_agent.tools.sql_validator import SQLValidator
from scripts.version_fsq_evaluation import QUESTION_REVISIONS, build_v2_dataset

DATABASE = Path("data/fsq/shanghai_places.db")
QUESTIONS = Path("data/fsq/eval/questions.json")
QUESTIONS_V1 = Path("data/fsq/eval/questions_v1.json")
QUESTIONS_V2 = Path("data/fsq/eval/questions_v2.json")
BEHAVIORS = Path("data/fsq/eval/behavior_scenarios.json")
QUESTIONS_V1_SHA256 = "9e9f3d51c3e40441e76de955bffdf26b8b88773603aeb5621fbe606595db6f5c"


def test_fsq_question_distribution_and_natural_language() -> None:
    dataset = load_evaluation_dataset(QUESTIONS)
    counts: dict[str, int] = {}
    forbidden = {
        "places",
        "categories",
        "place_categories",
        "canonical_district",
        "join",
        "group by",
        "null",
    }
    for case in dataset.cases:
        assert case.evaluation_group is not None
        counts[case.evaluation_group] = counts.get(case.evaluation_group, 0) + 1
        lowered = case.question.casefold()
        assert not any(term in lowered for term in forbidden)
        assert case.required_tables
        assert case.expected_query_shape
    assert counts == {
        "single_table": 6,
        "aggregate_topk": 8,
        "category_join": 8,
        "district": 4,
        "quality": 4,
    }


def test_all_gold_sql_matches_persisted_real_results() -> None:
    dataset = load_evaluation_dataset(QUESTIONS)
    validator = SQLValidator()
    executor = SQLExecutor(DATABASE, max_rows=200)
    for case in dataset.cases:
        assert case.gold_sql is not None
        safety = validator.validate(case.gold_sql, database_path=DATABASE)
        assert safety.safe is True
        assert safety.syntax_valid is True
        actual = executor.execute(case.gold_sql)
        assert actual.execution_success is True
        assert actual.truncated is False
        assert compare_execution_result(actual, case.expected_result) is True
        if "canonical_district" in case.gold_sql:
            assert case.notes is not None
            assert "50.5546%" in case.notes


def test_fsq_alias_policy_and_latest_refresh_case() -> None:
    dataset = load_evaluation_dataset(QUESTIONS)
    cases = {case.id: case for case in dataset.cases}

    assert cases["fsq_s04"].expected_result.strict_columns is False
    assert cases["fsq_c05"].expected_result.strict_columns is False

    latest = cases["fsq_s06"]
    assert latest.question == (
        "数据库中最近一次地点更新时间是哪一天？这一天共有多少个地点被更新？"
    )
    assert latest.gold_sql is not None
    actual = SQLExecutor(DATABASE).execute(latest.gold_sql)
    assert actual.execution_success is True
    assert actual.rows == [["2026-07-08", 280]]
    assert compare_execution_result(actual, latest.expected_result) is True


def test_questions_v1_is_an_exact_unchanged_snapshot() -> None:
    current = QUESTIONS.read_bytes()
    versioned = QUESTIONS_V1.read_bytes()

    assert versioned == current
    assert hashlib.sha256(current).hexdigest() == QUESTIONS_V1_SHA256


def test_questions_v2_clarifies_only_the_selected_definitions() -> None:
    v1 = load_evaluation_dataset(QUESTIONS_V1)
    v2 = load_evaluation_dataset(QUESTIONS_V2)
    v1_cases = {case.id: case for case in v1.cases}
    v2_cases = {case.id: case for case in v2.cases}

    assert v2.dataset_name == "fsq_shanghai_business_v2"
    assert v1_cases.keys() == v2_cases.keys()
    for case_id, question in QUESTION_REVISIONS.items():
        assert v2_cases[case_id].question == question
        assert v2_cases[case_id].question != v1_cases[case_id].question
    assert "涉及的地点数量" in v2_cases["fsq_c05"].question
    assert "Restaurant" in v2_cases["fsq_c07"].question
    assert "省级区域标注不计入" in v2_cases["fsq_a01"].question
    assert v2_cases["fsq_a07"].question == v1_cases["fsq_a07"].question
    assert v2_cases["fsq_a07"].expected_result.ordered is False

    changed = set(QUESTION_REVISIONS) | {"fsq_a07"}
    for case_id in v1_cases.keys() - changed:
        assert v2_cases[case_id].model_dump() == v1_cases[case_id].model_dump()

    forbidden = {"join", "group by", "null", "places", "categories"}
    for case_id in changed:
        lowered = v2_cases[case_id].question.casefold()
        assert not any(term in lowered for term in forbidden)


def test_questions_v2_gold_sql_matches_fresh_real_results() -> None:
    dataset = load_evaluation_dataset(QUESTIONS_V2)
    validator = SQLValidator()
    executor = SQLExecutor(DATABASE, max_rows=200)

    for case in dataset.cases:
        assert case.gold_sql is not None
        safety = validator.validate(case.gold_sql, database_path=DATABASE)
        assert safety.safe is True
        assert safety.syntax_valid is True
        actual = executor.execute(case.gold_sql)
        assert actual.execution_success is True
        assert actual.truncated is False
        assert compare_execution_result(actual, case.expected_result) is True


def test_questions_v2_generation_is_reproducible(tmp_path: Path) -> None:
    generated = build_v2_dataset(
        DATABASE,
        QUESTIONS_V1,
        tmp_path / "questions_v2.json",
    )

    assert generated.read_bytes() == QUESTIONS_V2.read_bytes()


def test_high_value_behavior_metrics_use_real_sqlite() -> None:
    dataset = load_behavior_dataset(BEHAVIORS)
    assert len(dataset.cases) == 5
    report = Evaluator(DATABASE, build_scripted_client).evaluate(
        dataset.cases[:4],
        dataset_name=dataset.dataset_name,
        agent_mode="function-calling",
    )
    metrics = report.mode_metrics["function-calling"]
    assert metrics.execution_accuracy.value == 1
    assert metrics.result_accuracy.value == 1
    assert metrics.repair_success_rate.value == 1
    assert metrics.refusal_accuracy.value == 1
    assert metrics.unsafe_sql_block_rate.value == 1


def test_memory_behavior_runs_as_two_isolated_agent_tasks() -> None:
    dataset = load_behavior_dataset(BEHAVIORS)
    case = next(item for item in dataset.cases if item.id == "fsq_b05_memory_follow_up")
    responses = case.fake_responses["function-calling"] + case.follow_up_responses
    fake = FakeLLMClient(responses)
    domain = load_domain_config("config/fsq_shanghai.json")
    agent = FunctionCallingAgent(DATABASE, fake, domain_context=domain.to_prompt())

    first = agent.run(case.question, session_id="fsq-memory")
    assert case.follow_up_question is not None
    second = agent.run(case.follow_up_question, session_id="fsq-memory")

    assert first.execution_success is True
    assert second.execution_success is True
    assert second.final_sql is not None and "'Xuhui'" in second.final_sql
    follow_up_prompt = fake.requests[2].messages[0].content or ""
    assert case.question in follow_up_prompt
    assert "'Pudong'" in follow_up_prompt
