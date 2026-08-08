"""Validation for the independent real-result FSQ holdout dataset."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from execsql_agent.evaluation.comparator import compare_execution_result
from execsql_agent.models import ExpectedResult
from execsql_agent.tools.sql_executor import SQLExecutor
from execsql_agent.tools.sql_validator import SQLValidator
from scripts.build_fsq_holdout import build_holdout

DATABASE = Path("data/fsq/shanghai_places.db")
HOLDOUT = Path("data/fsq/eval/questions_holdout_v1.json")
REFERENCES = (
    Path("data/fsq/eval/questions_v1.json"),
    Path("data/fsq/eval/questions_v2.json"),
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized_question(question: str) -> str:
    return re.sub(r"\W+", "", question.casefold(), flags=re.UNICODE)


def test_holdout_contains_twenty_unique_cases_with_required_distribution() -> None:
    payload = _load(HOLDOUT)
    cases = payload["cases"]
    assert isinstance(cases, list)
    assert len(cases) == 20
    assert [case["case_id"] for case in cases] == [
        f"fsq_h{index:02d}" for index in range(1, 21)
    ]
    assert len({case["case_id"] for case in cases}) == 20

    counts: dict[str, int] = {}
    for case in cases:
        group = case["tags"][1]
        counts[group] = counts.get(group, 0) + 1
        assert case["strict_columns"] is False
        assert isinstance(case["ordered"], bool)
        assert set(case) == {
            "case_id",
            "question",
            "tags",
            "gold_sql",
            "expected_result",
            "ordered",
            "strict_columns",
        }
    assert counts == {
        "single_filter_quality": 4,
        "aggregate_ratio_topk": 6,
        "category_join": 6,
        "district_analysis": 4,
    }


def test_holdout_questions_are_natural_specific_and_deterministic() -> None:
    cases = _load(HOLDOUT)["cases"]
    forbidden = {"join", "group by", "null", "canonical_district", "places"}
    tied_top_k = {
        "fsq_h07",
        "fsq_h08",
        "fsq_h09",
        "fsq_h11",
        "fsq_h12",
        "fsq_h15",
        "fsq_h16",
        "fsq_h17",
        "fsq_h18",
        "fsq_h19",
    }
    for case in cases:
        question = case["question"]
        lowered = question.casefold()
        assert not any(term in lowered for term in forbidden)
        assert not any(term in question for term in ("评分", "营业收入", "实时客流"))
        if case["case_id"] in tied_top_k:
            assert "相同" in question
        if "district_analysis" in case["tags"]:
            assert "明确映射到行政区" in question


def test_holdout_gold_sql_is_safe_and_expected_results_are_reproducible() -> None:
    cases = _load(HOLDOUT)["cases"]
    validator = SQLValidator()
    executor = SQLExecutor(DATABASE, max_rows=100)
    for case in cases:
        safety = validator.validate(case["gold_sql"], database_path=DATABASE)
        assert safety.safe is True
        assert safety.syntax_valid is True
        actual = executor.execute(case["gold_sql"])
        expected = ExpectedResult(
            columns=case["expected_result"]["columns"],
            rows=case["expected_result"]["rows"],
            ordered=case["ordered"],
            strict_columns=case["strict_columns"],
        )
        assert actual.execution_success is True
        assert actual.truncated is False
        assert compare_execution_result(actual, expected) is True


def test_holdout_has_no_question_or_sql_duplicates() -> None:
    holdout = _load(HOLDOUT)["cases"]
    references = [
        case
        for path in REFERENCES
        for case in _load(path)["cases"]
    ]
    holdout_questions = [_normalized_question(case["question"]) for case in holdout]
    reference_questions = {
        _normalized_question(case["question"]) for case in references
    }
    assert len(set(holdout_questions)) == len(holdout_questions)
    assert not set(holdout_questions) & reference_questions

    validator = SQLValidator()
    holdout_sql = [
        validator.validate(case["gold_sql"]).normalized_sql for case in holdout
    ]
    reference_sql = {
        validator.validate(case["gold_sql"]).normalized_sql for case in references
    }
    assert len(set(holdout_sql)) == len(holdout_sql)
    assert not set(holdout_sql) & reference_sql

    original_questions = [case["question"] for case in references]
    for case in holdout:
        closest = max(
            SequenceMatcher(
                None,
                case["question"],
                original,
                autojunk=False,
            ).ratio()
            for original in original_questions
        )
        assert closest < 0.9


def test_holdout_generation_is_reproducible(tmp_path: Path) -> None:
    generated = build_holdout(DATABASE, tmp_path / "questions_holdout_v1.json")

    assert generated.read_bytes() == HOLDOUT.read_bytes()
