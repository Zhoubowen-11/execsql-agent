from __future__ import annotations

import json
import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING))

from counterfactual_worlds import build_worlds, generate_cases  # noqa: E402
from grpo_rewards import qwen_tool_call  # noqa: E402
from grpo_test_suite_rewards import CounterfactualExecutionReward  # noqa: E402


def _score(case_index: int, tmp_path: Path, sql: str) -> tuple[float, object]:
    case = generate_cases()[case_index]
    worlds = build_worlds(case, tmp_path / case.case_id)
    reward = CounterfactualExecutionReward()
    values = reward(
        prompts=[case.question],
        completions=[qwen_tool_call(sql)],
        case_id=[case.case_id],
        worlds_json=[json.dumps(worlds)],
    )
    return values[0], reward.history[0]


def test_oracle_passes_all_worlds(tmp_path: Path) -> None:
    case = generate_cases()[0]
    value, breakdown = _score(0, tmp_path, case.oracle_sql)
    assert value == 1.0
    assert breakdown.passed_worlds == 4


def test_aggregation_scope_wrong_sql_is_partial(tmp_path: Path) -> None:
    case = generate_cases()[0]
    value, breakdown = _score(0, tmp_path, case.wrong_sql)
    assert value == 0.25
    assert [world.passed for world in breakdown.worlds] == [True, False, False, False]


def test_ranking_wrong_sql_is_partial(tmp_path: Path) -> None:
    case = generate_cases()[24]
    value, breakdown = _score(24, tmp_path, case.wrong_sql)
    assert value == 0.75
    assert breakdown.binary_reward == 1.0


def test_ties_limit_one_is_partial(tmp_path: Path) -> None:
    case = generate_cases()[48]
    value, breakdown = _score(48, tmp_path, case.wrong_sql)
    assert value == 0.25
    assert breakdown.execution_failure_count == 0


def test_ties_returning_every_group_is_partially_lucky(tmp_path: Path) -> None:
    case = generate_cases()[48]
    sql = case.wrong_sql.replace("LIMIT 1", "LIMIT 10")
    value, breakdown = _score(48, tmp_path, sql)
    assert value == 0.5
    assert [world.passed for world in breakdown.worlds] == [False, True, True, False]


def test_execution_failures_are_world_failures(tmp_path: Path) -> None:
    value, breakdown = _score(0, tmp_path, "SELECT missing_column FROM places")
    assert value == 0.0
    assert breakdown.execution_failure_count == 4
    assert breakdown.category == "execution_failure"


def test_unsafe_sql_keeps_negative_reward(tmp_path: Path) -> None:
    value, breakdown = _score(0, tmp_path, "DELETE FROM places")
    assert value == -1.0
    assert breakdown.category == "unsafe_sql"
    assert breakdown.worlds == []


def test_malformed_call_keeps_negative_reward(tmp_path: Path) -> None:
    case = generate_cases()[0]
    worlds = build_worlds(case, tmp_path / case.case_id)
    reward = CounterfactualExecutionReward()
    values = reward(
        prompts=[case.question],
        completions=["not a tool call"],
        case_id=[case.case_id],
        worlds_json=[json.dumps(worlds)],
    )
    assert values == [-0.3]
    assert reward.history[0].category == "malformed_tool_call"
