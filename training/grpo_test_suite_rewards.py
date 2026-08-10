"""Deterministic multi-world execution reward for counterfactual GRPO."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from grpo_rewards import SQLExecutionReward

from execsql_agent.evaluation.comparator import compare_execution_result
from execsql_agent.models import ExpectedResult
from execsql_agent.tools.sql_executor import SQLExecutor
from execsql_agent.tools.sql_validator import SQLValidator


@dataclass(frozen=True)
class WorldResult:
    world_id: str
    passed: bool
    executed: bool
    execution_success: bool
    truncated: bool
    error: str | None


@dataclass(frozen=True)
class TestSuiteBreakdown:
    case_id: str
    reward: float
    binary_reward: float
    category: str
    sql: str | None
    passed_worlds: int
    total_worlds: int
    execution_failure_count: int
    worlds: list[WorldResult]
    detail: str | None = None


class CounterfactualExecutionReward:
    """Score one SQL by its execution equivalence over four hidden worlds."""

    __name__ = "counterfactual_execution_reward"

    def __init__(self, *, num_generations: int | None = None) -> None:
        self.num_generations = num_generations
        self.validator = SQLValidator()
        self.history: list[TestSuiteBreakdown] = []

    @staticmethod
    def _coerce(values: Sequence[Any] | Any, size: int, name: str) -> list[Any]:
        if isinstance(values, (str, bytes, Path)):
            return [values] * size
        if isinstance(values, Sequence):
            result = list(values)
            if len(result) != size:
                raise ValueError(f"{name} length does not match completions")
            return result
        return [values] * size

    def __call__(
        self,
        prompts: Sequence[object],
        completions: Sequence[object],
        *,
        case_id: Sequence[str] | str,
        worlds_json: Sequence[str] | str,
        **_: object,
    ) -> list[float]:
        del prompts
        size = len(completions)
        case_ids = self._coerce(case_id, size, "case_id")
        world_payloads = self._coerce(worlds_json, size, "worlds_json")
        rewards: list[float] = []
        for completion, raw_case_id, raw_worlds in zip(
            completions, case_ids, world_payloads, strict=True
        ):
            current_case = str(raw_case_id)
            try:
                sql = SQLExecutionReward.extract_sql(completion)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                breakdown = TestSuiteBreakdown(
                    case_id=current_case,
                    reward=-0.3,
                    binary_reward=-0.3,
                    category="malformed_tool_call",
                    sql=None,
                    passed_worlds=0,
                    total_worlds=4,
                    execution_failure_count=0,
                    worlds=[],
                    detail=str(error),
                )
                self.history.append(breakdown)
                rewards.append(breakdown.reward)
                continue
            safety = self.validator.validate(sql)
            if not safety.safe:
                breakdown = TestSuiteBreakdown(
                    case_id=current_case,
                    reward=-1.0,
                    binary_reward=-1.0,
                    category="unsafe_sql",
                    sql=sql,
                    passed_worlds=0,
                    total_worlds=4,
                    execution_failure_count=0,
                    worlds=[],
                    detail=safety.reason,
                )
                self.history.append(breakdown)
                rewards.append(breakdown.reward)
                continue
            worlds = json.loads(str(raw_worlds))
            if not isinstance(worlds, list) or len(worlds) != 4:
                raise ValueError(f"{current_case}: expected exactly four worlds")
            world_results: list[WorldResult] = []
            for world in worlds:
                actual = SQLExecutor(str(world["database_path"]), max_rows=100).execute(sql)
                passed = False
                error = None
                if actual.execution_success:
                    expected = ExpectedResult.model_validate(world["expected_result"])
                    passed = compare_execution_result(actual, expected) is True
                else:
                    error = (
                        actual.error.message
                        if actual.error is not None
                        else actual.blocked_reason
                    )
                world_results.append(
                    WorldResult(
                        world_id=str(world["world_id"]),
                        passed=passed,
                        executed=actual.executed,
                        execution_success=actual.execution_success,
                        truncated=actual.truncated,
                        error=error,
                    )
                )
            passed_worlds = sum(result.passed for result in world_results)
            failures = sum(not result.execution_success for result in world_results)
            reward = passed_worlds / len(world_results)
            if reward == 1.0:
                category = "correct"
            elif reward > 0.0:
                category = "partial_semantic"
            elif failures:
                category = "execution_failure"
            else:
                category = "semantic_mismatch"
            breakdown = TestSuiteBreakdown(
                case_id=current_case,
                reward=reward,
                binary_reward=1.0 if world_results[0].passed else 0.0,
                category=category,
                sql=sql,
                passed_worlds=passed_worlds,
                total_worlds=len(world_results),
                execution_failure_count=failures,
                worlds=world_results,
            )
            self.history.append(breakdown)
            rewards.append(reward)
        if self.num_generations is not None and size % self.num_generations != 0:
            raise ValueError("completion batch does not contain complete reward groups")
        return rewards


def serialize_breakdown(value: TestSuiteBreakdown) -> dict[str, Any]:
    return asdict(value)
