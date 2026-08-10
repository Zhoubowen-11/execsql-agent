"""Deterministic SQLite execution reward for SQL-decision GRPO."""

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from execsql_agent.evaluation.comparator import compare_execution_result
from execsql_agent.models import ExpectedResult
from execsql_agent.tools.sql_executor import SQLExecutor
from execsql_agent.tools.sql_validator import SQLValidator

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class RewardBreakdown:
    """One auditable reward decision."""

    case_id: str
    reward: float
    category: str
    sql: str | None
    executed: bool
    execution_success: bool
    truncated: bool
    result_correct: bool | None
    detail: str | None = None


@dataclass(frozen=True)
class RewardGroupMetrics:
    """Rollout-category diagnostics for one prompt group."""

    case_id: str
    reward_mean: float
    reward_std: float
    zero_std: bool
    correct_rollout_rate: float
    semantic_mismatch_rate: float
    malformed_rate: float
    execution_failure_rate: float
    unsafe_rate: float


def normalized_group_advantages(rewards: Sequence[float]) -> list[float]:
    """Mirror GRPO reward normalization; constant rewards yield zero advantages."""

    if not rewards:
        raise ValueError("a reward group cannot be empty")
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    return [(reward - mean) / (std + 1.0e-4) for reward in rewards]


def reward_group_metrics(group: Sequence[RewardBreakdown]) -> RewardGroupMetrics:
    """Summarize exactly one prompt's rollout outcomes."""

    if not group:
        raise ValueError("a reward breakdown group cannot be empty")
    case_ids = {item.case_id for item in group}
    if len(case_ids) != 1:
        raise ValueError("a reward group contains multiple case IDs")
    size = len(group)
    rewards = [item.reward for item in group]

    def rate(category: str) -> float:
        return sum(item.category == category for item in group) / size

    reward_std = statistics.pstdev(rewards)
    return RewardGroupMetrics(
        case_id=group[0].case_id,
        reward_mean=statistics.fmean(rewards),
        reward_std=reward_std,
        zero_std=math.isclose(reward_std, 0.0, abs_tol=0.0),
        correct_rollout_rate=rate("correct"),
        semantic_mismatch_rate=rate("semantic_mismatch"),
        malformed_rate=rate("malformed_tool_call"),
        execution_failure_rate=rate("execution_failure"),
        unsafe_rate=rate("unsafe_sql"),
    )


class SQLExecutionReward:
    """Score generated ``validate_sql`` calls using the real safe SQLite stack."""

    __name__ = "sql_execution_reward"

    def __init__(
        self, *, max_rows: int = 100, num_generations: int | None = None
    ) -> None:
        if num_generations is not None and num_generations < 2:
            raise ValueError("num_generations must be at least two")
        self.max_rows = max_rows
        self.num_generations = num_generations
        self.validator = SQLValidator()
        self.history: list[RewardBreakdown] = []
        self.group_history: list[RewardGroupMetrics] = []

    @staticmethod
    def _completion_text(completion: object) -> str:
        if isinstance(completion, str):
            return completion
        if isinstance(completion, list) and len(completion) == 1:
            message = completion[0]
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return str(message["content"])
        raise ValueError("completion is not assistant text")

    @classmethod
    def extract_sql(cls, completion: object) -> str:
        """Extract exactly one SQL string from a Qwen ``validate_sql`` tool call."""

        text = cls._completion_text(completion).strip()
        matches = _TOOL_CALL_PATTERN.findall(text)
        if len(matches) != 1:
            raise ValueError("expected exactly one complete <tool_call> envelope")
        payload = json.loads(matches[0])
        if not isinstance(payload, dict) or payload.get("name") != "validate_sql":
            raise ValueError("completion must call validate_sql")
        arguments = payload.get("arguments")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict) or set(arguments) != {"sql"}:
            raise ValueError("validate_sql arguments must contain only sql")
        sql = arguments.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("sql must be a non-empty string")
        return sql

    @staticmethod
    def _coerce_metadata(values: Sequence[Any] | Any, size: int, name: str) -> list[Any]:
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
        database_path: Sequence[str] | str,
        expected_result_json: Sequence[str] | str,
        **_: object,
    ) -> list[float]:
        """Return one deterministic reward per completion and retain breakdowns."""

        del prompts
        size = len(completions)
        case_ids = self._coerce_metadata(case_id, size, "case_id")
        databases = self._coerce_metadata(database_path, size, "database_path")
        expected_json = self._coerce_metadata(
            expected_result_json, size, "expected_result_json"
        )
        history_start = len(self.history)
        rewards: list[float] = []
        for completion, raw_case_id, raw_database, raw_expected in zip(
            completions, case_ids, databases, expected_json, strict=True
        ):
            current_case_id = str(raw_case_id)
            try:
                sql = self.extract_sql(completion)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                breakdown = RewardBreakdown(
                    case_id=current_case_id,
                    reward=-0.3,
                    category="malformed_tool_call",
                    sql=None,
                    executed=False,
                    execution_success=False,
                    truncated=False,
                    result_correct=None,
                    detail=str(error),
                )
                self.history.append(breakdown)
                rewards.append(breakdown.reward)
                continue

            safety = self.validator.validate(sql)
            if not safety.safe:
                breakdown = RewardBreakdown(
                    case_id=current_case_id,
                    reward=-1.0,
                    category="unsafe_sql",
                    sql=sql,
                    executed=False,
                    execution_success=False,
                    truncated=False,
                    result_correct=None,
                    detail=safety.reason,
                )
                self.history.append(breakdown)
                rewards.append(breakdown.reward)
                continue

            expected = ExpectedResult.model_validate_json(str(raw_expected))
            actual = SQLExecutor(str(raw_database), max_rows=self.max_rows).execute(sql)
            if not actual.execution_success:
                detail = actual.error.message if actual.error is not None else actual.blocked_reason
                breakdown = RewardBreakdown(
                    case_id=current_case_id,
                    reward=-0.2,
                    category="execution_failure",
                    sql=sql,
                    executed=actual.executed,
                    execution_success=False,
                    truncated=False,
                    result_correct=None,
                    detail=detail,
                )
                self.history.append(breakdown)
                rewards.append(breakdown.reward)
                continue

            correct = compare_execution_result(actual, expected)
            reward = 1.0 if correct is True else 0.0
            category = "correct" if correct is True else "semantic_mismatch"
            breakdown = RewardBreakdown(
                case_id=current_case_id,
                reward=reward,
                category=category,
                sql=sql,
                executed=actual.executed,
                execution_success=True,
                truncated=actual.truncated,
                result_correct=correct,
            )
            self.history.append(breakdown)
            rewards.append(reward)
        if self.num_generations is not None:
            if size % self.num_generations != 0:
                raise ValueError("completion batch does not contain complete reward groups")
            current = self.history[history_start:]
            for start in range(0, size, self.num_generations):
                self.group_history.append(
                    reward_group_metrics(current[start : start + self.num_generations])
                )
        return rewards


def qwen_tool_call(sql: str) -> str:
    """Format SQL as the exact textual Qwen tool-call envelope used in audits."""

    payload = {"name": "validate_sql", "arguments": {"sql": sql}}
    return f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"
