"""Tests for deterministic GRPO SQL execution rewards."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from training.grpo_rewards import (
    SQLExecutionReward,
    normalized_group_advantages,
    qwen_tool_call,
)


def _database(tmp_path: Path) -> Path:
    path = tmp_path / "reward.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
        connection.executemany(
            "INSERT INTO items (name) VALUES (?)",
            [("alpha",), ("beta",), ("beta",)],
        )
    return path


def _expected(columns: list[str], rows: list[list[object]]) -> str:
    return json.dumps(
        {
            "columns": columns,
            "rows": rows,
            "ordered": True,
            "numeric_tolerance": 1.0e-6,
            "strict_columns": False,
        }
    )


def _score(
    reward: SQLExecutionReward,
    database: Path,
    completion: str,
    expected: str,
) -> float:
    return reward(
        prompts=["prompt"],
        completions=[completion],
        case_id=["case"],
        database_path=[str(database)],
        expected_result_json=[expected],
    )[0]


def test_correct_sql_reward(tmp_path: Path) -> None:
    database = _database(tmp_path)
    reward = SQLExecutionReward()
    assert _score(
        reward,
        database,
        qwen_tool_call("SELECT COUNT(*) AS total FROM items"),
        _expected(["total"], [[3]]),
    ) == 1.0
    assert reward.history[-1].category == "correct"


def test_semantic_mismatch_reward(tmp_path: Path) -> None:
    database = _database(tmp_path)
    reward = SQLExecutionReward()
    assert _score(
        reward,
        database,
        qwen_tool_call("SELECT COUNT(*) AS total FROM items WHERE name = 'beta'"),
        _expected(["total"], [[3]]),
    ) == 0.0
    assert reward.history[-1].category == "semantic_mismatch"


def test_execution_failure_reward(tmp_path: Path) -> None:
    database = _database(tmp_path)
    reward = SQLExecutionReward()
    assert _score(
        reward,
        database,
        qwen_tool_call("SELECT missing FROM items"),
        _expected(["total"], [[3]]),
    ) == -0.2
    assert reward.history[-1].category == "execution_failure"


def test_malformed_tool_call_reward(tmp_path: Path) -> None:
    database = _database(tmp_path)
    reward = SQLExecutionReward()
    assert _score(
        reward,
        database,
        "<tool_call>{not json}</tool_call>",
        _expected(["total"], [[3]]),
    ) == -0.3
    assert reward.history[-1].executed is False


def test_unsafe_sql_reward(tmp_path: Path) -> None:
    database = _database(tmp_path)
    reward = SQLExecutionReward()
    assert _score(
        reward,
        database,
        qwen_tool_call("DELETE FROM items"),
        _expected(["total"], [[3]]),
    ) == -1.0
    assert reward.history[-1].category == "unsafe_sql"
    assert reward.history[-1].executed is False


def test_truncated_success_is_incorrect(tmp_path: Path) -> None:
    database = _database(tmp_path)
    reward = SQLExecutionReward(max_rows=2)
    assert _score(
        reward,
        database,
        qwen_tool_call("SELECT id FROM items ORDER BY id"),
        _expected(["id"], [[1], [2], [3]]),
    ) == 0.0
    assert reward.history[-1].truncated is True
    assert reward.history[-1].category == "semantic_mismatch"


def test_zero_std_group_has_no_correctness_update_with_beta_zero() -> None:
    advantages = normalized_group_advantages([1.0, 1.0, 1.0, 1.0])
    assert advantages == [0.0, 0.0, 0.0, 0.0]
    beta = 0.0
    correctness_loss = -sum(advantages) / len(advantages)
    assert correctness_loss + beta == 0.0
