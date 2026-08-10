"""Run a deterministic offline audit of the counterfactual verifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from grpo_rewards import qwen_tool_call  # noqa: E402
from grpo_test_suite_rewards import CounterfactualExecutionReward  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def audit(tasks_path: Path) -> dict[str, Any]:
    tasks = _read_jsonl(tasks_path)
    selected: list[tuple[dict[str, Any], str, str]] = []
    for family in ("aggregation_scope", "ranking", "ties"):
        family_tasks = [item for item in tasks if item["template_family"] == family]
        selected.extend(
            [
                (family_tasks[0], family_tasks[0]["oracle_sql"], "oracle"),
                (family_tasks[1], family_tasks[1]["wrong_sql"], "typical_semantic_wrong"),
                (family_tasks[2], family_tasks[2]["wrong_sql"], "typical_semantic_wrong"),
                (
                    family_tasks[3],
                    "SELECT COUNT(*) AS unrelated_count FROM places",
                    "obvious_wrong",
                ),
            ]
        )

    verifier = CounterfactualExecutionReward()
    examples: list[dict[str, Any]] = []
    for task, sql, kind in selected:
        value = verifier(
            prompts=[task["prompt"]],
            completions=[qwen_tool_call(sql)],
            case_id=[task["case_id"]],
            worlds_json=[task["worlds_json"]],
        )[0]
        breakdown = verifier.history[-1]
        examples.append(
            {
                "case_id": task["case_id"],
                "family": task["template_family"],
                "question": task["question"],
                "candidate_kind": kind,
                "candidate_sql": sql,
                "per_world_pass": [world.passed for world in breakdown.worlds],
                "reward": value,
                "binary_world0_reward": breakdown.binary_reward,
                "execution_failure_count": breakdown.execution_failure_count,
                "explanation": (
                    "Oracle SQL matches every world."
                    if kind == "oracle"
                    else task["wrong_explanation"]
                    if kind == "typical_semantic_wrong"
                    else "The result shape and semantics do not answer the requested task."
                ),
            }
        )
    return {
        "example_count": len(examples),
        "reward_levels": sorted({item["reward"] for item in examples}),
        "binary_lucky_but_suite_partial_count": sum(
            item["binary_world0_reward"] == 1.0 and 0.0 < item["reward"] < 1.0
            for item in examples
        ),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/fsq/train/counterfactual/tasks_v1.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/fsq/train/counterfactual/offline_verifier_audit.json"),
    )
    args = parser.parse_args()
    report = audit(args.tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
