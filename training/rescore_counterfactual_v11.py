"""Rebuild only ranking/ties worlds and rescore saved V1 SFT completions."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from counterfactual_worlds import build_worlds, generate_cases  # noqa: E402
from grpo_test_suite_rewards import (  # noqa: E402
    CounterfactualExecutionReward,
    serialize_breakdown,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _summary(groups: list[dict[str, Any]]) -> dict[str, Any]:
    standard_deviations = [statistics.pstdev(group["suite_rewards"]) for group in groups]
    levels = Counter(
        reward for group in groups for reward in group["suite_rewards"]
    )
    zero_std = sum(value == 0.0 for value in standard_deviations)
    return {
        "group_count": len(groups),
        "zero_std_groups": zero_std,
        "zero_std_group_rate": zero_std / len(groups),
        "mixed_groups": len(groups) - zero_std,
        "mixed_group_rate": 1.0 - zero_std / len(groups),
        "mean_group_reward_std": statistics.fmean(standard_deviations),
        "unique_reward_levels": sorted(levels),
        "reward_level_counts": {str(key): value for key, value in sorted(levels.items())},
        "partial_reward_rollouts": sum(
            0.0 < reward < 1.0 for group in groups for reward in group["suite_rewards"]
        ),
    }


def rescore(
    *,
    tasks_path: Path,
    v1_audit_path: Path,
    world_dir: Path,
    output_tasks_path: Path,
) -> dict[str, Any]:
    tasks = _read_jsonl(tasks_path)
    cases = {case.case_id: case for case in generate_cases()}
    original = json.loads(v1_audit_path.read_text(encoding="utf-8"))
    original_groups = {group["case_id"]: group for group in original["groups"]}
    if len(tasks) != 72 or len(original_groups) != 72:
        raise ValueError("V1.1 requires the fixed 72-task V1 corpus")

    v11_tasks: list[dict[str, Any]] = []
    unchanged_aggregation = 0
    for item in tasks:
        updated = dict(item)
        if item["template_family"] == "aggregation_scope":
            unchanged_aggregation += 1
        else:
            worlds = build_worlds(cases[item["case_id"]], world_dir)
            updated["worlds_json"] = json.dumps(
                worlds, ensure_ascii=False, separators=(",", ":")
            )
        if updated["prompt"] != item["prompt"]:
            raise ValueError(f"prompt changed for {item['case_id']}")
        v11_tasks.append(updated)
    _write_jsonl(output_tasks_path, v11_tasks)

    verifier = CounterfactualExecutionReward(num_generations=4)
    rescored_groups: list[dict[str, Any]] = []
    for item in v11_tasks:
        old_group = original_groups[item["case_id"]]
        completions = list(old_group["completions"])
        history_start = len(verifier.history)
        rewards = verifier(
            prompts=[item["prompt"]] * 4,
            completions=completions,
            case_id=[item["case_id"]] * 4,
            worlds_json=[item["worlds_json"]] * 4,
        )
        breakdowns = verifier.history[history_start:]
        rescored_groups.append(
            {
                "case_id": item["case_id"],
                "template_family": item["template_family"],
                "split": item["split"],
                "suite_rewards": rewards,
                "completions": completions,
                "completion_hashes": [
                    hashlib.sha256(value.encode()).hexdigest() for value in completions
                ],
                "breakdowns": [serialize_breakdown(value) for value in breakdowns],
            }
        )

    family = {
        name: _summary(
            [group for group in rescored_groups if group["template_family"] == name]
        )
        for name in ("aggregation_scope", "ranking", "ties")
    }
    overall = _summary(rescored_groups)
    categories = Counter(
        breakdown["category"]
        for group in rescored_groups
        for breakdown in group["breakdowns"]
    )
    prior_categories = Counter(original["category_counts"])
    gate = {
        "overall_zero_std_below_55_percent": overall["zero_std_group_rate"] < 0.55,
        "aggregation_mixed_at_least_50_percent": family["aggregation_scope"][
            "mixed_group_rate"
        ]
        >= 0.50,
        "ranking_mixed_at_least_25_percent": family["ranking"]["mixed_group_rate"]
        >= 0.25,
        "ties_mixed_at_least_50_percent": family["ties"]["mixed_group_rate"] >= 0.50,
        "malformed_unchanged": categories["malformed_tool_call"]
        == prior_categories["malformed_tool_call"],
        "unsafe_unchanged": categories["unsafe_sql"] == prior_categories["unsafe_sql"],
    }
    gate["proceed_to_grpo"] = all(gate.values())
    return {
        "version": "counterfactual_v1.1",
        "source_completion_audit": str(v1_audit_path),
        "model_called": False,
        "task_count": len(v11_tasks),
        "rollout_count": len(v11_tasks) * 4,
        "aggregation_world_sets_unchanged": unchanged_aggregation,
        "ranking_ties_world_sets_rebuilt": len(v11_tasks) - unchanged_aggregation,
        "overall": overall,
        "family": family,
        "category_counts": dict(categories),
        "gate": gate,
        "groups": rescored_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--v1-audit", type=Path, required=True)
    parser.add_argument("--world-dir", type=Path, required=True)
    parser.add_argument("--output-tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = rescore(
        tasks_path=args.tasks,
        v1_audit_path=args.v1_audit,
        world_dir=args.world_dir,
        output_tasks_path=args.output_tasks,
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "groups"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
