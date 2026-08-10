"""Audit higher-exploration SFT rollouts on fixed V1.1 ranking/ties tasks."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from audit_grpo_hard import atomic_json, load_policy
from grpo_test_suite_rewards import CounterfactualExecutionReward, serialize_breakdown
from transformers import AutoTokenizer

FAMILIES = ("ranking", "ties")
NUM_GENERATIONS = 8


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _reward_summary(groups: list[dict[str, Any]]) -> dict[str, Any]:
    standard_deviations = [float(group["reward_std"]) for group in groups]
    levels = Counter(
        str(reward) for group in groups for reward in group["suite_rewards"]
    )
    zero_std = sum(value == 0.0 for value in standard_deviations)
    return {
        "group_count": len(groups),
        "unique_sql_groups": sum(group["unique_sql_count"] >= 2 for group in groups),
        "unique_sql_group_rate": sum(group["unique_sql_count"] >= 2 for group in groups)
        / len(groups),
        "mixed_groups": len(groups) - zero_std,
        "mixed_group_rate": (len(groups) - zero_std) / len(groups),
        "zero_std_groups": zero_std,
        "zero_std_group_rate": zero_std / len(groups),
        "mean_group_reward_std": statistics.fmean(standard_deviations),
        "unique_reward_levels": sorted({float(value) for value in levels}),
        "reward_level_counts": dict(levels),
    }


def _old_summary(groups: list[dict[str, Any]]) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for group in groups:
        rewards = list(map(float, group["suite_rewards"]))
        sql_values = {
            breakdown.get("sql")
            for breakdown in group["breakdowns"]
            if breakdown.get("sql") is not None
        }
        normalized.append(
            {
                "suite_rewards": rewards,
                "reward_std": statistics.pstdev(rewards),
                "unique_sql_count": len(sql_values),
            }
        )
    return _reward_summary(normalized)


def _category_rates(groups: list[dict[str, Any]]) -> dict[str, Any]:
    categories = Counter(
        breakdown["category"]
        for group in groups
        for breakdown in group["breakdowns"]
    )
    total = sum(categories.values())
    return {
        "rollout_count": total,
        "counts": dict(categories),
        "malformed_rate": categories["malformed_tool_call"] / total,
        "unsafe_rate": categories["unsafe_sql"] / total,
        "execution_failure_rate": categories["execution_failure"] / total,
    }


def run_audit(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    tasks: list[dict[str, Any]],
    old_report: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    selected = [item for item in tasks if item["template_family"] in FAMILIES]
    if len(selected) != 48 or Counter(item["template_family"] for item in selected) != {
        "ranking": 24,
        "ties": 24,
    }:
        raise ValueError("exploration audit requires exactly 24 ranking and 24 ties tasks")
    old_groups = [
        group for group in old_report["groups"] if group["template_family"] in FAMILIES
    ]
    if len(old_groups) != 48:
        raise ValueError("V1.1 report does not contain the fixed 48 comparison groups")

    torch.manual_seed(seed)
    verifier = CounterfactualExecutionReward(num_generations=NUM_GENERATIONS)
    groups: list[dict[str, Any]] = []
    for batch_start in range(0, len(selected), 2):
        batch = selected[batch_start : batch_start + 2]
        encoded = tokenizer(
            [item["prompt"] for item in batch],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(next(model.parameters()).device)
        prompt_width = int(encoded["input_ids"].shape[1])
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                do_sample=True,
                temperature=1.1,
                top_p=0.95,
                top_k=20,
                num_return_sequences=NUM_GENERATIONS,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completions = tokenizer.batch_decode(
            outputs[:, prompt_width:], skip_special_tokens=True
        )
        for item_index, item in enumerate(batch):
            start = item_index * NUM_GENERATIONS
            current = completions[start : start + NUM_GENERATIONS]
            history_start = len(verifier.history)
            rewards = verifier(
                prompts=[item["prompt"]] * NUM_GENERATIONS,
                completions=current,
                case_id=[item["case_id"]] * NUM_GENERATIONS,
                worlds_json=[item["worlds_json"]] * NUM_GENERATIONS,
            )
            breakdowns = verifier.history[history_start:]
            categories = Counter(entry.category for entry in breakdowns)
            sql_values = {entry.sql for entry in breakdowns if entry.sql is not None}
            reward_std = statistics.pstdev(rewards)
            groups.append(
                {
                    "case_id": item["case_id"],
                    "template_family": item["template_family"],
                    "split": item["split"],
                    "unique_raw_completion_count": len(set(current)),
                    "unique_sql_count": len(sql_values),
                    "correct_sql_count": categories["correct"],
                    "partial_reward_sql_count": categories["partial_semantic"],
                    "semantic_mismatch_count": categories["semantic_mismatch"],
                    "execution_failure_count": categories["execution_failure"],
                    "malformed_count": categories["malformed_tool_call"],
                    "unsafe_count": categories["unsafe_sql"],
                    "suite_rewards": rewards,
                    "reward_std": reward_std,
                    "unique_reward_level_count": len(set(rewards)),
                    "zero_std": reward_std == 0.0,
                    "mixed_group": reward_std > 0.0,
                    "raw_completions": current,
                    "extracted_sql": [entry.sql for entry in breakdowns],
                    "breakdowns": [serialize_breakdown(entry) for entry in breakdowns],
                }
            )
        completed = len(groups)
        if completed % 8 == 0 or completed == len(selected):
            print(
                json.dumps(
                    {
                        "stage": "exploration_progress",
                        "completed_groups": completed,
                        "total_groups": len(selected),
                    }
                ),
                flush=True,
            )
        del encoded, outputs
        torch.cuda.empty_cache()

    new_family = {
        family: _reward_summary(
            [group for group in groups if group["template_family"] == family]
        )
        for family in FAMILIES
    }
    old_family = {
        family: _old_summary(
            [group for group in old_groups if group["template_family"] == family]
        )
        for family in FAMILIES
    }
    old_safety = _category_rates(old_groups)
    new_safety = _category_rates(groups)
    safety_gate = {
        "malformed_not_materially_increased": new_safety["malformed_rate"]
        <= old_safety["malformed_rate"] + 0.02,
        "unsafe_not_materially_increased": new_safety["unsafe_rate"]
        <= old_safety["unsafe_rate"] + 0.01,
    }
    gate = {
        "ranking_mixed_at_least_25_percent": new_family["ranking"][
            "mixed_group_rate"
        ]
        >= 0.25,
        "ties_mixed_at_least_50_percent": new_family["ties"]["mixed_group_rate"]
        >= 0.50,
        **safety_gate,
    }
    gate["resume_counterfactual_grpo"] = all(gate.values())
    return {
        "experiment": "counterfactual_exploration_v1.1",
        "settings": {
            "seed": seed,
            "temperature": 1.1,
            "num_generations": NUM_GENERATIONS,
            "top_p": 0.95,
            "top_k": 20,
            "max_completion_length": 256,
        },
        "model_called": True,
        "task_count": len(selected),
        "rollout_count": len(selected) * NUM_GENERATIONS,
        "old": {"settings": old_report.get("settings"), "family": old_family},
        "new": {"family": new_family},
        "safety": {"old": old_safety, "new": new_safety},
        "gate": gate,
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--old-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tasks = _read_jsonl(args.tasks)
    old_report = json.loads(args.old_report.read_text(encoding="utf-8"))
    model = load_policy(args.model, args.adapter)
    report = run_audit(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        old_report=old_report,
        seed=args.seed,
    )
    atomic_json(args.output, report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "groups"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
