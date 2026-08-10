"""Audit binary versus counterfactual reward diversity using the original SFT policy."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from audit_grpo_hard import atomic_json, load_items, load_policy
from grpo_test_suite_rewards import CounterfactualExecutionReward, serialize_breakdown
from transformers import AutoTokenizer


def _group_summary(groups: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [list(map(float, group[key])) for group in groups]
    stds = [statistics.pstdev(group) for group in values]
    flattened = [reward for group in values for reward in group]
    return {
        "zero_std_groups": sum(std == 0.0 for std in stds),
        "zero_std_group_rate": sum(std == 0.0 for std in stds) / len(stds),
        "mean_group_reward_std": statistics.fmean(stds),
        "mixed_groups": sum(std > 0.0 for std in stds),
        "mixed_group_rate": sum(std > 0.0 for std in stds) / len(stds),
        "all_zero_groups": sum(group == [0.0] * 4 for group in values),
        "all_one_groups": sum(group == [1.0] * 4 for group in values),
        "unique_reward_levels": sorted(set(flattened)),
        "reward_level_counts": dict(Counter(str(value) for value in flattened)),
        "partial_reward_rollouts": sum(0.0 < value < 1.0 for value in flattened),
        "mean_reward": statistics.fmean(flattened),
    }


def audit_policy(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    items: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    reward = CounterfactualExecutionReward(num_generations=4)
    groups: list[dict[str, Any]] = []
    for batch_start in range(0, len(items), 4):
        batch = items[batch_start : batch_start + 4]
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
                temperature=1.0,
                top_p=0.95,
                top_k=20,
                num_return_sequences=4,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completions = tokenizer.batch_decode(
            outputs[:, prompt_width:], skip_special_tokens=True
        )
        for item_index, item in enumerate(batch):
            current = completions[item_index * 4 : (item_index + 1) * 4]
            history_start = len(reward.history)
            suite_rewards = reward(
                prompts=[item["prompt"]] * 4,
                completions=current,
                case_id=[item["case_id"]] * 4,
                worlds_json=[item["worlds_json"]] * 4,
            )
            breakdowns = reward.history[history_start:]
            groups.append(
                {
                    "case_id": item["case_id"],
                    "template_family": item["template_family"],
                    "split": item["split"],
                    "binary_rewards": [entry.binary_reward for entry in breakdowns],
                    "suite_rewards": suite_rewards,
                    "completions": current,
                    "breakdowns": [serialize_breakdown(entry) for entry in breakdowns],
                }
            )
        completed = len(groups)
        if completed % 12 == 0 or completed == len(items):
            print(
                json.dumps(
                    {
                        "stage": "counterfactual_rollout_progress",
                        "completed_groups": completed,
                        "total_groups": len(items),
                    }
                ),
                flush=True,
            )
        del encoded, outputs
        torch.cuda.empty_cache()
    categories = Counter(
        breakdown["category"]
        for group in groups
        for breakdown in group["breakdowns"]
    )
    binary = _group_summary(groups, "binary_rewards")
    suite = _group_summary(groups, "suite_rewards")
    gate = {
        "zero_std_reduced": suite["zero_std_group_rate"]
        < binary["zero_std_group_rate"],
        "partial_levels_present": all(
            level in suite["unique_reward_levels"] for level in (0.25, 0.5, 0.75)
        ),
        "many_partial_rollouts": suite["partial_reward_rollouts"] >= 24,
        "malformed_rate_controlled": categories["malformed_tool_call"]
        / (len(items) * 4)
        <= 0.1,
        "unsafe_rate_controlled": categories["unsafe_sql"] / (len(items) * 4) <= 0.05,
    }
    gate["proceed_to_grpo"] = all(gate.values())
    family: dict[str, dict[str, Any]] = {}
    for name in sorted({str(group["template_family"]) for group in groups}):
        selected = [group for group in groups if group["template_family"] == name]
        family[name] = {
            "group_count": len(selected),
            "binary": _group_summary(selected, "binary_rewards"),
            "suite": _group_summary(selected, "suite_rewards"),
        }
    return {
        "settings": {
            "seed": seed,
            "temperature": 1.0,
            "num_generations": 4,
            "max_completion_length": 256,
        },
        "task_count": len(items),
        "rollout_count": len(items) * 4,
        "binary": binary,
        "test_suite": suite,
        "category_counts": dict(categories),
        "family": family,
        "gate": gate,
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    items = load_items(args.tasks)
    model = load_policy(args.model, args.adapter)
    audit = audit_policy(model=model, tokenizer=tokenizer, items=items, seed=args.seed)
    atomic_json(args.output, audit)
    print(json.dumps({key: value for key, value in audit.items() if key != "groups"}, indent=2))


if __name__ == "__main__":
    main()
