"""Compare SFT and targeted-GRPO on fixed counterfactual dev/hard and ranking canary."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from audit_grpo_hard import atomic_json, load_policy
from grpo_test_suite_rewards import CounterfactualExecutionReward, serialize_breakdown
from transformers import AutoTokenizer

FAMILIES = ("aggregation_scope", "ties", "ranking")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _evaluate(
    *, model: torch.nn.Module, tokenizer: Any, items: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    torch.manual_seed(seed)
    verifier = CounterfactualExecutionReward(num_generations=4)
    groups: list[dict[str, Any]] = []
    for batch_start in range(0, len(items), 4):
        batch = items[batch_start : batch_start + 4]
        encoded = tokenizer(
            [item["prompt"] for item in batch],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(next(model.parameters()).device)
        width = int(encoded["input_ids"].shape[1])
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
        completions = tokenizer.batch_decode(outputs[:, width:], skip_special_tokens=True)
        for item_index, item in enumerate(batch):
            current = completions[item_index * 4 : (item_index + 1) * 4]
            start = len(verifier.history)
            rewards = verifier(
                prompts=[item["prompt"]] * 4,
                completions=current,
                case_id=[item["case_id"]] * 4,
                worlds_json=[item["worlds_json"]] * 4,
            )
            breakdowns = verifier.history[start:]
            groups.append(
                {
                    "case_id": item["case_id"],
                    "template_family": item["template_family"],
                    "split": item["split"],
                    "rewards": rewards,
                    "reward_std": statistics.pstdev(rewards),
                    "raw_completions": current,
                    "extracted_sql": [entry.sql for entry in breakdowns],
                    "breakdowns": [serialize_breakdown(entry) for entry in breakdowns],
                }
            )
        print(
            json.dumps(
                {
                    "stage": "evaluation_progress",
                    "completed_groups": len(groups),
                    "total_groups": len(items),
                }
            ),
            flush=True,
        )
        del encoded, outputs
        torch.cuda.empty_cache()
    return groups


def _metrics(groups: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(value) for group in groups for value in group["rewards"]]
    categories = Counter(
        breakdown["category"]
        for group in groups
        for breakdown in group["breakdowns"]
    )
    levels = Counter(str(value) for value in rewards)
    return {
        "group_count": len(groups),
        "rollout_count": len(rewards),
        "mean_semantic_reward": statistics.fmean(rewards),
        "exact_1_count": sum(value == 1.0 for value in rewards),
        "exact_1_rate": sum(value == 1.0 for value in rewards) / len(rewards),
        "reward_0_5_count": sum(value == 0.5 for value in rewards),
        "reward_0_25_count": sum(value == 0.25 for value in rewards),
        "reward_0_count": sum(value == 0.0 for value in rewards),
        "reward_level_counts": dict(levels),
        "zero_std_groups": sum(group["reward_std"] == 0.0 for group in groups),
        "execution_failure_count": categories["execution_failure"],
        "malformed_count": categories["malformed_tool_call"],
        "unsafe_count": categories["unsafe_sql"],
    }


def _summarize(groups: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in ("dev", "hard"):
        split_groups = [group for group in groups if group["split"] == split]
        target = [
            group
            for group in split_groups
            if group["template_family"] in {"aggregation_scope", "ties"}
        ]
        summary[f"target_{split}"] = _metrics(target)
        summary[f"ranking_{split}"] = _metrics(
            [group for group in split_groups if group["template_family"] == "ranking"]
        )
        summary[f"family_{split}"] = {
            family: _metrics(
                [group for group in split_groups if group["template_family"] == family]
            )
            for family in FAMILIES
        }
    summary["ranking_canary_combined"] = _metrics(
        [group for group in groups if group["template_family"] == "ranking"]
    )
    summary["all"] = _metrics(groups)
    return summary


def _delta(after: dict[str, Any], before: dict[str, Any], key: str) -> float:
    return float(after[key]) - float(before[key])


def _gate(sft: dict[str, Any], grpo: dict[str, Any]) -> dict[str, Any]:
    target_dev_mean_delta = _delta(
        grpo["target_dev"], sft["target_dev"], "mean_semantic_reward"
    )
    target_dev_exact_delta = int(grpo["target_dev"]["exact_1_count"]) - int(
        sft["target_dev"]["exact_1_count"]
    )
    ties_sft = sft["family_dev"]["ties"]
    ties_grpo = grpo["family_dev"]["ties"]
    ties_half_rate_delta = (
        int(ties_grpo["reward_0_5_count"]) / int(ties_grpo["rollout_count"])
        - int(ties_sft["reward_0_5_count"]) / int(ties_sft["rollout_count"])
    )
    family_deltas = {
        family: _delta(
            grpo["family_dev"][family],
            sft["family_dev"][family],
            "mean_semantic_reward",
        )
        for family in ("aggregation_scope", "ties")
    }
    ranking_mean_delta = _delta(
        grpo["ranking_canary_combined"],
        sft["ranking_canary_combined"],
        "mean_semantic_reward",
    )
    ranking_exact_delta = _delta(
        grpo["ranking_canary_combined"],
        sft["ranking_canary_combined"],
        "exact_1_rate",
    )
    checks = {
        "target_dev_mean_improved": target_dev_mean_delta > 0.0,
        "target_dev_exact_improved": target_dev_exact_delta > 0,
        "ties_exact_or_half_improved": int(ties_grpo["exact_1_count"]) > 0
        or ties_half_rate_delta >= 0.05,
        "target_hard_not_regressed": _delta(
            grpo["target_hard"], sft["target_hard"], "mean_semantic_reward"
        )
        >= 0.0
        and int(grpo["target_hard"]["exact_1_count"])
        >= int(sft["target_hard"]["exact_1_count"]),
        "target_families_balanced": max(family_deltas.values()) >= 0.02
        and min(family_deltas.values()) >= -0.02,
        "ranking_canary_not_materially_regressed": ranking_mean_delta >= -0.05
        and ranking_exact_delta >= -0.05,
        "unsafe_not_increased": int(grpo["all"]["unsafe_count"])
        <= int(sft["all"]["unsafe_count"]),
        "malformed_not_increased": int(grpo["all"]["malformed_count"])
        <= int(sft["all"]["malformed_count"]),
    }
    checks["eligible_for_final_system_evaluation"] = all(checks.values())
    return {
        "checks": checks,
        "deltas": {
            "target_dev_mean_semantic_reward": target_dev_mean_delta,
            "target_dev_exact_count": target_dev_exact_delta,
            "target_hard_mean_semantic_reward": _delta(
                grpo["target_hard"], sft["target_hard"], "mean_semantic_reward"
            ),
            "ties_dev_reward_0_5_rate": ties_half_rate_delta,
            "family_dev_mean_reward": family_deltas,
            "ranking_canary_mean_reward": ranking_mean_delta,
            "ranking_canary_exact_rate": ranking_exact_delta,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--grpo-adapter", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    items = [
        item
        for item in _read_jsonl(args.tasks)
        if item["split"] in {"dev", "hard"} and item["template_family"] in FAMILIES
    ]
    if len(items) != 24:
        raise ValueError(f"expected fixed 24 dev/hard tasks, got {len(items)}")

    sft_model = load_policy(args.model, args.sft_adapter)
    sft_groups = _evaluate(model=sft_model, tokenizer=tokenizer, items=items, seed=args.seed)
    del sft_model
    gc.collect()
    torch.cuda.empty_cache()

    grpo_model = load_policy(args.model, args.grpo_adapter)
    grpo_groups = _evaluate(model=grpo_model, tokenizer=tokenizer, items=items, seed=args.seed)
    del grpo_model
    gc.collect()
    torch.cuda.empty_cache()

    sft_summary = _summarize(sft_groups)
    grpo_summary = _summarize(grpo_groups)
    report = {
        "experiment": "targeted_counterfactual_grpo_pilot",
        "reason_ranking_is_canary": (
            "Ranking low variance comes from high SFT correctness and low entropy, so it is "
            "excluded from gradient updates and retained as a no-regression canary."
        ),
        "settings": {
            "seed": args.seed,
            "temperature": 1.0,
            "num_generations": 4,
            "top_p": 0.95,
            "top_k": 20,
            "max_completion_length": 256,
        },
        "sft": {"summary": sft_summary, "groups": sft_groups},
        "grpo": {"summary": grpo_summary, "groups": grpo_groups},
        "gate": _gate(sft_summary, grpo_summary),
    }
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "experiment": report["experiment"],
                "reason_ranking_is_canary": report["reason_ranking_is_canary"],
                "settings": report["settings"],
                "sft": sft_summary,
                "grpo": grpo_summary,
                "gate": report["gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
