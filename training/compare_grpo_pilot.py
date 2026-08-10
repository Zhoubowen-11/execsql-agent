"""Compare frozen SFT and GRPO adapters on fixed RL dev/hard splits."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from audit_grpo_hard import atomic_json, audit_policy, load_items, load_policy
from transformers import AutoTokenizer


def metrics(audit: dict[str, Any]) -> dict[str, Any]:
    groups = audit["groups"]
    categories = Counter(category for group in groups for category in group["categories"])
    total = len(groups) * 4
    family_counts: dict[str, dict[str, float | int]] = {}
    for family in sorted({str(group["template_family"]) for group in groups}):
        family_groups = [group for group in groups if group["template_family"] == family]
        correct = sum(
            category == "correct" for group in family_groups for category in group["categories"]
        )
        family_total = len(family_groups) * 4
        family_counts[family] = {
            "correct_count": correct,
            "total": family_total,
            "correct_rate": correct / family_total,
        }
    return {
        "groups": len(groups),
        "rollout_total": total,
        "correct": categories["correct"],
        "correct_rate": categories["correct"] / total,
        "semantic_mismatch": categories["semantic_mismatch"],
        "semantic_mismatch_rate": categories["semantic_mismatch"] / total,
        "execution_failure": categories["execution_failure"],
        "execution_failure_rate": categories["execution_failure"] / total,
        "malformed": categories["malformed_tool_call"],
        "malformed_rate": categories["malformed_tool_call"] / total,
        "unsafe": categories["unsafe_sql"],
        "unsafe_rate": categories["unsafe_sql"] / total,
        "zero_std_groups": sum(bool(group["metrics"]["zero_std"]) for group in groups),
        "family": family_counts,
    }


def compare(sft: dict[str, Any], grpo: dict[str, Any]) -> dict[str, Any]:
    sft_metrics = metrics(sft)
    grpo_metrics = metrics(grpo)
    family_delta = {
        family: {
            "sft_correct_count": values["correct_count"],
            "sft_correct_rate": values["correct_rate"],
            "grpo_correct_count": grpo_metrics["family"][family]["correct_count"],
            "grpo_correct_rate": grpo_metrics["family"][family]["correct_rate"],
            "delta_count": int(grpo_metrics["family"][family]["correct_count"])
            - int(values["correct_count"]),
            "delta_rate": float(grpo_metrics["family"][family]["correct_rate"])
            - float(values["correct_rate"]),
        }
        for family, values in sft_metrics["family"].items()
    }
    return {"sft": sft_metrics, "grpo": grpo_metrics, "family_delta": family_delta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--grpo-adapter", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--hard", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev-seed", type=int, default=20260812)
    parser.add_argument("--hard-seed", type=int, default=20260813)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dev_items = load_items(args.dev)
    hard_items = load_items(args.hard)
    results: dict[str, dict[str, Any]] = {}
    for policy_name, adapter in (("sft", args.sft_adapter), ("grpo", args.grpo_adapter)):
        model = load_policy(args.model, adapter)
        results[f"{policy_name}_dev"] = audit_policy(
            model=model,
            tokenizer=tokenizer,
            items=dev_items,
            temperature=1.0,
            batch_groups=4,
            seed=args.dev_seed,
        )
        results[f"{policy_name}_hard"] = audit_policy(
            model=model,
            tokenizer=tokenizer,
            items=hard_items,
            temperature=1.0,
            batch_groups=4,
            seed=args.hard_seed,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, audit in results.items():
        atomic_json(args.output_dir / f"{name}.json", audit)
    summary = {
        "settings": {
            "temperature": 1.0,
            "num_generations": 4,
            "max_completion_length": 256,
            "dev_seed": args.dev_seed,
            "hard_seed": args.hard_seed,
        },
        "mixed_dev": compare(results["sft_dev"], results["grpo_dev"]),
        "hard_audit": compare(results["sft_hard"], results["grpo_hard"]),
    }
    atomic_json(args.output_dir / "comparison_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
