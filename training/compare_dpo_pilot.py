"""Compare original SFT and DPO adapters on fixed RL-only dev/hard splits."""

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
    """Summarize execution-reward rollout outcomes with original counts."""

    groups = audit["groups"]
    categories = Counter(category for group in groups for category in group["categories"])
    total = len(groups) * 4
    family: dict[str, dict[str, float | int]] = {}
    for name in sorted({str(group["template_family"]) for group in groups}):
        selected = [group for group in groups if group["template_family"] == name]
        correct = sum(
            category == "correct" for group in selected for category in group["categories"]
        )
        family_total = len(selected) * 4
        family[name] = {
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
        "execution_failure": categories["execution_failure"],
        "malformed": categories["malformed_tool_call"],
        "unsafe": categories["unsafe_sql"],
        "zero_std_groups": sum(bool(group["metrics"]["zero_std"]) for group in groups),
        "family": family,
    }


def compare(sft: dict[str, Any], dpo: dict[str, Any]) -> dict[str, Any]:
    """Compare two audits generated with identical fixed sampling settings."""

    sft_metrics = metrics(sft)
    dpo_metrics = metrics(dpo)
    family_delta = {
        family: {
            "sft_correct_count": values["correct_count"],
            "sft_correct_rate": values["correct_rate"],
            "dpo_correct_count": dpo_metrics["family"][family]["correct_count"],
            "dpo_correct_rate": dpo_metrics["family"][family]["correct_rate"],
            "delta_count": int(dpo_metrics["family"][family]["correct_count"])
            - int(values["correct_count"]),
            "delta_rate": float(dpo_metrics["family"][family]["correct_rate"])
            - float(values["correct_rate"]),
        }
        for family, values in sft_metrics["family"].items()
    }
    return {"sft": sft_metrics, "dpo": dpo_metrics, "family_delta": family_delta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--dpo-adapter", type=Path, required=True)
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
    for name, adapter in (("sft", args.sft_adapter), ("dpo", args.dpo_adapter)):
        model = load_policy(args.model, adapter)
        results[f"{name}_dev"] = audit_policy(
            model=model,
            tokenizer=tokenizer,
            items=dev_items,
            temperature=1.0,
            batch_groups=4,
            seed=args.dev_seed,
        )
        results[f"{name}_hard"] = audit_policy(
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
    dev = compare(results["sft_dev"], results["dpo_dev"])
    hard = compare(results["sft_hard"], results["dpo_hard"])
    positive_families = sum(
        values["delta_count"] > 0 for values in dev["family_delta"].values()
    )
    ranking_delta = int(dev["family_delta"].get("ranking", {}).get("delta_count", 0))
    gate = {
        "mixed_dev_improves_by_more_than_two_rollouts": (
            int(dev["dpo"]["correct"]) - int(dev["sft"]["correct"]) > 2
        ),
        "hard_audit_not_regressed": int(hard["dpo"]["correct"])
        >= int(hard["sft"]["correct"]),
        "at_least_three_dev_families_improve": positive_families >= 3,
        "ranking_not_catastrophically_regressed": ranking_delta >= -2,
        "unsafe_not_increased": (
            int(dev["dpo"]["unsafe"]) <= int(dev["sft"]["unsafe"])
            and int(hard["dpo"]["unsafe"]) <= int(hard["sft"]["unsafe"])
        ),
        "malformed_not_increased": (
            int(dev["dpo"]["malformed"]) <= int(dev["sft"]["malformed"])
            and int(hard["dpo"]["malformed"]) <= int(hard["sft"]["malformed"])
        ),
    }
    gate["recommend_continue_dpo"] = all(gate.values())
    summary = {
        "settings": {
            "temperature": 1.0,
            "num_generations": 4,
            "max_completion_length": 256,
            "dev_seed": args.dev_seed,
            "hard_seed": args.hard_seed,
        },
        "mixed_dev": dev,
        "hard_audit": hard,
        "positive_dev_family_count": positive_families,
        "gate": gate,
    }
    atomic_json(args.output_dir / "comparison_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
