"""Sample SFT/GRPO policies on RL-hard prompts and classify reward diversity."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from grpo_rewards import SQLExecutionReward
from peft import PeftModel
from train_qlora_sft import inspect_quantization, load_quantized_base
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/root/autodl-tmp/models/Qwen3-8B")
DEFAULT_ADAPTER = (
    PROJECT_ROOT / "artifacts/qlora_sft/qwen3_8b_execsql_sft_v1_3epoch"
)
DEFAULT_CANDIDATES = (
    PROJECT_ROOT / "data/fsq/train/grpo/rl_train_hard_v1_candidates.jsonl"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data/fsq/train/grpo/audit_sft_temp_1_0.json"


def load_items(path: Path) -> list[dict[str, Any]]:
    """Load independent JSONL items without touching any evaluation dataset."""

    with path.open("r", encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    if not items or not all(isinstance(item, dict) for item in items):
        raise ValueError(f"invalid or empty RL-hard dataset: {path}")
    return items


def load_policy(model_path: Path, adapter_path: Path) -> torch.nn.Module:
    """Load a frozen 4-bit policy adapter for rollout-only auditing."""

    base = load_quantized_base(model_path)
    inspect_quantization(base)
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.config.use_cache = True
    model.eval()
    return model


def classify_group(rewards: list[float], categories: list[str]) -> str:
    """Assign the requested A/B/C/D difficulty class."""

    if rewards == [1.0, 1.0, 1.0, 1.0]:
        return "all_correct"
    if any(reward == 1.0 for reward in rewards):
        return "mixed"
    if all(category == "semantic_mismatch" for category in categories):
        return "all_semantic_wrong"
    return "execution_malformed_unsafe_dominated"


def audit_policy(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    items: list[dict[str, Any]],
    temperature: float,
    batch_groups: int,
    seed: int,
) -> dict[str, Any]:
    """Generate four rollouts per prompt and score every SQL against SQLite."""

    if batch_groups < 1:
        raise ValueError("batch_groups must be positive")
    torch.manual_seed(seed)
    reward = SQLExecutionReward(max_rows=100, num_generations=4)
    groups: list[dict[str, Any]] = []
    for batch_start in range(0, len(items), batch_groups):
        batch = items[batch_start : batch_start + batch_groups]
        encoded = tokenizer(
            [item["prompt"] for item in batch],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(next(model.parameters()).device)
        prompt_width = int(encoded["input_ids"].shape[1])
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                top_k=20,
                num_return_sequences=4,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completions = tokenizer.batch_decode(
            output_ids[:, prompt_width:], skip_special_tokens=True
        )
        for item_index, item in enumerate(batch):
            start = item_index * 4
            current_completions = completions[start : start + 4]
            breakdown_start = len(reward.history)
            rewards = reward(
                prompts=[item["prompt"]] * 4,
                completions=current_completions,
                case_id=[item["case_id"]] * 4,
                database_path=[item["database_path"]] * 4,
                expected_result_json=[item["expected_result_json"]] * 4,
            )
            breakdowns = reward.history[breakdown_start:]
            categories = [entry.category for entry in breakdowns]
            metrics = reward.group_history[-1]
            groups.append(
                {
                    "case_id": item["case_id"],
                    "template_family": item["template_family"],
                    "difficulty": item.get("difficulty", "unknown"),
                    "prompt_token_count": item["prompt_token_count"],
                    "rewards": rewards,
                    "categories": categories,
                    "classification": classify_group(rewards, categories),
                    "metrics": asdict(metrics),
                    "rollouts": [asdict(entry) for entry in breakdowns],
                }
            )
        if len(groups) % 10 == 0 or len(groups) == len(items):
            print(
                json.dumps(
                    {
                        "stage": "rollout_progress",
                        "completed_groups": len(groups),
                        "total_groups": len(items),
                        "temperature": temperature,
                    }
                ),
                flush=True,
            )
        del encoded, output_ids
        torch.cuda.empty_cache()

    class_counts = Counter(group["classification"] for group in groups)
    category_counts = Counter(
        category for group in groups for category in group["categories"]
    )
    rollout_count = len(groups) * 4
    mixed_stds = [
        float(group["metrics"]["reward_std"])
        for group in groups
        if group["classification"] == "mixed"
    ]
    token_lengths = [int(group["prompt_token_count"]) for group in groups]
    semantic_failures_by_family = Counter(
        group["template_family"]
        for group in groups
        for category in group["categories"]
        if category == "semantic_mismatch"
    )
    family_class_counts: dict[str, dict[str, int]] = {}
    difficulty_class_counts: dict[str, dict[str, int]] = {}
    for group in groups:
        family = str(group["template_family"])
        difficulty = str(group["difficulty"])
        classification = str(group["classification"])
        family_class_counts.setdefault(family, {})[classification] = (
            family_class_counts.setdefault(family, {}).get(classification, 0) + 1
        )
        difficulty_class_counts.setdefault(difficulty, {})[classification] = (
            difficulty_class_counts.setdefault(difficulty, {}).get(classification, 0) + 1
        )
    mixed_correct_counts = Counter(
        sum(reward == 1.0 for reward in group["rewards"])
        for group in groups
        if group["classification"] == "mixed"
    )
    summary = {
        "candidate_count": len(groups),
        "temperature": temperature,
        "seed": seed,
        "class_counts": dict(class_counts),
        "class_rates": {
            key: class_counts.get(key, 0) / len(groups)
            for key in (
                "all_correct",
                "mixed",
                "all_semantic_wrong",
                "execution_malformed_unsafe_dominated",
            )
        },
        "family_class_counts": family_class_counts,
        "difficulty_class_counts": difficulty_class_counts,
        "family_mixed_counts": {
            family: counts.get("mixed", 0)
            for family, counts in family_class_counts.items()
        },
        "mixed_correct_count_distribution": {
            f"{correct_count}_of_4": mixed_correct_counts.get(correct_count, 0)
            for correct_count in (1, 2, 3)
        },
        "correct_rollout_rate": category_counts.get("correct", 0) / rollout_count,
        "semantic_mismatch_rate": category_counts.get("semantic_mismatch", 0)
        / rollout_count,
        "malformed_rate": category_counts.get("malformed_tool_call", 0) / rollout_count,
        "execution_failure_rate": category_counts.get("execution_failure", 0)
        / rollout_count,
        "unsafe_rate": category_counts.get("unsafe_sql", 0) / rollout_count,
        "zero_std_group_fraction": sum(
            bool(group["metrics"]["zero_std"]) for group in groups
        )
        / len(groups),
        "mixed_reward_std": {
            "count": len(mixed_stds),
            "min": min(mixed_stds) if mixed_stds else None,
            "mean": statistics.fmean(mixed_stds) if mixed_stds else None,
            "median": statistics.median(mixed_stds) if mixed_stds else None,
            "max": max(mixed_stds) if mixed_stds else None,
        },
        "prompt_tokens": {
            "min": min(token_lengths),
            "mean": statistics.fmean(token_lengths),
            "max": max(token_lengths),
        },
        "semantic_failures_by_family": dict(semantic_failures_by_family),
    }
    return {"summary": summary, "groups": groups}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def write_difficulty_splits(
    *,
    items: list[dict[str, Any]],
    audit: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Create disjoint mixed train/dev and a stratified all-wrong audit subset."""

    by_id = {str(item["case_id"]): item for item in items}
    groups = audit.get("groups")
    if not isinstance(groups, list):
        raise ValueError("audit JSON has no groups array")
    mixed_by_family: dict[str, list[str]] = {}
    wrong_by_family: dict[str, list[str]] = {}
    for group in groups:
        if not isinstance(group, dict):
            continue
        family = str(group["template_family"])
        case_id = str(group["case_id"])
        classification = group.get("classification")
        if classification == "mixed":
            mixed_by_family.setdefault(family, []).append(case_id)
        elif classification in {
            "all_semantic_wrong",
            "execution_malformed_unsafe_dominated",
        }:
            wrong_by_family.setdefault(family, []).append(case_id)

    train_ids: list[str] = []
    dev_ids: list[str] = []
    for family in sorted(mixed_by_family):
        family_ids = sorted(mixed_by_family[family])
        if len(family_ids) >= 2:
            dev_ids.append(family_ids[0])
            train_ids.extend(family_ids[1:])
        else:
            train_ids.extend(family_ids)
    hard_ids = [
        case_id
        for family in sorted(wrong_by_family)
        for case_id in sorted(wrong_by_family[family])[:2]
    ]
    if set(train_ids) & set(dev_ids):
        raise ValueError("mixed train/dev split overlap")
    train_items = [by_id[case_id] for case_id in train_ids]
    dev_items = [by_id[case_id] for case_id in dev_ids]
    hard_items = [by_id[case_id] for case_id in hard_ids]
    atomic_jsonl(output_dir / "rl_train_hard_v1_mixed_train.jsonl", train_items)
    atomic_jsonl(output_dir / "rl_train_hard_v1_mixed_dev.jsonl", dev_items)
    atomic_jsonl(output_dir / "rl_train_hard_v1_hard_audit.jsonl", hard_items)
    return {
        "mixed_train_count": len(train_items),
        "mixed_dev_count": len(dev_items),
        "hard_audit_count": len(hard_items),
        "mixed_train_families": dict(
            Counter(item["template_family"] for item in train_items)
        ),
        "mixed_dev_families": dict(
            Counter(item["template_family"] for item in dev_items)
        ),
        "hard_audit_families": dict(
            Counter(item["template_family"] for item in hard_items)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-groups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-existing-audit", type=Path)
    parser.add_argument("--split-output-dir", type=Path)
    args = parser.parse_args()
    items = load_items(args.dataset)
    if args.split_existing_audit is not None:
        if args.split_output_dir is None:
            raise ValueError("--split-existing-audit requires --split-output-dir")
        audit = json.loads(args.split_existing_audit.read_text(encoding="utf-8"))
        split_summary = write_difficulty_splits(
            items=items, audit=audit, output_dir=args.split_output_dir
        )
        print(json.dumps(split_summary, ensure_ascii=False, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for policy rollout auditing")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_policy(args.model, args.adapter)
    audit = audit_policy(
        model=model,
        tokenizer=tokenizer,
        items=items,
        temperature=args.temperature,
        batch_groups=args.batch_groups,
        seed=args.seed,
    )
    atomic_json(args.output, audit)
    print(json.dumps(audit["summary"], ensure_ascii=False, indent=2))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
