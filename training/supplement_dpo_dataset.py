"""Append one fixed eight-rollout pass for DPO prompts missing a strict pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_grpo_hard import load_items, load_policy
from dpo_dataset import atomic_jsonl, audit_split, build_pairs, generate_split
from transformers import AutoTokenizer


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    items = load_items(args.train)
    item_by_id = {str(item["case_id"]): item for item in items}
    initial_audit = load_json(args.audit)
    missing_ids = [
        str(entry["case_id"]) for entry in initial_audit["train"]["missing_pairs"]
    ]
    if len(missing_ids) != 32 or len(set(missing_ids)) != 32:
        raise ValueError(f"expected 32 unique missing train prompts, got {len(missing_ids)}")
    missing_items = [item_by_id[case_id] for case_id in missing_ids]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_policy(args.model, args.adapter)
    supplement = generate_split(
        model=model,
        tokenizer=tokenizer,
        items=missing_items,
        split="train_supplement",
        base_seed=args.seed,
        adapter_identifier=str(args.adapter),
        rollout_index_offset=8,
    )
    if len(supplement) != 32 * 8:
        raise ValueError(f"expected 256 supplemental rollouts, got {len(supplement)}")
    original = load_jsonl(args.rollouts)
    merged = original + supplement
    pairs, missing = build_pairs(items=items, rollouts=merged)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(args.output_dir / "dpo_v1_rollouts_train_supplement.jsonl", supplement)
    atomic_jsonl(args.output_dir / "dpo_v1_rollouts_train_augmented.jsonl", merged)
    atomic_jsonl(args.output_dir / "dpo_v1_train_augmented.jsonl", pairs)
    audit = {
        "policy": str(args.adapter),
        "supplement_seed": args.seed,
        "supplement_prompt_count": len(missing_items),
        "supplement_rollout_count": len(supplement),
        "initial_strict_pair_count": int(initial_audit["train"]["strict_pair_count"]),
        "train": audit_split(tokenizer, items, merged, pairs, missing),
    }
    (args.output_dir / "dpo_v1_augmented_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
