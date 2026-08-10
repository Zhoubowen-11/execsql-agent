"""Generate raw SFT rollouts and strict semantic DPO preference pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from execsql_agent.evaluation.comparator import compare_execution_result
from execsql_agent.models import ExpectedResult
from execsql_agent.tools.sql_executor import SQLExecutor
from execsql_agent.tools.sql_validator import SQLValidator

if TYPE_CHECKING:
    import torch
    from grpo_rewards import SQLExecutionReward


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def classify_completion(
    *,
    completion: str,
    item: dict[str, Any],
    reward: SQLExecutionReward,
) -> dict[str, Any]:
    """Score one raw completion and retain validation/execution evidence."""

    history_start = len(reward.history)
    values = reward(
        prompts=[item["prompt"]],
        completions=[completion],
        case_id=[item["case_id"]],
        database_path=[item["database_path"]],
        expected_result_json=[item["expected_result_json"]],
    )
    breakdown = reward.history[history_start]
    if values != [breakdown.reward]:
        raise ValueError("reward history mismatch")
    parsed_tool_call = None
    validation = None
    execution = None
    result_correct = breakdown.result_correct
    if breakdown.sql is not None:
        parsed_tool_call = {
            "name": "validate_sql",
            "arguments": {"sql": breakdown.sql},
        }
        validation_result = SQLValidator().validate(
            breakdown.sql, database_path=item["database_path"]
        )
        validation = validation_result.model_dump(mode="json")
        if validation_result.safe:
            execution_result = SQLExecutor(item["database_path"], max_rows=100).execute(
                breakdown.sql
            )
            execution = execution_result.model_dump(mode="json")
            expected = ExpectedResult.model_validate_json(item["expected_result_json"])
            result_correct = compare_execution_result(execution_result, expected)
    return {
        "raw_assistant_completion": completion,
        "parsed_tool_call": parsed_tool_call,
        "extracted_sql": breakdown.sql,
        "validator_result": validation,
        "execution_result": execution,
        "result_correct": result_correct,
        "failure_kind": None if breakdown.category == "correct" else breakdown.category,
        "reward": breakdown.reward,
        "category": breakdown.category,
    }


def generate_split(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    items: list[dict[str, Any]],
    split: str,
    base_seed: int,
    adapter_identifier: str,
    rollout_index_offset: int = 0,
) -> list[dict[str, Any]]:
    """Generate eight raw completions per prompt under a fixed protocol."""

    import torch
    from grpo_rewards import SQLExecutionReward

    rows: list[dict[str, Any]] = []
    reward = SQLExecutionReward(max_rows=100)
    sampling_config = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "num_generations": 8,
        "max_completion_length": 256,
        "do_sample": True,
        "batch_groups": 2,
    }
    torch.manual_seed(base_seed)
    completed_prompts = 0
    for batch_start in range(0, len(items), 2):
        batch = items[batch_start : batch_start + 2]
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
                temperature=1.0,
                top_p=0.95,
                top_k=20,
                num_return_sequences=8,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        all_completions = tokenizer.batch_decode(
            output_ids[:, prompt_width:], skip_special_tokens=True
        )
        if len(all_completions) != len(batch) * 8:
            raise ValueError("batch did not return eight completions per prompt")
        for item_index, item in enumerate(batch):
            completions = all_completions[item_index * 8 : (item_index + 1) * 8]
            digest = hashlib.sha256(str(item["prompt"]).encode("utf-8")).hexdigest()
            for local_rollout_index, completion in enumerate(completions):
                outcome = classify_completion(completion=completion, item=item, reward=reward)
                rows.append(
                    {
                        "case_id": item["case_id"],
                        "split": split,
                        "family": item["template_family"],
                        "prompt_hash": digest,
                        "prompt_index": batch_start + item_index,
                        "rollout_index": rollout_index_offset + local_rollout_index,
                        "seed": base_seed,
                        **outcome,
                        "sampling_config": sampling_config,
                        "sft_adapter_identifier": adapter_identifier,
                    }
                )
        completed_prompts += len(batch)
        if completed_prompts % 10 == 0 or completed_prompts == len(items):
            print(
                json.dumps(
                    {
                        "stage": "dpo_rollout_progress",
                        "split": split,
                        "completed_prompts": completed_prompts,
                        "total_prompts": len(items),
                    }
                ),
                flush=True,
            )
        del encoded, output_ids
        torch.cuda.empty_cache()
    return rows


def build_pairs(
    *, items: list[dict[str, Any]], rollouts: list[dict[str, Any]]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Choose the first correct and first semantic mismatch by rollout index."""

    item_by_id = {str(item["case_id"]): item for item in items}
    rows_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in rollouts:
        rows_by_id.setdefault(str(row["case_id"]), []).append(row)
    pairs: list[dict[str, str]] = []
    missing: list[dict[str, object]] = []
    for case_id in item_by_id:
        rows = sorted(rows_by_id[case_id], key=lambda row: int(row["rollout_index"]))
        chosen = next(
            (
                row
                for row in rows
                if row["category"] == "correct"
                and row["result_correct"] is True
                and row["extracted_sql"] is not None
            ),
            None,
        )
        rejected = next(
            (
                row
                for row in rows
                if row["category"] == "semantic_mismatch"
                and row["result_correct"] is False
                and row["extracted_sql"] is not None
            ),
            None,
        )
        if chosen is None or rejected is None:
            missing.append(
                {
                    "case_id": case_id,
                    "family": item_by_id[case_id]["template_family"],
                    "reason": "missing_chosen" if chosen is None else "missing_semantic_rejected",
                    "categories": dict(Counter(str(row["category"]) for row in rows)),
                }
            )
            continue
        if chosen["raw_assistant_completion"] == rejected["raw_assistant_completion"]:
            raise ValueError(f"{case_id}: chosen and rejected completion are identical")
        pairs.append(
            {
                "case_id": case_id,
                "family": str(item_by_id[case_id]["template_family"]),
                "prompt": str(item_by_id[case_id]["prompt"]),
                "chosen": str(chosen["raw_assistant_completion"]),
                "rejected": str(rejected["raw_assistant_completion"]),
            }
        )
    return pairs, {"missing_pairs": missing}


def token_stats(tokenizer: Any, pairs: list[dict[str, str]]) -> dict[str, Any]:
    def summarize(values: list[int]) -> dict[str, float | int]:
        return {
            "min": min(values),
            "mean": statistics.fmean(values),
            "max": max(values),
        }

    prompt = [len(tokenizer(row["prompt"], add_special_tokens=False).input_ids) for row in pairs]
    chosen = [len(tokenizer(row["chosen"], add_special_tokens=False).input_ids) for row in pairs]
    rejected = [
        len(tokenizer(row["rejected"], add_special_tokens=False).input_ids) for row in pairs
    ]
    chosen_total = [
        len(tokenizer(row["prompt"] + row["chosen"], add_special_tokens=False).input_ids)
        for row in pairs
    ]
    rejected_total = [
        len(tokenizer(row["prompt"] + row["rejected"], add_special_tokens=False).input_ids)
        for row in pairs
    ]
    return {
        "prompt_tokens": summarize(prompt),
        "chosen_tokens": summarize(chosen),
        "rejected_tokens": summarize(rejected),
        "prompt_chosen_tokens": summarize(chosen_total),
        "prompt_rejected_tokens": summarize(rejected_total),
    }


def audit_split(
    tokenizer: Any,
    items: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    pairs: list[dict[str, str]],
    missing: dict[str, Any],
) -> dict[str, Any]:
    category_counts = Counter(str(row["category"]) for row in rollouts)
    prompts = [str(row["prompt"]) for row in pairs]
    pair_keys = [(row["chosen"], row["rejected"]) for row in pairs]
    leakage = sum(
        "expected_result" in (row["prompt"] + row["chosen"] + row["rejected"]).casefold()
        or "oracle_sql" in (row["prompt"] + row["chosen"] + row["rejected"]).casefold()
        for row in pairs
    )
    return {
        "prompt_count": len(items),
        "rollout_count": len(rollouts),
        "category_counts": dict(category_counts),
        "strict_pair_count": len(pairs),
        "pair_family_distribution": dict(Counter(row["family"] for row in pairs)),
        "duplicate_prompt_count": len(prompts) - len(set(prompts)),
        "duplicate_pair_count": len(pair_keys) - len(set(pair_keys)),
        "chosen_incorrect_count": 0,
        "rejected_non_semantic_count": 0,
        "unsafe_or_malformed_pair_count": 0,
        "leakage_count": leakage,
        **missing,
        "tokens": token_stats(tokenizer, pairs),
    }


def main() -> None:
    from audit_grpo_hard import load_items, load_policy
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=20260820)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_items = load_items(args.train)
    dev_items = load_items(args.dev)
    model = load_policy(args.model, args.adapter)
    adapter_id = str(args.adapter)
    train_rollouts = generate_split(
        model=model,
        tokenizer=tokenizer,
        items=train_items,
        split="train",
        base_seed=args.base_seed,
        adapter_identifier=adapter_id,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(args.output_dir / "dpo_v1_rollouts_train.jsonl", train_rollouts)
    train_pairs, train_missing = build_pairs(items=train_items, rollouts=train_rollouts)
    atomic_jsonl(args.output_dir / "dpo_v1_train.jsonl", train_pairs)
    dev_rollouts = generate_split(
        model=model,
        tokenizer=tokenizer,
        items=dev_items,
        split="dev",
        base_seed=args.base_seed + 1,
        adapter_identifier=adapter_id,
    )
    dev_pairs, dev_missing = build_pairs(items=dev_items, rollouts=dev_rollouts)
    atomic_jsonl(args.output_dir / "dpo_v1_rollouts_dev.jsonl", dev_rollouts)
    atomic_jsonl(args.output_dir / "dpo_v1_dev.jsonl", dev_pairs)
    audit = {
        "policy": adapter_id,
        "base_seed": args.base_seed,
        "train": audit_split(tokenizer, train_items, train_rollouts, train_pairs, train_missing),
        "dev": audit_split(tokenizer, dev_items, dev_rollouts, dev_pairs, dev_missing),
    }
    (args.output_dir / "dpo_v1_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
