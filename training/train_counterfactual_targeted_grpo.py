"""Run targeted counterfactual GRPO smoke or 32-step pilot from the SFT adapter."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from datasets import Dataset
from grpo_test_suite_rewards import CounterfactualExecutionReward
from train_grpo import GradientAuditCallback, load_trainable_sft_policy
from train_qlora_sft import (
    cuda_memory,
    lora_deltas,
    optimizer_supported,
    snapshot_lora_b,
)
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

TARGET_FAMILIES = ("aggregation_scope", "ties")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_items(path: Path, tokenizer: Any) -> list[dict[str, Any]]:
    source = _read_jsonl(path)
    items: list[dict[str, Any]] = []
    for item in source:
        if item["split"] != "train" or item["template_family"] not in TARGET_FAMILIES:
            continue
        prompt = str(item["prompt"])
        if any(marker in prompt for marker in ("expected_result", "oracle_sql", "database_path")):
            raise ValueError(f"reward metadata leaked into prompt: {item['case_id']}")
        items.append(
            {
                "prompt": prompt,
                "case_id": str(item["case_id"]),
                "worlds_json": str(item["worlds_json"]),
                "template_family": str(item["template_family"]),
                "prompt_token_count": len(
                    tokenizer(prompt, add_special_tokens=False).input_ids
                ),
            }
        )
    counts = Counter(item["template_family"] for item in items)
    if len(items) != 32 or counts != {"aggregation_scope": 16, "ties": 16}:
        raise ValueError(f"expected balanced 32-item target train split, got {counts}")
    return items


def _pilot_order(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family = {
        family: sorted(
            [item for item in items if item["template_family"] == family],
            key=lambda item: item["case_id"],
        )
        for family in TARGET_FAMILIES
    }
    ordered = [
        item
        for index in range(16)
        for item in (by_family["aggregation_scope"][index], by_family["ties"][index])
    ]
    if len({item["case_id"] for item in ordered}) != 32:
        raise ValueError("targeted pilot ordering repeated a semantic task")
    return ordered


def _smoke_item(
    items: list[dict[str, Any]], audit_path: Path
) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    eligible: list[str] = []
    item_ids = {item["case_id"] for item in items}
    for group in audit["groups"]:
        if group["case_id"] not in item_ids:
            continue
        reward_std = statistics.pstdev(group["suite_rewards"])
        if reward_std > 0.0:
            eligible.append(str(group["case_id"]))
    if not eligible:
        raise ValueError("V1.1 audit contains no nonzero-variance target train task")
    selected_id = eligible[0]
    return next(item for item in items if item["case_id"] == selected_id)


def _step_logs(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    logs = [dict(entry) for entry in history if "loss" in entry]
    if not logs or any(not math.isfinite(float(entry["loss"])) for entry in logs):
        raise ValueError("GRPO loss missing or non-finite")
    return logs


def _save_adapter(model: torch.nn.Module, tokenizer: Any, output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    files = sorted(path for path in output.rglob("*") if path.is_file())
    return {
        "path": str(output),
        "files": [str(path.relative_to(output)) for path in files],
        "total_bytes": sum(path.stat().st_size for path in files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    max_steps = 1 if args.mode == "smoke" else 32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    items = _load_items(args.tasks, tokenizer)
    training_items = (
        [_smoke_item(items, args.audit)] if args.mode == "smoke" else _pilot_order(items)
    )
    max_prompt_length = max(int(item["prompt_token_count"]) for item in items)
    print(
        json.dumps(
            {
                "stage": "dataset_ready",
                "mode": args.mode,
                "item_count": len(training_items),
                "family_counts": dict(
                    Counter(item["template_family"] for item in training_items)
                ),
                "case_ids": [item["case_id"] for item in training_items],
                "max_prompt_length": max_prompt_length,
            }
        ),
        flush=True,
    )

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    optimizer_supported()
    torch.cuda.empty_cache()
    model, policy_details = load_trainable_sft_policy(args.model, args.adapter)
    memory_after_load = cuda_memory()
    before = snapshot_lora_b(model)
    reward = CounterfactualExecutionReward(num_generations=4)
    gradient_audit = GradientAuditCallback()
    training_seed = 20260830 if args.mode == "smoke" else 42
    training_args = GRPOConfig(
        output_dir=str(args.output.with_name(args.output.name + "_trainer_state")),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1.0e-6,
        lr_scheduler_type="constant",
        warmup_steps=0,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=training_seed,
        data_seed=training_seed,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        num_generations=4,
        max_prompt_length=max_prompt_length,
        max_completion_length=256,
        use_vllm=False,
        beta=0.0,
        temperature=1.0,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=training_args,
        train_dataset=Dataset.from_list(training_items),
        processing_class=tokenizer,
        callbacks=[gradient_audit],
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    train_result = trainer.train()
    runtime = perf_counter() - started
    if trainer.state.global_step != max_steps:
        raise ValueError(f"expected {max_steps} steps, got {trainer.state.global_step}")
    logs = _step_logs(trainer.state.log_history)
    if len(logs) != max_steps:
        raise ValueError(f"expected {max_steps} step logs, got {len(logs)}")
    if not gradient_audit.seen or not gradient_audit.all_finite:
        raise ValueError("GRPO gradients missing or non-finite")
    if gradient_audit.max_abs_gradient <= 0.0:
        raise ValueError("GRPO gradient is zero")
    deltas = lora_deltas(model, before)
    max_delta = max(deltas.values())
    if max_delta <= 0.0:
        raise ValueError("trainable SFT LoRA did not update")
    if len(reward.history) != max_steps * 4:
        raise ValueError(f"expected {max_steps * 4} rewards, got {len(reward.history)}")

    family_by_case = {
        str(item["case_id"]): str(item["template_family"]) for item in training_items
    }
    step_curve: list[dict[str, Any]] = []
    for index, log in enumerate(logs):
        group = reward.history[index * 4 : (index + 1) * 4]
        rewards = [entry.reward for entry in group]
        case_ids = {entry.case_id for entry in group}
        if len(case_ids) != 1:
            raise ValueError("one optimizer step contains multiple case IDs")
        case_id = next(iter(case_ids))
        step_curve.append(
            {
                "step": index + 1,
                "case_id": case_id,
                "family": family_by_case[case_id],
                "rewards": rewards,
                "reward_mean": statistics.fmean(rewards),
                "reward_std": statistics.pstdev(rewards),
                "exact_1_count": sum(value == 1.0 for value in rewards),
                "partial_count": sum(0.0 < value < 1.0 for value in rewards),
                "zero_std": statistics.pstdev(rewards) == 0.0,
                "grad_norm": log.get("grad_norm"),
                "loss": float(log["loss"]),
                "malformed_count": sum(
                    entry.category == "malformed_tool_call" for entry in group
                ),
                "unsafe_count": sum(entry.category == "unsafe_sql" for entry in group),
            }
        )
    if args.mode == "smoke" and step_curve[0]["reward_std"] <= 0.0:
        raise ValueError("smoke generation produced a zero-variance reward group")
    observed_cases = [entry["case_id"] for entry in step_curve]
    if args.mode == "pilot":
        if len(set(observed_cases)) != 32:
            raise ValueError("pilot repeated prompts before covering all 32 tasks")
        observed_families = Counter(entry["family"] for entry in step_curve)
        if observed_families != {"aggregation_scope": 16, "ties": 16}:
            raise ValueError(f"pilot family imbalance: {observed_families}")

    all_rewards = [entry.reward for entry in reward.history]
    saved = _save_adapter(model, tokenizer, args.output)
    summary = {
        "stage": "complete",
        "mode": args.mode,
        "source_adapter": str(args.adapter),
        "global_step": trainer.state.global_step,
        "optimizer": "paged_adamw_8bit",
        "beta": 0.0,
        "temperature": 1.0,
        "policy": policy_details,
        "memory_after_model_load": memory_after_load,
        "loss": float(logs[-1]["loss"]),
        "grad_norm": logs[-1].get("grad_norm"),
        "gradient_max_abs": gradient_audit.max_abs_gradient,
        "lora_max_abs_delta": max_delta,
        "runtime_seconds": runtime,
        "trainer_runtime_seconds": train_result.metrics.get("train_runtime"),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "rollout_count": len(all_rewards),
        "mean_semantic_reward": statistics.fmean(all_rewards),
        "exact_pass_rate": sum(value == 1.0 for value in all_rewards) / len(all_rewards),
        "partial_reward_distribution": dict(
            Counter(str(value) for value in all_rewards if 0.0 < value < 1.0)
        ),
        "zero_std_step_ratio": sum(entry["zero_std"] for entry in step_curve)
        / len(step_curve),
        "malformed_count": sum(
            entry.category == "malformed_tool_call" for entry in reward.history
        ),
        "unsafe_count": sum(entry.category == "unsafe_sql" for entry in reward.history),
        "step_curve": step_curve,
        "saved_adapter": saved,
    }
    (args.output / "targeted_grpo_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
