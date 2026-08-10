"""One-step GRPO-V1 smoke training from the existing trainable SFT adapter."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import bitsandbytes as bnb
import torch
from datasets import Dataset
from grpo_dataset import build_grpo_items
from grpo_rewards import SQLExecutionReward, qwen_tool_call
from peft import PeftModel, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from train_qlora_sft import (
    cuda_memory,
    inspect_quantization,
    load_quantized_base,
    lora_deltas,
    optimizer_supported,
    snapshot_lora_b,
)
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/root/autodl-tmp/models/Qwen3-8B")
DEFAULT_ADAPTER = PROJECT_ROOT / "artifacts/qlora_sft/qwen3_8b_execsql_sft_v1_3epoch"
DEFAULT_TRAIN = PROJECT_ROOT / "data/fsq/train/sft/train_v1.jsonl"
DEFAULT_DATABASE = PROJECT_ROOT / "data/fsq/shanghai_places.db"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/grpo_smoke/qwen3_8b_execsql_grpo_v1_step1"


class GradientAuditCallback(TrainerCallback):
    """Capture finite gradient evidence immediately before the optimizer update."""

    def __init__(self) -> None:
        self.seen = False
        self.all_finite = True
        self.max_abs_gradient = 0.0
        self.max_abs_gradient_history: list[float] = []

    def on_pre_optimizer_step(
        self,
        args: object,
        state: object,
        control: object,
        model: torch.nn.Module | None = None,
        **kwargs: object,
    ) -> None:
        del args, state, control, kwargs
        if model is None:
            raise ValueError("Trainer callback did not receive the policy model")
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients:
            raise ValueError("No trainable parameter has a gradient")
        self.seen = True
        self.all_finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        self.max_abs_gradient = max(
            float(gradient.detach().abs().max().float().cpu()) for gradient in gradients
        )
        self.max_abs_gradient_history.append(self.max_abs_gradient)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def offline_reward_audit(
    *,
    train_path: Path,
    database_path: Path,
    reward: SQLExecutionReward,
) -> list[dict[str, object]]:
    """Confirm reward ordering on historical correct and deterministic bad SQL."""

    trajectories = _load_jsonl(train_path)[:3]
    audit: list[dict[str, object]] = []
    for row in trajectories:
        metadata = row["metadata"]
        execution = metadata["execution_result"]
        expected = json.dumps(
            {
                "columns": execution["columns"],
                "rows": execution["rows"],
                "ordered": True,
                "numeric_tolerance": 1.0e-6,
                "strict_columns": False,
            },
            ensure_ascii=False,
        )
        completions = [
            qwen_tool_call(metadata["gold_sql"]),
            qwen_tool_call("SELECT COUNT(*) AS deliberately_wrong FROM places WHERE 1 = 0"),
        ]
        rewards = reward(
            prompts=["offline"] * 2,
            completions=completions,
            case_id=[metadata["case_id"]] * 2,
            database_path=[str(database_path.resolve())] * 2,
            expected_result_json=[expected] * 2,
        )
        if rewards[0] != 1.0 or rewards[1] > rewards[0]:
            raise ValueError(f"offline reward ordering failed for {metadata['case_id']}: {rewards}")
        audit.append(
            {
                "case_id": metadata["case_id"],
                "historical_correct_reward": rewards[0],
                "deliberately_wrong_reward": rewards[1],
            }
        )
    return audit


def load_trainable_sft_policy(
    model_path: Path, adapter_path: Path
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Load the existing SFT LoRA as the trainable policy without adding an adapter."""

    base = load_quantized_base(model_path)
    quantization = inspect_quantization(base)
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    model.config.use_cache = False
    lora_modules = [name for name, module in model.named_modules() if isinstance(module, LoraLayer)]
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    unexpected = [name for name in trainable_names if "lora_" not in name]
    trainable_params, total_params = model.get_nb_trainable_parameters()
    linear4bit_count = sum(isinstance(module, bnb.nn.Linear4bit) for module in model.modules())
    if not lora_modules:
        raise ValueError("existing SFT adapter did not attach to any module")
    if unexpected:
        raise ValueError(f"unexpected trainable base parameters: {unexpected[:10]}")
    if trainable_params <= 0:
        raise ValueError("SFT adapter is not trainable")
    details = {
        "quantization": quantization,
        "linear4bit_module_count": linear4bit_count,
        "lora_target_module_count": len(lora_modules),
        "lora_target_module_examples": lora_modules[:20],
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_percentage": 100.0 * trainable_params / total_params,
        "trainable_name_examples": trainable_names[:12],
        "unexpected_trainable": unexpected,
        "adapter_source": str(adapter_path),
    }
    return model, details


def _step_log(history: list[dict[str, Any]]) -> dict[str, Any]:
    for entry in history:
        if "loss" in entry:
            return entry
    raise ValueError("GRPO Trainer did not log a step loss")


def _step_logs(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return every optimizer-step log and reject non-finite losses."""

    logs = [dict(entry) for entry in history if "loss" in entry]
    if not logs:
        raise ValueError("GRPO Trainer did not log optimizer steps")
    if any(not math.isfinite(float(entry["loss"])) for entry in logs):
        raise ValueError("GRPO Trainer logged a non-finite loss")
    return logs


def load_pilot_items(path: Path, tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load audited RL-hard items and recheck prompt isolation and token lengths."""

    source_items = _load_jsonl(path)
    items: list[dict[str, Any]] = []
    lengths: list[int] = []
    families: dict[str, int] = {}
    for item in source_items:
        prompt = item.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("pilot item has no prompt string")
        oracle_sql = item.get("oracle_sql")
        if isinstance(oracle_sql, str) and oracle_sql in prompt:
            raise ValueError(f"oracle SQL leaked into prompt: {item.get('case_id')}")
        if "expected_result" in prompt.casefold() or "oracle_sql" in prompt.casefold():
            raise ValueError(f"reward metadata leaked into prompt: {item.get('case_id')}")
        length = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        item["prompt_token_count"] = length
        lengths.append(length)
        family = str(item.get("template_family", "unknown"))
        families[family] = families.get(family, 0) + 1
        items.append(
            {
                "prompt": prompt,
                "case_id": str(item["case_id"]),
                "database_path": str(item["database_path"]),
                "expected_result_json": str(item["expected_result_json"]),
                "template_family": family,
                "prompt_token_count": length,
            }
        )
    if not items:
        raise ValueError("pilot dataset is empty")
    return items, {
        "item_count": len(items),
        "family_counts": dict(sorted(families.items())),
        "prompt_token_min": min(lengths),
        "prompt_token_mean": statistics.fmean(lengths),
        "prompt_token_max": max(lengths),
        "leakage_count": 0,
    }


def stratified_pilot_items(items: list[dict[str, Any]], step_count: int) -> list[dict[str, Any]]:
    """Select unique prompts in family round-robin order for a short pilot."""

    by_family: dict[str, list[dict[str, Any]]] = {}
    for item in sorted(items, key=lambda value: str(value["case_id"])):
        by_family.setdefault(str(item["template_family"]), []).append(item)
    families = sorted(by_family)
    if step_count < len(families):
        raise ValueError("stratified pilot needs at least one step per family")
    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < step_count:
        added = False
        for family in families:
            family_items = by_family[family]
            if round_index < len(family_items):
                selected.append(family_items[round_index])
                added = True
                if len(selected) == step_count:
                    break
        if not added:
            raise ValueError("not enough unique prompts for stratified pilot")
        round_index += 1
    return selected


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


def rollout_diversity_audit(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    items: list[dict[str, Any]],
    group_count: int,
) -> dict[str, Any]:
    """Sample four completions for several prompts without an optimizer update."""

    if group_count < 1:
        raise ValueError("group_count must be positive")
    selected: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    for item in items:
        family = str(item["template_family"])
        if family not in seen_families:
            selected.append(item)
            seen_families.add(family)
        if len(selected) == group_count:
            break
    if len(selected) < group_count:
        selected.extend(items[: group_count - len(selected)])

    model.eval()
    torch.manual_seed(314159)
    reward = SQLExecutionReward(max_rows=100)
    groups: list[dict[str, Any]] = []
    for item in selected:
        encoded = tokenizer(
            item["prompt"],
            return_tensors="pt",
            add_special_tokens=False,
        ).to(next(model.parameters()).device)
        prompt_length = int(encoded["input_ids"].shape[1])
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                top_k=20,
                num_return_sequences=4,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completions = tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)
        start = len(reward.history)
        rewards = reward(
            prompts=[item["prompt"]] * 4,
            completions=completions,
            case_id=[item["case_id"]] * 4,
            database_path=[item["database_path"]] * 4,
            expected_result_json=[item["expected_result_json"]] * 4,
        )
        group_breakdowns = reward.history[start:]
        reward_std = statistics.pstdev(rewards)
        groups.append(
            {
                "case_id": item["case_id"],
                "template_family": item["template_family"],
                "rewards": rewards,
                "reward_std": reward_std,
                "zero_std": reward_std == 0.0,
                "categories": [entry.category for entry in group_breakdowns],
                "executed_count": sum(entry.executed for entry in group_breakdowns),
            }
        )
        del encoded, outputs
        torch.cuda.empty_cache()
    zero_std_count = sum(bool(group["zero_std"]) for group in groups)
    return {
        "group_count": len(groups),
        "zero_std_group_count": zero_std_count,
        "zero_std_group_rate": zero_std_count / len(groups),
        "groups": groups,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--grpo-data", type=Path)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--diversity-groups", type=int, default=0)
    parser.add_argument("--diversity-only", action="store_true")
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--family-stratified", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.max_steps <= 30:
        raise ValueError("GRPO-V1 pilot supports between 1 and 30 optimizer steps")
    if args.beta != 0.0:
        raise ValueError("GRPO-V1 requires beta=0 until a frozen SFT reference exists")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.grpo_data is not None:
        items, data_summary = load_pilot_items(args.grpo_data, tokenizer)
        max_prompt_length = int(data_summary["prompt_token_max"])
        training_items = (
            stratified_pilot_items(items, args.max_steps) if args.family_stratified else items
        )
        if args.family_stratified:
            print(
                json.dumps(
                    {
                        "stage": "stratified_selection",
                        "item_count": len(training_items),
                        "family_counts": dict(
                            Counter(str(item["template_family"]) for item in training_items)
                        ),
                        "case_ids": [item["case_id"] for item in training_items],
                    }
                )
            )
        print(json.dumps({"stage": "pilot_dataset_audit", **data_summary}))
    else:
        items, dataset_stats = build_grpo_items(args.train, tokenizer, database_path=args.database)
        if dataset_stats.trajectory_count != 180 or dataset_stats.prompt_count != 180:
            raise ValueError(f"unexpected GRPO dataset size: {asdict(dataset_stats)}")
        if dataset_stats.leakage_count or dataset_stats.missing_sql_decision_turns:
            raise ValueError(f"GRPO dataset audit failed: {asdict(dataset_stats)}")
        print(json.dumps({"stage": "dataset_audit", **asdict(dataset_stats)}))
        audit_reward = SQLExecutionReward(max_rows=100)
        offline_audit = offline_reward_audit(
            train_path=args.train, database_path=args.database, reward=audit_reward
        )
        print(json.dumps({"stage": "offline_reward_audit", "cases": offline_audit}))
        max_prompt_length = dataset_stats.prompt_token_max
        smoke_item = max(items, key=lambda item: int(item["prompt_token_count"]))
        training_items = [smoke_item]
    if args.audit_only:
        return 0

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    optimizer_supported()

    smoke_item = max(items, key=lambda item: int(item["prompt_token_count"]))
    print(
        json.dumps(
            {
                "stage": "smoke_prompt",
                "case_id": smoke_item["case_id"],
                "template_family": smoke_item["template_family"],
                "prompt_tokens": smoke_item["prompt_token_count"],
                "max_prompt_length": max_prompt_length,
            }
        )
    )

    torch.cuda.empty_cache()
    model, policy_details = load_trainable_sft_policy(args.model, args.adapter)
    memory_after_model_load = cuda_memory()
    print(
        json.dumps(
            {
                "stage": "policy_ready",
                "memory": memory_after_model_load,
                "policy": policy_details,
            }
        )
    )

    if args.diversity_groups:
        diversity = rollout_diversity_audit(
            model=model,
            tokenizer=tokenizer,
            items=items,
            group_count=args.diversity_groups,
        )
        print(json.dumps({"stage": "rollout_diversity", **diversity}))
    if args.diversity_only:
        if not args.diversity_groups:
            raise ValueError("--diversity-only requires --diversity-groups")
        return 0

    before = snapshot_lora_b(model)

    reward = SQLExecutionReward(max_rows=100, num_generations=4)
    gradient_audit = GradientAuditCallback()
    training_args = GRPOConfig(
        output_dir=str(args.output.with_name(args.output.name + "_trainer_state")),
        max_steps=args.max_steps,
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
        seed=42,
        data_seed=42,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        num_generations=4,
        max_prompt_length=max_prompt_length,
        max_completion_length=256,
        use_vllm=False,
        beta=args.beta,
        temperature=args.temperature,
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
    result = trainer.train()
    runtime = perf_counter() - started
    if trainer.state.global_step != args.max_steps:
        raise ValueError(f"expected global_step={args.max_steps}, got {trainer.state.global_step}")
    logs = _step_logs(trainer.state.log_history)
    log = logs[-1]
    loss = float(log["loss"])
    if not gradient_audit.seen or not gradient_audit.all_finite:
        raise ValueError("GRPO gradients were absent or non-finite")
    deltas = lora_deltas(model, before)
    rollout_breakdowns = reward.history
    if len(rollout_breakdowns) != args.max_steps * 4:
        raise ValueError(
            f"expected {args.max_steps * 4} rollout rewards, got {len(rollout_breakdowns)}"
        )
    if not any(item.executed for item in rollout_breakdowns):
        raise ValueError("no generated rollout reached real SQLite execution")
    if len(reward.group_history) != args.max_steps:
        raise ValueError("reward group metrics do not match optimizer steps")
    zero_std_fraction = sum(group.zero_std for group in reward.group_history) / len(
        reward.group_history
    )
    if zero_std_fraction == 1.0:
        raise ValueError("all pilot reward groups have zero std; refusing to save adapter")
    peak_allocated = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)
    saved = _save_adapter(model, tokenizer, args.output)
    step_curve = []
    family_by_case = {str(item["case_id"]): str(item["template_family"]) for item in training_items}
    for index, (step_log, group) in enumerate(
        zip(logs, reward.group_history, strict=True), start=1
    ):
        group_breakdowns = rollout_breakdowns[(index - 1) * 4 : index * 4]
        step_curve.append(
            {
                "step": index,
                "loss": float(step_log["loss"]),
                "grad_norm": step_log.get("grad_norm"),
                "family": family_by_case[group.case_id],
                "rewards": [item.reward for item in group_breakdowns],
                "correct_count": sum(item.category == "correct" for item in group_breakdowns),
                "semantic_mismatch_count": sum(
                    item.category == "semantic_mismatch" for item in group_breakdowns
                ),
                "execution_failure_count": sum(
                    item.category == "execution_failure" for item in group_breakdowns
                ),
                "malformed_count": sum(
                    item.category == "malformed_tool_call" for item in group_breakdowns
                ),
                "unsafe_count": sum(item.category == "unsafe_sql" for item in group_breakdowns),
                **asdict(group),
            }
        )
    observed_cases = [group.case_id for group in reward.group_history]
    observed_family_counts = Counter(family_by_case[case_id] for case_id in observed_cases)
    if args.family_stratified and len(set(observed_cases)) != args.max_steps:
        raise ValueError("stratified pilot repeated prompts before covering its selection")
    summary = {
        "stage": "complete",
        "global_step": trainer.state.global_step,
        "optimizer": "paged_adamw_8bit",
        "beta": args.beta,
        "temperature": args.temperature,
        "loss": loss,
        "grad_norm": log.get("grad_norm"),
        "gradient_max_abs": gradient_audit.max_abs_gradient,
        "gradient_max_abs_history": gradient_audit.max_abs_gradient_history,
        "runtime_seconds": runtime,
        "trainer_runtime_seconds": result.metrics.get("train_runtime"),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "lora_max_abs_delta": max(deltas.values()),
        "lora_deltas": deltas,
        "rollout_count": len(rollout_breakdowns),
        "zero_std_group_fraction": zero_std_fraction,
        "correct_rollout_rate": sum(item.category == "correct" for item in rollout_breakdowns)
        / len(rollout_breakdowns),
        "semantic_mismatch_rate": sum(
            item.category == "semantic_mismatch" for item in rollout_breakdowns
        )
        / len(rollout_breakdowns),
        "malformed_rate": sum(item.category == "malformed_tool_call" for item in rollout_breakdowns)
        / len(rollout_breakdowns),
        "execution_failure_rate": sum(
            item.category == "execution_failure" for item in rollout_breakdowns
        )
        / len(rollout_breakdowns),
        "unsafe_rate": sum(item.category == "unsafe_sql" for item in rollout_breakdowns)
        / len(rollout_breakdowns),
        "step_curve": step_curve,
        "observed_unique_case_count": len(set(observed_cases)),
        "observed_family_counts": dict(observed_family_counts),
        "saved_adapter": saved,
    }
    (args.output / "pilot_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
