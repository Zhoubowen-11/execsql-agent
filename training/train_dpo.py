"""Train a short DPO pilot from identical train/reference SFT adapters."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import bitsandbytes as bnb
import torch
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from train_qlora_sft import (
    cuda_memory,
    inspect_quantization,
    load_quantized_base,
    optimizer_supported,
)
from transformers import AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/root/autodl-tmp/models/Qwen3-8B")
DEFAULT_ADAPTER = PROJECT_ROOT / "artifacts/qlora_sft/qwen3_8b_execsql_sft_v1_3epoch"
DEFAULT_TRAIN = PROJECT_ROOT / "data/fsq/train/dpo/dpo_v1_train.jsonl"
POLICY_ADAPTER = "policy"
REFERENCE_ADAPTER = "reference"


class GradientAuditCallback(TrainerCallback):
    """Retain finite-gradient evidence immediately before optimizer updates."""

    def __init__(self) -> None:
        self.seen = False
        self.all_finite = True
        self.max_abs_history: list[float] = []

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
            raise ValueError("DPO callback did not receive the policy model")
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if f".{POLICY_ADAPTER}." in name
            and parameter.requires_grad
            and parameter.grad is not None
        ]
        if not gradients:
            raise ValueError("no train-adapter parameter has a gradient")
        finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        self.seen = True
        self.all_finite = self.all_finite and finite
        self.max_abs_history.append(
            max(float(gradient.detach().abs().max().float().cpu()) for gradient in gradients)
        )


def load_jsonl(path: Path) -> list[dict[str, str]]:
    """Read DPO pairs and enforce the minimal public preference schema."""

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            required = {"case_id", "family", "prompt", "chosen", "rejected"}
            if set(raw) != required:
                raise ValueError(f"{path}:{line_number}: unexpected DPO fields: {set(raw)}")
            row = {key: str(raw[key]) for key in required}
            if row["chosen"] == row["rejected"]:
                raise ValueError(f"{row['case_id']}: chosen and rejected are identical")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty DPO dataset: {path}")
    return rows


def token_audit(tokenizer: Any, rows: list[dict[str, str]]) -> dict[str, Any]:
    """Measure exact TRL tokenization lengths and reject any planned truncation."""

    prompt_lengths: list[int] = []
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    chosen_totals: list[int] = []
    rejected_totals: list[int] = []
    for row in rows:
        tokenized = DPOTrainer.tokenize_row(row, tokenizer, None, None, False)
        prompt = len(tokenized["prompt_input_ids"])
        chosen = len(tokenized["chosen_input_ids"])
        rejected = len(tokenized["rejected_input_ids"])
        if not prompt or not chosen or not rejected:
            raise ValueError(f"{row['case_id']}: zero-length prompt or completion")
        prompt_lengths.append(prompt)
        chosen_lengths.append(chosen)
        rejected_lengths.append(rejected)
        chosen_totals.append(prompt + chosen)
        rejected_totals.append(prompt + rejected)

    def stats(values: list[int]) -> dict[str, float | int]:
        return {"min": min(values), "mean": statistics.fmean(values), "max": max(values)}

    max_prompt_length = max(prompt_lengths)
    max_completion_length = max(max(chosen_lengths), max(rejected_lengths))
    max_length = max(max(chosen_totals), max(rejected_totals))
    return {
        "pair_count": len(rows),
        "prompt_tokens": stats(prompt_lengths),
        "chosen_tokens": stats(chosen_lengths),
        "rejected_tokens": stats(rejected_lengths),
        "prompt_chosen_tokens": stats(chosen_totals),
        "prompt_rejected_tokens": stats(rejected_totals),
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
        "max_length": max_length,
        "truncation_count": 0,
    }


def stratified_pairs(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    """Choose unique pairs in family round-robin order."""

    by_family: dict[str, list[dict[str, str]]] = {}
    for row in sorted(rows, key=lambda value: value["case_id"]):
        by_family.setdefault(row["family"], []).append(row)
    if count < len(by_family):
        return [by_family[family][0] for family in sorted(by_family)[:count]]
    selected: list[dict[str, str]] = []
    offset = 0
    while len(selected) < count:
        added = False
        for family in sorted(by_family):
            if offset < len(by_family[family]):
                selected.append(by_family[family][offset])
                added = True
                if len(selected) == count:
                    break
        if not added:
            raise ValueError(f"cannot select {count} unique family-stratified pairs")
        offset += 1
    return selected


def _adapter_parameters(model: torch.nn.Module, adapter: str) -> dict[str, torch.nn.Parameter]:
    marker = f".{adapter}."
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if "lora_" in name and marker in name
    }


def _mapped_reference_name(train_name: str) -> str:
    return train_name.replace(f".{POLICY_ADAPTER}.", f".{REFERENCE_ADAPTER}.")


def _snapshot(parameters: dict[str, torch.nn.Parameter]) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()}


def _max_deltas(
    parameters: dict[str, torch.nn.Parameter], before: dict[str, torch.Tensor]
) -> dict[str, float]:
    return {
        name: float((parameter.detach().cpu() - before[name]).abs().max().float())
        for name, parameter in parameters.items()
    }


def load_two_adapter_policy(
    model_path: Path, adapter_path: Path
) -> tuple[PeftModel, dict[str, Any]]:
    """Load trainable and frozen copies of the exact same SFT adapter."""

    base = load_quantized_base(model_path)
    quantization = inspect_quantization(base)
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = PeftModel.from_pretrained(
        base, adapter_path, adapter_name=POLICY_ADAPTER, is_trainable=True
    )
    model.load_adapter(
        adapter_path, adapter_name=REFERENCE_ADAPTER, is_trainable=False
    )
    model.set_adapter(POLICY_ADAPTER)
    model.config.use_cache = False

    train = _adapter_parameters(model, POLICY_ADAPTER)
    reference = _adapter_parameters(model, REFERENCE_ADAPTER)
    if not train or len(train) != len(reference):
        raise ValueError("train/reference adapter parameter sets are incomplete")
    equality_deltas: dict[str, float] = {}
    for train_name, parameter in train.items():
        ref_name = _mapped_reference_name(train_name)
        if ref_name not in reference:
            raise ValueError(f"missing reference tensor for {train_name}")
        equality_deltas[train_name] = float(
            (parameter.detach().cpu() - reference[ref_name].detach().cpu()).abs().max().float()
        )
    if max(equality_deltas.values()) != 0.0:
        raise ValueError("train and reference adapters do not start bit-identical")

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    unexpected = [
        name
        for name in trainable_names
        if "lora_" not in name or f".{POLICY_ADAPTER}." not in name
    ]
    reference_trainable = [
        name for name in trainable_names if f".{REFERENCE_ADAPTER}." in name
    ]
    if unexpected or reference_trainable:
        raise ValueError(
            "unexpected trainability: "
            f"base/train={unexpected[:5]}, reference={reference_trainable[:5]}"
        )
    trainable_params = sum(
        parameter.numel() for parameter in train.values() if parameter.requires_grad
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    linear4bit_count = sum(isinstance(module, bnb.nn.Linear4bit) for module in model.modules())
    lora_modules = [name for name, module in model.named_modules() if isinstance(module, LoraLayer)]
    if not bool(getattr(model, "is_loaded_in_4bit", False)) or linear4bit_count == 0:
        raise ValueError("DPO policy is not a 4-bit model")
    return model, {
        "quantization": quantization,
        "linear4bit_module_count": linear4bit_count,
        "lora_target_module_count": len(lora_modules),
        "lora_target_module_examples": lora_modules[:20],
        "train_adapter_parameter_tensors": len(train),
        "reference_adapter_parameter_tensors": len(reference),
        "initial_adapter_max_abs_difference": max(equality_deltas.values()),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_percentage": 100.0 * trainable_params / total_params,
        "trainable_name_examples": trainable_names[:12],
        "base_4bit_frozen": all(
            not parameter.requires_grad
            for name, parameter in model.named_parameters()
            if f".{POLICY_ADAPTER}." not in name
        ),
        "reference_frozen": not reference_trainable,
    }


def save_train_adapter(model: PeftModel, tokenizer: Any, output: Path) -> dict[str, Any]:
    """Persist only the trained adapter and tokenizer, never the Base/reference."""

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model.set_adapter(POLICY_ADAPTER)
    model.save_pretrained(
        output, selected_adapters=[POLICY_ADAPTER], safe_serialization=True
    )
    tokenizer.save_pretrained(output)
    files = sorted(path for path in output.rglob("*") if path.is_file())
    adapter_dir = output / POLICY_ADAPTER
    if not (adapter_dir / "adapter_config.json").is_file():
        adapter_dir = output
    return {
        "path": str(output),
        "adapter_load_path": str(adapter_dir),
        "files": [str(path.relative_to(output)) for path in files],
        "total_bytes": sum(path.stat().st_size for path in files),
    }


def _step_logs(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    logs = [dict(entry) for entry in history if "loss" in entry]
    if not logs or any(not math.isfinite(float(entry["loss"])) for entry in logs):
        raise ValueError("DPO loss was missing or non-finite")
    return logs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, choices=(1, 30), required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    optimizer_supported()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = load_jsonl(args.train)
    dataset_audit = token_audit(tokenizer, rows)
    selection_count = 4 if args.max_steps == 1 else min(30, len(rows))
    selected = stratified_pairs(rows, selection_count)
    selection = {
        "count": len(selected),
        "family_counts": dict(Counter(row["family"] for row in selected)),
        "case_ids": [row["case_id"] for row in selected],
    }
    print(json.dumps({"stage": "dataset_audit", **dataset_audit}, ensure_ascii=False))
    print(json.dumps({"stage": "stratified_selection", **selection}, ensure_ascii=False))

    torch.cuda.empty_cache()
    model, adapter_audit = load_two_adapter_policy(args.model, args.adapter)
    memory_after_load = cuda_memory()
    print(
        json.dumps(
            {"stage": "two_adapter_policy", "memory": memory_after_load, **adapter_audit},
            ensure_ascii=False,
        )
    )
    train_parameters = _adapter_parameters(model, POLICY_ADAPTER)
    reference_parameters = _adapter_parameters(model, REFERENCE_ADAPTER)
    train_before = _snapshot(train_parameters)
    reference_before = _snapshot(reference_parameters)

    callback = GradientAuditCallback()
    training_args = DPOConfig(
        output_dir=str(args.output.with_name(args.output.name + "_trainer_state")),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
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
        loss_type="sigmoid",
        beta=0.1,
        disable_dropout=True,
        reference_free=False,
        label_smoothing=0.0,
        model_adapter_name=POLICY_ADAPTER,
        ref_adapter_name=REFERENCE_ADAPTER,
        max_prompt_length=int(dataset_audit["max_prompt_length"]),
        max_completion_length=int(dataset_audit["max_completion_length"]),
        max_length=int(dataset_audit["max_length"]),
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=1,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=Dataset.from_list(selected),
        processing_class=tokenizer,
        callbacks=[callback],
    )

    tokenized = trainer.train_dataset[0]
    prompt_tokens = len(tokenized["prompt_input_ids"])
    chosen_tokens = len(tokenized["chosen_input_ids"])
    rejected_tokens = len(tokenized["rejected_input_ids"])
    completion_mask_audit = {
        "case_id": selected[0]["case_id"],
        "prompt_tokens_masked": prompt_tokens,
        "chosen_completion_tokens_supervised": chosen_tokens,
        "rejected_completion_tokens_supervised": rejected_tokens,
        "prompt_supervised_tokens": 0,
        "no_truncation": (
            prompt_tokens + max(chosen_tokens, rejected_tokens)
            <= int(dataset_audit["max_length"])
        ),
    }
    if (
        prompt_tokens <= 0
        or chosen_tokens <= 0
        or rejected_tokens <= 0
        or not completion_mask_audit["no_truncation"]
    ):
        raise ValueError(f"completion mask audit failed: {completion_mask_audit}")
    print(json.dumps({"stage": "completion_mask_audit", **completion_mask_audit}))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    result = trainer.train()
    runtime = perf_counter() - started
    if trainer.state.global_step != args.max_steps:
        raise ValueError(
            f"expected global_step={args.max_steps}, got {trainer.state.global_step}"
        )
    logs = _step_logs(trainer.state.log_history)
    if not callback.seen or not callback.all_finite:
        raise ValueError("DPO gradients were absent or non-finite")
    train_deltas = _max_deltas(train_parameters, train_before)
    reference_deltas = _max_deltas(reference_parameters, reference_before)
    if max(train_deltas.values()) <= 0.0:
        raise ValueError("train SFT adapter received no optimizer update")
    if max(reference_deltas.values()) != 0.0:
        raise ValueError("frozen SFT reference adapter changed")
    if model.active_adapter not in (POLICY_ADAPTER, [POLICY_ADAPTER]):
        raise ValueError(f"unexpected active adapter after training: {model.active_adapter}")

    required_metrics = (
        "rewards/chosen",
        "rewards/rejected",
        "rewards/accuracies",
        "rewards/margins",
        "logps/chosen",
        "logps/rejected",
    )
    missing_metrics = [key for key in required_metrics if key not in logs[-1]]
    if missing_metrics:
        raise ValueError(f"TRL DPO metrics missing: {missing_metrics}")
    for entry in logs:
        numeric = [float(entry[key]) for key in required_metrics if key in entry]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("DPO reward/logprob metric is non-finite")

    peak_allocated = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)
    saved = save_train_adapter(model, tokenizer, args.output)
    summary = {
        "stage": "complete",
        "source_adapter": str(args.adapter),
        "global_step": trainer.state.global_step,
        "optimizer": "paged_adamw_8bit",
        "loss_type": "sigmoid",
        "beta": 0.1,
        "learning_rate": 1.0e-6,
        "dataset_audit": dataset_audit,
        "selection": selection,
        "completion_mask_audit": completion_mask_audit,
        "adapter_audit": adapter_audit,
        "step_logs": logs,
        "gradient_max_abs_history": callback.max_abs_history,
        "train_adapter_max_abs_delta": max(train_deltas.values()),
        "reference_adapter_max_abs_delta": max(reference_deltas.values()),
        "runtime_seconds": runtime,
        "trainer_runtime_seconds": result.metrics.get("train_runtime"),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "saved_adapter": saved,
    }
    (args.output / "dpo_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
