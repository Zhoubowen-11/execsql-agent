"""Full QLoRA SFT for ExecSQL-Agent.

Reuses the already smoke-tested QLoRA/model/collator helpers and the
validated assistant-only preprocessing pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import torch
from assistant_turn_preprocessing import (
    ProcessedTurn,
    load_jsonl,
    process_trajectory,
)
from torch.utils.data import Dataset
from train_qlora_sft import (
    AssistantOnlyCollator,
    attach_lora,
    cuda_memory,
    inspect_quantization,
    load_quantized_base,
    optimizer_supported,
)
from transformers import AutoTokenizer, Trainer, TrainingArguments

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = Path("/root/autodl-tmp/models/Qwen3-8B")
DEFAULT_TRAIN = PROJECT_ROOT / "data/fsq/train/sft/train_v1.jsonl"
DEFAULT_DEV = PROJECT_ROOT / "data/fsq/train/sft/dev_v1.jsonl"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts/qlora_sft/qwen3_8b_execsql_sft_v1_3epoch"
)

MAX_LENGTH = 4096


class AssistantTurnDataset(Dataset[dict[str, list[int]]]):
    """Dataset of already-tokenized assistant-only training turns."""

    def __init__(self, turns: list[ProcessedTurn]) -> None:
        self.turns = turns

    def __len__(self) -> int:
        return len(self.turns)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        turn = self.turns[index]
        return {
            "input_ids": turn.input_ids,
            "attention_mask": turn.attention_mask,
            "labels": turn.labels,
        }


def load_turns(
    *,
    tokenizer,
    path: Path,
    split: str,
    max_length: int,
) -> tuple[list[ProcessedTurn], int]:
    """Convert every trajectory into assistant-turn training samples."""

    trajectories = load_jsonl(path)
    turns: list[ProcessedTurn] = []

    for sample in trajectories:
        turns.extend(
            process_trajectory(
                tokenizer=tokenizer,
                sample=sample,
                split=split,
                max_length=max_length,
            )
        )

    if not turns:
        raise ValueError(f"{split} produced zero assistant-turn samples")

    return turns, len(trajectories)


def validate_turns(
    *,
    turns: list[ProcessedTurn],
    split: str,
) -> dict[str, object]:
    """Hard-stop validation before touching the 8B model."""

    zero_supervised = [
        turn.case_id for turn in turns if turn.supervised_tokens == 0
    ]
    context_supervised = [
        turn.case_id
        for turn in turns
        if turn.context_tokens_supervised != 0
    ]
    missing_turn_end = [
        turn.case_id
        for turn in turns
        if not turn.target_has_turn_end
    ]
    too_long = [
        turn.case_id
        for turn in turns
        if turn.sequence_tokens > MAX_LENGTH
    ]

    if zero_supervised:
        raise ValueError(
            f"{split}: zero-supervised samples: {zero_supervised[:5]}"
        )
    if context_supervised:
        raise ValueError(
            f"{split}: context tokens unexpectedly supervised: "
            f"{context_supervised[:5]}"
        )
    if missing_turn_end:
        raise ValueError(
            f"{split}: assistant targets missing turn-end token: "
            f"{missing_turn_end[:5]}"
        )
    if too_long:
        raise ValueError(
            f"{split}: sequence exceeds {MAX_LENGTH}: {too_long[:5]}"
        )

    lengths = [turn.sequence_tokens for turn in turns]
    supervised = [turn.supervised_tokens for turn in turns]

    return {
        "samples": len(turns),
        "sequence_min": min(lengths),
        "sequence_mean": sum(lengths) / len(lengths),
        "sequence_max": max(lengths),
        "supervised_min": min(supervised),
        "supervised_mean": sum(supervised) / len(supervised),
        "supervised_max": max(supervised),
        "tool_call_targets": sum(
            turn.target_type == "tool_call" for turn in turns
        ),
        "final_answer_targets": sum(
            turn.target_type == "final_answer" for turn in turns
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full ExecSQL-Agent Qwen3-8B QLoRA SFT."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support BF16")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError(
            "--gradient-accumulation-steps must be positive"
        )

    optimizer_supported()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither pad_token_id nor eos_token_id"
            )
        tokenizer.pad_token = tokenizer.eos_token

    train_turns, train_trajectory_count = load_turns(
        tokenizer=tokenizer,
        path=args.train,
        split="train",
        max_length=MAX_LENGTH,
    )
    dev_turns, dev_trajectory_count = load_turns(
        tokenizer=tokenizer,
        path=args.dev,
        split="dev",
        max_length=MAX_LENGTH,
    )

    train_stats = validate_turns(turns=train_turns, split="train")
    dev_stats = validate_turns(turns=dev_turns, split="dev")

    # v1 dataset contract.
    if train_trajectory_count != 180 or len(train_turns) != 720:
        raise ValueError(
            "Unexpected train dataset size: "
            f"trajectories={train_trajectory_count}, turns={len(train_turns)}"
        )
    if dev_trajectory_count != 20 or len(dev_turns) != 80:
        raise ValueError(
            "Unexpected dev dataset size: "
            f"trajectories={dev_trajectory_count}, turns={len(dev_turns)}"
        )

    print(
        json.dumps(
            {
                "stage": "data_ready",
                "train_trajectories": train_trajectory_count,
                "dev_trajectories": dev_trajectory_count,
                "train": train_stats,
                "dev": dev_stats,
            },
            ensure_ascii=False,
        )
    )

    torch.cuda.empty_cache()

    load_started = perf_counter()
    model = load_quantized_base(args.model)
    model_load_seconds = perf_counter() - load_started

    memory_after_4bit_load = cuda_memory()
    quantization = inspect_quantization(model)

    model, lora = attach_lora(model)

    print(
        json.dumps(
            {
                "stage": "model_ready",
                "model_load_seconds": model_load_seconds,
                "memory_after_4bit_load": memory_after_4bit_load,
                "quantization": quantization,
                "lora": lora,
            },
            ensure_ascii=False,
        )
    )

    trainer_output = Path("/tmp/execsql_qlora_full_trainer")

    training_args = TrainingArguments(
        output_dir=str(trainer_output),

        num_train_epochs=args.epochs,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,

        bf16=True,
        fp16=False,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        max_grad_norm=1.0,

        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,

        eval_strategy="epoch",
        save_strategy="no",
        label_names=["labels"],

        report_to="none",

        seed=42,
        data_seed=42,

        optim="paged_adamw_8bit",

        remove_unused_columns=False,
        push_to_hub=False,
        logging_nan_inf_filter=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=AssistantTurnDataset(train_turns),
        eval_dataset=AssistantTurnDataset(dev_turns),
        data_collator=AssistantOnlyCollator(tokenizer),
        processing_class=tokenizer,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    train_started = perf_counter()

    try:
        train_output = trainer.train()
    except torch.OutOfMemoryError:
        print(
            json.dumps(
                {
                    "stage": "training_failed",
                    "oom": True,
                }
            )
        )
        raise

    wall_seconds = perf_counter() - train_started

    train_loss = float(train_output.training_loss)

    if not math.isfinite(train_loss):
        raise ValueError(
            f"Final training loss is not finite: {train_loss}"
        )

    final_eval = trainer.evaluate()
    eval_loss = float(final_eval["eval_loss"])

    if not math.isfinite(eval_loss):
        raise ValueError(
            f"Final dev loss is not finite: {eval_loss}"
        )

    args.output.mkdir(parents=True, exist_ok=True)

    # Save PEFT adapter + tokenizer only; never save the 8B base model.
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    peak_allocated = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)

    expected_optimizer_steps = math.ceil(
        len(train_turns) / args.gradient_accumulation_steps
    ) * int(args.epochs)

    print(
        json.dumps(
            {
                "stage": "training_complete",
                "global_step": trainer.state.global_step,
                "expected_optimizer_steps": expected_optimizer_steps,
                "training_loss": train_loss,
                "final_eval_loss": eval_loss,
                "train_runtime_wall_seconds": wall_seconds,
                "peak_allocated_gib": peak_allocated / (1024**3),
                "peak_reserved_gib": peak_reserved / (1024**3),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
