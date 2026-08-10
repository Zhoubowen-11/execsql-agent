"""Summarize four existing fixed-seed SFT/GRPO rollout audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_grpo_pilot import compare


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-dev", type=Path, required=True)
    parser.add_argument("--grpo-dev", type=Path, required=True)
    parser.add_argument("--sft-hard", type=Path, required=True)
    parser.add_argument("--grpo-hard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dev = compare(load(args.sft_dev), load(args.grpo_dev))
    hard = compare(load(args.sft_hard), load(args.grpo_hard))
    positive_families = sum(values["delta_count"] > 0 for values in dev["family_delta"].values())
    gate = {
        "mixed_dev_improves_by_more_than_two_rollouts": (
            dev["grpo"]["correct"] - dev["sft"]["correct"] > 2
        ),
        "hard_audit_not_regressed": hard["grpo"]["correct"] >= hard["sft"]["correct"],
        "at_least_three_dev_families_improve": positive_families >= 3,
        "unsafe_not_increased_on_either_split": (
            dev["grpo"]["unsafe"] <= dev["sft"]["unsafe"]
            and hard["grpo"]["unsafe"] <= hard["sft"]["unsafe"]
        ),
        "malformed_not_increased_on_either_split": (
            dev["grpo"]["malformed"] <= dev["sft"]["malformed"]
            and hard["grpo"]["malformed"] <= hard["sft"]["malformed"]
        ),
    }
    gate["recommend_formal_grpo"] = all(gate.values())
    summary = {
        "settings": {
            "temperature": 1.0,
            "num_generations": 4,
            "max_completion_length": 256,
            "dev_seed": 20260812,
            "hard_seed": 20260813,
        },
        "mixed_dev": dev,
        "hard_audit": hard,
        "positive_dev_family_count": positive_families,
        "gate": gate,
    }
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
