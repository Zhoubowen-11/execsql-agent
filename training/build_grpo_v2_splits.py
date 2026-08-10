"""Merge RL-hard V2 audits and write leakage-safe deterministic splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(paths: list[Path]) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                case_id = str(item["case_id"])
                if case_id in items:
                    raise ValueError(f"duplicate case_id: {case_id}")
                items[case_id] = item
    return items


def read_groups(paths: list[Path]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for path in paths:
        groups.extend(json.loads(path.read_text(encoding="utf-8"))["groups"])
    return groups


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, action="append", required=True)
    parser.add_argument("--audits", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    items = read_jsonl(args.candidates)
    groups = read_groups(args.audits)
    if len(items) != 600 or len(groups) != 600:
        raise ValueError(f"expected 600 candidates/groups, got {len(items)}/{len(groups)}")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        if group["classification"] == "mixed":
            by_family[str(group["template_family"])].append(group)
    dev_ids: set[str] = set()
    for _family, family_groups in sorted(by_family.items()):
        ranked = sorted(
            family_groups,
            key=lambda group: (
                abs(sum(float(value) == 1.0 for value in group["rewards"]) - 2),
                str(group["case_id"]),
            ),
        )
        dev_ids.add(str(ranked[0]["case_id"]))
    mixed_ids = {str(group["case_id"]) for group in groups if group["classification"] == "mixed"}
    train_ids = mixed_ids - dev_ids
    hard_ids: set[str] = set()
    for family in sorted({str(group["template_family"]) for group in groups}):
        candidates = sorted(
            (
                group
                for group in groups
                if group["template_family"] == family
                and group["classification"]
                in {"all_semantic_wrong", "execution_malformed_unsafe_dominated"}
            ),
            key=lambda group: (str(group["difficulty"]), str(group["case_id"])),
        )
        hard_ids.update(str(group["case_id"]) for group in candidates[:4])
    split_ids = {"mixed_train": train_ids, "mixed_dev": dev_ids, "hard_audit": hard_ids}
    signatures: dict[str, set[str]] = {}
    for name, ids in split_ids.items():
        signatures[name] = {str(items[case_id]["parameter_signature"]) for case_id in ids}
        write_jsonl(
            args.output_dir / f"rl_train_hard_v2_{name}.jsonl",
            [items[case_id] for case_id in sorted(ids)],
        )
    for left, right in (
        ("mixed_train", "mixed_dev"),
        ("mixed_train", "hard_audit"),
        ("mixed_dev", "hard_audit"),
    ):
        if signatures[left] & signatures[right]:
            raise ValueError(f"parameter leakage between {left} and {right}")
    summary = {
        "candidate_count": len(items),
        "mixed_total": len(mixed_ids),
        "mixed_train": len(train_ids),
        "mixed_dev": len(dev_ids),
        "hard_audit": len(hard_ids),
        "mixed_by_family": dict(
            Counter(
                group["template_family"] for group in groups if group["classification"] == "mixed"
            )
        ),
        "mixed_correct_counts": dict(
            Counter(
                f"{sum(float(v) == 1.0 for v in group['rewards'])}_of_4"
                for group in groups
                if group["classification"] == "mixed"
            )
        ),
        "pilot_gate": {
            "mixed_train_at_least_100": len(train_ids) >= 100,
            "each_family_at_least_15": all(len(value) >= 15 for value in by_family.values()),
            "passed": len(train_ids) >= 100
            and all(len(value) >= 15 for value in by_family.values()),
        },
        "parameter_signature_overlap": 0,
    }
    (args.output_dir / "rl_train_hard_v2_split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
