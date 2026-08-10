"""Build the V2.1 first-mixed pool and stable eight-rollout splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_items(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                case_id = str(item["case_id"])
                if case_id in result:
                    raise ValueError(f"duplicate case_id: {case_id}")
                result[case_id] = item
    return result


def load_groups(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        for group in json.loads(path.read_text(encoding="utf-8"))["groups"]:
            case_id = str(group["case_id"])
            if case_id in result:
                raise ValueError(f"duplicate audit group: {case_id}")
            result[case_id] = group
    return result


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def choose_balanced(
    groups: list[dict[str, Any]], *, per_family: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chosen: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_family[str(group["template_family"])].append(group)
    for family_groups in by_family.values():
        ranked = sorted(
            family_groups,
            key=lambda group: (
                abs(int(group["eight_correct"]) - 4),
                str(group["case_id"]),
            ),
        )
        chosen.extend(ranked[:per_family])
        remaining.extend(ranked[per_family:])
    return chosen, remaining


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, action="append", required=True)
    parser.add_argument("--first-audits", type=Path, action="append", required=True)
    parser.add_argument("--second-audit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    items = load_items(args.candidates)
    first = load_groups(args.first_audits)
    if set(items) != set(first):
        raise ValueError("candidate and first-audit IDs do not match")
    mixed_ids = sorted(
        case_id for case_id, group in first.items() if group["classification"] == "mixed"
    )
    pool_path = args.output_dir / "rl_train_hard_v21_first_mixed_pool.jsonl"
    write_jsonl(pool_path, [items[case_id] for case_id in mixed_ids])
    initial = {
        "candidate_count": len(items),
        "first_mixed_count": len(mixed_ids),
        "first_mixed_by_family": dict(
            Counter(first[case_id]["template_family"] for case_id in mixed_ids)
        ),
        "pool_path": str(pool_path),
    }
    if args.second_audit is None:
        print(json.dumps(initial, ensure_ascii=False, indent=2))
        return
    second = load_groups([args.second_audit])
    if set(second) != set(mixed_ids):
        raise ValueError("second audit must contain exactly the first-mixed pool")
    stable_groups: list[dict[str, Any]] = []
    second_zero_std = 0
    eight_distribution: Counter[int] = Counter()
    for case_id in mixed_ids:
        first_rewards = [float(value) for value in first[case_id]["rewards"]]
        second_rewards = [float(value) for value in second[case_id]["rewards"]]
        second_zero_std += len(set(second_rewards)) == 1
        eight_correct = sum(value == 1.0 for value in first_rewards + second_rewards)
        eight_distribution[eight_correct] += 1
        group = dict(first[case_id])
        group["first_rewards"] = first_rewards
        group["second_rewards"] = second_rewards
        group["eight_correct"] = eight_correct
        if 2 <= eight_correct <= 6:
            stable_groups.append(group)
    dev, train_candidates = choose_balanced(stable_groups, per_family=4)
    train = sorted(
        train_candidates,
        key=lambda group: (
            abs(int(group["eight_correct"]) - 4),
            str(group["case_id"]),
        ),
    )[:130]
    hard: list[dict[str, Any]] = []
    for family in sorted({str(group["template_family"]) for group in first.values()}):
        candidates = sorted(
            (
                group
                for group in first.values()
                if group["template_family"] == family
                and group["classification"]
                in {"all_semantic_wrong", "execution_malformed_unsafe_dominated"}
            ),
            key=lambda group: (str(group["difficulty"]), str(group["case_id"])),
        )
        hard.extend(candidates[:6])
    split_groups = {"mixed_train": train, "mixed_dev": dev, "hard_audit": hard}
    split_signatures: dict[str, set[str]] = {}
    for name, groups in split_groups.items():
        split_signatures[name] = {
            str(items[str(group["case_id"])]["parameter_signature"]) for group in groups
        }
        write_jsonl(
            args.output_dir / f"rl_train_hard_v21_{name}.jsonl",
            [items[str(group["case_id"])] for group in groups],
        )
    overlap = 0
    names = list(split_signatures)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap += len(split_signatures[left] & split_signatures[right])
    if overlap:
        raise ValueError(f"parameter signatures overlap across splits: {overlap}")
    robust_by_family = Counter(group["template_family"] for group in stable_groups)
    train_by_family = Counter(group["template_family"] for group in train)
    dev_by_family = Counter(group["template_family"] for group in dev)
    unsafe = sum(
        category == "unsafe_sql" for group in second.values() for category in group["categories"]
    )
    malformed = sum(
        category == "malformed_tool_call"
        for group in second.values()
        for category in group["categories"]
    )
    summary = {
        **initial,
        "second_seed_zero_std_count": second_zero_std,
        "second_seed_zero_std_rate": second_zero_std / len(mixed_ids),
        "eight_correct_distribution": {str(key): eight_distribution[key] for key in range(9)},
        "robust_mixed_count": len(stable_groups),
        "robust_mixed_by_family": dict(robust_by_family),
        "mixed_train_count": len(train),
        "mixed_train_by_family": dict(train_by_family),
        "mixed_dev_count": len(dev),
        "mixed_dev_by_family": dict(dev_by_family),
        "hard_audit_count": len(hard),
        "second_seed_malformed_rate": malformed / (len(mixed_ids) * 4),
        "second_seed_unsafe_rate": unsafe / (len(mixed_ids) * 4),
        "parameter_signature_overlap": overlap,
        "pilot_gate": {
            "robust_train_at_least_110": len(train) >= 110,
            "mixed_dev_at_least_24": len(dev) >= 24,
            "each_train_family_at_least_12": all(
                train_by_family[family] >= 12 for family in robust_by_family
            ),
            "weak_families_recovered": train_by_family["multi_category_semantics"] >= 12
            and train_by_family["output_shape"] >= 12,
        },
    }
    summary["pilot_gate"]["passed"] = all(summary["pilot_gate"].values())
    summary_path = args.output_dir / "rl_train_hard_v21_stability_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
