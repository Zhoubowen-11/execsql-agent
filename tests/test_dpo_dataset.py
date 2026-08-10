from __future__ import annotations

import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING))

from dpo_dataset import build_pairs  # noqa: E402


def _item(case_id: str = "case_1") -> dict[str, str]:
    return {
        "case_id": case_id,
        "template_family": "ranking",
        "prompt": "prompt without reward metadata",
    }


def _rollout(index: int, category: str, completion: str) -> dict[str, object]:
    return {
        "case_id": "case_1",
        "rollout_index": index,
        "category": category,
        "result_correct": category == "correct",
        "extracted_sql": "SELECT 1",
        "raw_assistant_completion": completion,
    }


def test_build_pairs_uses_first_correct_and_first_semantic_rollout() -> None:
    rollouts = [
        _rollout(5, "correct", "chosen-late"),
        _rollout(2, "semantic_mismatch", "rejected-first"),
        _rollout(0, "execution_failure", "not-eligible"),
        _rollout(4, "correct", "chosen-first"),
        _rollout(3, "semantic_mismatch", "rejected-late"),
    ]

    pairs, audit = build_pairs(items=[_item()], rollouts=rollouts)

    assert pairs == [
        {
            "case_id": "case_1",
            "family": "ranking",
            "prompt": "prompt without reward metadata",
            "chosen": "chosen-first",
            "rejected": "rejected-first",
        }
    ]
    assert audit == {"missing_pairs": []}


def test_build_pairs_rejects_nonsemantic_negative() -> None:
    pairs, audit = build_pairs(
        items=[_item()],
        rollouts=[
            _rollout(0, "correct", "chosen"),
            _rollout(1, "malformed_tool_call", "bad"),
            _rollout(2, "unsafe_sql", "unsafe"),
            _rollout(3, "execution_failure", "failed"),
        ],
    )

    assert pairs == []
    assert audit["missing_pairs"][0]["reason"] == "missing_semantic_rejected"


def test_build_pairs_reports_missing_chosen() -> None:
    pairs, audit = build_pairs(
        items=[_item()],
        rollouts=[_rollout(0, "semantic_mismatch", "rejected")],
    )

    assert pairs == []
    assert audit["missing_pairs"][0]["reason"] == "missing_chosen"


def test_build_pairs_prefers_original_before_supplement() -> None:
    rollouts = [
        _rollout(9, "correct", "chosen-supplement"),
        _rollout(11, "semantic_mismatch", "rejected-supplement"),
        _rollout(4, "correct", "chosen-original"),
        _rollout(7, "semantic_mismatch", "rejected-original"),
    ]

    pairs, _ = build_pairs(items=[_item()], rollouts=rollouts)

    assert pairs[0]["chosen"] == "chosen-original"
    assert pairs[0]["rejected"] == "rejected-original"
