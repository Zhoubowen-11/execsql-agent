"""Build leak-free SQL-decision prompts for GRPO from SFT train trajectories."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass(frozen=True)
class GRPODatasetStats:
    """Audit statistics for SQL-decision prompt construction."""

    trajectory_count: int
    prompt_count: int
    missing_sql_decision_turns: int
    prompt_token_min: int
    prompt_token_mean: float
    prompt_token_max: int
    leakage_count: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _tool_call(message: dict[str, Any], name: str) -> dict[str, Any] | None:
    if message.get("role") != "assistant":
        return None
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        return None
    call = calls[0]
    if not isinstance(call, dict):
        return None
    function = call.get("function")
    if not isinstance(function, dict) or function.get("name") != name:
        return None
    return call


def find_sql_decision_index(messages: list[dict[str, Any]]) -> int | None:
    """Return the first assistant ``validate_sql`` turn after schema inspection."""

    observed_schema = False
    for index, message in enumerate(messages):
        if message.get("role") == "tool" and message.get("name") == "inspect_schema":
            observed_schema = True
            continue
        if observed_schema and _tool_call(message, "validate_sql") is not None:
            return index
    return None


def _expected_result(metadata: dict[str, Any]) -> dict[str, Any]:
    execution = metadata.get("execution_result")
    if not isinstance(execution, dict):
        raise ValueError("metadata.execution_result must be an object")
    if execution.get("execution_success") is not True:
        raise ValueError("training ground-truth execution was not successful")
    if execution.get("truncated") is True:
        raise ValueError("training ground-truth execution was truncated")
    columns = execution.get("columns")
    rows = execution.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise ValueError("ground-truth execution must contain columns and rows")
    return {
        "columns": columns,
        "rows": rows,
        "ordered": True,
        "numeric_tolerance": 1.0e-6,
        "strict_columns": False,
    }


def _detect_leakage(prompt: str, metadata: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    lowered = prompt.casefold()
    for forbidden in ("expected_result", "gold_sql", "correctness_label"):
        if forbidden in lowered:
            findings.append(forbidden)
    gold_sql = metadata.get("gold_sql")
    if isinstance(gold_sql, str) and gold_sql.strip() and gold_sql.strip() in prompt:
        findings.append("gold_sql_value")
    expected = metadata.get("execution_result")
    if expected is not None:
        serialized_expected = json.dumps(
            expected, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        if serialized_expected in prompt:
            findings.append("expected_result_value")
    return findings


def build_grpo_items(
    trajectories_path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    *,
    database_path: str | Path,
) -> tuple[list[dict[str, Any]], GRPODatasetStats]:
    """Serialize leak-free Qwen tool prompts and attach reward-only metadata."""

    rows = _read_jsonl(Path(trajectories_path))
    items: list[dict[str, Any]] = []
    missing = 0
    leaks = 0
    lengths: list[int] = []

    for row in rows:
        messages = row.get("messages")
        tools = row.get("tools")
        metadata = row.get("metadata")
        if not isinstance(messages, list) or not isinstance(tools, list):
            raise ValueError("each trajectory must contain messages and tools lists")
        if not isinstance(metadata, dict):
            raise ValueError("each trajectory must contain metadata")
        decision_index = find_sql_decision_index(messages)
        if decision_index is None:
            missing += 1
            continue
        history = messages[:decision_index]
        prompt = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
            enable_thinking=False,
        )
        if not isinstance(prompt, str):
            raise TypeError("chat template did not return text")
        findings = _detect_leakage(prompt, metadata)
        if findings:
            leaks += 1
            raise ValueError(
                f"reward metadata leaked into prompt for {metadata.get('case_id')}: "
                f"{', '.join(findings)}"
            )
        token_count = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        lengths.append(token_count)
        expected = _expected_result(metadata)
        items.append(
            {
                "prompt": prompt,
                "case_id": str(metadata["case_id"]),
                "database_path": str(Path(database_path).resolve()),
                "expected_result_json": json.dumps(
                    expected, ensure_ascii=False, separators=(",", ":")
                ),
                "template_family": str(metadata.get("template_family", "unknown")),
                "prompt_token_count": token_count,
            }
        )

    if not lengths:
        raise ValueError("no SQL-decision prompts were constructed")
    stats = GRPODatasetStats(
        trajectory_count=len(rows),
        prompt_count=len(items),
        missing_sql_decision_turns=missing,
        prompt_token_min=min(lengths),
        prompt_token_mean=statistics.fmean(lengths),
        prompt_token_max=max(lengths),
        leakage_count=leaks,
    )
    return items, stats


def main() -> None:
    """Print a read-only prompt audit without writing a training dataset."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    _, stats = build_grpo_items(
        args.train_data, tokenizer, database_path=args.database
    )
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
