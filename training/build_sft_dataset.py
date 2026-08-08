"""Build execution-verified FSQ Function Calling SFT datasets."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from sft_case_factory import SFTCase, generate_cases

from execsql_agent.models import ExecutionResult, ToolCallRequest, ToolCallResult
from execsql_agent.tools.registry import ToolRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TOOLS = {"inspect_schema", "validate_sql", "execute_sql"}
SEED_CASE_ID = re.compile(r"sft_seed_\d{3}\Z")
FULL_CASE_ID = re.compile(r"sft_v1_[a-j]_\d{3}\Z")


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return raw


def _normalized_question(question: str) -> str:
    return "".join(question.split()).casefold()


def _questions_v2(path: Path) -> set[str]:
    """Extract only normalized questions for exact de-duplication."""

    raw = _load_json_object(path)
    cases = raw.get("cases")
    if not isinstance(cases, list):
        raise ValueError("questions_v2.json must contain a cases array")
    questions: set[str] = set()
    for case in cases:
        if isinstance(case, dict) and isinstance(case.get("question"), str):
            questions.add(_normalized_question(case["question"]))
    return questions


def _load_seed_cases(path: Path, questions_v2_path: Path) -> tuple[str, list[dict[str, Any]]]:
    raw = _load_json_object(path)
    dataset_name = raw.get("dataset_name")
    cases = raw.get("cases")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("seed_cases.json requires a non-empty dataset_name")
    if not isinstance(cases, list) or len(cases) != 5:
        raise ValueError("seed_cases.json must contain exactly five cases")

    existing_questions = _questions_v2(questions_v2_path)
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, raw_case in enumerate(cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Seed case {index} must be a JSON object")
        case = dict(raw_case)
        case_id = case.get("case_id")
        question = case.get("question")
        sql = case.get("sql")
        inspect_tables = case.get("inspect_tables")
        answer = case.get("answer")
        if not isinstance(case_id, str) or not SEED_CASE_ID.fullmatch(case_id):
            raise ValueError(f"Invalid seed-only case_id: {case_id!r}")
        if case_id.startswith("fsq_") or case_id in seen_ids:
            raise ValueError(f"Forbidden or duplicate case_id: {case_id}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{case_id}: question must be non-empty")
        normalized = _normalized_question(question)
        if normalized in existing_questions:
            raise ValueError(f"{case_id}: question duplicates questions_v2.json")
        if normalized in seen_questions:
            raise ValueError(f"{case_id}: duplicate seed question")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError(f"{case_id}: sql must be non-empty")
        if not isinstance(inspect_tables, list) or not all(
            isinstance(table, str) for table in inspect_tables
        ):
            raise ValueError(f"{case_id}: inspect_tables must be a string array")
        if not isinstance(answer, dict):
            raise ValueError(f"{case_id}: answer must be an object")
        case.setdefault("template_family", case.get("failure_mode", "seed"))
        case.setdefault("difficulty", "medium")
        case.setdefault("parameter_signature", case_id)
        seen_ids.add(case_id)
        seen_questions.add(normalized)
        validated.append(case)
    return dataset_name, validated


def _tool_schemas(registry: ToolRegistry) -> list[dict[str, object]]:
    definitions = registry.definitions
    names = {definition.name for definition in definitions}
    if names != ALLOWED_TOOLS:
        raise ValueError(f"ToolRegistry allowlist changed: {sorted(names)}")
    return [
        {"type": "function", "function": definition.model_dump(mode="json")}
        for definition in definitions
    ]


def _dispatch(registry: ToolRegistry, call: ToolCallRequest) -> ToolCallResult:
    if call.name not in ALLOWED_TOOLS:
        raise ValueError(f"Forbidden tool: {call.name}")
    validation, result = registry.dispatch(call)
    if not validation.valid:
        raise ValueError(
            f"{call.id}: invalid {call.name} call: "
            f"{validation.error_code}: {validation.error_message}"
        )
    if not result.success:
        raise ValueError(
            f"{call.id}: {call.name} failed: {result.error_code}: {result.error_message}"
        )
    return result


def _assistant_tool_message(call: ToolCallRequest) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(
                        call.arguments or {},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            }
        ],
    }


def _tool_result_message(call: ToolCallRequest, result: ToolCallResult) -> dict[str, object]:
    return {
        "role": "tool",
        "tool_call_id": call.id,
        "name": call.name,
        "content": result.model_dump_json(),
    }


def _row_mapping(execution: ExecutionResult, row: list[object]) -> dict[str, object]:
    if len(execution.columns) != len(row):
        raise ValueError("Execution columns and row width differ")
    return dict(zip(execution.columns, row, strict=True))


def _render_answer(answer: dict[str, Any], execution: ExecutionResult) -> str:
    kind = answer.get("kind")
    template = answer.get("template")
    if not isinstance(template, str) or not template.strip():
        raise ValueError("answer.template must be non-empty")
    if kind == "scalar":
        if len(execution.rows) != 1:
            raise ValueError("A scalar answer requires exactly one result row")
        return template.format_map(_row_mapping(execution, execution.rows[0]))
    if kind == "rows":
        row_template = answer.get("row_template")
        if not isinstance(row_template, str) or not row_template.strip():
            raise ValueError("A rows answer requires row_template")
        rendered_rows: list[str] = []
        for rank, row in enumerate(execution.rows, start=1):
            values = _row_mapping(execution, row)
            values["rank"] = rank
            rendered_rows.append(row_template.format_map(values))
        return template.format(rows="；".join(rendered_rows))
    raise ValueError(f"Unknown answer kind: {kind!r}")


def _build_sample(
    *,
    dataset_name: str,
    split: str,
    case: dict[str, Any],
    registry: ToolRegistry,
    tools: list[dict[str, object]],
) -> dict[str, object]:
    case_id = str(case["case_id"])
    question = str(case["question"])
    sql = str(case["sql"])
    safety = registry.validator.validate(sql, database_path=registry.database_path)
    if not safety.safe or safety.syntax_valid is not True:
        raise ValueError(
            f"{case_id}: ground-truth SQL is not safe and compilable: "
            f"{safety.reason or safety.validation_error}"
        )

    calls = [
        ToolCallRequest(
            id=f"{case_id}_inspect",
            name="inspect_schema",
            arguments={"table_names": case["inspect_tables"]},
        ),
        ToolCallRequest(
            id=f"{case_id}_validate",
            name="validate_sql",
            arguments={"sql": sql},
        ),
        ToolCallRequest(
            id=f"{case_id}_execute",
            name="execute_sql",
            arguments={"sql": sql},
        ),
    ]
    if [call.name for call in calls] != [
        "inspect_schema",
        "validate_sql",
        "execute_sql",
    ]:
        raise ValueError(f"{case_id}: invalid tool sequence")

    messages: list[dict[str, object]] = [{"role": "user", "content": question}]
    execute_result: ToolCallResult | None = None
    for call in calls:
        result = _dispatch(registry, call)
        messages.append(_assistant_tool_message(call))
        messages.append(_tool_result_message(call, result))
        if call.name == "execute_sql":
            execute_result = result

    if execute_result is None:
        raise ValueError(f"{case_id}: missing execute_sql result")
    execution = registry.execution_result(execute_result)
    if execution is None or not execution.executed or not execution.execution_success:
        raise ValueError(f"{case_id}: execute_sql did not succeed")
    if execution.truncated:
        raise ValueError(f"{case_id}: final result is truncated")

    final_answer = _render_answer(case["answer"], execution)
    messages.append({"role": "assistant", "content": final_answer})
    return {
        "messages": messages,
        "tools": tools,
        "metadata": {
            "dataset_name": dataset_name,
            "split": split,
            "case_id": case_id,
            "template_family": case["template_family"],
            "difficulty": case["difficulty"],
            "parameter_signature": case["parameter_signature"],
            "gold_sql": sql,
            "execution_result": execution.model_dump(mode="json"),
        },
    }


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _serialize_sample(sample: dict[str, object]) -> str:
    return json.dumps(sample, ensure_ascii=False, separators=(",", ":"))


def _atomic_write_jsonl(path: Path, samples: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(_serialize_sample(sample))
            handle.write("\n")
    temporary.replace(path)


def _validate_jsonl(path: Path, *, expected_count: int, id_pattern: re.Pattern[str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != expected_count:
        raise ValueError(f"Expected {expected_count} JSONL lines, found {len(lines)}")
    for line_number, line in enumerate(lines, start=1):
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"JSONL line {line_number} is not an object")
        metadata = parsed.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"JSONL line {line_number} has no metadata object")
        case_id = metadata.get("case_id")
        if not isinstance(case_id, str) or not id_pattern.fullmatch(case_id):
            raise ValueError(f"JSONL line {line_number} contains a forbidden case id")
        execution = metadata.get("execution_result")
        if not isinstance(execution, dict):
            raise ValueError(f"JSONL line {line_number} has no execution_result")
        if (
            execution.get("execution_success") is not True
            or execution.get("truncated") is not False
        ):
            raise ValueError(f"JSONL line {line_number} has an invalid execution result")
        messages = parsed.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"JSONL line {line_number} has no messages array")
        tool_names = {
            call["function"]["name"]
            for message in messages
            if isinstance(message, dict)
            for call in message.get("tool_calls", [])
            if isinstance(call, dict) and isinstance(call.get("function"), dict)
        }
        if tool_names != ALLOWED_TOOLS:
            raise ValueError(f"JSONL line {line_number} has an invalid tool sequence")


def _split_cases(cases: list[SFTCase], seed: int) -> tuple[list[SFTCase], list[SFTCase]]:
    rng = random.Random(seed)
    by_family: dict[str, list[SFTCase]] = {}
    for case in cases:
        by_family.setdefault(case.template_family, []).append(case)
    train: list[SFTCase] = []
    dev: list[SFTCase] = []
    for family in sorted(by_family):
        family_cases = sorted(by_family[family], key=lambda item: item.case_id)
        rng.shuffle(family_cases)
        dev.extend(family_cases[:2])
        train.extend(family_cases[2:])
    rng.shuffle(train)
    rng.shuffle(dev)
    train_signatures = {case.parameter_signature for case in train}
    dev_signatures = {case.parameter_signature for case in dev}
    overlap = train_signatures & dev_signatures
    if overlap:
        raise ValueError(f"Train/dev parameter leakage: {sorted(overlap)}")
    return train, dev


def _deduplication_stats(
    cases: list[SFTCase], registry: ToolRegistry, questions_v2: set[str]
) -> tuple[int, int, int]:
    questions = [_normalized_question(case.question) for case in cases]
    normalized_sql: list[str] = []
    for case in cases:
        safety = registry.validator.validate(case.sql, database_path=registry.database_path)
        if not safety.safe or safety.syntax_valid is not True:
            raise ValueError(f"{case.case_id}: SQL failed preflight validation")
        normalized_sql.append(safety.normalized_sql)
    question_duplicates = len(questions) - len(set(questions))
    sql_duplicates = len(normalized_sql) - len(set(normalized_sql))
    collisions = sum(question in questions_v2 for question in questions)
    return question_duplicates, sql_duplicates, collisions


def _print_statistics(
    *,
    train_samples: list[dict[str, object]],
    dev_samples: list[dict[str, object]],
    question_duplicates: int,
    sql_duplicates: int,
    collisions: int,
    sample_seed: int,
) -> None:
    samples = train_samples + dev_samples
    family_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    message_total = 0
    max_line_bytes = 0
    for sample in samples:
        metadata = sample["metadata"]
        messages = sample["messages"]
        assert isinstance(metadata, dict)
        assert isinstance(messages, list)
        family_counts[str(metadata["template_family"])] += 1
        difficulty_counts[str(metadata["difficulty"])] += 1
        message_total += len(messages)
        max_line_bytes = max(max_line_bytes, len(_serialize_sample(sample).encode("utf-8")))

    print(f"total_samples={len(samples)}")
    print(f"train_samples={len(train_samples)}")
    print(f"dev_samples={len(dev_samples)}")
    family_json = json.dumps(dict(sorted(family_counts.items())), ensure_ascii=False)
    difficulty_json = json.dumps(dict(sorted(difficulty_counts.items())), ensure_ascii=False)
    print(f"template_family_counts={family_json}")
    print(f"difficulty_counts={difficulty_json}")
    print(f"question_duplicates={question_duplicates}")
    print(f"sql_duplicates={sql_duplicates}")
    print(f"questions_v2_collisions={collisions}")
    print(f"average_messages={message_total / len(samples):.2f}")
    print(f"max_jsonl_line_bytes={max_line_bytes}")

    rng = random.Random(sample_seed)
    print("sample_summaries=")
    for sample in rng.sample(samples, 5):
        metadata = sample["metadata"]
        messages = sample["messages"]
        assert isinstance(metadata, dict)
        assert isinstance(messages, list)
        user_message = messages[0]
        final_message = messages[-1]
        tool_calls = [
            call["function"]["name"]
            for message in messages
            if isinstance(message, dict)
            for call in message.get("tool_calls", [])
            if isinstance(call, dict) and isinstance(call.get("function"), dict)
        ]
        summary = {
            "case_id": metadata["case_id"],
            "split": metadata["split"],
            "template_family": metadata["template_family"],
            "difficulty": metadata["difficulty"],
            "question": user_message["content"] if isinstance(user_message, dict) else None,
            "gold_sql": metadata["gold_sql"],
            "tool_calls": tool_calls,
            "execution_result": metadata["execution_result"],
            "final_answer": final_message["content"] if isinstance(final_message, dict) else None,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_seed_dataset(args: argparse.Namespace) -> list[dict[str, object]]:
    database = _project_path(args.database)
    dataset_name, cases = _load_seed_cases(
        _project_path(args.seed_cases), _project_path(args.questions_v2)
    )
    registry = ToolRegistry(database)
    tools = _tool_schemas(registry)
    samples = [
        _build_sample(
            dataset_name=dataset_name,
            split="seed",
            case=case,
            registry=registry,
            tools=tools,
        )
        for case in cases
    ]
    _atomic_write_json(_project_path(args.tools_output), tools)
    _atomic_write_jsonl(_project_path(args.seed_output), samples)
    _validate_jsonl(_project_path(args.seed_output), expected_count=5, id_pattern=SEED_CASE_ID)
    return samples


def build_full_dataset(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    database = _project_path(args.database)
    if not database.is_file():
        raise FileNotFoundError(f"Database not found: {database}")
    registry = ToolRegistry(database)
    tools = _tool_schemas(registry)
    cases = generate_cases()
    questions_v2 = _questions_v2(_project_path(args.questions_v2))
    question_duplicates, sql_duplicates, collisions = _deduplication_stats(
        cases, registry, questions_v2
    )
    if question_duplicates or sql_duplicates or collisions:
        raise ValueError(
            "Dataset de-duplication failed: "
            f"question_duplicates={question_duplicates}, sql_duplicates={sql_duplicates}, "
            f"questions_v2_collisions={collisions}"
        )
    train_cases, dev_cases = _split_cases(cases, args.split_seed)
    train_samples = [
        _build_sample(
            dataset_name="fsq_shanghai_function_calling_sft_v1",
            split="train",
            case=case.as_dict(),
            registry=registry,
            tools=tools,
        )
        for case in train_cases
    ]
    dev_samples = [
        _build_sample(
            dataset_name="fsq_shanghai_function_calling_sft_v1",
            split="dev",
            case=case.as_dict(),
            registry=registry,
            tools=tools,
        )
        for case in dev_cases
    ]
    _atomic_write_json(_project_path(args.tools_output), tools)
    _atomic_write_jsonl(_project_path(args.train_output), train_samples)
    _atomic_write_jsonl(_project_path(args.dev_output), dev_samples)
    _validate_jsonl(_project_path(args.train_output), expected_count=180, id_pattern=FULL_CASE_ID)
    _validate_jsonl(_project_path(args.dev_output), expected_count=20, id_pattern=FULL_CASE_ID)
    _print_statistics(
        train_samples=train_samples,
        dev_samples=dev_samples,
        question_duplicates=question_duplicates,
        sql_duplicates=sql_duplicates,
        collisions=collisions,
        sample_seed=args.sample_seed,
    )
    return train_samples, dev_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build execution-verified FSQ Function Calling SFT data."
    )
    parser.add_argument("--mode", choices=["full", "seed"], default="full")
    parser.add_argument("--database", type=Path, default=Path("data/fsq/shanghai_places.db"))
    parser.add_argument(
        "--questions-v2", type=Path, default=Path("data/fsq/eval/questions_v2.json")
    )
    parser.add_argument("--tools-output", type=Path, default=Path("data/fsq/train/sft/tools.json"))
    parser.add_argument(
        "--seed-cases", type=Path, default=Path("data/fsq/train/sft/seed_cases.json")
    )
    parser.add_argument(
        "--seed-output", type=Path, default=Path("data/fsq/train/sft/train_seed_v1.jsonl")
    )
    parser.add_argument(
        "--train-output", type=Path, default=Path("data/fsq/train/sft/train_v1.jsonl")
    )
    parser.add_argument("--dev-output", type=Path, default=Path("data/fsq/train/sft/dev_v1.jsonl"))
    parser.add_argument("--split-seed", type=int, default=20260808)
    parser.add_argument("--sample-seed", type=int, default=20260808)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "seed":
        samples = build_seed_dataset(args)
        print(f"Generated {len(samples)} seed samples: {_project_path(args.seed_output)}")
        return 0
    train_samples, dev_samples = build_full_dataset(args)
    print(f"Generated train JSONL: {_project_path(args.train_output)}")
    print(f"Generated dev JSONL: {_project_path(args.dev_output)}")
    print(f"Exported ToolRegistry schemas: {_project_path(args.tools_output)}")
    print(f"Final split: train={len(train_samples)}, dev={len(dev_samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
