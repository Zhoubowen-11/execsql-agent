"""Build the versioned FSQ v2 dataset from validated real SQLite results."""

from __future__ import annotations

import argparse
from pathlib import Path

from execsql_agent.evaluation.synthetic import load_evaluation_dataset
from execsql_agent.models import EvaluationDataset, ExpectedResult
from execsql_agent.tools.sql_executor import SQLExecutor
from execsql_agent.tools.sql_validator import SQLValidator

DEFAULT_DATABASE = Path("data/fsq/shanghai_places.db")
DEFAULT_SOURCE = Path("data/fsq/eval/questions_v1.json")
DEFAULT_OUTPUT = Path("data/fsq/eval/questions_v2.json")

QUESTION_REVISIONS = {
    "fsq_c05": (
        "在同时标注了多种类型的地点中，哪些地点类型出现得最多？"
        "请按涉及的地点数量列出前十种。"
    ),
    "fsq_c07": (
        "类型名称中包含“Restaurant”的地点类型里，哪些覆盖的地点最多？"
        "请列出前十种。"
    ),
    "fsq_a01": (
        "原始城市或区县标注中，出现次数最多的十个名称是什么？"
        "省级区域标注不计入。"
    ),
}


def build_v2_dataset(database: Path, source: Path, output: Path) -> Path:
    """Re-execute every gold SQL and atomically write the clarified v2 dataset."""

    if not database.is_file():
        raise FileNotFoundError(f"FSQ database does not exist: {database}")
    if not source.is_file():
        raise FileNotFoundError(f"FSQ v1 dataset does not exist: {source}")
    if source.resolve() == output.resolve():
        raise ValueError("The v2 output must not overwrite the v1 source.")

    dataset = load_evaluation_dataset(source)
    validator = SQLValidator()
    executor = SQLExecutor(database, max_rows=200)
    cases = []
    for case in dataset.cases:
        if case.gold_sql is None:
            raise ValueError(f"Case {case.id} does not define gold_sql.")
        safety = validator.validate(case.gold_sql, database_path=database)
        if not safety.safe or safety.syntax_valid is not True:
            raise ValueError(
                f"Gold SQL validation failed for {case.id}: "
                f"{safety.model_dump_json()}"
            )
        result = executor.execute(case.gold_sql)
        if not result.execution_success or result.truncated:
            raise ValueError(
                f"Gold SQL execution failed for {case.id}: "
                f"{result.model_dump_json()}"
            )
        expected = ExpectedResult(
            columns=result.columns,
            rows=result.rows,
            ordered=False if case.id == "fsq_a07" else case.expected_result.ordered,
            numeric_tolerance=case.expected_result.numeric_tolerance,
            strict_columns=case.expected_result.strict_columns,
        )
        updates: dict[str, object] = {"expected_result": expected}
        if case.id in QUESTION_REVISIONS:
            updates["question"] = QUESTION_REVISIONS[case.id]
        cases.append(case.model_copy(update=updates))

    versioned = EvaluationDataset(
        dataset_name="fsq_shanghai_business_v2",
        cases=cases,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        versioned.model_dump_json(indent=2), encoding="utf-8", newline="\n"
    )
    temporary.replace(output)
    return output.resolve()


def main() -> None:
    """Parse paths and generate the FSQ v2 evaluation dataset."""

    parser = argparse.ArgumentParser(
        description="从 FSQ v1 和真实 SQLite 结果生成无歧义的 v2 评测集。"
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    created = build_v2_dataset(args.database, args.source, args.output)
    print(f"FSQ v2 evaluation dataset created: {created}")


if __name__ == "__main__":
    main()
