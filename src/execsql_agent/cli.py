"""Chinese CLI for single runs and deterministic Synthetic evaluation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from execsql_agent.agents.function_calling import FunctionCallingAgent
from execsql_agent.agents.pipeline import PipelineAgent
from execsql_agent.config import DomainConfig, load_domain_config
from execsql_agent.evaluation.evaluator import Evaluator
from execsql_agent.evaluation.reports import write_reports
from execsql_agent.evaluation.synthetic import (
    build_scripted_client,
    load_evaluation_dataset,
)
from execsql_agent.generation.sql_generator import SQLGenerator
from execsql_agent.llm.base import LLMClient
from execsql_agent.llm.fake import FakeLLMClient
from execsql_agent.llm.openai_compatible import OpenAICompatibleLLMClient
from execsql_agent.models import (
    EvaluationAgentMode,
    EvaluationCase,
    FunctionCallingAgentResult,
    LLMResponse,
    ResponseMode,
    ToolCallRequest,
)
from execsql_agent.tools.registry import ToolRegistry
from execsql_agent.tools.sql_executor import SQLExecutor
from execsql_agent.trajectory.logger import TrajectoryLogger


def _generation_response(sql: str, reason: str) -> LLMResponse:
    return LLMResponse(
        final_answer=json.dumps(
            {
                "sql": sql,
                "reason": reason,
                "referenced_tables": [],
                "referenced_columns": [],
            },
            ensure_ascii=False,
        ),
        response_mode=ResponseMode.PLAIN_FINAL,
    )


def _builtin_fake_responses(agent_mode: str) -> list[LLMResponse]:
    repaired_sql = (
        "SELECT c.customer_name, ROUND(SUM(o.total_amount), 2) AS total_spent "
        "FROM customers AS c JOIN orders AS o ON o.customer_id = c.customer_id "
        "WHERE o.status = 'completed' GROUP BY c.customer_id, c.customer_name "
        "ORDER BY total_spent DESC LIMIT 5"
    )
    if agent_mode == "pipeline":
        return [
            _generation_response(
                "SELECT customer_name FROM customer ORDER BY customer_name LIMIT 5",
                "演示第一次使用错误表名",
            ),
            _generation_response(repaired_sql, "根据 missing_table 修复表名和聚合查询"),
        ]
    return [
        LLMResponse(
            tool_calls=[
                ToolCallRequest(id="call_schema_1", name="inspect_schema", arguments={})
            ],
            response_mode=ResponseMode.NATIVE_TOOL_CALLS,
        ),
        LLMResponse(
            tool_calls=[
                ToolCallRequest(
                    id="call_bad_sql",
                    name="execute_sql",
                    arguments={"sql": "SELECT customer_name FROM customer LIMIT 5"},
                )
            ],
            response_mode=ResponseMode.NATIVE_TOOL_CALLS,
        ),
        LLMResponse(
            tool_calls=[
                ToolCallRequest(id="call_schema_2", name="inspect_schema", arguments={})
            ],
            response_mode=ResponseMode.NATIVE_TOOL_CALLS,
        ),
        LLMResponse(
            tool_calls=[
                ToolCallRequest(
                    id="call_fixed_sql",
                    name="execute_sql",
                    arguments={"sql": repaired_sql},
                )
            ],
            response_mode=ResponseMode.NATIVE_TOOL_CALLS,
        ),
        LLMResponse(
            final_answer="已根据真实 SQLite 查询结果返回消费金额最高的五名客户。",
            response_mode=ResponseMode.PLAIN_FINAL,
        ),
    ]


def _load_fake_responses(path: Path) -> list[LLMResponse]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("FakeLLM 响应文件根节点必须是 JSON 数组。")
    return [LLMResponse.model_validate(item) for item in raw]


def _build_client(agent_mode: str, llm_mode: str, response_file: Path | None) -> LLMClient:
    if llm_mode == "openai":
        return OpenAICompatibleLLMClient()
    responses = (
        _load_fake_responses(response_file)
        if response_file is not None
        else _builtin_fake_responses(agent_mode)
    )
    return FakeLLMClient(responses)


def _print_result(
    *,
    termination_reason: str,
    final_sql: str | None,
    execution_result: object,
    final_answer: str | None,
    trajectory_path: Path | str,
) -> None:
    print(f"终止原因：{termination_reason}")
    print(f"最终 SQL：{final_sql or '无'}")
    if hasattr(execution_result, "model_dump_json"):
        print(f"执行结果：{execution_result.model_dump_json()}")
    else:
        print("执行结果：无")
    if final_answer:
        print(f"最终回答：{final_answer}")
    print(f"轨迹文件：{trajectory_path}")


def _run_command(args: argparse.Namespace) -> int:
    database = Path(args.database)
    domain_config: DomainConfig | None = (
        load_domain_config(args.domain_config) if args.domain_config else None
    )
    trajectory_logger = TrajectoryLogger(Path(args.trajectory_file))
    client = _build_client(args.agent_mode, args.llm, args.fake_responses)
    run_mode = "deterministic/mock" if isinstance(client, FakeLLMClient) else "live"
    backend = type(client).__name__
    try:
        if args.agent_mode == "pipeline":
            agent = PipelineAgent(
                database,
                SQLGenerator(client),
                max_steps=args.max_steps or 3,
            )
            result = agent.run(args.question)
            trajectory = trajectory_logger.from_pipeline_result(
                result, llm_backend=backend, run_mode=run_mode
            )
            trajectory_path = trajectory_logger.log(trajectory)
            _print_result(
                termination_reason=result.termination_reason.value,
                final_sql=result.final_sql,
                execution_result=result.execution_result,
                final_answer=None,
                trajectory_path=trajectory_path,
            )
            return 0 if result.protocol_completed else 1

        registry = None
        if domain_config is not None:
            registry = ToolRegistry(
                database,
                executor=SQLExecutor(
                    database, max_rows=domain_config.default_result_limit
                ),
            )
        function_agent = FunctionCallingAgent(
            database,
            client,
            registry=registry,
            trajectory_logger=trajectory_logger,
            max_steps=args.max_steps or 5,
            llm_backend=backend,
            run_mode=run_mode,
            domain_context=(domain_config.to_prompt() if domain_config else None),
        )
        function_result = function_agent.run(args.question, session_id=args.session_id)
        _print_function_turns(function_result)
        _print_result(
            termination_reason=function_result.termination_reason.value,
            final_sql=function_result.final_sql,
            execution_result=function_result.execution_result,
            final_answer=function_result.final_answer,
            trajectory_path=function_result.trajectory_path or trajectory_logger.output_path,
        )
        print("Memory 摘要：")
        print(function_agent.memory_store.render_prompt(args.session_id))
        return 0 if function_result.protocol_completed else 1
    finally:
        if isinstance(client, OpenAICompatibleLLMClient):
            client.close()


def _evaluate_command(args: argparse.Namespace) -> int:
    dataset = load_evaluation_dataset(args.dataset)
    output_dir = Path(args.output_dir)
    domain_config = (
        load_domain_config(args.domain_config) if args.domain_config else None
    )

    def real_client_factory(
        _case: EvaluationCase,
        _mode: Literal["pipeline", "function-calling"],
    ) -> LLMClient:
        return OpenAICompatibleLLMClient()

    client_factory = real_client_factory if args.real_model else build_scripted_client
    evaluator = Evaluator(
        args.database,
        client_factory,
        trajectory_logger=TrajectoryLogger(output_dir / "trajectories.jsonl"),
        domain_context=(domain_config.to_prompt() if domain_config else None),
        llm_backend=("OpenAICompatibleLLMClient" if args.real_model else "FakeLLMClient"),
        run_mode=("live" if args.real_model else "deterministic/mock"),
    )
    report = evaluator.evaluate(
        dataset.cases,
        dataset_name=dataset.dataset_name,
        agent_mode=EvaluationAgentMode(args.agent_mode),
        seed=args.seed,
        tags=args.tag or [],
        limit=args.limit,
    )
    paths = write_reports(report, output_dir)
    if args.real_model:
        print("真实模型评测完成；指标来自本次实际模型调用和 SQLite 执行。")
    else:
        print("评测完成：deterministic/mock，仅表示流程可复现，不代表真实模型能力。")
    for mode, metrics in report.mode_metrics.items():
        print(
            f"{mode}：案例 {metrics.total_cases.numerator:g}，"
            f"执行成功率 {_display_metric(metrics.final_execution_success_rate.value)}，"
            f"结果正确率 {_display_metric(metrics.result_accuracy.value)}"
        )
    if report.comparison is not None:
        print(
            "结果正确率差值（Function Calling - Pipeline）："
            f"{_display_metric(report.comparison.result_accuracy_difference)}"
        )
    print(f"JSON 报告：{paths['json']}")
    print(f"CSV 报告：{paths['csv']}")
    print(f"Markdown 报告：{paths['markdown']}")
    return 0


def _display_metric(value: float | None) -> str:
    return "null" if value is None else f"{value:.4f}"


def _print_function_turns(result: FunctionCallingAgentResult) -> None:
    """Print bounded per-turn Function Calling facts without exposing credentials."""

    for step in result.steps:
        print(f"第 {step.turn_index} 轮：")
        for call in step.tool_calls:
            arguments = json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)
            print(f"  工具调用：{call.name} {arguments}")
            if call.name in {"validate_sql", "execute_sql"} and call.arguments:
                sql = call.arguments.get("sql")
                if isinstance(sql, str):
                    print(f"  SQL：{sql}")
        for tool_result in step.tool_results:
            summary = (
                f"success={tool_result.success}, executed={tool_result.executed}, "
                f"error={tool_result.error_code or 'none'}"
            )
            execution = ToolRegistry.execution_result(tool_result)
            if execution is not None:
                summary += (
                    f", execution_success={execution.execution_success}, "
                    f"rows={execution.returned_row_count}, "
                    f"sample={execution.rows[:3]}"
                )
            print(f"  执行结果：{summary}")


def build_parser() -> argparse.ArgumentParser:
    """Build single-run and Synthetic evaluation commands."""

    parser = argparse.ArgumentParser(description="ExecSQL-Agent 最小运行工具")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="运行单个 Text-to-SQL 问题")
    run_parser.add_argument(
        "--agent-mode",
        choices=["pipeline", "function-calling"],
        required=True,
        help="Agent 编排模式",
    )
    run_parser.add_argument("--database", required=True, help="SQLite 数据库路径")
    run_parser.add_argument("--question", required=True, help="中文自然语言问题")
    run_parser.add_argument(
        "--domain-config",
        type=Path,
        help="可选的严格 JSON 领域配置，仅作为模型上下文和结果行限制",
    )
    run_parser.add_argument(
        "--session-id",
        default="default",
        help="Function Calling 会话 ID；同一轨迹文件中按此隔离最近 5 轮上下文",
    )
    run_parser.add_argument(
        "--llm",
        choices=["fake", "openai"],
        default="fake",
        help="模型后端，默认使用 deterministic/mock FakeLLM",
    )
    run_parser.add_argument(
        "--fake-responses",
        type=Path,
        help="可选的 FakeLLM LLMResponse JSON 数组文件",
    )
    run_parser.add_argument(
        "--trajectory-file",
        type=Path,
        default=Path("data/trajectories/trajectories.jsonl"),
        help="JSONL 轨迹文件路径",
    )
    run_parser.add_argument("--max-steps", type=int, help="覆盖默认最大步骤数")
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="运行 deterministic/mock Synthetic 批量评测"
    )
    evaluate_parser.add_argument(
        "--agent-mode",
        choices=[mode.value for mode in EvaluationAgentMode],
        required=True,
        help="评测 Pipeline、Function Calling 或两者",
    )
    evaluate_parser.add_argument("--database", required=True, help="SQLite 数据库路径")
    evaluate_parser.add_argument("--dataset", required=True, type=Path, help="评测集路径")
    evaluate_parser.add_argument(
        "--output-dir", required=True, type=Path, help="JSON、CSV、Markdown 输出目录"
    )
    evaluate_parser.add_argument("--limit", type=int, help="最多运行的案例数")
    evaluate_parser.add_argument(
        "--tag", action="append", help="按标签筛选，可重复指定且需全部匹配"
    )
    evaluate_parser.add_argument("--seed", type=int, default=0, help="固定随机种子")
    evaluate_parser.add_argument(
        "--domain-config",
        type=Path,
        help="可选的严格 JSON 领域配置",
    )
    evaluate_parser.add_argument(
        "--real-model",
        action="store_true",
        help="使用 OPENAI_API_KEY、OPENAI_BASE_URL、OPENAI_MODEL 调用真实模型",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected agent and render Chinese errors."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            return _evaluate_command(args)
        return _run_command(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"运行失败：{error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
