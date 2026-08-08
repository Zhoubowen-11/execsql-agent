"""Batch evaluation for Pipeline and Function Calling on real SQLite results."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Literal
from uuid import uuid4

from execsql_agent.agents.function_calling import FunctionCallingAgent
from execsql_agent.agents.pipeline import PipelineAgent
from execsql_agent.evaluation.comparator import compare_execution_result
from execsql_agent.evaluation.metrics import calculate_mode_metrics, compare_modes
from execsql_agent.generation.sql_generator import SQLGenerator
from execsql_agent.llm.base import LLMClient
from execsql_agent.llm.fake import FakeLLMClient
from execsql_agent.llm.openai_compatible import OpenAICompatibleLLMClient
from execsql_agent.models import (
    BehaviorScenario,
    CaseEvaluation,
    EvaluationAgentMode,
    EvaluationCase,
    EvaluationReport,
    ExecutionResult,
    FailureKind,
    FunctionCallingAgentResult,
    LLMMessage,
    ModeMetrics,
    PipelineAgentResult,
    Trajectory,
)
from execsql_agent.tools.registry import ToolRegistry
from execsql_agent.trajectory.logger import TrajectoryLogger

ConcreteMode = Literal["pipeline", "function-calling"]
ClientFactory = Callable[[EvaluationCase, ConcreteMode], LLMClient]


class Evaluator:
    """Run isolated cases, compare real results, and aggregate auditable metrics."""

    def __init__(
        self,
        database_path: str | Path,
        client_factory: ClientFactory,
        *,
        trajectory_logger: TrajectoryLogger | None = None,
        domain_context: str | None = None,
        llm_backend: str = "FakeLLMClient",
        run_mode: str = "deterministic/mock",
    ) -> None:
        self.database_path = Path(database_path)
        self.client_factory = client_factory
        self.trajectory_logger = trajectory_logger
        self.domain_context = domain_context
        self.llm_backend = llm_backend
        self.run_mode = run_mode

    def evaluate(
        self,
        cases: Sequence[EvaluationCase],
        *,
        dataset_name: str,
        agent_mode: EvaluationAgentMode | str,
        seed: int = 0,
        tags: Sequence[str] = (),
        limit: int | None = None,
    ) -> EvaluationReport:
        """Evaluate a filtered case set; one case failure never stops later cases."""

        mode = EvaluationAgentMode(agent_mode)
        selected = [
            case
            for case in cases
            if not tags or all(tag in case.tags for tag in tags)
        ]
        Random(seed).shuffle(selected)
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be at least 1")
            selected = selected[:limit]

        if mode is EvaluationAgentMode.BOTH:
            concrete_modes: list[ConcreteMode] = ["pipeline", "function-calling"]
        elif mode is EvaluationAgentMode.PIPELINE:
            concrete_modes = ["pipeline"]
        else:
            concrete_modes = ["function-calling"]
        evaluations: list[CaseEvaluation] = []
        for concrete_mode in concrete_modes:
            for case in selected:
                try:
                    evaluations.append(
                        self._evaluate_case(case, dataset_name, concrete_mode)
                    )
                except Exception as error:
                    # The batch boundary deliberately records an isolated case crash and continues.
                    evaluations.append(
                        self._exception_case(
                            case, dataset_name, concrete_mode, str(error)
                        )
                    )

        metrics: dict[str, ModeMetrics] = {}
        for concrete_mode in concrete_modes:
            mode_cases = [
                case for case in evaluations if case.agent_mode == concrete_mode
            ]
            metrics[concrete_mode] = calculate_mode_metrics(
                mode_cases, include_tool_metrics=concrete_mode == "function-calling"
            )
        comparison = None
        if mode is EvaluationAgentMode.BOTH:
            comparison = compare_modes(
                metrics["pipeline"], metrics["function-calling"]
            )
        return EvaluationReport(
            dataset_name=dataset_name,
            database_id=self.database_path.stem,
            agent_mode=mode,
            llm_backend=self.llm_backend,
            run_mode=self.run_mode,
            seed=seed,
            selected_tags=list(tags),
            cases=evaluations,
            mode_metrics=metrics,
            comparison=comparison,
        )

    def _evaluate_case(
        self,
        case: EvaluationCase,
        dataset_name: str,
        agent_mode: ConcreteMode,
    ) -> CaseEvaluation:
        client = self.client_factory(case, agent_mode)
        max_steps = self._max_steps(case, agent_mode)
        started_at = perf_counter()
        try:
            if agent_mode == "pipeline":
                pipeline_agent = PipelineAgent(
                    self.database_path,
                    SQLGenerator(client),
                    max_steps=max_steps,
                )
                pipeline_result = pipeline_agent.run(case.question)
                duration_ms = (perf_counter() - started_at) * 1000
                evaluation, trajectory = self._pipeline_evaluation(
                    case, dataset_name, pipeline_result, client, duration_ms
                )
            else:
                function_agent = FunctionCallingAgent(
                    self.database_path,
                    client,
                    max_steps=max_steps,
                    llm_backend=type(client).__name__,
                    run_mode=self.run_mode,
                    domain_context=self.domain_context,
                )
                function_result = function_agent.run(case.question)
                duration_ms = (perf_counter() - started_at) * 1000
                evaluation, trajectory = self._function_evaluation(
                    case, dataset_name, function_result, client, duration_ms
                )
        finally:
            if isinstance(client, OpenAICompatibleLLMClient):
                client.close()
        if self.trajectory_logger is not None:
            self.trajectory_logger.log(trajectory)
        return evaluation

    @staticmethod
    def _max_steps(case: EvaluationCase, agent_mode: ConcreteMode) -> int:
        defaults = {"pipeline": 3, "function-calling": 5}
        if isinstance(case, BehaviorScenario):
            return case.max_steps.get(agent_mode, defaults[agent_mode])
        return defaults[agent_mode]

    def _pipeline_evaluation(
        self,
        case: EvaluationCase,
        dataset_name: str,
        result: PipelineAgentResult,
        client: LLMClient,
        duration_ms: float,
    ) -> tuple[CaseEvaluation, Trajectory]:
        executions = [
            step.execution
            for step in result.steps
            if step.execution is not None and step.execution.executed
        ]
        first_success = executions[0].execution_success if executions else None
        result_correct = compare_execution_result(
            result.execution_result, case.expected_result
        )
        message_history = self._client_message_history(client)
        trajectory = TrajectoryLogger.from_pipeline_result(
            result, llm_backend=type(client).__name__, run_mode=self.run_mode
        )
        unsafe_count = sum(
            step.safety is not None and not step.safety.safe for step in result.steps
        )
        failure = self._failure_kind(
            result_correct=result_correct,
            protocol_completed=result.protocol_completed,
            execution_success=result.execution_success,
            termination_reason=result.termination_reason.value,
            first_execution_success=first_success,
            invalid_tool_calls=0,
        )
        evaluation = CaseEvaluation(
            case_id=case.id,
            dataset_name=dataset_name,
            tags=case.tags,
            question=case.question,
            database_id=case.database_id,
            agent_mode="pipeline",
            protocol_completed=result.protocol_completed,
            execution_success=result.execution_success,
            answer_grounded=result.answer_grounded,
            result_correct=result_correct,
            expected_refusal=case.expected_refusal,
            refused=False,
            expected_unsafe_sql=case.expected_unsafe_sql,
            failure_kind=failure,
            termination_reason=result.termination_reason.value,
            final_sql=result.final_sql,
            expected_result=case.expected_result,
            gold_sql=case.gold_sql,
            required_tools=case.required_tools,
            expected_tool_sequences=case.expected_tool_sequences,
            actual_result=result.execution_result,
            first_execution_success=first_success,
            repair_succeeded=(
                result.execution_success if first_success is False else None
            ),
            trajectory_id=trajectory.trajectory_id,
            total_duration_ms=duration_ms,
            total_llm_turns=len(result.steps),
            total_tool_calls=0,
            total_sql_executions=len(executions),
            invalid_tool_calls=0,
            repeated_tool_calls=0,
            unsafe_sql_count=unsafe_count,
            message_history=message_history,
            error_message=self._execution_error_message(result.execution_result),
        )
        trajectory = trajectory.model_copy(
            update={
                "dataset_name": dataset_name,
                "tags": case.tags,
                "result_correct": result_correct,
                "message_history": message_history,
                "unsafe_sql_count": unsafe_count,
            }
        )
        return evaluation, trajectory

    def _function_evaluation(
        self,
        case: EvaluationCase,
        dataset_name: str,
        result: FunctionCallingAgentResult,
        client: LLMClient,
        duration_ms: float,
    ) -> tuple[CaseEvaluation, Trajectory]:
        calls = [call for step in result.steps for call in step.tool_calls]
        validations = [
            validation for step in result.steps for validation in step.validations
        ]
        tool_results = [
            tool_result for step in result.steps for tool_result in step.tool_results
        ]
        observations = [
            observation for step in result.steps for observation in step.tool_messages
        ]
        execute_results: list[ExecutionResult] = []
        for tool_result in tool_results:
            if tool_result.tool_name == "execute_sql":
                execution = ToolRegistry.execution_result(tool_result)
                if execution is not None:
                    execute_results.append(execution)
        executions = [execution for execution in execute_results if execution.executed]
        first_success = executions[0].execution_success if executions else None
        final_execution = execute_results[-1] if execute_results else None
        final_execution_success = bool(
            final_execution is not None and final_execution.execution_success
        )
        result_correct = compare_execution_result(
            final_execution, case.expected_result
        )
        refused = bool(
            result.protocol_completed
            and not execute_results
            and result.final_answer
        )
        invalid_codes = {"unknown_tool", "invalid_json_arguments", "invalid_arguments"}
        invalid_count = sum(
            validation.error_code in invalid_codes for validation in validations
        )
        repeated_count = sum(validation.duplicate for validation in validations)
        unsafe_count = sum(
            tool_result.error_code == "unsafe_sql" for tool_result in tool_results
        )
        failure = self._failure_kind(
            result_correct=result_correct,
            protocol_completed=result.protocol_completed,
            execution_success=final_execution_success,
            termination_reason=result.termination_reason.value,
            first_execution_success=first_success,
            invalid_tool_calls=invalid_count,
        )
        trajectory = TrajectoryLogger.from_function_result(
            result, llm_backend=type(client).__name__, run_mode=self.run_mode
        )
        message_history = trajectory.message_history
        evaluation = CaseEvaluation(
            case_id=case.id,
            dataset_name=dataset_name,
            tags=case.tags,
            question=case.question,
            database_id=case.database_id,
            agent_mode="function-calling",
            protocol_completed=result.protocol_completed,
            execution_success=final_execution_success,
            answer_grounded=result.answer_grounded,
            result_correct=result_correct,
            expected_refusal=case.expected_refusal,
            refused=refused,
            expected_unsafe_sql=case.expected_unsafe_sql,
            failure_kind=failure,
            termination_reason=result.termination_reason.value,
            final_sql=result.final_sql,
            final_answer=result.final_answer,
            expected_result=case.expected_result,
            gold_sql=case.gold_sql,
            required_tools=case.required_tools,
            expected_tool_sequences=case.expected_tool_sequences,
            actual_result=final_execution,
            first_execution_success=first_success,
            repair_succeeded=(
                any(execution.execution_success for execution in executions[1:])
                if first_success is False
                else None
            ),
            trajectory_id=result.trajectory_id,
            total_duration_ms=duration_ms,
            total_llm_turns=result.total_llm_turns,
            total_tool_calls=len(calls),
            total_sql_executions=len(executions),
            invalid_tool_calls=invalid_count,
            repeated_tool_calls=repeated_count,
            unsafe_sql_count=unsafe_count,
            message_history=message_history,
            assistant_tool_calls=calls,
            tool_observations=observations,
            actual_tool_sequence=[call.name for call in calls],
            tool_argument_validity=[
                validation.known_tool and validation.arguments_valid
                for validation in validations
            ],
            tool_execution_successes=[
                tool_result.success
                for tool_result in tool_results
                if tool_result.executed
            ],
            error_message=self._execution_error_message(final_execution),
        )
        trajectory = trajectory.model_copy(
            update={
                "dataset_name": dataset_name,
                "tags": case.tags,
                "result_correct": result_correct,
                "execution_result": final_execution,
                "execution_success": final_execution_success,
                "message_history": message_history,
                "assistant_tool_calls": calls,
                "tool_observations": observations,
                "invalid_tool_calls": invalid_count,
                "repeated_tool_calls": repeated_count,
                "unsafe_sql_count": unsafe_count,
            }
        )
        return evaluation, trajectory

    @staticmethod
    def _client_message_history(client: LLMClient) -> list[LLMMessage]:
        if isinstance(client, FakeLLMClient) and client.requests:
            return list(client.requests[-1].messages)
        return []

    @staticmethod
    def _execution_error_message(result: ExecutionResult | None) -> str | None:
        if result is None:
            return None
        if result.error is not None:
            return result.error.message
        return result.blocked_reason

    @staticmethod
    def _failure_kind(
        *,
        result_correct: bool | None,
        protocol_completed: bool,
        execution_success: bool,
        termination_reason: str,
        first_execution_success: bool | None,
        invalid_tool_calls: int,
    ) -> FailureKind | None:
        if result_correct is False:
            return FailureKind.SEMANTIC_MISMATCH
        termination_map = {
            "unsafe_sql": FailureKind.UNSAFE_SQL,
            "repeated_sql": FailureKind.REPEATED_SQL,
            "repeated_tool_call": FailureKind.REPEATED_TOOL_CALL,
            "max_steps_reached": FailureKind.MAX_STEPS,
            "model_error": FailureKind.MODEL_ERROR,
            "unrecoverable_error": FailureKind.UNRECOVERABLE_ERROR,
        }
        if termination_reason in termination_map:
            return termination_map[termination_reason]
        if protocol_completed and not execution_success:
            return FailureKind.UNGROUNDED_ANSWER
        if first_execution_success is False and not execution_success:
            return FailureKind.EXECUTION_ERROR
        if invalid_tool_calls and not protocol_completed:
            return FailureKind.INVALID_TOOL_CALL
        return None

    def _exception_case(
        self,
        case: EvaluationCase,
        dataset_name: str,
        agent_mode: ConcreteMode,
        error_message: str,
    ) -> CaseEvaluation:
        return CaseEvaluation(
            case_id=case.id,
            dataset_name=dataset_name,
            tags=case.tags,
            question=case.question,
            database_id=case.database_id,
            agent_mode=agent_mode,
            protocol_completed=False,
            execution_success=False,
            answer_grounded=False,
            result_correct=None,
            expected_refusal=case.expected_refusal,
            refused=False,
            expected_unsafe_sql=case.expected_unsafe_sql,
            failure_kind=FailureKind.UNRECOVERABLE_ERROR,
            termination_reason="unrecoverable_error",
            expected_result=case.expected_result,
            gold_sql=case.gold_sql,
            required_tools=case.required_tools,
            expected_tool_sequences=case.expected_tool_sequences,
            trajectory_id=str(uuid4()),
            total_duration_ms=0,
            total_llm_turns=0,
            total_tool_calls=0,
            total_sql_executions=0,
            invalid_tool_calls=0,
            repeated_tool_calls=0,
            unsafe_sql_count=0,
            error_message=error_message,
        )
