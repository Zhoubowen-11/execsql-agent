"""JSONL trajectory round-trip tests."""

from pathlib import Path

from execsql_agent.agents.function_calling import FunctionCallingAgent
from execsql_agent.llm.fake import FakeLLMClient
from execsql_agent.models import (
    LLMResponse,
    ResponseMode,
    ToolCallRequest,
    Trajectory,
)
from execsql_agent.trajectory.logger import TrajectoryLogger


def test_function_trajectory_round_trips_through_jsonl(
    demo_db: Path, tmp_path: Path
) -> None:
    path = tmp_path / "trajectories.jsonl"
    fake = FakeLLMClient(
        [
            LLMResponse(
                tool_calls=[
                    ToolCallRequest(
                        id="execute",
                        name="execute_sql",
                        arguments={"sql": "SELECT COUNT(*) FROM customers"},
                    )
                ],
                response_mode=ResponseMode.NATIVE_TOOL_CALLS,
            ),
            LLMResponse(
                final_answer="共有 10 名客户。",
                response_mode=ResponseMode.PLAIN_FINAL,
            ),
        ]
    )
    logger = TrajectoryLogger(path)

    result = FunctionCallingAgent(
        demo_db, fake, trajectory_logger=logger
    ).run("客户数量")

    raw_line = path.read_text(encoding="utf-8").strip()
    trajectory = Trajectory.model_validate_json(raw_line)
    assert trajectory.trajectory_id == result.trajectory_id
    assert trajectory.run_mode == "deterministic/mock"
    assert trajectory.steps[0].step_type == "function_calling"
    assert trajectory.protocol_completed is True
    assert "API_KEY" not in raw_line
    assert result.trajectory_path == str(path.resolve())
