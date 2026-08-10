"""FSQ Shanghai multi-turn smoke tests backed by the real read-only database."""

import json
from pathlib import Path

import pytest

from execsql_agent.agents.function_calling import FunctionCallingAgent
from execsql_agent.cli import main
from execsql_agent.config import load_domain_config
from execsql_agent.llm.fake import FakeLLMClient
from execsql_agent.models import LLMResponse, ResponseMode, ToolCallRequest
from execsql_agent.trajectory.logger import TrajectoryLogger

FSQ_DATABASE = Path("data/fsq/shanghai_places.db")
FSQ_CONFIG = Path("config/fsq_shanghai.json")


@pytest.fixture(scope="module")
def fsq_database() -> Path:
    if not FSQ_DATABASE.is_file():
        pytest.fail(
            "FSQ database is missing; run scripts/build_fsq_shanghai_db.py first."
        )
    return FSQ_DATABASE


def _calls(call_id: str, name: str, arguments: dict[str, object]) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=arguments)],
        response_mode=ResponseMode.NATIVE_TOOL_CALLS,
    )


def _final(answer: str) -> LLMResponse:
    return LLMResponse(
        final_answer=answer,
        response_mode=ResponseMode.PLAIN_FINAL,
    )


def _agent(
    database: Path,
    fake: FakeLLMClient,
    logger: TrajectoryLogger,
) -> FunctionCallingAgent:
    domain = load_domain_config(FSQ_CONFIG)
    return FunctionCallingAgent(
        database,
        fake,
        trajectory_logger=logger,
        domain_context=domain.to_prompt(),
    )


def _category_sql(locality_term: str, chinese_term: str) -> str:
    return (
        "SELECT c.category_name, COUNT(DISTINCT p.fsq_place_id) AS place_count "
        "FROM places AS p "
        "JOIN place_categories AS pc ON pc.fsq_place_id = p.fsq_place_id "
        "JOIN categories AS c ON c.category_id = pc.category_id "
        f"WHERE (LOWER(COALESCE(p.locality, '')) LIKE '%{locality_term}%' "
        f"OR COALESCE(p.locality, '') LIKE '%{chinese_term}%') "
        "GROUP BY c.category_id, c.category_name "
        "ORDER BY place_count DESC, c.category_name ASC LIMIT 5"
    )


def _coffee_region_sql(*, require_phone: bool) -> str:
    phone_filter = " AND p.tel IS NOT NULL AND TRIM(p.tel) <> ''" if require_phone else ""
    return (
        "SELECT p.locality, COUNT(DISTINCT p.fsq_place_id) AS coffee_shop_count "
        "FROM places AS p "
        "JOIN place_categories AS pc ON pc.fsq_place_id = p.fsq_place_id "
        "JOIN categories AS c ON c.category_id = pc.category_id "
        "WHERE c.category_name = 'Coffee Shop' "
        "AND p.locality IS NOT NULL AND TRIM(p.locality) <> ''"
        f"{phone_filter} "
        "GROUP BY p.locality "
        "ORDER BY coffee_shop_count DESC, p.locality ASC LIMIT 1"
    )


def _script(sql: str, prefix: str, answer: str) -> list[LLMResponse]:
    return [
        _calls(
            f"{prefix}-schema",
            "inspect_schema",
            {"table_names": ["places", "place_categories", "categories"]},
        ),
        _calls(f"{prefix}-validate", "validate_sql", {"sql": sql}),
        _calls(f"{prefix}-execute", "execute_sql", {"sql": sql}),
        _final(answer),
    ]


def test_pudong_then_xuhui_category_follow_up(
    fsq_database: Path, tmp_path: Path
) -> None:
    pudong_sql = _category_sql("pudong", "浦东")
    xuhui_sql = _category_sql("xuhui", "徐汇")
    fake = FakeLLMClient(
        _script(pudong_sql, "pudong", "已返回浦东前五类。")
        + _script(xuhui_sql, "xuhui", "已切换到徐汇。")
    )
    agent = _agent(
        fsq_database, fake, TrajectoryLogger(tmp_path / "region.jsonl")
    )

    pudong = agent.run("浦东数量最多的前五个地点分类？", session_id="region")
    xuhui = agent.run("那徐汇呢？", session_id="region")

    assert pudong.execution_result is not None
    assert xuhui.execution_result is not None
    assert pudong.execution_result.returned_row_count == 5
    assert xuhui.execution_result.returned_row_count == 5
    assert pudong.execution_result.rows != xuhui.execution_result.rows
    assert [call.name for step in pudong.steps for call in step.tool_calls] == [
        "inspect_schema",
        "validate_sql",
        "execute_sql",
    ]
    follow_up_prompt = fake.requests[4].messages[0].content or ""
    assert "浦东数量最多的前五个地点分类？" in follow_up_prompt
    assert "LOWER(COALESCE(p.locality" in follow_up_prompt
    assert "pudong" in follow_up_prompt
    assert "已返回浦东前五类。" in follow_up_prompt


def test_coffee_region_then_phone_filter(
    fsq_database: Path, tmp_path: Path
) -> None:
    coffee_sql = _coffee_region_sql(require_phone=False)
    phone_sql = _coffee_region_sql(require_phone=True)
    fake = FakeLLMClient(
        _script(coffee_sql, "coffee", "已返回咖啡店最多的区域。")
        + _script(phone_sql, "phone", "已限制为有电话的地点。")
    )
    agent = _agent(
        fsq_database, fake, TrajectoryLogger(tmp_path / "coffee.jsonl")
    )

    coffee = agent.run("上海咖啡店最多的区域？", session_id="coffee")
    phone = agent.run("只看有电话的地点。", session_id="coffee")

    assert coffee.execution_result is not None
    assert phone.execution_result is not None
    assert coffee.execution_result.returned_row_count == 1
    assert phone.execution_result.returned_row_count == 1
    assert phone.final_sql is not None
    assert "p.tel IS NOT NULL" in phone.final_sql
    follow_up_prompt = fake.requests[4].messages[0].content or ""
    assert "上海咖啡店最多的区域？" in follow_up_prompt
    assert "c.category_name = 'Coffee Shop'" in follow_up_prompt
    assert "已返回咖啡店最多的区域。" in follow_up_prompt


def test_fsq_sessions_do_not_share_context(
    fsq_database: Path, tmp_path: Path
) -> None:
    pudong_sql = _category_sql("pudong", "浦东")
    coffee_sql = _coffee_region_sql(require_phone=False)
    fake = FakeLLMClient(
        _script(pudong_sql, "session-a", "会话 A 完成。")
        + _script(coffee_sql, "session-b", "会话 B 完成。")
    )
    agent = _agent(
        fsq_database, fake, TrajectoryLogger(tmp_path / "isolated.jsonl")
    )

    agent.run("浦东数量最多的前五个地点分类？", session_id="session-a")
    agent.run("上海咖啡店最多的区域？", session_id="session-b")

    session_b_prompt = fake.requests[4].messages[0].content or ""
    assert "浦东数量最多的前五个地点分类？" not in session_b_prompt
    assert pudong_sql not in session_b_prompt
    assert [
        turn.question for turn in agent.memory_store.recent("session-a")
    ] == ["浦东数量最多的前五个地点分类？"]
    assert [
        turn.question for turn in agent.memory_store.recent("session-b")
    ] == ["上海咖啡店最多的区域？"]


def test_fsq_cli_loads_domain_config_and_prints_turn_details(
    fsq_database: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sql = _coffee_region_sql(require_phone=False)
    response_file = tmp_path / "responses.json"
    response_file.write_text(
        json.dumps(
            [response.model_dump(mode="json") for response in _script(sql, "cli", "完成。")],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "run",
            "--agent-mode",
            "function-calling",
            "--database",
            str(fsq_database),
            "--domain-config",
            str(FSQ_CONFIG),
            "--question",
            "上海咖啡店最多的区域？",
            "--session-id",
            "cli-fsq",
            "--fake-responses",
            str(response_file),
            "--trajectory-file",
            str(tmp_path / "cli.jsonl"),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "工具调用：inspect_schema" in output
    assert "工具调用：validate_sql" in output
    assert "工具调用：execute_sql" in output
    assert "SQL：SELECT p.locality" in output
    assert "执行结果：" in output
    assert "Memory 摘要：" in output
