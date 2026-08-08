"""Strict allowlist and real SQLite tests for ToolRegistry."""

from pathlib import Path

from execsql_agent.models import ToolCallRequest
from execsql_agent.tools.registry import ToolRegistry


def test_exports_exactly_three_strict_json_schemas(demo_db: Path) -> None:
    definitions = ToolRegistry(demo_db).definitions

    assert [definition.name for definition in definitions] == [
        "inspect_schema",
        "validate_sql",
        "execute_sql",
    ]
    assert all(definition.parameters["additionalProperties"] is False for definition in definitions)
    execute_schema = definitions[2].parameters
    assert execute_schema["required"] == ["sql"]


def test_unknown_tool_is_never_executed(demo_db: Path) -> None:
    validation, result = ToolRegistry(demo_db).dispatch(
        ToolCallRequest(id="bad", name="drop_database", arguments={})
    )

    assert validation.known_tool is False
    assert result.success is False
    assert result.executed is False
    assert result.error_code == "unknown_tool"


def test_invalid_json_arguments_are_never_executed(demo_db: Path) -> None:
    validation, result = ToolRegistry(demo_db).dispatch(
        ToolCallRequest(
            id="bad-json",
            name="execute_sql",
            arguments=None,
            raw_arguments="{bad",
        )
    )

    assert validation.arguments_valid is False
    assert result.executed is False
    assert result.error_code == "invalid_json_arguments"


def test_missing_type_and_extra_arguments_are_rejected(demo_db: Path) -> None:
    registry = ToolRegistry(demo_db)
    calls = [
        ToolCallRequest(id="missing", name="execute_sql", arguments={}),
        ToolCallRequest(id="type", name="execute_sql", arguments={"sql": 42}),
        ToolCallRequest(
            id="extra",
            name="execute_sql",
            arguments={"sql": "SELECT 1", "unexpected": True},
        ),
    ]

    results = [registry.dispatch(call)[1] for call in calls]

    assert all(result.executed is False for result in results)
    assert all(result.error_code == "invalid_arguments" for result in results)


def test_inspect_validate_and_execute_use_real_database(demo_db: Path) -> None:
    registry = ToolRegistry(demo_db)
    _, schema_result = registry.dispatch(
        ToolCallRequest(
            id="schema",
            name="inspect_schema",
            arguments={"table_names": ["customers"]},
        )
    )
    _, validation_result = registry.dispatch(
        ToolCallRequest(
            id="validate",
            name="validate_sql",
            arguments={"sql": "SELECT customer_id FROM customers"},
        )
    )
    _, execution_result = registry.dispatch(
        ToolCallRequest(
            id="execute",
            name="execute_sql",
            arguments={"sql": "SELECT COUNT(*) AS count FROM customers"},
        )
    )

    assert schema_result.success is True
    assert schema_result.output is not None
    assert schema_result.output["tables"][0]["name"] == "customers"
    assert validation_result.success is True
    assert execution_result.success is True
    typed_execution = registry.execution_result(execution_result)
    assert typed_execution is not None
    assert typed_execution.rows == [[10]]


def test_unsafe_sql_never_reaches_sqlite(demo_db: Path) -> None:
    registry = ToolRegistry(demo_db)

    _, validation = registry.dispatch(
        ToolCallRequest(
            id="validate", name="validate_sql", arguments={"sql": "DELETE FROM customers"}
        )
    )
    _, execution = registry.dispatch(
        ToolCallRequest(
            id="execute", name="execute_sql", arguments={"sql": "DELETE FROM customers"}
        )
    )

    assert validation.error_code == "unsafe_sql"
    assert validation.executed is False
    assert execution.error_code == "unsafe_sql"
    assert execution.executed is False
