# Repository Guidelines

## Scope and Structure

`PLAN.md` is the implementation contract. The MVP retains a Pipeline baseline and a native `FunctionCallingAgent`. Both share schema inspection, SQL validation, read-only execution, logging, and evaluation. Code belongs in `src/execsql_agent/`, tests in `tests/`, utilities in `scripts/`, defaults in `configs/`, and evaluation inputs under `data/`. Do not commit generated databases, reports, trajectories, credentials, or local environment files.

Do not add LangChain, LangGraph, multi-agent orchestration, complex planners, training pipelines, web APIs, Docker, or databases other than SQLite during the MVP.

## Architecture Boundaries

Keep Pipeline and Function Calling orchestration separate while reusing safe tools. Business modules depend on `LLMClient`, never a provider SDK. `ToolRegistry` exposes only `inspect_schema`, `validate_sql`, and `execute_sql`; validate names and Pydantic arguments before dispatch. Unknown tools and invalid arguments are observations, never executions. Tool outputs come from real Python and SQLite, including FakeLLM tests.

The Function Calling loop accepts native `tools`/`tool_calls` and a structured JSON fallback, supports multiple calls per model turn, feeds `ToolMessage` observations back to the model, and detects repeated calls and normalized SQL. Pipeline remains the stable baseline. Never collapse completion into one `success` flag: record `protocol_completed`, `execution_success`, `answer_grounded`, and nullable `result_correct` separately.

## Development Commands

Use the current verification workflow:

On Windows, run Python tooling through `.venv\Scripts\python.exe`; do not install project
dependencies into Conda or the system interpreter. Rebuild the FSQ Shanghai database with:

```powershell
.\.venv\Scripts\python.exe scripts\build_fsq_shanghai_db.py --places data\fsq\processed\shanghai_places_filtered.parquet --categories data\fsq\raw\release\dt=2026-07-09\categories\parquet\categories_000000.parquet --output data\fsq\shanghai_places.db
```

```bash
python -m pip install -e ".[dev]"
python scripts/create_demo_database.py
python -m pytest
python -m ruff check .
python -m mypy src
python -m execsql_agent.cli run --agent-mode pipeline --database data/demo.db --question "消费金额最高的五名客户是谁？"
python -m execsql_agent.cli run --agent-mode function-calling --database data/demo.db --question "消费金额最高的五名客户是谁？"
python -m execsql_agent.cli evaluate --agent-mode both --database data/demo.db --dataset data/synthetic/eval_questions.json --output-dir data/reports/synthetic
```

Update this guide and `README.md` whenever commands change.

## Style, Safety, and Testing

Use Python 3.11, four-space indentation, complete type annotations, focused docstrings, Pydantic v2 boundary models, Ruff, and mypy. Avoid broad exception handlers and oversized modules.

Open SQLite read-only, enable `query_only`, use an authorizer, reject multiple statements, and intercept unsafe SQL before execution. Allow one unexecuted repair after unsafe SQL; terminate on the next unsafe candidate. Record real elapsed time and use `sqlite3.set_progress_handler` only as a best-effort guard.

Use pytest, disposable SQLite databases, and FakeLLM—never paid APIs or production data. `ExecutionResult.returned_row_count` means only rows returned. Cover native and fallback calls, invalid tools and arguments, unsafe and repeated calls, direct answers, and cross-mode metrics. Primary tool metrics are exact sequence match, required-tool coverage, and invalid-call rate. Fake reports must say `deterministic/mock`.

## Commits and Pull Requests

Use Conventional Commits, for example `feat: add tool registry`. Pull requests should state scope, security impact, verification commands, agent modes tested, and report evidence when metrics change.
