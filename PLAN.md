# ExecSQL-Agent MVP 实施计划

## 1. 项目定位

在 3～5 天内实现一个 Python 3.11 Text-to-SQL 自纠错 Agent MVP。系统读取 SQLite Schema，生成或通过工具调用构造 SQL，进行安全校验和真实执行，在失败时利用 Observation 修复，并保存可用于自动评测和后续训练的数据轨迹。

MVP 同时保留两种实现模式：

1. **PipelineAgent**：固定流程、默认 `max_steps=3`，作为稳定、可复现、便于定位问题的 Baseline。
2. **FunctionCallingAgent**：原生最小 Function Calling 循环、默认 `max_steps=5`，由模型选择工具、接收工具 Observation 并决定继续调用或输出最终答案。

两种模式共享 Schema、SQL 安全、SQLite 执行、LLM 抽象、轨迹和评测基础设施。MVP 展示 Function Calling、Tool Use、Agent Loop、短期 Memory、Reflection、Replanning、错误恢复和自动评测，但不引入 LangChain、LangGraph、多智能体、复杂 Planner、训练流程、Web 前端、FastAPI、Docker、云部署或 SQLite 之外的数据库。

## 2. 已确认的设计原则

1. Pipeline Baseline 不移除、不改造成隐式 Function Calling；两种 Agent 使用独立编排器。
2. FunctionCallingAgent 只暴露 `inspect_schema`、`validate_sql`、`execute_sql` 三个工具。
3. 工具使用 JSON Schema 描述；所有工具名称和参数必须在执行前校验，未知工具或非法参数绝不分发。
4. 模型每轮可返回一个或多个 `tool_calls`；工具结果转换为 `ToolMessage` 后回填上下文。
5. OpenAI-compatible 客户端同时支持原生 `tools`/`tool_calls` 和结构化 JSON fallback，两种协议归一为同一个 `LLMResponse`。
6. FakeLLM 可模拟多轮、多工具调用和最终回答，但工具结果必须来自真实 Python 工具与真实 SQLite。
7. 评测样本必须包含 `expected_result`，`gold_sql` 可选。正确性以结果比较为主，不要求 SQL 文本一致。
8. 空结果默认执行成功；只有 `empty_result_repair=true` 时 Pipelin.\.venv\Scripts\python.exe -m execsql_agent.cli evaluate `
  --agent-mode function-calling `
  --database data\fsq\shanghai_places.db `
  --dataset data\fsq\eval\questions.json `
  --domain-config config\fsq_shanghai.json `
  --output-dir data\fsq\reports\real_model_smoke_v3 `
  --limit 5 `
  --real-modele 才将其作为可修复 Observation。评测器始终依据 `expected_result` 判断空结果是否正确。
9. unsafe SQL 必须在执行前拦截。观察到首个 unsafe SQL 后允许一次未执行修复；下一 SQL 候选仍不安全则以 `unsafe_sql` 终止。
10. `execution_error` 是 SQLite 明确执行错误；`semantic_mismatch` 是执行成功但结果与 `expected_result` 不一致，只由评测器判定。
11. `incorrect_join` 和 `aggregation_error` 不能声称可由规则可靠识别；规则不能确定时才由模型作可能性诊断。
12. Fake 轨迹、CLI 和报告必须标记 `deterministic/mock`，不得表示真实模型泛化能力。
13. 不承诺严格毫秒级 SQL 超时；记录真实耗时，可通过 `sqlite3.set_progress_handler` 防止明显长查询。
14. `data/demo.db` 不提交，只提交确定性创建脚本。README 和 CLI 中文为主，代码、结构化字段和日志字段保持英文。

## 3. 最终目录结构

```text
execsql-agent/
├── AGENTS.md
├── README.md
├── PLAN.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── configs/
│   └── default.yaml
├── data/
│   ├── synthetic/
│   │   ├── eval_questions.json
│   │   └── behavior_scenarios.json
│   ├── trajectories/
│   └── reports/
├── scripts/
│   └── create_demo_database.py
├── src/
│   └── execsql_agent/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── models.py
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── pipeline.py
│       │   └── function_calling.py
│       ├── llm/
│       │   ├── base.py
│       │   ├── fake.py
│       │   └── openai_compatible.py
│       ├── tools/
│       │   ├── registry.py
│       │   ├── schema_loader.py
│       │   ├── sql_validator.py
│       │   └── sql_executor.py
│       ├── generation/
│       │   └── sql_generator.py
│       ├── diagnosis/
│       │   └── error_diagnoser.py
│       ├── trajectory/
│       │   └── logger.py
│       └── evaluation/
│           └── evaluator.py
└── tests/
    ├── conftest.py
    ├── test_schema_loader.py
    ├── test_sql_validator.py
    ├── test_sql_executor.py
    ├── test_tool_registry.py
    ├── test_llm_clients.py
    ├── test_sql_generator.py
    ├── test_error_diagnoser.py
    ├── test_pipeline_agent.py
    ├── test_function_calling_agent.py
    ├── test_trajectory_logger.py
    ├── test_evaluator.py
    └── test_cli.py
```

`agents/` 明确隔离两种编排方式；`tools/registry.py` 只负责工具定义、参数校验和安全分发，不包含 Agent 决策；`sql_validator.py` 提供两种 Agent 共用的标准化和只读安全校验。

## 4. 技术与工程约定

运行依赖控制为 `pydantic`、`pydantic-settings`、`httpx` 和 `PyYAML`；CLI、SQLite、JSONL 和 CSV 优先使用标准库。开发依赖为 `pytest`、`pytest-cov`、`ruff` 和 `mypy`。项目使用 src-layout、Pydantic v2、完整类型注解、清晰 docstring 和职责单一的小型模块。

OpenAI-compatible 默认配置：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- endpoint：`/v1/chat/completions`

业务模块只依赖 `LLMClient` 抽象，不直接依赖供应商 SDK。

## 5. 模块职责

### 5.1 SchemaLoader

通过 `sqlite_master`、`PRAGMA table_info` 和 `PRAGMA foreign_key_list` 获取表、字段类型、主键和外键，输出结构化 Schema 和紧凑 Schema 文本。Schema 读取不得修改数据库。

### 5.2 SQLValidator

提供单语句检查、SQL 标准化和只读安全分类。拦截 INSERT、UPDATE、DELETE、DROP、ALTER、CREATE、REPLACE、ATTACH、DETACH、VACUUM 及危险 PRAGMA。必要时在只读连接中用 `EXPLAIN QUERY PLAN` 做尽力而为的编译验证，但不得执行查询或写入数据。

标准化只去除注释和末尾分号、压缩空白并统一大小写，不承诺 AST 级语义等价。关键词检查不是唯一安全边界。

### 5.3 SQLExecutor

使用 SQLite URI `mode=ro`、`PRAGMA query_only=ON` 和 authorizer 建立多层只读防护。`execute_sql` 内部必须再次调用安全校验，不能信任模型已调用 `validate_sql`。执行结果记录列、受限行数据、截断标记、原始 SQLite 错误和 `perf_counter` 实测耗时。

`sqlite3.set_progress_handler` 使用可配置指令预算提供尽力而为的长查询保护，不宣称严格毫秒级超时。空结果默认 `execution_success=true`。`returned_row_count` 只记录实际交付给调用方的行数；当 `truncated=true` 时不推断完整查询总行数。

### 5.4 三个公开工具

MVP 的 ToolRegistry 只注册以下工具：

| 工具 | 参数 | 行为 |
|---|---|---|
| `inspect_schema` | 可选 `table_names: list[str]` | 返回完整或指定表的真实结构化 Schema |
| `validate_sql` | `sql: str` | 标准化、安全检查并尽力编译验证，不执行查询 |
| `execute_sql` | `sql: str` | 重新执行完整安全检查后，在真实只读 SQLite 上查询 |

每个工具都由 Pydantic 参数模型导出 JSON Schema。额外字段默认禁止。ToolRegistry 按 allowlist 查找名称、解析 JSON 参数、执行模型校验，再进行分发。未知工具、JSON 解析失败或参数不合法时返回 `ToolCallResult(executed=false)`，并作为 Observation 反馈模型。

### 5.5 ToolRegistry

ToolRegistry 负责：

- 导出 `ToolDefinition` 列表；
- 校验调用 ID、工具名称和参数；
- 将已验证参数传给明确注册的 Python callable；
- 捕获并结构化工具边界错误，但不使用宽泛异常隐藏真实问题；
- 记录实际分发状态、执行结果和耗时；
- 保证任何未注册或未通过校验的调用不执行。

多个 tool_calls 按模型返回顺序串行处理；本阶段不并发执行。当前轮全部结果分别生成 ToolMessage，再统一回填下一轮上下文。

### 5.6 PipelineAgent Baseline

PipelineAgent 保留固定流程：加载 Schema → SQLGenerator 初次生成 → 安全验证 → SQLExecutor → ErrorDiagnoser → repair。默认 `max_steps=3`，保存历史 SQL、原始错误、诊断和修复建议。

首次 unsafe SQL 不执行，可进入一次修复；修复仍 unsafe 则终止。安全 SQL 标准化后若已执行过，不再重复执行。空结果默认成功；配置 `empty_result_repair=true` 时才允许诊断 `empty_result`。

### 5.7 FunctionCallingAgent

FunctionCallingAgent 默认 `max_steps=5`，其中 step 表示一次 LLM turn，而不是单个工具调用。流程如下：

1. 将用户问题、系统约束、历史消息和三个 `ToolDefinition` 发送给 LLM。
2. 将原生或 fallback 响应归一为 `LLMResponse`。
3. 若响应包含一个或多个 tool_calls，逐个交给 ToolRegistry 校验和分发。
4. 将每个 `ToolCallResult` 编码为对应 `ToolMessage`，按 `tool_call_id` 回填上下文。
5. 模型基于 Observation 继续调用工具或输出最终答案。
6. 达到 `max_steps`、重复调用、重复 SQL、连续 unsafe、模型错误或不可恢复错误时明确终止。

模型允许直接输出最终答案，协议层将正常结束并标记 `completion_kind=direct_answer`。若此前没有成功的 `execute_sql`，则必须记录 `protocol_completed=true`、`execution_success=false`、`answer_grounded=false`、`result_correct=null`；结果型评测不得把该回答当作已验证的数据库正确答案。

### 5.8 重复与 unsafe 检测

- **重复工具调用**：签名为 `tool_name + canonical_json(arguments)`。同一 LLM turn 内的完全相同调用直接拒绝；跨 turn 在没有产生新状态信息时再次调用也拒绝。
- **允许重新 inspect**：在新的执行错误 Observation 之后再次调用 `inspect_schema` 属于新的上下文，不判为重复，支持“报错后重新检查 Schema 并修复”。
- **重复 SQL**：只对 `execute_sql` 的标准化 SQL 做全轨迹去重；`validate_sql(sql=X)` 后首次 `execute_sql(sql=X)` 是正常流程，不算重复执行。
- **重复无效调用**：相同未知工具或相同非法参数再次出现时，以 `repeated_tool_call` 终止。
- **unsafe SQL**：无论来自 Pipeline、`validate_sql` 或 `execute_sql` 请求，都不进入数据库执行；首次反馈修复机会，下一 SQL 候选仍 unsafe 时以 `unsafe_sql` 终止。

### 5.9 LLMClient 与协议回退

真实客户端优先发送 OpenAI-compatible `tools`，解析 assistant message 中的 `tool_calls`、调用 ID、函数名和 JSON arguments。模型也可返回普通最终内容。

若服务明确不支持 tools/tool_calls，则切换到结构化 JSON fallback，在 Prompt 中加入等价工具 Schema，并要求以下两类响应之一：

```json
{"type":"tool_calls","tool_calls":[{"id":"call_1","name":"inspect_schema","arguments":{}}]}
```

```json
{"type":"final","final_answer":"..."}
```

若服务支持 tools 但不支持 `response_format`，客户端保留原生 tools，仅移除 `response_format`，再进行普通文本 JSON 解析。所有 fallback 最终都校验为 `LLMResponse`；无法解析时返回结构化 `LLMError`，不得猜测或执行不完整调用。

### 5.10 FakeLLMClient

FakeLLMClient 提供按 LLM turn 排列的响应队列，可模拟单个或多个 tool_calls、执行错误后的再次 inspect、修复调用、无效调用和最终答案。Fake 只产生 `LLMResponse`，ToolRegistry、SchemaLoader、SQLValidator 和 SQLExecutor 始终运行真实 Python 逻辑并访问真实 Demo SQLite。

### 5.11 ErrorDiagnoser

Pipeline 优先用规则识别明确的 syntax、missing table、missing column 和 ambiguous column 错误，规则不能处理时才调用 LLM。Function Calling 模式主要把结构化工具错误作为 Observation 交给模型，可附加规则诊断信息，但不替模型规划工具顺序。`incorrect_join` 和 `aggregation_error` 只能作为模型或评测分析的可能诊断。

### 5.12 TrajectoryLogger

JSONL 每行保存一个完整任务轨迹，顶层包含 `schema_version`、`agent_mode`、`llm_backend` 和 `run_mode`。Pipeline step 记录生成、验证、执行和诊断；Function Calling step 记录 LLMResponse、每个 ToolCallRequest、校验状态、ToolCallResult、ToolMessage、是否执行和耗时。

Fake 模式固定写入 `run_mode: deterministic/mock`。轨迹保留后续 tool-calling SFT、偏好数据和执行奖励所需字段，但本阶段不实现训练转换。

### 5.13 Evaluator

评测器支持 `pipeline`、`function-calling` 和 `both`。对比模式必须在相同数据库快照、相同问题和相同结果比较配置上分别运行两种 Agent。`expected_result` 和 `gold_sql` 不得进入 Agent Prompt。

推荐样本格式：

```json
{
  "id": "top_customers_001",
  "question": "消费金额最高的五名客户是谁？",
  "expected_result": {
    "columns": ["customer_name", "total_amount"],
    "rows": [["张三", 1200.0]],
    "ordered": true
  },
  "gold_sql": null,
  "expected_tool_sequences": [
    ["inspect_schema", "validate_sql", "execute_sql"],
    ["inspect_schema", "execute_sql"]
  ],
  "tags": ["aggregation", "limit"]
}
```

`expected_tool_sequences` 仅用于 Function Calling 的 `tool_sequence_exact_match_rate`，可提供多个可接受路径；`required_tools` 用于计算 `required_tool_coverage`。没有相应标注的案例不进入对应指标分母，报告必须显示有效样本数，不能推测期望工具路径。

结果比较规则固定：`ordered=true` 时按顺序比较，否则按多重集合比较；NULL 精确匹配；数字使用可配置容差；列名默认严格比较；空 rows 是合法期望结果。JSON、CSV 和 Markdown 必须从同一个 `EvaluationReport` 生成。

## 6. 核心 Pydantic 数据结构

### 6.1 Schema、SQL 与诊断

- `ColumnSchema`: `name`, `data_type`, `nullable`, `default_value`, `primary_key`
- `ForeignKeySchema`: `source_table`, `source_column`, `target_table`, `target_column`
- `TableSchema`: `name`, `columns`, `primary_keys`, `foreign_keys`
- `DatabaseSchema`: `database_id`, `tables`, `summary_text`
- `GenerationMode`: `initial_generation | repair`
- `SQLGeneration`: `sql`, `reason`, `referenced_tables`, `referenced_columns`
- `SafetyCheckResult`: `safe`, `normalized_sql`, `reason`, `blocked_operation`
- `ExecutionError`: `message`, `sqlite_error_name`, `sqlite_error_code`
- `ExecutionResult`: `executed`, `execution_success`, `columns`, `rows`, `returned_row_count`, `truncated`, `error`, `duration_ms`
- `DiagnosisSource`: `rule | llm`
- `ErrorDiagnosis`: `error_type`, `cause`, `repair_instruction`, `related_tables`, `related_columns`, `source`
- `FailureKind`: `execution_error | semantic_mismatch`

### 6.2 Function Calling 协议

- `ToolDefinition`: `name`, `description`, `parameters`（JSON Schema）
- `ToolCallRequest`: `id`, `name`, `arguments`, `raw_arguments`
- `ToolCallResult`: `tool_call_id`, `tool_name`, `tool_success`, `executed`, `output`, `error_code`, `error_message`, `duration_ms`
- `LLMResponse`: `final_answer`, `tool_calls`, `finish_reason`, `response_mode`, `raw_content`
- `ToolMessage`: `role="tool"`, `tool_call_id`, `name`, `content`
- `ResponseMode`: `native_tool_calls | json_fallback | plain_final`

`LLMResponse` 必须满足“有 final_answer”或“有至少一个 tool_call”之一，禁止两者皆空；若供应商同时返回两者，优先完成所有 tool_calls，最终答案留待下一轮确认，避免忽略工具结果。

### 6.3 Agent 与轨迹

- `ToolCallValidation`: 名称合法性、参数合法性、重复状态和错误信息
- `PipelineStep`: `step_type="pipeline"`、生成、验证、执行、诊断和 Observation
- `FunctionCallingStep`: `step_type="function_calling"`、LLM turn、LLMResponse、调用验证、ToolCallResult 和 ToolMessage
- `TerminationReason`: `completed`, `max_steps_reached`, `repeated_sql`, `repeated_tool_call`, `unsafe_sql`, `invalid_tool_call`, `model_error`, `unrecoverable_error`
- `CompletionKind`: `tool_grounded_answer | direct_answer | terminated`
- `Trajectory`: `schema_version`, `trajectory_id`, `agent_mode`, `question`, `database_id`, `schema_summary`, `llm_backend`, `run_mode`, `steps`, `final_sql`, `final_answer`, `protocol_completed`, `execution_success`, `answer_grounded`, `result_correct`, `termination_reason`, `total_steps`, `tool_call_count`, `execution_attempts`, `total_duration_ms`
- `AgentResult`: 最终答案、最终 SQL、真实执行结果、grounded 状态、终止状态和轨迹 ID

PipelineStep 与 FunctionCallingStep 使用 Pydantic discriminated union 存入统一 Trajectory，避免丢失任一模式特有字段。

### 6.4 评测

- `ExpectedResult`: `columns`, `rows`, `ordered` 和可选比较配置
- `EvaluationCase`: `id`, `question`, `expected_result`, 可选 `gold_sql`, `expected_tool_sequences`, `required_tools`, `tags`
- `CaseEvaluation`: Agent 模式、执行状态、结果正确性、FailureKind、调用统计和轨迹 ID
- `ToolMetrics`: 工具选择、参数合法、调用成功、无效与重复统计
- `ModeMetrics`: 单一 Agent 模式的执行与结果指标
- `ModeComparison`: Pipeline 与 Function Calling 的同集差值及逐案例对比
- `EvaluationMetrics`: 模式指标、错误类型指标和 ToolMetrics
- `EvaluationReport`: 后端标记、运行配置、指标和失败案例
- `LLMError`: `code`, `message`, `retryable`, `attempts`

`semantic_mismatch` 只存在于 CaseEvaluation，不得写成 SQLite ExecutionError。

## 7. 配置设计

`configs/default.yaml` 保存非秘密默认值：

- `pipeline.max_steps: 3`
- `function_calling.max_steps: 5`
- `agent.empty_result_repair: false`
- `executor.max_rows`
- `executor.progress_handler_ops`
- `llm.request_timeout_seconds`
- `llm.max_retries`
- `llm.function_calling_mode: auto`
- `llm.response_format_mode: auto`
- `evaluation.numeric_tolerance`
- `evaluation.strict_columns: true`

CLI 的 `run --agent-mode` 接受 `pipeline` 或 `function-calling`；`evaluate --agent-mode` 额外接受 `both`。API Key、Base URL 和模型名只从指定环境变量或显式 CLI 覆盖读取。

## 8. Agent 终止与成功语义

Pipeline 的 step 是 SQL 候选次数；Function Calling 的 step 是 LLM turn。两者分别使用自己的默认上限。

Function Calling 遇到未知工具或非法参数时，首次返回未执行的错误 ToolMessage，允许模型在剩余 step 内纠正；相同无效调用再次出现时以 `repeated_tool_call` 终止。若耗尽 step 则以 `max_steps_reached` 终止。单次无效调用不会伪装为工具执行失败。

Function Calling 和轨迹不使用单一 `success` 概括不同层面的完成状态，而是始终分别记录：

- `protocol_completed`：模型按协议输出最终答案并正常结束；
- `execution_success`：存在成功的真实 `execute_sql`；
- `answer_grounded`：最终答案建立在成功执行 Observation 上；
- `result_correct`：真实结果匹配 `expected_result`，未执行 SQL 时为 `null`；
- `direct_answer`：模型未调用执行工具直接回答，可正常完成但结果正确性记为不可验证，不计入结果正确案例。

因此直接回答固定为 `protocol_completed=true`、`execution_success=false`、`answer_grounded=false`、`result_correct=null`。Pipeline 轨迹也使用同一组字段，保证跨模式报告不存在含糊的单一成功状态。

模式对比以最终执行成功率和结果正确率为主要指标，不用“模型输出了文本”代替数据库任务成功。

## 9. 评测指标与口径

### 9.1 通用指标

- 总问题数。
- 首次执行成功率：第一次实际 SQL 执行无 SQLite 错误的案例数 / 总案例数。
- 最终执行成功率：最终获得 SQLite 成功结果的案例数 / 总案例数。
- 结果正确率：最终真实结果匹配 `expected_result` 的案例数 / 总案例数。
- 错误修复率：首次实际执行为 execution_error 且后来执行成功的案例数 / 首次执行错误案例数。
- 平均 SQL 候选数和平均真实执行次数。
- 重复 SQL 率、非法 SQL 率、错误类型数量、分错误类型修复率。
- semantic mismatch 数量及失败案例。

### 9.2 Function Calling 主要工具选择指标

- **tool_sequence_exact_match_rate**：实际工具名称序列与任一 `expected_tool_sequences` 完全一致的案例数 / 有期望序列标注的案例数。不再使用位置匹配数除以较长序列长度的近似算法。
- **required_tool_coverage**：每个案例实际调用到的必需工具数 / `required_tools` 数，再对有必需工具标注的案例取平均；重复调用不增加覆盖率。
- **invalid_tool_call_rate**：未知工具、arguments JSON 解析失败或参数校验失败的调用数 / 所有请求调用数。

参数合法率、工具调用成功率、平均工具调用次数和重复工具调用率可作为诊断性原始统计保留，但不替代上述三个主要工具选择指标。直接回答率与 grounded 回答率用于解释协议完成情况，不替代结果正确率。任一指标分母为零时输出 `null` 并报告有效样本数。

### 9.3 模式对比

在同一评测子集上输出 Pipeline 与 Function Calling 的最终执行成功率、结果正确率、错误修复率、平均执行次数和平均耗时，并提供绝对差值。Fake 对比报告必须明显标记 `deterministic/mock`，只说明流程与评测系统可复现，不推断真实模型能力。

若某指标分母为零，输出 `null` 和分母数，不输出伪造的 0% 或 100%。所有 JSON、CSV、Markdown 报告从同一个内存 EvaluationReport 生成。

## 10. 实施里程碑

### 里程碑 A：工程基础、Demo DB 与共享安全工具（已完成）

涉及文件：

- `pyproject.toml`, `.gitignore`, `.env.example`, `configs/default.yaml`
- `src/execsql_agent/config.py`, `models.py`
- `tools/schema_loader.py`, `tools/sql_validator.py`, `tools/sql_executor.py`
- `scripts/create_demo_database.py`
- `tests/conftest.py`, `test_schema_loader.py`, `test_sql_validator.py`, `test_sql_executor.py`

验收：项目可安装；Demo DB 可确定性重建；Schema 主外键正确；只读查询、空结果和行数截断正常；写操作、多语句、危险操作在执行前被阻止；记录真实耗时；progress handler 可停止构造的长查询。

### 里程碑 B：LLM 协议、Fake、多轮响应与 Pipeline Baseline（已完成）

涉及文件：

- `llm/base.py`, `llm/fake.py`, `llm/openai_compatible.py`
- `generation/sql_generator.py`, `diagnosis/error_diagnoser.py`
- `agents/pipeline.py`
- `tests/test_llm_clients.py`, `test_sql_generator.py`, `test_error_diagnoser.py`, `test_pipeline_agent.py`

验收：Pipeline 首次成功和修复流程可运行；Fake 完全离线；LLMResponse 可表达 final 与多个 tool_calls；原生 tools、无 response_format 和完整 JSON fallback 均有测试；模型错误结构化返回。

### 里程碑 C：ToolRegistry、FunctionCallingAgent、轨迹与 CLI（已完成）

涉及文件：

- `tools/registry.py`
- `agents/function_calling.py`
- `trajectory/logger.py`
- `cli.py`
- `tests/test_tool_registry.py`, `test_function_calling_agent.py`, `test_trajectory_logger.py`, `test_cli.py`

验收：三个工具严格 allowlist；未知工具和非法参数不执行；多轮 ToolMessage 正确关联 call ID；多工具调用顺序稳定；重复调用、重复 SQL、unsafe 修复和 max_steps 正确终止；两种 Agent 均可由 CLI 运行。

### 里程碑 D：批量评测、指标与模式对比（已实现，待审查）

涉及文件：

- `data/synthetic/eval_questions.json`, `data/synthetic/behavior_scenarios.json`
- `evaluation/comparator.py`, `evaluation/metrics.py`, `evaluation/evaluator.py`, `evaluation/reports.py`, `evaluation/synthetic.py`
- `tests/test_evaluator.py`
- `cli.py`

验收：至少 20 条真实执行问题；expected_result 比较覆盖有序、无序、空集、NULL 和数字容差；工具指标可从原始调用复算；Pipeline 与 Function Calling 在相同案例上对比；JSON、CSV 和 Markdown 内容一致；Fake 标识清晰。

### 里程碑 E：文档、边界加固与演示（未开始）

涉及文件：

- `README.md`
- 全部配置、测试和既有模块的必要修正

验收：pytest、Ruff、mypy 全部通过；按 README 从空环境安装；创建数据库；分别运行 Pipeline 和 Function Calling；演示失败后修复、多轮工具调用及模式对比报告。

## 11. 必须覆盖的测试场景

### 11.1 共享工具与 Pipeline

- Schema 表、字段、主键和外键读取。
- SELECT 成功、空结果成功、行数截断和 execution_error。
- 写操作、危险 PRAGMA、多语句、ATTACH 与 unsafe SQL 拦截。
- Pipeline 首次成功、missing table 修复、重复 SQL、连续 unsafe、模型错误和 max_steps。
- `empty_result_repair` 开启与关闭。

### 11.2 Function Calling

1. `inspect_schema → execute_sql → final answer`。
2. SQL 报错后再次 `inspect_schema`，生成修复 SQL并成功执行。
3. 未知工具返回 `executed=false`，Python callable 未被调用。
4. 缺字段、额外字段或错误类型参数不执行。
5. unsafe SQL 在 validate 或 execute 前被拦截，允许一次修复，连续 unsafe 终止。
6. 同轮和无新 Observation 的跨轮重复工具调用。
7. 重复标准化 SQL 不二次执行。
8. 超过 `max_steps=5` 明确终止。
9. 模型直接返回最终答案，标记 direct 和 ungrounded。
10. 单轮多个 tool_calls 与对应 ToolMessage 顺序、call ID 正确。
11. 原生 tool_calls、移除 response_format 回退和完整 JSON fallback。
12. FakeLLM 多轮响应，但 Schema 与 SQL 结果来自真实 SQLite。

### 11.3 评测与报告

- execution_error 与 semantic_mismatch 分离。
- `expected_result` 优先、`gold_sql` 可缺失。
- `tool_sequence_exact_match_rate` 支持多个可接受序列，`required_tool_coverage` 使用显式必需工具标注，两者均正确处理无标注分母。
- 参数合法、调用成功、无效、重复及平均调用次数可人工复算。
- Pipeline 与 Function Calling 使用同一案例集对比。
- 分母为零时输出 null。
- JSON、CSV、Markdown 使用同一 EvaluationReport。

## 12. 最终验收命令

```bash
python -m pip install -e ".[dev]"
python scripts/create_demo_database.py
python -m pytest
python -m execsql_agent.cli run \
  --agent-mode pipeline \
  --database data/demo.db \
  --question "消费金额最高的五名客户是谁？"
python -m execsql_agent.cli run \
  --agent-mode function-calling \
  --database data/demo.db \
  --question "消费金额最高的五名客户是谁？"
python -m execsql_agent.cli evaluate \
  --agent-mode both \
  --database data/demo.db \
  --dataset data/synthetic/eval_questions.json \
  --output-dir data/reports/synthetic
```

## 13. 最终验收标准

1. 从空 Python 3.11 环境可以安装，Demo DB 由脚本确定性创建且未提交仓库。
2. pytest、Ruff 和 mypy 全部通过，测试不访问真实付费 LLM API。
3. Pipeline Baseline 和 FunctionCallingAgent 均可端到端运行。
4. Function Calling 只暴露三个工具，并发送标准 JSON Schema。
5. 原生 tool_calls 与结构化 JSON fallback 都可运行。
6. 模型可一次返回多个 tool_calls，系统逐一验证、分发并回填 ToolMessage。
7. 未知工具和非法参数有可验证的 `executed=false` 记录，绝不执行。
8. unsafe SQL 从未进入 SQLite 执行，且“一次修复、再次 unsafe 终止”可演示。
9. 至少展示一个 Pipeline 首次失败后修复成功案例，以及一个 Function Calling 在 SQL 报错后重新 inspect 并修复成功案例。
10. 重复工具调用、重复 SQL、max_steps 和模型直接回答均有明确行为与轨迹。
11. 每轮 LLM、工具请求、参数验证、Observation、执行结果和终止原因写入 JSONL。
12. 至少 20 条问题在真实 Demo SQLite 上批量运行，expected_result 来自确定性数据。
13. JSON、CSV、Markdown 报告包含通用指标、工具指标和两种模式对比，所有数值来自真实运行。
14. Fake 模式可无 API Key 演示多轮 tool_calls，报告明确标记 `deterministic/mock`。
15. README 包含中文安装、配置、运行、测试、两种 Agent 演示和评测命令。
16. 没有 LangChain、LangGraph、多智能体、复杂 Planner 或任何当前阶段排除项。

## 14. 主要风险与控制

1. **供应商协议差异**：将 native tools、无 response_format 和完整 JSON fallback 归一到 LLMResponse，并用 Pydantic 严格校验。
2. **多工具调用边界**：串行处理、保持 call ID、每个调用独立结果，禁止未校验分发。
3. **重复检测误伤正常修复**：重复工具判断考虑是否产生新 Observation；执行错误后重新 inspect 明确允许；重复 execute SQL 始终阻止。
4. **直接回答造成虚假成功**：分开记录协议完成、execution success、grounded 和 result correct。
5. **工具选择路径不唯一**：数据集允许多个 expected_tool_sequences；无标注不进入准确率分母。
6. **SQL 安全绕过**：SQLValidator、只读 URI、query_only、authorizer 和执行器二次校验多层防护。
7. **Fake 被误解为泛化能力**：CLI、轨迹和所有报告统一标记 deterministic/mock。
8. **长查询阻塞**：progress handler 与真实耗时记录，明确不承诺严格时间截止。
9. **指标口径漂移**：所有格式由统一 EvaluationReport 生成，公式和零分母行为由单元测试固定。

## 15. 当前实施状态与阶段边界

里程碑 A、B 与 C 已完成并通过代码审查。里程碑 D 的 Synthetic 普通评测集、行为场景集、结果比较、批量评测、模式对比、三格式报告和 evaluate CLI 已实现并通过本地验收，等待代码审查确认。

获得明确确认前不得进入 Chinook、BIRD、真实模型测试、公开 Benchmark、训练数据转换或最终 README 包装；当前也不实现训练、多智能体、LangChain、LangGraph 或 Web 前端。后续任何改变工具白名单、安全边界、Agent 模式、轨迹格式或评测口径的调整，仍必须先更新本计划并获得确认。
