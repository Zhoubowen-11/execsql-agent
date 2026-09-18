<div align="center">

<h1>🤖 ExecSQL-Agent</h1>

<p><strong>基于 Qwen3 + vLLM + Function Calling 的可执行 SQL Agent</strong><br/>
支持 Execution Feedback、自纠错、离线评测、QLoRA/SFT 与 GRPO/RLVR 后训练</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/Model-Qwen3-7C3AED" alt="Qwen3"/>
  <img src="https://img.shields.io/badge/Inference-vLLM-0F766E" alt="vLLM"/>
  <img src="https://img.shields.io/badge/Training-PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/RL-verl%20v0.9.0%20%7C%20GRPO-2563EB" alt="verl GRPO"/>
  <img src="https://img.shields.io/badge/Database-SQLite-003B57?logo=sqlite&logoColor=white" alt="SQLite"/>
</p>

<p><a href="README.md">English</a> | <strong>简体中文</strong></p>

</div>

> ExecSQL-Agent 不只生成 SQL，而是让模型在受控工具边界内检查 Schema、验证并执行 SQL，
> 再依据真实数据库结果完成回答；同一套执行事实也用于离线评测、SFT 数据构建和 RLVR 奖励。

## 📌 项目简介

ExecSQL-Agent 是一个面向 SQLite 的 execution-grounded Text-to-SQL 项目。`FunctionCallingAgent`
让模型自主选择工具并依据 Tool Observation 继续推理；`PipelineAgent` 提供固定的生成、验证、
执行、诊断与修复流程，便于稳定对照。两条路径共享只读 SQL 安全边界、会话记忆与 Trajectory 记录。

## ✨ 核心能力

| 能力 | 仓库中的实现 |
|---|---|
| Agent Loop | 原生 `tools` / `tool_calls`；支持多 Tool Call、JSON fallback 与显式终止状态 |
| Tool Use 与自纠错 | 3 个 allowlisted 工具；执行结果或错误作为 Observation 回流模型 |
| 安全执行 | 单语句 SELECT/WITH、只读 URI、query-only、SQLite authorizer 与资源限制 |
| 上下文与审计 | 按 `session_id` 隔离的有限历史；Trajectory JSONL 可恢复、可回放 |
| Evaluation 与 SFT | 执行结果级 Offline Evaluation；Qwen3 assistant-only QLoRA/SFT |
| GRPO / RLVR | SQL Execution Verifier、verl 适配、vLLM group rollout 与非零更新 diagnostic |

## 🔍 与常见 Text-to-SQL 方案的能力对比

| 能力 | 常见 Text-to-SQL | ExecSQL-Agent |
|---|---|---|
| SQL Generation | 常见 | 支持 |
| Function Calling | 部分方案支持 | 原生 Tool Calls + JSON fallback |
| SQL Validation | 部分方案支持 | 单语句、只读、安全关键字与编译检查 |
| Database Execution | 部分方案支持 | 只读 SQLite 真实执行 |
| Execution Feedback | 部分方案支持 | Tool Observation 回流 Agent |
| Auto Repair | 部分方案支持 | Agent 重试 + Pipeline 诊断修复 |
| Session Context | 通常不是核心 | session_id 隔离的有界记忆 |
| Offline Evaluation | 实现方式各异 | 执行结果级比较与可复现报告 |
| SFT | 视方案而定 | Qwen3 assistant-only QLoRA |
| RLVR / GRPO | 研究型能力 | SQL Verifier + verl diagnostic smoke |

## 🏗️ 系统架构

~~~mermaid
flowchart TD
    U[用户问题] --> A[FunctionCallingAgent]
    A --> L[Qwen / OpenAI-compatible LLM]
    L --> F[Function Calling]
    F --> R[ToolRegistry]
    R --> T[SchemaLoader / SQLValidator / SQLExecutor]
    T --> D[(SQLite)]
    D --> O[Tool Observation]
    O --> A
    A -.-> M[Session Memory / TrajectoryLogger]
    M --> E[Offline Evaluator / Reports]
~~~

LLM 不直接接触数据库连接。调用先经过 Tool Registry 的名称、参数和重复调用检查，再进入只读 SQL 工具层；执行结果以结构化 Tool Message 回到 Agent。

## 🔁 Agent 执行流程

1. 按 <code>session_id</code> 加载有限历史，并准备 Schema、约束与工具定义。
2. LLM 返回原生 Tool Calls；不支持原生工具协议的服务可退化到受约束 JSON 格式。
3. Tool Registry 校验名称和 Pydantic 参数，SQL 通过只读安全检查后才会执行。
4. 查询结果或错误作为 Observation 回到上下文，模型可继续检查、修复或结束。
5. 最终回答、步骤、耗时与终止原因写入 Trajectory，供回放、评测与数据构建使用。

固定 Pipeline 则以显式的 generation / validation / execution / diagnosis / repair 状态推进，
用于可解释基线和行为对照。

## 🧩 核心模块

| 模块 | 作用 | 代码入口 |
|---|---|---|
| FunctionCallingAgent | 多轮 Tool Use、执行反馈、重复调用检测和显式终止 | [function_calling.py](src/execsql_agent/agents/function_calling.py) |
| PipelineAgent | 固定生成、诊断与修复流程 | [pipeline.py](src/execsql_agent/agents/pipeline.py) |
| ToolRegistry | 3 个工具的定义、参数校验与分发 | [registry.py](src/execsql_agent/tools/registry.py) |
| SQLValidator | 只读单语句扫描、阻断危险操作、SQLite 编译检查 | [sql_validator.py](src/execsql_agent/tools/sql_validator.py) |
| SQLExecutor | 只读 URI、query-only、authorizer、行数与进度限制 | [sql_executor.py](src/execsql_agent/tools/sql_executor.py) |
| Session / Trajectory | 有界会话记忆与 JSONL 轨迹持久化 | [trajectory/](src/execsql_agent/trajectory) |
| Evaluator | 结果比较、指标聚合与报告导出 | [evaluation/](src/execsql_agent/evaluation) |
| RLVR Verifier | 解析、验证、执行并给出可审计奖励 | [verifier.py](src/execsql_agent/rlvr/verifier.py) |

## 📊 Offline Evaluation

Offline Evaluation 使用真实 SQLite 查询结果作为事实来源。比较器支持：

- 有序结果与保留重复项的无序多重集；
- <code>NULL</code>、数值容差与可选的严格列名；
- 截断结果、拒答、协议完成、grounding、修复成功率和工具调用质量；
- JSON、CSV、Markdown 三种可审计报告。

仓库保留了同一 30-case FSQ Shanghai 评测协议下的真实报告：

| 指标 | Base Qwen3-8B | QLoRA-SFT |
|---|---:|---:|
| Result Accuracy | 16/30（53.33%） | 19/30（63.33%） |
| Execution Success | 30/30 | 30/30 |
| Protocol Completion | 27/30 | 30/30 |
| Grounded Answer | 27/30 | 30/30 |
| First Execution Success | 27/30 | 30/30 |

详细报告位于：

- [Base 评测报告](data/fsq/reports/qwen3_8b_v2_full_zero_shot_v1/evaluation_report.md)
- [SFT 评测报告](data/fsq/reports/qwen3_8b_v2_full_sft_v1/evaluation_report.md)
- [Base vs SFT 分析](data/fsq/reports/qwen3_8b_v2_full_sft_v1/BASE_VS_SFT_ANALYSIS.md)

> 这是一个 30-case、同领域项目评测，用于比较相同协议下的工程行为；不等同于通用
> Text-to-SQL Benchmark 结论。

## 🎯 QLoRA / SFT

SFT 数据由真实工具轨迹构建：Schema 检查、SQL 验证、数据库执行和最终回答均保留为消息序列。
预处理使用 Qwen3 原生 Tool Calling chat template，并设置 <code>enable_thinking=false</code>。

每个 assistant turn 独立成为一个训练样本：

- system、user、tool observation token 使用 <code>-100</code> mask；
- assistant tool call 与 final answer token 参与 loss；
- prefix mismatch、零监督 token、目标截断等情况会在加载模型前 hard stop。

| 配置 | 值 |
|---|---|
| Base Model | Qwen3-8B |
| Quantization | NF4 4-bit、double quantization、BF16 compute |
| LoRA | r=16，alpha=32，dropout=0.05，all-linear |
| 数据 | 180 条训练轨迹 / 20 条开发集轨迹 |
| 训练 | 默认 3 epochs，gradient accumulation=4，learning rate=2e-4 |
| 保存 | 仅保存 PEFT adapter 与 tokenizer，不提交 8B base model |

## 🚀 GRPO / RLVR

SQL 很适合 RLVR（Reinforcement Learning with Verifiable Rewards）：候选 SQL 可以在可信数据库上
真实执行，并将结果与私有 <code>expected_result</code> 比较，因此不需要另一个 LLM Judge 判断正确性。

~~~mermaid
flowchart TD
    P[Strict JSON Prompt] --> A[Qwen3 Actor]
    A --> R[vLLM Group Rollout × 4]
    R --> V[SQL Execution Verifier]
    V --> W[Verifiable Reward]
    W --> G[Group Relative Advantage]
    G --> C[Clipped Policy Objective]
    C --> B[Backward]
    B --> S[Optimizer Step]
~~~

仓库中的 <code>SQLExecutionVerifier</code> 固定使用以下奖励：

| Verifier 结果 | Reward |
|---|---:|
| 安全执行且结果匹配 | 1.0 |
| 安全执行但结果不匹配 | 0.2 |
| 格式错误、危险 SQL、编译/执行失败或不可可靠比较 | 0.0 |

私有的 <code>expected_result</code> 和数据库 SHA 位于 reward metadata，不进入模型 prompt；
reward callback 会校验可信数据库路径和 SHA-256，再调用相同的只读验证/执行链路。

## 🧪 GRPO Diagnostic 真实实验

仓库已固化一组可复现的 Qwen3-0.6B 单步 diagnostic：

| 项目 | 实测值 |
|---|---|
| Model | Qwen3-0.6B |
| Framework | verl v0.9.0 + vLLM |
| Rollout | n = 4 |
| Rewards | [1.0, 0.2, 1.0, 0.2] |
| Sequence Advantage | +0.866 / -0.866 |
| Policy Loss / Grad Norm | -0.07949 / 17.49 |
| Global Step | 0 → 1 |
| Backward / Optimizer Step | Yes / Yes |

> **重要：该实验用于验证 GRPO/RLVR 的完整训练闭环与非零策略更新，
> 不作为正式模型效果提升结论。**

可复现资产：

- [Diagnostic 配置](configs/verl_grpo_sql_diagnostic.yaml)
- [运行脚本](scripts/run_verl_grpo_diagnostic.sh)
- [实验记录](docs/experiments/qwen3_0.6b_grpo_diagnostic.md)
- [DEV-only 数据构建器](training/build_grpo_smoke_dataset.py)

<details>
<summary>🔧 GRPO Diagnostic 关键运行配置</summary>

| 项目 | 配置 |
|---|---|
| Prompt | Qwen3 <code>enable_thinking=false</code>；completion 仅允许 strict JSON SQL 对象 |
| Attention | SDPA |
| Remove Padding | false |
| Rollout Sampling | temperature=0.7，top_p=0.8，top_k=20 |
| vLLM Memory | gpu_memory_utilization=0.35 |
| Batch | train=1，PPO mini=1，micro=1 |
| FSDP | parameter offload + optimizer offload |
| Sampler | VLLM_USE_FLASHINFER_SAMPLER=0 |
| Checkpoint | diagnostic 默认不保存 checkpoint |

</details>

## 🧰 技术栈

| Layer | Technology |
|---|---|
| LLM | Qwen3-8B（SFT / Evaluation）、Qwen3-0.6B（GRPO diagnostic） |
| Inference | OpenAI-compatible Chat Completions、vLLM |
| Agent | Python、FunctionCallingAgent、PipelineAgent |
| Function Calling | 原生 tools/tool_calls、JSON fallback、Pydantic JSON Schema |
| Tools | Tool Registry、SchemaLoader、SQLValidator、SQLExecutor |
| Database | SQLite；DuckDB 用于 verl Parquet 边界 |
| Fine-tuning | PyTorch、Transformers、PEFT、bitsandbytes、QLoRA |
| RL | verl v0.9.0、GRPO、vLLM rollout、FSDP |
| Evaluation | Execution-result comparator、JSON/CSV/Markdown reports |
| Testing | pytest、FakeLLM、ruff、mypy |
| Engineering | httpx、timeout / retry、结构化 Pydantic models、YAML / environment config |

## 📁 项目结构

~~~text
execsql-agent/
├── src/execsql_agent/    # Agent、LLM 抽象、工具、安全执行、轨迹与评测
├── training/             # SFT 数据/训练、RLVR 数据、verl adapter 与 SQL reward
├── configs/              # 基础配置与 verl GRPO smoke/diagnostic 配置
├── scripts/              # 数据库、评测报告和 GRPO 运行脚本
├── docs/experiments/     # 可复现实验记录
├── tests/                # Agent、工具、评测、SFT/RLVR 契约测试
├── data/grpo_smoke/      # DEV-only SQLite 与 verl Parquet smoke 数据
└── data/fsq/             # FSQ 评测集、SFT 数据和已保存报告
~~~

## ⚡ Quick Start

### 1. 安装核心项目

~~~bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

python -m pip install -e ".[dev]"
python scripts/create_demo_database.py
~~~

训练依赖刻意与轻量 Agent 运行依赖分离；完成上述安装即可运行 FakeLLM、真实 SQLite 工具链和测试。

### 2. 运行本地 Function Calling Demo

~~~bash
python -m execsql_agent.cli run \
  --agent-mode function-calling \
  --database data/demo.db \
  --question "统计已完成订单消费金额最高的前五名客户。"
~~~

默认使用 deterministic <code>FakeLLMClient</code>，不访问网络，但 Schema、校验和 SQL 执行都是真实的。

### 3. 连接 OpenAI-compatible 模型服务

~~~bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_MODEL="your-served-model"

python -m execsql_agent.cli run \
  --agent-mode function-calling \
  --database data/demo.db \
  --llm openai \
  --question "按城市统计已完成订单数。"
~~~

## 🧭 Evaluation / Training 用法

### Deterministic Offline Evaluation

~~~bash
python -m execsql_agent.cli evaluate \
  --agent-mode both \
  --database data/demo.db \
  --dataset data/synthetic/eval_questions.json \
  --output-dir outputs/demo_evaluation
~~~

增加 <code>--real-model</code> 后会读取 <code>OPENAI_API_KEY</code>、
<code>OPENAI_BASE_URL</code> 和 <code>OPENAI_MODEL</code>，运行真实模型评测。

### 构建 SFT 数据

需要先按 [data/fsq/README.md](data/fsq/README.md) 构建本地
<code>data/fsq/shanghai_places.db</code>：

~~~bash
python training/build_sft_dataset.py --mode full
~~~

GPU 训练入口：

~~~bash
python training/train_qlora_sft_full.py \
  --model /path/to/Qwen3-8B \
  --train data/fsq/train/sft/train_v1.jsonl \
  --dev data/fsq/train/sft/dev_v1.jsonl \
  --output artifacts/qlora_sft/qwen3_8b_execsql_sft
~~~

### 运行 GRPO Diagnostic

需要已安装 verl v0.9.0、vLLM、PyTorch 及对应 GPU 依赖的 Linux 环境：

~~~bash
python -m training.build_grpo_smoke_dataset

MODEL_PATH=/path/to/Qwen3-0.6B \
bash scripts/run_verl_grpo_diagnostic.sh
~~~

脚本固定使用 GRPO、vLLM rollout 和 <code>n=4</code>。它只用于训练链路诊断，
不会替代正式实验设计。

## 🗺️ Roadmap

### 已完成

- [x] Function Calling Agent 与固定 Pipeline 基线
- [x] Schema Inspection、SQL Validation 和只读 Execution
- [x] Execution Feedback、重复调用防护与 SQL 修复
- [x] Session Memory 与 JSONL Trajectory
- [x] Execution-based Offline Evaluation 与真实报告
- [x] Qwen3-8B assistant-only QLoRA/SFT
- [x] SQL Execution Verifier、RLVR Dataset 与 verl adapter
- [x] verl GRPO infrastructure 与非零 advantage diagnostic

### 计划中

- [ ] 正式 Base / SFT / GRPO 对照评测
- [ ] 更完整的真实数据库 Benchmark
- [ ] 更大规模、可重复的 GRPO 实验与独立 holdout 验证

## ℹ️ 项目状态与边界

- 当前数据库后端仅支持 SQLite，不宣称兼容 PostgreSQL 或 MySQL。
- 当前入口是同步 CLI；仓库 main 不包含 FastAPI 服务或生产级高并发 API。
- FSQ 正式数据库需要从本地数据构建，模型权重、adapter 和训练 checkpoint 不提交到 Git。
- GRPO 部分已经验证 rollout → reward → advantage → backward → optimizer step，
  但尚未给出正式 Accuracy 提升结论。
- 可执行 SQL 仍可能语义错误，因此项目将 execution success 与 result correctness 分开统计。
- 30-case FSQ 结果是聚焦项目评测，不代表广泛领域上的模型泛化能力。

---

如果你关注 Agent 工程，可从
[FunctionCallingAgent](src/execsql_agent/agents/function_calling.py) 开始；
如果你关注后训练，可依次查看
[assistant-only SFT](training/assistant_turn_preprocessing.py)、
[SQL RLVR Verifier](src/execsql_agent/rlvr/verifier.py) 和
[GRPO diagnostic](docs/experiments/qwen3_0.6b_grpo_diagnostic.md)。
