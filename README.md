# ExecSQL-Agent

[简体中文](README_CN.md)

ExecSQL-Agent is an execution-grounded Text-to-SQL system for SQLite. It turns a natural-language question into a safe, verifiable tool-use trace:

```text
question -> inspect schema -> call tools -> validate SQL -> execute SQL
         -> use execution feedback -> answer -> log trajectory -> evaluate
```

The repository combines a deterministic Pipeline baseline, a native Function Calling agent, offline execution-based evaluation, and an assistant-only QLoRA SFT workflow for Qwen3-8B. The accepted v1 model is **Qwen3-8B + QLoRA SFT**.

## Why ExecSQL-Agent

Text-to-SQL quality is not just whether SQL parses. A useful agent must select valid tools, recover from database errors, ground its answer in actual rows, reject unsafe statements, and preserve enough evidence for reproducible evaluation. ExecSQL-Agent makes those behaviors explicit instead of hiding them behind a single success flag.

The primary business workload uses the FSQ OS Places Shanghai snapshot. The database is built deterministically from Parquet and contains 91,770 places, category metadata, many-to-many place/category links, and a conservative Shanghai district normalization.

## Architecture and agent workflow

```text
PipelineAgent --------------------+
                                  |
FunctionCallingAgent -> LLMClient +-> SchemaLoader / ToolRegistry
   ^                   native or       |- inspect_schema
   |                   JSON fallback   |- validate_sql
   +-- tool observations               `- execute_sql (read-only)
                                  |
                                  +-> TrajectoryLogger -> Evaluator -> reports
```

- `PipelineAgent` is the stable generate/execute/diagnose/repair baseline.
- `FunctionCallingAgent` supports native `tools`/`tool_calls`, multiple calls per turn, JSON fallback, repeated-call detection, and session-scoped short-term memory.
- `OpenAICompatibleLLMClient` isolates provider details; `FakeLLMClient` provides deterministic, network-free tests while all SQL still runs on real SQLite.
- `SchemaLoader`, `SQLValidator`, and `SQLExecutor` are domain-neutral. FSQ knowledge lives in `config/fsq_shanghai.json`, not in agent code.

## Tool use, safety, and trajectories

The Function Calling agent exposes only `inspect_schema`, `validate_sql`, and `execute_sql`. Tool names and Pydantic arguments are checked before dispatch. SQLite is opened read-only with query-only and authorizer protections; unsafe or multi-statement SQL never reaches execution.

Each JSONL trajectory records messages, assistant tool calls, observations, SQL results, execution timing, answerability, termination reason, and separate outcome semantics:

- `protocol_completed`
- `execution_success`
- `answer_grounded`
- `result_correct`

The final-query contract requires the last successful `execute_sql` call to return the complete rows and columns needed by the final answer.

## Evaluation

Correctness is determined by comparing the final real SQLite result with `expected_result`, not by SQL string equality or model prose. The comparator supports ordered or multiset rows, duplicate-row semantics, `NULL`, numeric tolerance, optional strict column names, and deterministic JSON/CSV/Markdown reports.

An evaluator denominator bug was corrected during v1 finalization. Previously, a successful but truncated result became `result_correct=null` and disappeared from accuracy. It now becomes `result_correct=false` with `semantic_mismatch`. Historical reports can be rescored without calling the model, and the trajectory hash is checked before and after rescoring:

```bash
python scripts/rescore_evaluation_report.py \
  data/fsq/reports/qwen3_8b_v2_full_zero_shot_v1
```

This changed the reported denominators from Base `16/29` and SFT `19/29` to the correct 30-case values below.

## QLoRA SFT

The SFT pipeline serializes each conversational trajectory with Qwen3's native tool-calling chat template and `enable_thinking=false`. Every assistant turn becomes one sample: system, user, and tool-observation tokens are masked, while assistant tool calls and final answers receive loss.

| Configuration | Value |
|---|---:|
| Base model | Qwen3-8B |
| Quantization | NF4 4-bit, double quantization, BF16 compute |
| LoRA | r=16, alpha=32, dropout=0.05, all-linear targets |
| Trainable parameters | 43,646,976 |
| Train data | 180 trajectories -> 720 assistant turns |
| Dev data | 20 trajectories -> 80 assistant turns |
| Training | 3 epochs, 540 optimizer steps |
| Hardware | RTX 4090D, peak allocated VRAM ~16.83 GiB |
| Runtime | ~48m25s |

No hidden chain-of-thought is trained. The base model and adapters are intentionally excluded from Git.

## Base vs SFT results

Both models use the same 30 questions, database, agent protocol, tools, system prompt, generation settings, and corrected evaluator.

| Metric | Base Qwen3-8B | QLoRA-SFT |
|---|---:|---:|
| Result accuracy | 16/30 (53.33%) | **19/30 (63.33%)** |
| Execution success | 100% | 100% |
| Protocol completion | 90% | **100%** |
| Answer grounded | 90% | **100%** |
| First execution success | 90% | **100%** |
| Avg. LLM turns | 3.933 | 4.000 |
| Avg. SQL executions | 1.133 | 1.000 |
| Avg. tool calls | 3.033 | 3.000 |

SFT improved result accuracy by **10.00 percentage points** and eliminated protocol-completion and grounding failures in this evaluation. Remaining errors are primarily executable-but-semantically-wrong SQL involving aggregation scope, filtering semantics, ranking, ties, multi-category semantics, and output shape.

Formal reports are under:

- `data/fsq/reports/qwen3_8b_v2_full_zero_shot_v1/`
- `data/fsq/reports/qwen3_8b_v2_full_sft_v1/`

## GRPO / RLVR diagnostic

The repository includes an execution-based `SQLExecutionVerifier` and a verl adapter for a small, reproducible GRPO diagnostic. Rewards come from real SQLite execution rather than an LLM judge:

| Outcome | Reward |
| --- | ---: |
| Correct execution result | `1.0` |
| Safe and executable, but wrong result | `0.2` |
| Malformed, unsafe, or failed execution | `0.0` |

The DEV diagnostic uses Qwen3-0.6B, vLLM group rollout with `n=4`, and strict JSON completions. One recorded group produced rewards `[1.0, 0.2, 1.0, 0.2]`, sequence advantages `±0.866`, policy loss `-0.07949`, grad norm `17.49`, and a real optimizer step from global step `0` to `1`.

> This validates the end-to-end GRPO training path and a non-zero parameter update. It is not evidence that GRPO improves SQL accuracy or outperforms the Base/SFT checkpoints.

Reproducible assets live in `configs/verl_grpo_sql_diagnostic.yaml`, `scripts/run_verl_grpo_diagnostic.sh`, and `docs/experiments/qwen3_0.6b_grpo_diagnostic.md`.

## Reproducibility

Install the core project with Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/create_demo_database.py
python -m pytest
python -m ruff check .
python -m mypy src
```

Build the FSQ database from local source Parquet files (the global raw release is not committed):

```bash
python scripts/build_fsq_shanghai_db.py \
  --places data/fsq/processed/shanghai_places_filtered.parquet \
  --categories data/fsq/raw/release/dt=2026-07-09/categories/parquet/categories_000000.parquet \
  --output data/fsq/shanghai_places.db
```

Run the Function Calling agent:

```bash
python -m execsql_agent.cli run \
  --agent-mode function-calling \
  --database data/fsq/shanghai_places.db \
  --domain-config config/fsq_shanghai.json \
  --llm openai \
  --question "在能够明确识别所属行政区的地点中，哪个区的咖啡店最多？"
```

OpenAI-compatible serving reads `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` from the process environment. `.env.example` is a reference template; the application does not automatically load a local `.env` file. Evaluate with the same protocol:

```bash
python -m execsql_agent.cli evaluate \
  --agent-mode function-calling \
  --database data/fsq/shanghai_places.db \
  --domain-config config/fsq_shanghai.json \
  --dataset data/fsq/eval/questions_v2.json \
  --output-dir data/fsq/reports/my_run \
  --real-model
```

Training environments and GPU dependencies are intentionally separate from the lightweight project dependencies. SFT entry points include `training/assistant_turn_preprocessing.py` and `training/train_qlora_sft_full.py`; the reproducible GRPO diagnostic is launched through `scripts/run_verl_grpo_diagnostic.sh`. These are training workflows, not distributed model checkpoints.

## Project structure

```text
src/execsql_agent/    agents, LLM abstraction, tools, trajectories, evaluator
training/             SFT plus RLVR dataset, reward, and verl adapter utilities
scripts/              deterministic database, dataset, and report utilities
tests/                agent, evaluator, data, training, and reward tests
config/               domain configuration for FSQ Shanghai
data/fsq/             documented inputs, evaluation sets, and compact reports
```

## Limitations

- SQLite only; the project does not claim PostgreSQL/MySQL compatibility.
- FSQ data cannot answer ratings, revenue, traffic, rent, routes, or real-time/future questions.
- District normalization is conservative and leaves ambiguous records unmapped.
- Executable SQL can still be semantically wrong; execution success is not result correctness.
- The 30-case result is a focused project evaluation, not a broad Text-to-SQL benchmark claim.
- Model weights, adapters, secrets, private holdout data, and large training artifacts are not distributed through Git. Compact DEV smoke databases/parquets and selected trajectories or evaluation reports are intentionally tracked when needed for reproducibility.
