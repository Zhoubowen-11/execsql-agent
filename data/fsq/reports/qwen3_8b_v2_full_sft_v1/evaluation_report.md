# Agent 评测报告

- 数据集：`fsq_shanghai_business_v2`
- 数据库：`shanghai_places`
- Agent 模式：`function-calling`
- 运行模式：`live`
- 随机种子：`0`
- 说明：指标来自本次真实模型调用及真实 SQLite 执行。

## function-calling

| 指标 | numerator | denominator | value |
|---|---:|---:|---:|
| `total_cases` | 30 | 1 | 30.000000 |
| `execution_accuracy` | 30 | 30 | 1.000000 |
| `protocol_completion_rate` | 30 | 30 | 1.000000 |
| `first_execution_success_rate` | 30 | 30 | 1.000000 |
| `final_execution_success_rate` | 30 | 30 | 1.000000 |
| `result_accuracy` | 19 | 30 | 0.633333 |
| `grounded_answer_rate` | 30 | 30 | 1.000000 |
| `repair_success_rate` | 0 | 0 | null |
| `refusal_accuracy` | 30 | 30 | 1.000000 |
| `unsafe_sql_block_rate` | 0 | 0 | null |
| `semantic_mismatch_count` | 11 | 1 | 11.000000 |
| `unsafe_sql_rate` | 0 | 30 | 0.000000 |
| `repeated_sql_rate` | 0 | 30 | 0.000000 |
| `average_llm_turns` | 120 | 30 | 4.000000 |
| `average_sql_executions` | 30 | 30 | 1.000000 |
| `average_duration_ms` | 167118 | 30 | 5570.591007 |
| `tool.tool_sequence_exact_match_rate` | 0 | 30 | 0.000000 |
| `tool.required_tool_coverage` | 30 | 30 | 1.000000 |
| `tool.valid_tool_argument_rate` | 90 | 90 | 1.000000 |
| `tool.tool_execution_success_rate` | 90 | 90 | 1.000000 |
| `tool.invalid_tool_call_rate` | 0 | 90 | 0.000000 |
| `tool.repeated_tool_call_rate` | 0 | 90 | 0.000000 |
| `tool.average_tool_calls` | 90 | 30 | 3.000000 |

## 失败案例

| case_id | mode | failure_kind | termination_reason |
|---|---|---|---|
| `fsq_a05` | `function-calling` | `semantic_mismatch` | `completed` |
| `fsq_c05` | `function-calling` | `semantic_mismatch` | `completed` |
| `fsq_c04` | `function-calling` | `semantic_mismatch` | `completed` |
| `fsq_s06` | `function-calling` | `semantic_mismatch` | `completed` |
| `fsq_q04` | `function-calling` | `semantic_mismatch` | `completed` |
| `fsq_d04` | `function-calling` | `semantic_mismatch` | `completed` |
| `fsq_c08` | `function-calling` | `semantic_mismatch` | `completed` |
| `fsq_a01` | `function-calling` | `semantic_mismatch` | `completed` |
| `fsq_q03` | `function-calling` | `semantic_mismatch` | `completed` |
| `fsq_a08` | `function-calling` | `semantic_mismatch` | `completed` |
| `fsq_a07` | `function-calling` | `semantic_mismatch` | `completed` |
