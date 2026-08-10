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
| `protocol_completion_rate` | 27 | 30 | 0.900000 |
| `first_execution_success_rate` | 27 | 30 | 0.900000 |
| `final_execution_success_rate` | 30 | 30 | 1.000000 |
| `result_accuracy` | 16 | 30 | 0.533333 |
| `grounded_answer_rate` | 27 | 30 | 0.900000 |
| `repair_success_rate` | 3 | 3 | 1.000000 |
| `refusal_accuracy` | 30 | 30 | 1.000000 |
| `unsafe_sql_block_rate` | 0 | 0 | null |
| `semantic_mismatch_count` | 14 | 1 | 14.000000 |
| `unsafe_sql_rate` | 0 | 30 | 0.000000 |
| `repeated_sql_rate` | 0 | 30 | 0.000000 |
| `average_llm_turns` | 118 | 30 | 3.933333 |
| `average_sql_executions` | 34 | 30 | 1.133333 |
| `average_duration_ms` | 134210 | 30 | 4473.681279 |
| `tool.tool_sequence_exact_match_rate` | 2 | 30 | 0.066667 |
| `tool.required_tool_coverage` | 29 | 30 | 0.966667 |
| `tool.valid_tool_argument_rate` | 91 | 91 | 1.000000 |
| `tool.tool_execution_success_rate` | 88 | 91 | 0.967033 |
| `tool.invalid_tool_call_rate` | 0 | 91 | 0.000000 |
| `tool.repeated_tool_call_rate` | 0 | 91 | 0.000000 |
| `tool.average_tool_calls` | 91 | 30 | 3.033333 |

## 失败案例

| case_id | mode | failure_kind | termination_reason |
|---|---|---|---|
| `fsq_c05` | `function-calling` | `semantic_mismatch` | `completed` |
| `fsq_c04` | `function-calling` | `semantic_mismatch` | `completed` |
| `fsq_q04` | `function-calling` | `semantic_mismatch` | `completed` |
| `fsq_c07` | `function-calling` | `max_steps` | `max_steps_reached` |
| `fsq_c08` | `function-calling` | `semantic_mismatch` | `completed` |
| `fsq_c06` | `function-calling` | `semantic_mismatch` | `max_steps_reached` |
| `fsq_s05` | `function-calling` | `semantic_mismatch` | `completed` |
| `fsq_d02` | `function-calling` | `semantic_mismatch` | `completed` |
| `fsq_a01` | `function-calling` | `semantic_mismatch` | `completed` |
| `fsq_a06` | `function-calling` | `semantic_mismatch` | `completed` |
| `fsq_c02` | `function-calling` | `semantic_mismatch` | `completed` |
| `fsq_c03` | `function-calling` | `semantic_mismatch` | `completed` |
| `fsq_a03` | `function-calling` | `semantic_mismatch` | `completed` |
| `fsq_a07` | `function-calling` | `semantic_mismatch` | `max_steps_reached` |
| `fsq_q02` | `function-calling` | `semantic_mismatch` | `completed` |
