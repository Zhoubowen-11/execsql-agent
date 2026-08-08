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
| `total_cases` | 5 | 1 | 5.000000 |
| `execution_accuracy` | 5 | 5 | 1.000000 |
| `protocol_completion_rate` | 5 | 5 | 1.000000 |
| `first_execution_success_rate` | 4 | 5 | 0.800000 |
| `final_execution_success_rate` | 5 | 5 | 1.000000 |
| `result_accuracy` | 3 | 4 | 0.750000 |
| `grounded_answer_rate` | 5 | 5 | 1.000000 |
| `repair_success_rate` | 1 | 1 | 1.000000 |
| `refusal_accuracy` | 5 | 5 | 1.000000 |
| `unsafe_sql_block_rate` | 0 | 0 | null |
| `semantic_mismatch_count` | 1 | 1 | 1.000000 |
| `unsafe_sql_rate` | 0 | 5 | 0.000000 |
| `repeated_sql_rate` | 0 | 5 | 0.000000 |
| `average_llm_turns` | 19 | 5 | 3.800000 |
| `average_sql_executions` | 6 | 5 | 1.200000 |
| `average_duration_ms` | 18708.3 | 5 | 3741.651043 |
| `tool.tool_sequence_exact_match_rate` | 1 | 5 | 0.200000 |
| `tool.required_tool_coverage` | 5 | 5 | 1.000000 |
| `tool.valid_tool_argument_rate` | 14 | 14 | 1.000000 |
| `tool.tool_execution_success_rate` | 13 | 14 | 0.928571 |
| `tool.invalid_tool_call_rate` | 0 | 14 | 0.000000 |
| `tool.repeated_tool_call_rate` | 0 | 14 | 0.000000 |
| `tool.average_tool_calls` | 14 | 5 | 2.800000 |

## 失败案例

| case_id | mode | failure_kind | termination_reason |
|---|---|---|---|
| `fsq_c05` | `function-calling` | `semantic_mismatch` | `completed` |
