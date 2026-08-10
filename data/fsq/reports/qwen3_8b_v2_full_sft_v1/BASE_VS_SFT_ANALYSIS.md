# Base vs SFT Post-SFT Evaluation

## 实验设置

- 数据集：`data/fsq/eval/questions_v2.json`，共 30 个 case，全部进入 `result_accuracy` 分母。
- 数据库：`data/fsq/shanghai_places.db`，由现有只读 `SQLExecutor` 执行。
- Agent：现有 `FunctionCallingAgent`，工具、系统指令、生成参数和模型输出均未修改；报告仅按修正后的截断结果口径做 deterministic rescore。
- Base：vLLM served model `Qwen3-8B` 的历史报告 `qwen3_8b_v2_full_zero_shot_v1`。
- SFT：vLLM LoRA model `ExecSQL-Qwen3-8B-SFT`，`OPENAI_ENABLE_THINKING=false`。
- SFT smoke：`fsq_s04` 正确完成 `inspect_schema → validate_sql → execute_sql → final answer`，之后才运行完整评测。

## Base vs SFT 指标

| 指标 | Base | SFT | 变化 |
| --- | ---: | ---: | ---: |
| Result accuracy | 16/30 (53.33%) | 19/30 (63.33%) | +3 case / +10.00 pp |
| Execution success | 30/30 (100%) | 30/30 (100%) | 0 |
| Protocol completion | 27/30 (90%) | 30/30 (100%) | +3 case / +10 pp |
| Grounded answer | 27/30 (90%) | 30/30 (100%) | +3 case / +10 pp |
| First execution success | 27/30 (90%) | 30/30 (100%) | +3 case / +10 pp |
| Repair success | 3/3 (100%) | 0/0 (null) | SFT 没有触发 repair 分母 |
| Required-tool coverage | 29/30 (96.67%) | 30/30 (100%) | +1 case |
| Valid tool arguments | 91/91 (100%) | 90/90 (100%) | 无退化 |
| Tool execution success | 88/91 (96.70%) | 90/90 (100%) | +3.30 pp |
| Invalid tool calls | 0 | 0 | 0 |
| Repeated tool calls | 0 | 0 | 0 |
| Unsafe SQL calls | 0 | 0 | 0 |
| Average LLM turns | 3.933 | 4.000 | +0.067 |
| Average tool calls | 3.033 | 3.000 | -0.033 |
| Average SQL executions | 1.133 | 1.000 | -0.133 |
| Average duration | 4473.68 ms | 5570.59 ms | +1096.91 ms |

`tool_sequence_exact_match_rate` 从 2/30 变为 0/30。SFT 的 30 个 case 全部使用 `inspect_schema → validate_sql → execute_sql`，而数据集主要把 `inspect_schema → execute_sql` 列为期望序列；因此该指标下降反映额外的合法验证调用，不等同于工具失败。

## 改善 case

Base 错、SFT 对，共 8 个：

- `fsq_a03`
- `fsq_a06`
- `fsq_c02`
- `fsq_c03`
- `fsq_c06`
- `fsq_d02`
- `fsq_q02`
- `fsq_s05`

协议完成与 grounded answer 各改善 3 个：`fsq_a07`、`fsq_c06`、`fsq_c07`。SFT 没有 invalid、repeated 或 unsafe tool call；SQL 执行总数从 34 降至 30。

## Regression case

Base 对、SFT 错，共 5 个：

| Case | 主要原因 |
| --- | --- |
| `fsq_a05` | 应按记录量降序，SFT 却先按年份降序，Top-5 年份错误。 |
| `fsq_a08` | 最终 SQL 多返回 `fsq_place_id`；最终回答还漏掉第二条同名 `Baker & Spice`，只列出 9 项。 |
| `fsq_d04` | 问题要求酒店分区，SQL 未关联分类、未筛选 `Hotel`，实际统计各区全部地点。 |
| `fsq_q03` | `MAX(date_refreshed)` 与全表 `COUNT(*)` 同层聚合，把最新日期当天数量错误写成 91770。 |
| `fsq_s06` | 与 `fsq_q03` 相同的聚合范围错误；正确数量为 280。 |

## Remaining failure case

Base 与 SFT 都错，共 6 个：

| Case | Evaluator failure_kind | 只读归因 |
| --- | --- | --- |
| `fsq_a01` | `semantic_mismatch` | 错误增加 `canonical_district IS NULL`，排除了大部分有效原始 `locality` 值。 |
| `fsq_a07` | `semantic_mismatch` | 三组数值正确，final answer 正确；SQL 将 `place_count` 和 `contact_status` 的列位置反转，结果形状不匹配。 |
| `fsq_c04` | `semantic_mismatch` | 按地点分组后返回大量 `[1]` 并被截断，模型错误地把返回上限 100 当成多分类地点总数；真实期望为 9415。 |
| `fsq_c05` | `semantic_mismatch` | `is_primary=1` 不能表示“地点拥有多个分类”；SQL 没有先筛出分类数大于 1 的地点。 |
| `fsq_c08` | `semantic_mismatch` | 两个计数和 final answer 正确，但 SQL 的两列位置与契约相反。 |
| `fsq_q04` | `semantic_mismatch` | SQL 多返回 `fsq_place_id`，且相同关闭日期缺少稳定次级排序；final answer 的十个名称和日期有工具证据。 |

SFT 的 11 个自动失败 case 均为 `semantic_mismatch`，没有 protocol、execution、invalid、repeated 或 unsafe 类型失败。

## 截断结果口径修正

`fsq_c04` 在 Base 与 SFT 中均执行了按地点分组的查询，返回大量 `[1]` 并被 100 行上限截断。两次 final answer 都错误地把返回行数 100 当成多分类地点总数；真实期望为 9415。修复后，成功执行但 `truncated=true` 的结果不再被视为不可验证，而是记为 `result_correct=false`、`failure_kind=semantic_mismatch` 并进入 accuracy 分母。真正没有 SQL 结果或执行失败的 case 仍保持 `result_correct=null`。两份报告仅从已保存的 case facts 重新评分，未请求模型，原始 trajectories 的 SHA-256 保持不变。

## 行为变化与风险

- SQL-result 层面有 8 个改善、5 个 regression，净改善 3 个。
- Protocol completion 与 grounding 均增加 3 个；execution success 保持 100%。
- Invalid/repeated tool call 都维持为 0，没有可归因于这两项的提升空间。
- SFT 的所有 case 都固定为 4 个 LLM turn、3 个工具调用、1 次 SQL 执行。这消除了 Base 的错误 SQL 后修复路径，但也表现出强烈的固定模板式工具调用。
- 没有出现过早 final answer、跳过 required tool 或重复 `execute_sql`；主要剩余问题是可执行但语义错误的 SQL，以及额外列/列位置违反结果形状。
- 平均时延增加约 1.10 秒；当前报告未提供 token usage，无法做可靠 token 对比。

## 保守解释

在完全相同的现有协议下，SFT 自动 `result_accuracy` 从 53.33% 提升到 63.33%，同时 protocol completion 和 grounding 达到 100%。该结果支持“训练后工具协议更稳定、部分 SQL 语义得到改善”，但不能证明全面泛化：只有 30 个同域 case，存在 5 个 Base→SFT regression 和 6 个共同失败。此外，所有 SFT case 使用同一工具模板，说明模型可能学习了训练轨迹格式，而不只是问题驱动的工具规划。应保留本次单次完整运行结果，不挑题重跑，并在后续独立 holdout 上再验证泛化。

## 报告完整性

- SFT report：30 条记录、30 个唯一 ID；无缺失、无重复、无额外 ID。
- `result_accuracy`、protocol、execution、grounding 的聚合分子分母均与逐 case 重算一致。
- Base report 仍位于独立目录，SFT 输出没有覆盖它。
