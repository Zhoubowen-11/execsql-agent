# ExecSQL-Agent

ExecSQL-Agent is an execution-feedback-driven SQL agent built around
schema inspection, SQL validation, database execution, automatic repair,
trajectory evaluation, and QLoRA SFT.

## Current Status

- Function-calling SQL agent
- Schema inspection / validation / execution tools
- Execution-feedback repair
- Trajectory logging and evaluation
- Qwen3-8B zero-shot evaluation
- Assistant-only SFT dataset construction
- QLoRA 1-step smoke training verified

## Baseline

Qwen3-8B zero-shot on the current evaluation set:

- Execution success: 100%
- Result accuracy: 55.17% (16/29)

Formal QLoRA SFT and post-SFT evaluation are in progress.
