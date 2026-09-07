# Canonical Blind Agent Protocol

This file defines the execution discipline for canonical Blind Rule Discovery after Stage 1 has produced the anonymous discovery workspace.

## Fail-closed execution rules

1. The canonical agent implementation and command must already exist before any discovery-data inspection or model call begins.
2. If no compliant pre-existing agent command is available, stop and report that the canonical agent cannot be run. Do not create, edit, copy, install, or patch an ad-hoc agent/runner to make the experiment proceed.
3. Do not perform manual or scripted exploratory analysis of `samples.csv`, `period_summary.csv`, or `feature_profile.csv` before the canonical agent run. In particular, do not pre-rank anonymous features, compute candidate thresholds, run median splits, or construct evidence prompts outside the agent itself.
4. Do not make trial LLM/model calls using discovery-derived evidence before the canonical run. The reported model-request count must cover the complete canonical discovery process, not only a final invocation.
5. The repository-generated `prompt.md` is part of the canonical agent input contract and must be supplied to the agent. Do not replace it with a separately constructed research prompt that drops or weakens its constraints.
6. `--research-seconds` is a hard safety ceiling, not a target. Stop early on genuine convergence. External provider request limits are ceilings only and must never be treated as search targets.
7. A canonical run may legitimately conclude that no sufficiently stable, simple, supported rule was found. Do not force a fallback rule merely to produce `rule.json`.
8. Rule freeze must happen before any private feature mapping or sealed holdout materialization. After holdout consumption, never tune against or re-evaluate the same holdout.

## Consumed R1 experiment

The local experiment previously produced under `backtest/blind_rule_discovery/output/rd_agent_run_01` is classified as:

`R1 — custom heuristic + DeepSeek blind discovery; holdout consumed; result FAIL.`

Its sealed holdout (`2025Q3`, `2025Q4`, `2026Q1`, `2026Q2`) has been consumed and must never again be treated as unseen holdout data for rule selection, tuning, or canonical one-shot validation.

The R1 frozen rule and its negative holdout result remain valid historical evidence and must be preserved. Do not rerun the same holdout simply because the discovery implementation was later judged non-canonical.

## Repository discipline for execution agents

Execution agents such as Gemini must treat the checked-out branch as immutable during a canonical experiment unless the user explicitly starts a separate code-change task. They may run tests and existing commands, inspect generated outputs required for reporting, and stop on errors. They must not edit source files, create research runner modules, modify sandbox policy, install repository-derived agent code into the Python environment, or commit changes while carrying out the experiment.
