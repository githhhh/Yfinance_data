# Retrospective Empirical Ceiling Protocol

This protocol is separate from `CANONICAL_AGENT_PROTOCOL.md`.

## Purpose

Estimate how much historical signal exists in the current point-in-time feature set when compact feature interactions are searched systematically. This is an explicitly coupled retrospective study. It is not a sealed-holdout experiment and cannot establish future generalization.

The already-consumed R1 periods (`2025Q3`, `2025Q4`, `2026Q1`, `2026Q2`) may be used here only as known historical data. They can never regain unseen-holdout status.

## Frozen execution entry point

The only approved execution entry for this study is:

```bash
python -m backtest.blind_rule_discovery.retrospective_ceiling_balanced ...
```

`retrospective_ceiling.py` is a lower-level helper/legacy entry and is not the approved study command.

## Search contract

- No LLM, RD-agent, DeepSeek, or external model participates in rule generation or selection.
- Use the existing explicit discovery feature allowlist plus point-in-time `M_*` market features.
- Threshold candidates come from deterministic empirical quantiles.
- Search every eligible single condition.
- Preserve threshold candidates from every searchable feature before interaction search.
- Exhaustively test pairs from that feature-balanced condition pool.
- Extend the strongest pairs to three-condition conjunctions with deterministic beam search.
- Test compact DNF/OR combinations from the strongest conjunction clauses, with at most 3 clauses and 6 total conditions.
- Rank with the committed multi-objective score rather than winner rate alone.
- Report the Pareto frontier rather than hiding alternative risk/coverage tradeoffs behind one scalar score.
- Run leave-one-quarter-out sensitivity on top rules.
- Run expanding-window rolling re-search: each fold must derive thresholds and choose its rule using past quarters only, then evaluate exactly the next quarter.

## Interpretation contract

The full-history best rule is an **empirical ceiling candidate**, not OOS evidence.

The rolling results are retrospective walk-forward diagnostics. Because the overall research protocol itself was designed after seeing historical R1 results, they are not a replacement for a future untouched one-shot holdout.

A useful result should show more than one attractive in-sample number. Review at minimum:

- resolved winner-rate lift versus the contemporaneous universe;
- 12-week excess-return p25/p50/p75;
- MAE and MFE;
- selection coverage and resolved support;
- active/evaluated quarters;
- fraction of quarters with positive relative lift;
- median and worst quarter lift;
- Pareto alternatives;
- leave-one-quarter-out degradation;
- rolling walk-forward pooled and per-quarter results;
- feature frequency across rolling folds.

## Execution discipline

An executor such as Gemini may run commands and inspect generated outputs only. It must not edit code, tune parameters after seeing results, add a new search method, call an LLM, or rerun with changed thresholds in order to improve the answer.

If the frozen command fails, report the failure. Do not repair it during the research execution. Code fixes belong in a separate audited development step before any new run.
