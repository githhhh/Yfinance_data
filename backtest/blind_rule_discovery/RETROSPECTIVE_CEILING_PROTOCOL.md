# Retrospective Empirical Ceiling Protocol

This protocol is separate from `CANONICAL_AGENT_PROTOCOL.md`.

## Purpose

Estimate how much historical signal exists in the current point-in-time feature set when compact feature interactions are searched systematically. This is an explicitly coupled retrospective study. It is not a sealed-holdout experiment and cannot establish future generalization.

The already-consumed R1/R2 periods (`2025Q3`, `2025Q4`, `2026Q1`, `2026Q2`) may be used here only as known historical data. They can never regain unseen-holdout status.

## Frozen R3 execution entry point

The only approved execution entry for the current R3 study is:

```bash
python -m backtest.blind_rule_discovery.retrospective_ceiling_exhaustive ...
```

Historical entries are retained only for reproducibility:

- `retrospective_ceiling.py`: lower-level/legacy helper.
- `retrospective_ceiling_balanced.py`: R2 execution implementation.

Neither historical entry is approved for a new R3 run.

## R3 search contract

- No LLM, RD-agent, DeepSeek, or external model participates in rule generation or selection.
- Use the existing explicit discovery feature allowlist plus point-in-time `M_*` market features.
- Threshold candidates come from deterministic empirical quantiles.
- Search every supported single condition.
- Exhaustively test **every compatible pair of generated quantile conditions**. The pair layer must not prune thresholds by feature rank or beam width.
- Beam search is permitted only after the exact two-condition layer, for three-condition conjunctions.
- Test compact DNF/OR combinations from the strongest conjunction clauses, with at most 3 clauses and 6 total conditions.
- Rank with the committed multi-objective score rather than winner rate alone.
- Report the Pareto frontier rather than hiding risk/coverage tradeoffs behind one scalar score.

## Supported-quarter contract

A rule is not eligible merely because it selects something in many quarters.

The frozen full-history defaults require all of the following:

- selected samples >= 40;
- resolved selected samples >= 30;
- active quarters >= 5;
- evaluable quarters >= 5;
- evaluable-quarter fraction >= 50% of active quarters;
- an evaluable quarter requires at least 5 resolved selected samples.

This prevents a rule such as `active=10, evaluable=4` from receiving a misleading 100% quarter-outperformance stability score.

## Quarter robustness contract

R3 produces two deliberately different quarter-removal analyses.

### Drop-One-Quarter Sensitivity

`drop_one_quarter_sensitivity.csv` keeps the globally discovered rule fixed and drops one quarter at a time before recomputing metrics. It only answers whether one historical quarter dominates the fixed rule's aggregate performance. It is **not** cross-validation and must not be called LOQ validation.

### True Leave-One-Quarter-Out Re-search

`leave_one_quarter_out_research.csv` must, for every held quarter:

1. remove that quarter from the research frame;
2. regenerate thresholds and re-search the rule using only the remaining quarters;
3. freeze the selected rule for that fold;
4. evaluate it only on the held quarter.

The held quarter must never appear in that fold's training-quarter list.

## Rolling contract

Expanding-window rolling re-search must:

1. use only quarters strictly earlier than the test quarter;
2. regenerate thresholds and re-search using the past training frame only;
3. freeze the best rule for the fold;
4. evaluate exactly the next quarter.

Rolling summaries must report both:

- positive-lift fraction among evaluable folds; and
- positive-lift fraction among **all** folds.

A zero-selection fold is a stability failure signal and remains in the all-fold denominator. Also report zero-selection count/fraction and a pooled contemporaneous universe winner-rate baseline across all rolling test quarters.

## Interpretation contract

The full-history best rule is an **empirical ceiling candidate**, not OOS evidence.

The exact pair layer establishes the best result found within the committed two-condition quantile grid under the committed support/score contract. Three-condition and DNF results remain approximate because they use deterministic beam expansion after the exact pair layer.

True LOQ re-search and rolling walk-forward are retrospective robustness diagnostics. Because the overall research protocol was designed after seeing historical R1/R2 results, neither replaces a future untouched one-shot holdout.

Review at minimum:

- resolved winner-rate lift versus the contemporaneous universe;
- 12-week excess-return p25/p50/p75;
- MAE and MFE;
- selection coverage and resolved support;
- active/evaluable quarters and evaluable-quarter fraction;
- fraction of evaluable quarters with positive relative lift;
- median and worst quarter lift;
- Pareto alternatives;
- fixed-rule drop-one-quarter sensitivity;
- true LOQ re-search held-quarter results;
- rolling pooled and per-quarter results;
- zero-selection rolling folds;
- feature frequency across rolling folds.

## Execution discipline

An executor such as Gemini may run commands and inspect generated outputs only. It must not edit code, tune parameters after seeing results, add a new search method, call an LLM, or rerun with changed thresholds in order to improve the answer.

If the frozen command or any preflight test fails, report the failure and stop. Do not repair it during research execution. Code fixes belong in a separate audited development step before any new run.
