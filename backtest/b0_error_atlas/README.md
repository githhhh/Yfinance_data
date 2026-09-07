# B0 Error Atlas v1

Purpose: determine whether current frozen PIT information can explain and potentially reduce two concrete B0 errors without modifying Production:

1. False negatives: B0 rejects/misses a stock that later becomes a clean large winner.
2. False positives: B0 selects a stock that later becomes a path failure.

This is not a B1 search.

## Error layers

Misses are separated into:

- Gate miss: current Production eligibility rejects the name.
- Selector miss: eligible but not selected into Top3.
- All unselected: either cause.

This prevents gate opportunity cost from being misdiagnosed as a ranking problem.

## Path-aware labels

Using the frozen next-open entry clock from the B0 Absolute Audit:

- clean_big_winner:
  terminal W4 >= +20% and Stop8 is never touched.
- rebound_big_winner:
  terminal W4 >= +20% but Stop8 is touched first or along the path.
- strict_path_failure:
  Stop8 occurs before Profit20, or terminal W4 <= -8%.
- same-day Stop8 + Profit20:
  ordering is ambiguous and excluded from binary recovery/veto tasks.

The selected-veto task compares path failures with selected names that finish >= +8% without Stop8.

## Feature sets

RAW_ONLY excludes all B0-derived fields.

It contains the pre-registered raw PIT feature manifest plus new strictly pre-snapshot derived features:

- downside volatility, max down day, gap-down structure;
- return skew, max drawdown;
- close-position / down-volume structure;
- SPY market momentum / volatility / drawdown regime;
- cross-sectional pool percentiles;
- sector-relative pool momentum and candidate breadth.

B0_AUGMENTED adds current B0 raw rank, pick order, Lane and reject reasons. This measures whether B0 priors add useful information beyond raw features.

## Analyses

- label counts and support by quarter;
- numeric effect size / sign-agnostic AUC / mutual information / missingness;
- quarterly direction stability;
- categorical target-rate and mutual-information tables;
- numeric feature redundancy:
  pairwise Spearman and PCA effective dimension;
- untuned Logistic-L2 and shallow Random Forest;
- chronological expanding-quarter evaluation only;
- top-feature pair interaction scan;
- chronological test-quarter permutation importance;
- missed-winner examples;
- bad-selected examples;
- reject-reason outcome map.

No random row split is allowed.

## Evidence boundary

The full history has only about 42 independent weekly snapshots and is reused retrospective evidence. Chronological folds reduce leakage but do not create untouched OOS evidence.

Pair candidates are preselected using full-history univariate results and therefore remain exploratory.

If current + newly derived PIT features still cannot separate FN/FP chronologically, that is evidence of an information-dimension problem, not proof that B0 is globally optimal.

Remaining unavailable information dimensions are explicitly listed in the materialized manifest/report.

## Run

Gemini/local runner must not patch source.

    git checkout codex/clean-latest-quant-trade-replay-pools
    git pull --ff-only

    /Users/dev/.conda/envs/quant_env/bin/python -m pytest -q       tests/test_b0_error_atlas.py

    /Users/dev/.conda/envs/quant_env/bin/python -m       backtest.b0_error_atlas.cli materialize

Then run the existing B0 / Track regression suite.

Successful execution should commit only:

    backtest/b0_error_atlas/output/

If execution fails, return the exact error and do not modify research source locally.
