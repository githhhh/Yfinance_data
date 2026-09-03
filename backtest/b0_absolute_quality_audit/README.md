# B0 Absolute Quality Audit

This audit answers a different question from Track C/D/E/F:

> How good or bad is the current Production B0 in absolute terms?

It does not search for a B1 and does not mutate Production.

## Core design

B0 is recomputed directly from:

    dashboard/skill_industry_eps_known.py

on the frozen replay panel. Historical helper columns such as b0_eligible and is_b0 are not trusted as the current baseline.

Two outcome systems are kept separate.

### Entry-aligned W4

Used only where a Production-style entry outcome is well defined:

- current B0 absolute W4 cohort quality;
- B0 vs exact feasible random portfolios inside the current B0-eligible universe;
- eligible-universe oracle capture;
- ranking monotonicity and rank buckets.

Primary eligible-random weeks require 100% W4 return/stop maturity across the eligible universe.

### Snapshot-close +28 calendar-day W4

Used for the raw Review Universe:

- raw signal random benchmark;
- eligibility winner retention / rejected winner rate;
- rejection-reason opportunity cost;
- simple de-anchored one-factor baselines;
- raw-universe oracle capture;
- SPY / QQQ comparison.

The raw Review Universe is simply:

    signal == True
    AND ibd_candidate_rule is non-empty

It is never conditioned on:

- b0_eligible;
- B0 Lane;
- B0 raw rank;
- B0 evidence/risk count;
- B0 reason codes.

Primary raw-universe inference requires 100% snapshot-W4 price coverage for that week. Partial-price weeks remain diagnostic only because missing tickers include delisted/acquired names and silently deleting them could bias the benchmark.

The fixed-capacity raw benchmark is intentionally de-anchored from B0 abstention: if the raw universe can supply three distinct-industry names, the benchmark uses three slots even when B0 selected fewer or zero names.

## Main health coordinates

1. Absolute B0 W4 distribution:
   mean, median, P10/P25/P75/P90, CVaR10, positive-week rate, Stop8, one-pick ruin, coverage.

2. Eligible random percentile:
   exact feasible same-N distinct-industry portfolio distribution whenever tractable; deterministic Monte Carlo only when combinations exceed the frozen limit.

3. Raw signal random percentile:
   - Primary: fixed capacity up to three distinct-industry positions, independent of B0's weekly pick count.
   - Secondary: Matched-N random, same weekly position count as B0, for conditional name-selection quality.

4. Oracle capture:
   (B0 - random mean) / (oracle - random mean).

5. Eligibility opportunity cost:
   top-20% winner retention, >=20% winner retention, rejected-winner rate, bottom-20% loser rejection, eligible-minus-rejected W4 lift.

6. Fine-ranking information:
   weekly Spearman of -eligible_rank vs W4, Top3-vs-all-eligible lift, rank buckets.

7. Simple de-anchored baselines:
   closest-to-trigger, entry-volume, EPS, 20-day momentum, 20-day relative strength vs SPY.

8. Market context:
   SPY and QQQ snapshot-W4 benchmarks when present in the frozen price cache.

9. W4 overlap robustness:
   four non-overlapping offsets plus 4-week moving-block bootstrap.

## Important interpretation rule

Weekly W4 cohorts overlap. They are selection-quality observations, not independent monthly trades and must not be directly annualized into CAGR.

All current historical evidence is retrospective: B0 and several of its components were developed with visibility into this period. Confidence intervals describe historical stability, not virgin OOS proof.

## Materialize

Gemini/local runner must not patch research source code.

    git checkout codex/clean-latest-quant-trade-replay-pools
    git pull --ff-only

    /Users/dev/.conda/envs/quant_env/bin/python -m pytest -q       tests/test_b0_absolute_quality_audit.py

    /Users/dev/.conda/envs/quant_env/bin/python -m       backtest.b0_absolute_quality_audit.cli materialize

Then run the existing Track/B0 regression suite.

Successful materialization should commit only:

    backtest/b0_absolute_quality_audit/output/

If execution fails, return the exact error instead of modifying source locally.
