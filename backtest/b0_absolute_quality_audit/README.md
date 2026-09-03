# B0 Absolute Quality Audit v1.1

This audit measures the current Production B0 without searching for a B1 and without modifying Production.

## Why v1.1 exists

The first absolute audit exposed useful structure but also three measurement problems:

1. Eligible-random percentile included many weeks with only one feasible portfolio, mechanically inflating the percentile.
2. Raw fixed-capacity results were based on snapshot-close outcomes and only 21 fully covered weeks, heavily concentrated in early underfilled regimes.
3. Reject-reason labels overlapped, so a descriptive label table could be misread as single-gate causal attribution.

v1.1 fixes all three and adds explicit capacity counterfactuals plus SPY/QQQ.

## Frozen data boundary

AUDIT_AS_OF_DATE = 2026-09-02

Bars after this date are ignored even if materialization is run later.

The audit first uses the existing frozen daily price cache. It then identifies only the tickers that cannot form a mature tradable outcome and downloads those symbols, plus SPY/QQQ, through the repository's existing YahooDataProvider:

- auto_adjust=False
- raw Yahoo OHLC precision
- ticker alias resolution from the existing audit utilities

Downloaded bars are written to this audit's output directory. The shared base price cache is never mutated.

## Tradable raw outcome

Raw-universe, Fill3, simple-rule and market comparisons use:

first trading-session open strictly after snapshot -> close at entry date + 28 calendar days

Stop8 includes the entry session low.

This avoids using the snapshot close even though the snapshot/pool is only known after that close.

The existing entry-aligned W4 outcome is retained only for the current B0-eligible ranking audit.

## Ranking quality

The headline eligible-random percentile uses only active-choice weeks:

- all eligible outcomes mature;
- B0 has at least one pick;
- there are more than one feasible distinct-industry portfolios.

Weeks with one feasible portfolio are reported as gate-locked/no-choice weeks and cannot inflate ranking percentile.

## Variable capacity: how B0 is evaluated

B0 is intentionally allowed to select 0–3 names in Production. v1.1 therefore uses two separate axes:

### Name-selection quality

Matched-N random uses the same weekly number of positions as B0. It asks:

Given the same amount of invested capital, did B0 choose better names?

### Capital-utilization quality

Raw fixed-capacity uses up to three distinct-industry names whenever the raw signal universe can supply them. It asks:

Was B0's decision to leave slots empty economically useful?

Raw fixed-capacity results are also split by original B0 pick count (0/1/2/3).

## Fill3 diagnostic ladder

Production is not changed. Four counterfactuals preserve every original B0 pick and may only fill unused slots:

1. B0_FILL3_RELAX_INDUSTRY
   - only already-eligible candidates;
   - relax distinct1 for empty slots;
   - isolates the industry constraint.

2. B0_FILL3_EPS_ONLY
   - keep distinct1;
   - fill only candidates rejected solely for eps_unknown.

3. B0_FILL3_SINGLE_REJECT
   - keep distinct1;
   - fill the highest B0-ranked candidate that failed exactly one hard gate;
   - this is the minimal general soft-gate candidate.

4. B0_FILL3_ANY_REJECT
   - keep known-industry distinct1;
   - any rejected candidate may fill;
   - diagnostic upper bound only.

No fill policy may replace an original B0 pick.

A Production move toward "prefer 3 names" is justified only if a minimal Fill3 policy improves underfilled-week capital return without material Stop/Ruin degradation. A higher fixed-3 random mean by itself is not enough.

## Eligibility audit

Winner retention is reported together with eligibility acceptance rate:

- winner retention / acceptance rate = winner enrichment;
- loser retention / acceptance rate = loser depletion/enrichment.

Final Top3 winner capture is compared with both:

- random Matched-N expected capture;
- mechanical fixed-3 expected capture.

Low recall is never interpreted without selectivity.

## Reject reasons

Two separate tables are produced:

- exclusive_rejection_summary.csv: candidates with exactly one reject reason; this is the relevant single-gate diagnostic.
- overlap_rejection_summary.csv: descriptive labels; not causal.
- rejection_combinations.csv: exact reason combinations.

## Simple baselines

Closest-to-trigger, entry volume, EPS, momentum20 and relative-SPY20:

- read only raw PIT features;
- use fixed capacity;
- require full feature capacity for a primary weekly comparison;
- use tradable next-open W4;
- report block-bootstrap CI, median edge, beat rate, mean without best 1/2 weeks.

A large right-tail mean is not called stable superiority.

## Outputs

Materialization writes only under:

backtest/b0_absolute_quality_audit/output/

including:

- B0_ABSOLUTE_QUALITY_HEALTH_CHECK.md
- b0_health_summary.json
- current_b0_state.csv
- eligible_random_weekly.csv
- raw_random_weekly.csv
- eligibility_gate_weekly.csv
- exclusive_rejection_summary.csv
- overlap_rejection_summary.csv
- rejection_combinations.csv
- capacity_policy_weekly.csv
- capacity_policy_summary.csv
- underfill_cause_summary.csv
- simple_baseline_summary.csv
- market_benchmark_summary.csv
- nonoverlap_offsets.csv
- yahoo_price_supplement.parquet
- yahoo_download_audit.csv
- run_manifest.json

## Local materialization

Gemini/local runner must not patch source.

Commands:

git checkout codex/clean-latest-quant-trade-replay-pools
git pull --ff-only

/Users/dev/.conda/envs/quant_env/bin/python -m pytest -q tests/test_b0_absolute_quality_audit.py

/Users/dev/.conda/envs/quant_env/bin/python -m backtest.b0_absolute_quality_audit.cli materialize

Then run the existing B0 / Track B/C/D/E/F regression suite.

If successful, commit only:

backtest/b0_absolute_quality_audit/output/

If anything fails, return the exact failure and do not patch the source locally.
