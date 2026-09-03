# Track F — Lane Taxonomy & Composition Audit

Track F is a research-only follow-up to Track C/D/E. Production B0 is not modified.

## Problem statement

The current five-Lane enum mixes multiple semantic dimensions:

- setup route (pullback vs non-pullback),
- demand confirmation,
- follow-through confirmation,
- non-ACTIONABLE pullback context,
- geometry failure.

Two concrete consequences are audited explicitly:

1. constructive_pullback has two branches: an ACTIONABLE confirmed-pullback branch and a
   non-ACTIONABLE context/radar branch.
2. standard_breakout is not necessarily a breakout route; a pullback rule with fresh demand
   but no follow-through also lands in standard_breakout.

Track F therefore decomposes each review candidate into orthogonal facts before testing any
portfolio composition rule.

## Orthogonal research taxonomy

- setup_route: pullback / non_pullback
- fresh_demand: near buy point AND entry volume >= 1.5x
- follow_through: EPS >=25% OR weekly volume >=1.3x
- quality_state:
  - confirmed = fresh_demand + follow_through
  - standard = fresh_demand without follow_through
  - incomplete = no complete fresh_demand
  - failure = Production clear geometry failure
- composition_group:
  - confirmed_non_pullback
  - confirmed_pullback
  - standard
  - incomplete
  - failure

For B0-eligible ACTIONABLE candidates, current fresh_demand_alpha and constructive_pullback
should map exactly to the two confirmed route groups.

## Frozen policies

Exact Production B0 is the only comparator.

Primary policies preserve distinct_1:

1. CONFIRMED_PARITY_FALLBACK
   - route-neutral confirmed quality first;
   - standard and incomplete remain fallback tiers.
   - This is anchored to Track C PULLBACK_PARITY and must reproduce its picks exactly.

2. CONFIRMED_ONLY_TOP3
   - only confirmed candidates from either route may fill up to three slots;
   - no forced fill.

3. FCS_MAX1
   - at most one confirmed_non_pullback, one confirmed_pullback and one standard;
   - no forced fill.

Each has a secondary pure_top3 version with industry dispersion removed. Those secondary
runs diagnose Lane x industry interaction only and must not be interpreted as a pure Lane effect.

## Evaluation

All policies use the exact B0 common universe and W4 3-slot capital accounting.

Outputs include:

- policy_evaluations.csv/json
- lane_taxonomy_rows.csv
- lane_taxonomy_matrix.csv
- lane_taxonomy_summary.json
- weekly_selection_composition.csv
- route_bucket_weekly.csv
- route_pair_summary.json
- parity_anchor.json
- track_f_decision.json
- TRACK_F_LANE_COMPOSITION_REPORT.md
- run_manifest.json

The current panel is retrospective for Track F because these hypotheses were formulated after
reviewing prior Track C/D/E outcomes. Historical Track-D segment names are used only as regime
slices, not as untouched OOS evidence.

A pre-registered historical-support gate is frozen in config.py before materialization. It requires
nonnegative mean and median spreads on both the 40-week Track-D history and its 18-week forward
slice, bounded CVaR/stop/ruin degradation, sufficient support, and a bounded bootstrap CI low.
Passing this gate only creates a future-shadow candidate; it never promotes Production from
retrospective evidence.

## Gemini local execution

Gemini must only run tests/materialization and must not modify research source code.

    git checkout codex/clean-latest-quant-trade-replay-pools
    git pull --ff-only

    /Users/dev/.conda/envs/quant_env/bin/python -m pytest -q tests/test_track_f_lane_composition_audit.py

    /Users/dev/.conda/envs/quant_env/bin/python -m backtest.track_f_lane_composition_audit.cli materialize

Then run the relevant Track B/C/D/E/F regression tests.

If successful, commit only:

    backtest/track_f_lane_composition_audit/output/

If any step fails, return the exact error and do not patch source locally.
