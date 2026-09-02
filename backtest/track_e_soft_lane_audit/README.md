# Track E — B0.1 Dry-Neutral + Soft Active-Lane Audit

This is a focused follow-up to Track D. It does **not** modify Production B0.

## Fixed challenger

Only one challenger is evaluated:

- pullback_v_is_dry=True keeps the positive dry_pullback evidence;
- pullback_v_is_dry=False is neutral and no longer adds pullback_not_dry;
- fresh_demand_alpha, constructive_pullback, and standard_breakout are softened:
  status and evidence/risk are compared before the original lane priority;
- incomplete_evidence and tail_risk remain structurally downgraded;
- distinct_1, eligibility and 3-slot capital accounting are unchanged.

The intended mechanism is explicit: a stronger standard_breakout may outrank a weaker
fresh_demand_alpha, while incomplete/tail candidates cannot jump the active-lane guard.

## Evidence boundary

This hypothesis was formed after reviewing Track D. Therefore the 40 snapshots through
2026-07-24 are retrospective mechanism evidence, not untouched OOS. Any later mature W4
snapshots are reported separately as post_track_d_shadow.

## Outputs

- comparison_summary.csv/json: paired B0 vs B0.1 metrics by Track-D segment.
- selection_events.csv/json: every weekly Top3 comparison, including lane swaps and W4 outcomes.
- event_summary.json: counts and targeted standard-in / fresh-out deltas.
- TRACK_E_SOFT_LANE_REPORT.md: concise report.
- run_manifest.json: source SHA and input/output hashes.

## Gemini local run

Gemini must not edit source code. Run:

    git checkout codex/clean-latest-quant-trade-replay-pools
    git pull --ff-only

    /Users/dev/.conda/envs/quant_env/bin/python -m pytest -q tests/test_track_e_soft_lane_audit.py

    /Users/dev/.conda/envs/quant_env/bin/python -m backtest.track_e_soft_lane_audit.cli materialize

After success, run the relevant regression tests and commit only:

    git add backtest/track_e_soft_lane_audit/output/
    git commit -m "docs(track_e): materialize soft-lane audit artifacts"
    git push origin codex/clean-latest-quant-trade-replay-pools

If any test/materialization step fails, do not patch source locally; return the complete error.
