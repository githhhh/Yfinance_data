# Track E v2 — Isolated Fresh-vs-Standard Lane Audit

Track E v1 was over-broad: it softened fresh_demand_alpha, constructive_pullback, and
standard_breakout together. In the materialized result every real Lane substitution was
fresh_demand_alpha -> constructive_pullback, so the intended standard-vs-fresh question
was not actually exercised.

Track E v2 fixes that design error.

## Single challenger

Production B0 is compared with one fixed B0.1 challenger:

- dry=True remains positive evidence;
- dry=False is neutral;
- build reward-only B0 ordering as a fixed rank skeleton;
- constructive_pullback, incomplete_evidence, and tail_risk keep their exact skeleton positions;
- only fresh_demand_alpha and standard_breakout are reordered among the rank slots those
  two target lanes already occupy;
- target ordering is status -> evidence/risk -> original lane -> remaining B0 tie-breaks;
- distinct_1, eligibility, and 3-slot capital accounting are unchanged.

This is a controlled mechanism intervention: no other Lane receives soft scoring.

## Audit outputs

comparison_summary.* contains paired B0 vs B0.1 portfolio metrics.

selection_events.* separately records:

- order_changed;
- membership_changed;
- target_rank_crossover (standard crossed a fresh candidate even if Top3 did not change);
- target_selection_swap (standard entered Top3 while fresh exited);
- W4 outcome deltas for both crossover levels.

run_manifest.json records the exact source SHA, panel hash, Production B0 hash, and v2 protocol.

## Evidence boundary

The hypothesis was formed after Track D, so <=2026-07-24 is retrospective mechanism
evidence. Later mature W4 snapshots are reported separately as post_track_d_shadow.

## Gemini local run

Gemini must not modify source code.

    git checkout codex/clean-latest-quant-trade-replay-pools
    git pull --ff-only

    /Users/dev/.conda/envs/quant_env/bin/python -m pytest -q tests/test_track_e_soft_lane_audit.py

    /Users/dev/.conda/envs/quant_env/bin/python -m backtest.track_e_soft_lane_audit.cli materialize

Then run regression tests. If successful, commit only generated Track E output.
If anything fails, return the exact error and do not patch source.
