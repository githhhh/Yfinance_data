# Track E v3 — Pairwise Standard-vs-Fresh Top3 Replacement

Track E v1/v2 were too broad because they changed global Lane ordering. V1 was dominated by
fresh_demand_alpha -> constructive_pullback substitutions; v2 still allowed target identities
to cross fixed non-target slots.

V3 isolates the actual question at the selected-slot level.

## Primary control

Hard B0 Lane with the already-supported dry semantic:

- dry=True remains positive evidence;
- dry=False is neutral;
- hard B0 Lane ordering remains unchanged;
- distinct_1 remains unchanged.

Production B0 with the old symmetric dry penalty is retained as a secondary reference only.

## Single challenger

Start from the dry-neutral hard-Lane Top3. Preserve every non-fresh selected slot.

An unselected standard_breakout may replace a selected fresh_demand_alpha only when it
Pareto-dominates the fresh candidate on independent entry-quality axes:

- risk_count <= fresh risk_count;
- absolute current_vs_ibd_candidate_pct <= fresh distance;
- ibd_entry_volume_ratio >= fresh entry volume;
- at least one axis strictly better.

There are no fitted weights.

EPS acceleration and weekly-volume follow-through are intentionally excluded from this
pairwise dominance rule because they are the evidence that creates fresh_demand_alpha;
using them would make fresh win by definition and make the experiment circular.

## Outputs

comparison_summary.* contains two comparator views per segment:

1. challenger vs dry-neutral hard-Lane control — primary causal Lane comparison;
2. challenger vs Production B0 — secondary net production reference.

selection_events.* records:

- raw Pareto-valid opportunity count;
- actual standard-in / fresh-out Top3 replacements;
- exact matched replacement pairs and W4 deltas;
- portfolio spread vs the primary control and Production B0;
- diagnostic control-vs-production differences.

run_manifest.json seals protocol, source SHA, panel hash and Production B0 hash.

## Evidence boundary

This is a post-Track-D hypothesis. <=2026-07-24 is retrospective mechanism evidence.
Later mature W4 observations are reported separately as post_track_d_shadow.

## Gemini local run

Gemini must not modify source code.

    git checkout codex/clean-latest-quant-trade-replay-pools
    git pull --ff-only

    /Users/dev/.conda/envs/quant_env/bin/python -m pytest -q tests/test_track_e_soft_lane_audit.py

    /Users/dev/.conda/envs/quant_env/bin/python -m backtest.track_e_soft_lane_audit.cli materialize

Then run relevant regressions. If successful, commit only Track E output.
If any step fails, return the exact error and do not patch source.
