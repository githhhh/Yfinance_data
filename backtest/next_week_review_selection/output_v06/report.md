# Next Week Review Selection v0.6 — Deterministic Discriminative Study

## Decision
- static status: NO_DISCOVERY_REFINEMENT
- static rule: B0_NO_EXPANSION
- adaptive status: NO_STABLE_DISCRIMINATIVE_RULE
- rd-agent: not used
- production authorization: NO

## Methodological warning
The historical formal OOS weeks have already been observed in prior v0.5 research.
v0.6 is therefore a disciplined retrospective confirmation, not a new sealed holdout.
Any candidate still requires future sealed weeks before production use.

## Static discovery rule — formal replay
evaluation_role,folds,expanded_fold_rate,opportunity_positive_rate,winner_lift_nonnegative_rate,loser_lift_nonworse_rate,mean_opportunity_delta,mean_winner_lift_delta,mean_loser_lift_delta,mean_attention_multiplier_vs_b0,median_attention_multiplier_vs_b0,mean_incremental_opportunities_per_added_review,mean_winner_lift_delta_2w,mean_loser_lift_delta_2w,winner_nonnegative_rate_2w,loser_nonworse_rate_2w,mean_winner_lift_delta_3w,mean_loser_lift_delta_3w,winner_nonnegative_rate_3w,loser_nonworse_rate_3w,mean_winner_lift_delta_4w,mean_loser_lift_delta_4w,winner_nonnegative_rate_4w,loser_nonworse_rate_4w
STATIC_DISCOVERY_RULE,5,0.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,1.0,,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0


## Adaptive discriminative policy — formal replay
evaluation_role,folds,expanded_fold_rate,opportunity_positive_rate,winner_lift_nonnegative_rate,loser_lift_nonworse_rate,mean_opportunity_delta,mean_winner_lift_delta,mean_loser_lift_delta,mean_attention_multiplier_vs_b0,median_attention_multiplier_vs_b0,mean_incremental_opportunities_per_added_review,mean_winner_lift_delta_2w,mean_loser_lift_delta_2w,winner_nonnegative_rate_2w,loser_nonworse_rate_2w,mean_winner_lift_delta_3w,mean_loser_lift_delta_3w,winner_nonnegative_rate_3w,loser_nonworse_rate_3w,mean_winner_lift_delta_4w,mean_loser_lift_delta_4w,winner_nonnegative_rate_4w,loser_nonworse_rate_4w
ADAPTIVE_DISCRIMINATIVE_POLICY,5,0.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,1.0,,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0


## Discovery feasible/Pareto candidates
_No data_

## Fold choices
fold,asof_cutoff,train_start,train_end,test_start,test_end,static_rule,adaptive_rule,adaptive_conditions
1,2026-03-13,2025-10-10,2026-03-06,2026-03-13,2026-04-02,B0_NO_EXPANSION,B0_NO_EXPANSION,
2,2026-04-10,2025-10-10,2026-04-02,2026-04-10,2026-05-01,B0_NO_EXPANSION,B0_NO_EXPANSION,
3,2026-05-08,2025-10-10,2026-05-01,2026-05-08,2026-05-29,B0_NO_EXPANSION,B0_NO_EXPANSION,
4,2026-06-05,2025-10-10,2026-05-29,2026-06-05,2026-06-26,B0_NO_EXPANSION,B0_NO_EXPANSION,
5,2026-07-02,2025-10-10,2026-06-26,2026-07-02,2026-07-24,B0_NO_EXPANSION,B0_NO_EXPANSION,


## Adaptive rule convergence
rule,fold_count,fold_share
B0_NO_EXPANSION,5,1.0


## Setup-balanced static sensitivity
_No data_

## Static-rule moving-block bootstrap
_No data_

## Guardrails
- attention multiplier cap: <= 1.50x B0
- fixed anchor: Near5 + UNCONFIRMED/BELOW_TRIGGER + >=2 evidence families + Geometry allow
- candidate refinements use at most two coarse, interpretable PIT conditions
- first 20 weeks choose one static discovery rule; it remains frozen across all formal replay folds
- expanding-train adaptive refinement is secondary
- no ML, no rd-agent, no C Rank, no ATR, no arbitrary decimal threshold search
