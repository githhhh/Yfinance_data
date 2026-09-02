# Track E v2 — Isolated Fresh-vs-Standard Lane Audit

## Fixed question

Can a stronger standard_breakout outrank a weaker fresh_demand_alpha when its status/evidence/risk profile is better?

## Controlled intervention

- pullback_v_is_dry=True remains positive evidence.
- pullback_v_is_dry=False is neutral.
- Build the reward-only B0 ranking skeleton first.
- Keep constructive_pullback, incomplete_evidence, and tail_risk at their exact skeleton positions.
- Reorder only fresh_demand_alpha and standard_breakout candidates among the slots those two lanes already occupy.
- Within those target slots compare status -> evidence/risk -> original Lane -> remaining B0 tie-breaks.
- distinct_1, eligibility, and 3-slot capital accounting remain unchanged.

This fixes Track E v1, where constructive_pullback was also softened and dominated every actual crossover.

## Evidence boundary

The hypothesis is post-Track-D. Evidence through 2026-07-24 is retrospective mechanism evidence, not untouched OOS.

## Paired portfolio comparison vs Production B0

| Segment | Support | Mean Δ | Median Δ | CVaR Δ | Stop Δ pp | Ruin Δ pp | Jaccard | CI low | CI high |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| discovery_train_18 | 18 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 | 1.0000 | 0.0000 | 0.0000 |
| purge_4 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 | 1.0000 | 0.0000 | 0.0000 |
| screening_6 | 6 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 | 1.0000 | 0.0000 | 0.0000 |
| confirmation_12 | 12 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 | 1.0000 | 0.0000 | 0.0000 |
| locked_forward_18 | 18 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 | 1.0000 | 0.0000 | 0.0000 |
| retrospective_all_40 | 40 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 | 1.0000 | 0.0000 | 0.0000 |
| post_track_d_shadow | 0 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 | 0.0000 | 0.0000 | 0.0000 |

## Mechanism-trigger audit

- Total snapshots: **42**
- Order-changed weeks: **2**
- Top3 membership-changed weeks: **0**
- Mature membership-changed weeks: **0**
- Fresh/standard rank-crossover weeks: **2**
- Mature rank-crossover weeks: **2**
- Rank-crossover mean W4 pair Δ: **-12.3328 pp**
- Rank-crossover median W4 pair Δ: **-12.3328 pp**
- Rank-crossover positive ratio: **0.0000**
- Top3 standard-in / fresh-out weeks: **0**
- Mature Top3 targeted swaps: **0**
- Top3 targeted mean W4 pair Δ: **N/A pp**
- Top3 targeted median W4 pair Δ: **N/A pp**
- Top3 targeted positive ratio: **N/A**
- Mean portfolio W4 spread on membership-changed weeks: **N/A pp**

## Targeted selection swaps

No mature Top3 standard-in / fresh-out event occurred.

## Interpretation rule

Track E v2 is a mechanism audit, not an automatic production promotion. The primary question is first whether the intended fresh/standard crossover actually fires, then whether the resulting rank/selection substitutions improve W4 outcome without degrading downside metrics.

Production B0 remains untouched.
