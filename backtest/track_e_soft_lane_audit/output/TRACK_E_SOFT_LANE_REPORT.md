# Track E — B0.1 Dry-Neutral + Soft Active-Lane Audit

## Question

Can a stronger standard_breakout outrank a weaker fresh_demand_alpha when evidence/risk is better, without removing the downgrade semantics of incomplete_evidence / tail_risk?

The challenger is intentionally singular:

- pullback_v_is_dry=True: keep positive evidence.
- pullback_v_is_dry=False: neutral; no risk penalty.
- fresh_demand_alpha, constructive_pullback, standard_breakout: soft hierarchy.
- incomplete_evidence, tail_risk: still structurally downgraded.
- distinct_1, eligibility, capital accounting and all other B0 semantics remain unchanged.

## Interpretation boundary

This hypothesis was formulated after reviewing Track D. Therefore all evidence through 2026-07-24 is retrospective mechanism evidence, not untouched OOS. Only later mature W4 snapshots may be described as post-Track-D shadow evidence.

## Segment comparison vs Production B0

| Segment | Support | Mean Δ | Median Δ | CVaR Δ | Stop Δ pp | Ruin Δ pp | Jaccard | CI low | CI high |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| discovery_train_18 | 18 | 0.9417 | 0.0000 | 0.0000 | -3.70 | -5.56 | 0.9444 | 0.0000 | 1.9819 |
| purge_4 | 4 | -1.3024 | -2.0754 | -1.0587 | 0.00 | 0.00 | 0.8000 | -1.3024 | -1.3024 |
| screening_6 | 6 | -4.7070 | -6.9738 | 3.5324 | -5.56 | 0.00 | 0.7000 | -4.7070 | 1.7239 |
| confirmation_12 | 12 | -1.2746 | -4.1297 | 0.0000 | 11.11 | 25.00 | 0.4750 | -3.6522 | 1.1497 |
| locked_forward_18 | 18 | -2.4188 | -4.1297 | 0.0000 | 5.55 | 16.67 | 0.5500 | -6.0319 | 0.1590 |
| retrospective_all_40 | 40 | -0.7949 | -2.0626 | -0.6735 | 0.83 | 5.00 | 0.7525 | -2.7493 | 0.7603 |
| post_track_d_shadow | 0 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 | 0.0000 | 0.0000 | 0.0000 |

## Decision-impact events

- Total panel snapshots: **42**
- Weeks where Top3 changed: **19**
- Weeks with targeted standard_breakout in / fresh_demand_alpha out: **0**
- Mature targeted swap weeks: **0**
- Targeted mean W4 pair delta: **N/A pp**
- Targeted median W4 pair delta: **N/A pp**
- Targeted positive pair ratio: **N/A**
- Mean portfolio W4 spread on changed weeks: **-1.7665 pp**
- Mature post-Track-D shadow weeks: **0**
- Changed post-Track-D shadow weeks: **0**

## Evidence readout

- Retrospective 40-snapshot portfolio effect: mean Δ -0.7949 pp, median Δ -2.0626 pp, CVaR Δ -0.6735 pp, stop Δ 0.83 pp.
- Track-D locked-forward 18-snapshot replay: mean Δ -2.4188 pp, median Δ -4.1297 pp, CVaR Δ 0.0000 pp.
- Former Track-D confirmation 12-snapshot replay: mean Δ -1.2746 pp, median Δ -4.1297 pp, CVaR Δ 0.0000 pp. This is not untouched for Track E.
- Post-Track-D shadow: 0 mature paired weeks, mean Δ 0.0000 pp.

## Targeted swap evidence

No mature Top3 event directly exercised standard_breakout in / fresh_demand_alpha out. The historical portfolio comparison alone cannot validate the exact crossover mechanism.

## Conclusion rule

Track E does **not** mutate Production B0. Historical results can support or reject the soft-Lane mechanism as a forward-shadow candidate, but production change requires genuinely future mature observations because this hypothesis was created after Track D.

The raw selection_events.csv is the primary audit artifact for determining whether the intended cross-Lane mechanism actually fired, rather than inferring mechanism quality only from aggregate portfolio returns.
