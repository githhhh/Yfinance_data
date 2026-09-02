# Track E v3 — Pairwise Standard-vs-Fresh Top3 Replacement

## Research question

Should an otherwise eligible standard_breakout be allowed to replace an already-selected fresh_demand_alpha when the standard candidate is unambiguously stronger on independent entry-quality axes?

## Controlled design

- Primary control: hard B0 Lane with dry=True reward and dry=False neutral.
- Production B0 is reported separately as a reference, not used to attribute the Lane effect.
- Challenger starts from the primary-control Top3.
- constructive_pullback, incomplete_evidence, tail_risk, and every non-fresh selected slot are frozen.
- Only an unselected standard_breakout may challenge a selected fresh_demand_alpha slot.
- Replacement requires unweighted Pareto dominance:
  - no more risk flags;
  - no worse absolute distance to buy point;
  - no weaker entry-volume ratio;
  - at least one of those axes strictly better.
- EPS>=25 and weekly-volume follow-through are excluded from the dominance test because they define fresh_demand_alpha itself; including them would make the test circular.
- distinct_1 and portfolio capacity are preserved.

This corrects Track E v1/v2, which altered global Lane ordering instead of isolating the specific Top3 replacement question.

## Evidence boundary

This is a post-Track-D hypothesis. Evidence through 2026-07-24 is retrospective mechanism evidence, not untouched OOS.

## Primary comparison — challenger vs dry-neutral hard-Lane control

| Segment | Support | Mean Δ | Median Δ | CVaR Δ | Stop Δ pp | Ruin Δ pp | Jaccard | CI low | CI high |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| discovery_train_18 | 18 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 | 1.0000 | 0.0000 | 0.0000 |
| purge_4 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 | 1.0000 | 0.0000 | 0.0000 |
| screening_6 | 6 | -0.3616 | -1.0847 | 0.0000 | 5.55 | 16.67 | 0.9167 | -0.3616 | 0.0000 |
| confirmation_12 | 12 | 0.2965 | -0.2852 | 0.0000 | 2.78 | 8.33 | 0.7667 | -0.5658 | 0.8958 |
| locked_forward_18 | 18 | 0.0772 | -0.2852 | 0.0000 | 3.70 | 11.11 | 0.8167 | -0.4764 | 0.7284 |
| retrospective_all_40 | 40 | 0.0347 | -0.4185 | 0.0000 | 1.67 | 5.00 | 0.9175 | -0.2082 | 0.2847 |
| post_track_d_shadow | 0 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 | 0.0000 | 0.0000 | 0.0000 |

## Mechanism support

- Total snapshots: **42**
- Weeks with at least one Pareto-valid standard-vs-fresh opportunity: **7**
- Pareto-valid candidate pairs: **12**
- Weeks with an actual Top3 replacement: **7**
- Actual replacement pairs: **8**
- Mature replacement weeks: **6**
- Mature replacement pairs: **7**
- Replacement-pair mean W4 Δ: **0.5953 pp**
- Replacement-pair median W4 Δ: **-4.7345 pp**
- Replacement-pair positive ratio: **0.4286**
- Mean portfolio W4 Δ on membership-changed weeks: **0.2315 pp**

## Dry-neutral control vs Production B0 diagnostic

- Order-changed weeks: **2**
- Membership-changed weeks: **0**

The production-reference comparison is retained to show the net B0.1 effect, but the primary Lane conclusion must use the dry-neutral hard-Lane control above.

## Production B0 reference

| Segment | Support | Mean Δ | Median Δ | CVaR Δ | Stop Δ pp | Ruin Δ pp | Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| discovery_train_18 | 18 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 | 1.0000 |
| purge_4 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 | 1.0000 |
| screening_6 | 6 | -0.3616 | -1.0847 | 0.0000 | 5.55 | 16.67 | 0.9167 |
| confirmation_12 | 12 | 0.2965 | -0.2852 | 0.0000 | 2.78 | 8.33 | 0.7667 |
| locked_forward_18 | 18 | 0.0772 | -0.2852 | 0.0000 | 3.70 | 11.11 | 0.8167 |
| retrospective_all_40 | 40 | 0.0347 | -0.4185 | 0.0000 | 1.67 | 5.00 | 0.9175 |
| post_track_d_shadow | 0 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 | 0.0000 |

## Mature targeted swaps

| Snapshot | Segment | Swap pairs | Portfolio Δ vs control |
| --- | --- | --- | ---: |
| 2026-05-01 | screening | [{"fresh_code": "BEN", "fresh_w4": 4.2077, "pair_delta_w4": -6.5085, "slot": 3, "standard_code": "GVA", "standard_w4": -2.3008}] | -2.1695 |
| 2026-05-15 | confirmation | [{"fresh_code": "AIZ", "fresh_w4": 2.8735, "pair_delta_w4": 13.5915, "slot": 3, "standard_code": "ALKS", "standard_w4": 16.465}] | 4.5305 |
| 2026-05-29 | confirmation | [{"fresh_code": "NBIX", "fresh_w4": 5.748, "pair_delta_w4": -8.1485, "slot": 2, "standard_code": "SHEN", "standard_w4": -2.4005}, {"fresh_code": "ILMN", "fresh_w4": 9.7606, "pair_delta_w4": 11.2839, "slot": 3, "standard_code": "PKOH", "standard_w4": 21.0445}] | 1.0452 |
| 2026-06-12 | confirmation | [{"fresh_code": "CWBC", "fresh_w4": 4.1877, "pair_delta_w4": -4.8469, "slot": 2, "standard_code": "FCBC", "standard_w4": -0.6592}] | -1.6156 |
| 2026-07-02 | confirmation | [{"fresh_code": "PACS", "fresh_w4": 3.7664, "pair_delta_w4": -4.7345, "slot": 3, "standard_code": "TRIN", "standard_w4": -0.9681}] | -1.5782 |
| 2026-07-24 | confirmation | [{"fresh_code": "PKG", "fresh_w4": -1.0723, "pair_delta_w4": 3.5298, "slot": 3, "standard_code": "NWFL", "standard_w4": 2.4575}] | 1.1766 |

## Interpretation rule

The result is only informative if Pareto-valid opportunities and actual replacements exist. If they do, judge the mechanism first on matched replacement-pair W4 deltas and then on portfolio mean/median/CVaR/stop behavior. Historical evidence can justify a forward shadow, but cannot by itself promote a production Lane change because the hypothesis was formed after Track D.

Production B0 remains untouched.
