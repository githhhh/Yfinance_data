# Track F — Lane Taxonomy & Composition Audit

## Why this track exists

Production B0 currently uses one Lane enum to represent several different concepts at once: setup route, evidence completeness, actionability context, and failure state. Track F does not change Production. It first decomposes Lane into orthogonal facts, then tests a small frozen set of composition policies against exact Production B0.

## Orthogonal interpretation

- setup_route: non_pullback / pullback
- fresh_demand: near buy point AND entry volume >= 1.5x
- follow_through: EPS >= 25% OR weekly volume >= 1.3x
- quality_state: confirmed / standard / incomplete / failure

For B0-eligible ACTIONABLE rows:

- fresh_demand_alpha = confirmed + non_pullback route
- constructive_pullback = confirmed + pullback route
- standard_breakout = standard quality; route may be pullback or non_pullback
- incomplete_evidence = incomplete quality
- tail_risk = geometry failure

The non-ACTIONABLE constructive_pullback context branch is recorded separately because it is a different semantic path and cannot enter Production Top3 eligibility.

## Taxonomy audit

- Total review rows: **2738**
- B0-eligible rows: **414**
- constructive_pullback rows: **1162**
- constructive actionable-confirmed branch rows: **258**
- constructive non-actionable-context branch rows: **904**
- constructive other rows: **0**
- eligible standard_breakout rows from pullback route: **25**
- eligible standard_breakout rows from non-pullback route: **11**

## Candidate-pool route diagnostic

This is descriptive supporting evidence, not a portfolio promotion gate. On weeks where both confirmed route groups have fully mature W4 outcomes:

- paired support weeks: **23**
- mean pullback minus non-pullback W4: **-5.2331 pp**
- median pullback minus non-pullback W4: **-6.2673 pp**
- positive-week ratio: **0.3043**
- mean stop-rate delta: **2.3417 pp**

## Frozen primary policies vs Production B0

All primary policies preserve Production B0 eligibility, symmetric dry semantics, 3-slot capital accounting, and distinct_1 industry dispersion.

- CONFIRMED_PARITY_FALLBACK: confirmed pullback/non-pullback have equal route priority; standard/incomplete remain fallback quality tiers.
- CONFIRMED_ONLY_TOP3: Top3 may come only from confirmed candidates of either route; no forced fill.
- FCS_MAX1: max one confirmed_non_pullback, one confirmed_pullback, one standard; no forced fill.

### retrospective_track_d_40

| Policy | Mean Δ | Median Δ | CVaR Δ | Stop Δ pp | Ruin Δ pp | Coverage | Full Top3 | Jaccard | CI low | CI high |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TRACK_F__CONFIRMED_PARITY_FALLBACK | -0.9673 | -2.5208 | -1.0315 | -0.00 | 2.50 | 75.83 | 57.50 | 0.7150 | -3.3501 | 0.7171 |
| TRACK_F__CONFIRMED_ONLY_TOP3 | -1.0242 | -2.7486 | -0.0198 | -0.83 | 2.50 | 71.67 | 52.50 | 0.6525 | -3.3752 | 0.7783 |
| TRACK_F__FCS_MAX1 | -1.4572 | -2.3234 | 1.5604 | -1.67 | 0.00 | 63.33 | 27.50 | 0.5800 | -3.8178 | 0.1366 |

## Secondary industry diagnostic

These policies use identical Lane logic but remove distinct_1. They are diagnostic only; they must not be used to attribute a Lane effect because Lane composition and industry concentration both change relative to Production B0.

| Policy | Mean Δ | Median Δ | CVaR Δ | Stop Δ pp | Ruin Δ pp | Coverage | Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TRACK_F__CONFIRMED_PARITY_FALLBACK_NO_IND | -1.1211 | -1.9481 | -2.1796 | 1.67 | 5.00 | 77.50 | 0.6767 |
| TRACK_F__CONFIRMED_ONLY_TOP3_NO_IND | -1.1780 | -2.5208 | -1.2851 | 0.83 | 5.00 | 73.33 | 0.6142 |
| TRACK_F__FCS_MAX1_NO_IND | -1.1526 | -2.3234 | 0.8881 | 0.00 | 2.50 | 64.17 | 0.5675 |

## Integrity anchor

Track F CONFIRMED_PARITY_FALLBACK is expected to reproduce the existing Track C PULLBACK_PARITY selector under distinct_1. This anchors the new orthogonal taxonomy to the previously tested structural policy.

- snapshots checked: **42**
- pick mismatches: **0**

## Pre-registered historical support decision

- Overall: **RETAIN_B0_WITHIN_TRACK_F_TESTED_COMPOSITIONS**
- Historical shadow candidates: **[]**

This gate is retrospective-only. Passing it cannot promote Production; it only identifies a policy worth observing on future unseen weeks.

## Evidence boundary

All currently materialized panel outcomes are retrospective for Track F because the Lane-composition hypotheses were formulated after observing prior Track C/D/E results. Track-D segment labels are reused only for regime consistency, not as untouched OOS.

## Interpretation rule

Track F is a mechanism/composition audit, not a search for the best historical policy. No thresholds are fitted and no policy is generated after seeing outcomes. A useful result must show coherent improvement across median/mean, downside and coverage without relying on one historical segment. Production B0 remains untouched.
