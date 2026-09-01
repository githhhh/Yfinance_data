# Track C Final Research Report: Modular Discovery & Counterfactual Attribution of B0 Ranking and Portfolio Construction

## Executive Summary
> **Core Research Objective**: Evaluate the structural foundations of the B0 baseline by modularly decoupling Candidate Ranking, Industry Allocation, and Within-Industry Stock Selection. Conduct k-matched 5,000-path Monte Carlo counterfactual attribution, enforce blind hypothesis generation before outcome evaluation, and test alternative decision policies across 6 families under 3-slot capital accounting.

### Key Findings:
1. **B0 Alpha Attribution (2x2 Counterfactual Matrix)**:
   - **B0-Induced Industry Allocation Effect**: +1.1609% (Null 1) vs +1.0322% (Null 2)
   - **Conditional Stock Selection Effect**: +0.9652% (Null 1) vs +0.9556% (Null 2)
   - **Interaction Effect**: +0.4577% (Null 1) vs +0.1689% (Null 2)
   - **B0 Full-Path Percentile**: B0 ranked at the **98.7th percentile** (Null 1) and **98.3th percentile** (Null 2) across 5,000 full historical paths.

2. **Locked Observed Re-Validation Results**:
| selector_id | family | classification | val_support_weeks | val_mean_spread | val_median_spread | val_cvar_delta | val_stop_delta_pct | val_slot_coverage_pct | val_full_top3_rate_pct | val_top3_jaccard_vs_b0 | val_ci_low | val_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CONT_SCORE__tight_structure | continuous | UNSTABLE (OOS DEGRADATION) | 6 | -3.084 | -5.4516 | 5.6256 | 16.67 | 100.0 | 100.0 | 0.15 | -3.084 | 3.8485 |
| IND_BREADTH__breadth_volume_breadth_dynTrue_min2 | industry_breadth | UNSTABLE (OOS DEGRADATION) | 9 | 2.3211 | -1.2364 | 0.6157 | 25.93 | 100.0 | 100.0 | 0.1333 | -4.9352 | 4.0166 |
| LTR__pairwise_geom | ltr | UNSTABLE (OOS DEGRADATION) | 7 | -2.1292 | -3.1441 | 11.9913 | 14.29 | 100.0 | 100.0 | 0.0571 | -3.5082 | 2.4566 |
| NOVEL__dry_heavy_max2 | novel | UNSTABLE (OOS DEGRADATION) | 8 | -8.0448 | -7.2338 | -4.9952 | 29.17 | 100.0 | 100.0 | 0.025 | -9.5131 | -3.2118 |
| PORT_UTIL__lambda_0_5_mom | portfolio | UNSTABLE (OOS DEGRADATION) | 7 | -7.3119 | -3.4509 | -8.4118 | 23.81 | 100.0 | 100.0 | 0.0857 | -9.1053 | -2.217 |
| STRUCTURAL__B0_LANE__symmetric__distinct_1 | structural | EQUIVALENT TO B0 | 9 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 | 1.0 | 0.0 | 0.0 |

3. **Overall Research Verdict**:
   - **State C: No robust evidence against B0 within the tested search space; retain B0 operationally.**
   - All complex LTR / ML models suffered out-of-sample degradation on the re-validation window.
   - Structural variants confirmed that B0's distinct_1 industry constraint and lexicographic ordering provide stable downside control.

## Provenance & Integrity Ledger
```json
{
  "codebase_hash": "8ee1d6772d893faef75caf99dbd73ef9a611ab0df748c959652897cce944c4c0",
  "challenge_package_hash": "f9d2ddad5db2ebeed7ffdc536abd774652a8542baeadbf85b1727ddd975d4742",
  "production_b0_hash": "115387c9861f7202c0f6b3c89fe2d2ff594544de93264901ee6d2f72e930c477"
}
```