# Track C Final Research Report: Modular Discovery & Counterfactual Attribution of B0 Ranking and Portfolio Construction

## Executive Summary
> **Core Research Objective**: Evaluate the structural foundations of the B0 baseline by modularly decoupling Candidate Ranking, Industry Allocation, and Within-Industry Stock Selection. Conduct k-matched 5,000-path Monte Carlo counterfactual attribution under strict selection-first maturity-second protocol, enforce blind hypothesis generation before outcome evaluation, and test alternative decision policies across 5 pre-registered discovery families under 3-slot capital accounting.

### Key Findings:
1. **B0 Alpha Attribution (2x2 Counterfactual Matrix)**:
   - **B0-Induced Industry Allocation Effect**: +1.1609% (Null 1) vs +1.0322% (Null 2)
   - **Conditional Stock Selection Effect**: +0.9652% (Null 1) vs +0.9556% (Null 2)
   - **Interaction Effect**: +0.4577% (Null 1) vs +0.1689% (Null 2)
   - **B0 Full-Path Percentile**: B0 ranked at the **98.7th percentile** (Null 1) and **98.3th percentile** (Null 2) across 5,000 full historical simulation paths.

2. **Locked Observed Re-Validation Results**:
| selector_id | family | classification | val_support_weeks | val_mean_spread | val_median_spread | val_cvar_delta | val_stop_delta_pct | val_slot_coverage_pct | val_full_top3_rate_pct | val_top3_jaccard_vs_b0 | val_ci_low | val_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CONT_SCORE__tight_structure | continuous | UNSTABLE (OOS DEGRADATION) | 6 | -3.084 | -5.4516 | 5.6256 | 16.67 | 100.0 | 100.0 | 0.15 | -3.084 | 3.8485 |
| IND_BREADTH__breadth_volume_breadth_dynFalse_min2 | industry_breadth | UNSTABLE (OOS DEGRADATION) | 9 | -0.7849 | -4.2774 | 0.6157 | 33.33 | 100.0 | 100.0 | 0.2 | -5.4203 | 0.1483 |
| LINEAR_RANK__linear_geom | linear_ranking | UNSTABLE (OOS DEGRADATION) | 7 | -2.1292 | -3.1441 | 11.9913 | 14.29 | 100.0 | 100.0 | 0.0571 | -3.5082 | 2.4566 |
| NOVEL_HEURISTIC__dry_heavy_max2 | novel_heuristic | UNSTABLE (OOS DEGRADATION) | 8 | -8.0448 | -7.2338 | -4.9952 | 29.17 | 100.0 | 100.0 | 0.025 | -9.5131 | -3.2118 |
| PORT_UTIL__lambda_0_5_mom | portfolio | UNSTABLE (OOS DEGRADATION) | 7 | -7.3119 | -3.4509 | -8.4118 | 23.81 | 100.0 | 100.0 | 0.0857 | -9.1053 | -2.217 |
| STRUCTURAL__SCORE_BEFORE_LANE__symmetric__distinct_1 | structural | UNSTABLE (OOS DEGRADATION) | 9 | 0.1689 | -2.1803 | 0.0 | 7.41 | 100.0 | 100.0 | 0.4111 | -1.3845 | 2.6282 |

3. **Overall Research Verdict**:
   - **State C: No robust evidence against B0 within the tested search space; retain B0 operationally.**
   - All multi-feature linear scoring, continuous, and novel heuristic challengers exhibited out-of-sample degradation on the locked re-validation window.
   - The complete B0 construction (incorporating lexicographic ranking and distinct_1 industry constraint) exhibited superior risk-adjusted stability; individual component contributions are detailed via the pre-registered Structural Ablation Grid.

## Provenance & Integrity Ledger
```json
{
  "codebase_hash": "3b40ca0824f8ef0bde6e503ed14c552b58366140649f1322399cb6d28f7d883f",
  "challenge_package_hash": "977afb39550f1795925d07d4876c6a9330f9711a08ec58246cd8c48be6f53564",
  "feature_manifest_hash": "a161edccf0f54ebe8d58d062553f8674f163cd630da67bab3648899019f913eb",
  "protocol_hash": "0da0b1930d6d48b2de16be232f563f062d32d8c1dfc64ccfe64b1f6d96223848",
  "b0_ablation_grid_hash": "c6e66d890f270441f9d8b44784f0c8a4252ca39b76321dae9c1fc7a2b8350032",
  "counterfactual_engine_hash": "eea0dad23d13c6e300fa743ac99341ad592e489107f7052044c463db010d541d",
  "evaluate_econometrics_hash": "bd864678483916f713ee50d262bd4c1484895ddd592ea1c2a6ddedaae79a746c",
  "discovery_runner_hash": "e560cbccd1b688323e38a542c967e248257d0c295ab6de9f1ef964c69d2ded7e",
  "production_b0_hash": "115387c9861f7202c0f6b3c89fe2d2ff594544de93264901ee6d2f72e930c477",
  "panel_hash": "d95ab4ba21831f72fdc2e434c49a42e6164d7fde6f72d187ad758f996347b5b5"
}
```