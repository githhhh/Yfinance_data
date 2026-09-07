# Track C Final Research Report: Modular Discovery & Counterfactual Attribution of B0 Ranking and Portfolio Construction

## Executive Summary
> **Core Research Objective**: Evaluate the structural foundations of the B0 baseline by modularly decoupling Candidate Ranking, Industry Allocation, and Within-Industry Stock Selection. Conduct k-matched 5,000-path Monte Carlo counterfactual attribution under strict selection-first maturity-second protocol, enforce blind hypothesis generation before outcome evaluation, and test alternative decision policies across 5 pre-registered discovery families under 3-slot capital accounting.

### Key Findings:
1. **B0 Alpha Attribution (2x2 Counterfactual Matrix)**:
   - **B0-Induced Industry Allocation Effect**: +1.2409% (Null 1) vs +1.1034% (Null 2)
   - **Conditional Stock Selection Effect**: +1.0317% (Null 1) vs +1.0215% (Null 2)
   - **Interaction Effect**: +0.4892% (Null 1) vs +0.1805% (Null 2)
   - **B0 Paired Full-Path Percentile**: on pathwise identical mature support, B0 beat **98.7%** of Null 1 paths and **98.3%** of Null 2 paths.

2. **Locked Observed Re-Validation Results**:
| selector_id | family | classification | val_support_weeks | val_mean_spread | val_median_spread | val_cvar_delta | val_stop_delta_pct | val_slot_coverage_pct | val_full_top3_rate_pct | val_top3_jaccard_vs_b0 | val_ci_low | val_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CONT_SCORE__atr_adjusted_entry_quality | continuous | UNSTABLE (NOT ROBUST ON OBSERVED VALIDATION) | 9 | -4.0438 | -5.6366 | 0.8126 | 18.52 | 100.0 | 100.0 | 0.2889 | -7.1595 | -0.9488 |
| IND_BREADTH__volume_based_two_position | industry_breadth | UNSTABLE (NOT ROBUST ON OBSERVED VALIDATION) | 9 | 0.7652 | -7.5105 | -1.0231 | 25.93 | 100.0 | 100.0 | 0.2889 | -4.72 | 3.0423 |
| LINEAR_RANK__actionable_ibd_quality | linear_ranking | UNSTABLE (NOT ROBUST ON OBSERVED VALIDATION) | 9 | -2.9015 | -3.7025 | -2.601 | 18.52 | 100.0 | 100.0 | 0.2667 | -3.808 | -1.1417 |
| NOVEL_HEURISTIC__shallow_base_strict_selection | novel_heuristic | UNSTABLE (NOT ROBUST ON OBSERVED VALIDATION) | 9 | -1.3972 | -6.0028 | 6.0734 | 25.93 | 100.0 | 100.0 | 0.1556 | -5.7566 | 0.4381 |
| PORT_UTIL__momentum_first_concentrator | portfolio | UNSTABLE (NOT ROBUST ON OBSERVED VALIDATION) | 9 | -4.7585 | -3.4509 | -8.4118 | 22.22 | 100.0 | 100.0 | 0.1667 | -6.2248 | -1.8461 |
| STRUCTURAL__SCORE_BEFORE_LANE__symmetric__distinct_1 | structural | UNSTABLE (NOT ROBUST ON OBSERVED VALIDATION) | 9 | 0.1689 | -2.1803 | 0.0 | 7.41 | 100.0 | 100.0 | 0.4111 | -1.3845 | 2.6282 |

3. **Overall Research Verdict**:
   - **State C: No robust evidence against B0 within the tested search space; retain B0 operationally.**
   - No locked challenger established a robust replacement for B0 on the observed re-validation window; some showed negative mean/median spreads, while others showed trade-offs or insufficient robustness.
   - RD-Agent natural-language hypotheses are provenance/rationale only. Every tested discovery policy is defined exclusively by its normalized frozen spec_params and common B0 eligibility universe.
   - The complete B0 construction remained the operational baseline within the tested search space; individual component contributions are assessed separately by the pre-registered Structural Ablation Grid.

## Provenance & Integrity Ledger
```json
{
  "challenge_package_hash": "44b5389b3de16b56e0045af492048d13192a6c27f1b924373865c008e845a7f6",
  "feature_manifest_hash": "a161edccf0f54ebe8d58d062553f8674f163cd630da67bab3648899019f913eb",
  "protocol_hash": "0da0b1930d6d48b2de16be232f563f062d32d8c1dfc64ccfe64b1f6d96223848",
  "b0_ablation_grid_hash": "c6e66d890f270441f9d8b44784f0c8a4252ca39b76321dae9c1fc7a2b8350032",
  "counterfactual_engine_hash": "f7deaea191e564051477e9efc77587a76fe7cca284b873995dc5f9e793d2a642",
  "evaluate_econometrics_hash": "9a6e9f77e02da990d9be9ee51e7b1322741f614ab144ba987e4d732135b9d2b0",
  "discovery_runner_hash": "6a81b09fb93620ed41fb78cb8e57e58fff46e92bbf2c4794e8304479ac499b2d",
  "rdagent_policy_bridge_hash": "bfe7ec632107465f7a7a3b88d033a215f2fb6e39a9f05f9d6098410952aebc90",
  "blind_prompt_hash": "eb4831a34c02393f15b44a990fdad23c3e6899ab172e7dd2351cd4e51444dcb9",
  "behavioral_dedup_hash": "d50b6c5c98464e2563c33268d186e23b447df904ac615fcbfcb5e0821cbad589",
  "production_b0_hash": "115387c9861f7202c0f6b3c89fe2d2ff594544de93264901ee6d2f72e930c477",
  "panel_hash": "d95ab4ba21831f72fdc2e434c49a42e6164d7fde6f72d187ad758f996347b5b5",
  "codebase_hash": "f20f50cfcd7e7d7083fc60029883a7982d8c320242d4ee4bd4d4f4488e2b5af4"
}
```