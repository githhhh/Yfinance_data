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
| CONT_SCORE__pullback_dry_volume_entry | continuous | UNSTABLE (OOS DEGRADATION) | 8 | -1.6774 | -3.0797 | 16.1435 | -0.0 | 100.0 | 100.0 | 0.0 | -4.738 | 3.9369 |
| IND_BREADTH__pullback_dryness_breadth | industry_breadth | UNSTABLE (OOS DEGRADATION) | 9 | -0.7849 | -4.2774 | 0.6157 | 33.33 | 100.0 | 100.0 | 0.2 | -5.4203 | 0.1483 |
| LINEAR_RANK__ibd_candidate_tight_entry | linear_ranking | UNSTABLE (OOS DEGRADATION) | 9 | 0.2651 | -5.853 | -5.959 | 25.93 | 100.0 | 100.0 | 0.1333 | -8.1704 | 0.7833 |
| NOVEL_HEURISTIC__dry_pullback_volume_breakout | novel_heuristic | UNSTABLE (OOS DEGRADATION) | 8 | -8.0448 | -7.2338 | -4.9952 | 29.17 | 100.0 | 100.0 | 0.025 | -9.5131 | -3.2118 |
| PORT_UTIL__conservative_momentum_concentrated | portfolio | UNSTABLE (OOS DEGRADATION) | 7 | -7.3119 | -3.4509 | -8.4118 | 23.81 | 100.0 | 100.0 | 0.0857 | -9.1053 | -2.217 |
| STRUCTURAL__SCORE_BEFORE_LANE__symmetric__distinct_1 | structural | UNSTABLE (OOS DEGRADATION) | 9 | 0.1689 | -2.1803 | 0.0 | 7.41 | 100.0 | 100.0 | 0.4111 | -1.3845 | 2.6282 |

3. **Overall Research Verdict**:
   - **State C: No robust evidence against B0 within the tested search space; retain B0 operationally.**
   - All multi-feature linear scoring, continuous, and novel heuristic challengers exhibited out-of-sample degradation on the locked re-validation window.
   - The complete B0 construction (incorporating lexicographic ranking and distinct_1 industry constraint) exhibited superior risk-adjusted stability; individual component contributions are detailed via the pre-registered Structural Ablation Grid.

## Provenance & Integrity Ledger
```json
{
  "challenge_package_hash": "1a27c415a6914e4504b74d3647c314f559c719668ecdf78375b37075a4780544",
  "feature_manifest_hash": "a161edccf0f54ebe8d58d062553f8674f163cd630da67bab3648899019f913eb",
  "protocol_hash": "0da0b1930d6d48b2de16be232f563f062d32d8c1dfc64ccfe64b1f6d96223848",
  "b0_ablation_grid_hash": "c6e66d890f270441f9d8b44784f0c8a4252ca39b76321dae9c1fc7a2b8350032",
  "counterfactual_engine_hash": "f7deaea191e564051477e9efc77587a76fe7cca284b873995dc5f9e793d2a642",
  "evaluate_econometrics_hash": "bd864678483916f713ee50d262bd4c1484895ddd592ea1c2a6ddedaae79a746c",
  "discovery_runner_hash": "805491d4793b3cf45c6a4ff060ec93b2aba5536f9ec86ebdbffd7c7166c969de",
  "rdagent_policy_bridge_hash": "bfe7ec632107465f7a7a3b88d033a215f2fb6e39a9f05f9d6098410952aebc90",
  "blind_prompt_hash": "ccc5e60c3a481a2afeb9941b6d31afc2ad1637219dd2b19a44a5c86545fdbd3a",
  "behavioral_dedup_hash": "d50b6c5c98464e2563c33268d186e23b447df904ac615fcbfcb5e0821cbad589",
  "production_b0_hash": "115387c9861f7202c0f6b3c89fe2d2ff594544de93264901ee6d2f72e930c477",
  "panel_hash": "d95ab4ba21831f72fdc2e434c49a42e6164d7fde6f72d187ad758f996347b5b5",
  "codebase_hash": "3e454bd2cadb14673f8aca9987ac5a6aea93f1d25418414a9fe352cfb395ac11"
}
```