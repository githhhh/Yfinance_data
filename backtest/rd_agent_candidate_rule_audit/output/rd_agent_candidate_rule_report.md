# Candidate-Level IBD Rule Audit and Portfolio Validation

Period label: `retrospective_final_test`. This is not a sealed holdout because the historical data has already been inspected in prior research.

## Coverage
- Candidate signal events: 2738
- Unique tickers: 1106
- Complete 8w outcomes: 1111
- Coverage regimes: early_low_coverage, late_high_coverage

## Rule Answers
1. Current pool is adequate for broad event-level diagnostics, but late/early coverage drift and censored recent weeks limit production claims.
2. ACTIONABLE keeps support as a hard eligibility boundary in current B0, but independent increment still needs prospective confirmation because status is partly derived.
3. Close > Trigger is RULE_NOT_IDENTIFIABLE in current pools: the negative observed group is absent after upstream entry logic.
4. Fresh Zone is more naturally continuous: near pivot is favorable, extension is risk, and below pivot is not proven as a universal hard gate.
5. Volume 1.5 is RULE_NOT_IDENTIFIABLE in current pools: known entry-volume observations do not include the below-1.5 comparison group.
6. Defensive Failure and Squat / Upper Shadow carry the clearest tail-risk concern; other geometry buckets should stay score/context.
7. `pullback_v_is_dry` is route-specific; base breakouts are NOT_APPLICABLE, and dry pullback is not established as a universal hard gate.
8. EPS evidence is PIT-limited: verified EPS >=25 can be a minor bonus, while unverified current-period Yahoo records are UNKNOWN before scoring.
9. Industry coverage is a risk-diversification constraint, not standalone alpha evidence.
10. Top1 is a capacity experiment and cannot replace Top3 from in-sample return.
11. Pivot +20% in three breakout weeks is a pattern-power event; it must not be confused with simulated entry +20%.
12. Eight-week hold can preserve power-trigger winners but increases capital occupancy; portfolio evidence is separated from selector evidence.
13. Eight-week hold can enlarge drawdown and idle capacity cost depending on capacity/cost settings.
14. B0 portfolio metrics are computed from an explicit equity curve: [{'total_return_pct': 36.218064, 'CAGR_pct': 46.245183, 'max_drawdown_pct': -8.350389, 'Calmar': 5.538088, 'Sharpe': 2.402883, 'Sortino': 3.881898, 'win_rate': 0.4, 'average_holding_days': 78.6, 'max_concurrent_positions': 3, 'variant': 'B0_PIT_VERIFIED', 'capacity': 3, 'cost_bps_per_side': 10, 'trades': 10, 'skipped_capacity': 74, 'stop_or_gap_stop': 4, 'profit_exits': 4, 'trade_power_triggers': 1, 'censored_positions': 2}]
15. Most balanced candidate in this run: `NO_STABLE_REPLACEMENT` with metrics []
16. No candidate rule set clears the pre-registered Pareto bar; current sample does not support replacing B0.
17. NO PRODUCTION SKILL CHANGE.
18. Prospective holdout must confirm any re-role from hard gate to score/risk/context.

## Machine Decisions
| rule_family     | machine_role   | evidence_status                               |   complete_outcomes |   independent_weeks | blocker                                                                                     | production_change   |
|:----------------|:---------------|:----------------------------------------------|--------------------:|--------------------:|:--------------------------------------------------------------------------------------------|:--------------------|
| Status          | UNKNOWN        | Insufficient Evidence                         |                 665 |                  34 |                                                                                             | False               |
| Close > Trigger | UNKNOWN        | RULE_NOT_IDENTIFIABLE                         |                 665 |                  35 | negative group not observed                                                                 | False               |
| Fresh Zone      | Context Only   | Insufficient Evidence                         |                 496 |                  35 |                                                                                             | False               |
| Entry Volume    | UNKNOWN        | RULE_NOT_IDENTIFIABLE                         |                 665 |                  34 | below 1.5 group not observed                                                                | False               |
| Geometry        | Risk Flag      | Promising / prospective confirmation required |                 665 |                  34 |                                                                                             | False               |
| Pullback        | Context Only   | Insufficient Evidence                         |                 456 |                  35 |                                                                                             | False               |
| EPS             | Context Only   | Insufficient Evidence                         |                 564 |                  34 | PIT blocker for unverified Yahoo current-period records; verified subset only.              | False               |
| Industry        | UNKNOWN        | No Treatment Contrast                         |                1111 |                  35 | Coverage rule changes opportunity set and risk concentration; not a standalone alpha claim. | False               |
| TopK            | UNKNOWN        | Insufficient Evidence                         |                   0 |                   0 | Capacity experiment only; cannot replace Top3 from in-sample return.                        | False               |

## B0 Atomic Ablation Snapshot
| variant                            |   selected_count |   added |   removed |   rank_changed |   affected_weeks |   Jaccard |   complete_outcome_count | treatment_contrast    |
|:-----------------------------------|-----------------:|--------:|----------:|---------------:|-----------------:|----------:|-------------------------:|:----------------------|
| B0 EPS >=25 hard/bonus/drop        |               84 |      31 |        44 |             23 |               26 |  0.509583 |                       38 | OK                    |
| B0 EPS unknown manual-review       |              101 |      39 |        35 |             28 |               25 |  0.5775   |                       46 | OK                    |
| B0 close trigger soft              |               97 |      32 |        32 |             30 |               19 |  0.6575   |                       45 | OK                    |
| B0 fresh continuous                |               97 |      32 |        32 |             30 |               19 |  0.6575   |                       45 | OK                    |
| B0 geometry failure only           |               97 |      32 |        32 |             30 |               19 |  0.6575   |                       45 | OK                    |
| B0 geometry soft                   |              113 |      49 |        33 |             30 |               32 |  0.511111 |                       53 | OK                    |
| B0 no entry_valid                  |               97 |      32 |        32 |             30 |               19 |  0.6575   |                       45 | OK                    |
| B0 no industry cover               |               99 |      35 |        33 |             33 |               20 |  0.646667 |                       44 | OK                    |
| B0 pullback dry bonus              |               97 |      32 |        32 |             30 |               19 |  0.6575   |                       45 | OK                    |
| B0 pullback dry drop               |               97 |      32 |        32 |             32 |               19 |  0.66     |                       45 | OK                    |
| B0 pullback dry hard               |               83 |      22 |        36 |             23 |               23 |  0.612917 |                       35 | OK                    |
| B0 status soft                     |              126 |      61 |        32 |             30 |               35 |  0.459524 |                       64 | OK                    |
| B0 status supplemental UNCONFIRMED |              126 |      61 |        32 |             30 |               35 |  0.459524 |                       66 | OK                    |
| B0 top1                            |               40 |      13 |        70 |              8 |               33 |  0.370833 |                       20 | OK                    |
| B0 volume route-specific           |               97 |      32 |        32 |             30 |               19 |  0.6575   |                       45 | OK                    |
| B0 volume soft                     |               97 |      32 |        32 |             30 |               19 |  0.6575   |                       45 | OK                    |
| B0_PIT_VERIFIED                    |               97 |       0 |         0 |              0 |                0 |  1        |                       42 | NO_TREATMENT_CONTRAST |
| B0_REPO_EXACT                      |               97 |       0 |         0 |              0 |                0 |  1        |                       42 | NO_TREATMENT_CONTRAST |
| R1_ATOMIC_IMPROVEMENTS             |               97 |      32 |        32 |             30 |               19 |  0.6575   |                       45 | OK                    |
| R2_BALANCED_SOFT                   |              126 |      62 |        33 |             30 |               36 |  0.447619 |                       62 | OK                    |

## Rule Set OOS
| variant                |   folds |   paired_weeks |   mean_oos_diff_pct |   median_oos_diff_pct |   worst_fold_diff_pct |   better_folds |
|:-----------------------|--------:|---------------:|--------------------:|----------------------:|----------------------:|---------------:|
| B0_PIT_VERIFIED        |       5 |             17 |             0       |               0       |               0       |              0 |
| B0_REPO_EXACT          |       5 |             17 |             0       |               0       |               0       |              0 |
| R1_ATOMIC_IMPROVEMENTS |       5 |             17 |            -2.90077 |              -1.09106 |              -4.73925 |              0 |
| R2_BALANCED_SOFT       |       5 |             17 |            -4.33177 |              -2.79249 |             -12.3739  |              1 |
| R3_MINIMAL_TECHNICAL   |       5 |             17 |            -4.65077 |              -1.83126 |              -9.1822  |              0 |

NO PRODUCTION SKILL CHANGE
