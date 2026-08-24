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
3. Close > Trigger is better treated as a risk flag/score input than promoted solely from this retrospective sample.
4. Fresh Zone is more naturally continuous: near pivot is favorable, extension is risk, and below pivot is not proven as a universal hard gate.
5. Volume 1.5 shows useful signal but no registered evidence for decimal-precise optimization; soft or route-specific treatment is more defensible.
6. Defensive Failure and Squat / Upper Shadow carry the clearest tail-risk concern; other geometry buckets should stay score/context.
7. `pullback_v_is_dry` is route-specific; base breakouts are NOT_APPLICABLE, and dry pullback is not established as a universal hard gate.
8. EPS evidence is PIT-limited: verified EPS >=25 can be a minor bonus, while unverified current-period Yahoo records are UNKNOWN before scoring.
9. Industry coverage is a risk-diversification constraint, not standalone alpha evidence.
10. Top1 is a capacity experiment and cannot replace Top3 from in-sample return.
11. Pivot +20% in three breakout weeks is a pattern-power event; it must not be confused with simulated entry +20%.
12. Eight-week hold can preserve power-trigger winners but increases capital occupancy; portfolio evidence is separated from selector evidence.
13. Eight-week hold can enlarge drawdown and idle capacity cost depending on capacity/cost settings.
14. B0 portfolio metrics are computed from an explicit equity curve: [{'total_return_pct': 2.461538, 'CAGR_pct': 1.225819, 'max_drawdown_pct': -8.536603, 'Calmar': 0.143596, 'Sharpe': 0.323661, 'Sortino': 0.221097, 'win_rate': 0.333333, 'average_holding_days': 111.0, 'max_concurrent_positions': 3, 'variant': 'B0_PIT_VERIFIED', 'capacity': 3, 'cost_bps_per_side': 10, 'trades': 3, 'skipped_capacity': 16, 'stop_or_gap_stop': 2, 'profit_exits': 1, 'trade_power_triggers': 0, 'censored_positions': 0}]
15. Most balanced candidate in this run: `R1_ATOMIC_IMPROVEMENTS` with metrics [{'total_return_pct': 2.461538, 'CAGR_pct': 1.225819, 'max_drawdown_pct': -8.536603, 'Calmar': 0.143596, 'Sharpe': 0.323661, 'Sortino': 0.221097, 'win_rate': 0.333333, 'average_holding_days': 111.0, 'max_concurrent_positions': 3, 'variant': 'R1_ATOMIC_IMPROVEMENTS', 'capacity': 3, 'cost_bps_per_side': 10, 'trades': 3, 'skipped_capacity': 16, 'stop_or_gap_stop': 2, 'profit_exits': 1, 'trade_power_triggers': 0, 'censored_positions': 0}]
16. Improvements, where present, come from selector softening plus exit-policy interaction; this harness reports them separately.
17. NO PRODUCTION SKILL CHANGE.
18. Prospective holdout must confirm any re-role from hard gate to score/risk/context.

## Machine Decisions
| rule_family     | machine_role     | evidence_status                               |   complete_outcomes |   independent_weeks | blocker                                                                                     | production_change   |
|:----------------|:-----------------|:----------------------------------------------|--------------------:|--------------------:|:--------------------------------------------------------------------------------------------|:--------------------|
| Status          | Hard Eligibility | Promising / prospective confirmation required |                1111 |                  42 |                                                                                             | False               |
| Close > Trigger | Risk Flag        | Promising / prospective confirmation required |                1111 |                  42 |                                                                                             | False               |
| Fresh Zone      | Major Score      | Promising / prospective confirmation required |                1111 |                  42 |                                                                                             | False               |
| Entry Volume    | Major Score      | Promising / prospective confirmation required |                1111 |                  42 |                                                                                             | False               |
| Geometry        | Risk Flag        | Promising / prospective confirmation required |                1111 |                  42 |                                                                                             | False               |
| Pullback        | Route-specific   | Promising / prospective confirmation required |                1111 |                  42 |                                                                                             | False               |
| EPS             | Minor Bonus      | Promising / prospective confirmation required |                1111 |                  42 | PIT blocker for unverified Yahoo current-period records; verified subset only.              | False               |
| Industry        | Context Only     | Promising / prospective confirmation required |                1111 |                  42 | Coverage rule changes opportunity set and risk concentration; not a standalone alpha claim. | False               |
| TopK            | UNKNOWN          | Insufficient Evidence                         |                   0 |                   0 | Capacity experiment only; cannot replace Top3 from in-sample return.                        | False               |

## B0 Atomic Ablation Snapshot
| variant                            |   selected_count |   added |   removed |   rank_changed |   affected_weeks |   Jaccard |   complete_outcome_count | treatment_contrast    |
|:-----------------------------------|-----------------:|--------:|----------:|---------------:|-----------------:|----------:|-------------------------:|:----------------------|
| B0 EPS >=25 hard/bonus/drop        |               84 |      13 |        30 |             10 |               20 |  0.667083 |                       38 | OK                    |
| B0 EPS unknown manual-review       |              101 |       0 |         0 |              0 |                0 |  1        |                       47 | NO_TREATMENT_CONTRAST |
| B0 close trigger soft              |              101 |       0 |         0 |              0 |                0 |  1        |                       47 | NO_TREATMENT_CONTRAST |
| B0 fresh continuous                |              101 |       3 |         3 |             14 |                3 |  0.9625   |                       46 | OK                    |
| B0 geometry failure only           |              101 |       0 |         0 |              0 |                0 |  1        |                       47 | NO_TREATMENT_CONTRAST |
| B0 geometry soft                   |              115 |      16 |         2 |              1 |               13 |  0.84127  |                       56 | OK                    |
| B0 no entry_valid                  |              101 |       0 |         0 |              0 |                0 |  1        |                       47 | NO_TREATMENT_CONTRAST |
| B0 no industry cover               |              106 |      12 |         7 |              2 |               11 |  0.871667 |                       45 | OK                    |
| B0 pullback dry bonus              |              101 |       0 |         0 |              0 |                0 |  1        |                       47 | NO_TREATMENT_CONTRAST |
| B0 pullback dry drop               |              101 |      14 |        14 |             14 |               13 |  0.825833 |                       46 | OK                    |
| B0 pullback dry hard               |               87 |      21 |        35 |             22 |               26 |  0.592083 |                       35 | OK                    |
| B0 status soft                     |              101 |       0 |         0 |              0 |                0 |  1        |                       47 | NO_TREATMENT_CONTRAST |
| B0 status supplemental UNCONFIRMED |              101 |       0 |         0 |              0 |                0 |  1        |                       47 | NO_TREATMENT_CONTRAST |
| B0 top1                            |               40 |       0 |        61 |              0 |               35 |  0.454167 |                       19 | OK                    |
| B0 volume route-specific           |              101 |       0 |         0 |              0 |                0 |  1        |                       47 | NO_TREATMENT_CONTRAST |
| B0 volume soft                     |              101 |       0 |         0 |              0 |                0 |  1        |                       47 | NO_TREATMENT_CONTRAST |
| B0_PIT_VERIFIED                    |              101 |       0 |         0 |              0 |                0 |  1        |                       47 | NO_TREATMENT_CONTRAST |
| B0_REPO_EXACT                      |              101 |       0 |         0 |              0 |                0 |  1        |                       47 | NO_TREATMENT_CONTRAST |
| R1_ATOMIC_IMPROVEMENTS             |              101 |       3 |         3 |             14 |                3 |  0.9625   |                       46 | OK                    |
| R2_BALANCED_SOFT                   |              123 |      27 |         5 |             14 |               19 |  0.757937 |                       60 | OK                    |

## Rule Set OOS
| variant                |   folds |   paired_weeks |   mean_oos_diff_pct |   median_oos_diff_pct |   worst_fold_diff_pct |   better_folds |
|:-----------------------|--------:|---------------:|--------------------:|----------------------:|----------------------:|---------------:|
| B0_PIT_VERIFIED        |       5 |             17 |            0        |                     0 |              0        |              0 |
| B0_REPO_EXACT          |       5 |             17 |            0        |                     0 |              0        |              0 |
| R1_ATOMIC_IMPROVEMENTS |       5 |             17 |            0.838503 |                     0 |             -0.235722 |              1 |
| R2_BALANCED_SOFT       |       5 |             17 |           -0.275528 |                     0 |            -10.7891   |              2 |
| R3_MINIMAL_TECHNICAL   |       5 |             17 |           -0.563137 |                     0 |             -4.5911   |              2 |

NO PRODUCTION SKILL CHANGE
