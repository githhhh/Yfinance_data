# R6 Risk Feature Stability

Known-history retrospective research. NOT AN UNTOUCHED HOLDOUT.
RD-Agent backend proposes bounded expressions; the evaluator and ranking are local and fixed.
This is a custom research loop, not the canonical fin_factor / CoSTEER experiment.
Population: reconstructed signal candidates with usable executable entries; not all listings or all signals.
Features are snapshot-PIT only. Risk flags are hypothetical removals, not portfolio P&L.
Stop First is a W3 path event, not a long-term loser label. No 12-week labels enter selection.
All outer rules were frozen before evaluation; each fold's prompts use purged past only.
Inner feedback is adaptively reused and is not independent validation.
No p-value/significance or causal claim. Repeat tickers and overlapping paths remain dependent.

## Quarterly Evidence

| Quarter | Source | Status | N flagged/evaluable | Matched snapshots | Stop lift | stop_capture | winner_loss | Supported |
|---|---|---|---:|---:|---:|---:|---:|---|
| 2024Q2 | rdagent | FROZEN | 74/74 | 11 | 0.0399 | 0.1288 | 0.2903 | True |
| 2024Q2 | simple | FROZEN | 93/93 | 13 | -0.0439 | 0.1364 | 0.3226 | True |
| 2024Q3 | rdagent | FROZEN | 252/251 | 13 | -0.0243 | 0.2531 | 0.2647 | True |
| 2024Q3 | simple | FROZEN | 252/251 | 13 | -0.0243 | 0.2531 | 0.2647 | True |
| 2024Q4 | rdagent | FROZEN | 170/169 | 13 | 0.0437 | 0.2128 | 0.2400 | True |
| 2024Q4 | simple | FROZEN | 59/58 | 12 | 0.1434 | 0.1106 | 0.1400 | True |
| 2025Q1 | rdagent | FROZEN | 102/100 | 12 | 0.0932 | 0.1667 | 0.2703 | True |
| 2025Q1 | simple | FROZEN | 114/111 | 10 | 0.1758 | 0.2305 | 0.2973 | True |
| 2025Q2 | rdagent | FROZEN | 106/106 | 13 | 0.0343 | 0.3286 | 0.2000 | True |
| 2025Q2 | simple | FROZEN | 65/65 | 12 | 0.0053 | 0.1857 | 0.2000 | True |
| 2025Q3 | rdagent | FROZEN | 54/54 | 12 | 0.4153 | 0.1833 | 0.1071 | True |
| 2025Q3 | simple | FROZEN | 71/71 | 13 | 0.3149 | 0.1778 | 0.1786 | True |
| 2025Q4 | rdagent | FROZEN | 103/103 | 13 | 0.1468 | 0.1694 | 0.2063 | True |
| 2025Q4 | simple | FROZEN | 103/103 | 13 | 0.1468 | 0.1694 | 0.2063 | True |
| 2026Q1 | rdagent | FROZEN | 96/95 | 13 | 0.1345 | 0.1442 | 0.2237 | True |
| 2026Q1 | simple | FROZEN | 92/91 | 12 | -0.0581 | 0.0962 | 0.1053 | True |
| 2026Q2 | rdagent | FROZEN | 0/0 | 0 | N/A | N/A | N/A | False |
| 2026Q2 | simple | FROZEN | 0/0 | 0 | N/A | N/A | N/A | False |

## Evidence Boundary

- rdagent: favorable risk/cost direction 2/9 total quarters; 8 supported, 1 unsupported/abstaining. These counts are descriptive; no automatic promotion.
- simple: favorable risk/cost direction 0/9 total quarters; 8 supported, 1 unsupported/abstaining. These counts are descriptive; no automatic promotion.

## RD-Agent Incremental Evidence

Paired on commonly supported quarters only; different coverage remains visible.
Positive stop-lift delta favors risk detection; positive winner-loss delta is a cost.
| Quarter | Stop-lift delta | Winner-loss delta | Coverage delta |
|---|---:|---:|---:|
| 2024Q2 | 0.0837 | -0.0323 | -0.0328 |
| 2024Q3 | 0.0000 | 0.0000 | 0.0000 |
| 2024Q4 | -0.0997 | 0.1000 | 0.1264 |
| 2025Q1 | -0.0826 | -0.0270 | -0.0171 |
| 2025Q2 | 0.0290 | 0.0000 | 0.1028 |
| 2025Q3 | 0.1004 | -0.0714 | -0.0210 |
| 2025Q4 | 0.0000 | 0.0000 | 0.0000 |
| 2026Q1 | 0.1926 | 0.1184 | 0.0041 |

## Feature Recurrence

Occurrence in a frozen expression is not independent feature attribution or a causal effect.
| Feature | Source | Selected folds | Supported folds | Positive risk/cost folds |
|---|---|---:|---:|---:|
| base_depth_pct | rdagent | 1 | 1 | 0 |
| dist_to_52w_high_pct | rdagent | 1 | 1 | 0 |
| eps_yoy_growth | rdagent | 1 | 1 | 1 |
| eps_yoy_growth | simple | 1 | 1 | 0 |
| ibd_entry_breakout_range_ratio | simple | 1 | 1 | 0 |
| pct_above_ceiling | rdagent | 1 | 1 | 0 |
| pullback_count | rdagent | 1 | 1 | 0 |
| pullback_pct | rdagent | 4 | 3 | 1 |
| pullback_pct | simple | 4 | 4 | 0 |
| pullback_v_is_dry | rdagent | 3 | 3 | 1 |
| pullback_v_is_dry | simple | 2 | 1 | 0 |
| touched_ema10_count | rdagent | 1 | 1 | 1 |
| volume_ratio | rdagent | 4 | 3 | 0 |
| volume_ratio | simple | 1 | 1 | 0 |

Compare RD-Agent and simple rules on the same quarters using outer_quarters.csv; coverage differs and this is not a Matched-N portfolio benchmark.
A high stop_capture with high winner_loss may only describe high dispersion.
Freeze any prospective candidate before collecting genuinely future evidence.

KEEP PRODUCTION FROZEN
