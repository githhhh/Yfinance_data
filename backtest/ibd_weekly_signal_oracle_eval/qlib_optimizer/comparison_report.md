# Qlib-Compatible Replay Pool Optimization

- Replay pool files: `43`
- Qlib backend available: `True`
- Qlib version: `0.9.7`
- Data modes: no-EPS and with-EPS are optimized independently, then compared.
- Walk-forward: each evaluation week selects the best rule using only prior weeks.
- Leakage guard: portfolio scoring rejects any `label_` column; labels are only used after weekly picks are fixed.
- Qlib usage: `qlib.init()` is required, and `qlib.contrib.evaluate.risk_analysis` computes risk metrics for each optimized weekly return series. If Qlib is unavailable this script stops instead of falling back.

## Execution Steps

1. Load every weekly `breakout_follow_pool.csv` under `backtest/ibd_skill_replay_pools/`.
2. Build two datasets from the same replay pool history: `no_eps` disables EPS lookup, `with_eps` uses the supplemental point-in-time EPS lookup and assumes it is correct.
3. Convert each dataset into a Qlib-style `datetime` / `instrument` panel with signal-time feature columns and future outcome label columns.
4. For each evaluation week after the minimum training window, score every candidate rule on prior weeks only, then apply the best historical rule to that week.
5. Evaluate that week's selected portfolio by realized return, top5 return hit, top5 gain hit, bottom5 loss hit, and 8% stop hit.
6. Run Qlib `risk_analysis(freq="week")` on the optimized weekly return series.

## Data Coverage

Rows without realized return are excluded from walk-forward optimization and portfolio statistics; they remain in the full panel for audit.

| eps_mode   |   signal_rows |   valid_return_rows |   missing_return_rows |   signal_weeks |   valid_return_weeks |   pool_file_weeks |   training_window_weeks |
|:-----------|--------------:|--------------------:|----------------------:|---------------:|---------------------:|------------------:|------------------------:|
| no_eps     |          2738 |                2074 |                   664 |             42 |                   42 |                43 |                       8 |
| with_eps   |          2738 |                2074 |                   664 |             42 |                   42 |                43 |                       8 |

## Summary

| eps_mode   | strategy               |   weeks |   picks |   avg_return_pct |   median_week_return_pct |   median_worst_pick_return_pct |   top5_return_week_rate |   top5_gain_week_rate |   bottom5_week_rate |   stop_week_rate |   unique_rules |
|:-----------|:-----------------------|--------:|--------:|-----------------:|-------------------------:|-------------------------------:|------------------------:|----------------------:|--------------------:|-----------------:|---------------:|
| no_eps     | walk_forward_best_rule |      34 |      83 |          27.9811 |                  18.5483 |                        7.5564  |                0.441176 |              0.441176 |            0.382353 |         0.647059 |              3 |
| with_eps   | walk_forward_best_rule |      33 |      81 |          28.0499 |                  18.6372 |                        6.63984 |                0.393939 |              0.363636 |            0.424242 |         0.666667 |              4 |

## Rule Space

| rule               | requires_eps_known   | requires_eps_pass   | features                                                                                         |
|:-------------------|:---------------------|:--------------------|:-------------------------------------------------------------------------------------------------|
| eps_known_balanced | True                 | False               | $eps_known, $eps_pass_25, $fresh_0_5, $entry_volume_ratio, $weekly_volume_follow, $near_52w_high |
| eps_pass_quality   | False                | True                | $eps_pass_25, $fresh_0_5, $entry_volume_ratio, $weekly_volume_follow, $near_52w_high             |
| technical_balanced | False                | False               | $fresh_0_5, $entry_volume_ratio, $weekly_volume_follow, $near_52w_high, $close_position          |
| fresh_demand       | False                | False               | $fresh_0_2, $entry_volume_ratio, $weekly_volume_follow, $near_52w_high                           |
| risk_conservative  | False                | False               | $eps_known, $fresh_0_5, $entry_volume_ratio, $weekly_volume_follow, $near_52w_high               |

## Output Files

| eps_mode   | panel                                                                         | weekly_choices                                                                          | weekly_rule_scores                                                                          | weekly_picks                                                                          | risk                                                                                  |
|:-----------|:------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
| no_eps     | backtest/ibd_weekly_signal_oracle_eval/qlib_optimizer/no_eps_qlib_panel.csv   | backtest/ibd_weekly_signal_oracle_eval/qlib_optimizer/no_eps_walk_forward_choices.csv   | backtest/ibd_weekly_signal_oracle_eval/qlib_optimizer/no_eps_walk_forward_rule_scores.csv   | backtest/ibd_weekly_signal_oracle_eval/qlib_optimizer/no_eps_walk_forward_picks.csv   | backtest/ibd_weekly_signal_oracle_eval/qlib_optimizer/no_eps_qlib_risk_analysis.csv   |
| with_eps   | backtest/ibd_weekly_signal_oracle_eval/qlib_optimizer/with_eps_qlib_panel.csv | backtest/ibd_weekly_signal_oracle_eval/qlib_optimizer/with_eps_walk_forward_choices.csv | backtest/ibd_weekly_signal_oracle_eval/qlib_optimizer/with_eps_walk_forward_rule_scores.csv | backtest/ibd_weekly_signal_oracle_eval/qlib_optimizer/with_eps_walk_forward_picks.csv | backtest/ibd_weekly_signal_oracle_eval/qlib_optimizer/with_eps_qlib_risk_analysis.csv |
