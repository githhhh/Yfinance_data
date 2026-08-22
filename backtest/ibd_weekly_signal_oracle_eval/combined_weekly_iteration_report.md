# Weekly Signal Oracle Evaluation Run Log

## Purpose

按周评估 skill 推荐质量。每周先用所有 `signal == True` 标的建立独立 oracle，再评估推荐列表是否命中当周大赢家、避开当周大输家，并比较 EPS-blind 与 EPS-enriched 两种输入模式。

## Step Logic

1. 固定输入：`backtest/ibd_skill_replay_pools/*/breakout_follow_pool.csv` 的 43 个成功 replay pool，范围 2025-10-10 至 2026-08-07；不修改 pool。
2. 固定收益窗口：从每个 `snapshot_date` 的 `ibd_candidate_price` 到 `2026-08-14`，使用 `results_pkl/stock_data_150826_1d.pkl` 计算 latest return、max gain、max drawdown、-8% stop。
3. 每周 universe：该周所有 `signal == True` 行；ACTIONABLE 与非 ACTIONABLE 都进入 winner/loser oracle。
4. 每周 winner/loser：latest return Top3/Top5、max gain Top5、latest return Bottom3/Bottom5、以及是否触发 -8% stop。所有排名只在同一周内比较。
5. 推荐生成：对同一 pool 调用 `rank_reasoning_candidates(..., universe='review', version='v3')`，再用可解释 variant 选择最多 3 个 ACTIONABLE 推荐。
6. EPS-blind 模式：在内存中关闭 `eps_pit.lookup.get_signal_eps`，所有 CSV 空 EPS 保持 missing。
7. EPS-enriched 模式：允许 `eps_pit.lookup.get_signal_eps(snapshot, code)` 作为 point-in-time 补源，按用户要求先假设其正确。
8. Variant 比较：测试行业覆盖、EPS 已知、EPS>=25、排除 `pullback_not_dry`、排除 `geometry_caution_not_failure`、Fresh Demand/Constructive Pullback 限定等组合。
9. 评分函数：`3*周Top5命中率 + 周max-gain Top5命中率 + 周中位平均收益/100 - 1.5*周Bottom5暴露率 - 周stop暴露率 - 0.8*pick Bottom5率 - 0.5*pick stop率`。
10. 规则沉淀：只采用跨周稳定的证据顺序和风险约束；禁止把具体 ticker、日期、收益率、中位数或命中率写成新门槛。

## Bug Fix During Evaluation

- 第一版临时脚本的 `fill_relaxed` 会错误放松 EPS 硬约束；已修正为 fallback 只能放松 lane/cleanliness，不能放松 `require_eps_known` 或 `require_eps_pass`。

## Current Best Rows

| eps_mode   | variant                               |   weeks |   picks |   median_week_avg_latest_return_pct |   avg_week_avg_latest_return_pct |   median_week_worst_pick_return_pct |   week_latest_top5_hit_rate |   week_gain_top5_hit_rate |   pick_latest_top5_rate |   week_bottom5_hit_rate |   pick_bottom5_rate |   week_stop_rate |   pick_stop_rate |      score |
|:-----------|:--------------------------------------|--------:|--------:|------------------------------------:|---------------------------------:|------------------------------------:|----------------------------:|--------------------------:|------------------------:|------------------------:|--------------------:|-----------------:|-----------------:|-----------:|
| with_eps   | skill_industry_eps_known              |      40 |     101 |                             14.9517 |                          24.9612 |                            10.009   |                    0.25     |                  0.25     |                0.108911 |                0.325    |            0.168317 |         0.325    |         0.158416 |  0.123156  |
| with_eps   | skill_industry_eps_known_no_dry_fail  |      40 |     101 |                             14.9517 |                          24.8578 |                            10.009   |                    0.25     |                  0.25     |                0.108911 |                0.325    |            0.168317 |         0.325    |         0.158416 |  0.123156  |
| with_eps   | clean_eps_pass_no_dry_no_geom_caution |      38 |      88 |                             14.9517 |                          23.1414 |                             9.15057 |                    0.236842 |                  0.210526 |                0.113636 |                0.315789 |            0.170455 |         0.315789 |         0.159091 |  0.065187  |
| with_eps   | risk_clean_eps_known                  |      40 |     101 |                             14.6643 |                          24.6703 |                            10.009   |                    0.25     |                  0.25     |                0.108911 |                0.35     |            0.178218 |         0.35     |         0.168317 |  0.0449107 |
| with_eps   | eps_pass_only                         |      38 |      88 |                             14.9517 |                          22.5105 |                             9.71673 |                    0.210526 |                  0.184211 |                0.102273 |                0.289474 |            0.159091 |         0.289474 |         0.147727 |  0.0404861 |
| with_eps   | fresh_demand_eps_pass_clean           |      38 |      88 |                             14.9517 |                          22.6283 |                             9.71673 |                    0.210526 |                  0.184211 |                0.102273 |                0.289474 |            0.159091 |         0.289474 |         0.147727 |  0.0404861 |
| with_eps   | fresh_or_constructive_eps_pass_clean  |      38 |      88 |                             14.9517 |                          22.5105 |                             9.71673 |                    0.210526 |                  0.184211 |                0.102273 |                0.289474 |            0.159091 |         0.289474 |         0.147727 |  0.0404861 |
| with_eps   | v3_core_top3                          |      40 |     106 |                             15.1289 |                          24.3838 |                             9.71673 |                    0.25     |                  0.25     |                0.103774 |                0.35     |            0.169811 |         0.375    |         0.169811 |  0.0305342 |
| no_eps     | v3_core_top3                          |      40 |     106 |                             14.6902 |                          24.6023 |                            10.2021  |                    0.25     |                  0.275    |                0.113208 |                0.4      |            0.179245 |         0.4      |         0.179245 | -0.0611167 |

