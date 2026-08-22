# Weekly Signal Oracle Evaluation Run Log

## Purpose

按周评估 skill 推荐质量。每周先用所有 `signal == True` 标的建立独立 oracle，再评估推荐列表是否命中当周大赢家、避开当周大输家，并比较 EPS-blind 与 EPS-enriched 两种输入模式。

## Step Logic

1. 固定输入：`backtest/ibd_skill_replay_pools/*/breakout_follow_pool.csv` 的 43 个成功 replay pool，范围 2025-10-10 至 2026-08-07；不修改 pool。
2. 固定收益窗口：从每个 `snapshot_date` 的 `ibd_candidate_price` 到 `2026-08-14`，使用 `results_pkl/stock_data_220826_1d.pkl` 计算 latest return、max gain、max drawdown、-8% stop。
3. 每周 universe：该周所有 `signal == True` 行；ACTIONABLE 与非 ACTIONABLE 都进入 winner/loser oracle。
4. 每周 winner/loser：latest return Top3/Top5、max gain Top5、latest return Bottom3/Bottom5、以及是否触发 -8% stop。所有排名只在同一周内比较。
5. 推荐生成：对同一 pool 调用 `rank_reasoning_candidates(..., universe='review', version='v3')`，比较现有 ACTIONABLE variants 与 `signal_shadow_top3`（所有 signal，保留 entry_status，最多 3 只的审计层；非正式推荐）。
6. EPS-blind 模式：在内存中关闭 `eps_pit.lookup.get_signal_eps`，所有 CSV 空 EPS 保持 missing。
7. EPS-enriched 模式：允许 `eps_pit.lookup.get_signal_eps(snapshot, code)` 作为 point-in-time 补源，按用户要求先假设其正确。
8. Variant 比较：测试行业覆盖、EPS 已知、EPS>=25、排除 `pullback_not_dry`、排除 `geometry_caution_not_failure`、Fresh Demand/Constructive Pullback 限定，以及 RD candidate 排序假设。
9. 评分函数：`3*周Top5命中率 + 周max-gain Top5命中率 + 周中位平均收益/100 - 1.5*周Bottom5暴露率 - 周stop暴露率 - 0.8*pick Bottom5率 - 0.5*pick stop率`。
10. 规则沉淀：只采用跨周稳定的证据顺序和风险约束；禁止把具体 ticker、日期、收益率、中位数或命中率写成新门槛。

## Bug Fix During Evaluation

- 第一版临时脚本的 `fill_relaxed` 会错误放松 EPS 硬约束；已修正为 fallback 只能放松 lane/cleanliness，不能放松 `require_eps_known` 或 `require_eps_pass`。

## Current Best Rows

| eps_mode   | variant                                   |   weeks |   picks |   median_week_avg_latest_return_pct |   avg_week_avg_latest_return_pct |   median_week_worst_pick_return_pct |   week_latest_top5_hit_rate |   week_gain_top5_hit_rate |   pick_latest_top5_rate |   week_bottom5_hit_rate |   pick_bottom5_rate |   week_stop_rate |   pick_stop_rate |    score |
|:-----------|:------------------------------------------|--------:|--------:|------------------------------------:|---------------------------------:|------------------------------------:|----------------------------:|--------------------------:|------------------------:|------------------------:|--------------------:|-----------------:|-----------------:|---------:|
| with_eps   | signal_shadow_top3                        |      42 |     126 |                             21.2473 |                          39.0878 |                             12.212  |                    0.452381 |                  0.47619  |               0.214286  |                0.380952 |            0.230159 |         0.380952 |         0.206349 | 0.806123 |
| no_eps     | signal_shadow_top3                        |      42 |     126 |                             22.5569 |                          36.6955 |                             12.1091 |                    0.452381 |                  0.47619  |               0.206349  |                0.5      |            0.238095 |         0.428571 |         0.206349 | 0.586681 |
| with_eps   | research_proximity_eps_pass_floor_guard   |      31 |      67 |                             15.2132 |                          30.3154 |                             10.0668 |                    0.225806 |                  0.225806 |               0.104478  |                0.16129  |            0.104478 |         0.193548 |         0.104478 | 0.484053 |
| with_eps   | research_proximity_eps_known_floor_guard  |      36 |      82 |                             15.3904 |                          28.6929 |                             11.5765 |                    0.194444 |                  0.222222 |               0.0853659 |                0.194444 |            0.109756 |         0.194444 |         0.097561 | 0.336763 |
| no_eps     | research_proximity_structural_floor_guard |      36 |      83 |                             18.4142 |                          27.9738 |                             12.0306 |                    0.222222 |                  0.25     |               0.108434  |                0.25     |            0.120482 |         0.277778 |         0.13253  | 0.28538  |
| with_eps   | research_proximity_structural_floor_guard |      36 |      83 |                             15.3904 |                          27.7972 |                             11.5765 |                    0.194444 |                  0.222222 |               0.0843373 |                0.222222 |            0.120482 |         0.222222 |         0.108434 | 0.253302 |
| with_eps   | clean_eps_pass_no_dry_no_geom_caution     |      38 |      88 |                             15.2132 |                          25.4418 |                              9.4822 |                    0.236842 |                  0.236842 |               0.113636  |                0.263158 |            0.147727 |         0.289474 |         0.147727 | 0.223245 |
| with_eps   | fresh_demand_eps_pass_clean               |      38 |      88 |                             15.527  |                          25.6007 |                             10.0015 |                    0.210526 |                  0.210526 |               0.102273  |                0.263158 |            0.147727 |         0.263158 |         0.136364 | 0.153117 |
| with_eps   | fresh_or_constructive_eps_pass_clean      |      38 |      88 |                             15.527  |                          25.4744 |                             10.0015 |                    0.210526 |                  0.210526 |               0.102273  |                0.263158 |            0.147727 |         0.263158 |         0.136364 | 0.153117 |
| with_eps   | eps_pass_only                             |      38 |      88 |                             15.527  |                          25.4744 |                             10.0015 |                    0.210526 |                  0.210526 |               0.102273  |                0.263158 |            0.147727 |         0.263158 |         0.136364 | 0.153117 |
| no_eps     | research_fresh_demand_proximity_first     |      40 |     101 |                             18.0377 |                          25.3563 |                             11.898  |                    0.275    |                  0.3      |               0.128713  |                0.375    |            0.168317 |         0.375    |         0.178218 | 0.144114 |
| with_eps   | skill_industry_eps_known                  |      40 |     101 |                             16.0251 |                          26.4208 |                             11.709  |                    0.225    |                  0.25     |               0.0990099 |                0.3      |            0.148515 |         0.3      |         0.148515 | 0.142182 |

