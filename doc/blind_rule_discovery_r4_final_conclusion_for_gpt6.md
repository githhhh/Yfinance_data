# Blind Rule Discovery R1-R4 客观数据结论（GPT-6 数据输入）

> 适用分支：`codex/clean-latest-quant-trade-replay-pools`  
> R4 实际执行 commit：`5f188d0262e518af736f0fa5e1f6ca3d3bf9197c`  
> 本文只汇总已经产生的实验数据与方法学边界，不给出生产策略建议，不指定 GPT-6 的研究方向。  
> R1-R4 已消费的数据均按各自协议处理；R2-R4 属于 retrospective known-history research，不是新的 unseen holdout。

---

## 1. 实验链路与状态

### R1

冻结规则：

```text
ibd_entry_volume_ratio >= 1.8
AND
eps_yoy_growth >= 15%
```

真正未见 holdout：

- holdout N：2,666
- selected N：322
- selected resolved winner rate：30.13%
- holdout universe resolved winner rate：32.34%
- selected 12w excess p50：+2.08%
- selected MAE p50：-9.29%

结果：selected winner rate 低于 universe baseline。

### R2

R2 在已知历史上做 feature-balanced retrospective search。

历史最佳规则：

```text
M_dist_52w_high >= -2.83%
AND
base_depth_pct >= -12.8%
AND
ibd_entry_close_position >= 0.9352
```

主要数据：

- selected N：43
- resolved N：32
- winner rate：65.63%
- universe winner rate：28.34%
- lift：+37.28pp
- coverage：0.48%
- evaluated quarters：4
- 12w excess p50：+7.90%
- MAE p50：-3.01%
- MFE p50：+19.96%

R2 rolling：

- 9 folds
- pooled selected winner rate：28.77%
- positive winner-rate-lift folds：3/9
- zero-selection folds：2/9
- median fold winner-rate lift：-10.17pp
- minimum fold lift：-31.64pp
- median fold 12w excess p50：-0.07%

R2 的 full-history 高胜率规则样本较小，且 rolling 未复现同等效果。

### R3

R3 对 committed two-condition quantile grid 做 exact compatible-pair enumeration，并加入更严格 support、true LOQ re-search 和 chronological rolling。

历史最佳规则：

```text
M_8w_drawdown <= -4.72%
AND
M_dist_52w_high >= -5.69%
```

主要数据：

- selected N：1,822
- resolved N：1,279
- winner rate：43.78%
- universe winner rate：28.34%
- lift：+15.44pp
- 12w excess p50：+2.05%
- MAE p50：-6.68%
- MFE p50：+16.19%

这两个条件均为 `M_*` broad-market features，因此该结果区分的是市场环境，不是同 snapshot 内的个股横截面排序。

R3 true LOQ：

- held quarters：15
- zero-selection quarters：8/15
- evaluable held quarters：7
- positive lift：3/15（all held quarters）
- positive lift：3/7（evaluable held quarters）
- median held-quarter lift：-0.48pp
- median held-quarter 12w excess p50：-2.09%

R3 chronological rolling：

- folds：9
- positive winner-rate-lift folds：3/9
- zero-selection folds：2/9
- selected total：94
- resolved total：73
- winners：21
- pooled selected winner rate：28.77%
- median evaluable fold lift：-10.17pp

R3 的 full-history market-regime pocket 没有在 rolling 中稳定复现为固定 transferable rule。

---

## 2. R4 研究定义

R4 将研究单位固定为真实可执行 trigger entry。

主要 3 周 outcome：

```text
Fast Winner 3w
= +20% before -8% within 15 trading sessions

Stop First 3w
= -8% before +20% within 15 trading sessions

Unresolved 3w
= neither threshold reached within 15 sessions

Ambiguous 3w
= intrabar ordering cannot be established
```

概率分母：所有 non-ambiguous executable entries；`unresolved_3w` 保留在分母。

R4 同时记录：

- W1/W2/W3/W4 return
- W1/W2/W3/W4 SPY excess return
- 3w / 4w MAE/MFE
- stop-first 后 12w 是否恢复到 +20%
- same-snapshot Fast Winner vs Stop First 横截面对照
- stock-only single / two-way interaction
- expanding-window rolling re-search
- rolling W3 label purge

R4 stock-condition search 禁止 `M_*`。

---

## 3. R4 执行审计事实

- pytest：100 / 100 passed
- 正式入口：`backtest.blind_rule_discovery.trigger_path_characterization_r4_causal_runner`
- runtime：约 11m50s
- LLM / DeepSeek / RD-agent：0
- stock interaction 中 `M_*`：0 命中
- rolling W3 overlap purge 后 overlap folds：0
- source/config 初始与最终 git 状态一致

---

## 4. R4 样本

- Replay candidate rows：10,686
- executable trigger entries：8,983
- censored：1,703
- censor reason：全部 `no_entry_within_buy_zone_window`
- evaluable entries：8,962
- ambiguous：21
- entry quarters：2022Q4–2026Q2
- stock/setup + causal execution features：22
- R3 favorable rows：1,822
- R3 favorable evaluable：1,819

22 个 stock/execution features：

```text
pullback_v_is_dry
ibd_entry_volume_ratio
ibd_entry_close_vs_trigger_pct
ibd_entry_close_position
ibd_entry_breakout_range_ratio
current_vs_ibd_candidate_pct
volume_ratio
pct_above_ceiling
touched_ema10_count
mbox_count
base_depth_pct
base_mbox_count
base_duration_weeks
pullback_count
pullback_duration_weeks
pullback_pct
pullback_pct_off_peak
eps_yoy_growth
dist_to_52w_high_pct
entry_delay_sessions
entry_extension_pct
entry_is_gap_or_open
```

---

## 5. All vs R3 favorable market

| 指标 | All | R3 favorable |
|---|---:|---:|
| Evaluable N | 8,962 | 1,819 |
| Fast Winner 3w | 5.55% | 5.55% |
| Stop First 3w | 27.30% | 20.18% |
| Unresolved 3w | 67.15% | 74.27% |
| Stop-first then 12w winner / stops | 21.09% | 34.33% |
| Persistent Stop / evaluable | 21.55% | 13.25% |
| 3w MAE p50 | -4.60% | -3.80% |
| 3w MFE p50 | +5.00% | +5.18% |
| 4w MAE p50 | -5.39% | -4.15% |
| 4w MFE p50 | +5.94% | +6.31% |
| 3w MFE/abs(MAE) | 1.086 | 1.363 |

W1-W4 stock return p25/p50/p75：

| Horizon | All | R3 favorable |
|---|---|---|
| W1 | -2.33% / +0.15% / +2.71% | -1.72% / +0.68% / +3.16% |
| W2 | -3.48% / +0.36% / +4.19% | -2.47% / +1.06% / +4.92% |
| W3 | -4.28% / +0.39% / +5.32% | -2.34% / +1.57% / +6.07% |
| W4 | -5.02% / +0.60% / +6.31% | -2.23% / +2.47% / +7.57% |

W1-W4 SPY excess p25/p50/p75：

| Horizon | All | R3 favorable |
|---|---|---|
| W1 | -2.38% / -0.03% / +2.40% | -2.12% / +0.17% / +2.43% |
| W2 | -3.81% / -0.12% / +3.55% | -3.19% / +0.13% / +3.64% |
| W3 | -4.66% / -0.30% / +4.27% | -3.86% / +0.11% / +4.51% |
| W4 | -5.73% / -0.46% / +4.89% | -4.60% / +0.01% / +5.05% |

数据事实：R3 favorable 与 All 的 3w Fast Winner rate 相同；Stop First、Persistent Stop 和 MAE 不同。

---

## 6. R4 单特征历史分箱数据

### `pullback_pct <= -14.4%`

- Fast Winner：11.03%
- Stop First：47.09%
- All baseline Stop First：27.30%
- Path Edge：-14.30pp
- higher-stop-risk quarter fraction：100%

该条件下 Fast Winner 与 Stop First 都高于总体，表现为更高 outcome dispersion。

### `current_vs_ibd_candidate_pct > 3.73%`

- Stop First：35.41%
- baseline：27.30%
- Stop First lift：+8.11pp
- higher-stop-risk quarter fraction：100%

### `pct_above_ceiling > 22.9%`

- Stop First：38.28%
- baseline：27.30%
- Stop First lift：+10.98pp
- higher-stop-risk quarter fraction：100%

### `entry_extension_pct > 3.46%`

- Stop First：32.67%

### `pullback_v_is_dry=False`

- Fast Winner：4.88%
- Stop First：30.32%

这些均为 retrospective historical bin statistics；R4 没有对这些单独做 prospective untouched validation。

---

## 7. R4 full-history stop-risk interaction 数据

历史最强 stop-risk interaction：

```text
current_vs_ibd_candidate_pct >= 3.73%
AND
pullback_pct <= -14.4%
```

数据：

- selected N：217
- Fast Winner：10.60%
- Stop First：60.83%
- All baseline Stop First：27.30%
- Stop First lift：+33.53pp
- Persistent Stop / evaluable：37.33%
- 12w recovery among stops：38.64%
- W3 return p50：-3.70%
- W3 excess p50：-4.91%
- 3w MAE p50：-11.60%
- 3w MFE p50：+6.80%
- higher-stop-risk quarter fraction：100%

这是从已知历史 threshold search 中得到的结果；quarter-direction statistic 也处于同一选择过程内。

---

## 8. Same-snapshot 横截面数据

Same-snapshot 只比较同一个 `snapshot_date` 的 Fast Winner 与 Stop First。Unresolved 不进入该 pair comparison。

`P(Winner feature value > Stop feature value)`：

| Feature | All | R3 favorable |
|---|---:|---:|
| pullback_v_is_dry | 0.5400 | 0.5737 |
| ibd_entry_volume_ratio | 0.4869 | 0.4751 |
| ibd_entry_close_vs_trigger_pct | 0.5081 | 0.4636 |
| ibd_entry_close_position | 0.5301 | 0.6322 |
| ibd_entry_breakout_range_ratio | 0.4508 | 0.3831 |
| current_vs_ibd_candidate_pct | 0.5173 | 0.4975 |
| volume_ratio | 0.4986 | 0.4118 |
| pct_above_ceiling | 0.5052 | 0.5782 |
| touched_ema10_count | 0.4733 | 0.5093 |
| mbox_count | 0.4741 | 0.4882 |
| base_depth_pct | 0.4198 | 0.4756 |
| base_duration_weeks | 0.4911 | 0.4473 |
| pullback_count | 0.4750 | 0.5635 |
| pullback_duration_weeks | 0.5099 | 0.5860 |
| pullback_pct | 0.4195 | 0.3980 |
| pullback_pct_off_peak | 0.5028 | 0.4828 |
| eps_yoy_growth | 0.5311 | 0.5715 |
| dist_to_52w_high_pct | 0.4007 | 0.4484 |
| entry_delay_sessions | 0.5198 | 0.5172 |
| entry_extension_pct | 0.5219 | 0.5039 |
| entry_is_gap_or_open | 0.4854 | 0.4763 |

`ibd_entry_close_position` favorable 补充数据：

- pair-weighted probability：0.6322
- matched snapshots：16
- equal-weight snapshot AUC p50：0.500
- snapshot median difference p50：-0.020
- positive-difference snapshot fraction：43.8%

R4 未计算 formal bootstrap CI / p-value / multiple-testing correction，因此本节数据属于 descriptive separation。

---

## 9. R4 favorable full-history interaction 数据

### `pullback_duration_weeks >= 8 AND volume_ratio <= 0.88`

- N：70
- Fast Winner：15.71%
- favorable baseline Fast Winner：5.55%
- Stop First：12.86%
- favorable baseline Stop First：20.18%
- Path Edge：+17.48pp
- W3 excess p50：+3.14%
- 3w MAE / MFE：-2.20% / +6.30%
- positive-quarter fraction：75%
- worst quarter edge：-4.4pp
- matched snapshot edge p50：+15.0pp

### `entry_extension_pct <= 1.4% AND pullback_duration_weeks >= 8`

- N：71
- Fast Winner：8.45%
- Stop First：9.86%
- Path Edge：+13.21pp
- W3 excess p50：+2.77%
- 3w MAE / MFE：-2.70% / +6.10%
- positive-quarter fraction：75%

### `base_depth_pct <= -32.2% AND pullback_pct >= -11.8%`

- N：148
- Fast Winner：8.78%
- Stop First：12.84%
- Path Edge：+10.57pp
- W3 excess p50：+1.60%
- positive-quarter fraction：100%
- worst quarter edge：+3.1pp

这些规则均来自 known-history search，属于 full-history retrospective results。

---

## 10. R4 搜索空间

每个 scope：

- generated stock conditions：148
- supported singles：148
- distinct-feature pairs tested：10,396
- supported pairs：10,009（all）/ 9,077（favorable）
- candidate rules：10,157（all）/ 9,225（favorable）
- `M_*` conditions：0

没有 formal multiple-testing correction。

---

## 11. R4 causal rolling 数据

Rolling 使用 expanding-window re-search，并在每折训练前 purge：

```text
exit_date_w3 >= test_quarter_start
```

累计：

- purged rows：1,339
- post-purge overlap folds：0
- test quarter in train：0

### All scope

| Test quarter | Selected/Evaluable | Fast lift | Stop reduction | Path Edge | W3 excess p50 |
|---|---:|---:|---:|---:|---:|
| 2024Q2 | 20/19 | +5.74pp | -6.50pp | -0.76pp | -0.11% |
| 2024Q3 | 65/65 | -0.54pp | -4.87pp | -5.41pp | -0.01% |
| 2024Q4 | 24/24 | +2.75pp | -2.17pp | +0.58pp | -2.00% |
| 2025Q1 | 15/15 | -5.44pp | -19.66pp | -25.09pp | -1.04% |
| 2025Q2 | 7/6 | +9.60pp | -16.06pp | -6.46pp | +2.77% |
| 2025Q3 | 8/8 | +5.31pp | +9.79pp | +15.10pp | -1.06% |
| 2025Q4 | 7/7 | +7.11pp | -14.62pp | -7.51pp | +5.16% |
| 2026Q1 | 21/21 | -7.89pp | -10.99pp | -18.88pp | -0.89% |
| 2026Q2 | 0/0 | - | - | - | - |

Summary：

- positive Path Edge：2/9 folds
- evaluable positive Path Edge：2/8
- median evaluable Path Edge：-5.93pp
- selected N：167
- evaluable N：165
- pooled selected Fast Winner：5.45%
- baseline Fast Winner on selected/traded folds：6.04%
- pooled selected Stop First：36.36%
- baseline Stop First on selected/traded folds：28.61%
- selected-fold pooled Path Edge：-8.34pp

### R3 favorable scope

| Test quarter | Selected/Evaluable | Fast lift | Stop reduction | Path Edge | W3 excess p50 |
|---|---:|---:|---:|---:|---:|
| 2024Q2 | 50/50 | -4.06pp | -3.61pp | -7.67pp | -2.01% |
| 2024Q3 | 14/14 | -3.54pp | +1.84pp | -1.70pp | +2.06% |
| 2024Q4 | 0/0 | - | - | - | - |
| 2025Q1 | 0/0 | - | - | - | - |
| 2025Q2 | 15/15 | -9.02pp | +2.46pp | -6.57pp | -1.97% |
| 2025Q3 | 0/0 | - | - | - | - |
| 2025Q4 | 47/47 | -2.76pp | -3.60pp | -6.36pp | +1.50% |
| 2026Q1 | 0/0 | - | - | - | - |
| 2026Q2 | 0/0 | - | - | - | - |

Summary：

- zero-selection：5/9 folds
- positive Path Edge：0/9
- evaluable positive Path Edge：0/4
- median evaluable Path Edge：-6.47pp
- selected N：126
- evaluable N：126
- pooled selected Fast Winner：2.38%
- baseline Fast Winner on selected/traded folds：5.63%
- pooled selected Stop First：23.02%
- baseline Stop First on selected/traded folds：21.55%
- selected-fold pooled Path Edge：-4.71pp

代码字段 `pooled_matched_path_edge_selected_folds` 实际是 selected/traded-fold pooled baseline comparison，不是严格 snapshot-by-snapshot matched metric。

---

## 12. 仅由 R1-R4 数据直接支持的结论

1. R1 冻结规则在真正未见 holdout 上没有超过 universe winner-rate baseline。
2. R2 full-history 存在高胜率、低覆盖的小样本 pocket，但 R2 rolling 没有复现同等效果。
3. R3 full-history 最强结果主要由 broad-market `M_*` 条件构成；它不是同 snapshot stock-ranking separation。
4. R3 favorable rows 与 All rows 的 3w Fast Winner rate 相同（5.55%），但 Stop First、Persistent Stop 和 MAE 较低。
5. R4 full-history 中存在多个高 Stop First 的 feature bins 和 two-condition pockets，幅度可达到 60.83% Stop First vs 27.30% overall baseline。
6. R4 favorable full-history 中也存在正 Path Edge 的两条件 pockets。
7. R4 same-snapshot 中部分 feature 的 Fast-Winner/Stop-First 分布存在描述性差异，但没有 formal significance correction。
8. R4 quality-ranked stock-rule causal rolling：All scope positive Path Edge 2/9；favorable scope 0/9；pooled selected-vs-selected-fold baseline 均为负。
9. 因此，R1-R4 数据没有给出一个在上述 causal rolling 设计下稳定复现的 1-2 条件 stock-selection rule。
10. R4 没有专门做“以 stop-risk 为唯一目标选择规则”的 causal rolling，因此仅凭 R4 不能判断 stop-risk rule 是否比 winner/quality rule 更稳定。

---

## 13. R5 当前状态

截至本文更新时：

- R5 stop-risk validation 代码、协议、测试已冻结在分支；
- 当前冻结 HEAD：`8e33b63aecf34c0204e898f8b30ae6b220081670`；
- R5 尚未由 Gemini 正式执行；
- 因此本文没有任何 R5 数据结论；
- 在 R5 输出产生并完成审计前，不能从 R1-R4 数据提前推断“risk-side 比 winner-side 更稳定”或相反结论。

---

## 14. 数据边界

- R2-R4 为 known-history retrospective research。
- R4 full-history Top rules 经大量候选筛选产生，没有 formal multiple-testing correction。
- Same-snapshot comparison 排除了 unresolved，因此它不是全候选直接 winner probability。
- 2026Q2 属于边界 entry cohort，样本极少，不应与完整季度等权理解。
- 目前没有新的 untouched prospective sample 用于验证 R4 full-history hypotheses。

---

## 15. 当前客观数据摘要

```text
R1 unseen holdout:
selected WR 30.13% < universe WR 32.34%

R3 favorable vs All:
Fast Winner 5.55% vs 5.55%
Stop First 20.18% vs 27.30%
Persistent Stop 13.25% vs 21.55%

R4 strongest historical stop pocket:
Stop First 60.83% vs baseline 27.30%

R4 causal rolling — All:
positive Path Edge 2/9
median evaluable Path Edge -5.93pp
pooled selected Path Edge vs selected-fold baseline -8.34pp

R4 causal rolling — Favorable:
positive Path Edge 0/9
median evaluable Path Edge -6.47pp
pooled selected Path Edge vs selected-fold baseline -4.71pp

R5:
not executed yet; no conclusion
```
