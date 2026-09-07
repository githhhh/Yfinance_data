# Blind Rule Discovery R4 最终审计结论（供 GPT-6 终审）

> 状态：R4 研究收口报告  
> 适用分支：`codex/clean-latest-quant-trade-replay-pools`  
> **实验实际执行 commit**：`5f188d0262e518af736f0fa5e1f6ca3d3bf9197c`  
> 本文新增只改变文档，不改变 R4 实验代码、参数或输出。  
> 研究属性：retrospective known-history characterization / robustness research；**不是新的 unseen holdout，也不是生产 Alpha 认证。**

---

## 1. 这份报告回答什么

R1-R3 主要回答了“历史上哪些规则/市场状态能筛出更高 winner-rate 子集”，但没有直接回答最初真正关心的问题：

> 当 BreakoutFollow 候选真正触发买点并形成可执行 entry 时，什么 PIT 个股/setup 特征更容易快速走对，什么特征更容易先触发 -8% 止损，以及这些差异是否能改善 W1-W4、MAE/MFE 和同市场下的横截面选择质量？

R4 因此把主要 outcome 固定为：

```text
Fast Winner 3w:
+20% before -8%, within 15 trading sessions

Stop First 3w:
-8% before +20%, within 15 trading sessions

Unresolved 3w:
neither threshold reached within 15 sessions

Ambiguous 3w:
intrabar ordering cannot be established causally
```

主要概率分母是：

```text
all non-ambiguous executable entries
```

`unresolved_3w` 保留在分母中，不为了抬高胜率而剔除。

R4 同时记录：

- W1/W2/W3/W4 return；
- W1/W2/W3/W4 SPY excess return；
- 3w/4w MAE/MFE；
- stop-first 后 12w 是否恢复到 +20%；
- same-snapshot winner vs stop-first 横截面对照；
- stock-only single / two-way interaction；
- expanding-window rolling re-search，并对训练样本实施 W3 label purge。

---

## 2. 研究链路背景：R1-R4 应如何理解

### R1

冻结规则：

```text
ibd_entry_volume_ratio >= 1.8
AND
eps_yoy_growth >= 15%
```

真正未见 holdout：

- selected resolved winner rate：约 30.13%；
- holdout universe winner rate：约 32.34%。

结果 FAIL。R1 后续被定性为 custom heuristic + DeepSeek blind discovery；已消费 holdout 永久视为 known history。

### R2

在已知历史上寻找 empirical ceiling，一度出现约 65% winner-rate 的低覆盖率历史 pocket，主要包含 `base_depth_pct + ibd_entry_close_position` 等结构。

但样本和 evaluable-quarter support 很弱，后续 R3 并未稳定复现，因此不能继续作为主结论。

### R3

R3 对全部 compatible two-condition quantile pairs 做更完整搜索，并增加真正 LOQ / rolling robustness。

历史最佳规则：

```text
M_8w_drawdown <= -4.72%
AND
M_dist_52w_high >= -5.69%
```

主要历史指标：

- selected N：1,822；
- resolved N：1,279；
- winner rate：43.78%；
- universe winner rate：28.34%；
- lift：+15.44pp；
- 12w excess p50：+2.05%；
- MAE p50：-6.68%；
- MFE p50：+16.19%。

但两个字段均为 `M_*` broad-market features，因此它主要是 **market regime information**，不是同一周内的 stock ranking alpha。

### R4

R4 不再让 `M_*` 进入 stock-condition search，而是直接研究 executable trigger entry 后的赢家、输家和风险收益路径。

---

## 3. R4 执行可信度

R4 正式执行基于 commit：

```text
5f188d0262e518af736f0fa5e1f6ca3d3bf9197c
```

执行事实：

- pytest：100 total / 100 passed / 0 failed；
- warnings：3 个 pandas concat deprecation warnings，非功能失败；
- 唯一批准入口：`backtest.blind_rule_discovery.trigger_path_characterization_r4_causal_runner`；
- 正式 runtime：约 11m50s；
- LLM / DeepSeek / RD-agent 调用：0；
- 初始与最终 source/config git 状态一致；
- `M_*` 在 stock interaction rule 中 0 命中；
- W3 rolling label purge 后 overlap fold 数：0。

因此本报告接受 R4 的执行链和因果 rolling 实现作为有效研究证据。

---

## 4. 样本与总体基准

### 4.1 样本规模

- Replay candidate rows：10,686；
- executable trigger entries：8,983；
- censored：1,703；
- censored 原因全部为 `no_entry_within_buy_zone_window`；
- entry quarters：2022Q4 到 2026Q2；
- stock/setup + causal execution features：22；
- R3 favorable regime rows：1,822。

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

### 4.2 全样本 vs R3 favorable market

| 指标 | All | R3 favorable | 差异 |
|---|---:|---:|---:|
| Selected N | 8,983 | 1,822 | - |
| Evaluable N | 8,962 | 1,819 | - |
| Ambiguous N | 21 | 3 | - |
| Fast Winner 3w | 497 / **5.55%** | 101 / **5.55%** | **≈0** |
| Stop First 3w | 2,447 / **27.30%** | 367 / **20.18%** | **-7.13pp** |
| Unresolved 3w | **67.15%** | **74.27%** | +7.12pp |
| Stop-first then 12w winner | 21.09% of stops | 34.33% of stops | +13.24pp |
| Persistent Stop / evaluable | **21.55%** | **13.25%** | **-8.30pp** |
| 3w MAE p50 | -4.60% | -3.80% | +0.80pp |
| 3w MFE p50 | +5.00% | +5.18% | +0.18pp |
| 4w MAE p50 | -5.39% | -4.15% | +1.24pp |
| 4w MFE p50 | +5.94% | +6.31% | +0.37pp |
| 3w MFE/abs(MAE) | 1.086 | 1.363 | +0.277 |

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

### 4.3 对 market regime 的准确解释

R3 favorable regime **没有提高 3 周内 +20% first-passage 概率**：5.55% vs 5.55%。

它最明确的效果是：

- 少触发 -8% stop；
- persistent stop 更少；
- MAE 更浅；
- stop 后仍能在 12w 恢复到 +20% 的比例更高；
- W1-W4 中位路径整体更平滑。

因此 R3 favorable regime 更接近：

> **risk / downside regime filter**

而不是：

> “更容易在 3 周内产生 +20% winner 的市场”。

---

## 5. R4 最可靠的历史风险 / 输家特征

### 5.1 深 pullback 是高方差、高 stop-risk，而不是单纯 winner feature

典型单特征：

```text
pullback_pct <= -14.4%
```

历史：

- Fast Winner：11.03%（高于 5.55%）；
- Stop First：47.09%（远高于 27.30%）；
- Path Edge：**-14.30pp**；
- stop-risk quarter direction：100%。

说明深 pullback 同时提高向上和向下极端路径，本质更像 **high-volatility / high-dispersion feature**，不能因为 Fast Winner 高就叫“赢家特征”。

类似现象也出现在：

- `base_depth_pct <= -46.1%`；
- `dist_to_52w_high_pct <= -5.45%`；
- 高 `ibd_entry_close_vs_trigger_pct`。

### 5.2 延伸/追高是更一致的 stop-risk 信息

单特征历史结果：

```text
current_vs_ibd_candidate_pct > 3.73%
```

- Stop First：35.41%；
- baseline：27.30%；
- stop lift：+8.11pp；
- higher-stop-risk quarter fraction：100%。

```text
pct_above_ceiling > 22.9%
```

- Stop First：38.28%；
- stop lift：+10.98pp；
- higher-stop-risk quarter fraction：100%。

较高 actual entry extension 也更差：

```text
entry_extension_pct > 3.46%
```

- Stop First：32.67%。

### 5.3 最强 stop-risk interaction

全样本历史最强风险组合：

```text
current_vs_ibd_candidate_pct >= 3.73%
AND
pullback_pct <= -14.4%
```

- N：217；
- Stop First：**60.83%**；
- baseline：27.30%；
- stop lift：**+33.53pp**；
- persistent stop：37.33% of evaluable；
- 12w recovery among stops：38.64%；
- Fast Winner：10.60%；
- W3 return p50：-3.7%；
- W3 excess p50：-4.91%；
- 3w MAE / MFE：-11.6% / +6.8%；
- higher-stop-risk quarter fraction：100%。

其它高风险组合大多仍由 `pullback_pct <= -14.4%` 与延伸、volume、off-peak、EPS 弱等字段叠加构成。

### 5.4 当前最强可保留结论之一

> **“深 pullback + 已明显延伸/追高”是当前数据中最一致、最强的 stop-first / loser-risk family。**

这类证据目前比任何 winner rule 更强，也更有可能成为未来 B0/Top3 设计中的 risk penalty / reject evidence，但是否进入生产逻辑仍需 GPT-6 终审及独立后续验证。

---

## 6. Same-snapshot 横截面：哪些个股特征在同一市场里有分离倾向

Same-snapshot 对照只比较同一 `snapshot_date` 中的 Fast Winner 与 Stop First，因此 broad-market state 被天然控制。

注意：这里比较的是两个极端 outcome group，**unresolved 不参与该 AUC/概率比较**；因此它不是“全体候选上的直接 winner probability”。

主要结果：

| Feature | All: P(Winner value > Stop value) | Favorable | 备注 |
|---|---:|---:|---|
| pullback_v_is_dry | 0.5400 | **0.5737** | favorable 有正向倾向 |
| ibd_entry_volume_ratio | 0.4869 | 0.4751 | 无稳定正向 |
| ibd_entry_close_vs_trigger_pct | 0.5081 | 0.4636 | 不稳定 |
| ibd_entry_close_position | 0.5301 | **0.6322** | 但 equal-weight snapshot AUC p50=0.500，方向并不稳定 |
| ibd_entry_breakout_range_ratio | 0.4508 | 0.3831 | winner 倾向较低值 |
| current_vs_ibd_candidate_pct | 0.5173 | 0.4975 | 接近无分离 |
| volume_ratio | 0.4986 | **0.4118** | favorable winner 倾向较低量比 |
| pct_above_ceiling | 0.5052 | 0.5782 | 有 favorable 正向倾向，但与 stop-risk 分箱存在张力 |
| touched_ema10_count | 0.4733 | 0.5093 | 弱 |
| mbox_count | 0.4741 | 0.4882 | 弱 |
| base_depth_pct | **0.4198** | 0.4756 | winner 倾向更深，但属于高波动特征 |
| base_duration_weeks | 0.4911 | 0.4473 | 弱/不稳定 |
| pullback_count | 0.4750 | 0.5635 | favorable 有正向倾向 |
| pullback_duration_weeks | 0.5099 | **0.5860** | favorable 较长 pullback 更偏 winner |
| pullback_pct | **0.4195** | **0.3980** | winner 倾向更深，但 stop-risk 同时大增 |
| pullback_pct_off_peak | 0.5028 | 0.4828 | 弱 |
| eps_yoy_growth | 0.5311 | **0.5715** | favorable 中较一致正向 |
| dist_to_52w_high_pct | **0.4007** | 0.4484 | winner 倾向离 52w high 更远，但同样可能反映波动 |
| entry_delay_sessions | 0.5198 | 0.5172 | 弱 |
| entry_extension_pct | 0.5219 | 0.5039 | 横截面弱；分箱 risk evidence 更强 |
| entry_is_gap_or_open | 0.4854 | 0.4763 | 弱 |

### 6.1 不应被夸大的字段

`ibd_entry_close_position` favorable pair-weighted probability 为 0.6322，看起来很高，但同时：

- matched snapshots：仅 16；
- equal-weight snapshot AUC p50：0.500；
- median within-snapshot difference：-0.020；
- positive-difference fraction：43.8%。

因此不能把它升级为“稳定赢家特征”。它更像少数大 snapshot 驱动的描述性现象。

### 6.2 相对值得继续保留为 hypothesis 的方向

在 favorable regime 内，多个视角相对一致的历史倾向包括：

- EPS YoY growth 较高；
- `pullback_v_is_dry=True`；
- pullback duration 较长；
- `volume_ratio` 较低/不过热。

但没有 p-value、bootstrap CI 或 multiple-testing correction，因此本文不使用“统计显著”一词，只称 **historical descriptive separation**。

---

## 7. Favorable market 下最有意思的历史正向 family

R4 full-history favorable scope 中，最有交易意义的结构不是“越暴量越好”，而是：

> **整理时间较充分 + 量能不过热 + entry 不追高。**

### 7.1 `pullback_duration_weeks >= 8 & volume_ratio <= 0.88`

- N：70；
- Fast Winner：**15.71%** vs favorable baseline 5.55%；
- Stop First：**12.86%** vs favorable baseline 20.18%；
- Path Edge：**+17.48pp**；
- W3 excess p50：**+3.14%**；
- 3w MAE / MFE：-2.2% / +6.3%；
- positive-quarter fraction：75%；
- worst quarter edge：-4.4%；
- matched snapshot edge p50：+15.0%。

### 7.2 `entry_extension_pct <= 0.014 & pullback_duration_weeks >= 8`

- N：71；
- Fast Winner：8.45%；
- Stop First：**9.86%**；
- Path Edge：**+13.21pp**；
- W3 excess p50：+2.77%；
- 3w MAE / MFE：-2.7% / +6.1%；
- positive-quarter fraction：75%。

### 7.3 `base_depth <= -32.2 & pullback_pct >= -11.8`

- N：148；
- Fast Winner：8.78%；
- Stop First：12.84%；
- Path Edge：+10.57pp；
- W3 excess p50：+1.60%；
- quarter edge positive：100%；
- worst quarter edge：+3.1%。

这些是 **retrospective historical pockets / hypothesis families**，不是已验证 Alpha。

---

## 8. 为什么不能把 Winner Top20 当成“赢家规则”

Winner-oriented view 是按 Fast Winner enrichment 排序，因此会天然偏爱 high-volatility setup。

例如：

```text
base_depth <= -46.1
AND
base_mbox <= 2
```

- Fast Winner：24.21%；
- Stop First：50.18%；
- Path Edge：-4.21pp；
- W3 MAE：-8.9%；
- matched snapshot edge p50：-2.0%。

这说明：

> “更容易三周 +20%”与“更好的风险收益路径”不是同一件事。

因此最终判断必须同时看：

- Fast Winner；
- Stop First；
- unresolved；
- persistent stop；
- W1-W4；
- MAE/MFE；
- quarter consistency；
- same-snapshot evidence；
- chronological rolling。

---

## 9. Search 空间与 multiple-testing 边界

R4 每个 scope：

- generated stock conditions：148；
- supported singles：148；
- distinct-feature pairs tested：10,396；
- supported pairs：10,009（all）/ 9,077（favorable）；
- candidate rules：10,157（all）/ 9,225（favorable）。

搜索没有使用 `M_*`，也没有 pair pruning。

但是 R4 没有进行 formal multiple-testing correction，因此：

> full-history Top rules 只能用于 hypothesis generation；真正的 anti-overfit 证据来自 causal rolling，而 rolling 最终为负。

---

## 10. Causal Rolling：本轮最终裁决

R4 rolling 使用 expanding-window re-search，并在每个 fold 前 purge 所有 `exit_date_w3 >= test quarter start` 的训练样本。

累计：

- purged rows：1,339；
- post-purge overlap folds：0；
- 每个 fold `train_max_exit_date_w3 < test_quarter_start`；
- test quarter 不在 train 中。

### 10.1 All scope rolling

| Test quarter | Frozen rule（缩写） | Selected/Evaluable | Fast lift | Stop reduction | Path Edge | W3 excess p50 |
|---|---|---:|---:|---:|---:|---:|
| 2024Q2 | close_vs_trigger high + touched_ema10<=0 | 20/19 | +5.74pp | -6.50pp | **-0.76pp** | -0.11% |
| 2024Q3 | range_ratio high + pullback_duration>=4 | 65/65 | -0.54pp | -4.87pp | **-5.41pp** | -0.01% |
| 2024Q4 | entry_volume high + volume_ratio low | 24/24 | +2.75pp | -2.17pp | **+0.58pp** | -2.00% |
| 2025Q1 | entry_volume high + pullback_off_peak | 15/15 | -5.44pp | -19.66pp | **-25.09pp** | -1.04% |
| 2025Q2 | entry_ext low + pullback_duration>=8 | 7/6 | +9.60pp | -16.06pp | **-6.46pp** | +2.77% |
| 2025Q3 | entry_ext low + pullback_duration>=8 | 8/8 | +5.31pp | +9.79pp | **+15.10pp** | -1.06% |
| 2025Q4 | entry_ext low + pullback_duration>=8 | 7/7 | +7.11pp | -14.62pp | **-7.51pp** | +5.16% |
| 2026Q1 | entry_ext low + pullback_duration>=8 | 21/21 | -7.89pp | -10.99pp | **-18.88pp** | -0.89% |
| 2026Q2 | zero selection / boundary cohort | 0/0 | - | - | - | - |

Summary：

- positive path-edge all-fold fraction：2/9 = **22.22%**；
- positive path-edge evaluable fraction：2/8 = **25.00%**；
- median evaluable path edge：**-5.93pp**；
- selected N / evaluable N：167 / 165；
- pooled selected Fast Winner：**5.45%**；
- baseline on selected/traded folds：**6.04%**；
- pooled selected Stop First：**36.36%**；
- baseline on selected/traded folds：**28.61%**；
- selected-fold pooled path edge：**-8.34pp**。

### 10.2 R3 favorable scope rolling

| Test quarter | Frozen rule（缩写） | Selected/Evaluable | Fast lift | Stop reduction | Path Edge | W3 excess p50 |
|---|---|---:|---:|---:|---:|---:|
| 2024Q2 | pct_above_ceiling low + volume_ratio low | 50/50 | -4.06pp | -3.61pp | **-7.67pp** | -2.01% |
| 2024Q3 | current_vs_ibd low + pullback_duration>=9 | 14/14 | -3.54pp | +1.84pp | **-1.70pp** | +2.06% |
| 2024Q4 | no favorable rows | 0/0 | - | - | - | - |
| 2025Q1 | no favorable rows | 0/0 | - | - | - | - |
| 2025Q2 | current_vs_ibd low + intraday trigger | 15/15 | -9.02pp | +2.46pp | **-6.57pp** | -1.97% |
| 2025Q3 | no favorable rows | 0/0 | - | - | - | - |
| 2025Q4 | base_duration + pullback_off_peak | 47/47 | -2.76pp | -3.60pp | **-6.36pp** | +1.50% |
| 2026Q1 | rule selected 0 | 0/0 | - | - | - | - |
| 2026Q2 | no favorable rows | 0/0 | - | - | - | - |

Summary：

- zero-selection folds：5/9 = 55.56%；
- positive path-edge all-fold fraction：**0/9**；
- positive path-edge evaluable fraction：**0/4**；
- median evaluable path edge：**-6.47pp**；
- selected N / evaluable N：126 / 126；
- pooled selected Fast Winner：**2.38%**；
- selected/traded-fold baseline：**5.63%**；
- pooled selected Stop First：**23.02%**；
- selected/traded-fold baseline：**21.55%**；
- selected-fold pooled path edge：**-4.71pp**。

### 10.3 一个命名纠正

代码字段：

```text
pooled_matched_path_edge_selected_folds
```

实际计算的是：

```text
pooled selected rates
vs
pooled baseline rates on the same selected/traded folds
```

它不是严格的 snapshot-by-snapshot matched metric。

因此本文统一称：

> **selected-fold pooled path edge**

真正的 same-snapshot rolling evidence 是每 fold 的 `test_matched_*` 指标以及 summary 中的 `matched_positive_edge_fold_fraction`。

该命名问题不改变负结论。

---

## 11. R4 最终结论：按证据强度分层

### A. 已确认的历史 market-regime 信息

R3 favorable regime 的主要作用是：

- 明显降低 stop-first；
- 降低 persistent stop；
- 改善 MAE；
- 提高 stop 后长期恢复比例。

它没有提高 3 周 +20% Fast Winner rate。

### B. 当前最强、最一致的历史 loser/risk information

```text
深 pullback
+
明显延伸 / 追高
```

是最强 stop-first family。

尤其：

```text
current_vs_ibd >= 3.73%
AND
pullback_pct <= -14.4%
```

历史 Stop First 达 60.83%。

这一方向比正向 winner rule 更值得优先进入 GPT-6 的最终策略审计。

### C. 有希望但尚未验证的正向 historical family

在 favorable regime 中：

```text
pullback duration 较充分
+
volume ratio 不过热
+
entry extension 较小
```

出现过明显更好的历史 path edge、MAE/MFE 和 W3 excess。

但 chronological causal rolling 没有维持，因此只能列为 hypothesis，不是规则。

### D. 弱到中等的 descriptive winner tendencies

同 snapshot / favorable scope 中值得保留观察的方向：

- EPS YoY 较高；
- dry pullback；
- pullback duration 较长；
- volume_ratio 较低。

`ibd_entry_close_position` 虽有 pair-weighted 高值，但 equal-weight snapshot evidence 不支持稳定方向，因此不应升级。

### E. 当前明确没有找到的东西

> **没有找到能够通过 causal chronological rolling 的稳定 1-2 条件 cross-sectional stock-selection Alpha。**

当前 full-history rule search 的最好结果在下一季度 re-search/freeze 后总体变差，而不是变好。

---

## 12. 为什么“赢家特征”比“输家特征”更难确认

R4 baseline 中：

- Fast Winner 仅 5.55%；
- Stop First 27.30%；
- Unresolved 67.15%。

`+20% within 15 sessions` 是一个很稀疏的极端事件。

因此：

1. winner characterization 天然统计功效低于 stop-risk characterization；
2. 高波动 setup 更容易同时碰到 +20% 和 -8%，容易产生“winner enrichment + 更大 stop risk”的双刃剑；
3. 只优化 Fast Winner rate 会偏向高波动 pocket；
4. 必须依赖 Path Edge、MAE/MFE、W1-W4 和 rolling 来区分真正质量与单纯波动。

这一点是 GPT-6 终审时应重点考虑的方法学问题。

---

## 13. 对 B0 / Top3 的当前意义

当前证据不支持直接写出一个新的 Top3 正向评分公式。

相对更有证据基础的方向是：

> **风险侧的降权 / reject 可能比正向加分更可辨识。**

例如：

- 深 pullback；
- 已明显 extended；
- 大幅高于 buy point / ceiling；
- 若与高波动结构同时出现，则 stop-first 风险显著放大。

但本文仍不建议直接修改生产 B0；应由 GPT-6 结合当前 B0 排序逻辑、实际推荐样本和研究证据做终审。

此外，`pullback_v_is_dry=False` 仍不应因为传统 heuristic 就机械视为大幅负面：R4 显示 dry=True 有一定 favorable separation，但远没有强到足以支持“一票否决”。

---

## 14. 研究中已经被推翻或降级的旧说法

以下内容不应再作为主结论引用：

1. “R1 的 entry volume + EPS 是有效 Alpha”——holdout FAIL。
2. “R2 的 shallow base + very high close position 是稳定核心 rule”——更严格 R3/R4 未支持稳定迁移。
3. “R3 65%/43% 等历史漂亮数字说明 stock ranking alpha”——主要是 market regime 或 retrospective pocket。
4. “favorable market 会提高 3w +20% winner rate”——R4 明确为 5.55% vs 5.55%，主要改善 downside。
5. “ibd_entry_close_position 在 favorable market 是已确认赢家特征”——pair-weighted 结果与 equal-weight snapshot evidence 冲突，不能升级。
6. “full-history winner Top20 就是好规则”——大量规则同时显著提高 stop-first，Path Edge 为负。

---

## 15. 给 GPT-6 的终审任务

请不要从头重复 R1-R4 的搜索，而应基于本报告与仓库代码做最终方法学和策略审计。

重点问题：

1. R4 的 `+20% / -8% within 15 sessions` 是否过度稀疏、过度偏向高波动 setup？
2. 是否应把 stop-risk prediction 作为比 winner prediction 更主要的可利用目标？
3. “深 pullback + extension”风险 family 是否在因果和特征语义上足够合理，可进入 B0 penalty 候选？
4. favorable market 内 `longer pullback + controlled volume + low entry extension` 为什么 full-history 很好但 rolling 崩溃？是阈值漂移、regime shift、样本稀疏，还是 feature 本身无稳定性？
5. same-snapshot comparison 是否需要进一步用 snapshot fixed effects / conditional logistic / rank-based model 来定量控制，而不是继续 threshold search？
6. 是否应该从“寻找固定规则”转向更连续、更保守的 ranking / risk penalty 估计？
7. 当前 22 个 PIT stock features 是否本身信息不足，下一步若继续研究，应该新增什么真正不同的信息，而不是继续调同一批阈值？
8. 在不重新消耗已知历史为“伪 holdout”的前提下，下一次真正 prospective / untouched 验证应该如何设计？

---

## 16. 最终一句话结论

> **经过 R1-R4，已经确认 BF 的 broad-market environment 对 downside risk 有明显影响，也找到若干强而一致的 stop-first 风险特征；但当前 22 个 PIT stock/execution features 尚未产生一个能通过 causal chronological rolling 的稳定 stock-selection rule。当前最可信的研究价值在“识别高失败风险并降权”，而不是声称已经发现一个可以替代 B0/Top3 的正向 Alpha。**
