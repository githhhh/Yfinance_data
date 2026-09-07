# Blind Rule Discovery 阶段结论与下一阶段研究方向

> 状态：阶段性正式研究报告  
> 适用分支：`codex/clean-latest-quant-trade-replay-pools`  
> 研究范围：R1 / R2 / R3 retrospective experiments 及其方法学审计  
> 重要边界：本文总结的是已知历史数据上的研究结论，不把任何结果表述为未来 OOS Alpha。

---

## 1. 研究最初目标

本研究的原始目标不是简单寻找一个历史胜率最高的规则，而是回答：

1. 当前 BreakoutFollow 候选在真实触发买点并成交时，哪些 point-in-time 个股特征与后续赢家/输家路径显著相关；
2. 是否存在明显优于现有 B0 / Top3 选择逻辑的简单稳定组合；
3. `pullback_v_is_dry`、突破质量、base、volume、EPS 等现有字段中，哪些是真正稳定信息，哪些只是当前规则中的人为假设；
4. 市场环境与个股选择效应能否分离，避免把 market timing uplift 错当成 stock-selection alpha。

研究期间逐步发现，早期实验的目标函数更偏向“搜索整体高 winner-rate 子集”，因此市场状态变量 `M_*` 容易主导结果，而真正的“赢家特征 / 输家特征”问题尚未被直接回答。

---

## 2. 数据与结果语义

### 2.1 候选与 PIT 边界

当前 retrospective 研究基于完整历史 Replay 候选，所有研究特征均要求在 signal / trigger 时点可知。

当前显式股票特征包括：

- `pullback_v_is_dry`
- `ibd_entry_volume_ratio`
- `ibd_entry_close_vs_trigger_pct`
- `ibd_entry_close_position`
- `ibd_entry_breakout_range_ratio`
- `current_vs_ibd_candidate_pct`
- `volume_ratio`
- `pct_above_ceiling`
- `touched_ema10_count`
- `mbox_count`
- `base_depth_pct`
- `base_mbox_count`
- `base_duration_weeks`
- `pullback_count`
- `pullback_duration_weeks`
- `pullback_pct`
- `pullback_pct_off_peak`
- `eps_yoy_growth`
- `dist_to_52w_high_pct`

并补充 broad-market PIT 特征：

- `M_4w_return`, `M_4w_drawdown`
- `M_8w_return`, `M_8w_drawdown`
- `M_12w_return`, `M_12w_drawdown`
- `M_dist_52w_high`

### 2.2 当前历史 outcome

当前 outcome 以真实 trigger 后未来最多 5 个交易日内是否可成交作为 entry gate，成交后使用 first-passage 逻辑判断：

- `clean_winner`：+20% 先于 -8%；
- `stopped_out_loser`：-8% 先触发，且之后未先成为 clean winner；
- `stop_out_then_winner`：先 -8%，之后再到 +20%，在主分类中仍属于 loser；
- `ambiguous_path`：同一 bar 内目标和止损顺序无法确定；
- `unresolved`：观察窗口内未触发上述主要路径。

同时记录 4w / 8w / 12w return、excess return、MAE 和 MFE。

---

## 3. R1：Canonical Blind 尝试及其结论

### 3.1 发生了什么

R1 最终生成的规则为：

```text
ibd_entry_volume_ratio >= 1.8
AND
eps_yoy_growth >= 15%
```

其 discovery 阶段看起来有一定改善，但 one-shot holdout 中：

- selected resolved winner rate：约 30.13%
- holdout universe winner rate：约 32.34%

因此该规则没有延续 winner separation。

### 3.2 方法学问题

R1 后续审计发现正式 research agent 过程被临时 ad-hoc runner 替代，存在：

- 正式 run 前已经进行了 discovery 分析；
- 自定义 heuristic + DeepSeek selector；
- 第一个满足较低 convergence 门槛的规则即可 early stop；
- fallback 逻辑会在无稳定规则时强制产生一个规则。

因此 R1 被重新定性为：

```text
R1 — custom heuristic + DeepSeek blind discovery
Holdout consumed
Result: FAIL
```

R1 已消费的时期不能再次恢复 unseen holdout 身份。

### 3.3 R1 可保留的事实

R1 的失败本身仍有效：

> `entry_volume >= 1.8 + EPS YoY >= 15%` 这一冻结规则在当时真正未见的 holdout 上没有表现出 winner-rate uplift。

但 R1 不足以证明“当前 feature set 没有 Alpha”。

---

## 4. R2：Feature-balanced retrospective ceiling

### 4.1 研究目的

R2 明确放弃“unseen holdout”主张，把所有已知历史数据作为 retrospective optimization dataset，尝试回答：

> 在当前 feature set 内，如果允许耦合，历史上简单规则最高能做到什么程度？

R2 引入：

- dense quantile singles；
- feature-balanced pair interaction；
- triple beam；
- compact DNF；
- Pareto frontier；
- rolling walk-forward。

### 4.2 R2 一度出现的强历史 pocket

R2 前排规则大量包含：

```text
浅 base
+
极高 breakout close position
+
market / volume / pullback 条件
```

典型最佳规则曾达到约：

- winner rate：65%+
- universe winner rate：约 28%
- 12w excess p50：约 +7.9%

但该规则：

- coverage 极低；
- resolved 样本仅几十个；
- active quarters 与真正 evaluable quarters 严重不一致；
- stability score 对少量可评估季度过于乐观。

### 4.3 R2 后续被推翻的部分

R2 的 `base_depth_pct + ibd_entry_close_position` 核心假设，在更严格 R3 中没有维持主导地位。

因此当前不应再把它作为主要稳定发现。

---

## 5. R3：Exhaustive exact pair + robustness

R3 修复了 R2 的几个关键方法学缺口：

1. pair 层真正遍历全部 compatible quantile pairs；
2. 增加 evaluable-quarter support gate；
3. fixed-rule Drop-One-Quarter 与真正 LOQ re-search 分离；
4. rolling 把 zero-selection fold 纳入稳定性失败统计；
5. rolling 计算 contemporaneous universe baseline。

### 5.1 搜索规模

R3 实际完成：

- generated conditions：816
- exact compatible pairs：322,087
- supported pairs：293,055
- triple candidates：15,704
- DNF candidates：194
- candidate rules：294,212
- Pareto rules：540

因此在已提交的 quantile grid 和 support contract 下，两条件 interaction 空间已经得到较充分搜索。

### 5.2 R3 历史最佳规则

Full-history best rule：

```text
M_8w_drawdown <= -4.72%
AND
M_dist_52w_high >= -5.69%
```

历史指标约为：

- selected N：1,822
- coverage：20.28%
- resolved N：1,279
- winner N：560
- winner rate：43.78%
- universe winner rate：28.34%
- winner-rate lift：+15.44pp
- 12w excess p50：+2.05%
- MAE p50：-6.68%
- MFE p50：+16.19%
- active / evaluated quarters：7 / 7
- positive quarter fraction：6 / 7

这已经不是 R2 那种几十个 resolved 样本的小 pocket，说明当前 PIT feature set 中确实存在明显的历史结构。

### 5.3 但该结构主要是 market regime，不是 stock-selection rule

R3 最佳规则的两个字段均为 `M_*` broad-market features。

其语义更接近：

> 市场最近经历过明显回撤，但仍处于距离 52 周高位较近的位置时，BF 候选整体历史质量较高。

同一个 signal snapshot / week 的候选股票拥有相同或高度相近的 `M_*` 状态，因此该规则主要回答“什么时候做 BF 更有利”，而不能回答“同一周应该买哪只股票”。

### 5.4 True LOQ re-search

R3 True LOQ：

- held quarters：15
- zero-selection：8 / 15
- evaluable：7 / 15
- positive winner lift：3 / 15
- positive / evaluable：3 / 7
- pooled selected resolved WR：33.76%
- median held-quarter lift：约 -0.48pp
- median held-quarter excess p50：约 -2.09%

因此：

> 历史上可以反复找到漂亮 regime，但 threshold / rule 在 held-quarter 迁移时并不稳定。

### 5.5 Expanding-window rolling

R3 rolling：

- folds：9
- zero-selection：4 / 9
- evaluable folds：5
- pooled selected WR：32.57%
- pooled universe WR：28.95%
- pooled nominal lift：+3.62pp
- positive-lift all-fold fraction：1 / 9
- median evaluable-fold lift：约 -2.0pp
- median 12w excess p50 across nonempty folds：约 -0.26%

注意 pooled +3.62pp 不能直接解释为 stock-selection alpha，因为大量 uplift 来自“策略只在某些市场时期产生 selection”的 market timing effect。

### 5.6 Rolling feature frequency

R3 rolling 最稳定重复出现的是：

- `M_dist_52w_high`：8 / 9 folds
- `M_12w_drawdown`：5 / 9 folds
- `M_8w_return`：2 / 9 folds

股票自身 feature 几乎没有稳定重复出现。

因此当前最可靠的阶段性结论是：

> **当前研究已经发现明显 market-regime information，但尚未证明一个稳定的 cross-sectional stock-selection rule。**

---

## 6. 当前已经确认、被否定与尚未回答的结论

### 6.1 已确认

1. 当前 PIT feature set 并非“完全没有信息”。
2. broad-market regime 对 BF candidate 的整体后续质量影响明显。
3. `M_dist_52w_high` 是当前 retrospective rolling 中最稳定重复出现的市场变量。
4. 历史上可以搜索出强 interaction pocket，但“历史漂亮”与“可迁移稳定”是完全不同的问题。
5. R1 的 `entry volume + EPS` 规则没有得到 holdout 支持。

### 6.2 当前被否定 / 不应继续引用为主结论

1. R1 不能代表真正充分的 autonomous RD-agent search。
2. R2 的 `base_depth + close_position` 强规则不足以视为稳定核心 interaction。
3. Full-history winner-rate 最高的规则不能直接视为 production Alpha。
4. market timing uplift 不能作为 B0 / Top3 ranking uplift。

### 6.3 尚未回答

最重要的缺口仍然是：

> **当 BF 候选在真实 trigger 处成交时，赢家和输家本身到底有什么不同？**

当前 R1–R3 并没有系统回答：

- 哪些股票 PIT 特征明显增加“先止损”的概率；
- 哪些 PIT 特征明显增加快速 +20% 的概率；
- winner / loser 在 W1 / W2 / W3 / W4 路径上有什么稳定差异；
- 哪些特征改善 MAE / MFE 风险收益质量；
- 在相同市场 regime、甚至同一个 snapshot/week 内，哪些股票特征还能区分 winner 与 loser；
- favorable market + favorable stock features 是否存在稳定 interaction。

---

## 7. 下一阶段唯一研究方向：Trigger-Time Winner / Loser Characterization

下一阶段不再继续“盲目搜索最高历史 winner-rate 规则”。

研究问题改为：

> **在 BreakoutFollow 候选实际触发买点并成交的时刻，哪些 PIT 个股特征及其与市场状态的 interaction，能够稳定提高未来 3 周内 +20% 先于 -8% 的概率、降低 -8% 先触发概率，并改善 W1–W4 收益、MAE/MFE 和超额收益路径？**

---

## 8. 新 outcome 设计

### 8.1 T0

```text
T0 = 实际可执行 trigger entry date / price
```

所有 predictor 必须是 T0 之前或 T0 当时可知的 PIT 数据。

### 8.2 Primary outcome：3 周 first-passage

建议以未来 15 sessions 为主窗口：

```text
Fast Winner 3w:
+20% before -8%, and +20% reached within 15 sessions

Stop First 3w:
-8% before +20%, within 15 sessions

Unresolved 3w:
neither threshold reached within 15 sessions

Ambiguous:
same-bar ordering cannot be determined
```

核心研究指标：

```text
P(Fast Winner 3w)
P(Stop First 3w)
FastWinner / StopFirst ratio
```

`Stop First` 不能简单视为 `1 - Fast Winner`，因为 unresolved 必须独立保留。

### 8.3 辅助 first-passage thresholds

为了判断是否存在稳定的单调 follow-through，可同时报告：

```text
+10% before -8% within 3w
+15% before -8% within 3w
+20% before -8% within 3w
```

如果同一个 feature bucket 对 +10 / +15 / +20 均呈稳定单调改善，证据强于只命中单一 +20% threshold。

### 8.4 W1–W4 路径

必须直接报告：

```text
W1 return / excess
W2 return / excess
W3 return / excess
W4 return / excess
```

每项至少包含：

- p25
- p50
- p75

目标是区分：

- 快速 follow-through；
- 先深回撤后修复；
- 长期最终上涨但 breakout 初期质量差。

### 8.5 风险收益路径

至少报告：

```text
3w MAE / MFE
4w MAE / MFE
MFE / abs(MAE)
```

并保留原有 12w outcome 作为长期参考，而不是主要 winner-characterization target。

---

## 9. Winner / Loser 必须进一步拆分

不能只把所有失败都扔进一个 loser bucket。

建议至少分为：

### A. Clean / Fast Winner

突破后不先触发 -8%，并快速达到目标。

研究问题：

> 什么特征提高“突破后立即走对”的概率？

### B. Stopped-out Loser

先触发 -8%，后续观察期也没有形成有效恢复。

研究问题：

> 什么特征增加真正 setup failure / immediate failure 的概率？

### C. Stop-out-then-Winner

先触发 -8%，之后再达到 +20%。

这类不能简单视为“坏股票”，它可能意味着：

- setup 本身有效；
- entry timing 较差；
- 固定 -8% stop 与 setup volatility 不匹配。

应独立研究，而不是与真正失败 setup 混在一起。

### D. Unresolved / Ambiguous

保持独立，不用于强行填充 winner / loser。

---

## 10. 单特征 Characterization 应先于组合搜索

对每个 stock PIT feature，先生成稳定的分层画像，不先追求一个最终 rule。

每个 feature 至少按 quantile / natural buckets 报告：

```text
sample N
resolved N
Fast Winner 3w %
Stop First 3w %
FastWinner / StopFirst ratio
W1 p25/p50/p75
W2 p25/p50/p75
W3 p25/p50/p75
W4 p25/p50/p75
W1-W4 excess
3w / 4w MAE
3w / 4w MFE
MFE / abs(MAE)
```

并检查：

- 是否存在单调性；
- direction 是否跨季度一致；
- threshold 是否只在很窄区间有效；
- missingness 是否改变结论。

只有在单特征画像后，才进入 2-way / 3-way interaction。

---

## 11. 必须控制 market regime

下一阶段不能再让 `M_*` 直接吞掉整个目标函数。

需要至少做三层分析：

### Layer 1：全样本

回答总体 winner / loser feature difference。

### Layer 2：固定 favorable / neutral / weak market regime

在相同 market state 内比较 stock features。

例如可以把 R3 已发现的 regime 作为一个 retrospective analysis stratum，但不能直接视为未来固定生产规则。

目标是回答：

> 市场条件相似时，winner 股票本身还有什么稳定差异？

### Layer 3：Within-snapshot cross-sectional comparison

这是与 B0 / Top3 最直接相关的一层。

```text
同一 snapshot / week
同一个 M_* market state
winner candidates
vs
loser candidates
```

这种设计天然控制 market timing。

只有在这一层仍存在稳定差异，才能开始声称是 stock-selection information，而不是 market-regime information。

---

## 12. Favorable market + stock feature interaction

最终可以研究：

```text
market regime
+
stock feature A
+
stock feature B
```

但评价目标必须同时包含：

- Fast Winner 3w probability
- Stop First 3w probability
- W1–W4 return / excess path
- MAE / MFE
- coverage / support
- cross-quarter consistency
- within-snapshot relative performance

不能只追求 winner-rate 最大值。

一个有价值的组合应表现为：

> 在合理 coverage 下，同时提高快速 follow-through、降低 stop-first、改善中位收益和 downside path，并且不是单一季度或单一 market regime 的偶然产物。

---

## 13. 下一阶段成功 / 失败标准

### 成功

至少发现一类 stock feature 或 compact interaction 满足大部分：

1. Fast Winner 3w 概率明显提高；
2. Stop First 3w 明显下降；
3. W2/W3/W4 return 或 excess 改善；
4. MAE 改善且 MFE 不恶化；
5. 同 snapshot 横截面对照仍保持方向；
6. 多季度重复，不依赖单一 threshold；
7. support 足够，不是几十个样本的小 pocket。

### 失败

如果充分做完：

- single-feature characterization；
- market-controlled analysis；
- within-snapshot comparison；
- compact interaction；
- rolling / quarter robustness；

仍没有稳定方向，则应明确记录：

> 当前 PIT stock features 对 trigger-entry 后短期 winner/loser path 的可利用 cross-sectional information 很弱；下一步应扩展 feature space，而不是继续在同一批字段中无限调 threshold。

---

## 14. 对后续 GPT / RD-agent 的约束

下一位研究模型不应重复 R1–R3 的路径。

必须首先区分：

```text
market timing effect
vs
stock-selection effect
```

不得：

- 把 `M_*` uplift 直接解释为 Top3 ranking alpha；
- 只报告 full-history best rule；
- 只优化 winner-rate；
- 把 `stop_out_then_winner` 与真正 setup failure 无差别混合；
- 忽略 W1–W4 路径；
- 忽略 zero-selection / low-support periods；
- 在没有 within-snapshot 对照时声称 stock-selection improvement。

应优先回答：

> **触发价成交时，什么特征让这只股票更可能快速走对、更少先触发止损，并获得更好的 W1–W4 风险收益路径？**

---

## 15. 当前阶段最终结论

经过 R1–R3，可以确定：

1. 现有 feature set 中存在明显历史信息，但目前最稳定的部分主要来自 broad-market regime；
2. 尚未发现一个足够稳定、能够直接替代 B0 / Top3 的 cross-sectional stock-selection rule；
3. R2 的浅 base + 强 close historical pocket 不足以作为稳定主结论；
4. 当前研究最大的未完成项不是“再多搜几个 rule”，而是系统回答 trigger-entry 时的 winner / loser path characterization；
5. 下一阶段应把 3 周 first-passage、W1–W4、MAE/MFE、market-controlled 和 within-snapshot analysis 作为核心研究契约。

因此，本阶段不继续扩大 retrospective threshold search。下一阶段应围绕 **Trigger-Time Winner / Loser Characterization** 独立开展。
