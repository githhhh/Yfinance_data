# Next Week Review Selection 研究协议 v0.3

日期：2026-09-02  
分支：`research/next-week-review-selection`  
研究基座：`codex/clean-latest-quant-trade-replay-pools`

## 核心假设

> **B0 ACTIONABLE-only** vs **B0 + Near Buy Point 的 UNCONFIRMED / BELOW_TRIGGER + 至少一个独立正证据族**。

目标不是最大化 Winner Recall，而是在有限 Review List 扩张下，提高**可交易大赢家捕获效率**，同时避免大输家暴露按列表扩张比例同步恶化。

## Primary R1

Primary R1 完整保留 B0，不重新过滤 ACTIONABLE。

Supplemental：
- UNCONFIRMED：默认 `-5% <= Vs Buy Point <= +5%`
- BELOW_TRIGGER：默认 `-5% <= Vs Buy Point < 0%`
- 至少 1 个独立正证据族

**Primary R1 不使用 Geometry hard reject。** Geometry 只作为独立 ablation。

EXTENDED 不进入核心 R1，继续独立 exploratory。

## 独立正证据族

四个 evidence families：

1. **Volume**：`ibd_entry_volume_ratio >= 1.5` OR `volume_ratio >= 1.3`
2. **EPS**：PIT `eps_yoy_growth >= 25`
3. **RS-near-high**：`dist_to_52w_high_pct > -5`
4. **Supply contraction**：适用时 `pullback_v_is_dry == True`

Entry Volume 与 Weekly Volume 同属 Volume family，最多计 1 条独立证据。

False / Missing = neutral，不扣分、不自动淘汰。

## 双时钟评价

### Snapshot clock
从周末后的第一个交易日开始。

回答：
> 周末把它放进 Review List 后，它后来是不是大赢家 / 大输家？

### Opportunity clock
只对实际形成 1W Review Opportunity 的标的：

- ACTIONABLE：周末 `latest_close` 为 anchor；
- 非 ACTIONABLE：未来 5 日首次收盘进入 frozen Pivot 0%~+5% 的当日收盘为 anchor。

从 anchor 之后分别观察 1W / 2W / 3W / 4W。

回答：
> 真正形成可交易机会之后，后续质量如何？

两个时钟不得混用。

## Winner / Loser Oracle

每周全部 active signals 上建立 Snapshot Oracle。

另在实际形成 Review Opportunity 且对应 horizon 完整的样本上建立 Opportunity Oracle。

两个时钟均保留：
- Return Top5
- MFE Top5
- Big Winner = Return Top5 OR MFE Top5
- Return Bottom5
- Severe Loser = MAE <= -8%
- Big Loser = Bottom5 OR Severe Loser

## 容量归一化

Winner Recall 不能单独决定优劣。

必须同时报告：
- Selection Coverage
- Winner Capture Lift = Winner Recall / Selection Coverage
- Loser Capture Lift = Loser Inclusion / Selection Coverage
- Incremental Opportunities / Added Review
- Attention Multiplier vs B0

理想方向：
- Winner Recall ↑
- Winner Capture Lift 不下降
- Loser Capture Lift 不上升
- Incremental Opportunities / Added Review > 0

## 两阶段自进化

### Stage 1：核心结构网格
仅 24 个规则：

- Near 下界：-3 / -5 / -7
- Supplemental Status：U / U+BELOW
- 最少证据族：1 / 2
- Geometry：allow / exclude

### Stage 2：证据族 ablation
只围绕 Stage 1 的少数 Pareto finalists：

- ALL
- NO_VOLUME
- NO_EPS
- NO_RS_NEAR_HIGH
- NO_SUPPLY_CONTRACTION

不得一次性让 100+ 规则在同一 OOS 选择层竞争。

## Walk-forward

- expanding window
- 默认首次 train >=20 周
- 每 4 周冻结 OOS test block
- 每 fold 的 Stage1 + Stage2 只用 train
- test 绝不参与该 fold 规则选择
- 同一 train-champion 至少经历 3 个 OOS folds 才有资格成为 retrospective candidate

允许输出：
`NO_STABLE_NEXT_WEEK_REVIEW_RULE`

## 统计稳健性

同时报告：

- micro aggregation
- weekly macro average
- paired 4-week moving-block bootstrap
- week-level directional stability

最终判断以跨周稳定性优先，不让信号特别多的周支配结论。

## 生产边界

不修改：
- 生产 Skill
- Dashboard
- Futu sync
- C Rank
- ATR
- Daily State Machine
