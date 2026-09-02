# Next Week Review Selection 研究协议 v0.2

日期：2026-09-02  
分支：`research/next-week-review-selection`  
研究基座：`codex/clean-latest-quant-trade-replay-pools`

## 1. 核心假设

验证：

> **Near Buy Point 的现实路径 + 至少一条独立正质量证据，是否比 ACTIONABLE-only 更适合作为周末 Review List 的准入机制。**

Review List 是“下周值得持续跟踪”的候选集合，不是 Top3 推荐、交易指令或组合构建。

## 2. B0 与 R1

### B0
所有周末 active signal 中：
- `signal == True`
- `ibd_candidate_rule` 非空
- `ibd_entry_status == ACTIONABLE`

全部进入 Review List。

### R1
R1 必须完整保留 B0，不重新过滤 ACTIONABLE，只额外加入 Supplemental Candidates：

- UNCONFIRMED：默认 `-5% <= Vs Buy Point <= +5%`
- BELOW_TRIGGER：默认 `-5% <= Vs Buy Point < 0%`
- 至少一条独立正证据
- 若存在明确 Geometry failure，可在规则变体中排除

**EXTENDED 不进入核心 R1**，单独 exploratory 分析其 retest 行为。

## 3. 正证据

只使用现有 Pool / PIT 字段：

- `ibd_entry_volume_ratio >= 1.5`
- `volume_ratio >= 1.3`
- PIT `eps_yoy_growth >= 25`
- `dist_to_52w_high_pct > -5`
- 适用 pullback rule 时 `pullback_v_is_dry == True`

语义：

- 正证据成立：支持入选
- False：neutral
- 缺失：neutral
- `pullback_v_is_dry == False`：不得负分或自动淘汰
- UNCONFIRMED 没有有效 breakout geometry / entry volume 时保持 UNKNOWN / N/A

不使用 C Rank、ATR 或新增技术指标。

## 4. 多周期 Outcome

周末选择只使用 snapshot PIT 信息。未来价格只用于评价。

- **1W / 5 sessions**：是否形成下周 Review opportunity；并记录 Return / MFE / MAE / -8% adverse touch
- **2W / 10 sessions**：初步 follow-through
- **3W / 15 sessions**：赢家形成质量
- **4W / 20 sessions**：持续性和尾部失败

1W Review Opportunity：
- 周末当前已经 ACTIONABLE；或
- 非 ACTIONABLE 在未来 5 个交易日收盘进入 frozen Pivot 的 0%~+5% 区域

## 5. Weekly Winner / Loser Oracle

Oracle 必须在**每周全部 active signals**中建立：

每个 1W/2W/3W/4W horizon：
- Return Top5
- MFE Top5
- Big Winner = Return Top5 OR MFE Top5
- Return Bottom5
- Severe Loser = MAE <= -8%
- Big Loser = Return Bottom5 OR Severe Loser

同时保留 Top10% / Bottom10% 作为敏感性口径。

## 6. 综合评价指标

至少报告：
- 1W Opportunity Recall
- 非 ACTIONABLE Opportunity Recall
- 1W/2W/3W/4W Winner Return Recall
- Winner MFE Recall
- Big Winner Recall
- Big Loser Inclusion / Exclusion
- Big Loser Density
- Severe Loser Exposure
- Median Return / MFE / MAE
- 平均 / 中位 / P95 Review List Size
- Opportunities per Review

强制输出：
- `missed_big_winners.csv`
- `included_big_losers.csv`

## 7. 自我优化规则空间

只允许在小型可解释规则语法中演化：

- Near Buy Point 下界：-3 / -5 / -7
- Supplemental Status：
  - UNCONFIRMED
  - UNCONFIRMED + BELOW_TRIGGER
- 最少正证据：>=1 / >=2
- 明确 Geometry failure：exclude / allow
- Positive-evidence leave-one-out：
  - ALL
  - NO_ENTRY_VOL
  - NO_WEEKLY_VOL
  - NO_EPS
  - NO_52W
  - NO_DRY

不得搜索任意小数阈值，不新增指标。

## 8. Walk-forward

使用 expanding-window：
- 默认至少 20 个历史周作为首次训练期
- 后续每 4 周作为一个冻结 OOS test block
- 每个 fold 只用过去数据选择 train champion
- test block 绝不参与该 fold 的规则选择

训练期选择采用 **Pareto + 稳定性**：

最大化：
- 1W Opportunity 增量
- 2W~4W Big Winner Recall 增量
- Big Loser Exclusion

最小化：
- Severe Loser Exposure
- Attention Cost

并以跨训练时间块方向稳定性和规则简单度打破平局。

允许输出：
`NO_STABLE_NEXT_WEEK_REVIEW_RULE`

## 9. Retrospective Champion

历史样本不是 sealed holdout，因此最终 champion 只能标记为：

`RETROSPECTIVE_CANDIDATE`

只有未来 prospective pools 验证后，才允许讨论生产 Skill / Futu 变更。

## 10. 生产边界

本研究不修改：
- 生产 `ibd-candidate-prescreen` Skill
- Dashboard
- Futu sync
- C Rank
- Daily State Machine
