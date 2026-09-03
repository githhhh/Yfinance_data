# Next Week Review Selection 研究协议 v0.4

日期：2026-09-03  
分支：`research/next-week-review-selection`

## 核心目标

验证：

> 在有限 Review 注意力下，Near Buy Point + 独立正证据能否比 ACTIONABLE-only 捕获更多真正可交易的大赢家，同时不让大输家暴露按列表扩张比例同步恶化。

Primary R1、四个 evidence families、双时钟、容量归一化规则保持 v0.3 不变。

## v0.4 新增：Price-Path Coverage Audit

历史 active signal 不允许因为缺少未来价格路径而静默退出分母。

每个 event 记录：

- price cache 是否存在 ticker
- ticker cache 首尾日期
- snapshot 后可用 session 数
- path state：
  - MISSING_SYMBOL
  - EMPTY_OR_INVALID_BARS
  - NO_FORWARD_BARS
  - SHORT_1W / SHORT_2W / SHORT_3W / SHORT_4W
  - COMPLETE_4W

必须按以下维度输出 coverage：

- snapshot week
- weekend status
- setup
- signal source
- ticker
- 1W 缺失明细

目的：解释 `active signals -> evaluable outcomes` 的缺口，并检查缺失是否与状态/规则/来源系统相关。

## v0.4 新增：Horizon-Aware As-Of Walk-Forward

废除旧的：

> 只有“整周 4W 覆盖 >= 某阈值”的 snapshot week 才进入优化。

新设计：

- 保留全部 snapshot weeks；
- 1W/2W/3W/4W 各自使用自己的可用历史；
- 每个 label 记录实际 `end_date`；
- 每个 Walk-forward fold 在 test block 第一个 snapshot close 作为 cutoff；
- train 中只有 `label_end_date <= cutoff` 的 outcome 可以被使用；
- cutoff 后才完成的 outcome 自动重新标为 censored；
- Oracle 必须在 as-of masking 后重新构造。

因此：

> 1W 可以使用更多近期训练历史，4W 自动少用几周，但不能为了 4W 把整个周从 1W/2W 研究中删除。

默认：

- first train = 20 snapshot weeks
- test block = 4 weeks
- expanding window
- train-only Stage1 + Stage2 selection
- 同一 train-selected rule 至少 3 个真实 OOS folds 才能成为 retrospective candidate

Champion 状态语义：

- OOS folds < 3 → `INSUFFICIENT_OOS_HISTORY`
- OOS folds >= 3 但无稳定规则 → `NO_STABLE_NEXT_WEEK_REVIEW_RULE`
- 达到稳定门槛 → `RETROSPECTIVE_CANDIDATE`

## 统计输出

继续同时使用：

- micro aggregation
- weekly macro aggregation
- paired moving-block bootstrap
- capacity-normalized Winner/Loser Capture Lift
- Snapshot clock
- Opportunity clock
- missed winners / included losers case audit

Bootstrap 不再只给 2W~4W 聚合均值；必须同时输出 2W、3W、4W horizon-specific 指标。

## 生产边界

本研究仍不修改：

- 生产 Skill
- Dashboard
- Futu sync
- C Rank
- ATR
- Daily State Machine
