# Next Week Review Selection Rule 研究协议

日期：2026-09-02  
状态：Research / Pre-registration v0.1  
分支：`research/next-week-review-selection`

## 1. 研究主体

### 1.1 核心问题

使用周末 `signal pool` 的 point-in-time（PIT）信息，从全部 `signal == True` 的 active signals 中选择一个**有限、可人工持续 Review 的 Next Week Review List**，并与当前实际工作流的 `ACTIONABLE-only` Futu 同步基线比较：

1. 是否减少未来 5 个交易日内值得 Review / 可能进入 Buy Point 区域的机会漏失；
2. 为此增加多少人工观察成本；
3. 新增候选的下一周风险收益质量是否明显恶化。

本研究不是 Top3 选股、不是未来收益预测、不是组合构建，也不是 Daily State Machine。

### 1.2 产品语义

目标输出是 **Review List / Watch List**，不是“直接给用户 3 只或 N 只买入”。

这与当前 IBD Review Dashboard 的工作流一致：先形成值得持续跟踪的候选列表，再由人工看图、状态更新和执行规则决定是否行动。

因此：

- `ibd_entry_status` 是当前时点的交易阶段，不再等同于“下周 Review 资格”；
- `ACTIONABLE` 不再拥有唯一 Futu 同步资格；
- `UNCONFIRMED`、`BELOW_TRIGGER`、`EXTENDED` 可以进入 Review List，但必须通过状态特定的可跟踪路径与质量检查；
- 本研究阶段不改变任何生产 Skill、Dashboard 或 Futu 同步代码。

## 2. 与现有 Skill 的关系

现有 `.agents/skills/ibd-candidate-prescreen/SKILL.md` 提供本研究的领域语义来源，但不直接复制其最终 Top3 决策：

### 2.1 复用

- Review Universe：`signal == True` 且 `ibd_candidate_rule` 非空；
- 缺失值三态语义：UNKNOWN / INFO_MISSING 不等于 FAIL；
- EPS 仅使用历史 PIT 可得值；
- `pullback_v_is_dry == True` 可作为正证据，`False` 不自动成为淘汰条件；
- Geometry 明确失败可作为结构风险；
- `volume_ratio`、`eps_yoy_growth`、`dist_to_52w_high_pct`、回踩 dry-up 等作为证据簇，而不是单一连续分数；
- 非 ACTIONABLE 的 Alpha Radar / Pullback Scout 思路，作为“值得后续人工观察”的先验语义。

### 2.2 不复用

- 不使用正式 Skill 的 `ACTIONABLE-only` 最终准入；
- 不使用 Top3 容量；
- 不使用 Industry 覆盖；
- 不使用 C Rank / `C_continuous` / `rank_C_continuous`；
- 不把 EPS 已知或 EPS >=25 设为 Watch List 硬门槛；
- 不把 `pullback_v_is_dry == False` 设为负分或硬拒绝；
- 不把突破日字段缺失自动视为非 ACTIONABLE 候选失败。

若本研究最终稳定成立，未来更合理的 Skill 结构应是：

```text
Weekend Active Signals
        ↓
Next Week Review List     ← 本研究
        ↓
人工/图表 Review
        ↓
Priority Review / Actionable subset
```

而不是直接从 active signals 跳到 Top3。

## 3. 数据与 PIT 边界

### 3.1 研究数据

优先复用已经审计过的历史基础设施：

- `backtest/ibd_skill_replay_pools/<snapshot>/breakout_follow_pool.csv`
- `backtest/ibd_skill_replay_pools/signal_eps_pit.csv`
- 已有日线价格缓存（由研究 runner 显式记录具体文件及 hash）
- `backtest/rd_agent_candidate_rule_audit/` 中已验证的 PIT 合并、价格裁剪和 label 工具可复用，但本研究独立输出，不覆盖旧实验。

当前已有审计结果显示可形成约 40+ 个独立周末 snapshot；研究 runner 必须再次输出实际可用周数、缺失周和覆盖漂移，不硬编码“42 周”。

### 3.2 禁止未来信息

周末选择时只能读取 snapshot 当日已经存在的字段。

冻结以下内容：

- `snapshot_date`
- `code`
- `signal`
- `ibd_candidate_rule`
- `ibd_candidate_price` / `ibd_trigger_price`
- `ibd_entry_status`
- 当周 pool 中全部正式字段
- 截至该 snapshot 可得的 EPS PIT

未来 5 个交易日价格只用于 outcome / opportunity label，不得反向修改周末特征、Pivot、Setup 或候选规则。

### 3.3 不新增字段

第一阶段只用现有 Pool 字段和已有 PIT 数据。

明确不加入：

- ATR
- RSI / MACD / ADX
- 新技术指标
- 新外部数据源
- 市场预测特征

ATR 如未来需要，仅作为 risk annotation 单独研究，不参与本阶段 Selection。

## 4. Baseline

### B0 — ACTIONABLE-only

B0 必须复现当前 Futu 同步语义，而不是 Skill Top3：

```text
signal == True
AND ibd_candidate_rule 非空
AND ibd_entry_status == ACTIONABLE
→ Next Week Review
```

B0 每周数量自然变化，不做 TopN，不使用 C Rank。

这是主比较锚点。

## 5. Next Week Review 候选规则族

本阶段只研究少量、可解释、预注册的规则，不做大规模参数挖掘。

### 5.1 统一结构失败

仅在 Geometry 字段有效时识别明确失败：

- `ibd_entry_breakout_range_ratio <= 0`；或
- `ibd_entry_close_position < 0.65`。

字段缺失不等于失败。对尚未形成有效突破日的 UNCONFIRMED，Geometry 缺失保持 UNKNOWN。

### 5.2 状态特定的可跟踪路径

使用 `current_vs_ibd_candidate_pct`，不把 Status 本身当唯一资格证。

默认研究窗口：

- ACTIONABLE：允许进入；
- UNCONFIRMED：`-5% <= Vs Buy Point <= +5%`；
- BELOW_TRIGGER：`-5% <= Vs Buy Point < 0%`；
- EXTENDED：`+5% < Vs Buy Point <= +10%`。

这组阈值是研究预注册值，不代表已经证明最优。

敏感性分析只允许少量经济意义阈值：

- 下方 Near Trigger：-3% / -5% / -7%；
- EXTENDED 上界：+7.5% / +10% / +15%。

不得在完整样本上搜索任意小数阈值。

### 5.3 正证据簇

定义 `support_count`，只累计正证据，不对缺失或未满足项扣分：

- `ibd_entry_volume_ratio >= 1.5`（字段已知时）；
- `volume_ratio >= 1.3`；
- PIT `eps_yoy_growth >= 25`；
- `dist_to_52w_high_pct > -5`；
- 适用回踩规则中 `pullback_v_is_dry == True`。

原则：

- EPS unknown = neutral；
- `pullback_v_is_dry == False` = risk/context，不扣分；
- entry volume missing for UNCONFIRMED = neutral；
- 不按 EPS、Volume 数值大小连续加权。

### 5.4 三个预注册研究变体

#### R1 — PATH

- 通过状态特定可跟踪路径；
- 排除明确 Geometry failure；
- 不要求最低 `support_count`。

目的：测量“仅解除 ACTIONABLE gate”本身能增加多少机会覆盖与注意力成本。

#### R2 — BALANCED

- R1 条件；
- `support_count >= 1`。

目的：在扩大状态覆盖后，使用至少一条独立正证据过滤明显弱候选。

#### R3 — STRICT

- R1 条件；
- `support_count >= 2`。

目的：测试更严格质量门槛是否能减少 Watch List 数量而不过度损失机会捕获。

R1/R2/R3 均不得使用 C Rank。

## 6. 数量与排序

### 6.1 先研究 Eligibility，再研究 Cap

首先记录每个变体自然产生的周度 Watch List 数量：

- mean
- median
- P75
- P90 / P95
- max

如果 R2/R3 已自然形成合理数量，则不强制 TopN。

### 6.2 Attention Frontier

为研究人工注意力成本，同时评估：

- Top 10
- Top 15
- Top 20

以及一个关键公平对照：

### R2 — Attention Matched

每周 R2 只保留与 B0 当周 ACTIONABLE 数量相同的 N 只。

这用于回答：

> 在**同样人工观察数量**下，从全部状态中选择是否比 ACTIONABLE-only 捕获更多下一周机会？

### 6.3 Cap 时的确定性优先级

只有需要 Cap 时才排序，且不使用 C Rank。

排序键：

1. `support_count` DESC；
2. `abs(current_vs_ibd_candidate_pct)` ASC；
3. 明确 Geometry failure 已在前置排除；
4. `code` 字典序；
5. CSV 原始行序。

Status 不作为首要排序键，避免再次把 ACTIONABLE 隐式变成质量排序。

## 7. 未来 5 个交易日 Outcome

本研究测的是“周末是否值得纳入 Review List”，不是交易系统收益。

### 7.1 Frozen Pivot

每只股票使用周末冻结的 `ibd_candidate_price` 作为未来 5 日路径参考，不在未来数据中重新发现 Pivot。

### 7.2 Review Opportunity

主 opportunity 定义为：

```text
weekend ACTIONABLE
OR
未来 5 个交易日任一收盘价进入 [Pivot, Pivot * 1.05]
```

并按周末初始状态记录事件来源：

- CURRENT_ACTIONABLE
- UNCONFIRMED_TO_ZONE
- BELOW_TO_ZONE
- EXTENDED_RETEST_TO_ZONE

这里的 “to zone” 只代表**值得人工 Review 的 Buy Point 区域事件**，不自动等价于有效买入信号。

不在第一阶段实现 Daily State Machine，不把 EXTENDED 回落自动改名为 ACTIONABLE_RETEST。

### 7.3 风险收益质量

从 snapshot 后第一个交易日 Open 作为统一观察基准，计算未来 5 个交易日：

- `forward_5d_return_pct`
- `mfe_5d_pct`
- `mae_5d_pct`
- `stop_8_within_5d`（仅作风险统计，不代表实际已开仓）

这些是 Review List 的路径质量指标，不冒充真实策略交易收益。

## 8. 主要评价指标

### 8.1 Opportunity Capture

```text
Capture Rate =
被该规则周末选中的 Review Opportunity 数
/
当周全部 Active Signals 中的 Review Opportunity 数
```

同时报告：

- 非 ACTIONABLE Incremental Capture；
- 各初始 Status 的机会转化率；
- B0 漏掉但 R1/R2/R3 捕获的事件数。

### 8.2 Attention Cost

- 平均/中位 Watch List Size；
- P90/P95 周度列表规模；
- 每新增 1 个 Review 候选带来的 incremental opportunity；
- Attention-Matched 对照。

### 8.3 Risk / Reward Quality

至少报告：

- median / mean 5D return；
- median MFE / MAE；
- MAE 分位数；
- -8% adverse touch rate；
- 周级最差候选路径；
- 不同 Status 来源的分层结果。

不以单一平均收益决定胜负。

## 9. 统计与防过拟合

### 9.1 Retrospective 性质

历史 replay 数据已经在其他研究中被观察过，因此本研究不能声称拥有真正 sealed holdout。

所有结果标记为：

`retrospective_pre_registered_replay`

### 9.2 时间阻断

按可用 snapshot 时间顺序：

- 前约 2/3：Discovery；
- 后约 1/3：Blocked Validation。

R1/R2/R3 规则在运行前冻结。敏感性阈值只在 Discovery 中比较；Validation 不允许根据结果再改规则。

同时做 week-block bootstrap / rolling robustness，避免把股票行当成独立同分布样本。

### 9.3 成功标准

不预设“新规则一定优于 B0”。

R2/R3 只有在以下方向同时成立时才值得进入下一阶段：

1. Capture 明显高于 B0；
2. Attention 增量可接受，或 Attention-Matched 下仍有增益；
3. 5D MAE / adverse tail 没有出现不可接受恶化；
4. 结果不是由极少数周或单一 Status 驱动；
5. Blocked Validation 方向与 Discovery 基本一致。

若不满足，结论允许是：

`NO STABLE NEXT-WEEK REVIEW RULE`

## 10. 研究输出

独立目录：

`backtest/next_week_review_selection/output/`

至少生成：

- `data_audit.md`
- `weekend_event_panel.csv/parquet`
- `opportunity_labels.csv`
- `weekly_selection_counts.csv`
- `baseline_vs_variants.csv`
- `attention_frontier.csv`
- `status_transition_summary.csv`
- `risk_reward_summary.csv`
- `blocked_validation.csv`
- `experiment_manifest.yaml`
- `research_report.md`

所有输出保留 `snapshot_date + code`，可回放到具体案例。

## 11. Skill / Futu 的生产变更边界

本研究分支不修改现有生产 Skill。

如果研究通过，再开单独生产变更：

1. Skill 增加独立的 **Next Week Review List**；
2. 该列表不是“正式推荐 Top3”，而是有限人工 Review Universe；
3. Futu 周末分组同步该 Review List，而不是 `ACTIONABLE-only`；
4. 每个候选输出：
   - Code
   - Weekend Status
   - Vs Buy Point
   - Setup
   - Review reason
   - Risk / missing info
5. 第二阶段才实现下周每日价格刷新与状态变更记录。

## 12. 当前明确不研究

- C Rank 是否有效；
- Top3 portfolio construction；
- 行业配额；
- ATR；
- Daily State Machine；
- 买卖/止损/仓位策略；
- 市场状态作为候选硬门槛；
- 新外部数据源。

这些问题与本研究主体解耦。
