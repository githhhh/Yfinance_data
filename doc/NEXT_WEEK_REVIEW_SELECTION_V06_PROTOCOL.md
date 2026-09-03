# Next Week Review Selection v0.6 — Deterministic Discriminative Study

日期：2026-09-03  
分支：`research/next-week-review-selection`

## 结论边界

v0.5 已经表明：

- ACTIONABLE-only 漏掉大量 next-week opportunities；
- broad R1 主要靠扩容提高 Recall；
- Adaptive evidence-family optimizer 没有提高单位 Review 的 Winner Capture Lift；
- train 结构 5/5 收敛到 `Near5 + UB + E2 + Geometry allow`。

v0.6 **不使用 rd-agent**，也不扩展到 ML。

目标是确定：

> 在现有 PIT 字段中，是否存在一个简单、可解释、注意力成本受控的 refinement，可以把 v0.5 的稳定结构进一步提纯。

## 重要统计声明

v0.5 的 Formal OOS 结果已经被研究者看到。

因此 v0.6 不能把同一批历史周重新称为“全新 sealed OOS”。

正式语义：

`retrospective_reused_history_confirmation`

即：

- 代码仍严格执行 train-only / frozen replay；
- 但这些历史测试周已经在前一研究迭代中被观察；
- 即使找到候选，也只能标记为 retrospective candidate；
- 真正 production authorization 必须等待未来 sealed weeks。

## 固定 Anchor

v0.6 不再搜索结构层，固定：

```
Near Buy Point <= 5%
status in {UNCONFIRMED, BELOW_TRIGGER}
>= 2 independent evidence families
Geometry allow
```

ACTIONABLE 仍完整保留。

EXTENDED 仍不进入核心 selector。

## Discriminative fields

仅使用周末已经存在的 PIT 字段：

- current_vs_ibd_candidate_pct
- base_depth_pct
- base_duration_weeks
- pullback_pct
- pullback_duration_weeks
- ibd_candidate_rule
- ibd_entry_volume_ratio
- volume_ratio
- PIT eps_yoy_growth
- dist_to_52w_high_pct
- pullback_v_is_dry
- 四个既有 evidence-family flags

禁止：

- 新技术指标
- ATR
- C Rank
- future-return feature
- 自动生成任意阈值
- ML
- rd-agent

## Discovery statistics

第一个 Formal OOS fold 之前的 20 个 snapshot weeks 作为 discovery window。

在 as-of cutoff 后：

1. 只取 Anchor supplemental cohort；
2. 输出 coarse bucket statistics；
3. Persistent Winner 定义：2W/3W/4W 中至少 2 个 horizon 为 Opportunity-clock Big Winner；
4. Persistent Loser 同理；
5. 输出固定交叉：
   - setup × Vs Buy Point
   - setup × base depth
   - Vs Buy Point × RS
   - base depth × dry-up
   - pullback depth × dry-up

这些统计只负责解释，不直接生成任意阈值。

## 固定候选库

候选 refinement 最多增加 2 个条件。

阈值全部预先固定为领域 coarse buckets，例如：

- |Vs Buy Point| <= 2 / 3
- base depth <= 33 / 50
- base duration 7–65
- pullback depth <= 15
- pullback duration 3–10
- pullback-like setup
- RS within 10%
- require Volume / EPS / RS / Supply evidence
- 少量具有领域含义的两条件组合

候选库规模固定为 25。

## Discovery rule selection

每条候选在 discovery train 上：

- 与 B0 比较；
- selection signature 去重；
- attention multiplier 必须 <= 1.50x B0；
- 必须实际增加可评估 Review；
- incremental opportunities / added review > 0；
- Opportunity Recall Δ > 0；
- 2–4W mean Winner Lift Δ >= 0；
- 2–4W mean Loser Lift Δ <= 0；
- 至少 2/3 horizons 同时满足 Winner Lift >=0 且 Loser Lift <=0；
- discovery train 切成 3 个 chronological blocks，至少 2/3 blocks 同时满足 Opportunity Δ > 0、Winner Lift Δ >= 0、Loser Lift Δ <= 0。

这一步用于避免某一个大信号周或单一时期主导 discovery 选择。因为 B0_NO_EXPANSION 是合法 fallback，所以稳定性门槛不会强迫一个差规则进入后续 replay。

没有满足者：

`B0_NO_EXPANSION`

满足者进入 Pareto：

- Winner Lift ↑
- Loser Lift ↓
- incremental opportunity efficiency ↑
- attention multiplier ↓

不使用加权综合分。

第一 discovery window 只选择 **一个 static discovery rule**。

## 两条验证轨

### A. Static discovery rule

第一 20 周选出的规则冻结，跨后续 5 个完整 4-week blocks 保持不变。

这是 v0.6 的主结论对象。

### B. Adaptive discriminative policy

每个 expanding train fold 可以重新选择 refinement。

只作为 secondary evidence，用于判断 refinement 是否存在时变性。

## Candidate gate

一个 retrospective discriminative candidate 至少要求：

- Formal folds >= 3
- expanded fold rate >= 60%
- Opportunity Δ > 0 fold rate >= 60%
- Winner Lift Δ >= 0 fold rate >= 60%
- Loser Lift Δ <= 0 fold rate >= 60%
- mean Opportunity Δ > 0
- mean Winner Lift Δ >= 0
- mean Loser Lift Δ <= 0
- mean attention multiplier <= 1.50
- incremental opportunity efficiency > 0
- 2W / 3W / 4W 各自 mean Winner Lift Δ >= 0
- 2W / 3W / 4W 各自 mean Loser Lift Δ <= 0

否则：

`NO_STABLE_DISCRIMINATIVE_RULE`

即使通过，也仍然不是 production authorization。

## 停止规则

如果 v0.6 的 static 与 adaptive 两轨都失败：

> 停止继续在相同 PIT 字段和同一批历史数据上扩大规则搜索。

下一步只能二选一：

1. 等待未来 sealed weeks；
2. 引入新的、事先定义的信息源/特征后重新预注册。

不继续通过增加参数、阈值或组合把这 42 周历史拟合穿。
