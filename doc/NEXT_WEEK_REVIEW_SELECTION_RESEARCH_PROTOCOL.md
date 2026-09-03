# Next Week Review Selection 研究协议 v0.5

日期：2026-09-03  
分支：`research/next-week-review-selection`

## 目标

v0.5 不再重复证明 Primary R1 的高召回特性，而是正式回答：

> **能否从 Supplemental Candidates 中找到一个 OOS 稳定、单位 Review 注意力效率更高的准入机制？**

最终目标不是最大化 Winner Recall，而是：

- Tradable Winner Capture Lift ↑
- Tradable Loser Capture Lift 不上升
- Incremental Opportunities / Added Review > 0
- Review List 自然规模尽量小
- 跨 OOS fold 方向稳定

## 1. Train provisional champion 必须进入 OOS

废除 v0.4 的训练期 hard veto：

`stability_floor >= 2/3` 不再是“能否进入 OOS”的门槛。

每个 formal fold：

1. 仅使用 train-as-of 数据
2. Stage 1 核心结构搜索
3. Stage 2 evidence-family ablation
4. 在 Pareto frontier 中选择一个 provisional champion
5. 冻结
6. 进入下一块 OOS

训练稳定性只用于 Pareto 后的 deterministic tie-break，不拥有最终否决权。

真正的否决权属于 OOS。

## 2. Pareto 目标

不构造加权综合分。

Pareto objectives：

- `Tradable Winner Capture Lift` 最大化
- `Tradable Loser Capture Lift` 最小化
- `Incremental Opportunities / Added Review` 最大化
- `Avg Watchlist Size Delta vs B0` 最小化

Pareto 内 tie-break：

1. training block stability
2. Winner Lift
3. Loser Lift
4. incremental opportunity efficiency
5. list size
6. rule simplicity
7. deterministic rule name

## 3. 选择集合去重

参数不同不等于研究假设不同。

每个 train fold 对每条规则计算：

`SHA256(sorted(snapshot_date, code) selected set)`

选择集合完全相同的规则视为一个 effective hypothesis。

代表规则：

- 优先 complexity 更低
- 再按 deterministic name

Stage 1 与 Stage 2 均先去重后再进入 Pareto/选 champion。

## 4. Formal OOS 与 Tail

默认：

- first train = 20 snapshot weeks
- formal test block = 4 snapshot weeks
- expanding window

只有完整 4-week test block 进入正式 OOS verdict。

当前 42 周预期结构：

- 5 个完整 formal OOS folds
- 最后 2 周为 `TAIL_EXPLORATORY`

Tail 可以计算，但不得进入：

- stability rate
- adaptive-policy verdict
- static-rule convergence verdict

## 5. Adaptive Policy

每个 fold 的 champion 可以不同。

因此主 OOS 对象不是“某一条固定规则”，而是：

> **只使用过去数据选择下一阶段 Review Rule 的 adaptive policy。**

正式输出：

- 每 fold train champion
- 每 fold OOS metrics
- 跨 fold adaptive aggregate
- Winner Lift direction rate
- Loser Lift non-worse rate
- Opportunity positive rate
- incremental opportunity efficiency
- attention multiplier

v0.5 retrospective adaptive candidate gate：

- formal folds >= 3
- Opportunity Δ > 0 的 fold 比例 >= 60%
- Winner Lift Δ >= 0 的 fold 比例 >= 60%
- Loser Lift Δ <= 0 的 fold 比例 >= 60%
- mean Opportunity Δ > 0
- mean Winner Lift Δ >= 0
- mean Loser Lift Δ <= 0
- mean Incremental Opportunities / Added Review > 0
- **2W / 3W / 4W 各 horizon 的 mean Winner Lift Δ 均 >= 0**
- **2W / 3W / 4W 各 horizon 的 mean Loser Lift Δ 均 <= 0**

最后两条防止“4W 正向把 2W/3W 的劣化平均掉”。

否则：

`NO_STABLE_ADAPTIVE_POLICY`

## 6. Rule Convergence

同时回答：

> train-only optimizer 是否反复收敛到相同结构？

分别统计：

- exact rule
- structural key：Near / Status / min evidence / Geometry
- evidence profile

只有 exact rule 至少出现在 3 个 formal folds，且其被选中 fold 的 OOS 方向达到稳定门槛，才标记：

`RETROSPECTIVE_CONVERGENT_RULE_CANDIDATE`

否则：

`NO_CONVERGENT_STATIC_RULE`

该标签仍不是 production authorization。

## 7. Setup-balanced sensitivity

历史 price-cache coverage 在 setup 间不均衡，因此正式补充：

- 按 `ibd_candidate_rule` 分层
- 每个有 >=10 个完整 1W outcome 的 setup 才进入 balanced summary
- setup 等权，不按样本行数加权

对 Primary R1 和 formal OOS Adaptive Policy 都输出：

- Opportunity Δ
- Winner Lift Δ
- Loser Lift Δ
- Severe loser exposure Δ
- Attention Δ
- incremental opportunity efficiency

若优势只来自少数高 coverage setup，不认为稳健。

## 8. 保持不变的边界

继续保留：

- B0 ACTIONABLE 全保留
- Primary R1 无 Geometry hard reject
- 4 个独立 evidence families
- False / Missing neutral
- Snapshot clock + Opportunity clock
- 1W/2W/3W/4W
- price-path coverage audit
- horizon-aware as-of censoring
- moving-block bootstrap
- EXTENDED exploratory only
- C Rank / ATR 不参与
- 不修改生产 Skill / Dashboard / Futu
