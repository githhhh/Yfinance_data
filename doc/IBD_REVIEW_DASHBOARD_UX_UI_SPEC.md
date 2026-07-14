# IBD Review Dashboard UX/UI 实施规范（最终版）

> 交付对象：Gemini Pro  
> 最终版本：2026-07-14  
> 本文定义页面最终结构、交互规则与验收标准。字段语义以 `BREAKOUT_FOLLOW_POOL_SCHEMA.md` 为准。

## 0. 核心体验要求

本版必须完整实现以下四项：

1. **明确的全量入口**：Status Queue 上方提供 `All Signals · N`，用户随时可以回到当前 Route 下的完整信号视图。
2. **详情紧邻决策结果**：Selected Row Detail 位于 Result Summary 与 Decision Table 之间，点击股票后不需要滚动到表格底部确认信息。
3. **高密度表格性能**：Decision Table 使用 AG Grid 行虚拟化，固定表头和左侧 Code 列，支持 80～300 行连续 Review。
4. **单一心流说明入口**：Header 提供唯一 `ⓘ` 图标，点击打开简短的 `IBD Review Flow` 说明窗；熟练用户可以完全忽略该入口。

## 1. 产品目标

Dashboard 是周三收盘后和周末收盘后的 IBD Review 工作台。用户在一个页面内完成：

1. 识别当前可行动标的。
2. 查看等待量价确认且接近候选价的标的。
3. 识别回落至候选价下方或已经延伸的标的。
4. 复制当前队列代码到 Futu。
5. 切换到 C Rank Reference 对照原始排名。

主路径为“状态队列 → 轻量筛选 → 表格 Review → 复制代码”。

## 2. 页面总结构

页面采用单列主工作区，模式切换位于顶部，不使用侧边栏。

```text
Header：名称、Snapshot、Pool 数量、模式切换
Status Queue：四个状态卡
Filter Bar：Route、Candidate Distance、量能阈值、Reset
Result Summary：结果数、当前排序、Copy Codes
Selected Row Detail
Decision Table
```

主工作区宽度随浏览器伸缩；桌面端内容宽度保持紧凑，避免筛选器横向拉得过长。

## 3. Header

左侧：

```text
Breakout Pool
Snapshot 2026-07-10 · 717 Total · 83 Active Signals
```

右侧使用分段按钮：

```text
[IBD Review] [C Rank Reference]
```

模式按钮右侧放置一个 `ⓘ` 图标按钮，悬停提示 `查看 IBD Review 心流`。点击打开 `IBD Review Flow` 说明窗。

说明窗固定为以下五步：

1. 查看全局：从 All Signals 确认当前信号总量和各状态工作量。
2. 决定优先级：先处理 ACTIONABLE，再查看接近 Candidate 的 UNCONFIRMED；BELOW_TRIGGER 用作等待队列，EXTENDED 用于识别追高风险。
3. 按需收窄：使用 Route、距离和量能缩小当前工作列表，并在 Selected Row Detail 核对单票。
4. 复制列表：将当前筛选结果复制到 Futu 做图表 Review 和执行计划。
5. C Rank 对照：C Rank Reference 只作为比较，不作为 IBD Review 的隐藏筛选条件。

说明窗默认关闭；点击关闭按钮、遮罩或按 `Esc` 关闭。打开和关闭说明窗不改变模式、筛选条件、排序及当前选中标的。全页面只设置这一个心流入口。

Snapshot 只显示日期，不显示 `00:00:00`。数据检查通过时显示简短状态 `Data Ready`。

## 4. IBD Review 模式

### 4.1 Status Queue

显示四个可点击状态卡，顺序固定：

```text
ACTIONABLE → UNCONFIRMED → BELOW_TRIGGER → EXTENDED
```

状态卡上方显示 `All Signals · N`。点击后清除 Status 选择，保留 Route、距离和 Weekly Vol 条件。Entry Vol 是确认状态专用条件，切换到 All 时清空。

每张卡包含：

| 状态 | 主数字 | 辅助信息 |
|:--|--:|:--|
| ACTIONABLE | 当前数量 | `0%～5% above candidate` |
| UNCONFIRMED | 当前数量 | `N within +3%` |
| BELOW_TRIGGER | 当前数量 | `Below candidate` |
| EXTENDED | 当前数量 | `Above +5%` |

交互：

- 初始状态为 All，`All Signals` 选中，四张状态卡均未选中。
- 点击卡片筛选对应状态。
- 再次点击已选卡片恢复 All。
- 卡片数量基于 `signal=True + 当前 Route`，不受距离和量能条件影响。

颜色只用于强化状态，文字标签始终保留：

- ACTIONABLE：绿色。
- UNCONFIRMED：琥珀色。
- BELOW_TRIGGER：红色。
- EXTENDED：中性灰蓝色。

### 4.2 Filter Bar

Filter Bar 单行优先排列，窄屏自动换行：

```text
Route | Distance Min | Distance Max | Entry Vol Min | Weekly Vol Min | Reset
```

规则：

- Route 默认 `All Routes`，选项来自当前 CSV。
- Candidate Distance 使用两个数字输入，单位为百分点。
- Distance 留空表示该侧无边界。
- Entry Vol Min 只在 `ACTIONABLE`、`BELOW_TRIGGER`、`EXTENDED` 状态启用。
- Weekly Vol Min 对所有状态启用。
- 控件范围和提示值基于 `signal=True + 当前 Route + 当前 Status`。
- 所有启用条件使用 AND。
- Reset 恢复 Route=All、Status=All、距离为空、量能阈值为空。

### 4.3 Result Summary

表格上方只保留一行：

```text
83 results · Sorted by Entry Status → C Rank          [Copy 83 Codes]
```

结果数以 Active Signal 为分母，不使用 Total Pool 作为漏斗分母。

### 4.4 Selected Row Detail

结果摘要下方、表格上方显示当前选中标的一行详情：

```text
DELL · Candidate 429.15 · Latest 434.97 · +1.36% · Reject: daily_volume_not_confirmed · 52W -7.35% · EPS +281.26%
```

详情只承担补充信息，不新增第二套筛选器。点击表格行后在原位置更新，使用户不需要滚动到表格底部确认单票信息。

### 4.5 Decision Table

默认表格列：

```text
Code
Entry Status
Route
Current vs Candidate
Latest Close
IBD Entry Volume / Reject Reason
Weekly Volume
C Rank
```

规则：

- `Code` 固定在左侧并支持单击复制。
- `Current vs Candidate` 显示正负号和 `%`，例如 `+2.43%`。
- `Latest Close` 显示两位小数。
- `IBD Entry Volume` 显示 `2.26x`；UNCONFIRMED 显示短驳回原因。
- `Weekly Volume` 显示 `0.93x`。
- `C Rank` 位于末列。
- 默认排序：ACTIONABLE、UNCONFIRMED、BELOW_TRIGGER、EXTENDED；同状态内按 C Rank 升序。
- 点击行更新 Selected Row Detail，不改变筛选结果。
- 表头始终固定；垂直滚动时保持可见。
- Code 列固定在左侧；横向滚动时保持可见。
- 行数据使用 AG Grid 虚拟化，不一次性渲染全部 DOM 行。

状态标签使用紧凑 Badge。表格行保持统一高度，避免将每行设计成卡片。

表格视区内只渲染可见行，80～300 行数据保持流畅滚动。用户点击某行后保持该行选中态，直到筛选结果移除该股票或用户选择另一行。

## 5. C Rank Reference 模式

C Rank Reference 是独立对照模式，保留原始排名语义。

顶部显示：

```text
C Rank Reference · 83 Active Signals
Top N: All / 10 / 20 / 30 / 50                [Copy Codes]
```

数据规则：

```text
signal=True
rank_C_continuous ASC
```

默认列：

```text
Code
C Rank
C Continuous
Entry Status
Current vs Candidate
Route
Weekly Volume
```

C Rank Reference 不继承 IBD Review 的 Route、Status、Distance 或 Volume 状态。切回 IBD Review 后恢复之前的 Review 筛选状态。

## 6. KPI 与图表

首页使用四个 Status Queue 数字表达工作量，不再单独排列 Median KPI。

IBD Review 主页面不放分析型图表。Route 的历史有效性、前瞻收益、MAE/MFE 和样本量进入独立历史分析报告，不参与当前队列心流。

## 7. 视觉规范

- 延续当前深色主题。
- 页面主色用于选中状态、主按钮和焦点。
- 标题、快照、模式切换位于同一视觉层级。
- 筛选区使用紧凑横排，不使用大面积折叠面板。
- 状态卡高度一致，数字优先于说明文字。
- 表格是页面最大视觉区域。
- 主要信息使用正常对比度；时间、说明和单位使用次级文本色。
- 颜色与文字标签同时表达状态。

## 8. 响应式规则

- 宽屏：四张状态卡同一行，Filter Bar 单行。
- 中等宽度：状态卡保持两列，Filter Bar 自动换行。
- 小屏：表格保留 Code、Status、Distance、Volume、C Rank，其余字段通过 Selected Row Detail 查看。

## 9. 数据与格式

| 字段 | 展示格式 |
|:--|:--|
| `snapshot_date` | `YYYY-MM-DD` |
| `latest_close` | `0.00` |
| `current_vs_ibd_candidate_pct` | `+0.00%; -0.00%` |
| `ibd_entry_volume_ratio` | `0.00x` |
| `volume_ratio` | `0.00x` |
| `dist_to_52w_high_pct` | `+0.00%; -0.00%` |
| `eps_yoy_growth` | `+0.00%; -0.00%` |
| `rank_C_continuous` | 整数 |

空值显示 `—`。

## 10. 当前快照验收基线

```text
Total Pool       717
Active Signal     83
ACTIONABLE        10
UNCONFIRMED       62
BELOW_TRIGGER      0
EXTENDED          11
UNCONFIRMED ≤ +3% 38
```

默认 IBD Review 显示83行；默认 C Rank Reference 显示83行。

## 11. 交互验收

- 页面打开后默认进入 IBD Review。
- Header 在首屏完整显示 Snapshot 和两个模式。
- 点击 `ⓘ` 打开心流说明窗；关闭后页面状态保持不变。
- 心流说明完整覆盖 All Signals、状态优先级、筛选、单票确认、Futu 和 C Rank Reference。
- `All Signals` 数量随 Route 更新；点击清除 Status 和 Entry Vol，保留 Route、Distance 与 Weekly Vol。
- Status 卡点击后表格、结果数和 Copy Codes 同步更新。
- Route 改变后 Status 卡数量同步更新。
- Distance 和 Volume 条件按 AND 更新结果。
- UNCONFIRMED 状态下 Entry Vol Min 处于不可编辑状态。
- Reset 一次恢复默认 Review 队列。
- 表格格式正确显示百分点和量比。
- 默认排序符合状态优先级和 C Rank 次级顺序。
- 80～300 行使用虚拟滚动，表头和 Code 列保持固定。
- C Rank Reference 严格按 Rank 升序，Top N 和 Copy Codes 正确。
- 两个模式的筛选状态相互独立。
- 页面首屏不出现分析型图表。
