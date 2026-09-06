# Static Breakout Pool Review Dashboard Spec

状态：当前唯一 Dashboard 心流 / 交互规范  
日期：2026-09-06

本文是静态 GitHub Pages Dashboard 的唯一交互规范。旧 Streamlit / AG Grid 时代的 Dashboard 设计文档已经删除；需要追溯设计演进时使用 Git 历史，不在当前文档目录保留重复规范。

## 1. 事实来源优先级

发生冲突时按以下顺序处理：

1. `quant_trade` 实际调用与 Yfinance_data 权威 Pool 发布契约；
2. `DESIGN_yfinance_data_midweek_review.md` 的数据 / 状态语义；
3. 本文的 Dashboard 心流、交互与公开展示契约。

不得为了视觉或前端实现修改权威 Pool 数据、Midweek projection、Entry Status、Breakout Price Quality 或跨仓库接口。

## 2. 产品目标与主心流

Dashboard 是高频 Review 工作台，不是分析报告。用户应沿同一页面连续完成：

```text
数据状态
→ Period / Scope
→ Change / Origin / Entry Status
→ 必要时 More Filters
→ Results / 排序 / Copy Codes
→ Selected Detail
→ 连续表格 Review
```

核心原则：

- **先知道当前数据语境，再筛选。**
- **快速条件优先，高级筛选按需展开。**
- **表格是主要工作区，详情紧贴结果。**
- **默认排序提供起点，表头排序允许即时探索。**
- **C Rank 只是 Reference，不作为隐藏 Gate。**
- 不新增首页分析型图表，不让辅助信息打断 Review 流。

## 3. 当前技术边界

运行时为纯静态页面：

```text
Authoritative Pool CSV
→ Python projection / normalization
→ dashboard/build_static.py
→ public dashboard.json
→ HTML / CSS / vanilla JS
→ GitHub Pages
```

- Python 是业务事实与投影权威层。
- 浏览器只负责展示、筛选、排序、选行、复制和响应式交互。
- 不重新引入 Streamlit、AG Grid、服务端状态或第二套交易规则。
- `dashboard/services/` 属于跨仓库共享契约，不因前端重构随意移动或改名。

## 4. 页面结构

固定从上到下：

```text
Header / Snapshot / Data State / IBD vs C Rank
Review Queue
  Period
  Scope
  What Changed / Signal Source（仅合法 Midweek comparison）
  Entry Status cards
More Filters
Results summary + Copy Codes
Selected Row Detail
Decision Table
```

### 4.1 Period / Scope

- Midweek 有合法完整周 baseline：默认 `Midweek Review + Changes`。
- Midweek 无合法 baseline：允许查看当前周中 Pool，但关闭 Carry / Change / Origin 比较。
- Weekend：固定完整周语境，Scope 为 `All Signals`。
- 切换 Period 时清理不兼容的临时筛选；同一 Period 内的普通操作不应无故重置其它维度。

### 4.2 快速筛选

Midweek comparison 可用时显示两组：

- `WHAT CHANGED`：Entered/Became Actionable、Left Actionable、Other Changes；
- `SIGNAL SOURCE`：New、Carry、Reconfirmed。

Change 与 Origin 可以组合；Clear 只清除这两组，不清 Status、Period、Scope 或高级 Filters。

### 4.3 Entry Status

顺序固定：

```text
ACTIONABLE → UNCONFIRMED → BELOW_TRIGGER → EXTENDED
```

颜色只能强化语义，文字始终存在。

### 4.4 More Filters

默认收起，显示 `None` 或 `N active`。当前高级条件：

- Setup；
- Vs Buy Point Min / Max；
- Entry Volume Min；
- Weekly Volume Min。

Reset 仅在有高级筛选时出现，并只重置高级筛选。

#### Range 控件语义

Range 控件必须基于**当前 Review 语境中的实际数据范围**，不能使用固定兜底范围冒充数据边界。

当前语境包括：

```text
Period
+ Scope
+ Change
+ Origin
+ Entry Status
+ Setup
```

数值 Range 本身不参与自己的边界计算，避免拖动后把可调范围越缩越小。

具体规则：

- `Vs Buy Point · Min` 默认停在当前数据实际最小值；
- `Vs Buy Point · Max` 默认停在当前数据实际最大值；
- `Entry Volume Min`、`Weekly Volume Min` 默认停在当前数据实际最小值；
- 默认完整范围属于**未启用筛选**，因此 `More Filters · None` 必须保持正确；
- 未启用时标题显示 `Full range`，不使用 `Any` 表示滑块端点；
- Range 下方固定展示当前语境的左右数据边界；
- 用户把 Min 拖离左边界、或 Max 拖离右边界后，该条件才变为 active；
- 当前语境变化后，Range 边界同步更新；旧阈值若已落在新数据范围之外，必须归一化，不能出现 UI 数值与实际过滤状态不一致；
- 缺失值不用于计算数值边界；完整范围状态仍保持“未启用过滤”的原语义。

## 5. Results 与连续 Review

### 5.1 默认排序

- Midweek + Changes + 合法 baseline：`Review Priority`；
- 其它 IBD Review：`C Rank`；
- C Rank Reference：`C Rank` best first。

默认排序是进入队列的起点，不是强制锁定。

### 5.2 表头排序

Decision Table 与 C Rank Reference 的可见字段必须支持点击表头排序：

- 第一次点击：升序；
- 再次点击：降序；
- 当前列显示明确的 `↑ / ↓`；
- 数值列按数值排序；
- Entry Status 按业务状态顺序；
- Breakout Price Quality 按业务质量强度顺序；
- 自定义排序后的选中行、键盘 ↑↓ Review 和 Copy 顺序均跟随当前可见顺序。

### 5.3 Breakout Price Quality 表头说明

该字段必须有独立说明入口，桌面 hover 与触屏点击均可访问。强度从强到弱：

```text
Powerful
Strong
Constructive
Marginal
Weak
```

说明使用倒三角 / 递减宽度与颜色深浅表达强度，并保留文字。语义固定为：

- Powerful：High close + full clearance
- Strong：One strong, one solid
- Constructive：Mixed but valid
- Marginal：Valid, little edge
- Weak：Low close

底部明确：`Price only: Close Position + Trigger Clearance. Volume is separate.`

不得在浏览器重新计算质量等级。

### 5.4 Selected Detail

Selected Detail 位于结果摘要和表格之间；选行后原地更新，不要求用户滚动到页面底部确认。至少覆盖：

- Buy Point / Setup；
- Vs Buy Point / Latest；
- Entry Status；
- C Rank；
- 展开后的 Daily Entry、Pullback、CANSLIM/Base 事实。

详情只解释当前行，不创建第二套筛选器。

## 6. 响应式与滚动

桌面优先保持 Review Queue、筛选入口、Results 与 Selected Detail 在表格前连续出现，减少视觉跳跃。

移动端：

- Period / Scope、Quick filters、Status cards 自动换行；
- 表格允许横向与纵向滚动，但 Code 列保持 sticky；
- 表格自身两个方向到边界时不使用 overscroll / bounce 弹性；页面外层正常纵向滚动不受影响；
- 表头排序和 Quality 说明必须支持触屏；
- 不为移动端复制第二套业务逻辑。

## 7. 数据与公开安全契约

GitHub Pages 是公网资源。

- `dashboard.json` 行数据只能来自 `dashboard.build_static.PUBLIC_DASHBOARD_ROW_FIELDS`；
- Pool 新增字段默认 **不发布**；
- 新字段只有在 UI 明确使用且确认可公开后才加入白名单；
- 禁止账户、持仓、成本、订单、broker account hash、OAuth token、API key、密码或私有研究数据进入 payload；
- 浏览器未显示但收到的数据也视为已公开，因此不能依赖“前端不渲染”作为安全边界。

## 8. 验收

Dashboard 修改至少验证：

```bash
python dashboard/self_check.py \
  --csv us/breakout_follow_pool.csv \
  --midweek-csv us/breakout_follow_pool_midweek.csv
python -m pytest dashboard/tests -q
node --check dashboard/app.js
node --check dashboard/table_enhancements.js
python dashboard/build_static.py --output /tmp/yfinance-dashboard-site
python security_scan.py --history
```

并检查：

- Midweek / Weekend 默认语境正确；
- Changes / All Signals 与 baseline 状态一致；
- Quick filters、Status、More Filters 不互相误重置；
- 默认 Range 显示当前语境真实边界，完整范围不显示 `Any`；
- Range 拖动后 active 状态与结果数量一致；
- 表头排序、Quality tooltip、选行、键盘 ↑↓、Copy 顺序一致；
- 表格横纵滚动到边界不产生自身 bounce；
- C Rank Reference 与 IBD Review 状态互不污染；
- 生成的 `dashboard.json` 不含白名单之外的 Pool 列。

## 9. 文档维护规则

Dashboard 心流、交互或静态展示行为发生变化时只更新本文。

- 不新增第二份 Dashboard UX / flow / UI alignment 规范；
- 过时方案直接删除，Git 历史承担追溯职责；
- 数据 schema 与 Midweek 数据状态设计仍分别由其专门文档维护，不和本文重复。
