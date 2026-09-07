# Static Breakout Pool Review Dashboard Spec

状态：当前唯一 Dashboard 心流 / 交互规范  
日期：2026-09-07

本文是静态 GitHub Pages Dashboard 的唯一交互规范。旧方案需要追溯时使用 Git 历史，不保留平行 UX 文档。

## 1. 产品目标与主心流

Dashboard 是高频 Review 工作台，不是分析报告：

```text
数据状态
→ Period / Scope
→ Change / Origin / Entry Status
→ 必要时 More Filters
→ Results / 排序 / Copy Codes
→ Selected Detail
→ 连续表格 Review
```

原则：

- 先确认当前数据语境，再筛选；
- 快速条件优先，高级筛选按需展开；
- 表格是主要工作区，详情紧贴结果；
- 默认排序只提供中性的 Review 起点，不引入未经验证的质量排名；
- 外部参考数据只能提供 context，不能成为隐藏 Gate；
- 不新增首页分析型图表，不让辅助信息打断 Review 流。

## 2. 技术与事实边界

```text
Authoritative Pool CSV
→ Python projection / normalization
→ optional current external reference enrichment
→ dashboard/build_static.py
→ public dashboard.json
→ HTML / CSS / vanilla JS
→ GitHub Pages
```

- Python / Pool 是交易事实与 Midweek projection 权威层；
- 浏览器只负责展示、筛选、排序、选行、复制和响应式交互；
- 不重新引入 Streamlit、AG Grid、服务端状态或第二套交易规则；
- `dashboard/services/` 是共享契约，不为纯 UI 改动随意破坏；
- 外部参考源失效时必须 fail-soft，不阻塞 Pool 或 Dashboard 发布；
- 不得为了视觉或前端实现修改权威 Pool、Entry Status、Breakout Price Quality 或跨仓库交易事实契约。

## 3. 页面结构

固定从上到下：

```text
Header / Snapshot / Data State
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

### 3.1 Period / Scope

- Midweek 有合法完整周 baseline：默认 `Midweek Review + Changes`；
- Midweek 无合法 baseline：允许查看当前周中 Pool，但关闭 Carry / Change / Origin 比较；
- Weekend：完整周语境，Scope 为 `All Signals`；
- 切换 Period 时清理不兼容的临时筛选。

### 3.2 快速筛选

Midweek comparison 可用时显示：

- `WHAT CHANGED`：Entered/Became Actionable、Left Actionable、Other Changes；
- `SIGNAL SOURCE`：New、Carry、Reconfirmed。

Change 与 Origin 可组合；Clear 只清除这两组。

### 3.3 Entry Status

顺序固定：

```text
ACTIONABLE → UNCONFIRMED → BELOW_TRIGGER → EXTENDED
```

颜色只强化语义，文字始终存在。

## 4. More Filters

默认收起，显示 `None` 或 `N active`：

- Setup；
- Vs Buy Point Min / Max；
- Entry Volume Min；
- Weekly Volume Min。

Reset 仅在有高级筛选时出现，并只重置高级筛选。

### Range 控件语义

Range 必须基于当前 Review 语境中的实际数据范围：

```text
Period + Scope + Change + Origin + Entry Status + Setup
```

规则：

- `Vs Buy Point · Min` 默认实际最小值；
- `Vs Buy Point · Max` 默认实际最大值；
- Entry / Weekly Volume Min 默认实际最小值；
- 完整范围属于未启用筛选，标题显示 `Full range`；
- 不使用 `Any` 冒充滑块端点；
- 数值 Range 不参与自己的边界计算；
- 语境变化后边界同步更新，旧阈值超出新范围时归一化；
- 缺失值不参与边界计算。

## 5. Results 与排序

### 5.1 默认排序

- Midweek + Changes + 合法 baseline：`Review Priority`；
- 其它 Review：按 `Code`，保持中性、可预测；
- RS 不作为默认排序、Gate、Top3 评分或 Review Priority 的组成部分。

### 5.2 表头排序

可见字段支持点击表头排序：

- 第一次点击升序，再次点击降序；
- 数值列按数值；
- Entry Status 按业务状态顺序；
- Breakout Price Quality 按质量强度顺序；
- RS 可按 percentile 手工排序，`N/A` 无论升降序都排最后；
- 自定义排序后的选中行、键盘 ↑↓ Review 与 Copy 顺序必须跟随当前可见顺序。

### 5.3 Breakout Price Quality

表头说明桌面 hover / 触屏点击都可访问：

```text
Powerful → Strong → Constructive → Marginal → Weak
```

语义：

- Powerful：High close + full clearance
- Strong：One strong, one solid
- Constructive：Mixed but valid
- Marginal：Valid, little edge
- Weak：Low close

底部固定：`Price only: Close Position + Trigger Clearance. Volume is separate.`

浏览器不得重新计算质量等级。

## 6. RS Reference

RS 是**当下横向强弱参考**，不是策略评分。

数据源：`Fred6725/rs-log` 的公开 `output/rs_stocks.csv`。该项目使用透明的 IBD-style RS 实现，但不是官方 IBD RS。

公开字段：

```text
rs_percentile
rs_1m_percentile
rs_3m_percentile
rs_6m_percentile
```

使用规则：

1. 构建时先读取 `rs_stocks.csv` 的最新 commit metadata；
2. 将该 commit 时间转换为 `America/New_York` 日期，作为 RS market date；
3. 使用同一个 commit SHA 固定读取对应版本的 `rs_stocks.csv`，禁止再从浮动 `main` 读取，避免 metadata / CSV 更新竞态；
4. Pool 的 `snapshot_date` 是最新实际市场数据日期；
5. 只有 `rs_market_date == pool.snapshot_date` 时才 join ticker；
6. 日期不一致、ticker 缺失、GitHub / rs-log 不可用、CSV schema 异常时显示 `N/A`；
7. RS 失败不得阻塞 Dashboard 构建或 Pool 发布；
8. 不要求本仓库保存 RS PIT 历史；源未来停止更新时继续显示 `N/A` 即可；
9. RS 获取只访问公开 GitHub API / raw 内容，不使用仓库 Token、API Key 或其它凭据。

主表直接原生显示当前 `RS` percentile；hover / focus / 触屏点击以及 Selected Detail 可查看当前、1M、3M、6M ago percentile、market date 与来源。不得通过额外字段替换层改变 Pool 事实。

禁止：

- 把 RS 当成官方 IBD RS；
- 用 RS 替代 Entry Status / Breakout Price Quality；
- 因 RS 高而自动提升 Top3；
- 因 RS 低 / N/A 而自动淘汰；
- 用当前 rs-log 数据倒推历史回测结论。

## 7. Selected Detail

Selected Detail 位于结果摘要和表格之间。至少覆盖：

- Buy Point / Setup；
- Vs Buy Point / Latest；
- Entry Status；
- RS Reference；
- 展开后的 Daily Entry、Pullback、CANSLIM/Base 事实。

详情只解释当前行，不创建第二套筛选器。

## 8. 响应式与滚动

- 移动端 Period / Scope、Quick filters、Status cards 自动换行；
- 表格允许横向与纵向滚动，Code 列保持 sticky；
- 表格自身两个方向到边界时关闭 overscroll / bounce；
- 页面外层正常纵向滚动不受影响；
- 表头排序、Quality / RS 说明必须支持触屏。

## 9. 公开安全契约

GitHub Pages 是公网资源：

- `dashboard.json` 行数据只能来自 `PUBLIC_DASHBOARD_ROW_FIELDS`；
- Pool 新增字段默认不发布；
- 禁止账户、持仓、成本、订单、broker account hash、OAuth token、API key、密码或私有研究数据进入 payload；
- 浏览器未显示但收到的数据同样视为公开；
- RS 公共数据获取不接收或转发仓库 Token、API Key 或其它凭据。

## 10. 验收

至少验证：

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
- Range 默认显示当前语境真实边界；
- Midweek Changes 仍按 Review Priority，其余默认 Code；
- RS 只在严格交易日匹配时显示，错日、ticker 缺失与外部失败为 N/A；
- RS 直接由主 UI 实现，不存在额外字段替换 patch 层；
- 表头排序、Quality tooltip、选行、键盘 ↑↓、Copy 顺序一致；
- 表格横纵滚动到边界不产生自身 bounce；
- 生成的 `dashboard.json` 不含白名单之外的 Pool 列。

## 11. 文档维护

Dashboard 心流、交互或静态展示行为变化时只更新本文。数据 schema 与 Midweek 数据状态仍由各自专门文档维护。
