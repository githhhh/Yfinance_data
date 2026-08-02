# Breakout Pool Midweek Review UI 对齐规格

状态：方案 1 与书面规格均已批准，实施中

日期：2026-08-02

范围：`Yfinance_data` 现有 Streamlit Dashboard 的展示层与浏览器验收

## 1. 目标

在不改变既有数据投影、跨项目接口和 AG Grid 核心能力的前提下，把现有 Breakout Pool Dashboard 的周中 Review 与本地 HTML 原型、参考截图对齐。最终页面应让用户按以下顺序连续完成复盘：

1. 确认数据快照是否可用；
2. 选择周中或周末数据语境；
3. 选择 Changes 或 All Signals 范围；
4. 通过 Change、Origin、Entry Status 快速收窄队列；
5. 必要时展开高级 Filters；
6. 查看结果、调整排序并复制当前结果 code；
7. 选中一行，理解完整周状态到当前状态的变化；
8. 在 AG Grid 中继续完成列拖动、Pin、排序和键盘 Review。

本规格不追求把原型中的静态样例数据复制到产品中。所有数量、日期、状态、选中行和表格内容必须来自真实运行数据与当前交互状态。

## 2. 事实来源与冲突优先级

发生冲突时按以下顺序决策：

1. `quant_trade` 实际调用代码：跨项目 import、参数、返回结构、异常语义和 ACTIONABLE 消费契约；
2. `doc/DESIGN_yfinance_data_midweek_review.md`：数据边界、投影规则、状态语义和 Dashboard 心流；
3. 本地 HTML 原型与两张截图：布局、视觉层级、尺寸、响应式和交互细节；
4. `Yfinance_data` 现有工程约定：在不违反前三项的情况下保留现有实现方式。

视觉参考文件：

- `doc/BF Midweek Review UI (2026_8_2 09：21：55).html`：Midweek Review；
- `doc/BF Midweek Review UI (2026_8_2 09：25：47).html`：Weekend Full Pool；
- `doc/截屏2026-08-01 下午8.08.39.png`：周中截图；
- `doc/截屏2026-08-01 下午8.08.47.png`：周末截图。

本规格只细化展示层。若视觉参考与数据/集成契约冲突，必须保留真实数据和接口语义，不得用静态 fixture 或修改 `quant_trade` 来迁就视觉稿。

## 3. 已批准的方案

采用“原生 Streamlit 组件重组 + 集中样式契约”的方案：

- 保留 Streamlit 的状态、回调、rerun、可访问按钮和测试方式；
- 保留已有数据服务、筛选、排序、Selected Row 和 AG Grid；
- 重组当前由零散 `st.columns`、`st.button` 和局部 CSS 组成的展示层，使组件分组与原型的视觉语义一致；
- 新增聚焦的 `dashboard/review_styles.py`，集中保存 `REVIEW_UI_CSS`、颜色 token、尺寸和响应式约束；
- `dashboard/app.py` 只保留页面结构、数据绑定和事件处理，避免继续扩张单文件内联 CSS；
- 不引入新的第三方依赖，不新建另一套页面，不以自定义前端应用替换 Streamlit。

拒绝 CSS-only 修补，因为当前差异包含两行快捷栏、额外 Manual 控件、emoji 状态点和错误的响应式 DOM 流，仅覆盖颜色与边距不能稳定修复。也不采用完整自定义前端组件，因为它会增加构建、依赖和双向状态同步复杂度，并给 AG Grid/Streamlit 交互带来不必要风险。

## 4. 改动边界

### 4.1 允许改动

- `dashboard/app.py` 中与页面组合、按钮呈现、Tooltip、复制控件和响应式标记有关的代码；
- 新增 `dashboard/review_styles.py`；
- 与展示层契约相关的 Dashboard 测试；
- 必要的可访问性属性和稳定选择器；
- 现有展示层 CSS 的删除、迁移和收敛。

### 4.2 必须保留

- `yfinance_data` 对 `quant_trade` 暴露的 import 路径、函数签名、返回结构和异常行为；
- current_pool 左表投影、周中/周末日期判定和 ACTIONABLE 结果；
- 所有动态计数和真实字段值；
- AG Grid 列拖动、Pin、排序、键盘选行和 Selected Row 联动；
- Filters 在两种模式下默认收起，真正切换模式时恢复收起；
- Clear、All Signals、重复点击当前 Mode、默认排序等既定状态语义；
- 源 CSV 只读边界。

### 4.3 非目标

- 不修改 `quant_trade`；
- 不修改 `us/`、`results_pkl/` 或任何源数据；
- 不写回投影结果；
- 不重做数据分析服务；
- 不复制原型中的 23、106、745 等静态数字；
- 不新增构建链或前端包管理器；
- 不创建 `docs/superpowers/`、`dashboard/artifacts/` 或新的独立 Dashboard。

## 5. 视觉系统

页面使用以下 token。实现允许通过 CSS 变量复用，但最终浏览器计算值必须与这里一致：

| Token | 值 | 用途 |
| --- | --- | --- |
| `--bg` | `#0c1016` | 页面背景 |
| `--panel` | `#151b23` | 卡片和 Selected Row |
| `--panel-soft` | `#111720` | 分段控件和辅助容器 |
| `--input` | `#202b3a` | 输入面 |
| `--line` | `#35404d` | 主边框 |
| `--line-soft` | `#2b3440` | 次级分隔线 |
| `--text` | `#f4f5f7` | 主文字 |
| `--muted` | `#9ca8b7` | 次级文字 |
| `--green` | `#35df65` | ACTIONABLE |
| `--cyan` | `#1fcdb4` | 选中/主强调 |
| `--blue` | `#2791ff` | EXTENDED/链接强调 |
| `--yellow` | `#ffd21f` | UNCONFIRMED |
| `--red` | `#f04444` | BELOW TRIGGER/错误 |

字体栈为 `Arial Narrow, Roboto Condensed, Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif`，正文基准为 `14px / 500`。所有计数使用 tabular numerals。按钮与信息图标必须有清晰的 hover、focus-visible、selected 和 disabled 状态。

颜色只能作为第二重表达；状态名称、筛选标签、选中对勾和禁用语义必须同时通过文字或控件状态表达。

## 6. 桌面布局契约

目标桌面视口为 1365px 宽。页面不要求对截图做像素级图像复制，但关键几何、层级和信息密度必须与原型一致。

| 区域 | 几何契约 |
| --- | --- |
| 页面壳 | `padding: 29px 28px 34px` |
| Header | `min-height: 78px`，底部分隔线，标题与右侧导航同一行 |
| 标题 | `29px / 800`，Data Ready 紧邻标题 |
| Snapshot | 上边距 `23px`；日期槽 `139px`，上下文槽 `211px` |
| 顶部导航 | Info `45×45px`；IBD Review/C Rank Reference 高 `45px` |
| Review section | 顶部内边距 `9px` |
| Section heading | `min-height: 50px` |
| Queue actions | 两列 `276px 268px`，列间距 `9px` |
| Mode switch | `276×40px`，两段式统一外壳 |
| Scope switch | `268×40px`，两段式统一外壳 |
| Context slot | 高 `48px`，外边距 `6px 0 8px` |
| Quick chips | 单行，高 `31px`；Change 与 Origin 之间有竖分隔线 |
| Subsection label | `10px`，位于 Status 卡片之前 |
| Status cards | 四列，间距 `16px`，每卡高 `70px` |
| Status orb | CSS 圆球 `19×19px`，带内高光和外发光，不用 emoji |
| Filters toggle | 高 `45px` |
| Results toolbar | `min-height: 56px` |
| Copy | `min-width: 180px; height: 44px` |
| Sort | `min-width: 120px; height: 44px` |
| Selected Row | 高 `60px`，列为 `194px repeat(4, minmax(0, 1fr))` |
| Grid | 紧接 Selected Row，下方保留现有 AG Grid 高度和能力 |

Mode、Scope、Status、Filters 顶边和 Results 顶边在 Midweek 与 Weekend 切换前后，目标桌面宽度下 `getBoundingClientRect()` 的同名基准位置差值不得超过 1px。

## 7. 组件设计

### 7.1 Header

- 左侧显示 Breakout Pool、Data Ready 和动态 Snapshot 摘要；
- Snapshot 的固定文字槽避免日期、陈旧天数和计数变化导致右侧内容跳动；
- 右侧保留 Info、IBD Review、C Rank Reference；
- 当前页面必须同时用选中样式和可访问状态表达，不能只依赖颜色。

### 7.2 Mode 与 Scope

- Mode 和 Scope 分别呈现为两个真正的视觉分段组，而不是四个互不相干的按钮；
- 底层仍使用原生 Streamlit button 与现有 callback/rerun；
- 每一段预留固定对勾槽，选中前后按钮宽度和文字位置不变；
- Midweek 默认 Scope 为 Changes；Weekend 默认 Scope 为 All Signals；
- Weekend 保留完整 Scope 外壳，Changes 在原位置禁用，All Signals 的位置和宽度不变；
- 重复点击当前 Mode 不重置任何用户状态；
- 真正切换 Mode 才重置不兼容筛选、Copy 反馈、Filters 展开状态和默认排序；
- Midweek 的显示值与实际排序均为 Review Priority；Weekend 均为 C Rank。

### 7.3 Review Context

Context slot 在两种模式中始终占同一高度和纵向位置。

Midweek 使用一条连续单行快捷栏，顺序固定为：

`CHANGE` → Became Actionable → Left Actionable → Other Changes → 分隔线 → `ORIGIN` → New → Carry → Reconfirmed → Clear。

每个 chip 包含：

- 7px 语义色点；
- 文本标签；
- 动态数量；
- 独立的 16px 信息触发器。

点击 chip 切换对应 Change 或 Origin 值；同一维度单选，不同维度可组合。Clear 只清除 Change 与 Origin，不清除 Status 或高级 Filters。

Weekend 在完全相同的 slot 中显示 Weekend Baseline 说明条，内容说明当前是完整周 Pool，未应用周中比较。Weekend 不显示或暗示 Change/Origin 数据。

### 7.4 Status Queue

固定显示 ACTIONABLE、UNCONFIRMED、BELOW TRIGGER、EXTENDED 四张卡。每卡显示 CSS orb、状态名称、动态数量和状态说明。卡片点击只切换 Status，且可与 Change/Origin 组合。

状态卡不得把 emoji 写入按钮标签。选中标记使用固定槽，不能改变卡片内部布局。Status 对应的 entry-volume 和 near-trigger 状态清理继续沿用现有业务规则。

### 7.5 Tooltip

以下 10 个心流入口必须有统一 Tooltip：

- Became Actionable、Left Actionable、Other Changes；
- New、Carry、Reconfirmed；
- ACTIONABLE、UNCONFIRMED、BELOW TRIGGER、EXTENDED。

每个 Tooltip 必须包含定义、数字统计口径和点击后的筛选效果。交互必须满足：

- 鼠标 hover；
- 键盘 focus；
- 独立触屏信息按钮；
- 触发器与浮层通过 `aria-describedby` 关联；
- Esc 关闭；
- 点击信息按钮不得触发对应筛选；
- 不使用浏览器原生 `title` 作为 Tooltip；
- 桌面浮层避免越过视口边界；760px 及以下呈现为底部浮层。

优先保留 Streamlit 原生按钮的键盘与点击语义，使用稳定的外层 key/class 和独立信息触发器实现视觉分组与 Tooltip。不得以不可交互的 Markdown 假按钮替代筛选按钮。

### 7.6 Filters

- 周中与周末初次进入都默认收起；
- 真正切换 Mode 后恢复收起；
- 展开按钮始终为整行 45px 控件，显示 Filters、当前活动筛选数量和 chevron；
- 现有高级筛选字段和业务约束不变；
- 展开/收起不能影响 Mode、Scope、Context 或 Status 的几何稳定性。

### 7.7 Results、Copy 与 Sort

- Results 左侧显示当前过滤结果数量，右侧只保留 Copy Codes 与 Sort；
- 移除始终可见的 Manual 控件；
- Copy Codes 复制当前过滤结果、按当前结果顺序生成的 code 列表；
- Clipboard API 失败时继续尝试 `execCommand('copy')` 兼容路径；
- 成功时按钮短暂显示 `✓ Copied (n)`，失败时显示明确失败反馈，随后恢复；
- Copy 反馈不新增一个与 Sort 并列的永久主控件；
- 空结果时按钮必须呈现明确且无误导的禁用/零数量状态；
- Sort 控件的选中值必须与实际 DataFrame 排序一致。

### 7.8 Selected Row 与 AG Grid

- Selected Row 固定为 60px，显示 code、完整周状态、当前状态、变化信息和关键价格/距离；
- 未选中行时保留同高占位，避免 Grid 上跳；
- 周中显示完整周到当前的变化；周末不伪造 Change/Origin；
- 保留表格的动态列、列拖动、Pin、排序、键盘选行与 Selected Row 联动；
- 周中主表显示必要的 Origin/Change 信息，周末表格不混用周中字段；
- 不用静态 HTML table 替换 AG Grid。

## 8. 状态与数据流

页面保持单一 Review state，由现有状态函数管理。展示层只读取 state 和分析结果，并通过既有事件函数更新 state。

```text
CSV / 投影服务
  → PoolAnalysisResult
  → Review state（mode、scope、quick filters、status、advanced filters、sort）
  → 当前结果 DataFrame 与动态计数
  → Mode/Scope/Context/Status/Results/Selected Row/AG Grid
```

不允许某个视觉组件自己重新计算另一套业务计数。所有 chip、Status、Results 和 Copy 数量都必须来源于同一过滤模型。Selected Row 必须来源于当前模式下的实际表格选中 code。

切换规则：

- Midweek → Weekend：Scope 设为 All Signals；清除 Change/Origin；清理只适用于周中的字段；Filters 收起；排序设为 C Rank；清理 Copy 反馈；
- Weekend → Midweek：Scope 设为 Changes；Filters 收起；排序设为 Review Priority；清理 Copy 反馈；
- 点击当前 Mode：不执行上述重置；
- All Signals：执行既定范围重置，不隐式修改 Mode；
- Clear：只清除 Change/Origin；
- Status：与 Change/Origin 和允许的高级筛选组合。

## 9. 响应式与动效

响应式断点使用原型契约：

### 1120px 及以下

- 页面壳缩为 `22px 18px 28px`；
- Status 改为两列；
- Selected Row 允许水平滚动，并给摘要列保留可读的最小宽度；
- Tooltip 调整边缘定位，避免越界。

### 760px 及以下

- Header、Section heading、Results toolbar 纵向排列；
- Queue actions 仍为两列且占满宽度；
- Context slot 允许水平滚动，快捷栏本身保持单行，chip 内部不能因选中状态重排；
- Status 改为一列；
- Copy 与 Sort 占满结果区宽度；
- Selected Row 水平滚动；
- Tooltip 变为固定底部浮层。

### 480px 及以下

- 页面壳为 `18px 12px`；
- 顶部导航占满宽度；
- Queue actions 改为一列，但每个分段组内部仍保持两段同高；
- 标题可缩至 25px。

当系统启用 `prefers-reduced-motion: reduce` 时，动画和 transition 必须近乎即时，且不得依赖动画表达状态变化。

## 10. 错误与降级

- 没有有效 midweek 数据时，Midweek Mode 原位禁用，Weekend 仍可工作；
- 数据错误沿用现有明确错误展示和失败语义，不用空的视觉稿掩盖；
- 没有 baseline 时不得显示错误 Carry；展示层只呈现投影服务给出的事实；
- Copy 失败必须显示可感知反馈，不能静默成功；
- Tooltip 或装饰样式失效时，核心按钮文字、计数和筛选仍可使用；
- CSS 装饰不得成为数据正确性或状态切换的依赖。

## 11. 实施顺序与测试策略

实施采用 TDD，并按以下边界推进：

1. 先写展示层契约失败测试；
2. 提取 `review_styles.py` 并建立 token/尺寸/断点契约；
3. 重组 Mode/Scope 和固定 Context slot；
4. 把 Quick filters 改为单行并统一 10 个 Tooltip；
5. 把 Status 改为无 emoji 的 CSS orb 卡片；
6. 对齐 Filters、Results、Copy/Sort 和 Selected Row；
7. 完成响应式、focus-visible、reduced-motion；
8. 运行完整自动化检查；
9. 在真实 Chrome 中完成桌面与窄屏验收。

### 11.1 静态契约测试

- CSS token、桌面尺寸、1120/760/480 断点存在且一致；
- Context 实现只有一个横向流程，不再按 CHANGE/ORIGIN 循环生成两行；
- Status 按钮标签不包含 emoji；
- Results 主控件不存在永久 Manual；
- Weekend Baseline 与 Midweek Context 使用同一 slot；
- `prefers-reduced-motion` 和 focus-visible 规则存在。

### 11.2 Streamlit/AppTest 行为测试

- Midweek/Weekend 默认 Scope 和默认排序正确；
- 重复点击当前 Mode 不重置状态；
- 真正切换 Mode 清理不兼容状态并收起 Filters；
- Weekend Changes 原位禁用；
- Change、Origin、Status 可以组合，数量与结果一致；
- Clear 不清除 Status；
- All Signals 执行明确范围重置；
- 10 个 Tooltip 的说明元数据完整；
- 周末不显示或排序周中字段；
- Copy 使用当前过滤结果。

### 11.3 数据与集成回归

继续运行既有接口兼容、投影和失败分支测试，证明展示层改造未破坏：

- `quant_trade` 使用的 import 路径、参数和返回结构；
- current_pool 左表规则；
- NEW、CARRY、RECONFIRMED 和 Effective Entry Status；
- 过期 midweek、缺失 baseline、无效价格/candidate、重复 code；
- Futu ACTIONABLE code 与统一投影结果。

### 11.4 真实 Chrome 验收

真实 Chrome 是最终 UI/交互验收环境，不能只以源码测试或 Streamlit AppTest 代替。验收矩阵包括：

- 1365px 桌面：分别截图 Midweek 与 Weekend，与本地 HTML/参考截图并排对照；
- 1120px、760px、480px：检查布局断点、横向滚动和不可重排约束；
- 通过 `getBoundingClientRect()` 记录 Mode、Scope、Status、Filters、Results 的位置，验证周中/周末对应顶边差值不超过 1px；
- 验证 Mode、Scope、六个 quick chip、Clear、四个 Status、Filters、Sort、Copy、表格选行；
- 验证 10 个 Tooltip 的 hover、focus、触屏信息按钮、Esc 和 `aria-describedby`；
- 验证 Weekend Changes disabled、All Signals 稳定位置、C Rank 实际排序；
- 验证 Midweek Review Priority 实际排序；
- 验证选中态、禁用态、focus-visible 和 reduced-motion。

若 Chrome 因权限未连接，自动化测试可继续推进，但不得据此宣称 UI 验收完成。最终完成声明必须附上真实 Chrome 的截图、几何测量和交互结果。

## 12. 完成判定

只有同时满足以下条件，UI 对齐工作才算完成：

- 当前 Dashboard 的结构、尺寸、层级和响应式符合本规格；
- 两种模式的数据、筛选、排序、复制和 Selected Row 行为符合既定业务契约；
- 10 个 Tooltip 满足统一内容与可访问交互要求；
- AG Grid 能力无回退；
- Dashboard 测试与 `dashboard/self_check.py` 在 Conda `quant_env` 中通过；
- 真实 Chrome 完成桌面、窄屏、交互和 `getBoundingClientRect()` 验收；
- 未修改 `quant_trade`、`us/`、`results_pkl/` 或源 CSV；
- 所有计数来自真实数据，不硬编码原型 fixture。
