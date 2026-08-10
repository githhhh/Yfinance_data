---
name: ibd-candidate-prescreen
description: 从 Dashboard 突破候选池中，基于当前已导出字段执行标准化、IBD-aligned 的候选预筛，结合大盘报告、买点新鲜度、突破日量价几何、当前阶段结构与辅助基本面信息，精选最多 3 只优先人工复核标的。用户请求“预筛标的”“IBD 分析”“突破池分析”“review 候选池”或周中/完整周候选复盘时使用。本 Skill 不是完整 CAN SLIM Review，也不替代图表人工判断。
---

# IBD Candidate Pre-screen

## 任务定位

- 基于候选池现有正式字段执行 IBD-aligned 预筛，目标是缩小人工 Review 范围，不是完成完整 CAN SLIM 认证。
- 输出 0～3 只“优先复核”；可另列最多 2 只“值得留意”和最多 3 只“暂不优先”作为上下文；宁缺毋滥。
- 规则优先。经验只用于正式规则仍完全接近的候选，不得覆盖 Critical、字段路由、Geometry、状态或排序。
- 不预测走势或目标价，不给仓位与买卖指令，不修改原始数据，不用未提供字段推导新指标。
- 所有计算在内存中完成；不得在项目目录生成或遗留临时脚本、中间文件或数据副本。

## 输入与前置步骤

1. 运行 `git submodule update --remote market_analysis`。若父仓库的 submodule 指针变化，在交付中说明。
2. 大盘唯一来源是 `market_analysis/output/market_report.json`。直接继承市场状态、派发日与板块信息，不重新判断趋势，也不因 Correction 自动淘汰合格候选。更新失败、文件缺失或 JSON 无效时停止，不改用其他信源。
3. 调用 `dashboard.data_utils.get_latest_pool_csv_path()`，根据文件内容中的 `snapshot_date` 比较 `us/breakout_follow_pool.csv` 与 `us/breakout_follow_pool_midweek.csv`；优先用 `dashboard.data_utils.load_pool_csv()` 加载最新文件。
4. CSV 必须用支持引号、转义和引号内换行的标准解析器。`ibd_candidate_extra` 含带逗号 JSON，禁止手工按逗号切分。任一数据行列数与 Header 不一致时停止并报告“候选池解析失败”，不得静默跳过。
5. 核心 Header 为 `code`、`signal`、`ibd_candidate_rule`、`ibd_entry_status`、`current_vs_ibd_candidate_pct`、`ibd_entry_volume_ratio`、`ibd_entry_close_vs_trigger_pct`、`ibd_entry_close_position`、`ibd_entry_breakout_range_ratio`、`sector`；缺少任一项时停止。其他字段缺失按下文状态处理。
6. 字符串去除首尾空白，`ibd_entry_status` 转大写。`signal`、`pullback_v_is_dry` 接受布尔值、大小写不敏感的 `true/false` 或 `1/0`。无法识别的 `signal` 作为数据错误转人工补数；无法识别的 `pullback_v_is_dry` 记 UNKNOWN。
7. 数值为空、非数字、NaN 或正负 Infinity 时视为缺失，不参与运算或排序。所有判定使用未四舍五入原值，只在最终报告中格式化。

## 状态语义

| 状态 | 含义 |
|---|---|
| PASS | 检查适用、字段有效、存在正式标准且满足 |
| FAIL | 检查适用、字段有效、存在正式标准但不满足 |
| CONTEXT | 路由字段有效但无正式阈值，只展示客观数值 |
| UNKNOWN | 当前技术/结构评估所需字段缺失或不可靠 |
| INFO_MISSING | 非决定性辅助字段缺失，只提示人工补数，不扣分、不参与分组或排序 |
| N/A | 检查不适用 |

- CONTEXT、INFO_MISSING、N/A 不计入 Major FAIL 或 Major UNKNOWN。
- Critical UNKNOWN 不参与报告排序，转人工补数。
- EPS 缺失是 INFO_MISSING，不得写成基本面差、不得将候选降入“值得留意”。

## 候选范围与展示分散

- 候选集合仅包含 `signal == True` 且 `ibd_candidate_rule` 非空的行。
- 非 ACTIONABLE 不进入“优先复核”。
- 板块占比按候选集合中的 ACTIONABLE 行计算：`该 sector 的 ACTIONABLE 数 / 全部 ACTIONABLE 数`；空 `sector` 记为 UNKNOWN。
- `sector` 上限只是 Top 3 报告的分散展示约束，不是 IBD 质量评分：同一 sector 最多 2 只。按排序逐只接收，触及上限时跳过并继续考察下一只；若确实跳过原本更高名次，报告必须说明。
- 若 sector 上限没有造成候选被跳过，不得在报告中展示 sector 数量、占比或“达到上限”等无效背景。
- 不再因某 sector 占比超过 50% 将上限降为 1。行业聚集可能是领导线索；不得把宽泛 sector 占比直接解释为行业强弱。可展示 `industry` 作为背景，但不自行生成行业排名。

## #4 / #5 阶段字段路由

这里区分当前候选生命周期阶段，只约束 Checklist #4 Depth 与 #5 Duration，不约束 #6 缩量。

| `ibd_candidate_rule` | 当前阶段 | #4 / #5 唯一字段 |
|---|---|---|
| `ceiling`、`ceiling_breakout` | 初始突破基底 | `base_depth_pct` / `base_duration_weeks` |
| `ceiling_pullback` 及其他非空值，如 `pivot`、`ma10_touch_confirm`、`three_weeks_tight` | 当前回踩/巩固 | `pullback_pct` / `pullback_duration_weeks` |
| 空或缺失 | UNKNOWN | 不读取另一组字段 |

- 严禁跨阶段替代、计算替代值或因字段缺失改读另一组字段。
- 路由字段有效时 #4/#5 记 CONTEXT；缺失记 UNKNOWN。
- `ceiling_pullback` 及其他 continuation 规则可额外展示 `base_depth_pct` / `base_duration_weeks` 作为母基底背景，但不替代当前 #4/#5，也不参与 Major 状态或排序。
- 不得把深度、时长或信号名解释成额外形态分类，不得创造“健康”“过深”“偏长”“太短”等未定义结论。

## 核心数值语义与展示规范

| 字段 | 计算语义与原始单位 | 强制展示 |
|---|---|---|
| `ibd_entry_close_position` | `(Close-Low)/(High-Low)`，0～1 比例 | `0.9414` → `收盘位置94.1%` |
| `ibd_entry_breakout_range_ratio` | `(Close-Trigger)/(High-Low)`，日振幅倍数 | `1.1873` → `穿透1.19×日振幅`；正文默认不展示 |
| `trigger_pos` | `(Trigger-Low)/(High-Low) = pos-rr` | 仅内部 Geometry 计算 |
| `ibd_entry_close_vs_trigger_pct` | `(Close-Trigger)/Trigger`，小数比例 | `0.0468` → `高于触发位4.68%` |
| `ibd_entry_volume_ratio` | 突破日成交量/基准均量 | `1.7459` → `量能1.75×均量` |
| `current_vs_ibd_candidate_pct` | 当前价相对候选买点，已是百分数 | `2.70` → `当前高于买点2.70%` |
| `base_depth_pct` | 基底回撤，已是带符号百分数 | `-20.2` → `基底深度20.2%` |
| `pullback_pct` | 当前回踩，已是带符号百分数 | `-13.8` → `回踩幅度13.8%` |
| `dist_to_52w_high_pct` | 当前价距 52 周高点，已是百分数 | `-2.24` → `低于52周高点2.24%` |
| `eps_yoy_growth` | EPS 同比，已是百分数 | `31.3` → `EPS同比+31.3%` |
| `volume_ratio` | 周线成交量/周均量 | `1.70` → `周线量能1.70×均量` |
| `*_duration_weeks` | 周数 | `6.0` → `6周`；非整数保留 1 位小数 |

强制规则：

- `ibd_entry_breakout_range_ratio` 不是百分比，可以大于 1。不得写成不带单位的“突破幅度1.19”，也不得解释成 19% 或 119%。必须展示时写“穿透1.19×日振幅”。
- 面向用户的“高于触发位多少”只使用 `ibd_entry_close_vs_trigger_pct`，按百分比格式化；它不参与 Geometry。
- `ibd_entry_close_position` 按百分比展示，保留 1 位小数；不得直接输出 `0.94`。
- 比例倍数统一保留 2 位小数并写 `×均量` 或 `×日振幅`；百分数通常保留 1～2 位小数。
- 使用方向词代替含混负号：写“回踩幅度13.8%”“低于52周高点2.24%”，不写“回踩-13.8%”“距高点-2.24%”。
- 当前价高于买点写“当前高于买点 x%”；低于时写“当前低于买点 x%”。
- 缺失值写“数据缺失”或省略对应短语，绝不显示为 0。
- `pos` 有限但不在 `[0,1]` 时视为数据错误并记 Critical UNKNOWN；`rr > 1` 可以合法表示 Trigger 低于当日 Low，不得截断。

## Breakout Geometry

只使用：

```text
pos = ibd_entry_close_position
rr = ibd_entry_breakout_range_ratio
trigger_pos = pos - rr
```

严格按顺序首次命中：

| 顺序 | 条件 | 分类 | 结果 |
|---:|---|---|---|
| 1 | `rr <= 0` | Defensive Failure | Critical FAIL |
| 2 | `pos < 0.65` | Squat / Upper Shadow | Critical FAIL |
| 3 | `trigger_pos <= 0` 且 `pos >= 0.80` | Full-range Breakout | PASS |
| 4 | `trigger_pos <= 0` 且 `0.65 <= pos < 0.80` | Faded Gap | PASS |
| 5 | `trigger_pos > 0` 且 `pos >= 0.80` 且 `rr >= 0.50` | Strong Finish | PASS |
| 6 | `trigger_pos > 0` 且 `pos >= 0.80` 且 `rr < 0.50` | Constructive Breakout | PASS |
| 7 | `trigger_pos > 0` 且 `0.65 <= pos < 0.80` 且 `rr >= 0.50` | Constructive Breakout | PASS |
| 8 | `trigger_pos > 0` 且 `0.65 <= pos < 0.80` 且 `rr < 0.50` | Marginal Breakout | PASS |

- Geometry 层级：`Full-range Breakout > Strong Finish > Faded Gap > Constructive Breakout > Marginal Breakout`。
- `rr <= 0` 或 `pos < 0.65` 一旦由有效数值确认，即使另一字段缺失仍为 Critical FAIL。其余 `pos` / `rr` 缺失为 Critical UNKNOWN。
- 报告只使用上述英文分类。不得仅凭 `rr` 或主观观感升级分类。
- `rr` 极低但大于 0 时仍按正式 Geometry 分类；若该值影响人工理解，可客观说明“仅小幅越过触发位”，不得另创淘汰阈值。

## IBD-aligned 分层检查表

| 级别 | # | 检查点 | 唯一标准 |
|---|---:|---|---|
| Critical | 1 | 买点新鲜度 | 仅用 `current_vs_ibd_candidate_pct`；`0%～5%` PASS，`>5%` 或 `<0%` FAIL，`<=2%` 为排序新鲜区 |
| Critical | 2 | 突破日放量 | `ibd_entry_volume_ratio >= 1.5` |
| Critical | 3 | 突破日质量 | Geometry 非 Defensive Failure、非 Squat / Upper Shadow |
| Major | 4 | 阶段深度 | 严格按路由；有效为 CONTEXT，缺失为 UNKNOWN |
| Major | 5 | 阶段时长 | 严格按路由；有效为 CONTEXT，缺失为 UNKNOWN |
| Major | 6 | 巩固期缩量 | `pullback_v_is_dry == True` PASS，`False` FAIL；`ceiling` / `ceiling_breakout` 为 N/A；其他规则适用，缺失或窗口错位为 UNKNOWN |
| Minor | 7 | 紧贴 52 周高点 | `dist_to_52w_high_pct > -5.0` |
| Major | 8 | 基本面辅助证据 | 有效值 `>=25` PASS，`<25` FAIL；缺失为 INFO_MISSING，不扣分、不影响分组或排序 |
| Minor | 9 | 净筹码吸纳 | 仅使用上游正式字段；当前未导出时 N/A，不自行推导 |
| Minor | 10 | 周线量能跟进 | `volume_ratio >= 1.3` 时仅作正向加分；否则省略 |

补充约束：

- Critical 任一 FAIL 即淘汰出“优先复核”和“值得留意”；只有 Critical 全部明确 PASS 才能进入前两组。
- 字段缺失不得写成 FAIL，INFO_MISSING 不得写成负面证据。
- EPS 已知低于 25% 只是一项 Major 顾虑，单独不得决定淘汰；只有与另一项明确 Major FAIL 同时存在时，才可能触发“Major FAIL 至少 2 项”。
- #6 只评价当前候选对应的回踩/巩固段，不评价确认突破日放量；明显错位的旧窗口记 UNKNOWN。
- Minor 不作淘汰条件；周线量能低于 1.3 或缺失不得成为顾虑。
- `ibd_entry_close_vs_trigger_pct` 只作突破日可读上下文，不参与 Geometry，也不得替代当前买点新鲜度。

## 分组与排序

每行按以下顺序首次命中并停止：

1. **人工补数**：`signal` 无法识别；排除在候选集合和报告排序之外。
2. **暂不优先**：候选存在任一 Critical FAIL；已知失败优先于同时存在的 UNKNOWN。
3. **人工补数**：无 Critical FAIL，但存在 Critical UNKNOWN。
4. **暂不优先**：Critical 全部 PASS，但不是 ACTIONABLE。
5. **暂不优先**：ACTIONABLE、Critical 全部 PASS，但明确 Major FAIL 至少 2 项。
6. **值得留意池**：ACTIONABLE、Critical 全部 PASS、Major FAIL 不超过 1，但存在真正的 Major UNKNOWN；明确写出结构证据缺口。仅 EPS 缺失不得进入本组。
7. **优先复核池**：ACTIONABLE、Critical 全部 PASS、Major FAIL 不超过 1、Major UNKNOWN 为 0；EPS INFO_MISSING 不影响资格。

优先复核池按以下顺序比较：

1. Geometry 层级；
2. 更少 Major FAIL；
3. 新鲜区 `current_vs_ibd_candidate_pct <= 2%`；
4. Minor #7：PASS > UNKNOWN > FAIL；
5. Minor #10 正向加分；
6. `code` 字典序，再按 CSV 原始行序。

值得留意池按以下顺序比较：

1. 更少 Major UNKNOWN；
2. 更少 Major FAIL；
3. Geometry 层级；
4. 新鲜区；
5. Minor #7 与 #10；
6. `code` 字典序，再按 CSV 原始行序。

不得把 EPS 缺失置于 EPS PASS 之后形成隐性扣分；已知 EPS FAIL 已通过 Major FAIL 数体现，不得二次扣分。排序完成后再应用同 sector 最多 2 只的展示上限。

## 执行流程

1. 更新并读取 Market Context。
2. 合规加载最新 CSV，确认周中或完整周快照，建立 ACTIONABLE sector 占比。
3. 先执行 Critical 与 Geometry，再执行路由后的 Major / Minor。
4. 按组内规则排序，应用展示分散上限。
5. 输出极简中文报告；标题与结论必须明确“周中分析”或“完整周分析”。

## 输出规范

- 正文最多 30 个非空行；省略空区块。
- 结论严格只用一个非空行说明最优先、值得留意及是否宁缺毋滥；除非用户明确要求，不输出各分组总数。
- 背景最多一个非空行，只在市场状态或展示分散上限确实影响结论时写；未触发的 sector 上限不得写入背景。
- 优先复核每只固定“突破日 / 优势 / 判断”3 行。
- 值得留意每只固定“突破日 / 信息缺口或顾虑 / 判断”3 行。
- 暂不优先最多 3 只代表，每只只写一个决定性原因；有亮点时先写亮点。
- 入选标的存在 1 项 Major FAIL 时，必须在“判断”中明确披露；不得只写优势。
- 入选标的 EPS 为 INFO_MISSING 时写“EPS 数据缺失，需人工复核”，不得因此使用“顾虑”“基本面不足”等负面措辞。
- 引用 #4/#5 时只写路由后的客观数值。Continuation 可另列母基底背景，但不得混写。
- 不外显 PASS/FAIL/CONTEXT/UNKNOWN/INFO_MISSING/N/A、检查数量或内部计分。
- 不解释代码字典序、CSV 行序或其他稳定性兜底规则；只写对人工 Review 有意义的事实。
- 不使用“完美”“健康”“完全共振”等超出正式证据的评价。

```markdown
# IBD 候选预筛（[周中分析 | 完整周分析]）

## 结论
[一句话结论]
[必要时写市场或展示分散背景]

## 优先复核
### [TICKER]
- **突破日：** [Geometry]｜收盘位置 [pos×100，1位小数]%｜高于触发位 [close_vs_trigger×100，2位小数]%｜量能 [entry volume，2位小数]×均量
- **优势：** [最多两个带规范化数字的理由]
- **判断：** [为何值得先人工 Review；披露适用的 Major 顾虑或 EPS 缺失]

## 值得留意
### [TICKER]
- **突破日：** [同上]
- **信息缺口：** [当前阶段真正缺失的关键结构证据]
- **判断：** [为何观察但不列第一优先]

## 暂不优先
- **[TICKER]：** [亮点可选]，但 [一个决定性原因 + 规范化数字]
```
