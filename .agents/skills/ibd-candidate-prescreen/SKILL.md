---
name: ibd-candidate-prescreen
description: 从 Dashboard 的 ACTIONABLE 突破候选池中执行标准化、IBD-aligned 人工预筛；项目模式强制先执行一次 market_analysis 子模块更新，但 Market Report 只独立展示、绝不阻断或影响候选预筛；随后生成不受 Industry 覆盖影响的完整原始质量排序、可审计决策轨迹及最多 3 只优先人工复核标的，并对每个 ticker 的来源字段与格式化数字执行交付硬核查。用户请求“预筛标的”“IBD 分析”“突破池分析”“review 候选池”或周中/完整周候选复盘时使用。本 Skill 不是未来表现预测、完整 CAN SLIM Review、行业领导者认证或投资建议。
---

# IBD Candidate Pre-screen

## 任务定位

- 目标固定为：**从当前 ACTIONABLE Pool 中，选出量价结构、买点新鲜度和阶段质量证据最强的最多 3 只，供人工优先复核。**
- 先产生完整、确定性的原始质量排序，再应用 Industry 覆盖规则；不得把最终展示顺序冒充原始质量顺序。
- 输出 0～3 只“优先复核”、0～2 只“值得留意”、独立“Alpha Radar（非 ACTIONABLE，仅观察）”、代表性“暂不优先”和完整候选决策轨迹；宁缺毋滥，不为凑满名额放宽规则。
- 只评估当前正式导出字段，不预测未来收益、走势或目标价，不给仓位与买卖指令，不完成完整 CAN SLIM 认证，不替代图表人工判断。
- 规则优先。历史复盘经验只能改进证据簇推理顺序，不得把事后收益、样本中位数或个别 ticker 的数值范围写成新门槛，也不得覆盖 Critical、阶段字段路由或覆盖选择。
- 跨模型一致性优先：候选名单、原始顺位、最终分组、理由码和数字必须来自确定性 artifact；模型只能解释 artifact，不得凭自然语言权衡重新排序或替换名单。
- 不修改原始数据，不用未提供字段推导新指标。所有计算在内存中完成，不在项目目录遗留临时脚本、中间文件或数据副本。
- 单次预筛不能证明所选三只未来表现最好。始终保留 `snapshot_date`、原始质量顺位、最终分组与决策原因，使后续历史回测可以检验规则；实时预筛中严禁读取未来价格或事后收益。

## 禁止推断边界

- `industry` 只用于优先人工复核名单的覆盖控制和同业补强路由；不得参与候选资格、原始质量排序或个股评分。
- `sector` 仅为描述性背景，不设展示上限，不参与排序或分组。
- 候选数量、候选占比、Pool 内顺位，以及某只股票在本次 Pool 的同业候选中顺位第一，均不得解释为行业强弱、行业领导性、资金共振或未来表现。
- 不得使用“行业领导者”“领先行业中的龙头”“行业代表”“Industry 代表”“覆盖代表”“该行业候选最多所以最强”等表述。只能写“本次候选池内，该 Industry 原始质量顺位更靠前且信息完整的候选”。
- 即使输入未来提供正式的 IBD 行业排名、行业相对强度或行业资金数据，也只能按来源与快照日期作为背景事实展示；不得据此自动认定个股为行业领导者，也不得覆盖个股量价、买点与阶段结构检查。
- 本 Skill 中“每个 Industry 最多 1 只进入优先复核”只是有限人工复核名额的覆盖规则，不是行业质量结论、持仓分散规则或组合风控结论。

## 输入与前置步骤

1. 先确定输入模式：
   - **项目模式**：当前工作区存在目标项目与 `market_analysis`，或用户要求自动分析项目内最新 Pool。
   - **独立 CSV 模式**：用户明确上传或指定某个 CSV，且项目仓库或 `market_analysis` 不可用。
2. **项目模式必须先尝试更新 Market Analysis，之后才能选择 Pool、读取报告或开始候选计算：**
   - 只运行一次 `git submodule update --init --remote market_analysis`，并记录命令成功或失败；不得用 `git -C market_analysis pull` 替代。
   - 命令成功后可记录 `git -C market_analysis rev-parse HEAD`。命令成功但 commit 未变化，或更新后报告仍是旧快照，都是允许的。
   - 命令非零退出、权限失败、网络失败、目录不存在或因本地修改无法更新时，简要披露失败并继续候选预筛；不得 reset、stash、覆盖本地修改、反复重试或改用互联网市场信源。
3. 项目模式完成上述更新尝试后，可从 `market_analysis/output/market_report.json` 读取大盘背景：
   - 文件存在且 JSON 有效时，忠实独立展示报告已有内容；市场日期只能来自报告正式字段，不得用文件修改时间、Git 时间或模型猜测补造。
   - 报告仍旧、Market 与 Pool 日期不同、正式日期缺失或多个日期字段冲突，均不得停止或改变候选预筛。能确定日期时同时展示实际日期；不能确定时写“Market Report 日期无法确定”。
   - 文件缺失或 JSON 无效时省略大盘背景并继续候选预筛，明确写“大盘背景不可用，本次未纳入”。
   - Market Status、报告日期、报告新旧、子模块更新结果及任何市场分布字段都不得进入候选资格、原始排序、Industry 覆盖或最终分组。
4. 用户明确提供 CSV 并限定“只分析该文件”时，直接使用该文件，不自动切换快照；否则调用 `dashboard.data_utils.get_latest_pool_csv_path()`，按文件内容中的 `snapshot_date` 比较 `us/breakout_follow_pool.csv` 与 `us/breakout_follow_pool_midweek.csv`。
5. 项目模式优先调用 `dashboard.data_utils.load_pool_csv(path)`，先确认实际函数签名；不得擅自把返回值解包成 `(df, is_midweek)`。若依赖不可用，立即改用 Python 标准库 `csv` 读取实际存在的候选文件并比较有效 `snapshot_date`；不得固定读取某一个文件，也不得反复探测多个 Python/Conda 环境。无法唯一确定最新快照时停止并报告。
6. **Market Report 与 Pool 不做快照时效门槛：** 不计算用于放行或停止的日期差，不要求同日或相差不超过若干天。两者日期不一致时只按原值并列展示，并注明“大盘背景独立展示，不参与候选预筛”。
7. 独立 CSV 模式不得尝试更新不存在的 submodule。若用户同时提供 Market Report，同样仅作独立背景展示；未提供时继续候选字段预筛，并明确写“大盘背景未提供，本次未纳入”，不得联网补齐或自行判断市场趋势。
8. 分析类型按以下顺序确定：用户明确指定 > 标准文件名明确包含 `midweek` 时为“周中分析” > 项目内标准完整周路径为“完整周分析” > 其他重命名或上传文件统一写“指定快照分析”。不得只凭 `snapshot_date` 猜测周中或完整周。
9. CSV 必须用支持引号、转义和引号内换行的标准解析器。`ibd_candidate_extra` 含带逗号 JSON，禁止手工按逗号切分。任一数据行列数与 Header 不一致时停止并报告“候选池解析失败”，不得静默跳过。
10. 核心 Header 为 `code`、`snapshot_date`、`signal`、`ibd_candidate_rule`、`ibd_entry_status`、`current_vs_ibd_candidate_pct`、`ibd_entry_volume_ratio`、`ibd_entry_close_vs_trigger_pct`、`ibd_entry_close_position`、`ibd_entry_breakout_range_ratio`、`industry`；缺少任一项时停止。Pool 内 `snapshot_date` 缺失、无效或不唯一时停止。`sector` 与其他辅助字段缺失时按下文状态处理。
11. 字符串去除首尾空白，`ibd_entry_status` 转大写。`signal`、`pullback_v_is_dry` 接受布尔值、大小写不敏感的 `true/false` 或 `1/0`。无法识别的 `signal` 作为数据错误转人工补数；无法识别的 `pullback_v_is_dry` 记 UNKNOWN。
12. 数值为空、非数字、NaN 或正负 Infinity 时视为缺失，不参与运算或排序。所有判断使用未四舍五入原值，只在报告中格式化。
13. EPS 只读取当前 ticker 当前行正式列 `eps_yoy_growth`。不得从 `ibd_candidate_extra`、其他 ticker、旧报告或模型记忆回填；缺失严格记 INFO_MISSING，等待数据层补齐。
14. 为每个候选绑定唯一原始 CSV 行。后续所有字段、判断和报告数字必须直接引用该评估记录，不得从自然语言摘要或另一 ticker 的记录重建。

## 状态语义

| 状态 | 含义 |
|---|---|
| PASS | 检查适用、字段有效、存在正式标准且满足 |
| FAIL | 检查适用、字段有效、存在正式标准但不满足 |
| CONTEXT | 字段有效但仅作客观背景，不构成淘汰或加分 |
| UNKNOWN | 当前技术或结构判断所需字段缺失、不可靠或错位 |
| INFO_MISSING | 非决定性辅助信息缺失；不降低原始质量顺位，但需要人工核验 |
| N/A | 检查不适用 |

- CONTEXT、INFO_MISSING、N/A 不计入 Major FAIL 或 Major UNKNOWN，也不得作为原始质量排序键。
- EPS 缺失是 INFO_MISSING，不是基本面差，不得写成 FAIL，也不降低原始质量顺位。只有候选凭其他证据进入本轮人工关注前沿时，EPS 缺失才使其具备“值得留意”资格；EPS 缺失本身不得提升前沿之外的低顺位候选。
- Critical UNKNOWN 不得伪装为 FAIL；无 Critical FAIL 时转人工补数。
- 所有适用检查必须保留 PASS / FAIL / UNKNOWN 三态，禁止用 `bool(None)`、`not pass`、默认 `False` 等方式把缺失折叠成失败。

```text
if value 缺失或无效:
    state = UNKNOWN
elif value 满足正式标准:
    state = PASS
else:
    state = FAIL
```

## 候选范围与双层结果

- Review Universe 仅包含 `signal == True` 且 `ibd_candidate_rule` 非空的行。
- “完整原始质量排序”只对其中 `ibd_entry_status == ACTIONABLE` 的候选编号；非 ACTIONABLE 不与 ACTIONABLE 混排。
- 非 ACTIONABLE Review Universe 行可进入单独的 **Alpha Radar**：只用于发现值得后续人工观察的强量价或高质量 pullback 线索，不编号为 ACTIONABLE 原始顺位，不进入优先复核/值得留意，不给买卖结论。
- Alpha Radar 拥有独立容量，必须从非 ACTIONABLE Review Universe 独立排序生成；不得用 ACTIONABLE 原始排序的剩余候选或最终分组 leftovers 占用、替代或遮蔽非 ACTIONABLE 发现名单。
- 评估非 ACTIONABLE Alpha Radar 时，突破日字段缺失是人工看图/补数提示，不是自动压制理由；若周线量能、EPS 辅助、接近 52 周高点、回踩/巩固结构等证据形成一致链路，可进入 radar，但必须标注缺失项和状态限制。
- 第一层是**个股原始质量排序**：只使用当前 ticker 的技术、量价、新鲜度、阶段证据和辅助确认簇；不使用 `industry`、`sector`、行业候选数量或最终展示名额。
- 第二层是**人工复核覆盖选择**：从原始排序顺次处理，最终最多 3 只“优先复核”，每个已知 Industry 最多 1 只。
- 两层结果必须同时保留。若 Industry 覆盖导致原始高顺位候选未进入优先复核，必须在决策轨迹中说明，不能改写其原始顺位。

将 `industry` 去除首尾空白并压缩连续空白，以大小写不敏感值作为覆盖键，同时保留原始显示值。空值不能当作一个共同 Industry 占用名额；它属于信息缺口。

## #4 / #5 阶段字段路由

这里只约束 Checklist #4 Depth 与 #5 Duration，不约束 #6 缩量。

| `ibd_candidate_rule` | 当前阶段 | #4 / #5 唯一字段 |
|---|---|---|
| `ceiling`、`ceiling_breakout` | 初始突破基底 | `base_depth_pct` / `base_duration_weeks` |
| `ceiling_pullback` 及其他非空值，如 `pivot`、`ma10_touch_confirm`、`three_weeks_tight` | 当前回踩或巩固 | `pullback_pct` / `pullback_duration_weeks` |
| 空或缺失 | UNKNOWN | 不读取另一组字段 |

- 严禁跨阶段替代、计算替代值或因字段缺失改读另一组字段。
- 路由字段有效时 #4/#5 记 CONTEXT；缺失记 UNKNOWN。
- Continuation 规则可额外展示 `base_depth_pct` / `base_duration_weeks` 作为母基底背景，但不替代当前 #4/#5，也不参与 Major 状态或排序。
- 不得把深度、时长或信号名解释成“健康”“过深”“偏长”“太短”等未定义结论。
- 本策略基于周线；`ma10_touch_confirm` 面向用户必须写“10周线触及确认”，不得写成“10日线”。

## 核心数值语义与展示规范

| 字段 | 计算语义与原始单位 | 强制展示 |
|---|---|---|
| `ibd_entry_close_position` | `(Close-Low)/(High-Low)`，0～1 比例 | `0.9414` → `收盘位置94.1%` |
| `ibd_entry_breakout_range_ratio` | `(Close-Trigger)/(High-Low)`，日振幅倍数 | `1.1873` → `穿透1.19×日振幅`；正文默认不展示 |
| `trigger_pos` | `(Trigger-Low)/(High-Low) = pos-rr` | 仅内部 Geometry 计算 |
| `ibd_entry_close_vs_trigger_pct` | `(Close-Trigger)/Trigger`，小数比例 | `0.0468` → `高于触发位4.68%` |
| `ibd_entry_volume_ratio` | 突破日成交量/基准均量 | `1.7459` → `突破日量能1.75×均量` |
| `current_vs_ibd_candidate_pct` | 当前价相对候选买点，原值已是百分数 | `2.70` → `当前高于买点2.70%` |
| `base_depth_pct` | 基底回撤，原值已是带符号百分数 | `-20.2` → `基底深度20.2%` |
| `pullback_pct` | 当前回踩，原值已是带符号百分数 | `-13.8` → `回踩幅度13.8%` |
| `dist_to_52w_high_pct` | 当前价距 52 周高点，原值已是百分数 | `-2.24` → `低于52周高点2.24%` |
| `eps_yoy_growth` | EPS 同比，原值已是百分数 | `31.3` → `EPS同比+31.3%` |
| `volume_ratio` | 周线成交量/周均量 | `1.70` → `周线量能1.70×均量` |
| `*_duration_weeks` | 周数 | `6.0` → `6周`；非整数保留 1 位小数 |

强制规则：

- `ibd_entry_breakout_range_ratio` 不是百分比，可以大于 1。不得写成不带单位的“突破幅度1.19”，也不得解释成 19% 或 119%；必须展示时写“穿透1.19×日振幅”。
- 面向用户的“高于触发位多少”只使用 `ibd_entry_close_vs_trigger_pct` 并按百分比格式化；它不参与 Geometry。
- “突破日量能”只读取同一 ticker 当前原始行的 `ibd_entry_volume_ratio`；`volume_ratio` 只能写成“周线量能”。两者不得交换、替代、混写或从相邻行复制。
- `ibd_entry_close_position` 按百分比展示并保留 1 位小数，不得直接输出 `0.94`。
- 倍数统一保留 2 位小数并写 `×均量` 或 `×日振幅`；百分数必须使用“交付前一致性校验”表规定的小数位，不得由模型自行选择 1 位或 2 位。
- 使用方向词代替含混负号：写“回踩幅度13.8%”“低于52周高点2.24%”，不写“回踩-13.8%”“距高点-2.24%”。
- 当前价高于买点写“当前高于买点 x%”；低于时写“当前低于买点 x%”。缺失值写“数据缺失”或省略，不得显示为 0。
- 先独立验证 `pos`：有限但不在 `[0,1]` 时记 Critical UNKNOWN，且不得再用该值测试 `pos < 0.65` 或其他 Geometry 分支。`rr > 1` 可以合法表示 Trigger 低于当日 Low，不得截断。

## Breakout Geometry

只使用：

```text
pos = ibd_entry_close_position
rr = ibd_entry_breakout_range_ratio
trigger_pos = pos - rr
```

先验证 `pos` 的有效范围，再严格按顺序首次命中：

| 顺序 | 条件 | 分类 | 结果 |
|---:|---|---|---|
| 1 | `pos` 有限但不在 `[0,1]` | UNKNOWN / Data Error | Critical UNKNOWN |
| 2 | `rr <= 0` | Defensive Failure | Critical FAIL |
| 3 | `pos < 0.65` | Squat / Upper Shadow | Critical FAIL |
| 4 | `pos` 或 `rr` 缺失 | UNKNOWN | Critical UNKNOWN |
| 5 | `trigger_pos <= 0` 且 `pos >= 0.80` | Full-range Breakout | PASS |
| 6 | `trigger_pos <= 0` 且 `0.65 <= pos < 0.80` | Faded Gap | PASS |
| 7 | `trigger_pos > 0` 且 `pos >= 0.80` 且 `rr >= 0.50` | Strong Finish | PASS |
| 8 | `trigger_pos > 0` 且 `pos >= 0.80` 且 `rr < 0.50` | Constructive Breakout | PASS |
| 9 | `trigger_pos > 0` 且 `0.65 <= pos < 0.80` 且 `rr >= 0.50` | Constructive Breakout | PASS |
| 10 | `trigger_pos > 0` 且 `0.65 <= pos < 0.80` 且 `rr < 0.50` | Marginal Breakout | PASS |

- PASS Geometry 分类用于解释突破日路径质量和人工看图重点，不代表天然收益排序。`Full-range Breakout`、`Strong Finish`、`Faded Gap`、`Constructive Breakout`、`Marginal Breakout` 均为非失败路径证据；其中 `Faded Gap` / `Constructive Breakout` / `Marginal Breakout` 只能写成“需人工确认突破后是否正常消化”，不得自动解释成弱势。
- 为保证全量排序确定性，完整 Geometry tie-breaker 顺序固定为 `Full-range Breakout > Strong Finish > Faded Gap > Constructive Breakout > Marginal Breakout > UNKNOWN > Squat / Upper Shadow > Defensive Failure`；该顺序只在证据簇与证据完整度仍并列时使用，不能越过更强的 Fresh Demand 或 Pullback 证据簇。
- `pos` 越界优先作为数据错误处理，不得把负数误判成 `pos < 0.65`。除此之外，`rr <= 0` 或有效 `pos < 0.65` 一旦确认，即使另一字段缺失仍为 Critical FAIL。
- 报告只使用上述英文分类，不得仅凭 `rr` 或主观观感升级分类。
- `rr` 极低但大于 0 时仍按正式 Geometry 分类；可客观说明“仅小幅越过触发位”，不得另创淘汰阈值。

## IBD-aligned 分层检查表

| 级别 | # | 检查点 | 唯一标准 |
|---|---:|---|---|
| Critical | 1 | 买点新鲜度 | 只用 `current_vs_ibd_candidate_pct`；缺失 UNKNOWN；`0%～5%` PASS，`>5%` 或 `<0%` FAIL，`0%～2%` 为排序新鲜区 |
| Critical | 2 | 突破日放量 | 只用 `ibd_entry_volume_ratio`；缺失 UNKNOWN；有效值 `>=1.5` PASS，否则 FAIL |
| Critical | 3 | 突破日质量 | Geometry 非 Defensive Failure、非 Squat / Upper Shadow |
| Major | 4 | 阶段深度 | 严格按阶段路由；有效为 CONTEXT，缺失为 UNKNOWN |
| Major | 5 | 阶段时长 | 严格按阶段路由；有效为 CONTEXT，缺失为 UNKNOWN |
| Major | 6 | 巩固期缩量 | `pullback_v_is_dry == True` PASS，`False` FAIL；`ceiling` / `ceiling_breakout` 为 N/A；其他规则字段缺失为 UNKNOWN |
| Minor | 7 | 紧贴 52 周高点 | `dist_to_52w_high_pct > -5.0` 为 PASS；缺失 UNKNOWN，否则 FAIL |
| Auxiliary | 8 | EPS 辅助信息 | 只读 `eps_yoy_growth`；`>=25` 为 PASS；有效但 `<25` 为 CONTEXT；缺失 INFO_MISSING。绝不计入 Major FAIL、淘汰或原始排序 |
| Minor | 9 | 净筹码吸纳 | 只使用上游正式字段；当前未导出时 N/A，不自行推导 |
| Minor | 10 | 周线量能跟进 | `volume_ratio >= 1.3` 时记一个二元正向加分；否则不加分 |

补充约束：

- Critical 任一 FAIL 即不能进入“优先复核”或“值得留意”；只有 Critical 全部明确 PASS 才能进入这两组。
- #4/#5 的 CONTEXT 表示已有阶段事实，不是 PASS；缺失才构成 Major UNKNOWN。
- #6 只读取当前行正式字段 `pullback_v_is_dry`，不评价突破日放量。当前未导出可验证的窗口日期或有效性字段，因此不得凭观感判断“旧窗口错位”；字段有效即按真值判定，缺失才记 UNKNOWN。
- EPS 是辅助信息而非决定项。已知低于 25% 只作客观披露，不记 Major FAIL；缺失不降低原始顺位，也不会自动进入“值得留意”，只在候选已凭其他证据进入人工关注前沿时提示人工核验。
- Minor 不作淘汰条件。#10 只比较“有加分 / 无加分”；所有 `volume_ratio >= 1.3` 的候选在该层完全并列，不得按 1.70、1.50、1.30 的实际大小继续排序。
- `ibd_entry_close_vs_trigger_pct` 只作可读上下文，不参与 Geometry，也不得替代当前买点新鲜度。

## 证据簇推理顺序

在完成 Critical、阶段路由和 Geometry 分类后，先判断候选属于哪类证据簇，再排序；不得把“图形更漂亮”单独置于“需求、跟进、基本面辅助和阶段证据更完整”之前。

| 证据簇 | 触发含义 | 排序含义 |
|---|---|---|
| Fresh Demand Alpha | 近买点、突破日放量明确，且 EPS 辅助、周线量能、接近 52 周高点或恢复强度中有多项确认 | 优先寻找旧规则能抓到的 IMAX 型需求扩张；Geometry 不完美只作人工看图提示 |
| Constructive Pullback | `ceiling_pullback`、`pivot`、`ma10_touch_confirm`、`three_weeks_tight` 等延续/回踩规则，近买点且量价证据明确 | 优先判断是突破后正常消化还是失败；dry pullback 是支持证据，非 dry 是风险提示 |
| Standard Breakout | Critical 通过但辅助确认较少 | 可进入排序，但不得因单一完美 Geometry 压过证据簇更完整的候选 |
| Incomplete Evidence | 关键字段通过但辅助证据不足或多项缺失 | 保留轨迹，通常不应进入优先复核前沿 |

证据完整度只使用二元/三态事实，不使用数值大小作过度拟合排序：买点新鲜、突破日放量达标、EPS 是否达到辅助门槛、周线量能是否跟进、是否接近 52 周高点、回踩是否缩量。EPS 仍然是辅助信息：不得作为淘汰条件，不得按 EPS 数值高低排序；但在其他关键证据已经通过时，`EPS >= 25` 可作为 Fresh Demand 证据簇的一项辅助确认。

非 ACTIONABLE Alpha Radar 的推理顺序：

1. 先剔除明确结构失败：Geometry Defensive Failure / Squat、当前低于候选买点、或其它可由当前字段确认的失败路径。
2. 再寻找证据链，而不是单项打分：接近买点或处在可解释的延伸/回踩状态、周线量能跟进、EPS 辅助达标、接近 52 周高点、pullback 结构或缩量证据。
3. 对 `EXTENDED` 与 `UNCONFIRMED` 分状态展示，保持 radar 属性；不得因它历史表现好而回填成 ACTIONABLE，也不得因 entry-volume 缺失直接排除。
4. 另设非 ACTIONABLE **Pullback Scout** 视角：`ceiling_pullback`、`pivot`、`ma10_touch_confirm`、`three_weeks_tight` 若近买点，且 pullback 结构与接近 52 周高点、周线量能跟进或缩量证据中至少形成一条一致链路，可进入人工看图观察；`pullback_v_is_dry == False`、Geometry caution、entry-volume 缺失均写入风险标签，不作为硬压制或自动淘汰。
5. Pullback Scout 不替代 ACTIONABLE 原始排序、优先复核或主 Alpha Radar；它只回答“哪些未确认回踩值得后续看图跟踪”，不能写成当前买点确认。
6. 排序解释必须写成“为何值得后续人工看图”，不是“为何现在应买入”。重点是没有先被明确失败打掉、过程和最终路径相对强，而不是事后收益最高。

## 原始质量分层与排序

先为每个 ACTIONABLE 候选建立技术分层：

1. **A｜可进入覆盖选择**：Critical 全部 PASS、Major UNKNOWN 为 0。当前规则中唯一可能出现明确 Major FAIL 的检查是 #6；它会降低原始顺位并必须披露，但单独不作淘汰。
2. **B｜结构信息待补**：Critical 全部 PASS，但存在 Major UNKNOWN。
3. **C｜关键数据待补**：无 Critical FAIL，但存在 Critical UNKNOWN。
4. **D｜关键条件不满足**：存在任一 Critical FAIL。

完整原始质量排序按以下键依次比较：

1. 技术分层 A > B > C > D；
2. 更少 Major FAIL；
3. 更少 Major UNKNOWN；
4. 证据簇：Fresh Demand Alpha > Constructive Pullback > Standard Breakout > Incomplete Evidence；
5. 更完整的证据确认项数量：买点新鲜、突破日放量达标、EPS 辅助达标、周线量能跟进、接近 52 周高点、适用时回踩缩量；
6. 更少非淘汰风险提示：例如当前巩固段未确认缩量、EPS 缺失、Industry 缺失、仅小幅越过触发位等；
7. 新鲜区 `0 <= current_vs_ibd_candidate_pct <= 2%`；
8. Geometry tie-breaker；Geometry UNKNOWN 排在已知 PASS 分类之后；
9. Minor #7：PASS > UNKNOWN > FAIL；
10. Minor #10：有二元加分 > 无加分；
11. `code` 字典序，再按 CSV 原始行序。

硬性约束：

- `industry`、`sector`、行业候选数量、EPS 数值大小和最终展示名额均不得进入上述排序键。EPS 只能以“辅助门槛是否达标/是否缺失”的状态参与证据完整度或风险提示，不能按实际同比数值排序。
- 原始顺位一经产生不得因 Industry 覆盖、EPS 人工核验或 Top 3 名额重排。
- 不得把原始质量顺位命名为预测排名、收益排名、行业排名或领导者排名。

## 最终分组与 Industry 覆盖

先冻结完整原始质量排序，再执行三个阶段。不得边排序边因 EPS 或 Industry 改变顺位。

### 阶段一：确定优先复核与优先截断位

按原始顺位逐只接收，只有同时满足以下条件的候选才能进入“优先复核”：

- 技术分层为 A；
- EPS 已知；EPS 是否达到 25% 不影响资格；
- Industry 已知；
- 该 Industry 尚未被优先复核覆盖；
- 优先复核尚未满 3 只。

若成功选出 3 只，`priority_cutline_rank` 为第 3 只优先复核候选的原始顺位；若最终不足 3 只，则设为 ACTIONABLE 原始排序的末位。该值表示优先复核选择为凑齐 3 个不同且信息完整的 Industry 实际扫描到哪里，不得直接作为“值得留意”的截止位。

### 阶段二：建立独立人工关注前沿

固定 `priority_limit = 3`、`watch_limit = 2`、`base_attention_capacity = priority_limit + watch_limit = 5`，然后计算：

```text
base_attention_rank = min(last_raw_rank, base_attention_capacity)
attention_frontier_rank = max(priority_cutline_rank, base_attention_rank)
```

- ACTIONABLE 排序为空时，`last_raw_rank`、`priority_cutline_rank`、`attention_frontier_rank` 均为 0。
- 该前沿至少覆盖原始排序前 5 名（若有），对应最多 3 只优先复核加最多 2 只值得留意的总人工关注容量。
- 若 EPS、Industry 或 Industry 覆盖使第 3 只优先复核顺延到 #6、#7 等更后顺位，前沿同步扩展至第 3 只优先复核的顺位，保留扫描过程中被跳过的高顺位候选。
- 若不足 3 只优先复核，`priority_cutline_rank = last_raw_rank`，表示已扫描完整排序；值得留意仍只按原始顺序最多接收 2 只，不为凑数降低资格门槛。
- 人工关注前沿只是有限人工复核容量的资格窗口，不是新的评分键，不改变原始顺位，也不保证窗口内每只候选都会展示。

硬性示例：若原始 #1～#3 全部进入优先复核且总候选不少于 5 只，则 `priority_cutline_rank = 3`、`attention_frontier_rank = 5`；不得再以“#4 超过第 3 只优先复核的顺位”为由，把符合下述条件的 #4 或 #5 直接归入暂不优先。

### 阶段三：从人工关注前沿精选值得留意

未进入优先复核的候选必须同时满足以下条件，才具备“值得留意”资格：

- 原始顺位 `<= attention_frontier_rank`；
- Critical 全部明确 PASS；
- 技术分层为 A 或 B；
- 存在至少一个明确路由原因：阶段结构信息缺口、EPS 信息缺失、Industry 信息缺失，或其 Industry 已被更高原始顺位候选覆盖。

将具备资格的候选保持原始顺序，最多接收 2 只进入“值得留意”。同业补强最多保留每个已覆盖 Industry 的最高原始顺位 1 只。把同一候选所有适用 Checklist 与覆盖字段缺口逐项列出，不得首次命中后遗漏另一项；`sector` 等可选描述性背景缺失不列为人工核验项。

- EPS 缺失只能解释为什么一个**已经凭技术质量进入人工关注前沿**的候选需要人工核验，不能让前沿之外、关键条件失败或非 ACTIONABLE 的候选升级。
- 值得留意已满 2 只后，其余具备资格的候选保留在完整决策轨迹，最终写“暂不优先—值得留意名额已满”；不得声称其技术质量不合格。
- 排在人工关注前沿之后（`raw_rank > attention_frontier_rank`）的信息缺失候选只在决策轨迹逐项记录全部缺失字段，不进入“值得留意”。信息缺失不得成为淘汰原因。
- 候选进入“值得留意”后，除路由原因外，还要列出该行所有其他适用 Checklist 与覆盖字段缺口。例如 EPS 或 Industry 缺失触发路由时，若 `dist_to_52w_high_pct` 同时缺失，也必须列为额外人工核验项，但 Minor 缺失本身不单独触发入组；不列 `sector` 等不参与检查或覆盖的可选背景字段。
- 第一个进入优先复核的 Industry 候选只能描述为“本次候选池内该 Industry 原始质量顺位更靠前且信息完整的候选”，不得称为该行业最强或行业领导者。

其他最终路由：

- 技术分层 D → **暂不优先**，写一个决定性 Critical 原因。
- 技术分层 C → **人工补数**，写明缺失的关键字段。
- 未达到人工关注前沿 → **暂不优先**，原因写“原始顺位超出本轮人工关注前沿”；不得把 EPS 缺失写成淘汰原因。
- 位于人工关注前沿内但没有明确值得留意路由原因，或值得留意名额已满 → **暂不优先**，分别写“优先复核名额已满且无信息核验/同业补强路由”或“值得留意名额已满”。
- 非 ACTIONABLE 不进入上述 ACTIONABLE 原始排序或覆盖循环；若具备强 Fresh Demand 或 Constructive Pullback 证据，放入单独的 Alpha Radar；否则在“排除记录”中写明当前状态。不论哪种情况，都不分配 ACTIONABLE 原始顺位。

参考伪代码：

```text
selected = []
covered_industries = set()
priority_limit = 3
watch_limit = 2
for item in raw_quality_ranking:
    if item.technical_tier != "A":
        continue
    if item.eps_state == INFO_MISSING or item.industry_key is missing:
        continue
    if item.industry_key in covered_industries:
        continue
    selected.append(item)
    covered_industries.add(item.industry_key)
    if len(selected) == priority_limit:
        break

last_raw_rank = raw_quality_ranking[-1].raw_rank if raw_quality_ranking else 0
priority_cutline_rank = selected[-1].raw_rank if len(selected) == priority_limit else last_raw_rank
base_attention_rank = min(last_raw_rank, priority_limit + watch_limit)
attention_frontier_rank = max(priority_cutline_rank, base_attention_rank)

watch_eligible = []
for item in raw_quality_ranking:
    if item in selected or item.raw_rank > attention_frontier_rank:
        continue
    if item.critical_all_pass and item.technical_tier in {"A", "B"}:
        reasons = collect_structure_eps_industry_and_same_industry_reasons(item)
        if reasons:
            watch_eligible.append(item)

watch = take_first_two_by_raw_rank(
    watch_eligible,
    at_most_one_same_industry_reinforcement_per_covered_industry=True,
)
```

## 执行流程与决策轨迹

1. 确定输入模式。项目模式必须先执行一次 `git submodule update --init --remote market_analysis`；在此之前不得读取 Market Report、选择 Pool 或开始候选计算。记录命令结果，但更新失败不阻断候选预筛。
2. 更新尝试后，项目模式读取可用的 `market_report.json` 并将其作为独立背景；报告旧、缺失、无效或日期无法确定时如实披露并继续。独立 CSV 模式只在用户提供报告时读取，否则记录“大盘背景未提供，本次未纳入”。
3. 合规加载目标 CSV，确认唯一 `snapshot_date`，并按前置规则确定“周中分析”“完整周分析”或“指定快照分析”。
4. 建立来源记录：`market_analysis_update_result + market_analysis_commit（若可得）+ market_snapshot_date（若可得）+ pool_snapshot_date + pool_path`。该记录只用于披露来源，任何字段都不得传入候选评分、排序或分组函数。
5. 生成确定性预筛 artifact，作为所有模型的唯一排序与报告骨架来源。项目内可运行：
   `conda run --no-capture-output -n quant_env python -m backtest.ibd_skill_iteration.deterministic_prescreen --pool [pool.csv] --snapshot-date [YYYY-MM-DD] --version v3 --json-out [artifact.json] --markdown-out [artifact.md]`
   若该脚本不可用，必须用同一套代码路径或停止说明，不能退回模型手工排序。
6. 为每行建立评估记录，至少保存：`snapshot_date`、`code`、原始行序、原始字段、解析值、阶段路由、所有检查状态、Geometry、证据簇、技术分层、原始质量顺位、EPS 状态、Industry 覆盖键、覆盖决策、Alpha Radar 资格、非 ACTIONABLE radar 顺位、最终分组、全部缺失项、决定性原因和格式化值。非 ACTIONABLE 记录不分配 ACTIONABLE 原始顺位。
7. 先执行 Critical 与 Geometry，再执行阶段路由后的 Major、Minor 与 EPS 辅助信息。
8. 生成完整 ACTIONABLE 原始质量排序；冻结顺位后再应用 EPS 人工核验与 Industry 覆盖选择。
9. 由 artifact 中同一评估记录模板化渲染报告；不得凭记忆、旧报告或其他 ticker 的句子手工补写数字，不得重排 `priority_top3`、`actionable_raw_top5`、`alpha_radar_top5`、`non_actionable_alpha_radar_top10` 或 `pullback_scout_top10`。
10. 输出完整候选排序与决策轨迹。每个 ACTIONABLE 候选至少显示：原始顺位、Code、Industry、技术分层的人类可读说明、最终分组和决策原因；无论最终分组为何，都在内部记录并在轨迹需要时逐项列出全部适用 Checklist 与覆盖字段缺口，不能只保留首次命中的缺口。
11. 执行交付前双向一致性校验。数字正确是交付硬门槛：任一候选数字无法追溯、取错 ticker、取错字段或格式化不一致时，必须从当前原始行重新渲染并重跑校验；仍无法确认时省略该数字事实，不得估算或带错发送。

决策轨迹使用可读链路，例如：

```text
可进入覆盖选择 → EPS 已知 → Industry 未覆盖 → 优先复核
人工关注前沿内 → EPS 已知 → Industry 已覆盖 → 值得留意（同业补强）
人工关注前沿内 → EPS 缺失 → 值得留意（需人工核验）
人工关注前沿外 → EPS 缺失 → 暂不优先（仅在轨迹记录缺失）
突破日量能 1.22×均量 < 1.50× → 暂不优先
```

## 交付前一致性校验

为报告中的每个候选数字保留 `ticker + source_field + raw_value + formatted_value` 四元组，并逐项验证：

| 报告短语 | 唯一来源 | 格式化 |
|---|---|---|
| 收盘位置 | `ibd_entry_close_position` | `raw * 100`，1 位小数，百分比 |
| 高于或低于触发位 | `ibd_entry_close_vs_trigger_pct` | `abs(raw) * 100`，2 位小数，百分比 |
| 突破日量能 | `ibd_entry_volume_ratio` | 2 位小数，`×均量` |
| 当前高于或低于买点 | `current_vs_ibd_candidate_pct` | 固定 2 位小数；原值已是百分数 |
| 基底或回踩幅度 | 路由后的 `base_depth_pct` / `pullback_pct` | 绝对值，1 位小数，百分比 |
| 距 52 周高点 | `dist_to_52w_high_pct` | 绝对值，固定 2 位小数，方向词 + 百分比 |
| EPS 同比 | `eps_yoy_growth` | 带方向符号，1 位小数，百分比 |
| 周线量能 | `volume_ratio` | 2 位小数，`×均量` |

最终发送前必须确认：

1. 项目模式已在读取 Market Report、选择 Pool 或候选计算之前执行一次 `git submodule update --init --remote market_analysis`；命令失败时已披露，但候选预筛仍继续。
2. Market Report 只作独立背景：报告新旧、日期差、缺失、无效与更新结果均未进入候选资格、原始排序、Industry 覆盖或最终分组。
3. 每个候选数字来自当前标题 ticker 的同一原始行，尤其逐一复核 `ibd_entry_volume_ratio`、`current_vs_ibd_candidate_pct` 与 `dist_to_52w_high_pct`，不得跨 ticker 复制或错位小数点。
4. 完成双向数字审计：从最终文本的每个候选数字反查 `ticker + source_field + raw_value + formatted_value` 四元组，同时从四元组检查最终文本中的格式化值。必须逐字符核对数字串与小数位；例如 MTUS 原值 `dist_to_52w_high_pct=-4.0301...` 只能渲染为“低于52周高点4.03%”，不得漂移为 0.43%。
5. 突破日量能与周线量能未混用；`ibd_entry_breakout_range_ratio` 未被写成百分比。
6. 原始顺位不含 Industry、Sector、行业候选数量或 EPS 数值大小；EPS 缺失只作为非淘汰风险提示，最终分组与覆盖决策没有反向污染原始顺位。
7. 优先复核不超过 3 只，且每个已知 Industry 最多 1 只；未凑满时没有降低门槛。
8. 因同 Industry 未入选的候选确实排在已入选同业候选之后，或更高顺位候选存在 EPS / Industry 信息缺口；决策轨迹必须能解释例外。
9. EPS 缺失没有被记失败或降低原始顺位；只有人工关注前沿内且最终进入“值得留意”的候选写出“EPS 数据缺失，需人工复核”。若该候选还有其他字段缺口，已全部列出。
10. 报告未从候选数量、占比或 Pool 内同业顺位推断行业强弱或行业领导者，也未出现“行业代表”“Industry 代表”或“覆盖代表”。
11. 市场背景忠于 `market_report.json`，没有自行补充市场判断；报告日期与 Pool 日期不一致时已同时展示实际日期并明确其不影响预筛。
12. ACTIONABLE 排序为空时，`last_raw_rank`、`priority_cutline_rank` 与 `attention_frontier_rank` 均为 0，并输出空的优先复核、值得留意和决策轨迹，不得访问不存在的末项。
13. 当原始 #1～#3 全部进入优先复核且总候选不少于 5 只时，人工关注前沿仍为 #5；没有把 #4、#5 误判为“超出第 3 只优先复核截断位”。

## 输出规范

- 结论先说明优先复核数量、是否宁缺毋滥，以及 Industry 覆盖只影响人工复核名单、不改变原始顺位。
- 项目模式在结论后固定输出一行独立大盘背景。更新成功且报告可读时写：`大盘背景（独立展示）：Market Report [market date/日期无法确定]｜Pool [pool date]｜已先执行 market_analysis 更新[｜commit short hash]｜不参与候选排序与分组`。日期不同可照实并列，不评价“过期”或“失配”。
- 更新命令失败时写：`大盘背景：market_analysis 更新失败（[简短原因]）；候选预筛继续，市场信息不参与排序与分组`。报告缺失或无效时写：`大盘背景不可用，本次未纳入；候选预筛不受影响`。独立 CSV 且无报告时固定写“大盘背景未提供，本次未纳入”。这些情况都不得停止候选输出。
- “优先复核”完整展示 0～3 只，每只固定“突破日 / 优势 / 判断”3 行。
- “值得留意”完整展示 0～2 只详细卡片；每只必须已经处在人工关注前沿，并明确写原因：结构信息缺口、EPS 人工核验、Industry 信息缺口或同业补强，同时列出当前行其他适用 Checklist 与覆盖字段缺口。
- “暂不优先”正文最多展示 3 只，固定取该最终分组中原始顺位最靠前的 3 只；每只只写一个或两个决定性原因。全部候选仍保留在完整决策轨迹中。
- 非停止路径最后始终输出“完整候选排序与决策轨迹”表，覆盖所有 ACTIONABLE 候选；不可只输出最终 3 只。
- 若 Review Universe 中存在非 ACTIONABLE radar 候选，必须输出独立“Alpha Radar（非 ACTIONABLE，仅观察）”小节；若没有候选，也写明“本轮无非 ACTIONABLE radar 候选通过证据链”。该小节不受 ACTIONABLE 优先复核/值得留意名额影响。
- Review Universe 中的非 ACTIONABLE 候选如需展示，放入不编号的“Alpha Radar”或“排除记录”，不得混入 ACTIONABLE 原始质量排序；Alpha Radar 必须明确写“非 ACTIONABLE，仅供后续观察/人工看图”，不得写成优先复核。
- 入选详情不外显 Critical / Major / Minor、PASS / FAIL / UNKNOWN / INFO_MISSING 等内部状态；决策轨迹改用人类可读原因。
- 优先复核或值得留意候选若 #6 明确为 False，必须客观写明当前巩固段未确认缩量；不得因它仍获有限名额而隐藏该事实。
- EPS 已知但低于 25% 时可客观显示实际同比值与辅助门槛，但不得把它写成淘汰原因或用于解释原始顺位。
- 引用 #4/#5 时只写路由后的客观数值。Continuation 可另列母基底背景，但不得混写。
- “优势”和“判断”只使用正式 Geometry 名称与当前 ticker 的客观字段事实。不得使用“完美”“健康”“完全共振”“几何饱满”“形态平稳”“结构完整”“穿透充分”“综合质量领先”等无正式字段支撑的评价。
- 描述 Industry 覆盖时不得写“作为某 Industry 行业代表进入”。写“该候选进入本次优先复核，且其 Industry 尚未被本次名单覆盖”。
- 不把“值得留意”理解成评分降低：同业补强与信息缺口只是人工关注前沿内的路由原因；信息缺失本身也不得把前沿外的低顺位候选提升进本组。

```markdown
# IBD 候选预筛（[周中分析 | 完整周分析 | 指定快照分析]）

## 结论
[一句话结论；说明最多 3 个不同 Industry 的优先人工复核名额]
[项目模式报告可读：大盘背景（独立展示）：Market Report [date/日期无法确定]｜Pool [date]｜已先执行 market_analysis 更新[｜commit short hash]｜不参与候选排序与分组]
[项目模式更新失败：大盘背景：market_analysis 更新失败（[简短原因]）；候选预筛继续，市场信息不参与排序与分组]
[项目模式报告不可用：大盘背景不可用，本次未纳入；候选预筛不受影响]
[独立 CSV 无报告：大盘背景未提供，本次未纳入]
[必要时写市场状态；不得从候选集中度推断行业强弱]

## 优先复核
### [TICKER]
- **突破日：** [Geometry]｜收盘位置 [pos×100，1位小数]%｜高于触发位 [close_vs_trigger×100，2位小数]%｜突破日量能 [entry volume，2位小数]×均量
- **优势：** [最多两个带规范化数字的量价、新鲜度或阶段理由]
- **判断：** [为什么进入优先人工复核；只写本次 Pool 决策，不称行业领导者]

## 值得留意
### [TICKER]
- **突破日：** [同上]
- **需核验或补强：** [EPS/Industry/结构信息缺口，或同业补强/名额原因]
- **判断：** [为什么仍值得人工观察；不改变原始质量顺位]

## Alpha Radar（非 ACTIONABLE，仅观察）
### [TICKER]
- **状态：** [EXTENDED / UNCONFIRMED]｜[rule]｜非 ACTIONABLE，不进入优先复核
- **证据链：** [周线量能/EPS/接近高点/pullback 结构等当前字段事实]
- **需人工确认：** [突破日字段缺失、延伸、未确认、非缩量等风险提示]

## 暂不优先
- **[TICKER]：** [亮点可选]，但 [决定性原因 + 规范化数字]

## 完整候选排序与决策轨迹
| 原始顺位 | Code | Industry | 技术证据 | 最终分组 | 决策轨迹 |
|---:|---|---|---|---|---|
| 1 | [TICKER] | [Industry/缺失] | [可进入覆盖选择/结构信息待补/关键条件不满足] | [分组] | [可审计链路] |
```
