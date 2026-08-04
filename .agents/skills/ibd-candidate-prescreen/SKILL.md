---
name: ibd-candidate-prescreen
description: 从 Dashboard 突破候选池中，以 IBD 资深图表分析师视角执行分层检查表，结合大盘报告与突破日几何，精选最多 3 只优先复核标的并输出极简复盘报告。当用户请求“预筛标的”“IBD 分析”“突破池分析”或“review 候选池”时触发。
---

# IBD Candidate Pre-screen

## 任务边界

- 对候选池执行 IBD 对齐的快速预筛，输出 0～3 只“优先复核”；可另列少量“值得留意”和“暂不优先”作为上下文；宁缺毋滥。
- 规则优先，经验只用于规则比较后仍接近的候选；不得覆盖 Critical、字段路由、Geometry 或正式 PASS/FAIL。
- 不预测走势或目标价，不给仓位与买卖指令，不修改原始数据，不使用未提供字段推导新指标。

## 输入与前置步骤

1. 先运行 `git submodule update --remote market_analysis`。若父仓库的 submodule 指针变化，在交付中说明。
2. 大盘唯一来源：`market_analysis/output/market_report.json`。直接继承市场状态、派发日和板块信息；不得重新判断趋势，也不得仅因 Correction 淘汰合格候选。更新失败、文件缺失或 JSON 无效时停止，不得改用其他信源。
3. 候选池：`us/breakout_follow_pool.csv`，优先通过 `dashboard.data_utils.load_pool_csv` 加载。
4. CSV 必须用支持引号和转义的标准读取方式；`ibd_candidate_extra` 含带逗号 JSON，禁止按逗号手工切分。若解析列数与 Header 不一致，停止并报告“候选池解析失败”。
5. 核心 Header 为 `code`、`signal`、`ibd_candidate_rule`、`ibd_entry_status`、`current_vs_ibd_candidate_pct`、`ibd_entry_volume_ratio`、`ibd_entry_close_position`、`ibd_entry_breakout_range_ratio`、`sector`；缺少任一项时停止。其他评估字段缺失按 UNKNOWN/N/A 处理，不视为 CSV 解析失败。
6. 逐行规范化字段：字符串去除首尾空白，`ibd_entry_status` 转大写；`signal`、`pullback_v_is_dry` 接受布尔值、大小写不敏感的 `true/false` 或 `1/0`。无法识别的 `signal` 作为数据错误转人工补数；无法识别的 `pullback_v_is_dry` 记 UNKNOWN。参与比较的数值若为空、非数字、NaN 或正负 Infinity，均视为缺失，不得参与运算或排序。

## 判定状态

| 状态 | 含义 |
|---|---|
| PASS | 检查适用、字段存在、存在正式标准且满足 |
| FAIL | 检查适用、字段存在、存在正式标准但不满足 |
| CONTEXT | 路由字段存在，但无正式阈值；只展示数值，不作 PASS/FAIL |
| UNKNOWN | 检查适用，但路由要求的正式字段为空或不存在 |
| N/A | 检查不适用 |

CONTEXT、N/A 不参与 Major FAIL/UNKNOWN 比较。无 Critical FAIL 时，Critical UNKNOWN 不进入报告排序，转人工补数；Major FAIL 不超过 1 时，Major UNKNOWN 不进入“优先复核”，归入“值得留意”竞争。

## 候选范围与板块风控

- 候选集合仅包含 `signal == True` 且 `ibd_candidate_rule` 非空的行；其他行不参与预筛。
- 非 ACTIONABLE 不进入“优先复核”：Critical UNKNOWN 转人工补数；Critical FAIL，或 Critical 全部 PASS 但非 ACTIONABLE，才可作为代表列入“暂不优先”。
- 板块占比以候选集合中的 ACTIONABLE 行计算：`该板块 ACTIONABLE 数 / 全部 ACTIONABLE 数`；空 `sector` 统一记为 UNKNOWN 板块。
- 板块数量上限只作用于“优先复核”：单一板块最多 2 只；占比 `> 50%` 时最多 1 只并提示拥挤。ACTIONABLE 集合为空时不触发。“值得留意”不受硬上限，同质量时优先分散。

## #4 / #5 阶段字段路由

本节区分的是标的入池后的生命周期阶段，不是 IBD 底部形态分类，只约束 Checklist #4 Depth 与 #5 Duration，不约束 #6 缩量。

- 标的必须先由 Ceiling 突破进入候选池。`ceiling` / `ceiling_breakout` / `ceiling_pullback` 使用 `base_*` 字段衡量这次入池突破的原始基底。
- 后续规则使用 `pullback_*` 字段跟踪入池后的回撤/巩固；不得再用原始基底字段代替。标的是否跌破 Ceiling 并出池由上游候选池决定，Skill 不重新推导。

| `ibd_candidate_rule` | 阶段 | #4 / #5 唯一字段 |
|---|---|---|
| `ceiling`、`ceiling_breakout`、`ceiling_pullback` | 入池基底 | `base_depth_pct` / `base_duration_weeks` |
| 其他非空值（如 `pivot`、`ma10_touch_confirm`、`three_weeks_tight`） | 后续回撤 | `pullback_pct` / `pullback_duration_weeks` |
| 空或缺失 | UNKNOWN | 不读取另一组字段 |

严禁跨阶段替代、计算替代值或因字段缺失改读另一组字段。

- 路由字段存在时，#4/#5 记 CONTEXT，只展示对应深度和时长，供人工复核；字段缺失记 UNKNOWN。
- 不得把深度、时长或信号名解释成额外的形态分类，也不得创造“过深”“偏长”“太短”等未定义阈值。

## Breakout Geometry

只使用正式字段：

```text
pos = ibd_entry_close_position
rr = ibd_entry_breakout_range_ratio
trigger_pos = pos - rr
```

严格按顺序首次命中，禁止仅凭 `rr` 或主观观感升级分类：

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

Geometry 排序：`Full-range Breakout > Strong Finish > Faded Gap > Constructive Breakout > Marginal Breakout`。报告只使用这些英文名称。若有限数值已触发 `rr <= 0` 或 `pos < 0.65`，即使另一字段缺失仍为 Critical FAIL；其余 `pos` / `rr` 缺失或非有限数值情况为 Critical UNKNOWN。

## IBD 分层检查表

| 级别 | # | 检查点 | 唯一标准 |
|---|---:|---|---|
| Critical | 1 | 买点新鲜度 | 仅用 `current_vs_ibd_candidate_pct`；`0%～5%` PASS，`>5%` 或 `<0%` FAIL，`<=2%` 排序优先 |
| Critical | 2 | 突破日放量 | `ibd_entry_volume_ratio >= 1.5` |
| Critical | 3 | 突破日质量 | Geometry 非 Defensive Failure、非 Squat / Upper Shadow |
| Major | 4 | 阶段深度 | 严格按 #4/#5 路由；字段存在为 CONTEXT，缺失为 UNKNOWN |
| Major | 5 | 阶段时长 | 严格按 #4/#5 路由；字段存在为 CONTEXT，缺失为 UNKNOWN |
| Major | 6 | 巩固期缩量 | `pullback_v_is_dry == True` PASS，`False` FAIL；`ceiling` / `ceiling_breakout` 为 N/A，`ceiling_pullback` 与非空 Continuation 适用，缺失为 UNKNOWN |
| Minor | 7 | 紧贴 52 周高点 | `dist_to_52w_high_pct > -5.0` |
| Major | 8 | 基本面支撑 | `eps_yoy_growth >= 25` |
| Minor | 9 | 净筹码吸纳 | 仅使用上游正式字段；当前未导出时 N/A，不自行推导 |
| Minor | 10 | 周线量能跟进 | `volume_ratio >= 1.3` 时仅作正向加分；否则省略 |

补充语义：

- Critical 任一 FAIL 即淘汰出“优先复核”和“值得留意”，但可选代表列入“暂不优先”；只有 Critical 全部明确 PASS 才能进入前两组。
- 字段缺失不得写成 FAIL，UNKNOWN 不得写成 PASS。
- #6 独立于 #4/#5 路由：`ceiling_pullback` 的 #4/#5 仍读 Base 字段，但 #6 读取 `pullback_v_is_dry`。
- Minor 不作淘汰条件；周线量能低于 1.3 或缺失不得成为顾虑。
- `ibd_entry_close_vs_trigger_pct` 只作突破上下文，不参与 Geometry，也不得替代 `current_vs_ibd_candidate_pct`。

## 分组、排序与入选

按以下顺序首次命中并停止，确保每行只归入一组：

1. **人工补数**：`signal` 无法识别；排除在候选集合和报告排序之外。
2. **暂不优先**：候选存在任一 Critical FAIL；已知失败优先于同时存在的 UNKNOWN。
3. **人工补数**：无 Critical FAIL，但存在 Critical UNKNOWN；不与已确认突破候选混排。
4. **暂不优先**：Critical 全部 PASS，但不是 ACTIONABLE。
5. **暂不优先**：ACTIONABLE、Critical 全部 PASS，但 Major FAIL 至少 2 项。
6. **值得留意池**：ACTIONABLE、Critical 全部 PASS、Major FAIL 不超过 1，但存在 Major UNKNOWN；明确写出关键缺口。
7. **优先复核池**：ACTIONABLE、Critical 全部 PASS、Major FAIL 不超过 1，且 Major UNKNOWN 为 0。

Major FAIL 为 1 时，报告必须在“判断”或“顾虑”中写明该项；Major FAIL 至少 2 项时不得进入前两组。组内依次比较：更少 Major FAIL → 更少 Major UNKNOWN（CONTEXT、N/A 不计）→ Geometry 层级 → 新鲜区 `current_vs_ibd_candidate_pct <= 2%` → EPS（`>=25%` 优于 UNKNOWN，UNKNOWN 优于 `<25%`）→ Minor（#7 按 PASS、UNKNOWN、FAIL 排序，再看 #10 正向加分）。完全并列时按 `code` 字典序、再按 CSV 原始行序，保证同一快照结果稳定。最后按排序顺序逐只接收“优先复核”，触及板块上限时跳过并继续考察下一只；不为填满名额而降低质量。

## 执行流程

1. 更新并读取 Market Context。
2. 合规加载 CSV，建立 ACTIONABLE 板块占比。
3. 先执行 Critical 与 Geometry，再执行适用的 Major / Minor。
4. 按统一顺序排序并应用板块限制。
5. 输出极简中文报告；不外显 PASS/FAIL/CONTEXT/UNKNOWN/N/A、检查数量或内部计分。

## 输出格式

- 正文最多 30 个非空行；省略空区块和不影响结论的数据。
- **结论**：一句话说明最优先、值得留意及是否宁缺毋滥。
- **背景**：仅在市场状态或板块拥挤实际影响结论时写一句。
- **优先复核**：0～3 只，每只固定“突破日 / 优势 / 判断”3 行。
- **值得留意**：0～2 只，仅收录突破突出但结构、基本面或关键证据不完整者，每只固定“突破日 / 顾虑 / 判断”3 行。
- **暂不优先**：最多 3 只代表，每只仅写一个决定性原因；有亮点时先写亮点。
- 只引用正式字段和明确规则，不扩展字段含义、创造阈值或使用“完全共振”等超出证据的表述。

```markdown
# IBD 候选预筛

## 结论
[一句话结论]
[必要时写市场或板块背景]

## 优先复核
### [TICKER]
- **突破日：** [Geometry]｜收盘位置 [pos]｜突破幅度 [rr]｜量能 [entry volume]x
- **优势：** [最多两个带数字的理由]
- **判断：** [为何值得先人工 Review]

## 值得留意
### [TICKER]
- **突破日：** [Geometry]｜收盘位置 [pos]｜突破幅度 [rr]｜量能 [entry volume]x
- **顾虑：** [最多两个关键顾虑或数据缺口]
- **判断：** [为何观察但不列第一优先]

## 暂不优先
- **[TICKER]：** [亮点可选]，但 [一个决定性原因 + 数字]
```
