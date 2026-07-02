# Breakout Pool 自定义筛选字段指南

本文基于 `doc/BREAKOUT_FOLLOW_POOL_SCHEMA.md`、真实 CSV 表头 `us/breakout_follow_pool.csv` 以及 `dashboard/field_config.py`，用于指导 pool 仪表板中用户进行自定义筛选操作。

## 一、筛选字段取舍原则

自定义筛选只保留能直接表达用户决策意图的字段：信号类型、IBD 入场确认、量价强度、结构风险、生命周期、行业分组。以下字段不进入自定义筛选：

1. 标识或批次元数据字段，例如 `code`、`snapshot_date`。
2. 可由其它字段组合或被更大粒度字段完全包含的字段，例如 `signal`、`signal_source`、`is_priority`、`C_continuous`。
3. 与其它字段重复、仅代表绝对价格/日期或只用于审计诊断的字段，例如 `ibd_candidate_signal_source`、`ibd_trigger_price`、`ibd_entry_rule`、`ibd_entry_date`、`ibd_entry_price`、`ibd_entry_reject_reason`、`ceiling`、`ceiling_date`。
4. 长 JSON 或底层诊断上下文字段，例如 `ibd_candidate_extra`。
5. 正负号容易误导且已有等价字段替代的字段，例如 `base_depth_pct`。
6. 易误导、逻辑不适合筛选或有效性有待验证的指标，例如 `pct_above_ceiling`（不能用于衡量日线买点延伸度）、`pullback_v_is_dry`（缩量回调有效性仍需验证）。

用户筛选表达式统一按 AND 组合。单个类别字段内部可使用 `in` / `not in` 表达多选，相当于字段内部 OR。

```text
Preset/Core Filters AND Advanced Field Filters
```

未启用字段不参与过滤。

## 二、建议保留的自定义筛选字段

### 1. 信号与形态路由

| 字段 | 类型 | 建议位置 | 保留原因 | 使用要点 |
|---|---|---|---|---|
| `ibd_candidate_rule` | category | Core | 代表底层 Resolver 的形态验证路由与细分生命周期 | 粒度最细且唯一映射上游叙事，完全替代 `signal_source`；不含 `m_breakout`（属于独立双底策略） |

### 2. IBD 入场确认

| 字段 | 类型 | 建议位置 | 保留原因 | 使用要点 |
|---|---|---|---|---|
| `ibd_entry_valid` | boolean | Core | 是否已完成日线价量确认 | 有效入场筛选的主开关 |
| `ibd_entry_volume_ratio` | number | Core | 确认日成交量强度 | 通常与 `ibd_entry_valid=True` 联用 |
| `ibd_entry_close_vs_trigger_pct` | number | Core | 确认日收盘相对触发价的强弱/延伸度 | CSV 中为小数比例，`0.05` 表示 5%；不能替代 `ibd_entry_valid` |

### 3. 周线量价与当周质量

| 字段 | 类型 | 建议位置 | 保留原因 | 使用要点 |
|---|---|---|---|---|
| `volume_ratio` | number | Core | 当周量比，衡量周线动能 | 可与 `is_bullish=True` 组合替代 `is_priority` |
| `is_bullish` | boolean | Advanced | 当周收盘不低于开盘 | 与 `volume_ratio>=1.3` 组合即原 `is_priority` 语义 |
| `hold_return` | number | Advanced | 自突破起算以来持仓收益 | CSV 为百分比点，`12.4` 表示 12.4% |

### 4. 结构位置与风险

| 字段 | 类型 | 建议位置 | 保留原因 | 使用要点 |
|---|---|---|---|---|
| `touched_ema10_count` | number | Core | 突破后 10 周线回踩确认轮次 | 值越高通常阶段越成熟 |
| `mbox_count` | number | Advanced | 突破后新突破的中期箱体数量 | 用于识别趋势阶梯数量 |
| `base_depth_abs` | number | Advanced | 基底洗盘深度的正数表达 | 优先使用该字段，不使用 `base_depth_pct` |
| `base_mbox_count` | number | Advanced | 基底内 M_BOX 数量 | 衡量基底内部换手充分度 |
| `pullback_count` | number | Advanced | 突破后显著回调轮次 | 过滤首波、二波或多轮调档状态 |
| `pullback_pct` | number | Advanced | 最近/本轮回撤最大洗盘深度 | CSV 为负数；`<=-10` 表示至少回撤 10% |
| `pullback_pct_off_peak` | number | Advanced | 当前收盘距本轮回撤高点的乖离 | `<0` 仍在修复，`>=0` 已越过波段高点 |

### 5. 分组字段

| 字段 | 类型 | 建议位置 | 保留原因 | 使用要点 |
|---|---|---|---|---|
| `sector` | category | Core/Advanced | GICS 板块分组 | 用于行业集中度和板块偏好 |
| `industry` | category | Core/Advanced | GICS 细分行业 | 用于进一步收窄行业主题 |

## 三、建议排除的自定义筛选字段

| 字段 | 排除原因 | 替代方式或保留用途 |
|---|---|---|
| `code` | 股票代码是行标识，不是策略条件；用户按代码查找属于搜索/quick filter 行为 | 表格固定列、quick filter、导出保留 |
| `snapshot_date` | 同一批 pool 通常共用生成日期，筛选后没有选股信息增量 | 展示数据批次或导出审计 |
| `signal` | 五大周线触发条件的 OR 复合字段 | preset/core 可作为基础股票池开关；自定义形态筛选直接用 `ibd_candidate_rule` |
| `signal_source` | 当周主信号叙事标签，已被粒度更细的 `ibd_candidate_rule` 完全包含与指代 | 建议从条件筛选中移除，仅在表格中展示主信号标签 |
| `ibd_candidate_signal_source` | 直接继承 `signal_source`，语义重复 | 仅用于底层诊断与排查 |
| `ibd_candidate_extra` | JSON 上下文，结构复杂且不适合 UI 条件筛选 | 详情展开、原始导出、开发排查 |
| `ibd_trigger_price` | 审计留存的触发价副本，绝对价格无筛选意义 | 仅在表格中作属性展示或审计 |
| `ibd_entry_rule` | 入场成功时回填 `ibd_candidate_rule`，无效时为空，筛选会混淆候选与确认 | 使用 `ibd_candidate_rule` + `ibd_entry_valid` |
| `ibd_entry_date` | 真实确认发生的绝对日期，筛选意义弱 | 建议仅在表格展示或作为基础审计字段 |
| `ibd_entry_price` | 建议成交入场绝对价格，不具备形态质量或强弱筛选意义 | 筛选强弱使用 `ibd_entry_close_vs_trigger_pct`；绝对价格仅作表格展示 |
| `ibd_entry_reject_reason` | 无效入场的具体驳回原因，仅作为诊断与解释文案 | 建议在表格详情或排查时查看，不进入条件筛选 |
| `ceiling` | 大型基底阻力/支撑绝对价位，不具备跨标的筛选意义 | 绝对价格仅在表格中作属性展示或计算基础 |
| `ceiling_date` | 大型基底构筑起始绝对日期，单独筛选无意义 | 与 `breakout_date` 配合计算“大型基底持续周数”，用于表格展示及后续筛查 |
| `pct_above_ceiling` | 距离大型基底天花板百分比，不能用来衡量买点过度延伸（延伸应针对触发价/枢轴点） | 建议从自定义条件筛选中排除，仅在表格中展示历史基底距离 |
| `pullback_v_is_dry` | 回撤窗口是否缩量属于启发式指标，预测与筛选有效性仍需验证 | 暂不作为条件筛选字段，仅在表格中展示原始字段供参考 |
| `base_depth_pct` | 与 `base_depth_abs` 等价但为负数，容易造成阈值方向误解 | 使用 `base_depth_abs` |
| `C_continuous` | 基于多字段百分位的综合打分，属于排序参考而非原子筛选条件 | 仅在 C Rank Reference Mode 展示 |
| `rank_C_continuous` | 由 `C_continuous` 和 tie-break 排序得到，属于排序结果 | 仅在 C Rank Reference Mode 展示 |
| `is_priority` | `volume_ratio>=1.3 AND is_bullish=True` 的复合便利字段 | 使用 `volume_ratio` + `is_bullish` 明确表达阈值 |

## 四、大型基底持续时间（周计数）衍生计算与表格渲染

`ceiling_date` 与 `breakout_date` 分别代表大型基底构筑的起始日期与首次越过天花板的突破日期。这两者的绝对历史时间差不能直接用于条件筛选，但可以用来精确计算**大型基底结构持续时间**。由于策略的核心分析周期为周线，我们以**周数 (`Weeks`)** 为单位进行标准化度量：

$$\text{base\_duration\_weeks} = \text{round}\left(\frac{\text{breakout\_date} - \text{ceiling\_date}}{7}\right)$$

### 1. 理论依据与业务意义
在 CAN SLIM 与 IBD 交易体系中，基底构筑的持续时间是衡量机构多头主力吸筹、洗盘充分度与筹码沉淀厚度的绝对关键指标（如标准杯柄或水平基底通常至少需巩固 7 周以上，大型 Stage 2 水平基底往往巩固数月至数年，底蕴越深厚，突破后的主升浪潜能越大）。

### 2. 表格展示与前端渲染指导
1. **优先作为表格专用展示列 (Table Display Column)**：
   * 在 Result Table 中将列名设为 **`Base Duration`** 或 **`基底周数`**。
   * 前端渲染时，强烈建议采用**简明紧凑格式**（例如 `24w`、`15 Weeks`，或附带颜色梯度的 Badge 徽章）进行渲染。这能让盯盘交易者在快速扫描表格时，瞬间直观感受到标的底蕴深浅与横盘跨度。
2. **后续评估加入自定义筛选**：
   * 现阶段优先满足前端表格的视觉渲染与盯盘感知；随着后续实战对基底跨度筛选需求的明确（如筛选寻找巩固 `>= 15 周` 的深厚基底），可作为独立的数值型筛选字段（`base_duration_weeks`）正式加入 Advanced Field Filters 体系。

## 五、实现建议

若后续调整 dashboard 字段配置，建议将 Advanced Field Filters 的候选字段改为本文“建议保留的自定义筛选字段”，而不是简单使用所有 Custom Mode 可显示字段。

`code` 仍应固定在表格左侧并参与 quick filter，但不作为 Advanced Field Filters 字段。`ibd_entry_reject_reason`、`ibd_entry_date`、`ibd_entry_price`、`ceiling`、`ceiling_date`、`pct_above_ceiling`、`pullback_v_is_dry`、`signal`、`signal_source` 建议仅作为表格列展示或作为衍生字段计算的基础，不进入条件筛选表达式。对于由 `ceiling_date` 和 `breakout_date` 衍生计算所得的基底周数字段，优先在表格中渲染展现。
