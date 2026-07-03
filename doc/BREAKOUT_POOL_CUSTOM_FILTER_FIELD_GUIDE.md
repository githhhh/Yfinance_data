# Breakout Pool 自定义筛选字段指南

本文基于策略主研发库白皮书 `quant_trade/strategy/doc/BREAKOUT_FOLLOW_POOL_SCHEMA.md`、真实 CSV 表头 `us/breakout_follow_pool.csv` 以及 `dashboard/field_config.py`，用于指导 pool 仪表板中用户进行自定义筛选操作。

## 一、筛选字段取舍原则

自定义筛选只保留能直接表达用户决策意图的字段：信号与形态路由、IBD 入场确认（日线检查）、当周线量价、趋势延伸&当前回撤深度、行业组别。以下字段不进入自定义筛选：

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

### 2. IBD 入场确认（日线检查）

**核心释义**：本组筛选专门针对**日线 (Daily Chart) 级别的买点检查**。与第三、第四部分的周线形态指标（如周量比、回踩 10 周线等）不同，`ibd_entry_*` 系列字段用于精确审计在**日线级别**突破发生当日的 K 线质量与量能表现，确保买点出现的一天具有真实的机构大单推进（放量、高收、大振幅）。

| 字段 | 类型 | 建议位置 | 保留原因 | 使用要点 |
|---|---|---|---|---|
| `ibd_entry_valid` | boolean | Core | 是否已在突破当红**日线**上完成价量有效确认 | 日线有效入场筛选的主开关 |
| `ibd_entry_volume_ratio` | number | Core | 突破当红**日线成交量**相对自身日均量的强度 | 通常与 `ibd_entry_valid=True` 联用，寻找日线爆发放量 |
| `ibd_entry_close_position` | number | Core | 突破当红**单日 K 线**收盘价在日内最高最低价区间内的相对位置（0~1） | 靠近 1 表示收盘处于日内高位（如 `>=0.5` 表示强势高收，过滤日线长上影疲软确认） |
| `ibd_entry_breakout_range_ratio` | number | Core | 突破当红**单日 K 线**振幅比率 | 衡量突破单日价格区间波动比率；振幅扩大且收盘在高位，反映日线强劲多头推升动能 |

### 3. 当周线量价

| 字段 | 类型 | 建议位置 | 保留原因 | 使用要点 |
|---|---|---|---|---|
| `volume_ratio` | number | Core | 当周量比，衡量周线动能 | 可与 `is_bullish=True` 组合替代 `is_priority` |
| `is_bullish` | boolean | Advanced | 当周收盘不低于开盘 | 与 `volume_ratio>=1.3` 组合即原 `is_priority` 语义 |

### 4. 趋势延伸&当前回撤深度

| 字段 | 类型 | 建议位置 | 保留原因 | 使用要点 |
|---|---|---|---|---|
| `touched_ema10_count` | number | Core | 突破后 10 周线回踩确认轮次 | 值越高通常阶段越成熟 |
| `pullback_pct` | number | Advanced | 最近/本轮回撤最大洗盘深度 | CSV 为负数；`<=-10` 表示至少回撤 10% |

### 5. 行业组别

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
| `ibd_entry_price` | 建议成交入场绝对价格，不具备形态质量或强弱筛选意义 | 筛选强弱使用 `ibd_entry_close_position` 和 `ibd_entry_breakout_range_ratio`；绝对价格仅作表格展示 |
| `ibd_entry_close_vs_trigger_pct` | 不能准确衡量突破日 K 线的视觉形态与动能强度（如收盘价位置与振幅） | 突破强弱筛选由 `ibd_entry_close_position` 与 `ibd_entry_breakout_range_ratio` 替代；该字段仅作表格展示或历史参考 |
| `ibd_entry_reject_reason` | 无效入场的具体驳回原因，仅作为诊断与解释文案 | 建议在表格详情或排查时查看，不进入条件筛选 |
| `ceiling` | 大型基底阻力/支撑绝对价位，不具备跨标的筛选意义 | 绝对价格仅在表格中作属性展示或计算基础 |
| `ceiling_date` | 大型基底构筑起始绝对日期，单独筛选无意义 | 与 `breakout_date` 配合计算“大型基底持续周数”，用于表格展示及后续筛查 |
| `pct_above_ceiling` | 距离大型基底天花板百分比，不能用来衡量买点过度延伸（延伸应针对触发价/枢轴点） | 建议从自定义条件筛选中排除，仅在表格中展示历史基底距离 |
| `pullback_v_is_dry` | 回撤窗口是否缩量属于启发式指标，预测与筛选有效性仍需验证 | 暂不作为条件筛选字段，仅在表格中展示原始字段供参考 |
| `base_depth_pct` | 与 `base_depth_abs` 等价但为负数，容易造成阈值方向误解 | 使用 `base_depth_abs` |
| `C_continuous` | 基于多字段百分位的综合打分，属于排序参考而非原子筛选条件 | 仅在 C Rank Reference Mode 展示 |
| `rank_C_continuous` | 由 `C_continuous` 和 tie-break 排序得到，属于排序结果 | 仅在 C Rank Reference Mode 展示 |
| `is_priority` | `volume_ratio>=1.3 AND is_bullish=True` 的复合便利字段 | 使用 `volume_ratio` + `is_bullish` 明确表达阈值 |
| `hold_return` | 突破后的持仓累计收益表现，属于事后跟踪指标而非入场前条件筛选项 | 仅在 Result Table 及监控报表中展示表现 |
| `mbox_count` | 中期箱体数量属于趋势阶段阶梯参考，非原子筛选条件 | 仅在表格列与详情中作阶段参考展示 |
| `base_depth_abs` | 基底洗盘深度属于结构参考，实际突破选股中洗盘过滤由回调深度 `pullback_pct` 承担 | 仅作表格属性展示 |
| `base_mbox_count` | 基底内换手阶梯参考，不进入条件筛选漏斗 | 仅在表格与详情中展示 |
| `pullback_count` | 突破后显著回调轮次属于波段修盘跟踪指标 | 仅在表格列中作参考展示 |
| `pullback_pct_off_peak` | 距回撤高点的乖离度，属于修盘进度跟踪 | 仅在表格与图表中展示 |
| `base_duration_weeks` | 衡量基底构筑标准化跨度，已开箱写入 CSV，为极致精简漏斗不强制进入常规筛选项 | 重点作为表格中 `Base Duration` 列展示；进阶排查可用于自定义排序 |

## 四、实现建议

若后续调整 dashboard 字段配置，建议将 Advanced Field Filters 的候选字段严格遵循本文“建议保留的自定义筛选字段”，做到精简和原子化，避免过度设计。

`code` 仍应固定在表格左侧并参与 quick filter，但不作为条件筛选字段。`ibd_entry_reject_reason`、`ibd_entry_date`、`ibd_entry_price`、`ibd_entry_close_vs_trigger_pct`、`hold_return`、`ceiling`、`ceiling_date`、`pct_above_ceiling`、`pullback_v_is_dry`、`signal`、`signal_source` 等均仅作表格列展示或图表参考，不进入交互条件筛选漏斗。对于 `base_duration_weeks`（基底周数），直接读取 CSV 中的物理字段在表格中紧凑呈现（如 `24w`）。
