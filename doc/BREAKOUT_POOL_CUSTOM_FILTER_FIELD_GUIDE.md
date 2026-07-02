# Breakout Pool 自定义筛选字段指南

本文基于 `doc/BREAKOUT_FOLLOW_POOL_SCHEMA.md`、真实 CSV 表头 `us/breakout_follow_pool.csv` 以及 `dashboard/field_config.py`，用于指导 pool 仪表板中用户进行自定义筛选操作。

## 一、筛选字段取舍原则

自定义筛选只保留能直接表达用户决策意图的字段：信号类型、IBD 入场确认、量价强度、结构风险、生命周期、行业分组。以下字段不进入自定义筛选：

1. 标识或批次元数据字段，例如 `code`、`snapshot_date`。
2. 可由其它字段组合得到的复合字段，例如 `signal`、`is_priority`、`C_continuous`。
3. 与其它字段重复或只用于审计回放的字段，例如 `ibd_candidate_signal_source`、`ibd_trigger_price`、`ibd_entry_rule`。
4. 长 JSON 或底层诊断上下文字段，例如 `ibd_candidate_extra`。
5. 正负号容易误导且已有等价字段替代的字段，例如 `base_depth_pct`。

用户筛选表达式统一按 AND 组合。单个类别字段内部可使用 `in` / `not in` 表达多选，相当于字段内部 OR。

```text
Preset/Core Filters AND Advanced Field Filters
```

未启用字段不参与过滤。

## 二、建议保留的自定义筛选字段

### 1. 信号与形态路由

| 字段 | 类型 | 建议位置 | 保留原因 | 使用要点 |
|---|---|---|---|---|
| `signal_source` | category | Core | 代表当周唯一主信号叙事 | 与 `ibd_candidate_rule` 联用区分具体形态阶段 |
| `ibd_candidate_rule` | category | Core | 代表日线 Resolver 的实际验证路由 | `ceiling_breakout` 下必须用它区分 `ceiling` 与 `ceiling_pullback` |

### 2. IBD 入场确认

| 字段 | 类型 | 建议位置 | 保留原因 | 使用要点 |
|---|---|---|---|---|
| `ibd_entry_valid` | boolean | Core | 是否已完成日线价量确认 | 有效入场筛选的主开关 |
| `ibd_entry_date` | date | Advanced | 入场确认发生日期 | 适合筛选近期确认 |
| `ibd_entry_price` | number | Advanced | 实际建议成交入场价 | 可用于价格区间偏好 |
| `ibd_entry_volume_ratio` | number | Core | 确认日成交量强度 | 通常与 `ibd_entry_valid=True` 联用 |
| `ibd_entry_close_vs_trigger_pct` | number | Core | 确认日收盘相对触发价的强弱/延伸度 | CSV 中为小数比例，`0.05` 表示 5%；不能替代 `ibd_entry_valid` |
| `ibd_entry_reject_reason` | category | Advanced | 无效入场的具体驳回原因 | 仅在 `ibd_entry_valid=False` 时有解释价值 |

### 3. 周线量价与当周质量

| 字段 | 类型 | 建议位置 | 保留原因 | 使用要点 |
|---|---|---|---|---|
| `volume_ratio` | number | Core | 当周量比，衡量周线动能 | 可与 `is_bullish=True` 组合替代 `is_priority` |
| `is_bullish` | boolean | Advanced | 当周收盘不低于开盘 | 与 `volume_ratio>=1.3` 组合即原 `is_priority` 语义 |
| `pullback_v_is_dry` | boolean | Core | 回撤窗口是否缩量健康 | 常用于过滤健康调档 |
| `hold_return` | number | Advanced | 自突破起算以来持仓收益 | CSV 为百分比点，`12.4` 表示 12.4% |

### 4. 结构位置与风险

| 字段 | 类型 | 建议位置 | 保留原因 | 使用要点 |
|---|---|---|---|---|
| `pct_above_ceiling` | number | Core | 当前距离大型基底天花板的延伸度 | CSV 为百分比点；常用 `<=10` 过滤不过度延伸 |
| `touched_ema10_count` | number | Core | 突破后 10 周线回踩确认轮次 | 值越高通常阶段越成熟 |
| `mbox_count` | number | Advanced | 突破后新突破的中期箱体数量 | 用于识别趋势阶梯数量 |
| `ceiling` | number | Advanced | 大型基底天花板阻力/支撑价位 | 绝对价格筛选，业务含义弱于百分比字段 |
| `ceiling_date` | date | Advanced | 大型基底构筑起始日期 | 可用于筛选基底时间跨度 |
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
| `signal` | 五大周线触发条件的 OR 复合字段 | preset/core 可作为基础股票池开关；自定义形态筛选用 `signal_source` + `ibd_candidate_rule` |
| `ibd_candidate_signal_source` | 直接继承 `signal_source`，语义重复 | 使用 `signal_source` |
| `ibd_candidate_extra` | JSON 上下文，结构复杂且不适合 UI 条件筛选 | 详情展开、原始导出、开发排查 |
| `ibd_trigger_price` | 审计留存的触发价副本，与 `ibd_candidate_price` 重复 | 如需价格筛选使用 `ibd_candidate_price` 或 `ibd_entry_price` |
| `ibd_entry_rule` | 入场成功时回填 `ibd_candidate_rule`，无效时为空，筛选会混淆候选与确认 | 使用 `ibd_candidate_rule` + `ibd_entry_valid` |
| `base_depth_pct` | 与 `base_depth_abs` 等价但为负数，容易造成阈值方向误解 | 使用 `base_depth_abs` |
| `C_continuous` | 基于多字段百分位的综合打分，属于排序参考而非原子筛选条件 | 仅在 C Rank Reference Mode 展示 |
| `rank_C_continuous` | 由 `C_continuous` 和 tie-break 排序得到，属于排序结果 | 仅在 C Rank Reference Mode 展示 |
| `is_priority` | `volume_ratio>=1.3 AND is_bullish=True` 的复合便利字段 | 使用 `volume_ratio` + `is_bullish` 明确表达阈值 |

## 四、字段逻辑关系

### 1. 周线主信号与候选路由

`signal_source` 是策略层主叙事标签；同周多个信号并发时按白皮书优先级只保留一个主标签：

```text
ceiling_breakout > pivot > three_weeks_tight_breakout > 10_wk_ema_touch_confirm > ""
```

`ibd_candidate_rule` 是日线 Resolver 的验证路由。两者不是简单一一对应，尤其 `ceiling_breakout` 包含两个阶段：

| 用户意图 | 推荐筛选表达式 | 业务含义 |
|---|---|---|
| 初始天花板突破 | `signal_source=ceiling_breakout AND ibd_candidate_rule=ceiling` | 首次越过大型基底顶部平台 |
| 天花板回踩确认 | `signal_source=ceiling_breakout AND ibd_candidate_rule=ceiling_pullback` | 突破后在平台上方回踩并反弹 |
| Pivot 突破 | `signal_source=pivot AND ibd_candidate_rule=pivot` | 中/小型箱体阻力突破 |
| 三周紧缩再启动 | `signal_source=three_weeks_tight_breakout AND ibd_candidate_rule=three_weeks_tight` | 至少三周紧缩后向上突破 |
| 10 周线回踩反弹 | `signal_source=10_wk_ema_touch_confirm AND ibd_candidate_rule=ma10_touch_confirm` | 触及 10 周线缓冲带后反弹确认 |

### 2. 候选触发价与日线确认

`ibd_candidate_rule` 与 `ibd_candidate_price` 共同定义日线确认候选。`ibd_entry_valid=True` 表示日线 Resolver 已经同时满足：

```text
Daily_High > ibd_candidate_price
AND Daily_Close > ibd_candidate_price
AND Daily_Volume / SMA(Daily_Volume, 50) >= 1.5
```

因此：

1. 查找可操作确认标的时，先用 `ibd_entry_valid=True`。
2. 比较确认质量时，再加 `ibd_entry_volume_ratio`、`ibd_entry_close_vs_trigger_pct`。
3. 查找无效原因时，用 `ibd_entry_valid=False AND ibd_entry_reject_reason in (...)`。
4. 不要用 `ibd_entry_close_vs_trigger_pct>0` 代替 `ibd_entry_valid=True`，该字段只诊断确认日收盘强弱。

### 3. 结构风险与量价质量

结构字段建议按目标组合，而不是单字段孤立使用：

| 目标 | 推荐组合 | 含义 |
|---|---|---|
| 不追高的有效突破 | `ibd_entry_valid=True AND pct_above_ceiling<=10 AND ibd_entry_close_vs_trigger_pct BETWEEN 0 AND 0.05` | 已确认且距离结构位不过度延伸 |
| 强量确认 | `ibd_entry_valid=True AND ibd_entry_volume_ratio>=1.5` | 日线确认量能达到 IBD 契约下限或更强 |
| 周线动能优先 | `volume_ratio>=1.3 AND is_bullish=True` | 显式表达原 `is_priority` 语义 |
| 健康回调 | `pullback_v_is_dry=True AND pullback_pct BETWEEN -15 AND -3` | 回撤有洗盘且未深度破坏结构 |
| 首波/早期形态 | `pullback_count<=1 AND touched_ema10_count<=1` | 突破后经历的调档和均线回踩较少 |
| 基底充分 | `base_depth_abs>=30 AND base_mbox_count>=1` | 有较充分洗盘与基底内部换手 |

### 4. 时间字段边界

`snapshot_date` 是数据批次，不用于用户自定义筛选。其余日期字段有不同业务含义：

| 字段 | 业务含义 | 典型用法 |
|---|---|---|
| `breakout_date` | 首次站上 ceiling 的起算周 | 筛选近期突破或老突破延续 |
| `ibd_entry_date` | 日线真实确认日期 | 筛选近期可操作入场 |
| `ceiling_date` | 大型基底构筑起始日期 | 分析基底年龄 |

## 五、实现建议

若后续调整 dashboard 字段配置，建议将 Advanced Field Filters 的候选字段改为本文“建议保留的自定义筛选字段”，而不是简单使用所有 Custom Mode 可显示字段。

`code` 仍应固定在表格左侧并参与 quick filter，但不作为 Advanced Field Filters 字段。`ibd_entry_reject_reason` 建议按 category 枚举渲染，而不是普通长文本。`signal` 可保留在 preset/core base universe 中，但不建议进入 Advanced Field Filters。
