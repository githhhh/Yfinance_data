# Breakout Pool：IBD Review 状态漏斗重构实施规格

> 交付对象：Gemini Pro  
> 本文是本轮唯一实施范围。不要自行扩展需求。

## 1. 本轮目标

把 Dashboard 从“字段分组筛选器”改成围绕 IBD 入场生命周期的 Review 工具，明确区分：

- 尚未完成日线确认；
- 已确认且当前仍在买区；
- 已确认但已经延伸；
- 确认后跌回触发价下方。

本轮不改变策略信号、IBD Resolver、C Rank 或 Pattern 算法。

## 2. 必须遵守的范围

### 2.1 本轮必须完成

1. Pool 新增三个物理字段：
   - `latest_close`
   - `current_vs_candidate_pct`
   - `ibd_entry_status`
2. Dashboard 改为 IBD 状态漏斗。
3. Route 保留为分类筛选。
4. Pattern、日线量比、周线量比保留为可选质量筛选。
5. Structure、Sector、Industry 从筛选区移除。
6. Result Table 提前到图表之前。
7. 增加真实 CSV 自测和边界案例测试。
8. 同步 Schema、筛选字段指南和 Dashboard 设计文档。

### 2.2 本轮禁止扩展

- 不修改 Custom Mode 当前继承的 CSV/C Rank 顺序。
- 不调整 `C_continuous`、`rank_C_continuous`、`is_priority`。
- 不做 Structure 按 Route 联动。
- 不新增通用 Advanced Filters。
- 不修改五类 `breakout_pattern` 的计算规则。
- 不重做图表，只调整图表位置。
- 不修复 52 周高点派生字段。
- 不增加每日任务、通知或实时行情。
- 不引入新的复杂状态或综合评分。

如发现以上问题，只记录，不在本轮顺手修改。

### 2.3 更新频率不变

- 周三收盘生成周中 `Provisional` 快照。
- 周末收盘生成正式 `Final` 快照。
- 三个新字段都描述当前快照，不代表实时盘中状态。
- 本轮不增加每日扫描；Dashboard 定位为 Review 工具，不是实时入场提醒。

## 3. 为什么改成状态漏斗

当前 `ibd_entry_valid=True` 只表示信号周内某天曾满足日线突破、收盘站稳和成交量确认，不表示当前仍可买。

典型差异：

| 股票 | 当前 Pool 含义 | 正确 Review 状态 |
|---|---|---|
| PHVS | IBD Valid，距候选买点约 +1.9% | `ACTIONABLE` |
| TRAX | IBD Valid，距候选买点约 +45% | `EXTENDED` |
| PENG | IBD Valid，距候选买点约 +14.7% | `EXTENDED` |
| DELL | 位于早期 Pivot 买区附近，但日线量能未确认 | `UNCONFIRMED` |

因此真正的决策链只有：

```text
Active Signal
  ├─ UNCONFIRMED
  └─ IBD Confirmed
       ├─ ACTIONABLE
       ├─ EXTENDED
       └─ BELOW_TRIGGER
```

Route 是分类；Pattern 和量比是质量筛选。它们不是与 Entry Status 同级的生命周期状态。

## 4. 新字段契约

三个字段必须由 Pool 生成层写入 CSV。Dashboard 只读取和筛选，不自行用 `ceiling` 反推价格。

### 4.1 `latest_close`

| 属性 | 规定 |
|---|---|
| 类型 | `float` |
| 含义 | `snapshot_date` 对应最新一根周线的收盘价 |
| 来源 | 生成本期 Pool 时使用的最新周线 `Close` |

禁止通过下面的公式反推：

```python
ceiling * (1 + pct_above_ceiling / 100)
```

### 4.2 `current_vs_candidate_pct`

| 属性 | 规定 |
|---|---|
| 类型 | `float` |
| 单位 | 百分点，例如 `1.36` 表示 `+1.36%` |
| 含义 | 最新收盘价相对本次日线候选触发价的位置 |

唯一计算公式：

```python
current_vs_candidate_pct = (
    latest_close / ibd_candidate_price - 1
) * 100
```

#### 为什么不用 `ibd_trigger_price`

当前真实 Pool 中，83 只 Signal 都有 `ibd_candidate_price`，但 `ibd_trigger_price` 只在 21 只 IBD Valid 行中有值。若使用 `ibd_trigger_price`，62 只未确认候选无法计算当前位置。

Schema 已规定 `ibd_candidate_price` 是日线 Resolver 必须跨越的真实 Trigger Price。因此本字段必须使用 `ibd_candidate_price`，不要增加 fallback。

当 `latest_close` 或 `ibd_candidate_price` 缺失、非有限数或候选价 `<= 0` 时，本字段为 `NaN`。

### 4.3 `ibd_entry_status`

| 属性 | 规定 |
|---|---|
| 类型 | `str` |
| 允许值 | `UNCONFIRMED`、`ACTIONABLE`、`EXTENDED`、`BELOW_TRIGGER` |

唯一判定逻辑：

```python
if signal is not True:
    ibd_entry_status = None
elif current_vs_candidate_pct is invalid:
    ibd_entry_status = None
elif ibd_entry_valid is not True:
    ibd_entry_status = "UNCONFIRMED"
elif current_vs_candidate_pct < 0:
    ibd_entry_status = "BELOW_TRIGGER"
elif current_vs_candidate_pct <= 5:
    ibd_entry_status = "ACTIONABLE"
else:
    ibd_entry_status = "EXTENDED"
```

重要规则：

- `ibd_entry_valid=False` 时，无论价格在触发价上方还是下方，都属于 `UNCONFIRMED`。
- `ACTIONABLE` 必须同时满足 `ibd_entry_valid=True` 和 `0 <= current_vs_candidate_pct <= 5`。
- `EXTENDED` 只表示不适合按当前买点追入，不表示股票质量差。
- 本轮不细分 `5%～10%`、`10%～15%` 等延伸区间。
- 不新增 `DATA_INVALID` 状态；数据异常保持空值并由自测报错。

## 5. Dashboard 最终 Review 流程

### 5.1 固定基础池

Custom Mode 始终从以下条件开始：

```text
signal=True
```

该条件不可关闭，但必须在页面上显式显示，不能成为隐藏条件。

### 5.2 Route

只保留单选：

```text
All
ceiling
ceiling_pullback
pivot
ma10_touch_confirm
three_weeks_tight
```

字段使用 `ibd_candidate_rule`。Route 只是分类，不代表优劣。

### 5.3 Entry Status

Route 下方显示状态按钮或单选：

```text
All
UNCONFIRMED
ACTIONABLE
EXTENDED
BELOW_TRIGGER
```

每个状态显示数量。数量基于：

```text
signal=True + 当前 Route
```

计算状态数量时，不能提前应用 Pattern、日线量比或周线量比筛选。

`All` 为默认值，不能默认只显示 `ACTIONABLE`，否则会隐藏 DELL 这类等待量能确认的候选。

### 5.4 Optional Quality Filters

只保留三个可选条件：

1. `breakout_pattern`
2. `ibd_entry_volume_ratio`
3. `volume_ratio`

默认均为 All/全范围，不主动减少行数。

规则：

- Status 为 `UNCONFIRMED` 时，禁用 Pattern 和 `ibd_entry_volume_ratio`，因为没有有效突破日数据。
- Status 为 `ACTIONABLE`、`EXTENDED` 或 `BELOW_TRIGGER` 时才启用日线质量条件。
- Status 为 `All` 时不应用日线质量默认条件；避免自动排除未确认候选。
- `volume_ratio` 可用于任何状态，但默认不设置 `>=1.3`。
- 本轮不提供 `current_vs_candidate_pct` 独立滑块；它用于状态计算和表格展示。

### 5.5 不再出现在筛选区的字段

以下字段继续保留在 CSV、表格和导出中，但不提供筛选控件：

```text
touched_ema10_count
pullback_pct
pullback_count
base_depth_pct
base_duration_weeks
sector
industry
is_bullish
```

## 6. 页面布局

必须按下面顺序渲染：

```text
1. Snapshot / Active Signal 信息
2. Route
3. Entry Status 数量与选择
4. Optional Quality Filters
5. Current Filters 摘要与最终行数
6. Result Table
7. KPI / Charts
8. Download CSV
```

图表移到 Result Table 后面；本轮不改图表聚合逻辑。可以使用 expander 折叠图表，但不是强制要求。

页面应明确显示：

```text
Total Pool: 717
Active Signal: 83
Route: All
Status: All
Final Rows: 83
```

选择 Route 后，状态数量随 Route 更新。选择 Status 和质量条件后，只更新 Final Rows，不重新定义状态含义。

## 7. 默认表格视图

新增并默认选择 `IBD Decision` 视图，列顺序固定为：

```text
code
signal_source
ibd_candidate_rule
ibd_entry_status
latest_close
ibd_candidate_price
current_vs_candidate_pct
ibd_entry_date
ibd_entry_price
ibd_entry_volume_ratio
breakout_pattern
volume_ratio
ibd_entry_reject_reason
```

要求：

- `code` 继续 Pin 在左侧。
- `current_vs_candidate_pct` 显示为百分比，例如 `+1.36%`。
- `ibd_entry_reject_reason` 对 `UNCONFIRMED` 必须可见。
- 原 `All Fields` 和其它 Column View 保留。
- 本轮不修改默认行排序逻辑。

## 8. 实现文件与职责

### 8.1 Pool 生成层

定位实际生成 `breakout_follow_pool.csv` 的代码，不要猜测新文件名。

完成：

- 从本期最新周线读取 `latest_close`；
- 计算 `current_vs_candidate_pct`；
- 计算 `ibd_entry_status`；
- 将三个字段写入最终 CSV；
- 不修改现有 IBD Resolver 和排名公式。

### 8.2 `dashboard/field_config.py`

- 注册三个新字段及格式。
- 新增 `IBD Decision` 列视图。
- 将筛选白名单收敛为 Route、Status 和三个 Optional Quality 字段。
- Structure、Grouping 字段继续保留为表格字段。
- 不修改 C Rank 相关配置。

### 8.3 `dashboard/data_utils.py`

- Normalize 三个新字段。
- 新增构建 Route 下状态数量的纯函数。
- 状态数量与最终筛选必须是两个步骤，禁止用最终过滤结果反算状态数量。
- `apply_filters()` 通用实现保持不变，除非新字符串字段确实需要最小适配。

建议纯函数接口：

```python
def build_entry_status_counts(signal_df: pd.DataFrame) -> dict[str, int]:
    ...
```

返回至少包含：

```text
ALL
UNCONFIRMED
ACTIONABLE
EXTENDED
BELOW_TRIGGER
```

### 8.4 `dashboard/app.py`

- 重排页面顺序。
- 删除 Structure、Grouping 筛选 UI。
- 用 Entry Status 替换 `IBD Entry Valid` 主控件。
- 保留 Route 和三个 Optional Quality 控件。
- 实现状态数量展示及控件禁用规则。
- 表格默认切到 `IBD Decision`。
- 图表移到表格后。

### 8.5 测试与自测

至少更新：

```text
dashboard/tests/test_filters.py
dashboard/tests/test_table_config.py
dashboard/tests/test_app_static.py
dashboard/self_check.py
```

不要只做源码字符串存在性断言；核心状态必须使用 DataFrame 数据测试。

## 9. 必须通过的业务测试

### 9.1 状态边界

构造最小 DataFrame 验证：

| valid | current_vs_candidate_pct | 预期状态 |
|---|---:|---|
| False | +2.0 | `UNCONFIRMED` |
| True | 0.0 | `ACTIONABLE` |
| True | +5.0 | `ACTIONABLE` |
| True | +5.01 | `EXTENDED` |
| True | -0.01 | `BELOW_TRIGGER` |
| True | NaN | 空值/数据错误 |

### 9.2 状态互斥与守恒

对任一正常 Signal Pool：

```text
UNCONFIRMED
+ ACTIONABLE
+ EXTENDED
+ BELOW_TRIGGER
= signal=True 行数
```

任何 Signal 行状态为空时，`self_check.py` 必须失败。

### 9.3 2026-07-10 真实 CSV 基线

使用本次提供的完整 Pool，生成新字段后必须得到：

```text
Total Pool       717
Active Signal     83
UNCONFIRMED       62
ACTIONABLE        10
EXTENDED          11
BELOW_TRIGGER      0
```

边界股票预期：

```text
PHVS -> ACTIONABLE
TRAX -> EXTENDED
PENG -> EXTENDED
DELL -> UNCONFIRMED
```

DELL 在该快照中约高于系统 Pivot 候选价 1.3%，但因 `daily_volume_not_confirmed` 必须保持 `UNCONFIRMED`，不能因为价格在候选价上方而进入 `ACTIONABLE`。

### 9.4 筛选行为

- Route=All、Status=All、质量条件默认时返回全部 83 只 Signal。
- Status=ACTIONABLE 返回 10 行。
- Status=UNCONFIRMED 时 Pattern 和 IBD Entry Volume 不生效。
- 选择 Pattern 不得改变 `ibd_entry_status` 原值。
- 周量比筛选只改变 Final Rows，不改变状态计数。
- C Rank Reference Mode 行为与本轮修改前完全一致。

## 10. 文档同步

### `BREAKOUT_FOLLOW_POOL_SCHEMA.md`

增加三个字段的类型、公式、状态枚举和边界规则。

### `BREAKOUT_POOL_CUSTOM_FILTER_FIELD_GUIDE.md`

筛选字段收敛为：

```text
ibd_candidate_rule
ibd_entry_status
breakout_pattern
ibd_entry_volume_ratio
volume_ratio
```

`latest_close`、`current_vs_candidate_pct` 为核心展示字段，但本轮不提供直接筛选控件。

### `BREAKOUT_POOL_LOCAL_DASHBOARD_DESIGN.md`

删除旧的五阶段字段漏斗描述，改为：

```text
Route
→ Entry Status
→ Optional Quality Filters
→ Result Table
```

### `BREAKOUT_QUADRANT_QUANTITATIVE_METHODOLOGY.md`

只补充一句：

> Breakout Pattern 仅描述确认当日 K 线质量，不表示当前仍处于可买区间；当前可执行性以 `ibd_entry_status` 为准。

## 11. 实施顺序

严格按顺序执行：

1. 修改 Pool 生成层并生成带新字段的真实 CSV。
2. 更新 Schema。
3. 更新 Dashboard field config 和 normalize。
4. 增加状态计数及筛选纯函数测试。
5. 重构页面布局与控件。
6. 更新表格视图。
7. 更新 self-check 和真实 CSV 基线测试。
8. 更新其余文档。
9. 运行完整测试和 self-check。

禁止先写 UI、最后再猜字段语义。

## 12. 完成标准

只有同时满足以下条件才算完成：

- 三个新字段由 Pool 生成层稳定输出。
- 83 只 Signal 的状态数量满足 `62 + 10 + 11 + 0 = 83`。
- DELL 不会因位于候选价上方而被错误归入 ACTIONABLE。
- 默认页面不会把 Pool 自动硬筛到 3 只。
- Structure、Sector、Industry 不再出现在筛选区。
- Result Table 位于图表之前。
- C Rank Reference Mode 无行为变化。
- `self_check.py` 与 pytest 全部通过。
- 四份文档语义一致。

若真实数据无法满足基线，先排查字段生成和单位，不得通过修改预期数量让测试通过。
