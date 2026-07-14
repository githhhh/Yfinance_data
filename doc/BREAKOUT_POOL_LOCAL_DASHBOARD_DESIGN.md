# Breakout Pool 本地仪表板设计与实施规范

> 交付对象：Gemini Pro  
> 字段定义、公式和空值规则以 `BREAKOUT_FOLLOW_POOL_SCHEMA.md` 为准。本文规定 Dashboard 的页面结构、筛选流程、展示顺序和验收标准。

---

## 1. 产品定位

Dashboard 是读取 `breakout_follow_pool.csv` 的本地 Streamlit Review 工具，服务于周三收盘后和周末收盘后的两次固定复盘。

页面完成三项任务：

1. 从 `signal=True` 的标的中识别当前 IBD 生命周期状态。
2. 使用候选价距离和量能条件形成可处理的 Review 队列。
3. 通过结果表完成查看、复制代码和导出。

页面以 Result Table 为核心，筛选器位于表格之前，KPI 和辅助图表围绕当前结果提供摘要。

### 1.1 快照频率

- 周三收盘后生成周中快照。
- 周末收盘后生成正式快照。
- `snapshot_date` 标识当前 CSV 的数据截止日期。
- `latest_close` 与 `current_*` 字段均表达该快照时点的状态。

---

## 2. 数据契约

Dashboard 直接读取 Pool 生成层写入的物理字段。

### 2.1 IBD Review 核心字段

```text
latest_close
current_vs_ibd_candidate_pct
ibd_entry_status
```

### 2.2 参考字段

```text
eps_yoy_growth
price_52_week_high
dist_to_52w_high_pct
```

### 2.3 字段职责

| 字段 | Dashboard 用途 |
|:--|:--|
| `latest_close` | 显示快照收盘价 |
| `current_vs_ibd_candidate_pct` | 候选价距离筛选、展示和排序 |
| `ibd_entry_status` | IBD Review 的主决策状态 |
| `ibd_entry_volume_ratio` | 已确认突破的日线量能质量 |
| `volume_ratio` | 当前周线量能参考 |
| `ibd_candidate_rule` | Route 分类 |
| `rank_C_continuous` | 同状态内的次级排序参考 |

Dashboard 对核心字段执行 Schema 校验和抽样复算；Pool 生成层是字段值的唯一生产者。

---

## 3. 技术结构

```text
Yfinance_data/
  dashboard/
    run_app.py
    app.py
    data_utils.py
    field_config.py
    table_view.py
    self_check.py
    requirements.txt
    README.md
    tests/
      test_filters.py
      test_charts.py
      test_table_config.py
      test_app_static.py
      test_app_runtime.py
    .streamlit/
      config.toml
```

启动方式：

```bash
python dashboard/run_app.py --csv /path/to/breakout_follow_pool.csv
```

### 3.1 文件职责

| 文件 | 职责 |
|:--|:--|
| `app.py` | 页面布局、控件状态收集和渲染 |
| `data_utils.py` | CSV 标准化、状态计数、筛选、排序和 KPI 数据纯函数 |
| `field_config.py` | 字段标签、类型、格式、列视图和筛选配置 |
| `table_view.py` | AG Grid、Pin、拖列、复制和表头展示 |
| `self_check.py` | 使用真实 CSV 校验 Schema、公式、状态守恒和筛选结果 |
| `tests/` | 使用固定 fixture 验证业务纯函数与页面配置 |

### 3.2 依赖

```text
streamlit
pandas
numpy
plotly
streamlit-aggrid
python-dateutil
pytest
```

### 3.3 缓存

`@st.cache_data` 的缓存键包含：

```text
CSV path + st_mtime_ns + file size
```

同一路径文件更新后，页面加载新快照。

---

## 4. 页面模式

页面提供两个互斥模式：

```text
IBD Review（默认）
C Rank Reference
```

### 4.1 IBD Review

围绕 Entry Status、候选价距离和量能形成决策队列。

### 4.2 C Rank Reference

展示原始 C Rank 队列：

```text
signal=True
rank_C_continuous ASC
Top N: All / 10 / 20 / 30 / 50
```

该模式独立使用自己的过滤和排序状态。

---

## 5. IBD Review 漏斗

决策链：

```text
signal=True
→ Route
→ Entry Status
→ Current vs IBD Candidate
→ IBD Entry Volume Ratio
→ Weekly Volume Ratio
→ Review Table
```

启用的条件使用 AND 组合。

### 5.1 基础池

IBD Review 的基础集合为：

```text
signal=True
```

页面顶部显示：

```text
Total Pool | Active Signal | Snapshot Date
```

### 5.2 Route

字段：

```text
ibd_candidate_rule
```

默认值为 `All`，选项从当前 CSV 的有效值生成。标准 Route 包括：

```text
ceiling
ceiling_pullback
pivot
ma10_touch_confirm
three_weeks_tight
```

### 5.3 Entry Status

控件顺序：

```text
All
ACTIONABLE
UNCONFIRMED
BELOW_TRIGGER
EXTENDED
```

默认值为 `All`。每个状态显示数量，计数集合为：

```text
signal=True + 当前 Route
```

| 状态 | Review 含义 |
|:--|:--|
| `ACTIONABLE` | 已确认，当前位于候选价上方 0%～5% |
| `UNCONFIRMED` | 有 Signal 和候选价，等待合格日线量价确认 |
| `BELOW_TRIGGER` | 曾确认，当前位于候选价下方 |
| `EXTENDED` | 已确认，当前超过候选价 5% |

### 5.4 候选价距离

字段：

```text
current_vs_ibd_candidate_pct
```

使用最小值和最大值范围控件，单位为百分点：

```text
-5.0 = 候选价下方 5%
+3.0 = 候选价上方 3%
```

默认范围覆盖当前阶段全部有效值。

典型使用方式：

- `UNCONFIRMED`：寻找候选价附近的等待确认标的。
- `BELOW_TRIGGER`：使用例如 `-5%～0%` 控制回撤 Review 深度。
- `ACTIONABLE`：查看 0%～5% 买入区内的已确认标的。

### 5.5 量能条件

保留两个范围控件：

```text
ibd_entry_volume_ratio
volume_ratio
```

规则：

- 两个控件默认覆盖当前阶段全部有效值。
- `ibd_entry_volume_ratio` 应用于 `ACTIONABLE`、`BELOW_TRIGGER` 和 `EXTENDED`。
- `UNCONFIRMED` 使用 `ibd_entry_reject_reason` 查看待确认原因。
- `volume_ratio` 适用于所有状态。

### 5.6 筛选顺序

```python
signal_df = filter_signal(pool_df)
route_df = filter_route(signal_df, selected_route)
status_counts = build_entry_status_counts(route_df)
status_df = filter_status(route_df, selected_status)
distance_df = filter_candidate_distance(status_df, distance_range)
final_df = filter_quality(distance_df, entry_volume_range, weekly_volume_range)
```

状态数量固定从 `route_df` 计算，最终行数从 `final_df` 计算。

---

## 6. 页面布局

页面从上到下排列：

```text
Title + Snapshot Summary
Mode Switch
Filter Bar
Active Filter Summary
Result Count + KPI
Result Table
Optional Charts
```

### 6.1 Filter Bar

单行优先排列：

```text
Route | Entry Status | Candidate Distance | Entry Volume | Weekly Volume
```

窄屏时自动换行。

### 6.2 Active Filter Summary

示例：

```text
Active Signal: 83 → Status: UNCONFIRMED (62) → Distance: -3% to +3% → Result: 18
```

### 6.3 Result Count

```text
Showing 18 of 83 Active Signals
```

---

## 7. Result Table

### 7.1 默认列视图：IBD Decision

```text
code
snapshot_date
ibd_candidate_rule
ibd_entry_status
latest_close
ibd_candidate_price
current_vs_ibd_candidate_pct
ibd_entry_date
ibd_entry_price
ibd_entry_volume_ratio
ibd_entry_reject_reason
volume_ratio
eps_yoy_growth
price_52_week_high
dist_to_52w_high_pct
rank_C_continuous
```

展示规则：

- `code` Pin Left。
- 表头使用 `field_config.py` 的友好名称。
- 百分点字段显示符号和 `%`，例如 `+1.36%`。
- `ibd_entry_reject_reason` 对 `UNCONFIRMED` 保持可见。
- `rank_C_continuous` 位于末列，作为同状态内的次级顺序解释。

### 7.2 Column View

```text
IBD Decision（默认）
IBD Entry
Signal
Volume/Pullback
Reference
All Fields
```

Column View 只改变显示列，筛选结果保持一致。

### 7.3 默认排序

IBD Review 默认排序：

```text
ACTIONABLE
UNCONFIRMED
BELOW_TRIGGER
EXTENDED
```

同一状态内：

```text
rank_C_continuous ASC
```

空 Rank 排在最后。页面显示：

```text
Sort: Entry Status priority → C Rank
```

### 7.4 交互

- 表格列支持排序、拖动和调整宽度。
- 支持复制单个代码和复制当前结果代码列表。
- 导出使用当前筛选结果和当前排序。
- 导出字段名与标准 Pool Schema 一致。

---

## 8. KPI 与辅助图表

KPI 基于最终筛选结果：

```text
Final Rows
Median Candidate Distance
Median IBD Entry Volume Ratio
Median Weekly Volume Ratio
```

无适用数据时显示 `n/a`。

Route Quality 可作为折叠图表，统计集合为：

```text
signal=True + 当前 Route，Entry Status 过滤前
```

图表位于 Result Table 之后。

---

## 9. 实现接口

`data_utils.py` 提供：

```python
load_pool_csv(path)
normalize_pool_df(df)
apply_filters(df, filters)
apply_sort(df, sort_specs)
build_entry_status_counts(route_df)
build_kpis(filtered_df)
```

### 9.1 数据标准化

- Boolean 使用显式映射，NA 保持为空。
- 数值字段统一使用 `pd.to_numeric(errors="coerce")`。
- 状态字段接受 Schema 规定的四个枚举或空值。
- 价格分母校验为有限数且大于 0。
- 核心字段缺失时显示 Schema Error，并列出缺失字段。

---

## 10. 测试与自检

运行：

```bash
pytest dashboard/tests -q
python dashboard/self_check.py --csv /path/to/breakout_follow_pool.csv
```

### 10.1 Schema 与公式

- `latest_close`、`current_vs_ibd_candidate_pct`、`ibd_entry_status` 存在。
- `current_vs_ibd_candidate_pct` 抽样复算与 Schema 公式一致。
- `dist_to_52w_high_pct` 抽样复算与 Schema 公式一致。
- 核心字段的空值和有限数规则通过边界测试。

### 10.2 Entry Status 边界

```text
signal=False                                      → None
signal=True, distance invalid                     → None / Schema Error
signal=True, valid=False, distance=2.0            → UNCONFIRMED
signal=True, valid=True, distance=-0.01           → BELOW_TRIGGER
signal=True, valid=True, distance=0               → ACTIONABLE
signal=True, valid=True, distance=5               → ACTIONABLE
signal=True, valid=True, distance=5.01            → EXTENDED
```

正常 Signal 集合满足：

```text
UNCONFIRMED + ACTIONABLE + BELOW_TRIGGER + EXTENDED = signal=True rows
```

### 10.3 筛选测试

- 单条件筛选结果正确。
- 多条件 AND 组合结果正确。
- Entry Status 计数保持基于 `signal_df + route`。
- 两个页面模式拥有独立状态。
- 空结果显示清晰提示和零行表格。

### 10.4 表格测试

- 默认列顺序与本规范一致。
- `code` 固定在左侧。
- 百分点格式、空值和日期格式正确。
- 默认排序稳定。
- 复制和导出顺序与当前表格一致。

### 10.5 真实 CSV 回归基线

2026-07-10 fixture：

```text
Total Pool       717
Active Signal     83
UNCONFIRMED       62
ACTIONABLE        10
BELOW_TRIGGER      0
EXTENDED          11
```

该 fixture 用于回归测试；每期运行以当前 CSV 的实际计数为准。

---

## 11. 实施顺序

1. 在 Pool 生成层完成核心字段写入和状态计算。
2. 完成 `data_utils.py` 的标准化、筛选、排序、计数和 KPI 纯函数。
3. 完成 `field_config.py` 的字段格式和 Column View。
4. 完成 IBD Review 漏斗与模式隔离。
5. 完成 Result Table、复制和导出。
6. 完成 KPI 与 Route Quality 折叠图表。
7. 运行单元测试和真实 CSV 自检。

---

## 12. 完成标准

- 页面默认进入 IBD Review。
- `signal=True` 构成固定基础池。
- Entry Status 默认显示 `All` 并展示四态计数。
- 候选价距离和两个量能范围可独立组合筛选。
- Result Table 位于页面主体位置。
- 默认排序为 Entry Status 优先、C Rank 次级。
- C Rank Reference 保持原始排名顺序。
- 两个模式状态互相独立。
- 单元测试与真实 CSV 自检全部通过。
- README 包含启动命令、依赖安装、数据路径和测试命令。
