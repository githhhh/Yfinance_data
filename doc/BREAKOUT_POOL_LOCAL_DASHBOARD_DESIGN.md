# Breakout Pool 本地分析面板设计 v0.1.24

## 1. 最终定位

这是一个**本地 Streamlit 分析面板**，用于读取 `breakout_follow_pool.csv`，快速验证不同字段筛选和排序组合。

第一版只强化三件事：

```text
字段组合筛选能力
强表格展示能力
少量关键图表辅助感知
```

页面主角是 Result Table。图表只做辅助，不承担筛选主流程。

---

## 2. 放置位置与启动

目录固定：

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
    .streamlit/
      config.toml
```

启动：

```bash
python dashboard/run_app.py --csv /path/to/breakout_follow_pool.csv
```

默认读取：

```text
dashboard/data/breakout_follow_pool.csv
```

---

## 3. 复杂度边界

### 3.1 文件职责

```text
app.py           Streamlit 页面组织与状态收集
data_utils.py    CSV 读取、字段清洗、筛选、排序、图表聚合纯函数
field_config.py  字段配置、preset、默认列、可筛选字段、可排序字段
table_view.py    AG Grid 表格封装
self_check.py    本地 CSV 自测脚本：验证筛选逻辑、排序逻辑、图表聚合数据
run_app.py       单命令启动入口
tests/           pytest 单元测试：使用小样本 fixture 验证纯函数
```

页面层不写复杂业务判断。筛选、排序、聚合必须放在 `data_utils.py`，方便测试。

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

`streamlit-aggrid` 只用于 Result Table，因为表格需要拖列和 pin 列。其它部分尽量使用 Streamlit 原生组件。

---

## 4. 页面布局

采用紧凑左右布局。

```text
┌────────────────────┬───────────────────────────────────────────────┐
│ Sidebar Filters    │ Active Filters / KPI                          │
│                    │ Compact Charts                                │
│                    │ Sort Bar                                      │
│                    │ Result Table                                  │
└────────────────────┴───────────────────────────────────────────────┘
```

右侧顺序固定：

```text
1. Active Filter Summary，一行 chips
2. KPI，一行 4 个数字
3. 2 个核心图表，单行并排，高度约 220px
4. Sort Bar，一行
5. Result Table，占主要空间
```

不要再增加说明卡片、长字段卡片、测试预览卡片、逻辑注释卡片。

---

## 5. 模式设计

页面只有两个互斥模式。

### 5.1 Custom Filter Mode，默认

用于字段组合筛选和排序实验。

Custom Mode 中完全隔离以下字段：

```text
C_continuous
rank_C_continuous
is_priority
```

这些字段不出现在 Custom Mode 的筛选、排序、默认列、Advanced Field Filters 中。

### 5.2 C Rank Reference Mode

只用于查看旧 C Rank 参照队列。

固定规则：

```text
signal=True
rank_C_continuous asc
```

显示范围：

```text
All rows / Top 10 / Top 20 / Top 30 / Top 50
```

C Rank Mode 不使用 Custom Filters，不使用 Sort Bar，不参与图表联动。

---

## 6. 筛选设计

筛选分两层。

```text
Core Filters：高频字段，直接显示
Advanced Field Filters：任意字段组合，默认折叠，按需添加
```

所有启用条件统一使用 AND 组合。

```text
Core Filters AND Advanced Field Filters
```

未启用字段完全不参与过滤。

---

## 7. Sidebar：核心筛选

### 7.1 Preset，只保留 3 个

#### IBD 有效突破，默认

```text
signal=True
ibd_entry_valid=True
sort_1=ibd_entry_volume_ratio desc
sort_2=ibd_entry_close_vs_trigger_pct desc
```

#### Ceiling Pullback 回撤确认

```text
signal=True
signal_source=ceiling_breakout
ibd_candidate_rule=ceiling_pullback
sort_1=pct_above_ceiling asc
```

注意：`ceiling_pullback` 是 `ibd_candidate_rule`，不是独立 `signal_source`。

#### MA Touch Count

```text
signal=True
sort_1=touched_ema10_count desc
```

### 7.2 Core Filters，默认可见

只放最常用字段。

| 分组 | 字段 | 控件 | 默认 |
|---|---|---|---|
| Signal | `signal` | select | True |
| Signal | `signal_source` | select | All |
| Candidate | `ibd_candidate_rule` | select | All |
| IBD Entry | `ibd_entry_valid` | select | True |
| IBD Entry | `ibd_entry_volume_ratio` | compact range | auto |
| IBD Entry | `ibd_entry_close_vs_trigger_pct` | compact range | auto |

### 7.3 Secondary Filters，默认折叠

用于后置收窄，不放在第一屏核心位置。

| 分组 | 字段 | 控件 | 默认 |
|---|---|---|---|
| Risk / Structure | `pct_above_ceiling` | compact range | auto |
| Risk / Structure | `touched_ema10_count` | compact range | auto |
| Risk / Structure | `volume_ratio` | compact range | auto |
| Grouping | `sector` | searchable multiselect | All |
| Grouping | `industry` | searchable multiselect | All |

---

## 8. Advanced Field Filters

### 8.1 目标

支持对 Custom Mode 可显示字段进行任意组合筛选。

默认不启用，不展开，不铺满侧边栏。

入口样式：

```text
Advanced filters · 0 active
+ Add filter
```

### 8.2 添加方式

每条高级筛选由 4 部分组成：

```text
Enable / Field / Operator / Value
```

示例：

```text
volume_ratio >= 1.3
pullback_v_is_dry is True
hold_return >= 0
breakout_date after 2026-05-01
code contains CR
```

最终组合：

```text
signal=True
AND ibd_entry_valid=True
AND volume_ratio>=1.3
AND pullback_v_is_dry=True
AND hold_return>=0
```

### 8.3 操作符

| 字段类型 | 操作符 |
|---|---|
| Boolean | is True / is False |
| Category | in / not in |
| Number | >= / <= / between / is empty / not empty |
| Date | after / before / between / is empty / not empty |
| Text | contains / equals / startswith / non-empty |

### 8.4 可选字段

Advanced Field Filters 的字段来源：

```text
Custom Mode 可显示字段 - Custom Mode 隔离字段
```

禁止字段：

```text
C_continuous
rank_C_continuous
is_priority
```

---

## 9. Active Filter Summary

右侧顶部显示当前实际生效条件。

形式是一行 chips，不做大卡片。

示例：

```text
Preset: IBD 有效突破 · Rows: 42/212 · signal=True · ibd_entry_valid=True · volume_ratio>=1.3 · Sort: ibd_entry_volume_ratio desc → close_vs_trigger desc
```

要求：

```text
只显示启用条件
Advanced filters 必须显示
排序必须显示
C Rank Mode 使用独立摘要
```

---

## 10. KPI，一行即可

保留 4 个：

```text
Filtered Rows
IBD Valid Rate
Median IBD Volume Ratio
Median Close vs Trigger
```

KPI 只基于当前筛选结果计算。

---

## 11. 图表，只保留 2 个

图表不是主流程，只提供快速全局感知。

### 11.1 IBD Valid Rate by Signal Source

类型：横向 100% stacked bar，替代普通 stacked count bar。

```text
y = signal_source
x = percentage
segments = ibd_entry_valid True / False
right_label = valid_count / total_count + valid_rate
hover = signal_source, valid_count, invalid_count, total_count, valid_rate
```

用途：更直观地观察不同信号类型的 IBD 有效确认率。

设计要求：

```text
每个 signal_source 一行
绿色段表示 ibd_entry_valid=True
灰色/弱色段表示 ibd_entry_valid=False
右侧直接显示 77/81 · 95% 这类摘要
按照 total_count desc 或 signal_source 固定顺序排列
空结果显示 empty state
```

不要使用普通堆叠柱状图只展示 count，因为它不够直观，难以一眼比较不同信号类型的有效确认比例。

### 11.2 Volume × Close Strength

类型：scatter。

```text
x = ibd_entry_volume_ratio
y = ibd_entry_close_vs_trigger_pct
hover = code, signal_source, ibd_candidate_rule, ibd_entry_price, pct_above_ceiling
```

用途：对应默认排序逻辑，观察放量强度与收盘确认质量。

### 11.3 图表交互

只使用 Plotly 原生能力：

```text
hover 查看完整字段
legend click 隐藏/显示分类
scatter zoom / box select / lasso select
```

图表点击不修改全局筛选。筛选只通过 Sidebar 控件完成。

### 11.4 后续扩展位

保留 `chart_registry` 扩展位，但 v0.1 不默认显示更多图表。

---

## 12. Sort Bar

放在表格上方，一行展示。

最多三层排序：

```text
sort_1 field + direction
sort_2 field + direction
sort_3 field + direction
```

排序字段来自 Custom Mode 可显示字段，但排除：

```text
C_continuous
rank_C_continuous
is_priority
```

---

## 13. Result Table，核心模块

### 13.1 表格组件

使用 `streamlit-aggrid`，封装在 `table_view.py`。

必须支持：

```text
固定 code 列
横向滚动
拖动列顺序
pin / unpin 列
列宽调整
单列排序
quick filter
复制单元格 / 行
```

### 13.2 默认 pin

```text
code pinned left
```

用户可以额外 pin 常用列，例如：

```text
signal_source
ibd_entry_valid
ibd_entry_volume_ratio
```

不保存用户列状态。刷新后恢复默认。

### 13.3 默认列

默认 18～20 列，按业务链路排列。

```text
code
sector
industry
signal_source
ibd_candidate_rule
ibd_candidate_price
ibd_entry_valid
ibd_entry_date
ibd_entry_price
ibd_entry_volume_ratio
ibd_entry_close_vs_trigger_pct
pct_above_ceiling
touched_ema10_count
volume_ratio
pullback_v_is_dry
pullback_count
pullback_pct_off_peak
hold_return
breakout_date
ceiling
```

不包含：

```text
C_continuous
rank_C_continuous
is_priority
ibd_entry_reject_reason
ibd_candidate_extra
```

长字段不放默认列，避免表格变脏。原始数据导出仍保留。

### 13.4 Column View

只保留简单切换：

```text
Core / IBD / Risk / Full Custom
```

`Full Custom` 用 multiselect 选择显示列。

表格可显示字段，也可被 Advanced Field Filters 选择为筛选字段。

### 13.5 导出

导出当前筛选结果，保留原始 CSV 全部字段。

---

## 14. field_config.py

所有字段统一配置，避免散落在页面代码中。

```python
FIELD_CONFIG = {
    "ibd_entry_volume_ratio": {
        "type": "number",
        "label": "IBD Entry Volume Ratio",
        "group": "IBD Entry",
        "filterable": True,
        "sortable": True,
        "custom_mode": True,
        "c_rank_mode": False,
        "default_table": True,
        "advanced_filter": True,
        "format": "0.00x",
        "help": "突破当日成交量相对前 50 均量的比例",
    },
}
```

字段配置必须驱动：

```text
筛选控件
Advanced Field Filters
排序字段
表格列
图表 hover
```

---

## 15. data_utils.py 纯函数

必须提供：

```python
load_pool_csv(path: str) -> pd.DataFrame
normalize_pool_df(df: pd.DataFrame) -> pd.DataFrame
build_filter_specs(ui_state) -> list[FilterSpec]
apply_filters(df: pd.DataFrame, filters: list[FilterSpec]) -> pd.DataFrame
apply_sort(df: pd.DataFrame, sort_specs: list[SortSpec]) -> pd.DataFrame
build_active_filter_summary(filters, sort_specs) -> list[str]
build_kpis(df: pd.DataFrame) -> dict
build_chart_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]
```

---

## 16. 自测与测试要求，必须实现

这个面板的主要风险不是 UI，而是筛选组合、排序结果和图表聚合口径错误。实现完成后必须提供可运行的自测脚本，并在交付前跑通。

要求保留两类测试：

```text
pytest 单元测试：验证纯函数，使用小型 fixture 数据。
self_check.py 自测脚本：读取真实 breakout_follow_pool.csv，验证实际筛选结果和图表数据口径。
```

自测命令：

```bash
python dashboard/self_check.py --csv /path/to/breakout_follow_pool.csv
pytest dashboard/tests -q
```

`self_check.py` 不启动 Streamlit，只调用 `data_utils.py` 与 `field_config.py` 中的纯函数，输出简洁结果：

```text
[PASS] load and normalize
[PASS] preset: IBD 有效突破
[PASS] preset: Ceiling Pullback 回撤确认
[PASS] preset: MA Touch Count
[PASS] advanced filters AND logic
[PASS] sort specs
[PASS] chart: IBD Valid Rate by Signal Source aggregation
[PASS] chart: Volume × Close Strength row source
[PASS] mode isolation
```

## 20. Filter Funnel Layout Follow-up (2026-07-02)

本轮根据 `BREAKOUT_POOL_CUSTOM_FILTER_FIELD_GUIDE.md` 重新收敛筛选入口：

```text
Custom Filter
1 Route
2 Entry Confirmation & Strength
3 Weekly Volume & Price
4 Structure
5 Grouping
```

### 20.1 Custom Filter 不再使用 Preset

页面筛选不再以 preset 作为入口，避免隐藏默认条件。筛选条件全部显式来自漏斗阶段与顶部 Active
Chips。后台仍保留 preset 纯函数与 self_check 覆盖，用于验证历史筛选口径与筛选引擎正确性。

### 20.2 筛选字段白名单

自定义筛选仅保留具有直接决策价值的字段：

```text
ibd_candidate_rule
ibd_entry_valid
ibd_entry_volume_ratio
ibd_entry_close_vs_trigger_pct
volume_ratio
is_bullish
touched_ema10_count
pullback_pct
sector
industry
```

其中 `ibd_entry_volume_ratio` 与 `ibd_entry_close_vs_trigger_pct` 只在
`ibd_entry_valid=True` 时启用。

`hold_return`、`mbox_count`、`base_depth_abs`、`base_mbox_count`、`pct_above_ceiling`、
`pullback_pct_off_peak`、`pullback_v_is_dry` 等字段不进入筛选漏斗，仅作为表格观察字段。

### 20.3 表格全字段逻辑分组

Result Table 默认显示 `All Fields`，列顺序按业务链路组织：

```text
Identity / Grouping
Signal / Route
IBD Candidate / Daily Entry
Base / Structure
Weekly Volume / Pullback
C Rank Reference
```

表格新增 `base_duration_weeks` 衍生列，按
`round((breakout_date - ceiling_date) / 7)` 计算，仅用于展示。

### 20.4 C Rank Reference 模式说明

`C Rank Reference` 作为独立模式保留，切换后必须显示：

```text
signal=True
rank_C_continuous asc
Top N selector only
Custom filters ignored
```

并展示 `C_continuous` 的公式参考与 tie-break 语义，避免与 Custom Filter 漏斗混淆。

### 20.5 漏斗筛选可见性修正

Custom Filter 模式不再展示独立 Sort Bar，避免 `sort_1`、`sort_2`、`direction_1` 等底层字段名干扰筛选主流程。

漏斗阶段标题使用纯业务名称，不展示内部数字编号：

```text
Route
Entry Confirmation & Strength
Weekly Volume & Price
Structure
Grouping
```

每个漏斗 tab 标题显示当前启用条件数量，例如 `Structure (2)`。页面顶部必须按漏斗阶段展示当前筛选条件：

```text
Current Filters
Rows: 42/743
Route (1): IBD Candidate Rule: pivot
Entry Confirmation & Strength (2): IBD Entry Valid: True; IBD Entry Volume Ratio: 1.5 to 4.0
```

无启用条件的阶段显示 `All`，让用户能一眼判断哪些漏斗阶段参与了过滤。

任何一项失败都必须 `exit(1)`，不能只打印 warning。

### 16.1 测试重点

测试只围绕核心功能。

### 16.2 Normalize

```text
Boolean 字段正确转换
Number 字段正确转换
Date 字段正确转换
空值不报错
```

### 16.3 Preset

```text
IBD 有效突破：signal=True AND ibd_entry_valid=True
Ceiling Pullback：signal=True AND signal_source=ceiling_breakout AND ibd_candidate_rule=ceiling_pullback
MA Touch Count：signal=True，按 touched_ema10_count desc
```

### 16.4 Advanced Filters

```text
未启用字段不参与过滤
启用字段与 Core Filters 进行 AND 组合
多个高级字段全部 AND 组合
category / number / boolean / date / text 操作符正确
```

### 16.5 Sort

```text
一层排序正确
两层排序正确
三层排序正确
空值排序稳定
```

### 16.6 Mode Isolation

```text
Custom Mode 不出现 C_continuous / rank_C_continuous / is_priority
C Rank Mode 不受 Custom Filters 影响
C Rank Mode 固定 signal=True + rank_C_continuous asc
```

### 16.7 Table

```text
code 默认 pinned left
默认列符合配置
Full Custom 只能选择 Custom Mode 允许字段
导出 CSV 保留原始全部字段
```

### 16.8 Chart

```text
IBD Valid Rate by Signal Source 的 valid_count + invalid_count 合计等于当前筛选结果行数
Scatter 只使用当前筛选结果
空结果显示 empty state
```

### 16.9 self_check.py 必须验证的真实 CSV 逻辑

`self_check.py` 使用真实 CSV 进行端到端口径验证，不能只检查程序是否报错。

必须至少验证以下内容：

#### Preset 结果验证

每个 preset 都要用一份显式 pandas mask 进行交叉验证。

```python
# IBD 有效突破
expected = df[(df["signal"] == True) & (df["ibd_entry_valid"] == True)]
actual = apply_filters(df, build_preset_filters("ibd_valid_breakout"))
assert set(actual["code"]) == set(expected["code"])
```

```python
# Ceiling Pullback 回撤确认
expected = df[
    (df["signal"] == True)
    & (df["signal_source"] == "ceiling_breakout")
    & (df["ibd_candidate_rule"] == "ceiling_pullback")
]
```

```python
# MA Touch Count
expected = df[df["signal"] == True].sort_values("touched_ema10_count", ascending=False)
```

#### Advanced Filters 组合验证

至少构造 3 组组合条件：

```text
number + boolean：volume_ratio >= 1.3 AND pullback_v_is_dry=True
category + number：signal_source in [...] AND ibd_entry_volume_ratio >= 1.5
date/text：breakout_date after ... AND code contains ...
```

每组都要用手写 pandas mask 对比 `apply_filters()` 的结果 code 集合。

#### 排序验证

至少验证：

```text
一层排序：ibd_entry_volume_ratio desc
两层排序：ibd_entry_volume_ratio desc + ibd_entry_close_vs_trigger_pct desc
三层排序：ibd_entry_valid desc + ibd_entry_volume_ratio desc + pct_above_ceiling asc
```

排序验证以 `code` 顺序列表为准。

#### 图表数据验证

图表的数据必须来自当前筛选后的 DataFrame。

`IBD Valid Rate by Signal Source`：

```text
valid_count + invalid_count 合计 == 当前筛选结果行数
按 signal_source + ibd_entry_valid groupby 的结果 == build_chart_data 输出
valid_rate = valid_count / total_count，total_count 为 0 时显示 0 或空态
每个 signal_source 输出一行，包含 valid_count / invalid_count / total_count / valid_rate_pct
```

`Volume × Close Strength`：

```text
散点图行数 == 当前筛选结果中 x/y 所需字段可用的行数
散点图 code 集合是当前筛选结果 code 集合的子集
hover 字段必须存在：code, signal_source, ibd_candidate_rule, ibd_entry_price, pct_above_ceiling
```

#### 模式隔离验证

```text
Custom Mode 可筛选字段中不能出现 C_continuous / rank_C_continuous / is_priority
Custom Mode 默认列中不能出现 C_continuous / rank_C_continuous / is_priority
C Rank Mode 固定 signal=True + rank_C_continuous asc
C Rank Mode 不读取 Custom Filters 和 Sort Bar
```

### 16.10 交付验收要求

Codex 完成实现后，必须在回复中附上：

```text
运行命令
测试命令
self_check.py 输出摘要
pytest 输出摘要
```

未跑自测，视为未完成交付。

---

## 17. Codex 实现指令

请在 `Yfinance_data/dashboard/` 下实现本地 Streamlit 分析面板。

重点：

```text
1. 表格是主角，图表只保留 2 个。
2. 左侧只显示核心筛选；其它字段通过 Advanced Field Filters 按需添加。
3. Advanced Field Filters 支持 Custom Mode 可显示字段，默认不启用。
4. 所有启用筛选统一 AND 组合。
5. Result Table 使用 streamlit-aggrid，必须支持拖列、pin 列，并默认固定 code。
6. Custom Mode 完全隔离 C_continuous / rank_C_continuous / is_priority。
7. C Rank Reference Mode 独立，只按 signal=True + rank_C_continuous asc 展示。
8. 筛选、排序、表格列、图表聚合必须有 pytest 测试。
9. 必须实现 dashboard/self_check.py，并用真实 CSV 验证筛选逻辑和图表聚合口径。
10. 交付前必须运行：python dashboard/self_check.py --csv <csv_path> 和 pytest dashboard/tests -q。
```

## 18. Senior Engineer Code Review (v0.1.24 实现评估)

这是一个基于 Senior Engineer 视角、遵循极简主义（Ponytail）和“红蓝对抗”预演机制的深度 Code Review。

整体来看，该 Dashboard 模块采用了非常优秀的**数据与视图解耦**架构：`field_config.py` 充当统一配置中心，`data_utils.py` 封装纯函数式数据处理，`app.py` 负责渲染。这种配置驱动（Configuration-driven）的设计使得可维护性和扩展性极高。

但在深入代码细节与执行上下文中，我发现了几个严重的**性能隐患**和**冗余设计**。以下是具体的 Review 报告：

### 🔴 1. 严重的性能与内存隐患 (Red Team 视角)

Streamlit 的生命周期是**“每次交互重新执行整个脚本”**。在此机制下，当前代码存在致命的性能消耗：

*   **缺失数据缓存 (致命)**：
    `app.py` 的 `main()` 中直接调用 `df = load_pool_csv(args.csv)`。由于没有任何缓存机制，用户每次在侧边栏调整一个 Filter，程序都会去磁盘重新读取一次 CSV、并重新执行所有字符串截取和类型转换 (`normalize_pool_df`)。对于量化分析数百上千条数据的场景，这会导致明显的 UI 卡顿。
    **最优解修改：** 必须在 `data_utils.py` 或 `app.py` 中使用 `@st.cache_data` 装饰器包裹加载逻辑。
*   **滥用深拷贝 (`df.copy()`)**：
    `data_utils.py` 中所有的图表聚合函数（如 `_build_signal_quality_matrix_data`、`_build_structure_action_map_data` 等）在开头都使用了 `working = df.copy()`。这会导致内存中同时存在多个全量 DataFrame 副本。实际上聚合操作只需要用到 3~4 列。
    **最优解修改：** 绝不要全量 copy。应当仅截取需要的列进行计算，或使用 `df.assign()`。例如：`working = df[['signal_source', 'ibd_entry_valid']].copy()`。
*   **非向量化的 Map 操作 (降维打击)**：
    `data_utils.py` 中的布尔掩码函数实现非常低效：
    ```python
    def _true_mask(series: pd.Series) -> pd.Series:
        return series.map(lambda value: False if pd.isna(value) else bool(value) is True).astype(bool)
    ```
    使用 `lambda` 使得 Pandas 放弃了底层的 C 级加速，退化为 Python 层面的逐行循环。
    **最优解修改：** 使用纯向量化操作，性能可提升几十倍：
    `return series.fillna(False).astype(bool)`

### 🟡 2. 代码冗余与极简主义违背 (Ponytail 视角)

贯彻“能少写一行绝不多写”的极简主义，代码中存在一些无用逻辑和过度依赖：

*   **死代码 (Dead Code)**：
    `data_utils.py` 中的 `build_filter_specs(ui_state: dict[str, Any]) -> list[FilterSpec]` 完全是废弃代码，`app.py` 并没有调用它，而是手动组装了 Filter 列表。应该直接删除。
*   **奇怪的函数签名**：
    `apply_c_rank_mode` 函数签名中接受了 `filters` 和 `sort_specs`，但在第一行就 `del filters, sort_specs`。如果没有通过高阶函数强制约束接口，这毫无意义，应该直接从签名中剔除。
*   **依赖管理臃肿**：
    `requirements.txt` 中引入了 `pytest`，但是工程内部使用的测试是 `self_check.py`，它自带 `if __name__ == '__main__':` 运行逻辑，并没有用到 pytest 框架。同时引入了 `numpy`，虽然 pandas 依赖它，但如果没有直接在代码里 `import numpy` 进行特殊矩阵操作，就不应该在顶级依赖中声明。

### 🟢 3. 健壮性与兜底逻辑评估 (First Principles 视角)

*   **过滤器的“兜底掩盖”风险**：
    在 `_coerce_float` 和 `_coerce_timestamp` 中，如果类型转换失败，使用了 `try...except` 吞掉异常并返回 `None`。如果上游的数据清洗（如数据源改版导致某列混入了无法解析的字符）出现问题，由于此处的静默处理，Dashboard 会显示为空数据或错误过滤，而不是明确报错。
    **建议：** 考虑到量化系统的严谨性，建议在 `normalize_pool_df` 阶段就进行严格校验，渲染阶段的过滤只做逻辑匹配。

---

### 🛠️ 建议的修改切口 (Action Plan)

1.  在 `data_utils.py` 的 `load_pool_csv` 加上 `@st.cache_data`，根除 UI 卡顿。
2.  用 `.fillna(False).astype(bool)` 重写 `_true_mask` 和 `_false_mask`，移除 Lambda 函数。
3.  重构 `_build_*_data` 绘图函数，移除无脑的 `df.copy()` 避免内存暴涨。
4.  清理废弃的死代码 `build_filter_specs`，精简 `requirements.txt` 和无用的函数参数。

## 19. Review Follow-up Fixes (2026-06-30)

本轮 review 后发现并修复了 v0.1.24 性能优化提交中的 3 个回归风险。

### 19.1 Streamlit CSV 缓存失效策略

问题：

`app.py` 中新增的 `@st.cache_data` 只以 CSV 路径作为缓存 key。若定时任务覆盖同一路径的
`breakout_follow_pool.csv`，Dashboard rerun 时可能继续使用旧 DataFrame。

修复：

`cached_load_pool_csv(path, cache_fingerprint)` 增加文件指纹参数，指纹由
`(st_mtime_ns, st_size)` 组成。CSV 同路径被覆盖后，只要修改时间或文件大小变化，Streamlit
缓存 key 会同步变化，从而重新读取并标准化数据。

回归测试：

`test_csv_cache_fingerprint_changes_when_same_path_is_rewritten` 覆盖同一路径重写后的指纹变化。

### 19.2 Boolean Mask 的 pandas FutureWarning

问题：

`_true_mask/_false_mask` 使用 `series.fillna(...).astype(bool)` 后，pandas 会触发 object dtype
静默 downcasting 的 `FutureWarning`。这会污染测试输出，并增加未来 pandas 升级后的行为风险。

修复：

改为 `series.to_numpy(dtype=bool, na_value=...)` 后再包装回 `pd.Series`，保留原语义：

```text
_true_mask:  NA -> False
_false_mask: NA -> False
```

回归测试：

`test_boolean_masks_do_not_emit_pandas_downcasting_warning` 同时覆盖 `_true_mask` 和 `_false_mask`
的返回值与 warning 行为。

### 19.3 self_check 失败上下文

问题：

`self_check.py` 的异常路径从原先的 `[FAIL] {label}: ...` 退化为直接 `raise`，导致失败时只有
traceback，缺少当前失败检查项，定位成本上升。

修复：

恢复 `print(f"[FAIL] {label}: {exc}", file=sys.stderr)`，并保持返回码为 `1`。

回归测试：

`test_self_check_reports_setup_failures_with_label` 覆盖 CSV 缺失时的 `[FAIL] setup:` 输出。

### 19.4 验证记录

修复后执行：

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest dashboard/tests -q -p no:cacheprovider
```

结果：

```text
30 passed
```

同时执行真实 CSV 自检：

```text
PYTHONDONTWRITEBYTECODE=1 python dashboard/self_check.py --csv us/breakout_follow_pool.csv
```

结果：

```text
[PASS] load and normalize
[PASS] preset: Review All Signals
[PASS] preset: IBD Valid Breakout
[PASS] preset: Action Clean Entry
[PASS] preset: Ceiling Breakout
[PASS] preset: Ceiling Pullback
[PASS] preset: Pivot Review
[PASS] preset: 10W EMA Touch
[PASS] advanced filters AND logic
[PASS] sort specs
[PASS] chart: Signal Quality Matrix aggregation
[PASS] chart: Structure Action Map row source
[PASS] chart: Sector Concentration aggregation
[PASS] chart: IBD Valid Rate by Signal Source aggregation
[PASS] chart: Volume x Close Strength row source
[PASS] mode isolation
```

## 21. IBD Review Funnel Refactor (2026-07-13)

按规范 `BREAKOUT_POOL_IBD_REVIEW_FUNNEL_REFACTOR_SPEC.md` 将看板筛选结构重构为“三阶段 IBD 漏斗”，并以 `IBD Decision` 为默认表格视图。

### 21.1 漏斗三阶段分组

```text
1. Route
2. Entry Status
3. Optional Quality Filters
```

- **Route**: `signal=True` 为默认不可关闭底座；可单选 `ibd_candidate_rule` (`All`, `ceiling`, `ceiling_pullback`, `pivot`, `ma10_touch_confirm`, `three_weeks_tight`)。
- **Entry Status**: 统计基于 `signal=True + 当前 Route` 下候选的 `ibd_entry_status` 数量，提供状态选择 (`All`, `UNCONFIRMED`, `ACTIONABLE`, `EXTENDED`, `BELOW_TRIGGER`)。
- **Optional Quality Filters**: 提供三个进阶质量条件：`breakout_pattern`、`ibd_entry_volume_ratio`、`volume_ratio`。当状态为 `UNCONFIRMED` 时自动禁用日线突破形态与成交量比筛选。

### 21.2 默认视图 IBD Decision

表格默认加载 `IBD Decision` 视图字段组合：

```text
code, snapshot_date, ibd_candidate_rule, ibd_entry_status, latest_close, ibd_candidate_price, current_vs_ibd_candidate_pct, ibd_entry_price, ibd_entry_volume_ratio, volume_ratio, breakout_pattern, sector, industry
```
