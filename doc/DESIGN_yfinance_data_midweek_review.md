# Yfinance_data 周中 Review 实现设计

## 1. 文档边界

本文只定义 `Yfinance_data` 的改动。

`quant_trade` 已独立完成数据生产改造，本设计将其视为两个稳定输入：

```text
breakout_follow_pool.csv
breakout_follow_pool_midweek.csv
```

`Yfinance_data` 负责：

1. 判断 midweek 文件是否关联当前完整周；
2. 使用 `midweek current_pool` 作为左表构建 Review Projection；
3. 为 Dashboard 提供周中变化数据；
4. 为 `quant_trade` 提供周中 Futu ACTIONABLE code 接口；
5. 保持完整周 Dashboard 原有体验。

不负责：

- 重新运行 BF 策略；
- 改变 signal、candidate 或 resolver 规则；
- 修改两个源 CSV；
- 接管通知、统计和其他推送内容。

---

## 2. 用户目标

完整周 Dashboard 回答：

> 周末筛选出了什么？

周中 Review 回答：

> 和完整周相比，现在什么值得行动，什么发生了变化？

用户进入周中页面后的阅读顺序：

1. 确认当前周中快照和完整周基线；
2. 查看刚变 ACTIONABLE、新 signal、Carry 和状态恶化；
3. 从变化/来源快捷筛选进入同一张 Review 主表；
4. 查看某一行的完整周 → 周中变化原因；
5. 必要时切回原始完整周 Pool 核对基线。

---

## 3. 输入约束

### 3.1 完整周文件

```text
breakout_follow_pool.csv
```

含义：最近一次成功完整周任务的真实 Pool。

### 3.2 周中文件

```text
breakout_follow_pool_midweek.csv
```

含义：最近一次成功周中任务的真实 Pool，可能属于当前周期，也可能已经过期。

### 3.3 必需校验

两个文件必须校验：

- CSV 可读取；
- `code` 存在、非空、唯一；
- `snapshot_date` 存在且可解析；
- 文件内有效 `snapshot_date` 唯一；
- signal、candidate、latest_close、IBD 字段满足当前 schema；
- Boolean 和数值字段使用现有统一归一化函数。

任一文件不得因为 Dashboard 展示需要被修改。

---

## 4. Review 周关联

midweek 文件长期存在，不能通过文件存在性判断当前模式。判断分成两步：

1. 先按业务日期判断当前是否允许进入周中窗口；
2. 再按两个文件的 `snapshot_date` 判断它们是否属于同一个 Review 周。

### 4.1 业务窗口

| 业务日期 | 默认窗口 | 默认数据 |
| --- | --- | --- |
| 周二至周五 | `MIDWEEK_WINDOW` | 有效周中投影；无有效周中数据时降级 |
| 周六至周一 | `COMPLETE_WINDOW` | `breakout_follow_pool.csv` |

```python
def resolve_window(window_date: date) -> PoolWindow:
    if window_date.weekday() in {1, 2, 3, 4}:  # Tue-Fri
        return MIDWEEK_WINDOW
    return COMPLETE_WINDOW                    # Sat-Mon
```

`window_date` 是 Dashboard/任务统一配置的业务日期。业务时区必须显式配置，并与现有调度时区一致，不能依赖服务器进程的隐式本地时区：

```python
window_date = datetime.now(ZoneInfo(settings.business_timezone)).date()
```

它只决定默认展示窗口，不参与 CSV 命名，也不能替代文件内的 `snapshot_date`。

周六至周一：

- 默认直接展示完整周文件；
- 不执行周中左连接；
- 长期保留的 midweek 文件不触发周中模式；
- Dashboard 仍可保留显式入口查看最近一次周中结果，但不得作为默认视图或 Futu 输入。

周二至周五：

- 只有周中文件与完整周基线关联有效时，才进入正式 `MIDWEEK`；
- midweek 缺失、过期或属于上一 Review 周时，不得因为文件仍存在而误用；
- 周二尚未产生本周 midweek 时，继续展示完整周数据；
- 存在本周 midweek 但没有有效完整周基线时，进入 `MIDWEEK_WITHOUT_VALID_BASELINE`。

### 4.2 完整周目标周

完整周 `snapshot_date` 始终是实际最后交易日，不是任务运行日或文件生成日：

| 完整周 `snapshot_date` | `review_week_start` |
| --- | --- |
| 普通周 Friday | 下周一 |
| Friday 休市时的 Thursday | 下周一 |

该映射由 `bf_snapshot.complete_target_week()` 提供；需要持久化分析的 canonical 完整周
标签由 `bf_snapshot.complete_snapshot_week()` 提供。周末、周一和普通 stale Thursday
都不是合法的完整周 `snapshot_date`，必须 fail closed。

注意：这里的“Friday/Thursday”是完整周文件中的美股实际行情 `snapshot_date`，与
4.1 的 Dashboard 业务窗口不是同一个概念。

### 4.3 midweek 所属周

```python
def monday_of_week(value: date) -> date:
    return value - timedelta(days=value.weekday())
```

### 4.4 最终模式

```python
complete_week = complete_target_week(complete_date)
midweek_week = monday_of_week(midweek_date)
window = resolve_window(window_date)

if window == COMPLETE_WINDOW:
    mode = COMPLETE
elif midweek_file_missing:
    mode = COMPLETE
elif midweek_week < complete_week:
    mode = COMPLETE
elif midweek_week == complete_week and midweek_date > complete_date:
    mode = MIDWEEK
elif midweek_week > complete_week:
    mode = MIDWEEK_WITHOUT_VALID_BASELINE
else:
    mode = COMPLETE
```

| 模式 | Dashboard | Futu 接口 |
| --- | --- | --- |
| `COMPLETE` | 原完整周页面 | 不使用周中投影 |
| `MIDWEEK` | 周中 Review，可切回完整周 | 返回投影 ACTIONABLE |
| `MIDWEEK_WITHOUT_VALID_BASELINE` | 周中原始 Pool + 警告 | 只返回本次真实 ACTIONABLE，不 Carry |

完整周日期不能被 `bf_snapshot` authority 证明为合法完整周时，fail closed，不猜测关联关系。

---

## 5. 建议代码结构

以下为职责建议，实际路径可按现有项目结构调整：

```text
dashboard/
  services/
    bf_pool_context.py       # 文件加载、校验、模式判断
    bf_midweek_review.py     # 左表投影、变化与摘要
    bf_actionable.py         # Futu ACTIONABLE code

  components/
    review_context.py        # 日期和模式切换
    review_metrics.py        # 范围、变化、来源与状态汇总
    review_table.py          # Review 主表字段配置
    review_detail.py         # 行详情
```

核心 services 不得依赖：

- Streamlit session state；
- AgGrid UI 配置；
- Futu SDK；
- Telegram SDK。

UI 组件只消费 service 返回结果，不重新实现业务判断。

---

## 6. 公共分析接口

### 6.1 高层接口

```python
analyze_breakout_follow_pool(
    complete_pool_path: Path,
    midweek_pool_path: Path | None,
) -> PoolAnalysisResult
```

建议结果：

```python
@dataclass
class PoolAnalysisResult:
    mode: PoolMode
    complete_snapshot_date: date | None
    midweek_snapshot_date: date | None
    review_week_start: date | None
    current_review: DataFrame
    exited_pool: DataFrame
    summary: dict[str, int]
    actionable_codes: tuple[str, ...]
    warnings: tuple[str, ...]
```

### 6.2 纯投影接口

```python
build_midweek_review(
    current_pool: DataFrame,
    complete_pool: DataFrame,
) -> MidweekReviewResult
```

必须满足：

- 不读写文件；
- 不读取系统日期；
- 不修改输入 DataFrame；
- 不执行网络和外部同步；
- 相同输入产生相同输出。

### 6.3 Futu 接口

```python
get_midweek_futu_actionable_codes(
    complete_pool_path: Path,
    midweek_pool_path: Path,
) -> ActionableResult
```

返回：

```text
mode
actionable_codes
complete_snapshot_date
midweek_snapshot_date
review_week_start
warnings
```

`quant_trade` 只消费该接口的 code 集合，不消费 Dashboard UI 字段。

---

## 7. 初步投影算法

### 7.1 左表

左连接只在 `MIDWEEK` 模式执行。

```text
左表：breakout_follow_pool_midweek.csv   # 当前周中真实 Pool
右表：breakout_follow_pool.csv           # 当前 Review 周的完整周基线
连接键：code
连接方式：left join，one-to-one
```

左表决定主 Review 的成员集合、当前价格和本次真实结果；右表只提供同一 code 的完整周对照事实，不拥有把 code 补回主表的权限。

```python
joined = midweek_pool.merge(
    complete_pool,
    on="code",
    how="left",
    suffixes=("_current", "_complete"),
    sort=False,
    validate="one_to_one",
)
```

合并后必须满足：

```text
set(current_review.code) == set(midweek_pool.code)
len(current_review) == len(midweek_pool)
```

不得使用 `outer join`，也不允许从完整周补回周中已退出的 code。

#### code 分类

| code 位置 | 分类 | 主 Review | 处理规则 |
| --- | --- | --- | --- |
| current 有、complete 无 | `CURRENT_ONLY` | 保留 | 新进入 Pool；只能使用 current 事实 |
| current 有、complete 有 | `MATCHED` | 保留 | 按 Current Signal / Carry 规则投影 |
| current 无、complete 有 | `EXITED` | 不进入 | 只进入退出统计/列表，不进入 Futu |

#### 字段所有权

| 字段类型 | 优先来源 |
| --- | --- |
| Pool 成员、`latest_close`、本次 signal/result | current 左表 |
| current signal 存在时的 candidate/entry/route | current 左表，整行原子优先 |
| Carry 的 candidate 和完整周 resolver 事实 | complete 右表 |
| C Rank、base、pivot、volume 等研究字段 | 默认 current；Carry 不重新计算完整周结构 |
| `review_*` | 内存投影生成，不写回任一 CSV |

完整周的值只能按明确的 Carry 映射使用，不能用通用 `fillna(complete)` 把 current 的空值逐列补齐，否则会制造一条从未真实存在过的混合 Signal。

退出集合：

```python
exited_codes = set(complete_pool.code) - set(midweek_pool.code)
```

`exited_codes` 只进入 Dashboard 的退出列表，不进入 Review 主表和周中 Futu。

在 `COMPLETE` 模式不执行上述 merge：Dashboard 直接消费完整周文件，midweek 文件即使存在也被窗口规则忽略。

### 7.2 Signal 来源

```python
if signal_current:
    origin = RECONFIRMED_SIGNAL if signal_complete else NEW_SIGNAL
elif signal_complete:
    origin = CARRY_SIGNAL
else:
    origin = NO_SIGNAL
```

### 7.3 Current Signal

当 `signal_current=True`：

- 使用 current 的 signal、candidate、IBD 结果；
- current 行原子优先；
- 即使 current `ibd_entry_valid=False`，也不回退完整周 ACTIONABLE。

### 7.4 Carry

Carry 条件：

```text
code 仍在 current_pool
signal_current=False
signal_complete=True
```

Carry 使用：

- complete candidate；
- complete `ibd_entry_valid` 等 resolver 事实；
- current `latest_close`；
- 现有统一状态函数重算价差和 Effective 状态。

Carry 不重算完整周 base、pivot、volume、pullback 或 C Rank。

### 7.5 Effective 状态

```python
if signal_current:
    effective_status = ibd_entry_status_current
elif signal_complete:
    effective_status = calculate_entry_status(
        signal=True,
        entry_valid=ibd_entry_valid_complete,
        candidate_price=ibd_candidate_price_complete,
        latest_close=latest_close_current,
    )
else:
    effective_status = None
```

当前价或 candidate 无效时返回空状态，不能沿用完整周旧 ACTIONABLE。

### 7.6 ACTIONABLE code

```python
actionable_codes = set(
    current_review.loc[
        current_review["review_watch_active"]
        & (
            current_review["review_effective_entry_status"]
            == "ACTIONABLE"
        ),
        "code",
    ]
)
```

完整周 code 只有仍存在于 current_pool，且使用 current 价格投影后仍为 ACTIONABLE，才进入周中 Futu。

---

## 8. Review 字段

源字段不覆盖，运行时追加：

| 字段 | 用途 |
| --- | --- |
| `review_pool_change` | ENTERED / STAYED |
| `review_signal_origin` | NEW / RECONFIRMED / CARRY / NONE |
| `review_watch_active` | 是否属于当前观察 Signal |
| `review_candidate_price` | 最终采用的 candidate |
| `review_baseline_entry_status` | 完整周状态 |
| `review_effective_entry_status` | 当前 Effective 状态 |
| `review_current_vs_candidate_pct` | 当前价格相对最终 candidate |
| `review_entry_change` | 状态变化 |
| `review_priority` | 默认 UI 排序 |
| `review_futu_actionable` | 是否进入周中 Futu |

状态变化至少支持：

```text
BECAME_ACTIONABLE
STILL_ACTIONABLE
ACTIONABLE_TO_EXTENDED
ACTIONABLE_TO_BELOW_TRIGGER
STATUS_CHANGED
UNCHANGED
```

---

## 9. 周中 Dashboard 心流

### 9.1 设计原则：在现有仪表板上增量修改

周中 Review 不新建另一套 Dashboard，也不改变用户已经熟悉的页面结构。

必须保留：

- `Breakout Pool` 标题、Data Ready 和 Snapshot 行；
- `IBD Review / C Rank Reference` 顶部导航；
- `Review Queue` 及四个 Entry Status 卡片；
- 现有 Filters、结果操作区、Selected Row 汇总条；
- 现有 AG Grid 字段、列拖动、Pin、排序和表头帮助。

周中只增加三类信息：

1. 当前快照与完整周基线的关系；
2. 变化类型的入口；
3. 行级变化及状态前后关系。

### 9.2 第一眼：确认当前处于哪种视图

Snapshot 行在周中显示：

```text
Snapshot 2026-07-29 · Midweek · baseline 2026-07-24
```

`Review Queue` 右侧增加轻量切换：

```text
[ Midweek Review ] [ Weekend Full Pool ]   [ Changes (23) ] [ All Signals (106) ]
```

第一组是数据模式，第二组是当前 Review 范围。二者必须分开，避免出现“Midweek Changes 已选中但同时显示 All Signals 已选中”的语义冲突。

切换不跳页、不创建第二张表，只替换同一张表的数据视图和周中专属字段。

默认选中项由业务窗口决定：周二至周五优先 `Midweek Review + Changes`，周六至周一选中 `Weekend Full Pool + All Signals`；若周中窗口没有有效 midweek，则降级选中完整周并显示数据提示。

### 9.3 第二眼：先处理变化，再看当前状态

周中模式在原四个状态卡上方增加一条紧凑快捷筛选栏，不再增加第二排大卡片：

```text
CHANGE  [ Became Actionable 8 ] [ Left Actionable 3 ] [ Other Changes 12 ]
ORIGIN  [ New 12 ] [ Carry 8 ] [ Reconfirmed 3 ]                 [ Clear ]
```

`CHANGE` 与 `ORIGIN` 是两个独立维度：

- Change 表达“发生了什么”；
- Origin 表达“Signal 从哪里来”；
- 同组单选，不同组可以组合；
- 状态卡可以继续叠加过滤，不清除 Change/Origin；
- Clear 只清除快捷筛选，不修改展开式 Filters。

快捷栏下方仍显示原有四个 Entry Status 卡：

```text
ACTIONABLE / UNCONFIRMED / BELOW TRIGGER / EXTENDED
```

因此用户心流是：

```text
默认只看变化 → 按 Change/Origin 缩小 → 看当前状态 → 用原 Filters 深入分析
```

#### 心流卡片悬浮说明（强制）

周中心流中的所有可点击卡片都必须提供悬浮说明，不能只依靠标签、颜色或用户记忆理解含义。覆盖范围包括：

- `CHANGE` 的 3 个快捷筛选卡片；
- `ORIGIN` 的 3 个快捷筛选卡片；
- 原有 4 个 Entry Status 卡片。

每个悬浮说明统一回答三件事：

1. **定义**：这张卡表示什么；
2. **统计口径**：卡片数字如何计算；
3. **点击效果**：点击后会筛选哪些行，以及会与哪些条件组合。

建议文案如下：

| 卡片 | 悬浮说明 |
| --- | --- |
| Became Actionable | 完整周基线不是 ACTIONABLE、但本次周中 Effective Status 已变为 ACTIONABLE。数字为当前 Review 范围内符合条件的数量；点击后按此变化筛选，并保留已选 Origin、Status 和 Filters。 |
| Left Actionable | 完整周基线为 ACTIONABLE、但本次周中已变为 EXTENDED、BELOW_TRIGGER、UNCONFIRMED 或空状态。数字为当前 Review 范围内的离开数量；点击后只看这些状态恶化或离开买入区的标的。 |
| Other Changes | 除“进入 ACTIONABLE”和“离开 ACTIONABLE”外，其余基线状态与当前状态不同的标的。点击后查看其他状态迁移，并与 Origin、Status 和 Filters 组合。 |
| New | 本次周中 `signal_current=True`，完整周没有有效 Signal。使用 current 行的 candidate、entry 和 route；点击后只看本次新增 Signal。 |
| Carry | 本次周中没有新 Signal，但完整周 Signal 仍被观察，且 code 仍存在于 current_pool。使用完整周 candidate 和 current 价格重算状态；点击后只看延续 Signal。 |
| Reconfirmed | 完整周和本次周中都存在 Signal。本次 current 行整体优先，不使用完整周字段逐列补齐；点击后只看再次确认的 Signal。 |
| ACTIONABLE | 当前 Effective Status 位于现有规则定义的可执行买入区。数字随 Changes / All Signals 范围及其他已选条件更新；点击后叠加 ACTIONABLE 状态筛选。 |
| UNCONFIRMED | 当前 Signal 尚未满足现有统一 Entry 确认条件。数字按当前范围统计；点击后叠加 UNCONFIRMED 状态筛选。 |
| BELOW TRIGGER | 当前价格低于有效 candidate / trigger。数字按当前范围统计；点击后叠加 BELOW_TRIGGER 状态筛选。 |
| EXTENDED | 当前价格已超过现有追价上限。数字按当前范围统计；点击后叠加 EXTENDED 状态筛选。 |

实现规范：

- 使用统一的自定义 Tooltip/Popover，不使用浏览器原生 `title`，避免多行说明被压成一行且无法控制暗色样式；
- 桌面端悬浮约 `250–350ms` 后显示；键盘聚焦时显示；触屏端点击卡片内的 `ⓘ` 显示，但不能触发筛选；
- 大状态卡右上角常驻低对比度 `ⓘ`；紧凑快捷卡片的 `ⓘ` 默认弱化，悬浮或聚焦时增强，避免首屏出现过多视觉噪声；
- Tooltip 采用“标题 + 2～3 行短说明”，最大宽度约 `320px`，允许换行，不遮挡当前卡片数字；
- Tooltip 必须使用当前 Dashboard 的深色背景、细边框和正文层级，不新增一套高饱和颜色；
- Tooltip 只解释，不承载必须点击的操作；鼠标离开、按 `Esc`、切换卡片或滚动时关闭；
- 卡片必须保留原点击筛选行为；`ⓘ` 的点击事件需阻止冒泡；
- 使用 `aria-describedby`、可聚焦触发点和可读文本，保证键盘用户能获得同样说明；
- Tooltip 中的状态名、阈值与统计必须由同一字段配置/状态函数生成，禁止在 UI 中复制第二套业务规则。

### 9.4 Filters 和结果操作区

第一阶段不增加新的复杂筛选字段：

- Route、Distance、Entry Vol、Weekly Vol 和 Reset 原样保留；
- Filters 容器默认收起，只显示紧凑标题栏；
- 周中和周末模式都默认收起，模式切换后也恢复收起；
- 标题栏显示 `No filters applied` 或 `N active`，隐藏时已有条件仍继续生效；
- 用户点击标题栏后展开完整筛选区，展开状态只属于当前 UI 会话；
- `Changes` 是周中默认范围，只排除 `UNCHANGED`；
- `All Signals` 返回全部当前 Pool，并清除 Change、Origin、Status 和展开式筛选；
- 四个状态卡的数量随 `Changes/All Signals` 范围切换，点击后与 Change/Origin 组合；
- 默认排序为 `Change Priority → Entry Status`；
- 用户仍可切换 C Rank 或原有排序方式；
- Copy Codes 复制当前过滤结果，不改变原行为。

### 9.5 Selected Row 与主表

Selected Row 不新增第六个汇总单元格，保持现有五段宽度：

- Code 单元格旁增加 `NEW / CARRY / RECONF.` 小标签；
- Code 下方显示紧凑变化摘要；
- Entry Status 继续显示 `完整周状态 → 当前状态`；
- 变化原因继续使用现有 Entry/Reason 或详情悬浮。

主表只在 `Code` 后增加一列 `Change`，使用组合标签直接表达来源与结果：

```text
NEW → ACTIONABLE
CARRY → ACTIONABLE
ACTIONABLE → EXTENDED
ACTIONABLE → BELOW
NEW · UNCONFIRMED
STILL ACTIONABLE
```

避免同时增加 `Origin`、`Previous Status`、`Current Status` 多个新列，防止破坏现有高密度表格。

### 9.6 周末完整池

切换至 `Weekend Full Pool` 后：

- 隐藏变化/来源快捷栏、Origin 标签和 `Change` 列；
- Snapshot 恢复完整周日期与 freshness；
- Selected Row 恢复原有结构；
- 不应用 Carry，不混入周中价格；
- 原四个状态卡、Filters、排序和 AG Grid 行为保持不变。
- 四个 Entry Status 卡仍保留统一悬浮说明；其数字和点击效果改为完整周 Pool 口径，不显示周中基线/变化描述。

---

## 10. 高保真 UI 规范

本设计附带可交互高保真原型：`BF Midweek Review UI`。原型以当前 Breakout Pool 仪表板为视觉母版，不再使用侧边栏或右侧详情面板。

### 10.1 现有组件映射

| 当前组件 | 周中修改 | Weekend 模式 |
| --- | --- | --- |
| Snapshot 行 | 增加 Midweek 和 baseline 日期 | 原样 |
| Review Queue 标题区 | 模式与 Review 范围分开控制 | 选中 Full Pool + All Signals |
| 快捷筛选 | 一条紧凑 CHANGE/ORIGIN 组合栏 | 隐藏 |
| 四个状态卡 | 样式原样，数量随 Review 范围变化 | 使用完整池数量 |
| Filters | 字段与逻辑不变，容器默认收起并显示启用数量 | 同样默认收起 |
| 结果操作区 | 默认显示变化优先排序 | 原排序 |
| Selected Row | Code 内增加 Origin 和变化摘要，保持五段结构 | 隐藏周中摘要 |
| AG Grid | Code 内增加 Origin，Code 后增加 Change 列 | 隐藏 Origin 与 Change |

其中 `CHANGE`、`ORIGIN` 和四个 Entry Status 的全部卡片必须实现 9.3 定义的统一悬浮说明；这属于心流组件的必需行为，不是可选增强。

### 10.2 颜色与强调

| 信息 | 颜色 |
| --- | --- |
| ACTIONABLE / Became Actionable | 绿色 |
| New Signal | 青色 |
| Carry | 中性灰蓝 |
| Reconfirmed | 弱蓝灰 |
| ACTIONABLE → EXTENDED | 原 EXTENDED 蓝色 |
| ACTIONABLE → BELOW | 原 BELOW_TRIGGER 红色 |
| BELOW_TRIGGER | 红色 |
| EXTENDED | 蓝色 |
| 无变化 | 中性灰 |

约束：

- 颜色必须和文字标签同时出现；
- 变化标签使用左侧细色条和弱背景，不整行高饱和填充；
- `ACTIONABLE` 仍由原状态绿色表达；
- `Carry` 是来源提示，不能盖过当前状态；
- `Left Actionable` 不使用统一橙色，直接使用目标状态颜色；
- 表格列宽增加应最小化，优先使用组合标签。

### 10.3 原型交互

- Midweek Review / Weekend Full Pool 原地切换；
- Midweek 默认进入 Changes，而不是 106 条 All Signals；
- Change、Origin 和 Status 三个维度可组合过滤；
- Change、Origin 和四个 Entry Status 的每张卡片均可悬浮/聚焦查看定义、统计口径与点击效果；
- All Signals 一键回到完整当前 Pool 并清空条件；
- Filters 默认收起，点击标题栏展开/收起；
- 收起时展示有效筛选数量，避免隐藏条件不可见；
- Route、Distance 和 Weekly Vol 继续组合过滤；
- Review Priority / C Rank / Distance 排序；
- 点击表格行更新 Selected Row 汇总；
- Copy Codes 复制当前结果；
- 桌面保持信息密度，窄屏只做换行和水平滚动。

---

## 11. 缓存与性能

数据量在数千行以内，投影可直接使用 pandas。

建议缓存键：

```text
complete 文件修改时间/内容指纹
midweek 文件修改时间/内容指纹
schema version
```

缓存内容：

- `PoolAnalysisResult`；
- summary；
- actionable code。

筛选、排序和选中行属于 UI 状态，不写入投影缓存。

---

## 12. 失败与降级

| 失败 | Dashboard | Futu 接口 |
| --- | --- | --- |
| midweek 不存在 | COMPLETE | 不调用 |
| midweek 过期 | COMPLETE | 不调用 |
| midweek 无效 | COMPLETE + warning | 返回错误，不替换 Futu |
| 完整周无效 | 周中原始 Pool + warning | 只返回本次真实 ACTIONABLE 或 fail closed |
| 无有效基线 | 周中原始 Pool，无 Carry | 本次真实 ACTIONABLE |
| 投影失败 | 周中原始 Pool，无变化字段 | 返回错误，不替换 Futu |

源 CSV 永不因分析失败被修改。

---

## 13. 核心测试

### 13.1 周关联

- 周二至周五解析为 `MIDWEEK_WINDOW`；
- 周六至周一解析为 `COMPLETE_WINDOW`；
- `COMPLETE_WINDOW` 即使存在有效 midweek 文件也不执行投影；
- 周二尚无本周 midweek 时继续使用完整周；
- 周五/六/日/一完整周日期映射正确；
- 上一周期 midweek 自动失效；
- 新一周期完整周漏跑时禁止 Carry；
- 周二至周四完整周日期 fail closed。

### 13.2 投影

- current signal 原子优先；
- Carry 只保留仍在 current_pool 的 code；
- Carry 使用 current 价格重算状态；
- current 退出的完整周 code 只进入 exited_pool；
- 主 Review code 集合严格等于 current_pool；
- 重复 code、无效价格和 candidate 正确失败。

### 13.3 UI

- MIDWEEK 在现有 Breakout Pool 页面内进入变化心流；
- Changes/All Signals 范围切换与结果数量一致；
- Change、Origin、Status 可组合且计数/过滤结果一致；
- 10 张心流卡片均存在统一 Tooltip；悬浮、键盘聚焦和触屏 `ⓘ` 均可读取，且 `ⓘ` 不触发筛选；
- Tooltip 文案与投影字段、状态函数和动态计数一致，不存在 UI 自行复制的第二套规则；
- 周中/完整周切换不混合字段；
- Review 专属筛选不会污染完整周模式；
- Selected Row 汇总条与表格选中同步；
- Weekend 模式隐藏变化/来源快捷栏、Origin 标签和 Change 列；
- COMPLETE/MIDWEEK 首次进入及相互切换后 Filters 均保持默认收起；
- 原 Filters、状态卡和 AG Grid 行为无回归；
- ACTIONABLE code 与 Futu 接口一致。

---

## 14. 验收标准

1. `Yfinance_data` 自动识别 COMPLETE/MIDWEEK；
2. 周二至周五进入周中窗口，周六至周一进入完整周窗口；
3. Review 主表严格执行 current_pool 左表规则，右表不得补回 code 或逐列填充；
4. Carry 与 Effective 状态只存在于内存投影；
5. Dashboard 与 Futu 接口使用同一投影；
6. 过期 midweek 不需要删除即可自动失效；
7. 用户无需离开现有仪表板，即可明确区分本次 Signal、Carry 和状态变化；
8. 用户可一键切回完整周原始 Pool；
9. 投影失败不污染源文件，也不错误替换 Futu；
10. 自动化测试覆盖窗口、周关联、左表、状态和 UI 核心交互。
11. 周中全部变化、来源和 Entry Status 卡片都有可访问的悬浮说明，能够解释定义、统计口径与点击效果。

---

## 15. 非目标

- 不修改 `quant_trade`；
- 不修改通知和其他推送；
- 不修改策略或 resolver；
- 不写回 ReviewResult；
- 不新增 metadata/manifest；
- 不构建历史 midweek 回放；
- 第一阶段不接入交易所节假日日历。
