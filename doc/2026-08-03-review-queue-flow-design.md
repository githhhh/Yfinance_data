# Review Queue 心流优化设计

日期：2026-08-03

## 目标

在不改变 Midweek、Weekend 和 C Rank 数据口径的前提下，消除状态与筛选语义冲突，提升字段可读性、横向比较效率和连续复盘反馈。

公开用户验收以 `1366×768` Chrome 首屏为基准，并覆盖 1280、1366、1440 三种宽度。首屏必须完整显示 Review Queue、筛选入口、结果工具栏和选中行详情；不得依赖横向滚动、裁切、省略号或颜色猜测语义。

## 交互设计

### 数据状态

- 数据契约失败时继续显示 `Schema / Data Error`。
- 数据加载成功时，页头徽标跟随快照时效显示 `Data Fresh`、`Data Aging`、`Data Stale` 或 `Data Loaded`。
- Midweek 有独立快照与有效基线时显示中性的 `Data Loaded`，避免把两个日期简化成单一时效结论。

### 范围与筛选

- 高级筛选入口改为 `More Filters · None` 或 `More Filters · N active`。
- Changes / All Signals 属于主范围，不计入 More Filters。
- 快速 Change / Origin 条件由 `Clear N` 清除；无条件时按钮禁用。
- Clear 不改变 Midweek / Weekend、Changes / All Signals、状态卡或高级筛选。
- 周中快速筛选分为 `WHAT CHANGED` 与 `SIGNAL SOURCE`；1280px 以上保持一行，更窄时允许两组上下排列。
- 六个快速筛选项使用响应式 Grid，数量和说明图标占独立固定槽；不得出现横向滚动、裁切或覆盖。
- Clear 仅在选中快速筛选条件时显示，但无筛选时保留相同布局槽位，避免 Queue 位移。
- Weekend 不显示无效的 Changes 数量；范围区域改为静态 `All Signals · N`。

### 用户可见语言

- 表格、详情、筛选选项和排序摘要使用简短英文标签，不显示内部枚举名。
- 路由示例：`ceiling_pullback` → `Ceiling Pullback`，`ma10_touch_confirm` → `MA10 Touch`。
- 拒绝原因示例：`daily_volume_not_confirmed` → `Volume Not Confirmed`。
- C Rank 标题和规则使用 `Active Signals`、`C Rank · Best First`。
- 中文解释保留在帮助文本或悬浮说明中；底层数据和值不被改写。
- 快速筛选文案固定为 `Entered Buy Zone`、`Left Buy Zone`、`Other Changes`、`New Signal`、`Carried Over`、`Reconfirmed`。
- 快速筛选同时使用颜色、符号和文字：进入为绿色进入箭头，离开为珊瑚红离开箭头，其他变化为橙色变化符号，新信号为青色加号，延续为灰色延续箭头，再确认为紫色确认符号。
- 公开用户术语统一为 `Buy Point`、`Vs Buy Point`、`Setup`、`Weekly Vol`；底层字段名保持不变。
- `C Rank` 与 `Review Priority` 保留英文主文案，并提供简明中文解释。

### C Rank 比较

- 固定顺序为 `Code | C Rank | Continuous C | Status | Vs Candidate | Route | W Vol | Latest`。
- 仅 Code 固定在左侧；C Rank 不再固定在右侧。
- C Rank 与 Continuous C 保持相邻，避免跨屏对照。

### Copy 与连续复盘

- Copy 默认使用深色描边次级样式；成功后短暂显示绿色 `Copied N Codes`。
- 未选中行时显示 `Select a row · Use ↑↓ to review`。
- 选中行时显示 `CODE · X of N`，位置基于当前筛选及排序结果重新计算。
- 每个视图在当前 Streamlit 会话内分别维护 visited code 集合；选择一行即标记，筛选和排序不清除，页面会话重建后清空。
- visited 行只使用轻微背景差异，不改变数据，也不影响排序和筛选。
- Copy 数量始终来自当前可见结果；筛选变化后立即更新，成功反馈继续显示 `Copied N Codes`。

### 首屏布局与默认排序

- 页头顶部留白控制在 28–32px，Header 分隔线到 Review Queue 控制在 18–22px，并减少状态卡到 More Filters 的区块间距，使表格整体上移约 20–30px。
- 周期与范围控件删除勾号，字号为 13–14px，使用选中背景和描边表达状态；弱标签使用 `PERIOD`、`SCOPE`。
- 删除 Copy 右侧排序下拉框。Midweek Changes 固定默认 `Review Priority`；Midweek All Signals、Weekend Full Pool 和 C Rank Reference 固定默认 `C Rank`。其他排序由表头完成，`Vs Buy Point` 表头承担距离排序。
- 选中行详情使用自动高度，目标最小高度 68–76px；禁止固定高度裁切、负 margin、绝对定位或与表格重叠。窄屏字段可换行，容器同步增高。

## 实现边界

- 状态标签、字段值映射和位置计算使用独立纯函数，便于单元测试。
- 表格通过隐藏的 visited 支持列和行样式表达已检查状态。
- 保持现有数据加载、分析、筛选和排序逻辑不变。
- 不修改 `us/`、`results_pkl/`，不新增依赖。

## 验证

- 为时效徽标、显示值映射、C Rank 列固定规则、More Filters、Clear N、位置计算和 visited 行样式添加失败优先测试。
- 运行相关 dashboard 测试、完整 `dashboard/tests`，以及 `dashboard/self_check.py --csv us/breakout_follow_pool.csv`。
- 检查三种视图的标题、摘要和表格列顺序，确认内部字段名不再出现在主界面。
- 在 1280、1366、1440 三种宽度检查 Midweek/Weekend、Changes/All Signals、鼠标/键盘选行、快速筛选无选中/单选/多选/Clear、Copy 数量和表头排序。
