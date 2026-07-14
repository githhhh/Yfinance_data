# 突破监控池本地分析面板 (Breakout Pool Local Dashboard)

用于 `breakout_follow_pool.csv` 的本地 Streamlit 分析与筛选面板。

## 启动运行

```bash
python dashboard/run_app.py --csv us/breakout_follow_pool.csv
```

## 自测与校验

```bash
python dashboard/self_check.py --csv us/breakout_follow_pool.csv
python -m pytest dashboard/tests -q
```

## 核心逻辑说明

- **组合筛选与交易决策漏斗**：顶部漏斗折叠栏提供标准化决策过滤：
  1. **Route**（路径筛选）：可选 `signal=True` 下的不同触发规则（如 `SMA40W_RECOVERY` 等）；
  2. **Entry Status**（入场状态）：可按四态分类（`ACTIONABLE` / `UNCONFIRMED` / `BELOW_TRIGGER` / `EXTENDED`）实时单选筛选并统计数量；
  3. **Optional Quality Filters**（进阶质量过滤）：动态范围筛选（如距离百分比 `current_vs_ibd_candidate_pct`、突破量比 `ibd_entry_volume_ratio`、周线量比 `volume_ratio`），仅针对当前选择的入场状态进行数值边界推导（当选择 `UNCONFIRMED` 时自动禁用日内量比筛选）。
- **默认排序逻辑**：按照 `ACTIONABLE` -> `UNCONFIRMED` -> `BELOW_TRIGGER` -> `EXTENDED` 的业务优先级排序；同状态内依次按 `latest_close / ibd_candidate_price` 下降（更有利贴近触发点优先）以及 `rank_C_continuous` 升序排位。
- **页面视图（Column View）**：提供 `IBD Decision`（默认）、`IBD Entry`、`Signal`、`Volume/Pullback`、`Reference`、`All Fields` 六种标准化视图切片，仅切换表格展示列，筛选与排位结果保持恒定。
- **辅助感知图表**：`Route Quality` 折叠图表接收经经过路径筛选（Route）后的筛选结果 (`route_df`) 动态渲染。
- **数据源与 Schema 硬校验**：`self_check.py` 与 `data_utils.py` 在加载 CSV 原始数据后第一时间执行 Schema 与物理字段完整性硬校验，所有核心决策字段（如 `latest_close`、`current_vs_ibd_candidate_pct` 等）严格由物理层生成提供，缺失则直接拦截报错，拒绝非物理字段的主观脑补与隐式补全。
