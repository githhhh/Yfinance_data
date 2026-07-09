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

- **组合筛选逻辑**：自定义筛选（Custom Filter）模式下，所有开启的基础条件与高阶过滤条件采用 `AND`（且）逻辑叠加。
- **交易决策漏斗**：自定义筛选模式废弃了原有的预设（Preset）和独立的排序栏，整体改造为标准交易决策漏斗分组：路径（Route）、入场确认与强度（Entry & Strength）、周线量价（Weekly Vol & Price）、形态结构（Structure）、分类筛选（Grouping）。
- **聚焦生效信号标的**：在自定义筛选模式中，`Route -> IBD Candidate Rule` 默认选择 `All`（收敛于当期生效信号 `signal=True` 的候选标的池）。
- **漏斗折叠栏与摘要**：漏斗折叠面板上方实时显示各分组开启的条件数；当前生效的全部过滤规则会在页面上方与图表容器顶部清晰展示。
- **日内强度联动开关**：日内量价强度筛选条件仅在勾选 `ibd_entry_valid=True`（仅有效突破）时开启调节。
- **突破象限（Breakout Quadrant）**：入场确认与强度部分将收盘位置 `ibd_entry_close_position`（`[0.0, 1.0]`，分界线 `0.70`）与突破量幅比 `ibd_entry_breakout_range_ratio`（`> 0`，分界线 `1.5x`）聚合成了直观的“突破象限”下拉项，便于筛选强力突破（Q1 Power）或洗盘潜伏（Q4 Stealth）等形态。
- **C 级排序参考模式（C Rank Reference）**：该模式会忽略所有自定义筛选条件，先限定 `signal=True`，然后严格按 `rank_C_continuous asc` 连续排位顺序展示。
- **排位计算公式参考**：在 C 级排序模式中，页面表格上方会固定展示本模式采用的固定规则及排名公式参考。
- **表格列展示策略**：结果表格（Result Table）默认展示 `All Fields` 全量业务字段（按业务逻辑归类排序），并且额外包含动态计算出来的基底周期列 `base_duration_weeks`。
- **辅助感知图表**：图表仅用于数据分布辅助感知，接收当前已筛选好的 DataFrame 渲染，互不干扰全局筛选逻辑。
- **默认图表结构**：默认显示三类精炼图表维度：`Route Quality`（触发路径质量）、`Trend × Volume Map`（均线回踩次数与成交量放大散点图），以及全宽显示的行业板块分布图（Sector Concentration）。
