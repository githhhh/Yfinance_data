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

- **交互心流与极简工作台布局**：
  1. **Header Bar**：极简显示 `Breakout Pool`、数据就绪标记、Snapshot 日期及全池统计，右侧提供轻量级 `ⓘ` 流程说明按钮与 `IBD Review` / `C Rank Reference` 模式切换；
  2. **Review Queue（决策状态卡筛）**：合并了状态统计卡与筛选按钮，单行展示 `All Signals` 及四态（`🟢 ACTIONABLE` / `🟡 UNCONFIRMED` / `🔴 BELOW_TRIGGER` / `🔵 EXTENDED`），单击整卡即可完成精准切换；
  3. **Filters（单行质量过滤）**：单行排布 `Route (Rule)` 路径筛选以及 `Distance Min/Max %`、`Entry Vol Min (x)`、`Weekly Vol Min (x)` 输入框；未填写时默认不设限制，并在处于 `UNCONFIRMED` 状态时自动禁用日内量比筛选；
  4. **Selected Row Detail（选股参数带）**：单行极简横向参数带，紧随表格上方实时突出当前所选候选股（或默认第一行）的 Candidate Price、Current vs Candidate、Entry Status 及 C Rank 指标。
- **默认排序逻辑**：按照 `Entry Status`（即 `ACTIONABLE` -> `UNCONFIRMED` -> `BELOW_TRIGGER` -> `EXTENDED` 的业务优先级） -> `rank_C_continuous` 升序 -> `code` 升序排位。
- **决策表格与视图**：核心决策视角（`IBD Review` 模式）下固定提供标准化的 `IBD Decision` 决策字段序列，去除干扰性冗余切换，确保页面高度极简，实现表头与多行决策数据首屏即见。
- **数据源与 Schema 硬校验**：`self_check.py` 与 `data_utils.py` 在加载 CSV 原始数据后第一时间执行 Schema 与物理字段完整性硬校验，所有核心决策字段（如 `latest_close`、`current_vs_ibd_candidate_pct` 等）严格由物理层生成提供，缺失则直接拦截报错，拒绝非物理字段的主观脑补与隐式补全。
