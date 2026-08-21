# 改进：固定窗口收益标准化 — 消除路径长度偏差

## 问题描述

当前 `evaluate_weekly_signal_oracle.py` 使用 `END_DATE = "2026-08-14"` 计算所有 pool 的 `latest_return`。这导致：

- **2026-01-02 pool** 的收益窗口 ≈ 32 周（225 个交易日），Top5 绝对收益可达 +108%
- **2026-08-07 pool** 的收益窗口 ≈ 1 周（5 个交易日），Top5 绝对收益仅 +1%

这造成了**系统性的路径长度偏差**：

1. 早期 pool 天然产生更大的绝对收益差异，Winner Top5 排名更明确
2. 晚期 pool 收益差异被压缩到噪声级别，Top5 命中几乎不可能
3. 综合指标（`week_top5_hit_rate`）被早期周主导，晚期周实质上是分母噪声

### 实际数据佐证

从 `with_eps_weekly_report.md` best variant (`skill_industry_eps_known`) 逐周明细：

- **2026-01 ~ 2026-04（15 周）**：hit_latest_top5_count 合计 **8 次**，含 DELL +108%、NYAX +96%、MRVL +74%
- **2026-05 ~ 2026-08（16 周）**：hit_latest_top5_count 合计 **0 次**，最高周均收益仅 +27%

**结论：当前 Top5 命中率 25.8% 本质上是 8/31 ≈ 25.8%，全部由前半段贡献。**

## 改进方案

在现有 `latest_return`（至 END_DATE）基础上，增加 **固定窗口 forward return** 作为补充评估维度。

### 窗口定义

| 窗口 | 交易日数 | 含义 | 字段后缀 |
|---|---|---|---|
| 4W | 20 个交易日 | 短期动量兑现 | `_4w` |
| 8W | 40 个交易日 | 中期趋势持续 | `_8w` |
| 12W | 60 个交易日 | 季度级确认 | `_12w` |
| 至今 | 到 END_DATE | 保持现有逻辑不变 | `_latest`（现有） |

### 需要修改的文件

#### 1. `evaluate_weekly_signal_oracle.py`

##### 新增常量

```python
FORWARD_WINDOWS = {"4w": 20, "8w": 40, "12w": 60}
```

##### 修改 `build_signal_universe()`

对每个 signal 行，除了现有的 `compute_path_metrics(..., end_date=END_DATE)` 之外，对每个固定窗口也计算一组 path metrics。

计算固定窗口的 `end_date` 方式：从 `snapshot_date` 起，在价格 bars 中向前数 N 个交易日，取第 N 个交易日的日期作为窗口 `end_date`。如果价格数据不足 N 个交易日，该窗口的收益记为 `None`。

universe 行增加字段：
- `latest_return_pct_4w`, `max_gain_pct_4w`, `max_drawdown_pct_4w`, `hit_stop_8pct_4w`
- `latest_return_pct_8w`, `max_gain_pct_8w`, `max_drawdown_pct_8w`, `hit_stop_8pct_8w`
- `latest_return_pct_12w`, `max_gain_pct_12w`, `max_drawdown_pct_12w`, `hit_stop_8pct_12w`

##### 新增辅助函数 `nth_trading_day()`

```python
def nth_trading_day(price_bars: pd.DataFrame, snapshot_date: str, n: int) -> str | None:
    """返回 snapshot_date 起第 n 个交易日的日期字符串。数据不足则返回 None。"""
```

从 `price_bars` 中筛选 `index >= snapshot_date` 的行，如果行数 >= n，返回第 n 行的日期；否则返回 None。

##### 修改 `add_oracle_ranks()`

对每个窗口分别做 groupby + rank，生成：
- `latest_rank_4w`, `gain_rank_4w`, `loss_rank_4w`
- `latest_rank_8w`, `gain_rank_8w`, `loss_rank_8w`
- `latest_rank_12w`, `gain_rank_12w`, `loss_rank_12w`

保持现有的 `latest_rank` / `gain_rank` / `loss_rank`（基于至今收益）不变。

**注意**：某些 snapshot 的部分标的在短窗口内可能缺少价格数据（`latest_return_pct_4w` 为 None），排名时应仅基于有有效值的标的。

##### 修改 picks 的命中判定

对每个 pick，除了现有的 `hit_latest_top5` 等，增加：
- `hit_latest_top5_4w`, `hit_latest_top5_8w`, `hit_latest_top5_12w`
- `hit_gain_top5_4w`, `hit_gain_top5_8w`, `hit_gain_top5_12w`
- `hit_loss_bottom5_4w`, `hit_loss_bottom5_8w`, `hit_loss_bottom5_12w`

##### 修改 weekly 聚合和 summary

对每个窗口单独计算一组汇总指标，输出到 summary 的新列：
- `week_latest_top5_hit_rate_4w`, `week_latest_top5_hit_rate_8w`, `week_latest_top5_hit_rate_12w`
- 对应的 bottom5、stop、pick 级别指标也按窗口各算一组

评分函数保持现有逻辑作为 `score_latest`。对每个固定窗口，用相同的权重公式计算 `score_4w`, `score_8w`, `score_12w`。

##### 修改报告渲染

在每个模式报告（`render_mode_report`）中新增一个 section：

```markdown
## 固定窗口收益对比

| variant | score_latest | score_4w | score_8w | score_12w | 稳定性 |
```

「稳定性」列 = 各窗口 score 的变异系数 CV（标准差/均值），CV 越低越稳定。

#### 2. 输出文件变更

- `*_signal_universe_oracle.csv`：增加固定窗口收益列和排名列
- `*_variant_summary.csv`：增加固定窗口的汇总指标列和 score 列
- `*_weekly_report.md`：增加「固定窗口收益对比」section
- `combined_variant_summary.csv` / `run_log.md`：增加固定窗口 score 列

#### 3. 不变的部分

- `backtest/ibd_skill_replay/core.py` 中的 `compute_path_metrics()` 不需要修改。固定窗口的 `end_date` 由调用方计算后传入即可。
- `backtest/ibd_skill_iteration/core.py` 中的 `rank_reasoning_candidates()` 不需要修改。推荐生成逻辑不依赖收益窗口。
- 现有 `latest_return`（至 END_DATE）的全部逻辑和输出保持不变，新增维度是**追加**而非替代。
- SKILL.md 不需要修改。

## 验证标准

1. **回归测试**：现有输出（`*_variant_summary.csv` 中 `score` 列，`*_weekly_report.md` 中 variant 总结表）不变。新增列是追加的。
2. **窗口有效性**：`2026-08-07` pool 的 4W 窗口（到 ~2026-09-04）在价格数据中是否有足够交易日？如果 `END_DATE = "2026-08-14"` 之后无数据，则该 pool 的 4W/8W/12W 返回 None 是正确的。
3. **文档更新**：`run_log.md` 的 Step Logic 部分增加固定窗口说明。
4. **结果提交**：重跑脚本后，将 `/private/tmp/ibd_weekly_signal_oracle_eval/` 下的输出复制到 `backtest/ibd_weekly_signal_oracle_eval/` 并提交。

## 预期效果

- 如果 skill 推荐在固定窗口下仍有显著的 Top5 命中率，说明 skill 质量是真实的
- 如果固定窗口 score 显著低于 latest score，说明当前的高评分主要来自路径长度偏差
- 固定窗口结果更适合跨周比较，因为每周的评估标准一致
