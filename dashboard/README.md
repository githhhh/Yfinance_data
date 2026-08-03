# Breakout Pool Dashboard

用于 `breakout_follow_pool.csv` 的 IBD 候选复盘与优先级筛选。

## 启动

```bash
python dashboard/run_app.py --csv us/breakout_follow_pool.csv
```

## Review 心流

**状态 → 价格位置 → 日线确认 → 周线量能 → C Rank 对照**

1. **选状态**
   - `ACTIONABLE`：已确认，位于 Candidate 上方 0%–5%，优先 Review。
   - `UNCONFIRMED`：尚未满足日线确认；优先查看 `Near Trigger ≤ +3%`。
   - `BELOW TRIGGER`：曾有效确认，但当前回到 Candidate 下方。
   - `EXTENDED`：已超过 Candidate +5%，避免追高。

2. **看价格位置**
   - `Vs Candidate` 显示最新收盘价相对 Candidate Price 的距离。

3. **看突破确认**
   - `Entry / Reason`：日线突破确认。成功显示日线量比，未确认显示原因。

4. **看周线量能**
   - `W Vol`：当前周成交量相对 10 周均量的倍数。

5. **用 C Rank 对照**
   - `C Rank` 越小越靠前，仅用于同状态候选之间的质量对照。

## 操作

- 点击状态卡进入对应 Review 队列；各筛选条件按 **AND** 组合。
- 点击表格行的其他单元格，更新上方 Selected Row Detail。
- 点击 `Code` 只复制单个股票代码，不切换选中行。
- 悬停 Selected Code 查看 CANSLIM / Base、Pullback 与 Daily Entry 详情。
- `Copy N Codes` 按当前筛选及排序复制全部代码；失败时使用 `Manual`。
- `C Rank Reference` 是独立对照视图，不改变 IBD Review 的筛选状态。

## 自检

```bash
python dashboard/self_check.py --csv us/breakout_follow_pool.csv
python -m pytest dashboard/tests -q
```

数据契约通过后，完整周视图按快照时效显示 `Data Fresh`、`Data Aging` 或 `Data Stale`；Midweek 双日期上下文显示中性的 `Data Loaded`。
