# Breakout Pool Dashboard

`dashboard/` 是 `breakout_follow_pool.csv` / `breakout_follow_pool_midweek.csv` 的静态 IBD Review 页面与共享投影逻辑。

运行时不再依赖 Streamlit。Python 只负责读取权威 Pool、执行既有字段正规化与 Midweek projection，并在 GitHub Actions 中生成静态 JSON；浏览器只负责展示、筛选、排序与交互。

## Review 心流

**状态 → 价格位置 → 日线确认 → 周线量能 → C Rank**

- `ACTIONABLE`：已确认，位于 Buy Point 上方 0%–5%。
- `UNCONFIRMED`：尚未满足日线确认。
- `BELOW TRIGGER`：有效信号当前低于 Buy Point。
- `EXTENDED`：已超过 Buy Point +5%。
- Midweek Review 继续使用完整周 Pool 作为合法 baseline；没有合法 baseline 时 fail closed 到当前周中快照，不启用 Carry / Change / Origin 比较。
- Midweek `Changes` 固定按 `Review Priority`；其它 Review 视图固定按 `C Rank`。
- `C Rank Reference` 保持独立，只对 Active Signals 做横向参考。

## 本地构建

```bash
python dashboard/self_check.py \
  --csv us/breakout_follow_pool.csv \
  --midweek-csv us/breakout_follow_pool_midweek.csv
python -m pytest dashboard/tests -q
python dashboard/build_static.py --output /tmp/yfinance-dashboard-site
python -m http.server 8000 --directory /tmp/yfinance-dashboard-site
```

浏览器访问 `http://localhost:8000` 即可检查与 GitHub Pages 相同的静态产物。

## 部署

`.github/workflows/deploy-review-dashboard.yml` 监听 `main` 上权威 BF Pool 的最终提交以及 Dashboard 自身修改：

```text
quant_trade scheduled run
  → Yfinance_data raw-data update
  → quant_trade BreakoutFollow + IBD enrichment
  → Yfinance_data authority publish / validate
  → pool.commit() pushes breakout_follow_pool*.csv to main
  → Deploy Review Dashboard
  → build_static.py
  → GitHub Pages
```

因此 Pages 没有第二套 weekly/midweek 调度，也不会从原始行情下载 workflow 提前发布半成品。Pool 发布失败时，Pages 保持上一份成功部署的快照。

GitHub Pages 只发布仓库中原本就公开的 BreakoutFollow Pool 派生数据；不得把账户、持仓、API Key 或其它私有交易数据加入静态 payload。
