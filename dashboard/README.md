# Breakout Pool Dashboard

`dashboard/` 是 `breakout_follow_pool.csv` / `breakout_follow_pool_midweek.csv` 的静态 IBD Review 页面与共享投影逻辑。

当前 UI/心流规范以 [`doc/STATIC_REVIEW_DASHBOARD_SPEC.md`](../doc/STATIC_REVIEW_DASHBOARD_SPEC.md) 为准；旧 Streamlit / AG Grid 文档只作为历史设计参考。

运行时不依赖 Streamlit。Python 负责读取权威 Pool、执行字段正规化与 Midweek projection，并生成静态 JSON；浏览器只负责展示、筛选、排序、选行、复制与响应式交互。

## Review 心流

```text
数据状态
→ Period / Scope
→ Change / Origin / Entry Status
→ More Filters（按需）
→ Results / 表头排序 / Copy Codes
→ Selected Detail
→ 连续表格 Review
```

- `ACTIONABLE`：已确认，位于 Buy Point 上方 0%–5%。
- `UNCONFIRMED`：尚未满足日线确认。
- `BELOW TRIGGER`：有效信号当前低于 Buy Point。
- `EXTENDED`：已超过 Buy Point +5%。
- Midweek Review 使用合法完整周 Pool 作为 baseline；没有合法 baseline 时关闭 Carry / Change / Origin 比较。
- Midweek `Changes` 默认按 `Review Priority`；其它 Review 视图默认按 `C Rank`。
- 默认排序只是入口；所有可见表头支持点击升/降序。
- `Breakout Price Quality` 表头保留强度说明，质量计算仍完全来自 Python 权威层。
- `C Rank Reference` 独立，只对 Active Signals 做横向参考。

## 本地构建与验证

```bash
python dashboard/self_check.py \
  --csv us/breakout_follow_pool.csv \
  --midweek-csv us/breakout_follow_pool_midweek.csv
python -m pytest dashboard/tests -q
python dashboard/build_static.py --output /tmp/yfinance-dashboard-site
python security_scan.py --history
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

## Public payload 安全边界

GitHub Pages 是公网资源。`dashboard/build_static.py` 通过 `PUBLIC_DASHBOARD_ROW_FIELDS` 显式白名单输出行字段：

- Pool 新增列不会自动进入 `dashboard.json`；
- 新字段只有在 UI 明确消费且确认可公开后才加入白名单；
- 不得把账户、持仓、成本、订单、broker account hash、API Key、OAuth Token 或其它私有交易数据加入静态 payload。

完整仓库安全约束见 [`SECURITY.md`](../SECURITY.md)。
