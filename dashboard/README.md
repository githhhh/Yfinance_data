# Breakout Pool Dashboard

`dashboard/` 是 `breakout_follow_pool.csv` / `breakout_follow_pool_midweek.csv` 的静态 IBD Review 页面与共享投影逻辑。

当前且唯一的 UI / 心流 / 交互规范是 [`doc/STATIC_REVIEW_DASHBOARD_SPEC.md`](../doc/STATIC_REVIEW_DASHBOARD_SPEC.md)。过时 Dashboard 方案直接删除，需要追溯时使用 Git 历史，不保留平行规范。

运行时不依赖 Streamlit。Python 负责读取权威 Pool、执行字段正规化、Midweek projection，以及可失效的当前 RS reference enrichment；浏览器只负责展示、筛选、排序、选行、复制与响应式交互。

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
- Midweek `Changes` 默认按 `Review Priority`；其它 Review 默认按 `Code`。
- 主表 `RS` 只引用 `Fred6725/rs-log` 最新公开数据；只有 RS market date 与 Pool `snapshot_date` 严格一致才显示，否则 `N/A`。
- RS 只是 context，不进入 Gate、Top3、Review Priority 或默认排序，也不保存为本仓库 PIT 历史。
- `Breakout Price Quality` 仍由 Python 权威层生成，表头保留强度说明。
- More Filters Range 使用当前 Period / Scope / Change / Origin / Status / Setup 语境下的实际数据边界；完整范围显示 `Full range`。
- 表格横纵两个方向允许滚动，但表格自身到边界时关闭 overscroll / bounce。

## 本地构建与验证

```bash
python dashboard/self_check.py \
  --csv us/breakout_follow_pool.csv \
  --midweek-csv us/breakout_follow_pool_midweek.csv
python -m pytest dashboard/tests -q
node --check dashboard/app.js
node --check dashboard/table_enhancements.js
python dashboard/build_static.py --output /tmp/yfinance-dashboard-site
python security_scan.py --history
python -m http.server 8000 --directory /tmp/yfinance-dashboard-site
```

浏览器访问 `http://localhost:8000` 即可检查与 GitHub Pages 相同的静态产物。`build_static.py` 获取 RS 失败、RS 日期不匹配或 ticker 缺失时仍正常构建，对应 RS 显示 `N/A`。

## 部署

`.github/workflows/deploy-review-dashboard.yml` 监听 `main` 上权威 BF Pool 的最终提交以及 Dashboard 自身修改：

```text
quant_trade scheduled run
  → Yfinance_data raw-data update
  → quant_trade BreakoutFollow + IBD enrichment
  → Yfinance_data authority publish / validate
  → pool.commit() pushes breakout_follow_pool*.csv to main
  → Deploy Review Dashboard
  → optional current rs-log enrichment
  → build_static.py
  → GitHub Pages
```

Pool push 仍是数据发布权威触发。由于 `rs-log` 通常在美股收盘后的约 `01:30 UTC` 才发布对应市场日 RS，而周末 Pool 可能更早提交，Dashboard 额外在 **周四、周六 `03:00 UTC`** 重建一次当前已发布 Pool，只用于补齐 exact-date RS reference。这个定时任务不会重新下载行情、不会重新计算 Pool、不会改变 `snapshot_date`，也不是第二套 weekly / midweek 数据调度。

因此 Pages 不会从原始行情下载 workflow 提前发布半成品。Pool 发布失败时，Pages 保持上一份成功部署的快照；RS 外部源失败不会阻塞部署，后续 Pool push 或定时 refresh 可再次补齐。

## Public payload 安全边界

GitHub Pages 是公网资源。`dashboard/build_static.py` 通过 `PUBLIC_DASHBOARD_ROW_FIELDS` 显式白名单输出行字段：

- Pool 新增列不会自动进入 `dashboard.json`；
- RS 只发布当前 percentile 和 1M / 3M / 6M ago percentile；
- RS 公共数据获取不使用仓库 Token、API Key 或其它凭据；
- 不得把账户、持仓、成本、订单、broker account hash、API Key、OAuth Token 或其它私有交易数据加入静态 payload。

完整仓库安全约束见 [`SECURITY.md`](../SECURITY.md)。
