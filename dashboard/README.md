# Breakout Pool Dashboard

`dashboard/` 是 `breakout_follow_pool.csv` / `breakout_follow_pool_midweek.csv` 的静态 IBD Review 页面与共享投影逻辑。

当前且唯一的 UI / 心流 / 交互规范是 [`doc/STATIC_REVIEW_DASHBOARD_SPEC.md`](../doc/STATIC_REVIEW_DASHBOARD_SPEC.md)。过时 Dashboard 方案直接删除，需要追溯时使用 Git 历史，不保留平行规范。

运行时不依赖 Streamlit。Python 只负责读取权威 Pool、字段正规化和 Midweek projection；浏览器负责展示、筛选、排序、选行、复制与响应式交互。RS 是浏览器加载页面后独立拉取的公开参考信息，不进入 Python Pool projection 或 Pages 构建流程。

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
- 主表 `RS` 在页面主体已经加载后，由 `rs_runtime.js` 从 [`Fred6725 / rs-log`](https://github.com/Fred6725/rs-log) 拉取最新公开数据。
- RS 只是 context，不进入 Gate、Top3、Review Priority、默认排序或 `dashboard.json`，也不保存为本仓库 PIT 历史。
- RS 表头承担来源、更新时间、与 Pool snapshot 的时间状态以及 `Refresh / Retry`；RS cell 只显示当前 percentile，点击仍按普通表格行为选中该行；Selected Detail 显示当前 / 1M / 3M / 6M。
- 首次加载显示 `—`；RS 比 Pool 旧或新都继续显示并在表头标明状态；拉取失败或 ticker 不存在时显示 `N/A`。所有状态都不阻塞也不刷新主体 Pages。
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
node --check dashboard/rs_runtime.js
python dashboard/build_static.py --output /tmp/yfinance-dashboard-site
python security_scan.py --history
python -m http.server 8000 --directory /tmp/yfinance-dashboard-site
```

浏览器访问 `http://localhost:8000` 即可检查与 GitHub Pages 相同的静态产物。静态站点构建本身不访问 RS；即使 `rs-log` 不可用，Dashboard 主体仍正常发布和加载，RS 位置显示 `N/A` 并可从表头执行 Retry。

## 部署

`.github/workflows/deploy-review-dashboard.yml` 只监听 `main` 上权威 BF Pool 的最终提交以及 Dashboard 自身修改：

```text
quant_trade scheduled run
  → Yfinance_data raw-data update
  → quant_trade BreakoutFollow + IBD enrichment
  → Yfinance_data authority publish / validate
  → pool.commit() pushes breakout_follow_pool*.csv to main
  → Deploy Review Dashboard
  → build_static.py
  → GitHub Pages

browser opens Pages
  → render authoritative dashboard.json first
  → optional fetch Fred6725/rs-log
  → fill RS reference cells only
```

因此 Pool / Pages 发布链路完全不依赖 RS。没有 RS schedule、没有 RS-only Pages refresh、没有 RS date gate。RS 无论正常、落后、缺失或请求失败，都只影响 RS 自己的显示状态。

## Public payload 安全边界

GitHub Pages 是公网资源。`dashboard/build_static.py` 通过 `PUBLIC_DASHBOARD_ROW_FIELDS` 显式白名单输出行字段：

- Pool 新增列不会自动进入 `dashboard.json`；
- `dashboard.json` 不包含 C Rank / Continuous C，也不包含 RS percentile；
- RS 数据由浏览器直接访问公开 GitHub API / raw 内容，不经过本仓库 Token、API Key 或其它凭据；
- 不得把账户、持仓、成本、订单、broker account hash、API Key、OAuth Token 或其它私有交易数据加入静态 payload。

完整仓库安全约束见 [`SECURITY.md`](../SECURITY.md)。
