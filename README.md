# Yfinance_data

Public market-data and BreakoutFollow review repository.

This fork extends the upstream data-download workflow with screening, normalized BreakoutFollow pool publication, Midweek/complete-week review projection, and a static GitHub Pages dashboard.

## Main areas

- `DataStore.py`, `*_screener.py` — public market-data download/screening.
- `us/` — published screening and BreakoutFollow pool CSV data.
- `dashboard/` — shared review projection plus the static GitHub Pages UI.
- `doc/` — schema, Midweek projection and Dashboard design contracts.
- `market_analysis` — private submodule reference; its private contents are not part of this public repository.

## Dashboard

Current review-flow and interaction contract:

`doc/STATIC_REVIEW_DASHBOARD_SPEC.md`

Local verification:

```bash
python dashboard/self_check.py \
  --csv us/breakout_follow_pool.csv \
  --midweek-csv us/breakout_follow_pool_midweek.csv
python -m pytest dashboard/tests -q
python dashboard/build_static.py --output /tmp/yfinance-dashboard-site
```

## Public repository security

Everything committed here, including Git history and GitHub Pages output, must be treated as public.

Do not commit brokerage accounts/positions/orders, API keys, OAuth tokens, passwords, private keys, `.env` files, credential JSON, or private research data.

The static Dashboard uses an explicit public field whitelist so new Pool columns are not automatically published. Run the repository scanner when changing credentials/integrations/public output:

```bash
python security_scan.py --history
```

See `SECURITY.md` for the full boundary and incident procedure.
