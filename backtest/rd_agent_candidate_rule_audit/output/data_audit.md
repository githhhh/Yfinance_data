# Candidate Event Rule Audit - Data Audit

- Pool directories with CSV: 43
- Non-empty pool weeks: 42
- Pool raw rows: 13055
- Signal ticker-week events after deterministic de-dup: 2738
- Unique signal tickers: 1106
- Duplicate snapshot/code rows: 0
- ACTIONABLE/UNCONFIRMED/EXTENDED: 733/1529/476
- PIT EPS verified/blocked-or-unknown: 2368/370
- Complete 8w labels: 1111

The local schema document is a migration pointer to the quant_trade SSOT. Field meanings not inferable from repository consumers remain schema blockers for production hard gates.

Coverage regimes are assigned from observed pool row count: weeks with at least 500 raw rows are `late_high_coverage`; earlier weeks are `early_low_coverage`.
