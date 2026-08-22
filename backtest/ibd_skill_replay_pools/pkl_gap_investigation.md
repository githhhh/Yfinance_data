# Historical Pkl Gap Investigation

## Scope

- Replay request range: 2025-07-04 through 2026-08-07, excluding 2026-08-14.
- Clean replay result: 58 weeks audited, 43 successful pool CSV weeks, 15 recorded historical pkl gap weeks.
- Failed gap weeks: 2025-07-04, 2025-07-11, 2025-07-18, 2025-07-25, 2025-08-01, 2025-08-08, 2025-08-15, 2025-08-22, 2025-08-29, 2025-09-05, 2025-09-12, 2025-09-19, 2025-09-26, 2025-10-03, 2025-10-17.

## Selection Policy

- Replay uses the same live-style quant_trade inputs: separate daily and weekly pkl payloads.
- Accepted historical files must match `stock_data_DDMMYY_1d.pkl` and `stock_data_DDMMYY_1wk.pkl`.
- Daily pkl max date must equal the snapshot week's expected last trading day.
- Weekly pkl max date must fall inside the snapshot week and must not exceed the expected last trading day.
- `old_pool` carries forward only from the immediately previous successful replay week; after a missing pkl week it resets instead of jumping over the gap.

## Findings

- 2025-07 through 2025-09 git history mostly contains legacy untyped files such as `stock_data_010825.pkl` or timestamped single files such as `stock_data_110725_040017.pkl`.
- Legacy untyped pkl spot checks loaded as daily OHLCV dictionaries without a weekly-period marker; the sample universe also included non-US symbols, so these files were not mixed into the clean US daily/weekly replay data source.
- A 30-day candidate search around missing 2025-07/2025-08 weeks found no acceptable `_1d`/`_1wk` pair.
- 2025-10-03 and 2025-10-17 did have new-format candidates, but the best daily pkl max dates were 2025-10-02 and 2025-10-16 respectively, so they were not complete-Friday snapshots.

## Decision

- No synthetic weekly resampling or legacy single-pkl inference was used.
- Missing weeks are retained as auditable metadata gaps, not as pool CSV inputs.
- The research dataset should use the 43 successful clean replay pool CSV weeks plus the 15 explicit gap records.
