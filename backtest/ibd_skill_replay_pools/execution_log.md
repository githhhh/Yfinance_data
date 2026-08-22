# Latest Quant Trade Historical Pkl Replay Execution Log

## Scope

- Rebuild complete-week breakout/follow pools from 2025-07-04 to 2026-08-07.
- Use the latest checked-out quant_trade dev logic for pool generation.
- Use git-history pkl blobs from this Yfinance_data repository as point-in-time market data.
- Do not use existing historical pool CSV files as inputs.
- Do not write production `us/` pool files, publish, commit from the strategy pipeline, send Telegram, connect Futu, or update databases.
- Carry `old_pool` chronologically: first replay week cold-starts with an empty set; each successful week provides the next week's old_pool codes.

## Procedure

1. Enumerate complete NYSE weeks from the requested start date up to but excluding the configured production week.
2. For each snapshot week, scan git history commits touching `results_pkl` from the expected close date through the configured search window.
3. In each candidate commit tree, inspect available `stock_data_*_1d.pkl` and `stock_data_*_1wk.pkl` blobs by reading their internal price dates.
4. Select the earliest commit whose daily pkl max date equals the snapshot close and whose weekly pkl max date stays inside the snapshot week without exceeding the close.
5. Load selected pkl blobs directly with `git show <commit>:<path>`, run quant_trade `core_run` in replay mode with the carried old_pool set, then run the IBD entry enrichment helper against the replay output only.
6. Clear current-snapshot EPS values, recompute 52-week-high fields from the selected as-of daily pkl, validate schema/null semantics, and write per-week metadata.

## Commits

- quant_trade repo: `/Users/tbin/Documents/quant_trade`
- quant_trade commit: `bdf2f59a41f09fb649d44bf11b6b223bd7bb77e6`
- Yfinance_data commit at run start: `4ba83b42aaa20949a3b6ecfa03189b44f0a2b57c`

## Weekly Pkl Mapping

| snapshot_date | status | old_pool | new_pool | old_pool_source | pkl_commit | commit_date | daily_pkl | daily_max | weekly_pkl | weekly_max | future_before_clip | clipped | rows |
|---|---|---:|---:|---|---|---|---|---|---|---|---:|---:|---:|
| 2025-07-04 | failed_missing_historical_pkl | 0 | 0 | cold_start |  |  |  | None |  | None | False | False | 0 |
| 2025-07-11 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-07-04 |  |  |  | None |  | None | False | False | 0 |
| 2025-07-18 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-07-11 |  |  |  | None |  | None | False | False | 0 |
| 2025-07-25 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-07-18 |  |  |  | None |  | None | False | False | 0 |
| 2025-08-01 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-07-25 |  |  |  | None |  | None | False | False | 0 |
| 2025-08-08 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-08-01 |  |  |  | None |  | None | False | False | 0 |
| 2025-08-15 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-08-08 |  |  |  | None |  | None | False | False | 0 |
| 2025-08-22 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-08-15 |  |  |  | None |  | None | False | False | 0 |
| 2025-08-29 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-08-22 |  |  |  | None |  | None | False | False | 0 |
| 2025-09-05 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-08-29 |  |  |  | None |  | None | False | False | 0 |
| 2025-09-12 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-09-05 |  |  |  | None |  | None | False | False | 0 |
| 2025-09-19 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-09-12 |  |  |  | None |  | None | False | False | 0 |
| 2025-09-26 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-09-19 |  |  |  | None |  | None | False | False | 0 |
| 2025-10-03 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-09-26 |  |  |  | None |  | None | False | False | 0 |
| 2025-10-10 | success | 0 | 112 | reset_after_missing_pkl:2025-10-03 | dd41b95f69ae | 2025-10-11T05:10:11+00:00 | results_pkl/stock_data_111025_1d.pkl | 2025-10-10 | results_pkl/stock_data_111025_1wk.pkl | 2025-10-06 | False | False | 112 |
| 2025-10-17 | failed_missing_historical_pkl | 112 | 0 | previous_replay_week:2025-10-10 |  |  |  | None |  | None | False | False | 0 |
| 2025-10-24 | success | 0 | 120 | reset_after_missing_pkl:2025-10-17 | 3f2880d3441a | 2025-10-25T08:28:40+00:00 | results_pkl/stock_data_251025_1d.pkl | 2025-10-24 | results_pkl/stock_data_251025_1wk.pkl | 2025-10-20 | False | False | 120 |
| 2025-10-31 | success | 120 | 124 | previous_replay_week:2025-10-24 | 98312a23d0f0 | 2025-11-02T09:50:03+00:00 | results_pkl/stock_data_021125_1d.pkl | 2025-10-31 | results_pkl/stock_data_021125_1wk.pkl | 2025-10-27 | False | False | 124 |
| 2025-11-07 | success | 124 | 113 | previous_replay_week:2025-10-31 | 0468d33bdfa4 | 2025-11-08T01:54:14+00:00 | results_pkl/stock_data_081125_1d.pkl | 2025-11-07 | results_pkl/stock_data_081125_1wk.pkl | 2025-11-03 | False | False | 113 |
| 2025-11-14 | success | 113 | 103 | previous_replay_week:2025-11-07 | 99ffa68215ea | 2025-11-15T08:49:50+00:00 | results_pkl/stock_data_151125_1d.pkl | 2025-11-14 | results_pkl/stock_data_151125_1wk.pkl | 2025-11-10 | False | False | 103 |
| 2025-11-21 | success | 103 | 94 | previous_replay_week:2025-11-14 | 9ab8d8a56cbe | 2025-11-23T11:35:34+00:00 | results_pkl/stock_data_231125_1d.pkl | 2025-11-21 | results_pkl/stock_data_231125_1wk.pkl | 2025-11-17 | False | False | 94 |
| 2025-11-28 | success | 94 | 100 | previous_replay_week:2025-11-21 | 0b4ead26a066 | 2025-11-29T10:16:38+00:00 | results_pkl/stock_data_291125_1d.pkl | 2025-11-28 | results_pkl/stock_data_291125_1wk.pkl | 2025-11-24 | False | False | 100 |
| 2025-12-05 | success | 100 | 101 | previous_replay_week:2025-11-28 | 10585c7bd3f5 | 2025-12-07T10:30:54+00:00 | results_pkl/stock_data_071225_1d.pkl | 2025-12-05 | results_pkl/stock_data_071225_1wk.pkl | 2025-12-01 | False | False | 101 |
| 2025-12-12 | success | 101 | 99 | previous_replay_week:2025-12-05 | 40a54fce10cd | 2025-12-13T14:09:20+00:00 | results_pkl/stock_data_131225_1d.pkl | 2025-12-12 | results_pkl/stock_data_131225_1wk.pkl | 2025-12-08 | False | False | 99 |
| 2025-12-19 | success | 99 | 100 | previous_replay_week:2025-12-12 | 00364a9091f9 | 2025-12-20T13:41:55+00:00 | results_pkl/stock_data_201225_1d.pkl | 2025-12-19 | results_pkl/stock_data_201225_1wk.pkl | 2025-12-15 | False | False | 100 |
| 2025-12-26 | success | 100 | 94 | previous_replay_week:2025-12-19 | 02512d3a22de | 2025-12-27T11:38:15+00:00 | results_pkl/stock_data_271225_1d.pkl | 2025-12-26 | results_pkl/stock_data_271225_1wk.pkl | 2025-12-22 | False | False | 94 |
| 2026-01-02 | success | 94 | 83 | previous_replay_week:2025-12-26 | 512cdf643d7b | 2026-01-04T09:41:31+00:00 | results_pkl/stock_data_040126_1d.pkl | 2026-01-02 | results_pkl/stock_data_040126_1wk.pkl | 2026-01-02 | False | False | 83 |
| 2026-01-09 | success | 83 | 91 | previous_replay_week:2026-01-02 | 48c09cb7a0be | 2026-01-11T03:28:15+00:00 | results_pkl/stock_data_110126_1d.pkl | 2026-01-09 | results_pkl/stock_data_110126_1wk.pkl | 2026-01-05 | False | False | 91 |
| 2026-01-16 | success | 91 | 94 | previous_replay_week:2026-01-09 | cf1eda790d1c | 2026-01-17T03:29:49+00:00 | results_pkl/stock_data_170126_1d.pkl | 2026-01-16 | results_pkl/stock_data_170126_1wk.pkl | 2026-01-12 | False | False | 94 |
| 2026-01-23 | success | 94 | 100 | previous_replay_week:2026-01-16 | 68266829e415 | 2026-01-25T09:14:53+00:00 | results_pkl/stock_data_250126_1d.pkl | 2026-01-23 | results_pkl/stock_data_250126_1wk.pkl | 2026-01-19 | False | False | 100 |
| 2026-01-30 | success | 100 | 100 | previous_replay_week:2026-01-23 | e8f1f6beb7ba | 2026-02-01T10:57:16+00:00 | results_pkl/stock_data_010226_1d.pkl | 2026-01-30 | results_pkl/stock_data_010226_1wk.pkl | 2026-01-26 | False | False | 100 |
| 2026-02-06 | success | 100 | 0 | previous_replay_week:2026-01-30 | 7f69cd69a54e | 2026-02-06T14:03:28+00:00 | results_pkl/stock_data_060226_1d.pkl | 2026-02-06 | results_pkl/stock_data_060226_1wk.pkl | 2026-02-02 | False | False | 0 |
| 2026-02-13 | success | 0 | 119 | previous_replay_week:2026-02-06 | 2686d5dc75b9 | 2026-02-16T08:57:56+00:00 | results_pkl/stock_data_160226_1d.pkl | 2026-02-13 | results_pkl/stock_data_160226_1wk.pkl | 2026-02-09 | False | False | 119 |
| 2026-02-20 | success | 119 | 119 | previous_replay_week:2026-02-13 | f2601d55f997 | 2026-02-23T11:10:59+00:00 | results_pkl/stock_data_230226_1d.pkl | 2026-02-20 | results_pkl/stock_data_230226_1wk.pkl | 2026-02-16 | False | False | 119 |
| 2026-02-27 | success | 119 | 111 | previous_replay_week:2026-02-20 | b47f5aef7255 | 2026-02-28T03:57:36+00:00 | results_pkl/stock_data_280226_1d.pkl | 2026-02-27 | results_pkl/stock_data_280226_1wk.pkl | 2026-02-23 | False | False | 111 |
| 2026-03-06 | success | 111 | 106 | previous_replay_week:2026-02-27 | d71ff76d4b96 | 2026-03-07T07:43:24+00:00 | results_pkl/stock_data_070326_1d.pkl | 2026-03-06 | results_pkl/stock_data_070326_1wk.pkl | 2026-03-02 | False | False | 106 |
| 2026-03-13 | success | 106 | 102 | previous_replay_week:2026-03-06 | 3b937f7b03f0 | 2026-03-14T10:22:09+00:00 | results_pkl/stock_data_140326_1d.pkl | 2026-03-13 | results_pkl/stock_data_140326_1wk.pkl | 2026-03-09 | False | False | 102 |
| 2026-03-20 | success | 102 | 90 | previous_replay_week:2026-03-13 | ba5dfeff7b8b | 2026-03-22T10:51:30+00:00 | results_pkl/stock_data_220326_1d.pkl | 2026-03-20 | results_pkl/stock_data_220326_1wk.pkl | 2026-03-16 | False | False | 90 |
| 2026-03-27 | success | 90 | 92 | previous_replay_week:2026-03-20 | 99aa4f93abdb | 2026-03-29T12:51:08+00:00 | results_pkl/stock_data_290326_1d.pkl | 2026-03-27 | results_pkl/stock_data_290326_1wk.pkl | 2026-03-23 | False | False | 92 |
| 2026-04-02 | success | 92 | 141 | previous_replay_week:2026-03-27 | bc0e12d8a9a4 | 2026-04-03T09:29:57+00:00 | results_pkl/stock_data_030426_1d.pkl | 2026-04-02 | results_pkl/stock_data_030426_1wk.pkl | 2026-03-30 | False | False | 141 |
| 2026-04-10 | success | 141 | 148 | previous_replay_week:2026-04-02 | 0088c5f6c596 | 2026-04-12T10:40:16+00:00 | results_pkl/stock_data_120426_1d.pkl | 2026-04-10 | results_pkl/stock_data_120426_1wk.pkl | 2026-04-06 | False | False | 148 |
| 2026-04-17 | success | 148 | 254 | previous_replay_week:2026-04-10 | d85477135f28 | 2026-04-19T08:14:26+00:00 | results_pkl/stock_data_190426_1d.pkl | 2026-04-17 | results_pkl/stock_data_190426_1wk.pkl | 2026-04-13 | False | False | 254 |
| 2026-04-24 | success | 254 | 380 | previous_replay_week:2026-04-17 | 629496d38cf7 | 2026-04-26T10:58:45+00:00 | results_pkl/stock_data_260426_1d.pkl | 2026-04-24 | results_pkl/stock_data_260426_1wk.pkl | 2026-04-20 | False | False | 380 |
| 2026-05-01 | success | 380 | 451 | previous_replay_week:2026-04-24 | b58507d9dab9 | 2026-05-01T23:06:08+00:00 | results_pkl/stock_data_010526_1d.pkl | 2026-05-01 | results_pkl/stock_data_010526_1wk.pkl | 2026-04-27 | False | False | 451 |
| 2026-05-08 | success | 451 | 513 | previous_replay_week:2026-05-01 | b8528c7ad9b2 | 2026-05-09T00:09:46+00:00 | results_pkl/stock_data_090526_1d.pkl | 2026-05-08 | results_pkl/stock_data_090526_1wk.pkl | 2026-05-04 | False | False | 513 |
| 2026-05-15 | success | 513 | 515 | previous_replay_week:2026-05-08 | 57a5ed9505f0 | 2026-05-16T00:10:59+00:00 | results_pkl/stock_data_160526_1d.pkl | 2026-05-15 | results_pkl/stock_data_160526_1wk.pkl | 2026-05-11 | False | False | 515 |
| 2026-05-22 | success | 515 | 554 | previous_replay_week:2026-05-15 | e67712c4fd7e | 2026-05-23T00:10:26+00:00 | results_pkl/stock_data_230526_1d.pkl | 2026-05-22 | results_pkl/stock_data_230526_1wk.pkl | 2026-05-18 | False | False | 554 |
| 2026-05-29 | success | 554 | 570 | previous_replay_week:2026-05-22 | c6434c1dfeaf | 2026-05-30T09:32:27+00:00 | results_pkl/stock_data_300526_1d.pkl | 2026-05-29 | results_pkl/stock_data_300526_1wk.pkl | 2026-05-25 | False | False | 570 |
| 2026-06-05 | success | 570 | 594 | previous_replay_week:2026-05-29 | d06d396e2911 | 2026-06-06T00:10:47+00:00 | results_pkl/stock_data_060626_1d.pkl | 2026-06-05 | results_pkl/stock_data_060626_1wk.pkl | 2026-06-01 | False | False | 594 |
| 2026-06-12 | success | 594 | 674 | previous_replay_week:2026-06-05 | 77697d555969 | 2026-06-13T00:09:42+00:00 | results_pkl/stock_data_130626_1d.pkl | 2026-06-12 | results_pkl/stock_data_130626_1wk.pkl | 2026-06-08 | False | False | 674 |
| 2026-06-18 | success | 674 | 670 | previous_replay_week:2026-06-12 | 95a7307e8ee3 | 2026-06-20T00:10:18+00:00 | results_pkl/stock_data_200626_1d.pkl | 2026-06-18 | results_pkl/stock_data_200626_1wk.pkl | 2026-06-15 | False | False | 670 |
| 2026-06-26 | success | 670 | 731 | previous_replay_week:2026-06-18 | 1845fa873eb0 | 2026-06-27T00:11:33+00:00 | results_pkl/stock_data_270626_1d.pkl | 2026-06-26 | results_pkl/stock_data_270626_1wk.pkl | 2026-06-22 | False | False | 731 |
| 2026-07-02 | success | 731 | 760 | previous_replay_week:2026-06-26 | d26c0b9aa70e | 2026-07-04T00:10:45+00:00 | results_pkl/stock_data_040726_1d.pkl | 2026-07-02 | results_pkl/stock_data_040726_1wk.pkl | 2026-06-29 | False | False | 760 |
| 2026-07-10 | success | 760 | 716 | previous_replay_week:2026-07-02 | 0aa934244672 | 2026-07-11T00:11:11+00:00 | results_pkl/stock_data_110726_1d.pkl | 2026-07-10 | results_pkl/stock_data_110726_1wk.pkl | 2026-07-06 | False | False | 716 |
| 2026-07-17 | success | 716 | 754 | previous_replay_week:2026-07-10 | 9c931b7da64b | 2026-07-19T06:49:18+00:00 | results_pkl/stock_data_190726_1d.pkl | 2026-07-17 | results_pkl/stock_data_190726_1wk.pkl | 2026-07-13 | False | False | 754 |
| 2026-07-24 | success | 754 | 745 | previous_replay_week:2026-07-17 | 7b00d0c457d8 | 2026-07-25T00:10:31+00:00 | results_pkl/stock_data_250726_1d.pkl | 2026-07-24 | results_pkl/stock_data_250726_1wk.pkl | 2026-07-24 | False | False | 745 |
| 2026-07-31 | success | 745 | 742 | previous_replay_week:2026-07-24 | be2c948b8e55 | 2026-08-03T10:09:00+00:00 | results_pkl/stock_data_030826_1d.pkl | 2026-07-31 | results_pkl/stock_data_030826_1wk.pkl | 2026-07-27 | False | False | 742 |
| 2026-08-07 | success | 742 | 776 | previous_replay_week:2026-07-31 | 8bff577a843f | 2026-08-08T00:11:16+00:00 | results_pkl/stock_data_080826_1d.pkl | 2026-08-07 | results_pkl/stock_data_080826_1wk.pkl | 2026-08-03 | False | False | 776 |
