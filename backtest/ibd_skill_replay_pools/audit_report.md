# Latest Quant Trade Replay Pool Audit

- Quant trade commit: `bdf2f59a41f09fb649d44bf11b6b223bd7bb77e6`
- Weeks processed: 58
- Success weeks: 43
- Failed weeks: 15
- Weeks using clipped data: 0
- Chronological boundary check: passed (first=2025-07-04, last=2026-08-07, excluded_2026_08_14=True)
- Replay old_pool carry-forward check: passed
- Future-date leak check after clip: passed
- Schema check on successful pool weeks: passed
- Missing historical pkl weeks recorded as data gaps: 15
- Historical git pkl source check: passed
- IBD resolver field check: passed (signal_candidates=2738, ibd_entry_valid_nonempty=2738, valid_entries=1209, valid_entry_price_nonempty=1209, invalid_entries=1529, invalid_reject_nonempty=1529)
- Side-effect isolation check: passed
- Production pool write/publish/commit/Futu/Telegram/database side effects: disabled by replay wrapper.
- Old `ibd_skill_replay_pools` contents are treated as untrusted and replaced by this clean replay baseline.

## Week Status

| snapshot_date | status | old_pool | new_pool | old_pool_source | rows | data_source | pkl_commit | daily_pkl | weekly_pkl | daily_max | weekly_max | clipped | schema | failure_reason |
|---|---:|---:|---:|---|---:|---|---|---|---|---|---|---:|---|---|
| 2025-07-04 | failed_missing_historical_pkl | 0 | 0 | cold_start | 0 | historical_git |  |  |  | None | None | False | failed_missing_historical_pkl | No git-history daily/weekly pkl pair with matching internal as-of dates |
| 2025-07-11 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-07-04 | 0 | historical_git |  |  |  | None | None | False | failed_missing_historical_pkl | No git-history daily/weekly pkl pair with matching internal as-of dates |
| 2025-07-18 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-07-11 | 0 | historical_git |  |  |  | None | None | False | failed_missing_historical_pkl | No git-history daily/weekly pkl pair with matching internal as-of dates |
| 2025-07-25 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-07-18 | 0 | historical_git |  |  |  | None | None | False | failed_missing_historical_pkl | No git-history daily/weekly pkl pair with matching internal as-of dates |
| 2025-08-01 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-07-25 | 0 | historical_git |  |  |  | None | None | False | failed_missing_historical_pkl | No git-history daily/weekly pkl pair with matching internal as-of dates |
| 2025-08-08 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-08-01 | 0 | historical_git |  |  |  | None | None | False | failed_missing_historical_pkl | No git-history daily/weekly pkl pair with matching internal as-of dates |
| 2025-08-15 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-08-08 | 0 | historical_git |  |  |  | None | None | False | failed_missing_historical_pkl | No git-history daily/weekly pkl pair with matching internal as-of dates |
| 2025-08-22 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-08-15 | 0 | historical_git |  |  |  | None | None | False | failed_missing_historical_pkl | No git-history daily/weekly pkl pair with matching internal as-of dates |
| 2025-08-29 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-08-22 | 0 | historical_git |  |  |  | None | None | False | failed_missing_historical_pkl | No git-history daily/weekly pkl pair with matching internal as-of dates |
| 2025-09-05 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-08-29 | 0 | historical_git |  |  |  | None | None | False | failed_missing_historical_pkl | No git-history daily/weekly pkl pair with matching internal as-of dates |
| 2025-09-12 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-09-05 | 0 | historical_git |  |  |  | None | None | False | failed_missing_historical_pkl | No git-history daily/weekly pkl pair with matching internal as-of dates |
| 2025-09-19 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-09-12 | 0 | historical_git |  |  |  | None | None | False | failed_missing_historical_pkl | No git-history daily/weekly pkl pair with matching internal as-of dates |
| 2025-09-26 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-09-19 | 0 | historical_git |  |  |  | None | None | False | failed_missing_historical_pkl | No git-history daily/weekly pkl pair with matching internal as-of dates |
| 2025-10-03 | failed_missing_historical_pkl | 0 | 0 | reset_after_missing_pkl:2025-09-26 | 0 | historical_git |  |  |  | None | None | False | failed_missing_historical_pkl | No git-history daily/weekly pkl pair with matching internal as-of dates |
| 2025-10-10 | success | 0 | 112 | reset_after_missing_pkl:2025-10-03 | 112 | historical_git | dd41b95f | stock_data_111025_1d.pkl | stock_data_111025_1wk.pkl | 2025-10-10 | 2025-10-06 | False | passed |  |
| 2025-10-17 | failed_missing_historical_pkl | 112 | 0 | previous_replay_week:2025-10-10 | 0 | historical_git |  |  |  | None | None | False | failed_missing_historical_pkl | No git-history daily/weekly pkl pair with matching internal as-of dates |
| 2025-10-24 | success | 0 | 120 | reset_after_missing_pkl:2025-10-17 | 120 | historical_git | 3f2880d3 | stock_data_251025_1d.pkl | stock_data_251025_1wk.pkl | 2025-10-24 | 2025-10-20 | False | passed |  |
| 2025-10-31 | success | 120 | 124 | previous_replay_week:2025-10-24 | 124 | historical_git | 98312a23 | stock_data_021125_1d.pkl | stock_data_021125_1wk.pkl | 2025-10-31 | 2025-10-27 | False | passed |  |
| 2025-11-07 | success | 124 | 113 | previous_replay_week:2025-10-31 | 113 | historical_git | 0468d33b | stock_data_081125_1d.pkl | stock_data_081125_1wk.pkl | 2025-11-07 | 2025-11-03 | False | passed |  |
| 2025-11-14 | success | 113 | 103 | previous_replay_week:2025-11-07 | 103 | historical_git | 99ffa682 | stock_data_151125_1d.pkl | stock_data_151125_1wk.pkl | 2025-11-14 | 2025-11-10 | False | passed |  |
| 2025-11-21 | success | 103 | 94 | previous_replay_week:2025-11-14 | 94 | historical_git | 9ab8d8a5 | stock_data_231125_1d.pkl | stock_data_231125_1wk.pkl | 2025-11-21 | 2025-11-17 | False | passed |  |
| 2025-11-28 | success | 94 | 100 | previous_replay_week:2025-11-21 | 100 | historical_git | 0b4ead26 | stock_data_291125_1d.pkl | stock_data_291125_1wk.pkl | 2025-11-28 | 2025-11-24 | False | passed |  |
| 2025-12-05 | success | 100 | 101 | previous_replay_week:2025-11-28 | 101 | historical_git | 10585c7b | stock_data_071225_1d.pkl | stock_data_071225_1wk.pkl | 2025-12-05 | 2025-12-01 | False | passed |  |
| 2025-12-12 | success | 101 | 99 | previous_replay_week:2025-12-05 | 99 | historical_git | 40a54fce | stock_data_131225_1d.pkl | stock_data_131225_1wk.pkl | 2025-12-12 | 2025-12-08 | False | passed |  |
| 2025-12-19 | success | 99 | 100 | previous_replay_week:2025-12-12 | 100 | historical_git | 00364a90 | stock_data_201225_1d.pkl | stock_data_201225_1wk.pkl | 2025-12-19 | 2025-12-15 | False | passed |  |
| 2025-12-26 | success | 100 | 94 | previous_replay_week:2025-12-19 | 94 | historical_git | 02512d3a | stock_data_271225_1d.pkl | stock_data_271225_1wk.pkl | 2025-12-26 | 2025-12-22 | False | passed |  |
| 2026-01-02 | success | 94 | 83 | previous_replay_week:2025-12-26 | 83 | historical_git | 512cdf64 | stock_data_040126_1d.pkl | stock_data_040126_1wk.pkl | 2026-01-02 | 2026-01-02 | False | passed |  |
| 2026-01-09 | success | 83 | 91 | previous_replay_week:2026-01-02 | 91 | historical_git | 48c09cb7 | stock_data_110126_1d.pkl | stock_data_110126_1wk.pkl | 2026-01-09 | 2026-01-05 | False | passed |  |
| 2026-01-16 | success | 91 | 94 | previous_replay_week:2026-01-09 | 94 | historical_git | cf1eda79 | stock_data_170126_1d.pkl | stock_data_170126_1wk.pkl | 2026-01-16 | 2026-01-12 | False | passed |  |
| 2026-01-23 | success | 94 | 100 | previous_replay_week:2026-01-16 | 100 | historical_git | 68266829 | stock_data_250126_1d.pkl | stock_data_250126_1wk.pkl | 2026-01-23 | 2026-01-19 | False | passed |  |
| 2026-01-30 | success | 100 | 100 | previous_replay_week:2026-01-23 | 100 | historical_git | e8f1f6be | stock_data_010226_1d.pkl | stock_data_010226_1wk.pkl | 2026-01-30 | 2026-01-26 | False | passed |  |
| 2026-02-06 | success | 100 | 0 | previous_replay_week:2026-01-30 | 0 | historical_git | 7f69cd69 | stock_data_060226_1d.pkl | stock_data_060226_1wk.pkl | 2026-02-06 | 2026-02-02 | False | passed |  |
| 2026-02-13 | success | 0 | 119 | previous_replay_week:2026-02-06 | 119 | historical_git | 2686d5dc | stock_data_160226_1d.pkl | stock_data_160226_1wk.pkl | 2026-02-13 | 2026-02-09 | False | passed |  |
| 2026-02-20 | success | 119 | 119 | previous_replay_week:2026-02-13 | 119 | historical_git | f2601d55 | stock_data_230226_1d.pkl | stock_data_230226_1wk.pkl | 2026-02-20 | 2026-02-16 | False | passed |  |
| 2026-02-27 | success | 119 | 111 | previous_replay_week:2026-02-20 | 111 | historical_git | b47f5aef | stock_data_280226_1d.pkl | stock_data_280226_1wk.pkl | 2026-02-27 | 2026-02-23 | False | passed |  |
| 2026-03-06 | success | 111 | 106 | previous_replay_week:2026-02-27 | 106 | historical_git | d71ff76d | stock_data_070326_1d.pkl | stock_data_070326_1wk.pkl | 2026-03-06 | 2026-03-02 | False | passed |  |
| 2026-03-13 | success | 106 | 102 | previous_replay_week:2026-03-06 | 102 | historical_git | 3b937f7b | stock_data_140326_1d.pkl | stock_data_140326_1wk.pkl | 2026-03-13 | 2026-03-09 | False | passed |  |
| 2026-03-20 | success | 102 | 90 | previous_replay_week:2026-03-13 | 90 | historical_git | ba5dfeff | stock_data_220326_1d.pkl | stock_data_220326_1wk.pkl | 2026-03-20 | 2026-03-16 | False | passed |  |
| 2026-03-27 | success | 90 | 92 | previous_replay_week:2026-03-20 | 92 | historical_git | 99aa4f93 | stock_data_290326_1d.pkl | stock_data_290326_1wk.pkl | 2026-03-27 | 2026-03-23 | False | passed |  |
| 2026-04-02 | success | 92 | 141 | previous_replay_week:2026-03-27 | 141 | historical_git | bc0e12d8 | stock_data_030426_1d.pkl | stock_data_030426_1wk.pkl | 2026-04-02 | 2026-03-30 | False | passed |  |
| 2026-04-10 | success | 141 | 148 | previous_replay_week:2026-04-02 | 148 | historical_git | 0088c5f6 | stock_data_120426_1d.pkl | stock_data_120426_1wk.pkl | 2026-04-10 | 2026-04-06 | False | passed |  |
| 2026-04-17 | success | 148 | 254 | previous_replay_week:2026-04-10 | 254 | historical_git | d8547713 | stock_data_190426_1d.pkl | stock_data_190426_1wk.pkl | 2026-04-17 | 2026-04-13 | False | passed |  |
| 2026-04-24 | success | 254 | 380 | previous_replay_week:2026-04-17 | 380 | historical_git | 629496d3 | stock_data_260426_1d.pkl | stock_data_260426_1wk.pkl | 2026-04-24 | 2026-04-20 | False | passed |  |
| 2026-05-01 | success | 380 | 451 | previous_replay_week:2026-04-24 | 451 | historical_git | b58507d9 | stock_data_010526_1d.pkl | stock_data_010526_1wk.pkl | 2026-05-01 | 2026-04-27 | False | passed |  |
| 2026-05-08 | success | 451 | 513 | previous_replay_week:2026-05-01 | 513 | historical_git | b8528c7a | stock_data_090526_1d.pkl | stock_data_090526_1wk.pkl | 2026-05-08 | 2026-05-04 | False | passed |  |
| 2026-05-15 | success | 513 | 515 | previous_replay_week:2026-05-08 | 515 | historical_git | 57a5ed95 | stock_data_160526_1d.pkl | stock_data_160526_1wk.pkl | 2026-05-15 | 2026-05-11 | False | passed |  |
| 2026-05-22 | success | 515 | 554 | previous_replay_week:2026-05-15 | 554 | historical_git | e67712c4 | stock_data_230526_1d.pkl | stock_data_230526_1wk.pkl | 2026-05-22 | 2026-05-18 | False | passed |  |
| 2026-05-29 | success | 554 | 570 | previous_replay_week:2026-05-22 | 570 | historical_git | c6434c1d | stock_data_300526_1d.pkl | stock_data_300526_1wk.pkl | 2026-05-29 | 2026-05-25 | False | passed |  |
| 2026-06-05 | success | 570 | 594 | previous_replay_week:2026-05-29 | 594 | historical_git | d06d396e | stock_data_060626_1d.pkl | stock_data_060626_1wk.pkl | 2026-06-05 | 2026-06-01 | False | passed |  |
| 2026-06-12 | success | 594 | 674 | previous_replay_week:2026-06-05 | 674 | historical_git | 77697d55 | stock_data_130626_1d.pkl | stock_data_130626_1wk.pkl | 2026-06-12 | 2026-06-08 | False | passed |  |
| 2026-06-18 | success | 674 | 670 | previous_replay_week:2026-06-12 | 670 | historical_git | 95a7307e | stock_data_200626_1d.pkl | stock_data_200626_1wk.pkl | 2026-06-18 | 2026-06-15 | False | passed |  |
| 2026-06-26 | success | 670 | 731 | previous_replay_week:2026-06-18 | 731 | historical_git | 1845fa87 | stock_data_270626_1d.pkl | stock_data_270626_1wk.pkl | 2026-06-26 | 2026-06-22 | False | passed |  |
| 2026-07-02 | success | 731 | 760 | previous_replay_week:2026-06-26 | 760 | historical_git | d26c0b9a | stock_data_040726_1d.pkl | stock_data_040726_1wk.pkl | 2026-07-02 | 2026-06-29 | False | passed |  |
| 2026-07-10 | success | 760 | 716 | previous_replay_week:2026-07-02 | 716 | historical_git | 0aa93424 | stock_data_110726_1d.pkl | stock_data_110726_1wk.pkl | 2026-07-10 | 2026-07-06 | False | passed |  |
| 2026-07-17 | success | 716 | 754 | previous_replay_week:2026-07-10 | 754 | historical_git | 9c931b7d | stock_data_190726_1d.pkl | stock_data_190726_1wk.pkl | 2026-07-17 | 2026-07-13 | False | passed |  |
| 2026-07-24 | success | 754 | 745 | previous_replay_week:2026-07-17 | 745 | historical_git | 7b00d0c4 | stock_data_250726_1d.pkl | stock_data_250726_1wk.pkl | 2026-07-24 | 2026-07-24 | False | passed |  |
| 2026-07-31 | success | 745 | 742 | previous_replay_week:2026-07-24 | 742 | historical_git | be2c948b | stock_data_030826_1d.pkl | stock_data_030826_1wk.pkl | 2026-07-31 | 2026-07-27 | False | passed |  |
| 2026-08-07 | success | 742 | 776 | previous_replay_week:2026-07-31 | 776 | historical_git | 8bff577a | stock_data_080826_1d.pkl | stock_data_080826_1wk.pkl | 2026-08-07 | 2026-08-03 | False | passed |  |
