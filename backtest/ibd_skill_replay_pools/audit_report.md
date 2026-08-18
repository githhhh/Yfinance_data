# Latest Quant Trade Replay Pool Audit

- Quant trade commit: `4c0840186ef7bb213bc8d13a849005bb97b7da35`
- Weeks processed: 32
- Success weeks: 32
- Failed weeks: 0
- Weeks using clipped data: 0
- Boundary check: passed (first=2026-01-02, last=2026-08-07, excluded_2026_08_14=True)
- Future-date leak check after clip: passed
- Schema check: passed
- Historical git pkl source check: passed
- IBD resolver field check: passed (signal_candidates=2500, ibd_entry_valid_nonempty=2500, valid_entries=1132, valid_entry_price_nonempty=1132, invalid_entries=1368, invalid_reject_nonempty=1368)
- Side-effect isolation check: passed
- Production pool write/publish/commit/Futu/Telegram/database side effects: disabled by replay wrapper.
- Old `ibd_skill_replay_pools` contents are treated as untrusted and replaced by this clean replay baseline.

## Week Status

| snapshot_date | status | rows | data_source | pkl_commit | daily_pkl | weekly_pkl | daily_max | weekly_max | clipped | schema | failure_reason |
|---|---:|---:|---|---|---|---|---|---|---:|---|---|
| 2026-01-02 | success | 83 | historical_git | 512cdf64 | stock_data_040126_1d.pkl | stock_data_040126_1wk.pkl | 2026-01-02 | 2026-01-02 | False | passed |  |
| 2026-01-09 | success | 91 | historical_git | 48c09cb7 | stock_data_110126_1d.pkl | stock_data_110126_1wk.pkl | 2026-01-09 | 2026-01-05 | False | passed |  |
| 2026-01-16 | success | 94 | historical_git | cf1eda79 | stock_data_170126_1d.pkl | stock_data_170126_1wk.pkl | 2026-01-16 | 2026-01-12 | False | passed |  |
| 2026-01-23 | success | 100 | historical_git | 68266829 | stock_data_250126_1d.pkl | stock_data_250126_1wk.pkl | 2026-01-23 | 2026-01-19 | False | passed |  |
| 2026-01-30 | success | 100 | historical_git | e8f1f6be | stock_data_010226_1d.pkl | stock_data_010226_1wk.pkl | 2026-01-30 | 2026-01-26 | False | passed |  |
| 2026-02-06 | success | 0 | historical_git | 7f69cd69 | stock_data_060226_1d.pkl | stock_data_060226_1wk.pkl | 2026-02-06 | 2026-02-02 | False | passed |  |
| 2026-02-13 | success | 119 | historical_git | 2686d5dc | stock_data_160226_1d.pkl | stock_data_160226_1wk.pkl | 2026-02-13 | 2026-02-09 | False | passed |  |
| 2026-02-20 | success | 119 | historical_git | f2601d55 | stock_data_230226_1d.pkl | stock_data_230226_1wk.pkl | 2026-02-20 | 2026-02-16 | False | passed |  |
| 2026-02-27 | success | 111 | historical_git | b47f5aef | stock_data_280226_1d.pkl | stock_data_280226_1wk.pkl | 2026-02-27 | 2026-02-23 | False | passed |  |
| 2026-03-06 | success | 106 | historical_git | d71ff76d | stock_data_070326_1d.pkl | stock_data_070326_1wk.pkl | 2026-03-06 | 2026-03-02 | False | passed |  |
| 2026-03-13 | success | 102 | historical_git | 3b937f7b | stock_data_140326_1d.pkl | stock_data_140326_1wk.pkl | 2026-03-13 | 2026-03-09 | False | passed |  |
| 2026-03-20 | success | 90 | historical_git | ba5dfeff | stock_data_220326_1d.pkl | stock_data_220326_1wk.pkl | 2026-03-20 | 2026-03-16 | False | passed |  |
| 2026-03-27 | success | 92 | historical_git | 99aa4f93 | stock_data_290326_1d.pkl | stock_data_290326_1wk.pkl | 2026-03-27 | 2026-03-23 | False | passed |  |
| 2026-04-02 | success | 141 | historical_git | bc0e12d8 | stock_data_030426_1d.pkl | stock_data_030426_1wk.pkl | 2026-04-02 | 2026-03-30 | False | passed |  |
| 2026-04-10 | success | 148 | historical_git | 0088c5f6 | stock_data_120426_1d.pkl | stock_data_120426_1wk.pkl | 2026-04-10 | 2026-04-06 | False | passed |  |
| 2026-04-17 | success | 254 | historical_git | d8547713 | stock_data_190426_1d.pkl | stock_data_190426_1wk.pkl | 2026-04-17 | 2026-04-13 | False | passed |  |
| 2026-04-24 | success | 380 | historical_git | 629496d3 | stock_data_260426_1d.pkl | stock_data_260426_1wk.pkl | 2026-04-24 | 2026-04-20 | False | passed |  |
| 2026-05-01 | success | 451 | historical_git | b58507d9 | stock_data_010526_1d.pkl | stock_data_010526_1wk.pkl | 2026-05-01 | 2026-04-27 | False | passed |  |
| 2026-05-08 | success | 513 | historical_git | b8528c7a | stock_data_090526_1d.pkl | stock_data_090526_1wk.pkl | 2026-05-08 | 2026-05-04 | False | passed |  |
| 2026-05-15 | success | 515 | historical_git | 57a5ed95 | stock_data_160526_1d.pkl | stock_data_160526_1wk.pkl | 2026-05-15 | 2026-05-11 | False | passed |  |
| 2026-05-22 | success | 554 | historical_git | e67712c4 | stock_data_230526_1d.pkl | stock_data_230526_1wk.pkl | 2026-05-22 | 2026-05-18 | False | passed |  |
| 2026-05-29 | success | 570 | historical_git | c6434c1d | stock_data_300526_1d.pkl | stock_data_300526_1wk.pkl | 2026-05-29 | 2026-05-25 | False | passed |  |
| 2026-06-05 | success | 594 | historical_git | d06d396e | stock_data_060626_1d.pkl | stock_data_060626_1wk.pkl | 2026-06-05 | 2026-06-01 | False | passed |  |
| 2026-06-12 | success | 674 | historical_git | 77697d55 | stock_data_130626_1d.pkl | stock_data_130626_1wk.pkl | 2026-06-12 | 2026-06-08 | False | passed |  |
| 2026-06-18 | success | 670 | historical_git | 95a7307e | stock_data_200626_1d.pkl | stock_data_200626_1wk.pkl | 2026-06-18 | 2026-06-15 | False | passed |  |
| 2026-06-26 | success | 731 | historical_git | 1845fa87 | stock_data_270626_1d.pkl | stock_data_270626_1wk.pkl | 2026-06-26 | 2026-06-22 | False | passed |  |
| 2026-07-02 | success | 760 | historical_git | d26c0b9a | stock_data_040726_1d.pkl | stock_data_040726_1wk.pkl | 2026-07-02 | 2026-06-29 | False | passed |  |
| 2026-07-10 | success | 716 | historical_git | 0aa93424 | stock_data_110726_1d.pkl | stock_data_110726_1wk.pkl | 2026-07-10 | 2026-07-06 | False | passed |  |
| 2026-07-17 | success | 754 | historical_git | 9c931b7d | stock_data_190726_1d.pkl | stock_data_190726_1wk.pkl | 2026-07-17 | 2026-07-13 | False | passed |  |
| 2026-07-24 | success | 745 | historical_git | 7b00d0c4 | stock_data_250726_1d.pkl | stock_data_250726_1wk.pkl | 2026-07-24 | 2026-07-24 | False | passed |  |
| 2026-07-31 | success | 742 | historical_git | be2c948b | stock_data_030826_1d.pkl | stock_data_030826_1wk.pkl | 2026-07-31 | 2026-07-27 | False | passed |  |
| 2026-08-07 | success | 776 | historical_git | 8bff577a | stock_data_080826_1d.pkl | stock_data_080826_1wk.pkl | 2026-08-07 | 2026-08-03 | False | passed |  |
