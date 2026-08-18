# Final Data Quality Audit

- Weeks: 32 (2026-01-02 to 2026-08-07, excluding 2026-08-14)
- Rows: 11895; signal rows: 2500
- Distinct column sets: 1
- Non-EPS abnormal empty fields: {}
- EPS signal empty values intentionally blanked: 2500
- 52w recompute mismatches: 0
- IBD valid entries: 1132; invalid entries with reject path checked: 1368
- Repairable fallback fields: {'industry': 1584, 'sector': 1584}
- Optional gap fields: {'pullback_v_is_dry': 2540, 'ibd_candidate_extra': 10095, 'pullback_duration_weeks': 1870, 'pullback_pct': 1870, 'pullback_pct_off_peak': 1870}
- Issues: none

## Weekly Pkl Boundary

| snapshot_date | rows | signal | pkl_commit | daily_pkl | daily_max | weekly_pkl | weekly_max | clipped |
|---|---:|---:|---|---|---|---|---|---:|
| 2026-01-02 | 83 | 15 | 512cdf643d7b | results_pkl/stock_data_040126_1d.pkl | 2026-01-02 | results_pkl/stock_data_040126_1wk.pkl | 2026-01-02 | False |
| 2026-01-09 | 91 | 40 | 48c09cb7a0be | results_pkl/stock_data_110126_1d.pkl | 2026-01-09 | results_pkl/stock_data_110126_1wk.pkl | 2026-01-05 | False |
| 2026-01-16 | 94 | 21 | cf1eda790d1c | results_pkl/stock_data_170126_1d.pkl | 2026-01-16 | results_pkl/stock_data_170126_1wk.pkl | 2026-01-12 | False |
| 2026-01-23 | 100 | 14 | 68266829e415 | results_pkl/stock_data_250126_1d.pkl | 2026-01-23 | results_pkl/stock_data_250126_1wk.pkl | 2026-01-19 | False |
| 2026-01-30 | 100 | 17 | e8f1f6beb7ba | results_pkl/stock_data_010226_1d.pkl | 2026-01-30 | results_pkl/stock_data_010226_1wk.pkl | 2026-01-26 | False |
| 2026-02-06 | 0 | 0 | 7f69cd69a54e | results_pkl/stock_data_060226_1d.pkl | 2026-02-06 | results_pkl/stock_data_060226_1wk.pkl | 2026-02-02 | False |
| 2026-02-13 | 119 | 27 | 2686d5dc75b9 | results_pkl/stock_data_160226_1d.pkl | 2026-02-13 | results_pkl/stock_data_160226_1wk.pkl | 2026-02-09 | False |
| 2026-02-20 | 119 | 18 | f2601d55f997 | results_pkl/stock_data_230226_1d.pkl | 2026-02-20 | results_pkl/stock_data_230226_1wk.pkl | 2026-02-16 | False |
| 2026-02-27 | 111 | 27 | b47f5aef7255 | results_pkl/stock_data_280226_1d.pkl | 2026-02-27 | results_pkl/stock_data_280226_1wk.pkl | 2026-02-23 | False |
| 2026-03-06 | 106 | 9 | d71ff76d4b96 | results_pkl/stock_data_070326_1d.pkl | 2026-03-06 | results_pkl/stock_data_070326_1wk.pkl | 2026-03-02 | False |
| 2026-03-13 | 102 | 10 | 3b937f7b03f0 | results_pkl/stock_data_140326_1d.pkl | 2026-03-13 | results_pkl/stock_data_140326_1wk.pkl | 2026-03-09 | False |
| 2026-03-20 | 90 | 8 | ba5dfeff7b8b | results_pkl/stock_data_220326_1d.pkl | 2026-03-20 | results_pkl/stock_data_220326_1wk.pkl | 2026-03-16 | False |
| 2026-03-27 | 92 | 10 | 99aa4f93abdb | results_pkl/stock_data_290326_1d.pkl | 2026-03-27 | results_pkl/stock_data_290326_1wk.pkl | 2026-03-23 | False |
| 2026-04-02 | 141 | 45 | bc0e12d8a9a4 | results_pkl/stock_data_030426_1d.pkl | 2026-04-02 | results_pkl/stock_data_030426_1wk.pkl | 2026-03-30 | False |
| 2026-04-10 | 148 | 69 | 0088c5f6c596 | results_pkl/stock_data_120426_1d.pkl | 2026-04-10 | results_pkl/stock_data_120426_1wk.pkl | 2026-04-06 | False |
| 2026-04-17 | 254 | 96 | d85477135f28 | results_pkl/stock_data_190426_1d.pkl | 2026-04-17 | results_pkl/stock_data_190426_1wk.pkl | 2026-04-13 | False |
| 2026-04-24 | 380 | 80 | 629496d38cf7 | results_pkl/stock_data_260426_1d.pkl | 2026-04-24 | results_pkl/stock_data_260426_1wk.pkl | 2026-04-20 | False |
| 2026-05-01 | 451 | 105 | b58507d9dab9 | results_pkl/stock_data_010526_1d.pkl | 2026-05-01 | results_pkl/stock_data_010526_1wk.pkl | 2026-04-27 | False |
| 2026-05-08 | 513 | 128 | b8528c7ad9b2 | results_pkl/stock_data_090526_1d.pkl | 2026-05-08 | results_pkl/stock_data_090526_1wk.pkl | 2026-05-04 | False |
| 2026-05-15 | 515 | 66 | 57a5ed9505f0 | results_pkl/stock_data_160526_1d.pkl | 2026-05-15 | results_pkl/stock_data_160526_1wk.pkl | 2026-05-11 | False |
| 2026-05-22 | 554 | 87 | e67712c4fd7e | results_pkl/stock_data_230526_1d.pkl | 2026-05-22 | results_pkl/stock_data_230526_1wk.pkl | 2026-05-18 | False |
| 2026-05-29 | 570 | 112 | c6434c1dfeaf | results_pkl/stock_data_300526_1d.pkl | 2026-05-29 | results_pkl/stock_data_300526_1wk.pkl | 2026-05-25 | False |
| 2026-06-05 | 594 | 108 | d06d396e2911 | results_pkl/stock_data_060626_1d.pkl | 2026-06-05 | results_pkl/stock_data_060626_1wk.pkl | 2026-06-01 | False |
| 2026-06-12 | 674 | 268 | 77697d555969 | results_pkl/stock_data_130626_1d.pkl | 2026-06-12 | results_pkl/stock_data_130626_1wk.pkl | 2026-06-08 | False |
| 2026-06-18 | 670 | 145 | 95a7307e8ee3 | results_pkl/stock_data_200626_1d.pkl | 2026-06-18 | results_pkl/stock_data_200626_1wk.pkl | 2026-06-15 | False |
| 2026-06-26 | 731 | 211 | 1845fa873eb0 | results_pkl/stock_data_270626_1d.pkl | 2026-06-26 | results_pkl/stock_data_270626_1wk.pkl | 2026-06-22 | False |
| 2026-07-02 | 760 | 120 | d26c0b9aa70e | results_pkl/stock_data_040726_1d.pkl | 2026-07-02 | results_pkl/stock_data_040726_1wk.pkl | 2026-06-29 | False |
| 2026-07-10 | 716 | 83 | 0aa934244672 | results_pkl/stock_data_110726_1d.pkl | 2026-07-10 | results_pkl/stock_data_110726_1wk.pkl | 2026-07-06 | False |
| 2026-07-17 | 754 | 174 | 9c931b7da64b | results_pkl/stock_data_190726_1d.pkl | 2026-07-17 | results_pkl/stock_data_190726_1wk.pkl | 2026-07-13 | False |
| 2026-07-24 | 745 | 106 | 7b00d0c457d8 | results_pkl/stock_data_250726_1d.pkl | 2026-07-24 | results_pkl/stock_data_250726_1wk.pkl | 2026-07-24 | False |
| 2026-07-31 | 742 | 114 | be2c948b8e55 | results_pkl/stock_data_030826_1d.pkl | 2026-07-31 | results_pkl/stock_data_030826_1wk.pkl | 2026-07-27 | False |
| 2026-08-07 | 776 | 167 | 8bff577a843f | results_pkl/stock_data_080826_1d.pkl | 2026-08-07 | results_pkl/stock_data_080826_1wk.pkl | 2026-08-03 | False |
