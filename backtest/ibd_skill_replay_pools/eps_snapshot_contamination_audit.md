# EPS Snapshot Contamination Audit

## 结论

- Historical replay pool `eps_yoy_growth` is not point-in-time safe.
- Non-empty EPS values are current `us/stage2/stage2_whitelist.csv` snapshot values injected by quant_trade pool enrichment.
- Empty signal EPS values are also a current-snapshot artifact: the ticker was not covered by the current Stage2 EPS source at enrichment time.
- Therefore EPS should not be used for historical skill backtest ranking until a point-in-time EPS supplement/enriched layer is built.

## Evidence Summary

- Weeks audited: 32
- Pool rows: 15693
- Signal rows: 3817
- EPS non-empty rows: 13918
- EPS non-empty rows exactly matching current Stage2: 13918
- EPS non-empty rows not matching current Stage2: 0
- Signal EPS non-empty rows: 3409
- Signal EPS rows exactly matching current Stage2: 3409
- Signal EPS missing rows: 408
- Unique EPS codes: 762
- Multi-week EPS codes: 739
- Multi-week EPS codes with changed EPS value: 0

## 每周审计

| snapshot_date | rows | signal | eps_nonempty | stage2_exact_match | nonmatch | signal_eps_nonempty | signal_stage2_match | signal_eps_missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-01-02 | 306 | 29 | 270 | 270 | 0 | 28 | 28 | 1 |
| 2026-01-09 | 346 | 149 | 304 | 304 | 0 | 131 | 131 | 18 |
| 2026-01-16 | 380 | 145 | 337 | 337 | 0 | 130 | 130 | 15 |
| 2026-01-23 | 376 | 64 | 334 | 334 | 0 | 58 | 58 | 6 |
| 2026-01-30 | 407 | 106 | 363 | 363 | 0 | 97 | 97 | 9 |
| 2026-02-06 | 482 | 265 | 439 | 439 | 0 | 250 | 250 | 15 |
| 2026-02-13 | 497 | 92 | 445 | 445 | 0 | 77 | 77 | 15 |
| 2026-02-20 | 497 | 60 | 448 | 448 | 0 | 53 | 53 | 7 |
| 2026-02-27 | 459 | 54 | 411 | 411 | 0 | 46 | 46 | 8 |
| 2026-03-06 | 407 | 26 | 367 | 367 | 0 | 20 | 20 | 6 |
| 2026-03-13 | 360 | 20 | 324 | 324 | 0 | 16 | 16 | 4 |
| 2026-03-20 | 331 | 32 | 293 | 293 | 0 | 25 | 25 | 7 |
| 2026-03-27 | 346 | 69 | 308 | 308 | 0 | 61 | 61 | 8 |
| 2026-04-02 | 409 | 151 | 366 | 366 | 0 | 140 | 140 | 11 |
| 2026-04-10 | 474 | 286 | 425 | 425 | 0 | 261 | 261 | 25 |
| 2026-04-17 | 452 | 174 | 398 | 398 | 0 | 156 | 156 | 18 |
| 2026-04-24 | 447 | 78 | 398 | 398 | 0 | 68 | 68 | 10 |
| 2026-05-01 | 472 | 122 | 416 | 416 | 0 | 102 | 102 | 20 |
| 2026-05-08 | 487 | 136 | 429 | 429 | 0 | 118 | 118 | 18 |
| 2026-05-15 | 469 | 59 | 409 | 409 | 0 | 49 | 49 | 10 |
| 2026-05-22 | 514 | 106 | 448 | 448 | 0 | 96 | 96 | 10 |
| 2026-05-29 | 505 | 99 | 442 | 442 | 0 | 89 | 89 | 10 |
| 2026-06-05 | 530 | 121 | 464 | 464 | 0 | 109 | 109 | 12 |
| 2026-06-12 | 613 | 292 | 544 | 544 | 0 | 270 | 270 | 22 |
| 2026-06-18 | 591 | 119 | 521 | 521 | 0 | 102 | 102 | 17 |
| 2026-06-26 | 669 | 239 | 593 | 593 | 0 | 214 | 214 | 25 |
| 2026-07-02 | 691 | 129 | 615 | 615 | 0 | 116 | 116 | 13 |
| 2026-07-10 | 683 | 85 | 604 | 604 | 0 | 70 | 70 | 15 |
| 2026-07-17 | 713 | 181 | 629 | 629 | 0 | 163 | 163 | 18 |
| 2026-07-24 | 731 | 114 | 645 | 645 | 0 | 103 | 103 | 11 |
| 2026-07-31 | 259 | 36 | 228 | 228 | 0 | 33 | 33 | 3 |
| 2026-08-07 | 790 | 179 | 701 | 701 | 0 | 158 | 158 | 21 |

## 处理建议

- Do not fill `eps_yoy_growth` inside the clean replay pool from current screener snapshots.
- Keep clean replay pool as price/structure baseline and mark EPS as snapshot-contaminated for historical evaluation.
- Build a separate EPS supplement table with `snapshot_date`, `code`, `eps_yoy_growth`, `source`, `source_asof_date`, `period_end`, and `point_in_time_safe`.
- Only use EPS in skill backtests when the supplement row can prove the value was available on or before the snapshot week.
