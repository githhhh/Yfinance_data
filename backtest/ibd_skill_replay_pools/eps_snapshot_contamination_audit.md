# EPS Snapshot Contamination Audit

## 结论

- Historical replay pool `eps_yoy_growth` has been cleared to remove current Stage2 snapshot contamination.
- Current replay pools intentionally keep EPS blank until a point-in-time EPS supplement layer is available.
- Current `us/stage2/stage2_whitelist.csv` can still provide a snapshot-only reference, but it is not point-in-time safe for historical ranking.
- Therefore EPS must not be used in historical skill backtest scoring from the clean replay pool itself.

## Evidence Summary

- Weeks audited: 32
- Pool rows: 15693
- Signal rows: 3817
- EPS non-empty rows remaining in replay pools: 0
- EPS non-empty rows exactly matching current Stage2: 0
- Signal EPS non-empty rows remaining: 0
- Signal EPS missing rows requiring PIT supplement: 3817
- Signal EPS rows with current snapshot-only source: 3409
- Signal EPS rows unresolved by current snapshot: 408
- Unique EPS codes remaining: 0
- Multi-week EPS codes remaining: 0
- Multi-week EPS codes with changed EPS value: 0

## 每周审计

| snapshot_date | rows | signal | eps_nonempty | stage2_exact_match | signal_eps_missing | current_snapshot_available | current_snapshot_unresolved |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-01-02 | 306 | 29 | 0 | 0 | 29 | 28 | 1 |
| 2026-01-09 | 346 | 149 | 0 | 0 | 149 | 131 | 18 |
| 2026-01-16 | 380 | 145 | 0 | 0 | 145 | 130 | 15 |
| 2026-01-23 | 376 | 64 | 0 | 0 | 64 | 58 | 6 |
| 2026-01-30 | 407 | 106 | 0 | 0 | 106 | 97 | 9 |
| 2026-02-06 | 482 | 265 | 0 | 0 | 265 | 250 | 15 |
| 2026-02-13 | 497 | 92 | 0 | 0 | 92 | 77 | 15 |
| 2026-02-20 | 497 | 60 | 0 | 0 | 60 | 53 | 7 |
| 2026-02-27 | 459 | 54 | 0 | 0 | 54 | 46 | 8 |
| 2026-03-06 | 407 | 26 | 0 | 0 | 26 | 20 | 6 |
| 2026-03-13 | 360 | 20 | 0 | 0 | 20 | 16 | 4 |
| 2026-03-20 | 331 | 32 | 0 | 0 | 32 | 25 | 7 |
| 2026-03-27 | 346 | 69 | 0 | 0 | 69 | 61 | 8 |
| 2026-04-02 | 409 | 151 | 0 | 0 | 151 | 140 | 11 |
| 2026-04-10 | 474 | 286 | 0 | 0 | 286 | 261 | 25 |
| 2026-04-17 | 452 | 174 | 0 | 0 | 174 | 156 | 18 |
| 2026-04-24 | 447 | 78 | 0 | 0 | 78 | 68 | 10 |
| 2026-05-01 | 472 | 122 | 0 | 0 | 122 | 102 | 20 |
| 2026-05-08 | 487 | 136 | 0 | 0 | 136 | 118 | 18 |
| 2026-05-15 | 469 | 59 | 0 | 0 | 59 | 49 | 10 |
| 2026-05-22 | 514 | 106 | 0 | 0 | 106 | 96 | 10 |
| 2026-05-29 | 505 | 99 | 0 | 0 | 99 | 89 | 10 |
| 2026-06-05 | 530 | 121 | 0 | 0 | 121 | 109 | 12 |
| 2026-06-12 | 613 | 292 | 0 | 0 | 292 | 270 | 22 |
| 2026-06-18 | 591 | 119 | 0 | 0 | 119 | 102 | 17 |
| 2026-06-26 | 669 | 239 | 0 | 0 | 239 | 214 | 25 |
| 2026-07-02 | 691 | 129 | 0 | 0 | 129 | 116 | 13 |
| 2026-07-10 | 683 | 85 | 0 | 0 | 85 | 70 | 15 |
| 2026-07-17 | 713 | 181 | 0 | 0 | 181 | 163 | 18 |
| 2026-07-24 | 731 | 114 | 0 | 0 | 114 | 103 | 11 |
| 2026-07-31 | 259 | 36 | 0 | 0 | 36 | 33 | 3 |
| 2026-08-07 | 790 | 179 | 0 | 0 | 179 | 158 | 21 |

## 处理建议

- Keep `eps_yoy_growth` empty in clean replay pools.
- Build a separate EPS supplement table with `snapshot_date`, `code`, `eps_yoy_growth`, `source`, `source_asof_date`, `period_end`, and `point_in_time_safe`.
- Only join EPS into enriched research pools when `point_in_time_safe=True`.
