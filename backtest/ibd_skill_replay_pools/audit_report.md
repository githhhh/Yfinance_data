# Latest Quant Trade Replay Pool Audit

- Quant trade commit: `4c0840186ef7bb213bc8d13a849005bb97b7da35`
- Weeks processed: 32
- Success weeks: 32
- Failed weeks: 0
- Weeks using clipped data: 32
- Boundary check: passed (first=2026-01-02, last=2026-08-07, excluded_2026_08_14=True)
- Future-date leak check after clip: passed
- Schema check: passed
- IBD resolver field check: passed (signal_candidates=3817, ibd_entry_valid_nonempty=3817, valid_entries=1619, valid_entry_price_nonempty=1619, invalid_entries=2198, invalid_reject_nonempty=2198)
- Side-effect isolation check: passed
- Production pool write/publish/commit/Futu/Telegram/database side effects: disabled by replay wrapper.
- Old `ibd_skill_replay_pools` contents are treated as untrusted and replaced by this clean replay baseline.

## Week Status

| snapshot_date | status | rows | clipped | schema | failure_reason |
|---|---:|---:|---:|---|---|
| 2026-01-02 | success | 306 | True | passed |  |
| 2026-01-09 | success | 346 | True | passed |  |
| 2026-01-16 | success | 380 | True | passed |  |
| 2026-01-23 | success | 376 | True | passed |  |
| 2026-01-30 | success | 407 | True | passed |  |
| 2026-02-06 | success | 482 | True | passed |  |
| 2026-02-13 | success | 497 | True | passed |  |
| 2026-02-20 | success | 497 | True | passed |  |
| 2026-02-27 | success | 459 | True | passed |  |
| 2026-03-06 | success | 407 | True | passed |  |
| 2026-03-13 | success | 360 | True | passed |  |
| 2026-03-20 | success | 331 | True | passed |  |
| 2026-03-27 | success | 346 | True | passed |  |
| 2026-04-02 | success | 409 | True | passed |  |
| 2026-04-10 | success | 474 | True | passed |  |
| 2026-04-17 | success | 452 | True | passed |  |
| 2026-04-24 | success | 447 | True | passed |  |
| 2026-05-01 | success | 472 | True | passed |  |
| 2026-05-08 | success | 487 | True | passed |  |
| 2026-05-15 | success | 469 | True | passed |  |
| 2026-05-22 | success | 514 | True | passed |  |
| 2026-05-29 | success | 505 | True | passed |  |
| 2026-06-05 | success | 530 | True | passed |  |
| 2026-06-12 | success | 613 | True | passed |  |
| 2026-06-18 | success | 591 | True | passed |  |
| 2026-06-26 | success | 669 | True | passed |  |
| 2026-07-02 | success | 691 | True | passed |  |
| 2026-07-10 | success | 683 | True | passed |  |
| 2026-07-17 | success | 713 | True | passed |  |
| 2026-07-24 | success | 731 | True | passed |  |
| 2026-07-31 | success | 259 | True | passed |  |
| 2026-08-07 | success | 790 | True | passed |  |
