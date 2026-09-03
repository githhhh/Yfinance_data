# Next Week Review Selection - Data Audit

- pool files loaded: 43
- active-signal weeks: 42
- first snapshot: 2025-10-10
- last snapshot: 2026-08-07
- active-signal events: 2738
- walk-forward OOS folds: 6
- verified PIT EPS rows: 2467
- complete snapshot-clock 1w: 1944/2738 across 42 weeks
- complete opportunity-clock 1w: 1532
- complete snapshot-clock 2w: 1944/2738 across 42 weeks
- complete opportunity-clock 2w: 1529
- complete snapshot-clock 3w: 1788/2738 across 41 weeks
- complete opportunity-clock 3w: 1415
- complete snapshot-clock 4w: 1683/2738 across 40 weeks
- complete opportunity-clock 4w: 1338

Guardrails:
- Price-path missingness has dedicated audits by week/status/setup/source/ticker.
- Walk-forward training is horizon-aware and as-of censored by actual label end date.
- No all-or-nothing 4W mature-week gate is used.
- Primary R1 has no Geometry hard reject.
- Volume is one independent evidence family.
- False/missing positive evidence is neutral.
- Snapshot and opportunity clocks are separate.
- C Rank and ATR are not used.
