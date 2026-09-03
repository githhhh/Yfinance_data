# Next Week Review Selection - Data Audit

- pool files loaded: 43
- active-signal weeks: 42
- first snapshot: 2025-10-10
- last snapshot: 2026-08-07
- active-signal events: 2738
- 4W-mature evaluation weeks: 19
- 4W-mature evaluation events: 2069
- verified PIT EPS rows: 2467
- complete snapshot-clock 1w: 1944/2738
- complete opportunity-clock 1w: 1532
- complete snapshot-clock 2w: 1944/2738
- complete opportunity-clock 2w: 1529
- complete snapshot-clock 3w: 1788/2738
- complete opportunity-clock 3w: 1415
- complete snapshot-clock 4w: 1683/2738
- complete opportunity-clock 4w: 1338

Guardrails:
- Primary R1 has no Geometry hard reject.
- Volume is one evidence family even if entry and weekly volume both confirm.
- False/missing positive evidence is neutral.
- Snapshot and opportunity clocks are kept separate.
- Winner recall is capacity-normalized with selection coverage/capture lift.
- Rule evolution is two-stage, not one 144-rule sweep.
- Weekly macro metrics + paired moving-block bootstrap are reported.
- C Rank and ATR are not used.
