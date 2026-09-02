# Next Week Review Selection - Data Audit

- pool files loaded: 43
- active-signal weeks: 42
- first snapshot: 2025-10-10
- last snapshot: 2026-08-07
- active-signal events: 2738
- 4W-mature evaluation weeks: 19
- 4W-mature evaluation events: 2069
- verified PIT EPS rows: 2467
- complete 1w outcomes: 1944/2738
- complete 2w outcomes: 1944/2738
- complete 3w outcomes: 1788/2738
- complete 4w outcomes: 1683/2738

Guardrails:
- Weekend selector reads only frozen pool fields + verified PIT EPS.
- Forward returns/MFE/MAE/oracle flags are evaluation-only.
- ACTIONABLE baseline rows are never re-filtered by research variants.
- EXTENDED is excluded from the core selector.
- False/missing positive evidence is neutral.
- C Rank and ATR are not used.
