# Schema Review

- Confirmed: fully read `doc/BREAKOUT_FOLLOW_POOL_SCHEMA.md`; it is an 18-line migration notice, not the full field whitepaper.
- Primary field semantics used for this run: repository consumer code plus the pointed SSOT at `/Users/tbin/Documents/quant_trade/strategy/doc/BREAKOUT_FOLLOW_POOL_SCHEMA.md`.
- Ambiguity/inconsistency: yfinance_data no longer carries the full schema, so field semantics cannot be verified from this repository alone. Recommended fix: replace the migration notice with a pinned commit/path reference or vendor a read-only schema snapshot.
- Isolated fields before use: no field with unclear formula was converted into a hard gate. Base/pullback fields are context only unless the existing rule route makes them applicable.
- PIT EPS check: `signal_eps_pit.csv` was audited for `effective_date <= snapshot_date` and Yahoo rows where `effective_date == current_period`. Those Yahoo rows are marked `UNVERIFIED_AVAILABILITY` and excluded from formal EPS evidence.
