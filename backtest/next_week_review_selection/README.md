# Next Week Review Selection Research

Research-only experiment on branch `research/next-week-review-selection`, based on
`codex/clean-latest-quant-trade-replay-pools`.

Core hypothesis:

`B0 ACTIONABLE-only`
vs
`B0 + Near-Buy-Point UNCONFIRMED/BELOW_TRIGGER with at least one positive quality evidence`.

The experiment evaluates 1W/2W/3W/4W winner capture, loser exclusion, MFE/MAE,
attention cost, and expanding-window walk-forward stability.

Run:

```bash
conda run --no-capture-output -n quant_env \
  python -m backtest.next_week_review_selection.run
```

Focused tests:

```bash
conda run --no-capture-output -n quant_env \
  python -m pytest tests/test_next_week_review_selection.py -q
```

Important:
- ACTIONABLE rows are never re-filtered by research variants.
- EXTENDED is exploratory only.
- False/missing positive evidence is neutral.
- C Rank and ATR are excluded.
- Outputs are retrospective research artifacts, not production authorization.
