# Next Week Review Selection Research

Branch: `research/next-week-review-selection`

v0.4 fixes two remaining research blockers:

1. Explicit price-path coverage audit explains why active signals do or do not have 1W/2W/3W/4W outcomes.
2. Walk-forward is horizon-aware and as-of censored by each label's actual end date. The old all-or-nothing 4W mature-week gate is removed.

Primary R1 remains:

`B0 ACTIONABLE-only + Near-Buy-Point UNCONFIRMED/BELOW_TRIGGER + >=1 independent positive evidence family`.

Run:

```bash
conda run --no-capture-output -n quant_env \
  python -m pytest tests/test_next_week_review_selection.py -q

conda run --no-capture-output -n quant_env \
  python -m backtest.next_week_review_selection.run
```

Do not change parameters during the formal replay. Generated outputs are retrospective research artifacts only.
