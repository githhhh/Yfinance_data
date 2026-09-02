# Next Week Review Selection Research

Branch: `research/next-week-review-selection`

This research compares:

`B0 ACTIONABLE-only`

vs

`B0 + Near-Buy-Point UNCONFIRMED/BELOW_TRIGGER + >=1 independent positive evidence family`.

v0.3 changes:
- Primary R1 has no Geometry hard reject.
- Entry + weekly volume form one Volume evidence family.
- Snapshot clock and post-opportunity clock are both measured.
- Winner/loser recall is capacity-normalized with capture lift.
- Rule evolution is two-stage: 24 structural rules, then evidence-family leave-one-out only around finalists.
- Weekly macro metrics and paired moving-block bootstrap are produced.

Run tests:

```bash
conda run --no-capture-output -n quant_env \
  python -m pytest tests/test_next_week_review_selection.py -q
```

Run research:

```bash
conda run --no-capture-output -n quant_env \
  python -m backtest.next_week_review_selection.run
```

Do not modify parameters during the run. Outputs remain retrospective research artifacts, not production authorization.
