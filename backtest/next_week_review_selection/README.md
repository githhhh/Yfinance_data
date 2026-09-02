# Next Week Review Selection Research

Research-only implementation for:

> From the frozen weekend active-signal pool, build a limited IBD-style next-week review/watch list and compare it with the current ACTIONABLE-only Futu-sync baseline.

Formal preregistration:
`doc/NEXT_WEEK_REVIEW_SELECTION_RESEARCH_PROTOCOL.md`

## Scope

- Existing pool fields only.
- Verified historical PIT EPS only.
- No ATR or new indicators.
- No C Rank in eligibility or ordering.
- No production Skill / Dashboard / Futu changes.
- Forward 5 sessions are labels only.

Primary variants:

- `B0_ACTIONABLE_ONLY`
- `R1_PATH`
- `R2_BALANCED`
- `R3_STRICT`
- `R2_BALANCED_ATTENTION_MATCHED`
- R2 attention caps 10 / 15 / 20

## Run

Use the project Conda environment:

```bash
conda run --no-capture-output -n quant_env \
  python -m backtest.next_week_review_selection.run
```

Focused tests:

```bash
conda run --no-capture-output -n quant_env \
  python -m pytest tests/test_next_week_review_selection.py -q
```

Default output:

`backtest/next_week_review_selection/output/`

The report is retrospective evidence. It does not authorize a production Skill or Futu-sync change.
