# Next Week Review Selection Research

Latest controlled experiment: **v0.6 deterministic discriminative study**.

v0.6 deliberately does **not** use rd-agent or ML.

It fixes the v0.5 train-converged anchor:

`Near5 + UNCONFIRMED/BELOW_TRIGGER + >=2 evidence families + Geometry allow`

and asks whether existing PIT fields can refine that supplemental cohort while keeping review attention <= **1.50x B0**.

Important: the historical test weeks were already observed in v0.5, so v0.6 is a **retrospective reused-history confirmation**, not a new sealed holdout.

Run tests:

```bash
/Users/dev/.conda/envs/quant_env/bin/python \
  -m pytest tests/test_next_week_review_selection.py \
  tests/test_next_week_review_discriminative.py -q
```

Run only v0.6:

```bash
/Users/dev/.conda/envs/quant_env/bin/python \
  -m backtest.next_week_review_selection.run_discriminative
```

Outputs are written to:

`backtest/next_week_review_selection/output_v06/`

Do not change parameters during the formal run.
