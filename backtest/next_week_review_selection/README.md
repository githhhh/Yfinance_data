# Next Week Review Selection Research

Branch: `research/next-week-review-selection`

v0.5 turns the experiment into a real optimizer OOS test:

- every formal fold must choose one train-only provisional champion;
- training stability ranks candidates but cannot block OOS entry;
- only full 4-week test blocks count formally;
- the final partial block is `TAIL_EXPLORATORY`;
- behaviorally identical rule parameterizations are de-duplicated by selection signature;
- the main OOS object is the adaptive train-select-next-block policy;
- exact/structural rule convergence is reported separately;
- setup-balanced sensitivity checks whether apparent gains depend on uneven setup coverage.

Run without changing parameters:

```bash
/Users/dev/.conda/envs/quant_env/bin/python \
  -m pytest tests/test_next_week_review_selection.py -q

/Users/dev/.conda/envs/quant_env/bin/python \
  -m backtest.next_week_review_selection.run
```

Generated outputs are retrospective research artifacts only.
