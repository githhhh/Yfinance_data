# Blind Rule Discovery

This is the canonical discovery experiment for replay candidates. Comparator-specific audit code is not an input to this workflow.

## Research contract

- Universe: every upstream `signal=True` replay candidate; no downstream rank/score/selection filter.
- Entry: next trading session Open after the weekly snapshot, so the price is executable without hindsight.
- Path labels over 60 trading sessions: `clean_winner` reaches +20% before -8%; `stop_out_then_winner` reaches -8% first and is **not** a training winner; same-bar stop/target is `ambiguous_path`; no +20% is `loser`.
- Outcomes also include 4/8/12-week returns, SPY excess returns, MAE and MFE.
- Feature surface: numeric/boolean fields only, prior decision artifacts removed, remaining names replaced with `X###`. The alias map stays outside the agent workspace.
- Time structure: monthly and quarterly distribution profiles use counts and quantiles, plus contemporaneous SPY return/drawdown. No global mean is used.
- Holdout: the most recent 4 quarters are sealed by default. They are not copied into the agent workspace.
- Research runtime: external agent execution is hard-capped at 3600 seconds and runs from a temporary copy containing only agent-facing files.
- Freeze: `rule.json` is hashed and frozen before holdout/comparator evaluation.

## CLI

```bash
python -m backtest.blind_rule_discovery.runner \
  --replay-root backtest/latest_quant_trade_replay/output \
  --daily-pkl /path/to/full_history_daily.pkl \
  --benchmark-pkl /path/to/benchmark_daily.pkl \
  --output-root backtest/blind_rule_discovery/output \
  --agent-command '<your RD-agent command that reads prompt.md and samples.csv>'
```

The daily source must include future bars beyond each historical snapshot. A separate benchmark pickle is optional when SPY is already present in the daily source.
