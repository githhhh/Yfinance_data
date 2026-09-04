# Blind Rule Discovery

Canonical discovery experiment for replay candidates. B0/ranking/comparator code is not an input to discovery.

## Research contract

- Universe: every upstream `signal=True` replay candidate; no downstream rank/score/Top3 filter.
- Entry: use the replay-time trigger/pivot. Over the next 5 sessions, a cross from below enters at trigger; a gap/open enters only up to +5% above trigger; extended opens are not retroactively bought on a same-day dip. Missing trigger means censored. No future low/rebound can become a hindsight buy point. Outcome OHLC is split-safe: canonical runs require `Adj Close` and derive a continuous adjusted OHLC path; raw unadjusted outcomes are refused unless explicitly overridden.
- First-passage labels over 60 sessions:
  - `clean_winner`: +20% before -8%.
  - `stopped_out_loser`: -8% and no later +20% in-window.
  - `stop_out_then_winner`: -8% first, later +20%; **still a trading loser**.
  - `unresolved`: neither boundary; kept separate instead of being forced into loser.
  - same-bar target/stop: `ambiguous_path`.
- Features: explicit point-in-time allowlist only. Stock fields are anonymized to `X###`; prior rank/score/status/selection artifacts are absent by construction.
- Market context: only `M_*` trailing features known at signal time are agent-facing. Full month/quarter SPY return/drawdown is reviewer-only and created after rule freeze.
- Dossier: discovery-only monthly/quarterly counts and quantiles; no global mean. Input pool order is discarded before opaque sample IDs are assigned, so legacy sort order cannot leak.
- Maturity: partial outcome quarters are excluded; a quarter is eligible only when its quarter-end signal date could have a full 60-session future window from the benchmark calendar. Missing ticker/future-price outcomes are counted as censored attrition.
- Holdout: latest 4 quarters by default, plus a **purge/embargo** removing any discovery sample whose 12-week outcome window touches holdout start.
- Isolation: `--agent-command` is refused unless `--sandbox-prefix` is supplied. The wrapper must be an OS/container sandbox exposing only the isolated workspace and required runtime.
- Rule: DNF JSON, at most 3 clauses / 6 conditions, conditions may reference only `X###` or `M_*`. `Y_*`, dates and IDs are rejected. Tiny period-specific rules are rejected before freeze (support gate only; no discovery-performance threshold).
- Freeze barrier: rule validation + SHA freeze happen before feature map, embargo rows or holdout rows are written.
- Holdout: after freeze, the exact frozen rule is applied once and reported by quarter using counts/rates/quantiles plus realized SPY quarter return/drawdown. Selected winner rate is shown beside the holdout-universe winner rate. A `holdout_consumed.json` ledger then blocks silent re-evaluation in the same output root.
- RD-agent runtime: hard cap 3600 seconds.

## CLI

```bash
python -m backtest.blind_rule_discovery.runner \
  --replay-root backtest/latest_quant_trade_replay/output \
  --daily-pkl /path/to/full_history_daily.pkl \
  --output-root backtest/blind_rule_discovery/output \
  --agent-command '<RD-agent command reading prompt.md/samples.csv>' \
  --sandbox-prefix '<OS/container sandbox wrapper; {workspace} is supported>'
```

Running without `--agent-command` prepares only the blind discovery workspace and public metadata; private feature mappings and holdout data are deliberately not materialized.
