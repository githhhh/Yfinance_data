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
- Maturity: partial outcome quarters are excluded; a quarter is eligible only when its quarter-end signal date could have a full outcome window. Missing ticker/future-price outcomes are counted as censored attrition.
- Holdout: latest 4 quarters by default, plus a **purge/embargo** removing any discovery sample whose 12-week outcome window touches holdout start.
- Isolation: `--agent-command` is refused unless `--sandbox-prefix` is supplied. The same wrapper is preflight-probed and must be unable to read the repository root or an external sentinel before RD-agent is launched.
- Rule: DNF JSON, at most 3 clauses / 6 conditions, conditions may reference only `X###` or `M_*`. `Y_*`, dates and IDs are rejected. Tiny period-specific rules are rejected before freeze.
- Freeze barrier: rule validation + SHA freeze happen before feature map, embargo rows or holdout rows are written.
- Holdout: after freeze, the exact frozen rule is applied once and reported by quarter. `holdout_consumed.json` prevents silent re-evaluation in the same output root.
- RD-agent runtime: hard cap 3600 seconds.

## Data prerequisites

Canonical research must satisfy both of these independently:

1. **Replay-history depth**: enough historical replay candidate quarters must already exist before the sealed holdout. Do not reduce `--holdout-quarters` merely to make a short dataset run. The current committed `backtest/ibd_skill_replay_pools` starts in 2025Q4, so it is useful for pipeline validation but is not long enough for the default four-quarter sealed-holdout research protocol.
2. **Outcome-price depth/quality**: a full-history daily price pickle must cover every replay candidate through its future outcome window and include `Adj Close` plus the benchmark symbol. The current `results_pkl/stock_data_290826_1d.pkl` is raw OHLCV-only and is therefore unsuitable for canonical outcome labeling.

For a meaningful discovery run, regenerate replay pools farther back in time using a long-history point-in-time source, then use a split-safe full-history daily outcome source. Prefer at least 8 discovery quarters before the four sealed holdout quarters; more is better.

The canonical benchmark is `SPY`. If an intentionally equivalent benchmark is used, pass it explicitly with `--spy-code` and document the change before research begins.

## Local verification

From repository root:

```bash
git switch codex/clean-latest-quant-trade-replay-pools
git pull

/Users/dev/.conda/envs/quant_env/bin/python -m pytest \
  tests/test_blind_rule_discovery.py \
  tests/test_blind_rule_discovery_sandbox.py -q
```

## Stage 1: prepare the blind discovery workspace

Use the actual replay-pool root. The command below is a pipeline/preflight example only until long-history replay pools and a split-safe daily source are available.

```bash
rm -rf backtest/blind_rule_discovery/output/local_run

/Users/dev/.conda/envs/quant_env/bin/python \
  -m backtest.blind_rule_discovery.runner \
  --replay-root backtest/ibd_skill_replay_pools \
  --daily-pkl /path/to/full_history_daily_with_adj_close_and_spy.pkl \
  --output-root backtest/blind_rule_discovery/output/local_run
```

At this stage only blind/public discovery artifacts are created. `private_feature_map.json`, `sealed_holdout.csv` and holdout results must not exist.

## Stage 2: run RD-agent on macOS

A restrictive macOS profile is provided at `backtest/blind_rule_discovery/sandbox/macos_agent.sb`. Verify `sandbox-exec` exists first:

```bash
command -v sandbox-exec
```

Then run with the real installed RD-agent command. Use a fresh output root for each research attempt:

```bash
/Users/dev/.conda/envs/quant_env/bin/python \
  -m backtest.blind_rule_discovery.runner \
  --replay-root /path/to/long_history_replay_pools \
  --daily-pkl /path/to/full_history_daily_with_adj_close_and_spy.pkl \
  --output-root backtest/blind_rule_discovery/output/rd_agent_run_01 \
  --agent-command '<installed RD-agent command that reads prompt.md/samples.csv and writes rule.json>' \
  --sandbox-prefix "sandbox-exec -D WORKSPACE={workspace} -D RUNTIME=/Users/dev/.conda/envs/quant_env -f $(pwd)/backtest/blind_rule_discovery/sandbox/macos_agent.sb"
```

The sandbox preflight is fail-closed. If the profile can read outside the isolated workspace, RD-agent is not started.

Do not use `--allow-unadjusted-outcomes` for the canonical research run.
