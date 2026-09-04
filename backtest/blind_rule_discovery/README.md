# Blind Rule Discovery

Canonical strategy-blind discovery experiment for upstream replay candidates. Existing B0 rank/score/Top3/comparator outputs are not discovery inputs.

## Research contract

- **Universe of observations:** every upstream `signal=True` candidate produced by the reconstructed latest-logic replay. No downstream rank/score/selection filter is applied.
- **Executable entry:** use the replay-time trigger/pivot. During the next 5 sessions, a cross from below enters at trigger; a gap/open enters only up to +5% above trigger. Extended opens are not retroactively bought on a same-day dip. Missing/untriggered entries are censored.
- **Outcome path:** 60 trading sessions from executable entry. `clean_winner` is +20% before -8%; `stop_out_then_winner` is still a trading loser; `unresolved` and same-bar/entry-day ambiguity remain separate.
- **Price basis:** canonical research uses Yahoo OHLC from `auto_adjust=False`, recorded as split-adjusted/non-dividend-adjusted price history. Do **not** multiply the path by dividend `Adj Close` factors. The bundle records and hashes this price contract.
- **Intervals:** `1d` and `1wk` are downloaded independently from Yahoo. Replay therefore receives the same interval semantics as production rather than a locally resampled weekly approximation.
- **Calendar:** replay snapshot weeks come from actual SPY daily sessions. Historical holidays are not inferred from a hard-coded calendar.
- **Features:** an explicit point-in-time allowlist is anonymized to `X###`. Prior ranks/scores/status/selection artifacts are excluded by construction. Agent-facing `M_*` market features are trailing facts known at signal time.
- **Time split:** default requirement is at least 8 discovery quarters plus 4 sealed holdout quarters. A 12-week purge/embargo removes discovery rows whose outcome window touches the holdout.
- **RD-agent isolation:** OS/container sandbox is mandatory, preflight-probed, and execution is hard-capped at 3600 seconds.
- **Frozen rule:** machine-readable DNF JSON, max 3 clauses / 6 conditions, executable fields limited to `X###` and `M_*`. Freeze/hash occurs before private feature mapping or holdout materialization.
- **Holdout:** the frozen rule is evaluated once; `holdout_consumed.json` blocks silent repeated tuning against the same output root.

## Canonical historical reconstruction

The old committed `backtest/ibd_skill_replay_pools` starts too late for formal discovery. It is now used only as one source of known symbols when constructing the research universe.

The canonical local reconstruction is fixed to:

- price history start: `2017-01-01`;
- old-pool warmup: `2022-07-01` through the start of analysis;
- persisted replay analysis: `2022Q4` through `2026Q1` (`2022-10-01` through `2026-03-27` weekly snapshots);
- minimum successful persisted quarters: 14;
- blind Stage 1: minimum 8 discovery quarters + 4 sealed holdout quarters after maturity and purge checks.

The research universe is the union of current strategy inputs, symbols already seen in committed replay pools, symbols in the current seed daily cache, and SPY/^GSPC. **This is not a point-in-time historical listings database.** Delisted/non-current symbols that are absent from every known source cannot be recovered, so survivorship remains a documented limitation. Conclusions from this experiment apply to the reconstructed current-known upstream universe, not an unbiased census of every historically listed US stock.

## Provenance / fail-close chain

Canonical execution verifies every stage:

1. `research_data` writes `research_daily.pkl`, `research_weekly.pkl`, and `price_manifest.json` with provider parameters, yfinance version, joint coverage, source limitation, and SHA-256 hashes.
2. `replay_builder` independently requires >=98% joint 1d/1wk coverage, direct Yahoo `1wk`, 5Y pre-warmup context, identical daily/weekly symbol sets, zero warmup/analysis failures, and >=14 successful quarters. Replay EPS PIT cache is redirected into ignored local work data instead of modifying tracked `us/signal_eps_pit_replay.csv`.
3. `replay_builder` hashes exactly the persisted `*/breakout_follow_pool.csv` files into `research_replay_preflight.json`.
4. blind `runner` verifies the exact daily-pkl SHA and replay dataset digest before Stage 1. Wrong/stale/tampered stage outputs are rejected.
5. after candidate maturity filtering and purge, blind `runner` again requires >=8 discovery quarters before writing the agent workspace.

`--allow-unadjusted-outcomes` and `--allow-unverified-replay` are debug escape hatches only. They must not be used for canonical research.

## Local run

Run from the `Yfinance_data` repository root. Adjust `QUANT_TRADE` only if your local path differs.

```bash
git switch codex/clean-latest-quant-trade-replay-pools
git pull

PY=/Users/dev/.conda/envs/quant_env/bin/python
QUANT_TRADE=/Users/dev/Documents/quant_trade

$PY -m pytest \
  tests/test_blind_rule_discovery.py \
  tests/test_blind_rule_discovery_sandbox.py \
  tests/test_blind_rule_discovery_data_pipeline.py \
  tests/test_blind_rule_discovery_provenance.py -q
```

### 1. Build canonical long-history price bundle

This downloads both Yahoo `1d` and direct `1wk` history. Generated data is under gitignored `backtest/blind_rule_discovery/work/`.

```bash
rm -rf backtest/blind_rule_discovery/work/prices

$PY -m backtest.blind_rule_discovery.research_data \
  --start 2017-01-01 \
  --output-dir backtest/blind_rule_discovery/work/prices
```

Expected files:

```text
backtest/blind_rule_discovery/work/prices/research_daily.pkl
backtest/blind_rule_discovery/work/prices/research_weekly.pkl
backtest/blind_rule_discovery/work/prices/price_manifest.json
```

The builder requires SPY and >=98% joint daily/weekly symbol coverage by default.

### 2. Rebuild long-history replay pools with latest quant_trade logic

```bash
rm -rf backtest/blind_rule_discovery/work/replay_pools

$PY -m backtest.blind_rule_discovery.replay_builder \
  --quant-trade-path "$QUANT_TRADE"
```

This first runs the Q3-2022 warmup to reconstruct chronological `old_pool`, then persists weekly replay pools from 2022Q4 through 2026Q1. Any failed warmup week or analysis week aborts the canonical run instead of continuing with broken state.

Inspect:

```bash
cat backtest/blind_rule_discovery/work/replay_pools/research_replay_preflight.json
cat backtest/blind_rule_discovery/work/replay_pools/research_replay_report.md
```

The successful preflight must show zero warmup/analysis failures, complete expected week count, and at least 14 successful quarters.

### 3. Prepare blind Stage 1 workspace

```bash
rm -rf backtest/blind_rule_discovery/output/local_run

$PY -m backtest.blind_rule_discovery.runner \
  --replay-root backtest/blind_rule_discovery/work/replay_pools \
  --daily-pkl backtest/blind_rule_discovery/work/prices/research_daily.pkl \
  --output-root backtest/blind_rule_discovery/output/local_run
```

At this point only blind/public artifacts may exist. In particular, before rule freeze these must **not** exist:

```text
private_feature_map.json
sealed_holdout.csv
holdout_report.csv
```

Inspect:

```bash
cat backtest/blind_rule_discovery/output/local_run/experiment_metadata.json
ls -la backtest/blind_rule_discovery/output/local_run/agent_workspace
```

`experiment_metadata.json` must report verified replay provenance, >=8 discovery quarters, and 4 sealed holdout quarters.

### 4. Run the <=1h RD-agent research

Use a **fresh output root**. The macOS sandbox profile is `backtest/blind_rule_discovery/sandbox/macos_agent.sb`.

```bash
command -v sandbox-exec

$PY -m backtest.blind_rule_discovery.runner \
  --replay-root backtest/blind_rule_discovery/work/replay_pools \
  --daily-pkl backtest/blind_rule_discovery/work/prices/research_daily.pkl \
  --output-root backtest/blind_rule_discovery/output/rd_agent_run_01 \
  --agent-command '<installed RD-agent command that reads prompt.md/samples.csv and writes rule.json>' \
  --sandbox-prefix "sandbox-exec -D WORKSPACE={workspace} -D RUNTIME=/Users/dev/.conda/envs/quant_env -f $(pwd)/backtest/blind_rule_discovery/sandbox/macos_agent.sb"
```

The sandbox must fail its isolation probe if it can see the repository or any external sentinel. Only after the agent returns a valid rule does the runner freeze it, materialize the private feature map/holdout, and perform the one-shot holdout evaluation.
