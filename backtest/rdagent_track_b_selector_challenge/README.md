# RD-Agent Track B: Breaking B0 Ranking + Top3 Selection Challenge

This module contains the complete research pipeline for Track B, evaluating models and Top3 portfolio construction selectors on the full `Signal` (`signal == True`) and `ACTIONABLE` (`is_actionable == 1`) universes.

## Workflow & Commands

```bash
# 1. Run empirical diagnostics on pullback_v_is_dry, lane monotonicity, and universe sizes
python -m backtest.rdagent_track_b_selector_challenge.cli diagnostic

# 2. Run RD-Agent factor discovery (with fixed budget)
python -m backtest.rdagent_track_b_selector_challenge.cli rdagent --step-n 3

# 3. Run 4-stage factor audit (leakage, semantic direction, redundancy, replay)
python -m backtest.rdagent_track_b_selector_challenge.cli audit

# 4. Run purged walk-forward cross-validation on Train and evaluate all selector families
python -m backtest.rdagent_track_b_selector_challenge.cli evaluate

# 5. Seal research lock manifest
python -m backtest.rdagent_track_b_selector_challenge.cli seal

# 6. Run unit tests
python -m pytest -q tests/test_rdagent_track_b_selector_challenge.py
```

## Structure

- `config.py`: Date splits, constants, feature definitions, universes.
- `panel.py`: PIT candidate panel loader without restricting to `b0_eligible`.
- `rdagent_bridge.py`: Train-only safe data preparation, RD-Agent 0.8.0 integration, provenance logging.
- `factor_audit.py`: 4-stage audit (Leakage, Semantic Direction, Redundancy, Deterministic Replay).
- `selectors.py`: Pure Rank Top3, Distinct Industry Top3, Portfolio-Aware Top3.
- `evaluate.py`: Purged walk-forward CV on Train, sealed validation evaluation, paired bootstrap, champion matrix.
- `diagnostics.py`: `pullback_v_is_dry` empirical distribution, lane monotonicity, factor IC.
- `cli.py`: Unified CLI for running each phase.
- `raw_rdagent/`: Sanitized raw RD-Agent factor scripts.
- `output/`: All generated CSV, JSON, Parquet, and Markdown artifacts.
