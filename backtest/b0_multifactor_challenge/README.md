# B0 Multi-Factor Champion Challenge

Research-only harness using the existing EPS_RECALIBRATED_V2 replay pools, frozen candidate outcomes and frozen price cache. It never refreshes data or changes production.

## Local baseline challenge

```bash
python -m backtest.b0_multifactor_challenge.cli prepare
python -m backtest.b0_multifactor_challenge.cli diagnostics --feature-mode f0
python -m backtest.b0_multifactor_challenge.cli evaluate --feature-mode f0
python -m backtest.b0_multifactor_challenge.cli diagnostics --feature-mode f1
python -m backtest.b0_multifactor_challenge.cli evaluate --feature-mode f1
```

F0 uses existing B0/pool information as continuous features. F1 adds PIT-safe MA/momentum/volume/volatility features derived from the already-frozen daily price cache. Ridge and ElasticNet require scikit-learn; LightGBM is optional.

## Genuine Microsoft RD-Agent factor discovery

Install current Microsoft RD-Agent (pip install rdagent) and configure its LLM backend. Then:

```bash
python -m backtest.b0_multifactor_challenge.cli rdagent --steps 2
```

The bridge uses the official rdagent fin_factor proposal + Factor CoSTEER coding stages. Agent source data is **Train-only through 2026-05-22**, with all outcome/B0 membership columns removed. We deliberately do not use RD-Agent's stock Qlib runner as the champion judge because it is a different universe/protocol.

Take a generated factor.py and replay it on the full frozen panel without exposing holdout labels to the agent:

```bash
python -m backtest.b0_multifactor_challenge.cli replay-factor --name <factor_name> --factor-py <rdagent_workspace>/factor.py
python -m backtest.b0_multifactor_challenge.cli diagnostics --feature-mode agent
python -m backtest.b0_multifactor_challenge.cli evaluate --feature-mode agent
```

## Frozen research protocol

- Candidate history: existing 40+ replay pools only.
- EPS: EPS_RECALIBRATED_V2 only.
- Price/outcomes: existing frozen artifacts only.
- Track A universe: exact pre-industry-cover B0 eligibility semantics from dashboard.skill_industry_eps_known.
- Train end: 2026-05-22.
- 2026-05-29..2026-08-07: contaminated historical validation, never model selection or agent input.
- Forward shadow starts 2026-08-28.
- W1/W2/W4 primary; W3 diagnostic.
- Expanding walk-forward inside Train with a 4-snapshot purge before each validation block.
- Challenger scores every eligible candidate first, selects Top3 with distinct industries, then applies horizon maturity. No survivor reweighting.
- B0 paired comparison uses exact overlapping snapshot/horizon support.
- No new production rule/weight is created by this harness.
