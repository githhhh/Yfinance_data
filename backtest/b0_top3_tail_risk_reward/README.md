# B0 Top3 Tail Risk / Reward

Standalone downstream research topic for the frozen B0 historical audit.

## Question

When B0 actually recommends three names, is the portfolio's historical payoff driven by:
- broad basket strength,
- a few large winners,
- single-name blowups,
- or a favorable right-tail / left-tail asymmetry?

## Protocol

Primary scope: weeks with exactly 3 B0 picks.

Inputs are existing audited research artifacts only:
- `../b0_top3_quality_audit/output/b0_rank_position_weekly_detail.csv`
- `../b0_top3_quality_audit/output/three_tier_weekly_comparison.csv`
- `../b0_top3_quality_audit/output/b0_path_quality_to_asof.csv`

No production selector, EPS PIT, frozen price cache, or historical outcomes are modified.

Run:

```bash
python -m backtest.b0_top3_tail_risk_reward.analyze
```

Outputs:
- `output/tail_summary.csv`
- `output/internal_concentration_summary.csv`
- `output/weekly_tail_detail.csv`
- `output/path_summary.csv`
- `output/B0_TOP3_TAIL_RISK_REWARD_REPORT.md`

Definitions:
- CVaR10: mean of the worst ceil(10% × N) portfolio weeks.
- Top10 mean: mean of the best ceil(10% × N) portfolio weeks.
- Tail Ratio10 = Top10 mean / abs(CVaR10).
- Loss concentration = abs(worst losing pick) / sum(abs(all losing picks)).
- Gain concentration = best winning pick / sum(all winning picks).
- One-pick-ruins = equal-weight Top3 < 0 while the other two names average > 0.
- Stop-capped = existing Three-Tier horizon convention; stop-hit outcomes are capped at -8%, not gap-through-open execution.
