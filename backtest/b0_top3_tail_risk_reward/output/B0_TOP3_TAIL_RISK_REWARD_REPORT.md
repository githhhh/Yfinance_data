# B0 Top3 Left-Tail / Right-Tail Risk-Reward Study

**Research scope**: historical weeks where B0 actually selected exactly 3 names.  
**Primary sample**: 25 full-Top3 weeks; W4 raw common-support is 23 mature weeks.  
**Upstream artifacts**: `b0_rank_position_weekly_detail.csv`, `three_tier_weekly_comparison.csv`, `b0_path_quality_to_asof.csv`.

## Executive conclusion

B0 Top3 is not a smooth three-name basket. Its historical W4 shape is **positively right-skewed but highly single-name concentrated on both tails**.

- Raw W4 median: **+3.77%**
- Raw W4 CVaR10: **-11.47%**
- Raw W4 Top10% mean: **+26.81%**
- Raw W4 Tail Ratio10: **2.34**
- In negative W4 weeks, **66.7%** were cases where the other two names averaged positive but one loser dragged the equal-weight Top3 below zero.
- Average worst-pick share of total negative-name losses in negative W4 weeks: **80.9%**.
- In the best 10% W4 weeks, the best single name contributed **85.3%** of all positive-name gains on average.

The implication is important: **Top3 diversification is useful, but B0's realized edge still behaves like a small basket designed to capture a large winner while containing single-name blowups.**

## 1. Portfolio tail shape

| Horizon | Mode | Weeks | Median | P10 | CVaR10 | P90 | Top10 Mean | Tail Ratio10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| W1 | Raw | 25 | 0.69% | -4.39% | -5.75% | 3.91% | 6.21% | 1.08 |
| W1 | Stop-capped | 25 | 0.62% | -2.74% | -5.13% | 3.38% | 5.53% | 1.08 |
| W2 | Raw | 25 | 0.10% | -7.32% | -10.23% | 8.18% | 11.40% | 1.11 |
| W2 | Stop-capped | 25 | 0.05% | -4.32% | -5.56% | 6.91% | 9.89% | 1.78 |
| W4 | Raw | 23 | **3.77%** | -7.17% | **-11.47%** | 17.84% | **26.81%** | **2.34** |
| W4 | Stop-capped | 23 | **1.34%** | -2.63% | **-4.83%** | 8.79% | **14.11%** | **2.92** |

### Interpretation

- W1 is almost tail-symmetric: raw Tail Ratio10 ≈ **1.08**.
- W2 begins to benefit from stop-capping: Tail Ratio10 rises from **1.11** to **1.78**.
- W4 is where B0's asymmetry becomes visible: raw Top10 mean (**+26.81%**) is more than twice the magnitude of raw CVaR10 (**-11.47%**), giving Tail Ratio10 **2.34**.

## 2. What the -8% stop changes

On the same 23 mature W4 raw-support weeks:

- CVaR10 improves from **-11.47%** to **-4.83%** — roughly **57.9%** reduction in left-tail magnitude.
- Top10 mean falls from **+26.81%** to **+14.11%** — roughly **47.4%** reduction in right-tail capture.
- Median falls from **+3.77%** to **+1.34%**.
- Tail Ratio10 nevertheless improves from **2.34** to **2.92**.

So the stop is doing exactly what a convexity control should do — **compressing the left tail more aggressively than the right tail** — but it also creates a real opportunity cost by exiting names that later recover and become large winners.

> Important protocol note: `stop_capped` is the existing Three-Tier horizon convention: once stop-8 is hit by the horizon, the horizon value is capped at -8%. It is not gap-through-open execution. Gap-stop risk is therefore reported separately below.

## 3. Internal Top3 concentration

| Horizon | Negative Weeks | One-Pick-Ruins / Negative | Avg Loss Concentration | Avg Gain Concentration | Left10 Worst-Pick Concentration | Right10 Best-Pick Concentration |
|---|---:|---:|---:|---:|---:|---:|
| W1 | 8 | 25.0% | 74.7% | 75.4% | 68.8% | 66.3% |
| W2 | 10 | 30.0% | 72.6% | 76.5% | 65.4% | 67.0% |
| W4 | 9 | 66.7% | 80.9% | 72.4% | 62.8% | 85.3% |

### W4 is the key result

- **6 / 9 negative W4 weeks** were one-pick-ruins-the-basket cases.
- Across all negative W4 weeks, the worst name accounts for about **80.9%** of negative-name loss magnitude.
- Across the best 10% W4 weeks, the best name accounts for about **85.3%** of positive-name gains.

This means the same structural property drives both tails: **single-name concentration**.

## 4. Path risk across the 75 picks in full-Top3 weeks

| Metric | Count | Rate |
|---|---:|---:|
| Stop8 before Profit20 | 28 | 37.3% |
| Profit20 before Stop8 | 24 | 32.0% |
| Stop8 ever | 30 | 40.0% |
| Gap stop | 5 | 6.7% |
| Profit20 hit | 30 | 40.0% |

Path odds:

[
\frac{P(Profit20\ before\ Stop8)}{P(Stop8\ before\ Profit20)}
= 0.86
]

At the individual-pick path level, +20-before-stop is therefore **not** more frequent than stop-before-+20. The portfolio's favorable W4 right tail comes from the magnitude of the winners, not from a dominant per-name path win probability.

## 5. Research verdict

**PASS — B0 shows historical positive W4 tail asymmetry, but not broad-based basket alpha.**

The clean interpretation is:

1. **Diversification matters**: three slots reduce dependence on correctly identifying Rank1.
2. **Left-tail risk is mostly single-name risk**: a bad name can dominate basket losses.
3. **Right-tail reward is also mostly single-name reward**: the strongest weeks depend heavily on one exceptional winner.
4. **The -8% stop improves tail efficiency**, especially by W2/W4, but materially sacrifices rebound/right-tail participation.
5. This supports treating B0 as an **equal-weight Top3 bucket selector with explicit single-name risk control**, not as a fine-ranking engine.

## Limitations

- Full-Top3 sample is only 25 weeks; W4 raw common-support is 23 weeks.
- Tail10 means only 3 weeks at this sample size; Tail20 is included in machine-readable output as robustness.
- Historical period is the existing frozen replay period; no new data or production rules were introduced.
- Stop-capped horizon results are not gap-through-open execution; gap-stop frequency is separately measured from path outcomes.
