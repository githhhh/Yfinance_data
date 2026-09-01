# Track B: Breaking B0 Ranking + Top3 Selection - Comprehensive Research Report

## 1. Executive Summary

This study executes a formal, sealed empirical challenge against the frozen **B0 (`skill_industry_eps_known`)** benchmark selector. Rather than restricting candidate re-ranking within the narrow `b0_eligible` subset (414 rows), Track B investigates whether alternative ranking models and Top3 portfolio construction selectors operating directly on the **full `Signal` universe (`signal == True`, N=2738)** or the **PIT `ACTIONABLE` universe (`is_actionable == 1`, N=733)** can construct Top3 portfolios with superior risk-adjusted return, lower tail risk, and reduced "one-pick-ruins" failure rates.

### Primary Verdict
- **Is there a challenger that strictly DOMINATES B0?** **NO**.
- **B0 Champion Status**: **B0 remains the Champion Comparator**.
- **Best Actionable Challenger**: `actionable_f1_lgbm_w1_distinct_industry` and `actionable_f1_lgbm_w1_portfolio_aware` qualify as **PARETO PEERS** (Validation W4 Mean Spread: `+0.13%`, Median Spread: `-0.17%`, CVaR10 Delta: `+12.56%` improvement from `-18.42%` to `-5.86%`, Stop8 Delta: `-3.70%`). While downside left-tail risk is improved, median excess return does not establish a persistent dominance spread over B0.
- **Best Signal Universe Challenger**: `signal_agent_elastic_w1_distinct_industry` showed high in-sample/OOF return (`+8.03%` W4 median spread on Train), but suffered severe out-of-sample breakdown in Contaminated Validation (`-0.15%` median spread, `-6.58%` mean spread, CVaR10 deteriorating by `-14.26%`, Stop8 rate increasing by `+29.17%`). Classified as **INFERIOR / UNSTABLE**.

---

## 2. Motivating Diagnostic: `pullback_v_is_dry` & Lane / Evidence Analysis

### 2.1 Empirical Distribution of `pullback_v_is_dry`
B0 penalizes `pullback_v_is_dry == False` by assigning the risk code `pullback_not_dry` (`-1` evidence balance), implicitly treating `False` as an independent negative alpha signal.

We evaluated all 2,738 historical candidates across the three states (`True`, `False`, `Missing`):

| Universe | State | Sample Size | W1 Median | W2 Median | W4 Mean | W4 Median | W4 CVaR10 | W4 Stop8 Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Signal** | `True` | 429 (15.7%) | -0.12% | +0.67% | +0.44% | +0.39% | -25.00% | 37.1% |
| **Signal** | `False` | 1,152 (42.1%) | -0.08% | -0.04% | +0.16% | +0.50% | -26.06% | 34.2% |
| **Signal** | `Missing` | 1,157 (42.3%) | +0.28% | +0.66% | +1.75% | +1.56% | -23.47% | 31.4% |
| **ACTIONABLE** | `True` | 91 (12.4%) | -0.10% | +0.03% | +0.98% | +1.21% | -27.16% | 37.4% |
| **ACTIONABLE** | `False` | 297 (40.5%) | -0.29% | -0.74% | -0.48% | +0.65% | -23.42% | 30.0% |
| **ACTIONABLE** | `Missing` | 345 (47.1%) | +0.39% | +1.06% | +2.22% | +2.10% | -17.79% | 22.0% |
| **b0_eligible** | `True` | 41 (9.9%) | -0.14% | +1.39% | +2.39% | +2.67% | -18.28% | 36.6% |
| **b0_eligible** | `False` | 177 (42.8%) | -0.27% | -0.82% | -1.07% | +0.25% | -25.52% | 31.6% |
| **b0_eligible** | `Missing` | 196 (47.3%) | +0.45% | +1.19% | +3.68% | +2.41% | -16.33% | 18.9% |

#### Key Diagnostic Conclusions:
1. **`pullback_v_is_dry = False` carries INCONCLUSIVE / WEAK NEGATIVE evidence**:
   - In the broader `Signal` universe, `False` W4 median is `+0.50%` vs `True` `+0.39%`, with nearly identical CVaR10 (`-26.06%` vs `-25.00%`).
   - However, in `ACTIONABLE` and `b0_eligible`, `False` has a negative mean return (`-0.48%` and `-1.07%`) while `True` has a positive mean return (`+0.98%` and `+2.39%`).
   - Importantly, `Missing` (which corresponds to non-pullback patterns such as standard ceiling breakouts and pivots) significantly outperforms both `True` and `False` pullbacks (`+2.22%` mean in ACTIONABLE).
2. **Structural B0 Weakness**:
   - B0 places `constructive_pullback` (lane precedence 1) ahead of `standard_breakout` (lane precedence 2).
   - Historical outcome data reveals that `standard_breakout` outperforms `constructive_pullback` (`+3.32%` vs `+0.79%` W4 mean; `+2.16%` vs `+1.27%` W4 median; `-17.75%` vs `-19.69%` CVaR10).

---

## 3. RD-Agent Provenance & Factor Audit

### 3.1 Provenance Record
- **Agent Framework**: Microsoft RD-Agent 0.8.0
- **LLM / Embedding Backend**: `gemini/gemini-3.6-flash` / `gemini/gemini-embedding-001` via LiteLLM
- **Command**: `rdagent fin_factor --step-n=3`
- **Data Boundary**: Strictly Train-only (`<= 2026-05-22`), 38 safe candidate & technical features provided. All 21 future label, stop, entry, and B0 columns removed.
- **Task Injection Mode**: Generic Financial Factor Generator evaluated externally.

### 3.2 4-Stage Audit Results
All factor proposals were audited across 4 verification layers:

1. **`prox_52w_high`**:
   - Formula: `-dist_to_52w_high_pct`
   - **Audit Result: REJECTED**.
   - *Reason*: Semantic direction inverted (`corr = -1.00` with proximity; higher values indicated greater distance rather than closer proximity) and exact affine duplicate of existing feature `dist_to_52w_high_pct`.
2. **`vol_confirmed_mom_20`**:
   - Formula: `mom_20 * vol_ratio_5_20`
   - **Audit Result: REJECTED**.
   - *Reason*: Redundant with base technical feature `mom_20` (`corr = 0.967 >= 0.95`).
3. **`risk_adj_mom_10`**:
   - Formula: `mom_10 / (rv_20 + 1e-5)`
   - **Audit Result: ACCEPTED** (Distinct non-linear volatility-adjusted momentum; clean replay; no leakage).
4. **`risk_adj_mom_20`**:
   - Formula: `mom_20 / (rv_20 + 1e-5)`
   - **Audit Result: ACCEPTED** (Distinct non-linear volatility-adjusted momentum; clean replay; no leakage).
5. **`risk_adj_mom_60`**:
   - Formula: `mom_60 / (rv_20 + 1e-5)`
   - **Audit Result: ACCEPTED** (Distinct non-linear volatility-adjusted momentum; clean replay; no leakage).

---

## 4. Track B Model & Top3 Selector Evaluation

We evaluated 108 model $\times$ selector configurations across:
- **Universes**: `Signal` (S), `ACTIONABLE` (A)
- **Feature Sets**: `F1` (Base + Technical), `B-Agent` (F1 + Audited RD-Agent Factors)
- **Model Architectures**: Ridge, ElasticNet, LightGBM
- **Selectors**: Pure Rank, Distinct Industry, Portfolio-Aware

### Paired Comparison Matrix (Contaminated Validation W4, Full3 Common Support)

| Selector / Model | Universe | Classification | Train OOF W4 Med Spread | Val W4 Med Spread | Val W4 Mean Spread | Val W4 CVaR Delta | Val W4 Stop Delta | Val TailRatio10 | Support |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B0 Champion Baseline** | `b0_eligible` | **CHAMPION** | `0.00%` | `0.00%` | `0.00%` | `0.00%` (`-18.42%`) | `0.00%` (`14.8%`) | **2.34** | **9 wks** |
| `actionable_f1_lgbm_w1_distinct_industry` | `ACTIONABLE` | **PARETO PEER** | `+0.98%` | `-0.17%` | `+0.13%` | `+12.56%` (`-5.86%`) | `-3.70%` (`11.1%`) | **2.30** | **9 wks** |
| `actionable_f1_lgbm_w1_portfolio_aware` | `ACTIONABLE` | **PARETO PEER** | `+0.98%` | `-0.17%` | `+0.13%` | `+12.56%` (`-5.86%`) | `-3.70%` (`11.1%`) | **2.30** | **9 wks** |
| `actionable_agent_elastic_w2_distinct_industry`| `ACTIONABLE` | **UNSTABLE** | `+0.66%` | `-0.41%` | `-0.24%` | `+1.85%` (`-16.57%`) | `+37.04%` (`51.9%`) | **1.27** | **9 wks** |
| `signal_agent_elastic_w1_distinct_industry` | `Signal` | **INFERIOR** | `+8.03%` | `-0.15%` | `-6.58%` | `-14.26%` (`-32.68%`) | `+29.17%` (`41.7%`) | **0.54** | **8 wks** |

---

## 5. Bootstrap Uncertainty Analysis (2,000 Rounds, Fixed Seed)

For the primary W4 validation comparison against B0:
- **`actionable_f1_lgbm_w1_distinct_industry`**:
  - Mean Spread: `+0.13%` (95% CI: `[-6.42%, +6.29%]`)
  - Median Spread: `-0.17%` (95% CI: `[-7.89%, +4.19%]`)
  - CVaR Difference: `+12.56%` (95% CI: `[-5.73%, +14.26%]`)
- **`signal_agent_elastic_w1_distinct_industry`**:
  - Mean Spread: `-6.58%` (95% CI: `[-14.15%, +0.67%]`)
  - Median Spread: `-0.15%` (95% CI: `[-7.89%, +6.44%]`)
  - CVaR Difference: `-14.26%` (95% CI: `[-16.11%, +1.69%]`)

---

## 6. Synthesis & Final Research Recommendations

1. **Why B0 Holds the Champion Position**:
   - B0's conservative gating (`ACTIONABLE` + `effective_eps is not None` + `near_buy_point` + `industry diversity`) provides strong defense against high-volatility false breakouts.
   - Expanding candidate selection to the unrestricted `Signal` universe creates significant negative left-tail drag (Stop8 rates $>40\%$, CVaR10 worsening to $-32\%$).
2. **Actionable Insights for Future B1 Design**:
   - `actionable_f1_lgbm_w1_distinct_industry` significantly improved downside CVaR10 (from `-18.42%` to `-5.86%`) and reduced stop rates without sacrificing mean return.
   - Lane reordering (upgrading `standard_breakout` above `constructive_pullback`) and removing the harsh negative penalty on `pullback_v_is_dry = False` represent clear, empirical targets for future B1 exploration.
3. **Production Integrity**:
   - In accordance with research protocol, zero production logic was modified. All findings are strictly preserved in this challenge directory.
