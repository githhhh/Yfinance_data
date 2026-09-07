# Current Production B0 — Absolute Quality Health Check v1.3

## Executive coordinates

No composite PASS/FAIL score is used. B0 is evaluated on two separate axes:

- **Name selection quality**: compare B0 with random portfolios using the same weekly N.
- **Capital utilization quality**: compare B0 with a de-anchored fixed-capacity 3-slot portfolio and with explicit Fill3 counterfactuals.

This prevents B0's 0/1/2-stock weeks from being either unfairly punished in the ranking audit or automatically protected by forcing the benchmark to hold the same amount of cash.

Raw-universe and market comparisons use a tradable outcome:

**first trading-session open after the snapshot -> close at entry date + 28 calendar days**.

Yahoo supplementation and all benchmark outcomes are frozen at **2026-09-02**.

## 1. Absolute Production B0

| Metric | Value |
| --- | ---: |
| Mature entry-aligned weeks | 40 |
| Mean W4 capital return | 3.47% |
| Median W4 capital return | 2.96% |
| P10 | -5.42% |
| CVaR10 | -9.97% |
| Positive week rate | 60.0% |
| Mean slot coverage | 77.0% |
| Full Top3 rate | 59.5% |

Pick-count distribution across all snapshots: **{'0': 2, '1': 8, '2': 7, '3': 25}**.

The absolute W4 cohorts overlap and must not be annualized as independent monthly trades.

## 2. Ranking quality — only weeks where ranking actually had a choice

- Eligible/mature weeks: **38**
- Gate-locked / one-feasible-portfolio weeks: **17**
- **Active-choice weeks: 21**
- Active-choice mean feasible percentile: **64.7%**
- Active-choice median feasible percentile: **66.0%**
- Mean B0 edge vs eligible random: **3.22pp**
- Median edge: **1.95pp**
- Beat-random week rate: **76.2%**
- 4-week block-bootstrap mean-edge CI: **[0.32pp, 6.68pp]**
- Aggregate oracle capture on active-choice weeks: **25.0%**

This is the cleanest estimate of B0's incremental ranking skill after hard eligibility. Weeks with only one feasible portfolio are excluded from the percentile headline because there was no ranking decision to evaluate.

## 3. Whole-system quality vs raw signal universe — tradable next-open outcome

- Strict 100%-covered support: **22 weeks** (52.4% of all snapshots)
- Mean raw price coverage across all weeks: **96.5%**
- Mean fixed-3 percentile: **50.0%**
- Median fixed-3 percentile: **53.6%**
- Mean capital spread vs raw fixed-3 random: **-0.30pp**
- Median spread: **0.24pp**
- Beat fixed-3 random week rate: **50.0%**
- Mean-edge block-bootstrap CI: **[-3.74pp, 3.17pp]**
- Raw-universe aggregate oracle capture: **-1.0%**

### Raw fixed-3 result split by B0's actual weekly position count

| B0 picks | Weeks | B0 mean | Fixed-3 random mean | Mean Δ | Median Δ | Beat rate | Median percentile |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2 | 0.00% | 2.18% | -2.18pp | -2.18pp | 0.0% | 34.2% |
| 1 | 7 | 3.17% | 3.96% | -0.79pp | 2.52pp | 57.1% | 66.2% |
| 2 | 6 | 3.34% | 5.11% | -1.77pp | 0.24pp | 50.0% | 53.3% |
| 3 | 7 | 6.75% | 4.76% | 1.99pp | 4.91pp | 57.1% | 76.3% |

Matched-N is retained as a name-selection diagnostic only:
- Mean matched-N edge: **1.62pp**
- Median matched-N edge: **2.27pp**
- Beat-rate: **65.0%**

### Temporal support coverage

Strict-support results are interpreted together with their calendar concentration.

| Comparison | Quarter | Support weeks | Total snapshots | Support rate |
| --- | --- | ---: | ---: | ---: |
| raw_fixed3_primary | 2025Q4 | 7 | 11 | 63.6% |
| raw_fixed3_primary | 2026Q1 | 12 | 12 | 100.0% |
| raw_fixed3_primary | 2026Q2 | 1 | 13 | 7.7% |
| raw_fixed3_primary | 2026Q3 | 2 | 6 | 33.3% |
| simple_momentum_20 | 2025Q4 | 7 | 11 | 63.6% |
| simple_momentum_20 | 2026Q1 | 12 | 12 | 100.0% |
| simple_momentum_20 | 2026Q2 | 1 | 13 | 7.7% |
| simple_momentum_20 | 2026Q3 | 2 | 6 | 33.3% |

## 4. Underfill / cash policy — Fill3 counterfactual ladder

### Why B0 is underfilled

| Cause | Weeks | Mean original picks | Mature next-open weeks |
| --- | ---: | ---: | ---: |
| ELIGIBILITY_SHORTAGE | 16 | 1.25 | 16 |
| FULL | 25 | 3.00 | 24 |
| INDUSTRY_CONSTRAINT | 1 | 2.00 | 1 |

### Fill3 counterfactuals

Every Fill3 policy preserves all original B0 picks. It may only fill empty slots; it never replaces an original pick. This isolates the value of cash/underfill rather than creating a new ranking system.

- RELAX_INDUSTRY: only relax distinct-industry when already-eligible names remain.
- EPS_ONLY: fill only with candidates rejected solely for EPS unknown; distinct1 remains.
- SINGLE_REJECT: fill with the highest B0-ranked candidate that failed exactly one hard gate.
- ANY_REJECT: diagnostic upper bound; any rejected known-industry candidate may fill.

| Policy | Scope | Weeks | Mean Δ vs B0 | Median Δ | Beat B0 | Slot-stop exposure Δ | Any Stop/≤-8 week Δ | Full3 | Added-pick mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0_ORIGINAL | all_mature | 41 | 0.00pp | 0.00pp | 0.0% | 0.00pp | 0.00pp | 58.5% | N/A |
| B0_ORIGINAL | underfilled_only | 17 | 0.00pp | 0.00pp | 0.0% | 0.00pp | 0.00pp | 0.0% | N/A |
| B0_FILL3_RELAX_INDUSTRY | all_mature | 41 | -0.30pp | 0.00pp | 0.0% | 1.63pp | 4.88pp | 61.0% | -18.61% |
| B0_FILL3_RELAX_INDUSTRY | underfilled_only | 17 | -0.73pp | 0.00pp | 0.0% | 3.92pp | 11.76pp | 5.9% | -18.61% |
| B0_FILL3_EPS_ONLY | all_mature | 41 | 0.22pp | 0.00pp | 7.3% | 0.81pp | 2.44pp | 61.0% | 6.71% |
| B0_FILL3_EPS_ONLY | underfilled_only | 17 | 0.53pp | 0.00pp | 17.6% | 1.96pp | 5.88pp | 5.9% | 6.71% |
| B0_FILL3_SINGLE_REJECT | all_mature | 41 | 0.86pp | 0.00pp | 26.8% | 8.13pp | 7.32pp | 100.0% | 3.37% |
| B0_FILL3_SINGLE_REJECT | underfilled_only | 17 | 2.07pp | 1.97pp | 64.7% | 19.61pp | 17.65pp | 100.0% | 3.37% |
| B0_FILL3_ANY_REJECT | all_mature | 41 | 1.24pp | 0.00pp | 29.3% | 7.32pp | 7.32pp | 100.0% | 4.43% |
| B0_FILL3_ANY_REJECT | underfilled_only | 17 | 2.98pp | 2.12pp | 70.6% | 17.65pp | 17.65pp | 100.0% | 4.43% |

### Per-pick quality — removes the mechanical effect of holding more names

| Policy | Cohort | Picks | Mean W4 | Median W4 | P10 | CVaR10 | Positive | Stop8/pick | Final W4≤-8 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0_FILL3_RELAX_INDUSTRY | original_b0 | 22 | 6.16% | 5.15% | -5.91% | -9.60% | 72.7% | 31.8% | 4.5% |
| B0_FILL3_RELAX_INDUSTRY | added_fill | 2 | -18.61% | -18.61% | -25.16% | -26.80% | 0.0% | 100.0% | 100.0% |
| B0_FILL3_EPS_ONLY | original_b0 | 22 | 6.16% | 5.15% | -5.91% | -9.60% | 72.7% | 31.8% | 4.5% |
| B0_FILL3_EPS_ONLY | added_fill | 4 | 6.71% | 4.62% | -0.44% | -2.04% | 75.0% | 25.0% | 0.0% |
| B0_FILL3_SINGLE_REJECT | original_b0 | 22 | 6.16% | 5.15% | -5.91% | -9.60% | 72.7% | 31.8% | 4.5% |
| B0_FILL3_SINGLE_REJECT | added_fill | 29 | 3.65% | 5.92% | -17.19% | -19.96% | 58.6% | 34.5% | 17.2% |
| B0_FILL3_ANY_REJECT | original_b0 | 22 | 6.16% | 5.15% | -5.91% | -9.60% | 72.7% | 31.8% | 4.5% |
| B0_FILL3_ANY_REJECT | added_fill | 29 | 5.25% | 6.30% | -15.13% | -17.98% | 62.1% | 31.0% | 13.8% |

Added fills by exact reject reason:

| Policy | Reject reason | Picks | Mean W4 | Median W4 | Positive | Stop8/pick | Final W4≤-8 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0_FILL3_RELAX_INDUSTRY | (none) | 2 | -18.61% | -18.61% | 0.0% | 100.0% | 100.0% |
| B0_FILL3_EPS_ONLY | eps_unknown | 4 | 6.71% | 4.62% | 75.0% | 25.0% | 0.0% |
| B0_FILL3_SINGLE_REJECT | eps_unknown | 3 | 7.84% | 5.92% | 66.7% | 33.3% | 0.0% |
| B0_FILL3_SINGLE_REJECT | non_actionable | 26 | 3.16% | 5.95% | 57.7% | 34.6% | 19.2% |
| B0_FILL3_ANY_REJECT | eps_unknown | 3 | 7.84% | 5.92% | 66.7% | 33.3% | 0.0% |
| B0_FILL3_ANY_REJECT | non_actionable | 23 | 3.80% | 6.09% | 56.5% | 34.8% | 17.4% |
| B0_FILL3_ANY_REJECT | non_actionable|eps_unknown | 3 | 13.74% | 12.55% | 100.0% | 0.0% | 0.0% |

### Idealized Stop8 execution scenario

For every name that triggers Stop8, scenario return is set to exactly -8%; otherwise terminal W4 is used. This is optimistic no-slippage execution and should be read as a scenario, not realized fills.

| Policy | Scope | Weeks | Stop8-exec mean | Stop8-exec median | Mean Δ vs B0 | Median Δ | Beat B0 | 95% mean-edge CI | Worst | P10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| B0_ORIGINAL | all_mature | 41 | 2.18% | 0.68% | 0.00pp | 0.00pp | 0.0% | [0.00pp, 0.00pp] | -8.00% | -3.23% |
| B0_ORIGINAL | underfilled_only | 17 | 1.49% | 0.68% | 0.00pp | 0.00pp | 0.0% | [0.00pp, 0.00pp] | -5.33% | -2.67% |
| B0_FILL3_RELAX_INDUSTRY | all_mature | 41 | 2.05% | 0.68% | -0.13pp | 0.00pp | 0.0% | [-0.33pp, 0.00pp] | -8.00% | -3.23% |
| B0_FILL3_RELAX_INDUSTRY | underfilled_only | 17 | 1.18% | 0.68% | -0.31pp | 0.00pp | 0.0% | [-0.63pp, 0.00pp] | -5.33% | -2.67% |
| B0_FILL3_EPS_ONLY | all_mature | 41 | 2.35% | 1.21% | 0.17pp | 0.00pp | 7.3% | [-0.20pp, 0.70pp] | -8.00% | -3.23% |
| B0_FILL3_EPS_ONLY | underfilled_only | 17 | 1.90% | 1.21% | 0.41pp | 0.00pp | 17.6% | [-0.31pp, 1.63pp] | -5.33% | -2.67% |
| B0_FILL3_SINGLE_REJECT | all_mature | 41 | 2.82% | 4.06% | 0.64pp | 0.00pp | 24.4% | [-0.25pp, 2.27pp] | -8.00% | -4.52% |
| B0_FILL3_SINGLE_REJECT | underfilled_only | 17 | 3.04% | 5.18% | 1.55pp | 0.35pp | 58.8% | [-0.75pp, 5.40pp] | -8.00% | -8.00% |
| B0_FILL3_ANY_REJECT | all_mature | 41 | 3.08% | 4.06% | 0.90pp | 0.00pp | 26.8% | [-0.03pp, 2.40pp] | -8.00% | -4.23% |
| B0_FILL3_ANY_REJECT | underfilled_only | 17 | 3.65% | 5.18% | 2.16pp | 1.69pp | 64.7% | [-0.01pp, 5.62pp] | -8.00% | -4.49% |

Portfolio-level any-stop/≤-8-week deltas rise mechanically when more positions are held; they are exposure diagnostics, not proof that each added name is intrinsically riskier. Per-pick Stop8 and terminal-left-tail rates above are the quality comparison.

Interpretation rule: do not change Production to always-3 merely because fixed-3 random has a higher mean. The minimal-change candidate is B0_FILL3_SINGLE_REJECT; only a coherent gain on underfilled weeks without material per-pick or portfolio left-tail deterioration would justify future shadow testing.

## 5. Eligibility gate quality

- Strict support: **22 weeks**
- Raw candidate events: **631**
- Eligible candidate events: **88**
- Acceptance rate: **13.9%**
- Top20 winner retention: **15.7%**
- Winner enrichment vs random acceptance: **1.12x**
- Bottom20 loser retention: **9.7%**
- Loser retention vs random acceptance: **0.70x**
- Final B0 Top3 winner-capture rate: **10.4%**
- Winner-capture enrichment vs Matched-N random: **1.62x**
- Winner-capture enrichment vs mechanical fixed-3 random: **0.97x**
- Mean eligible-minus-rejected W4 lift: **2.00pp**
- Median weekly gate lift: **3.83pp**

Low recall must therefore be interpreted together with the gate's acceptance rate. A gate accepting only a small fraction of the universe cannot be judged solely by the percentage of all future winners that it rejects.

## 6. Reject-reason audit — exclusive attribution separated from overlap

### Exclusive single-reason rejects

| Sole reject reason | Events | Weeks | Mean W4 | Median W4 | Positive | Stop8 | Top20 winner | >=20% winner |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| non_actionable | 376 | 22 | 2.45% | -0.68% | 46.5% | 38.8% | 20.5% | 11.7% |
| clear_geometry_failure | 67 | 18 | -0.71% | -0.73% | 43.3% | 37.3% | 14.9% | 4.5% |
| eps_unknown | 11 | 8 | 11.32% | 2.70% | 63.6% | 27.3% | 27.3% | 9.1% |

### Overlapping reason labels — descriptive only, not causal

| Reason label | Events | Multi-reason share | Mean W4 | Median W4 | Top20 winner |
| --- | ---: | ---: | ---: | ---: | ---: |
| non_actionable | 459 | 18.1% | 2.57% | -0.20% | 21.4% |
| clear_geometry_failure | 94 | 28.7% | 1.31% | -0.67% | 17.0% |
| eps_unknown | 81 | 86.4% | 4.27% | 1.15% | 27.2% |

Reason-combination counts are separately materialized; overlapping labels must never be used to claim that one gate caused the rejection outcome.

## 7. Fine-ranking information

- Mature eligible-universe Spearman support: **38 weeks**
- Mean weekly Spearman (-rank vs W4): **0.094**
- Median Spearman: **0.130**
- Positive-Spearman weeks: **66.7%**
- Selected mean minus all-eligible mean: **2.42pp** mean / **0.00pp** median

Rank buckets:

| Rank bucket | Rows | Weeks | Mean W4 | Median W4 | Positive | Stop8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rank_1_3 | 93 | 38 | 4.18% | 2.66% | 62.4% | 29.0% |
| rank_4_6 | 54 | 19 | 0.54% | 1.20% | 53.7% | 35.2% |
| rank_7_10 | 54 | 14 | -3.29% | -0.68% | 46.3% | 31.5% |
| rank_11_plus | 171 | 12 | 2.04% | 1.47% | 60.2% | 26.3% |

A strong Top bucket does not imply globally monotonic fine ranking; the entire bucket curve and Spearman distribution must be considered.

## 8. Simple raw-PIT baselines — tradable next-open outcome

| Baseline | Weeks | Mean W4 | Median W4 | Mean Δ | Median Δ | Beat B0 | 95% mean-edge CI | Mean w/o best1 | Mean w/o best2 | Stop exposure Δ | Any Stop/≤-8 week Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| closest_to_trigger | 22 | 2.49% | 1.34% | -1.58pp | 0.19pp | 50.0% | [-6.62pp, 4.01pp] | -2.73pp | -3.35pp | 21.21pp | 22.73pp |
| entry_volume | 19 | 6.42% | 2.62% | 2.49pp | -0.18pp | 47.4% | [-4.97pp, 8.38pp] | 0.42pp | -1.34pp | 22.81pp | 15.79pp |
| eps | 22 | 1.73% | 2.75% | -2.34pp | -1.16pp | 45.5% | [-9.56pp, 4.75pp] | -3.47pp | -4.63pp | 30.30pp | 40.91pp |
| momentum_20 | 22 | 11.70% | 4.15% | 7.64pp | 4.06pp | 63.6% | [2.19pp, 17.26pp] | 4.52pp | 3.05pp | 37.88pp | 50.00pp |

### Idealized Stop8 execution scenario for simple baselines

| Baseline | Weeks | Stop8-exec mean | Stop8-exec median | Mean Δ vs B0 | Median Δ | Beat B0 | 95% mean-edge CI | Worst | P10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| closest_to_trigger | 22 | -0.06% | -1.17% | -1.75pp | -2.25pp | 45.5% | [-4.37pp, 0.97pp] | -8.00% | -7.77% |
| entry_volume | 19 | 4.29% | 2.11% | 3.10pp | -2.39pp | 42.1% | [-2.46pp, 7.09pp] | -8.00% | -8.00% |
| eps | 22 | 0.42% | -1.23% | -1.27pp | -2.10pp | 40.9% | [-2.94pp, 1.60pp] | -8.00% | -7.85% |
| momentum_20 | 22 | 3.79% | -0.33% | 2.10pp | -2.67pp | 31.8% | [-2.08pp, 7.04pp] | -8.00% | -8.00% |

Relative-SPY 20-session momentum is retained as a PIT diagnostic field, not as an independent Top3 baseline: subtracting the same SPY return from every candidate in one snapshot cannot change that snapshot's cross-sectional momentum ranking.

Large mean gains with flat/negative medians or strong best-week dependence are treated as right-tail hypotheses, not as proof of stable superiority.

### Momentum20 gate-location diagnostic

This answers whether the raw momentum baseline's incremental names were already inside the B0 eligible universe or mainly outside the hard gates.

| Cohort | Picks | Share of momentum picks | Selected by B0 | Mean W4 | Median W4 | Positive | Stop8/pick | Final W4≤-8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 66 | 100.0% | 16.7% | 11.70% | 3.57% | 60.6% | 59.1% | 21.2% |
| eligible | 11 | 16.7% | 100.0% | 20.71% | 13.10% | 81.8% | 45.5% | 18.2% |
| gate_outside | 55 | 83.3% | 0.0% | 9.90% | 2.20% | 56.4% | 61.8% | 21.8% |

Gate-outside momentum picks by exact reject reason:

| Reject reason | Picks | Share of gate-outside momentum | Mean W4 | Median W4 | Positive | Stop8/pick | Final W4≤-8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clear_geometry_failure | 3 | 5.5% | 4.46% | 0.17% | 66.7% | 0.0% | 0.0% |
| clear_geometry_failure|eps_unknown | 2 | 3.6% | 34.24% | 34.24% | 50.0% | 50.0% | 50.0% |
| eps_unknown | 1 | 1.8% | 2.70% | 2.70% | 100.0% | 0.0% | 0.0% |
| non_actionable | 38 | 69.1% | 9.75% | 0.79% | 50.0% | 68.4% | 26.3% |
| non_actionable|clear_geometry_failure | 4 | 7.3% | 7.37% | 8.42% | 100.0% | 50.0% | 0.0% |
| non_actionable|clear_geometry_failure|eps_unknown | 1 | 1.8% | -3.79% | -3.79% | 0.0% | 100.0% | 0.0% |
| non_actionable|eps_unknown | 6 | 10.9% | 10.65% | 12.30% | 66.7% | 66.7% | 16.7% |

## 9. SPY / QQQ benchmark — Yahoo, same tradable clock

| Benchmark | Weeks | Benchmark mean | B0 capital mean | B0 vs fully invested | B0 vs exposure-matched | Active-pick selection spread | Full-spread 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SPY | 41 | 1.53% | 3.63% | 2.10pp | 2.51pp | 3.76pp | [-0.39pp, 5.12pp] |
| QQQ | 41 | 1.79% | 3.63% | 1.84pp | 2.29pp | 3.51pp | [-1.32pp, 5.25pp] |

## 10. Non-overlap stability — ranking, raw fixed3, and momentum20

| Comparison | Offset | Weeks | Value mean | Benchmark mean | Spread mean | Spread median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| active_choice_b0_vs_eligible_random | 0 | 6 | 4.19% | 2.07% | 2.11pp | 1.62pp |
| active_choice_b0_vs_eligible_random | 1 | 5 | 3.82% | 2.66% | 1.16pp | 1.95pp |
| active_choice_b0_vs_eligible_random | 2 | 5 | 1.43% | -0.37% | 1.80pp | 1.73pp |
| active_choice_b0_vs_eligible_random | 3 | 5 | 9.33% | 1.30% | 8.03pp | 5.36pp |
| b0_vs_raw_fixed3_next_open | 0 | 6 | 3.25% | 2.81% | 0.44pp | -1.92pp |
| b0_vs_raw_fixed3_next_open | 1 | 6 | 0.88% | 4.51% | -3.63pp | -1.59pp |
| b0_vs_raw_fixed3_next_open | 2 | 5 | 4.45% | 3.71% | 0.74pp | 0.69pp |
| b0_vs_raw_fixed3_next_open | 3 | 5 | 8.48% | 6.72% | 1.76pp | 3.31pp |
| momentum20_vs_b0_next_open | 0 | 6 | 6.64% | 3.25% | 3.39pp | 6.47pp |
| momentum20_vs_b0_next_open | 1 | 6 | 13.93% | 0.88% | 13.05pp | 5.12pp |
| momentum20_vs_b0_next_open | 2 | 5 | 7.02% | 4.45% | 2.57pp | -0.37pp |
| momentum20_vs_b0_next_open | 3 | 5 | 19.79% | 8.48% | 11.31pp | 8.11pp |
| momentum20_vs_b0_idealized_stop8 | 0 | 6 | 1.85% | 2.65% | -0.80pp | -4.97pp |
| momentum20_vs_b0_idealized_stop8 | 1 | 6 | 5.24% | 1.39% | 3.85pp | 2.98pp |
| momentum20_vs_b0_idealized_stop8 | 2 | 5 | -0.16% | 2.49% | -2.64pp | -2.67pp |
| momentum20_vs_b0_idealized_stop8 | 3 | 5 | 8.31% | 0.11% | 8.21pp | -2.67pp |

## Evidence boundary

Retrospective audit. B0 was developed with visibility into this history. Yahoo supplementation is frozen at 2026-09-02 and used only for outcome completion / benchmarks; no future bar after the frozen as-of date is allowed.

This report is a retrospective measurement instrument. It can identify where B0 appears strong or weak and which minimal counterfactual deserves future shadow testing; it cannot convert historical reuse into untouched OOS proof.

## Provenance

- source_git_sha: a00083ef1018735fa1963f7591740763c9e05e1c
- protocol_version: b0_absolute_quality_v1_3
- production_b0_hash: 115387c9861f7202c0f6b3c89fe2d2ff594544de93264901ee6d2f72e930c477
- panel_hash: d95ab4ba21831f72fdc2e434c49a42e6164d7fde6f72d187ad758f996347b5b5
- base_price_cache_hash: 1dfced4c23c639478dd42f0188648823f1c2cafea796fc1a043c56d87d55eb4f
- yahoo_supplement_hash: 7592cb7f55c3b301d06bd38882fd10b4e2302cc569b38e3287461721ad040ba0
- audit_as_of_date: 2026-09-02
