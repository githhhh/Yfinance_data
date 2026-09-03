# Current Production B0 — Absolute Quality Health Check

## Executive coordinates

- No arbitrary PASS/FAIL score is used. The report exposes the raw coordinates, uncertainty and oracle headroom; final interpretation must use them jointly.

This audit measures the current Production B0 as-is. B0 is never used to define the raw-signal benchmark universe. Current B0 eligibility/rank/Top3 are recomputed from Production source; the frozen panel's old b0_eligible/is_b0 helper columns are not trusted.

Two outcome systems are deliberately separated:

1. **Entry-aligned W4** — used for B0-eligible ranking/random/oracle comparisons.
2. **Snapshot-close +28 calendar-day W4** — used for the raw signal universe, eligibility opportunity cost, simple baselines, and SPY/QQQ. This avoids pretending non-ACTIONABLE raw signals have a comparable Production entry event.

## 1. Absolute B0 W4 cohort quality

| Metric | Current B0 |
| --- | ---: |
| Mature/cash weeks | 40 |
| Mean capital-adjusted W4 | 3.47% |
| Median capital-adjusted W4 | 2.96% |
| P10 | -5.42% |
| P25 | -1.26% |
| P75 | 7.11% |
| P90 | 12.75% |
| CVaR10 | -9.97% |
| Positive-week rate | 60.0% |
| Worst cohort | -16.66% |
| Best cohort | 32.01% |
| Mean capital Stop8 | 20.83% |
| One-pick-ruin week rate | 47.5% |
| Mean slot coverage | 77.0% |
| Full Top3 rate | 59.5% |
| Zero-pick weeks | 2 |

Moving-block (4-week) bootstrap for the overlapping W4 cohorts:

- Mean 95% CI: **[0.96%, 5.79%]**
- Median 95% CI: **[-0.04%, 4.37%]**

These are cohort-selection diagnostics, not a tradable CAGR. Weekly W4 windows overlap.

## 2. Does B0 ranking add value inside its own eligible universe?

- Strict common-maturity support: **38 weeks**
- Median weekly feasible-portfolio percentile: **99.7%**
- Mean weekly feasible-portfolio percentile: **80.5%**
- Mean W4 edge vs exact eligible distinct-industry random: **1.78%**
- Median W4 edge vs random: **0.00%**
- Beat-random-mean week rate: **55.3%**
- Edge block-bootstrap mean CI: **[0.16%, 3.92%]**
- Eligible-universe aggregate oracle capture: **25.0%**

Interpretation: this section isolates the ranking/selection layer after the current hard gates. A percentile near 50% means the detailed B0 ranking is not doing much once eligibility has already done the filtering.

## 3. Does the whole B0 system beat the raw signal universe?

- Raw benchmark support with required price coverage: **21 weeks**
- Median weekly percentile vs raw-signal **fixed-capacity** distinct-industry random: **43.1%**
- Mean weekly percentile: **50.4%**
- Mean snapshot-W4 edge vs raw random: **-0.40%**
- Median snapshot-W4 edge: **-1.46%**
- Beat-random-mean week rate: **47.6%**
- Edge block-bootstrap mean CI: **[-3.32%, 1.89%]**
- Raw-universe aggregate oracle capture: **-1.5%**

- Conditional Matched-N raw percentile median (name-selection only): **75.0%**

This is the most important total-system coordinate: raw signal names are not pre-filtered by b0_eligible or Lane. The **primary fixed-capacity benchmark** mechanically fills up to three distinct-industry slots whenever the raw universe can do so, so B0 abstention/underfill is evaluated rather than copied into the benchmark. Matched-N is retained only as a conditional name-selection diagnostic.

## 4. Eligibility gate: how many future winners does B0 retain or reject?

- Support weeks: **21**
- Mean raw price-outcome coverage: **94.1%**
- Top-20% future-winner retention by B0 eligibility: **17.1%**
- Top-20% future-winner capture by final B0 picks: **12.6%**
- Winner rate among rejected candidates: **20.4%**
- Bottom-20% future-loser rejection rate: **92.8%**
- >=20% big-winner retention: **15.7%**
- Mean eligible-minus-rejected snapshot-W4 lift: **1.99%**
- Median weekly gate lift: **4.49%**

### Hard-reject reason diagnostics

| Reject reason | Rows | Weeks | Mean W4 | Median W4 | Top20 winner rate | >=20% winner rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| non_actionable | 386 | 21 | 2.17% | 0.28% | 19.7% | 9.6% |
| clear_geometry_failure | 71 | 18 | 1.88% | 0.74% | 23.9% | 8.5% |
| eps_unknown | 73 | 19 | 3.55% | 2.10% | 24.7% | 8.2% |

## 5. Fine-ranking information across the entire eligible universe

- Strict-maturity support: **38 weeks**
- Mean weekly Spearman (-eligible_rank vs W4): **0.094**
- Median weekly Spearman: **0.130**
- Positive-Spearman week rate: **66.7%**
- B0 selected mean minus all-eligible mean W4: **2.42%** mean / **0.00%** median

Rank-bucket outcome profile:

| Eligible rank bucket | Candidate rows | Weeks | Mean W4 | Median W4 | Positive | Stop8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rank_1_3 | 93 | 38 | 4.18% | 2.66% | 62.4% | 29.0% |
| rank_4_6 | 54 | 19 | 0.54% | 1.20% | 53.7% | 35.2% |
| rank_7_10 | 54 | 14 | -3.29% | -0.68% | 46.3% | 31.5% |
| rank_11_plus | 171 | 12 | 2.04% | 1.47% | 60.2% | 26.3% |

## 6. Simple de-anchored rules from the raw signal universe

These rules do not use b0_eligible, B0 Lane, B0 rank, reason_codes, or future outcomes. They use one raw PIT feature plus distinct-industry selection and a de-anchored fixed capacity of up to three positions. Therefore they also challenge B0 abstention/underfill instead of copying B0's weekly position count.

| Baseline | Weeks | Mean W4 | Median W4 | Mean Δ vs B0 | Median Δ vs B0 | Beat B0 | Pick coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| closest_to_trigger | 21 | 2.88% | 1.26% | -0.68% | -0.44% | 42.9% | 100.0% |
| entry_volume | 21 | 5.89% | 3.80% | 2.33% | 1.31% | 57.1% | 95.2% |
| eps | 21 | 1.94% | 4.75% | -1.62% | 0.10% | 52.4% | 100.0% |
| momentum_20 | 21 | 9.17% | 4.75% | 5.61% | -0.09% | 47.6% | 100.0% |
| rel_spy_20 | 21 | 0.00% | 0.00% | -3.56% | -2.94% | 28.6% | 0.0% |

## 7. Market benchmark

- B0 active-pick snapshot-W4 mean spread vs SPY: **N/A** (0 weeks)
- B0 capital-adjusted spread vs exposure-matched SPY: **N/A**
- B0 capital-adjusted spread vs fully invested SPY: **N/A**
- B0 active-pick snapshot-W4 mean spread vs QQQ: **N/A** (0 weeks)
- B0 capital-adjusted spread vs fully invested QQQ: **N/A**

## 8. Four-offset non-overlap stability

Each row takes every fourth weekly cohort. This removes W4 horizon overlap within an offset.

| Comparison | Offset | Weeks | Value mean | Value median | Benchmark mean | Spread mean | Spread median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| b0_absolute_entry_w4 | 0 | 10 | 0.39% | -0.04% | N/A | N/A | N/A |
| b0_absolute_entry_w4 | 1 | 10 | 1.11% | 3.27% | N/A | N/A | N/A |
| b0_absolute_entry_w4 | 2 | 10 | 4.58% | 4.61% | N/A | N/A | N/A |
| b0_absolute_entry_w4 | 3 | 10 | 7.80% | 4.15% | N/A | N/A | N/A |
| b0_vs_eligible_random | 0 | 10 | 3.94% | 2.24% | 0.82% | 3.12% | 0.00% |
| b0_vs_eligible_random | 1 | 10 | 8.36% | 4.81% | 6.90% | 1.46% | 0.00% |
| b0_vs_eligible_random | 2 | 9 | 1.39% | -0.07% | 0.18% | 1.21% | 0.00% |
| b0_vs_eligible_random | 3 | 9 | 0.37% | 3.61% | -0.85% | 1.22% | 0.83% |
| b0_vs_raw_random_fixed_capacity | 0 | 6 | 2.08% | 0.00% | 2.17% | -0.08% | -2.50% |
| b0_vs_raw_random_fixed_capacity | 1 | 5 | 2.27% | 3.51% | 4.34% | -2.07% | 1.43% |
| b0_vs_raw_random_fixed_capacity | 2 | 5 | 4.32% | 4.42% | 4.44% | -0.11% | -1.46% |
| b0_vs_raw_random_fixed_capacity | 3 | 5 | 5.86% | 2.94% | 5.25% | 0.61% | 2.81% |

## 9. Evidence boundary

Retrospective diagnostic only. Current B0 and its components were developed with substantial visibility into this historical period; no p-value or CI is treated as virgin OOS proof.

The most reliable interpretation hierarchy is:

1. raw-signal percentile / edge = total B0 selection-system quality;
2. eligible-random percentile / edge = incremental ranking quality after hard gates;
3. eligibility winner retention = gate opportunity-cost quality;
4. oracle capture = how much headroom remains;
5. four-offset consistency = whether overlapping W4 labels exaggerate stability.

No single mean return or p-value is treated as a sufficient verdict.

## Provenance

- source_git_sha: 0ac4f69ba39722091df10553cfcf4ea677dde4e0
- production_b0_hash: 115387c9861f7202c0f6b3c89fe2d2ff594544de93264901ee6d2f72e930c477
- panel_hash: d95ab4ba21831f72fdc2e434c49a42e6164d7fde6f72d187ad758f996347b5b5
- price_cache_hash: 1dfced4c23c639478dd42f0188648823f1c2cafea796fc1a043c56d87d55eb4f
- snapshots: **42** (2025-10-10 .. 2026-08-07)
