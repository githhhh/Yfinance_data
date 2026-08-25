# B0 Quality vs Matched-N Random Report

**Frozen baseline commit**: `7cc3a439c89dd7e52135c7418590563b58b3c97e`  
**Primary benchmark**: Matched-N Random, 1,000 weekly draws, same holding count as B0 each week  
**Scope**: Phase 1/2 frozen historical audit artifacts under `backtest/b0_top3_quality_audit`  

## Executive Conclusion

B0 shows a positive but uneven selection-quality edge against equal-position Matched-N Random. The strongest direct evidence is W4: B0's median weekly portfolio return is +4.60% versus Matched-N Random P50 +0.47%, with a paired median spread of +3.33%. W4 percentile quality is also better than W1, with median weekly random percentile 65.52.

W1 should remain conservatively worded. B0's W1 paired median spread is +0.45%, and the existing three-tier decomposition does **not** prove W1 ranking alpha. The cleaner interpretation is: screening alpha is visible, W1 ranking alpha is unconfirmed, and W4 ranking alpha is a promising historical signal that must be validated by the forward shadow ledger starting 2026-08-28.

## 1. B0 vs Matched-N Random P50

Zero-pick weeks are excluded from return and percentile denominators. Each horizon also applies common censoring: if any B0 pick lacks a valid outcome for that horizon, that week is excluded for that horizon.

| horizon   |   valid_weeks |   b0_median_return_pct |   matched_random_p50_median_return_pct |   paired_spread_median_pct |   b0_mean_return_pct |   matched_random_p50_mean_return_pct |   paired_spread_mean_pct |   beat_random_p50_rate_pct |
|:----------|--------------:|-----------------------:|---------------------------------------:|---------------------------:|---------------------:|-------------------------------------:|-------------------------:|---------------------------:|
| W1        |            40 |                 0.9859 |                                 0.3164 |                     0.4513 |               1.3131 |                               0.1817 |                   1.1313 |                      55    |
| W2        |            40 |                 1.5747 |                                 0.1788 |                     1.5435 |               2.3847 |                               0.2606 |                   2.124  |                      57.5  |
| W4        |            38 |                 4.603  |                                 0.4693 |                     3.3309 |               3.9783 |                               1.3458 |                   2.6325 |                      63.16 |

## 2. Weekly Random Percentile Quality

| horizon   |   valid_percentile_weeks |   median_percentile |   mean_percentile |   weeks_gt_p50_pct |   weeks_gt_p75_pct |   weeks_gt_p90_pct |
|:----------|-------------------------:|--------------------:|------------------:|-------------------:|-------------------:|-------------------:|
| W1        |                       40 |              54.975 |           56.4872 |              55    |              30    |              17.5  |
| W2        |                       40 |              64.09  |           57.4362 |              57.5  |              45    |              12.5  |
| W4        |                       38 |              65.52  |           59.8313 |              63.16 |              39.47 |              13.16 |

## 3. Downside Quality and Upside Capture

Downside rows are better when lower, except worst-pick return where less negative is better. Upside rows are better when higher. Random columns use each week's Matched-N random P50, except winner/all-stopped week rates, which use the random probability of that weekly event.

| category   | metric                         |   valid_weeks |   b0_median |   matched_random_p50_median |   paired_spread_median |   b0_mean |   matched_random_p50_mean |   b0_better_than_random_p50_rate_pct |
|:-----------|:-------------------------------|--------------:|------------:|----------------------------:|-----------------------:|----------:|--------------------------:|-------------------------------------:|
| Downside   | Stop8 before Profit20 rate     |            40 |     33.3333 |                     66.6667 |                 0      |   47.0833 |                   58.75   |                                32.5  |
| Downside   | Stop8 ever rate                |            40 |     41.6667 |                     66.6667 |                 0      |   52.5    |                   65.8333 |                                37.5  |
| Downside   | Gap-stop rate                  |            40 |      0      |                      0      |                 0      |    6.6667 |                    0.8333 |                                 0    |
| Downside   | All picks stopped week rate    |            40 |      0      |                     27.49   |                -8.75   |   30      |                   35.3882 |                                70    |
| Downside   | Worst pick as-of return        |            40 |     -6.2961 |                     -7.9197 |                 0.5782 |   -6.8463 |                   -6.9893 |                                50    |
| Upside     | Profit20 pick rate             |            40 |     41.6667 |                     50      |                 0      |   46.25   |                   50      |                                12.5  |
| Upside     | Weeks with >=1 Profit20 winner |            40 |    100      |                     80.23   |                12.1    |   77.5    |                   69.7348 |                                77.5  |
| Upside     | As-of mean max gain            |            40 |     22.202  |                     24.5836 |                -0.0613 |   38.2492 |                   25.9343 |                                47.5  |
| Upside     | W1 mean max gain               |            40 |      3.9491 |                      3.9165 |                 0.1889 |    5.4342 |                    4.097  |                                52.5  |
| Upside     | W2 mean max gain               |            40 |      6.3053 |                      4.2882 |                 2.146  |    7.1066 |                    4.5119 |                                62.5  |
| Upside     | W4 mean max gain               |            38 |      6.9859 |                      4.2816 |                 2.7622 |    8.9066 |                    5.1918 |                                65.79 |

## 4. Stability

| horizon   | segment                                        |   valid_weeks |   beat_random_p50_rate_pct |   paired_spread_median_pct |   paired_spread_mean_pct |   median_percentile |
|:----------|:-----------------------------------------------|--------------:|---------------------------:|---------------------------:|-------------------------:|--------------------:|
| W1        | All valid weeks                                |            40 |                      55    |                     0.4513 |                   1.1313 |              54.975 |
| W1        | Early half                                     |            20 |                      55    |                     0.522  |                   2.0077 |              58.835 |
| W1        | Late half                                      |            20 |                      55    |                     0.4513 |                   0.2549 |              54.975 |
| W1        | Train-era weeks 1-30                           |            30 |                      56.67 |                     0.5386 |                   1.6382 |              57.575 |
| W1        | Contaminated historical validation weeks 31-40 |            10 |                      50    |                     0.1783 |                  -0.3891 |              51.55  |
| W2        | All valid weeks                                |            40 |                      57.5  |                     1.5435 |                   2.124  |              64.09  |
| W2        | Early half                                     |            20 |                      75    |                     3.1093 |                   4.5417 |              79.05  |
| W2        | Late half                                      |            20 |                      40    |                    -1.4371 |                  -0.2936 |              36.185 |
| W2        | Train-era weeks 1-30                           |            30 |                      63.33 |                     1.682  |                   2.9218 |              68.9   |
| W2        | Contaminated historical validation weeks 31-40 |            10 |                      40    |                    -1.5336 |                  -0.2692 |              36.035 |
| W4        | All valid weeks                                |            38 |                      63.16 |                     3.3309 |                   2.6325 |              65.52  |
| W4        | Early half                                     |            19 |                      68.42 |                     5.1554 |                   4.9302 |              76.52  |
| W4        | Late half                                      |            19 |                      57.89 |                     1.7413 |                   0.3347 |              63.93  |
| W4        | Train-era weeks 1-30                           |            30 |                      63.33 |                     4.7377 |                   3.2641 |              69.565 |
| W4        | Contaminated historical validation weeks 31-40 |             8 |                      62.5  |                     1.7498 |                   0.2637 |              64.27  |

## 5. Alpha Decomposition: L0 / L1 / B0

L0 is blind random from the signal pool. L1 is random after production-like eligibility screening and industry de-duplication. B0 is the deterministic production ranking/selection layer. Therefore:

- **Screening alpha** = L1 minus L0.
- **Ranking alpha** = B0/L2 minus L1.
- **Total alpha** = B0/L2 minus L0.

| horizon       |   mature_eval_weeks |   l0_signal_median_pct |   l1_screened_median_pct |   b0_l2_median_pct |   screening_alpha_weekly_spread_median_pct |   ranking_alpha_weekly_spread_median_pct |   active_rank_weeks_count |   active_rank_spread_ranking_pct |   active_rank_win_rate_b0_vs_l1_pct |   p_val_ranking_wilcoxon | interpretation                                                                        |
|:--------------|--------------------:|-----------------------:|-------------------------:|-------------------:|-------------------------------------------:|-----------------------------------------:|--------------------------:|---------------------------------:|------------------------------------:|-------------------------:|:--------------------------------------------------------------------------------------|
| W1            |                  40 |                 0.1385 |                   0.233  |             0.6535 |                                    -0.0328 |                                        0 |                        21 |                           0      |                             47.619  |                   0.433  | W1 ranking alpha not proven; screening/total lift is visible but weak.                |
| W2            |                  40 |                -0.2741 |                   0.4064 |             0.5809 |                                     0.6646 |                                        0 |                        21 |                           0      |                             47.619  |                   0.4445 | Supportive context; not the primary ranking-alpha claim.                              |
| W4            |                  38 |                -0.8725 |                   0.7277 |             1.2573 |                                     1.0081 |                                        0 |                        19 |                           2.0774 |                             52.6316 |                   0.0299 | W4 ranking alpha is a promising historical signal, pending forward shadow validation. |
| AsOf executed |                  40 |                -0.8332 |                   0.8019 |             2.0443 |                                     0      |                                        0 |                        21 |                           0.8432 |                             57.1429 |                   0.1262 | Supportive context; not the primary ranking-alpha claim.                              |

## 6. Methodology Notes

- **Matched-N Random**: each random portfolio samples the same number of names B0 actually selected that week. A 1-pick B0 week is compared with random 1-pick portfolios; a 3-pick B0 week is compared with random 3-pick portfolios.
- **Common censoring**: B0 and random portfolios require all sampled picks to have valid entry and horizon outcomes. No survivor reweighting is allowed.
- **Maturity**: W1/W2/W4 denominators are horizon-specific and exclude immature or missing-outcome weeks.
- **0-pick rule**: 2 zero-pick week(s) out of 42 calendar benchmark rows are marked not applicable and excluded from return, percentile, downside and upside denominators. Active Matched-N rows: 40.
- **Historical validation caveat**: Weeks 31-40 are contaminated historical validation because prior research already touched that period. They are useful for one-way audit reporting, not pure out-of-sample proof.
- **Forward shadow start**: unbiased validation starts from the pre-registered 2026-08-28 forward shadow ledger for B0, Pure Freshness, and Pure Close Position.

## 7. Reproducibility

Generated by:

```bash
PYTHONPATH=. python backtest/b0_top3_quality_audit/generate_b0_quality_vs_matched_random_report.py
```

Generated companion CSVs:

- `b0_quality_vs_matched_random_horizon_summary.csv`
- `b0_quality_vs_matched_random_percentile_summary.csv`
- `b0_quality_vs_matched_random_downside_upside_summary.csv`
- `b0_quality_vs_matched_random_stability_summary.csv`
- `b0_quality_vs_matched_random_alpha_decomposition.csv`
- `b0_quality_vs_matched_random_weekly_detail.csv`
