# B0 Quality vs Matched-N Random Report

**Frozen baseline commit**: `7cc3a439c89dd7e52135c7418590563b58b3c97e`  
**Primary benchmark**: Matched-N Random, 1,000 weekly draws, same holding count as B0 each week  
**Scope**: Phase 1/2 frozen historical audit artifacts under `backtest/b0_top3_quality_audit`  

## Executive Conclusion

B0 shows a positive but uneven selection-quality edge against equal-position Matched-N Random. The strongest direct evidence is W4: B0's median weekly portfolio return is +4.60% versus Matched-N Random P50 +0.47%, with a paired median spread of +3.68%. W4 percentile quality is also better than W1, with median weekly random percentile 72.57.

W1 should remain conservatively worded. B0's W1 paired median spread is +0.58%, and the existing three-tier decomposition does **not** prove W1 ranking alpha. The cleaner interpretation is: screening alpha is visible, W1 ranking alpha is unconfirmed, and W4 ranking alpha is a promising historical signal that must be validated by the forward shadow ledger starting 2026-08-28.

## 1. B0 vs Matched-N Random P50

Zero-pick weeks are excluded from return and percentile denominators. Each horizon also applies common censoring: if any B0 pick lacks a valid outcome for that horizon, that week is excluded for that horizon.

| horizon   |   valid_weeks |   b0_median_return_pct |   matched_random_p50_median_return_pct |   paired_spread_median_pct |   b0_mean_return_pct |   matched_random_p50_mean_return_pct |   paired_spread_mean_pct |   beat_random_p50_rate_pct |
|:----------|--------------:|-----------------------:|---------------------------------------:|---------------------------:|---------------------:|-------------------------------------:|-------------------------:|---------------------------:|
| W1        |            40 |                 1.1325 |                                 0.3164 |                     0.5798 |               1.4208 |                               0.1817 |                   1.2391 |                      57.5  |
| W2        |            40 |                 1.7404 |                                 0.1788 |                     1.682  |               2.6748 |                               0.2606 |                   2.4141 |                      60    |
| W4        |            38 |                 4.603  |                                 0.4693 |                     3.6791 |               4.9089 |                               1.3458 |                   3.563  |                      65.79 |

## 2. Weekly Random Percentile Quality

| horizon   |   valid_percentile_weeks |   median_percentile |   mean_percentile |   weeks_gt_p50_pct |   weeks_gt_p75_pct |   weeks_gt_p90_pct |
|:----------|-------------------------:|--------------------:|------------------:|-------------------:|-------------------:|-------------------:|
| W1        |                       40 |              57.575 |           57.8377 |              57.5  |              35    |              17.5  |
| W2        |                       40 |              68.9   |           58.822  |              60    |              47.5  |              15    |
| W4        |                       38 |              72.57  |           62.0203 |              65.79 |              44.74 |              18.42 |

## 3. Downside Quality and Upside Capture

Downside rows are better when lower, except worst-pick return where less negative is better. Upside rows are better when higher. Random columns use each week's Matched-N random P50, except winner/all-stopped week rates, which use the random probability of that weekly event.

| category   | metric                         |   valid_weeks |   b0_median |   matched_random_p50_median |   paired_spread_median |   b0_mean |   matched_random_p50_mean |   b0_better_than_random_p50_rate_pct |
|:-----------|:-------------------------------|--------------:|------------:|----------------------------:|-----------------------:|----------:|--------------------------:|-------------------------------------:|
| Downside   | Stop8 before Profit20 rate     |            40 |     33.3333 |                     66.6667 |                 0      |   47.0833 |                   58.75   |                                35    |
| Downside   | Stop8 ever rate                |            40 |     50      |                     66.6667 |                 0      |   52.5    |                   65.8333 |                                37.5  |
| Downside   | Gap-stop rate                  |            40 |      0      |                      0      |                 0      |    6.6667 |                    0.8333 |                                 0    |
| Downside   | All picks stopped week rate    |            40 |      0      |                     27.49   |                -8.75   |   30      |                   35.3882 |                                70    |
| Downside   | Worst pick as-of return        |            40 |     -6.2961 |                     -7.9197 |                 0.5782 |   -7.099  |                   -6.9893 |                                50    |
| Upside     | Profit20 pick rate             |            40 |     50      |                     50      |                 0      |   48.75   |                   50      |                                17.5  |
| Upside     | Weeks with >=1 Profit20 winner |            40 |    100      |                     80.23   |                12.75   |   80      |                   69.7348 |                                80    |
| Upside     | As-of mean max gain            |            40 |     23.1713 |                     24.5836 |                 0.6531 |   39.5221 |                   25.9343 |                                52.5  |
| Upside     | W1 mean max gain               |            40 |      3.873  |                      3.9165 |                 0.212  |    5.537  |                    4.097  |                                52.5  |
| Upside     | W2 mean max gain               |            40 |      6.4742 |                      4.2882 |                 2.1625 |    7.2591 |                    4.5119 |                                62.5  |
| Upside     | W4 mean max gain               |            38 |      7.3814 |                      4.2816 |                 3.2187 |    9.8934 |                    5.1918 |                                68.42 |

## 4. Stability

| horizon   | segment                                        |   valid_weeks |   beat_random_p50_rate_pct |   paired_spread_median_pct |   paired_spread_mean_pct |   median_percentile |
|:----------|:-----------------------------------------------|--------------:|---------------------------:|---------------------------:|-------------------------:|--------------------:|
| W1        | All valid weeks                                |            40 |                      57.5  |                     0.5798 |                   1.2391 |              57.575 |
| W1        | Early half                                     |            20 |                      55    |                     0.522  |                   2.0077 |              58.835 |
| W1        | Late half                                      |            20 |                      60    |                     0.5798 |                   0.4704 |              57.575 |
| W1        | Train-era weeks 1-30                           |            29 |                      55.17 |                     0.4101 |                   1.5434 |              55.15  |
| W1        | Contaminated historical validation weeks 31-40 |            11 |                      63.64 |                     0.8626 |                   0.4367 |              67.09  |
| W2        | All valid weeks                                |            40 |                      60    |                     1.682  |                   2.4141 |              68.9   |
| W2        | Early half                                     |            20 |                      75    |                     3.1093 |                   4.5417 |              79.05  |
| W2        | Late half                                      |            20 |                      45    |                    -1.0134 |                   0.2866 |              41.05  |
| W2        | Train-era weeks 1-30                           |            29 |                      65.52 |                     1.753  |                   3.1487 |              71.9   |
| W2        | Contaminated historical validation weeks 31-40 |            11 |                      45.45 |                    -1.7186 |                   0.4777 |              35.5   |
| W4        | All valid weeks                                |            38 |                      65.79 |                     3.6791 |                   3.563  |              72.57  |
| W4        | Early half                                     |            19 |                      68.42 |                     5.1554 |                   4.9302 |              76.52  |
| W4        | Late half                                      |            19 |                      63.16 |                     2.0461 |                   2.1959 |              64.61  |
| W4        | Train-era weeks 1-30                           |            29 |                      65.52 |                     5.1554 |                   4.1478 |              74.2   |
| W4        | Contaminated historical validation weeks 31-40 |             9 |                      66.67 |                     2.0461 |                   1.679  |              64.61  |

## 5. Alpha Decomposition: L0 / L1 / B0

L0 is blind random from the signal pool. L1 is random after production-like eligibility screening and industry de-duplication. B0 is the deterministic production ranking/selection layer. Therefore:

- **Screening alpha** = L1 minus L0.
- **Ranking alpha** = B0/L2 minus L1.
- **Total alpha** = B0/L2 minus L0.

| horizon       |   mature_eval_weeks |   l0_signal_median_pct |   l1_screened_median_pct |   b0_l2_median_pct |   screening_alpha_weekly_spread_median_pct |   ranking_alpha_weekly_spread_median_pct |   active_rank_weeks_count |   active_rank_spread_ranking_pct |   active_rank_win_rate_b0_vs_l1_pct |   p_val_ranking_wilcoxon | interpretation                                                                        |
|:--------------|--------------------:|-----------------------:|-------------------------:|-------------------:|-------------------------------------------:|-----------------------------------------:|--------------------------:|---------------------------------:|------------------------------------:|-------------------------:|:--------------------------------------------------------------------------------------|
| W1            |                  40 |                 0.107  |                   0.3193 |             0.8302 |                                    -0.0474 |                                        0 |                        21 |                           0      |                             47.619  |                   0.4688 | W1 ranking alpha not proven; screening/total lift is visible but weak.                |
| W2            |                  40 |                -0.366  |                   0.3261 |             0.7127 |                                     0.9162 |                                        0 |                        21 |                           0      |                             47.619  |                   0.286  | Supportive context; not the primary ranking-alpha claim.                              |
| W4            |                  38 |                -0.714  |                   0.8902 |             1.7125 |                                     1.0431 |                                        0 |                        19 |                           2.3786 |                             57.8947 |                   0.0262 | W4 ranking alpha is a promising historical signal, pending forward shadow validation. |
| AsOf executed |                  40 |                -0.8431 |                   2.141  |             2.3388 |                                     0.1603 |                                        0 |                        21 |                           2.1722 |                             57.1429 |                   0.0854 | Supportive context; not the primary ranking-alpha claim.                              |

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
