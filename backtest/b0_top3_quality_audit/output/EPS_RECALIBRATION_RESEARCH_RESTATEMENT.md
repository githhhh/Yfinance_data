# EPS Recalibration Research Restatement

- Old EPS baseline ref: 593bd333181da4fe301b3f61397c7bc95ac86ced
- Data revision: EPS_RECALIBRATED_V2
- Rule change: NO
- Production selector change: NO
- Price data change: NO
- Champion reselection: NO

## Candidate / B0 impact

- E0 membership changed candidates: 22
- E0 affected weeks: 10
- B0 selected-count changed weeks: 0
- B0 code-set/order changed weeks: 6
- B0 order-only changed weeks: 2

## EPS PIT Data Revision

- Historical pools: 43
- Signal rows: 2738
- Old resolved: 2380
- New resolved: 2467
- EPS value changed: 553
- Unknown -> known: 103
- Known -> unknown: 16
- Source changed: 138
- EPS >= 25 state changed: 239
- Provider errors: 0
- Future leakage violations: 0

## All-Historical Old -> New: Three-Tier

| Contribution / horizon | Metric | Old | New | Delta |
| :--- | :--- | ---: | ---: | ---: |
| Week 1 Executed Return | weekly_spread_screening_pct | -0.0328 | -0.0474 | -0.0146 |
| Week 1 Executed Return | weekly_spread_ranking_pct | 0.0000 | 0.0000 | 0.0000 |
| Week 1 Executed Return | weekly_spread_total_pct | 0.4771 | 0.9710 | 0.4940 |
| Week 1 Executed Return | win_rate_l1_vs_l0_pct | 47.5000 | 47.5000 | 0.0000 |
| Week 1 Executed Return | win_rate_l2_vs_l1_pct | 25.0000 | 25.0000 | 0.0000 |
| Week 1 Executed Return | p_val_screening_wilcoxon | 0.7346 | 0.6948 | -0.0397 |
| Week 1 Executed Return | p_val_ranking_wilcoxon | 0.4330 | 0.4688 | 0.0358 |
| Week 2 Executed Return | weekly_spread_screening_pct | 0.6646 | 0.9162 | 0.2516 |
| Week 2 Executed Return | weekly_spread_ranking_pct | 0.0000 | 0.0000 | 0.0000 |
| Week 2 Executed Return | weekly_spread_total_pct | 2.0759 | 2.7543 | 0.6784 |
| Week 2 Executed Return | win_rate_l1_vs_l0_pct | 57.5000 | 57.5000 | 0.0000 |
| Week 2 Executed Return | win_rate_l2_vs_l1_pct | 25.0000 | 25.0000 | 0.0000 |
| Week 2 Executed Return | p_val_screening_wilcoxon | 0.1798 | 0.0911 | -0.0887 |
| Week 2 Executed Return | p_val_ranking_wilcoxon | 0.4445 | 0.2860 | -0.1585 |
| Week 4 Executed Return | weekly_spread_screening_pct | 1.0081 | 1.0431 | 0.0349 |
| Week 4 Executed Return | weekly_spread_ranking_pct | 0.0000 | 0.0000 | 0.0000 |
| Week 4 Executed Return | weekly_spread_total_pct | 2.3633 | 2.7697 | 0.4064 |
| Week 4 Executed Return | win_rate_l1_vs_l0_pct | 65.7895 | 63.1579 | -2.6316 |
| Week 4 Executed Return | win_rate_l2_vs_l1_pct | 26.3158 | 28.9474 | 2.6316 |
| Week 4 Executed Return | p_val_screening_wilcoxon | 0.0955 | 0.1017 | 0.0062 |
| Week 4 Executed Return | p_val_ranking_wilcoxon | 0.0299 | 0.0262 | -0.0037 |
| Executed Return (to As-Of Secondary) | weekly_spread_screening_pct | 0.0000 | 0.1603 | 0.1603 |
| Executed Return (to As-Of Secondary) | weekly_spread_ranking_pct | 0.0000 | 0.0000 | 0.0000 |
| Executed Return (to As-Of Secondary) | weekly_spread_total_pct | 0.4907 | 0.1584 | -0.3323 |
| Executed Return (to As-Of Secondary) | win_rate_l1_vs_l0_pct | 47.5000 | 52.5000 | 5.0000 |
| Executed Return (to As-Of Secondary) | win_rate_l2_vs_l1_pct | 30.0000 | 30.0000 | 0.0000 |
| Executed Return (to As-Of Secondary) | p_val_screening_wilcoxon | 0.4736 | 0.3857 | -0.0879 |
| Executed Return (to As-Of Secondary) | p_val_ranking_wilcoxon | 0.1262 | 0.0854 | -0.0408 |

## B0 vs Matched-N Random

| Horizon | Metric | Old | New | Delta |
| :--- | :--- | ---: | ---: | ---: |
| W1 | b0_median_return_pct | 0.9859 | 1.1325 | 0.1466 |
| W1 | matched_random_p50_median_return_pct | 0.3164 | 0.3164 | 0.0000 |
| W1 | paired_spread_median_pct | 0.4513 | 0.5798 | 0.1285 |
| W1 | b0_mean_return_pct | 1.3131 | 1.4208 | 0.1077 |
| W1 | matched_random_p50_mean_return_pct | 0.1817 | 0.1817 | 0.0000 |
| W1 | paired_spread_mean_pct | 1.1313 | 1.2391 | 0.1078 |
| W1 | beat_random_p50_rate_pct | 55.0000 | 57.5000 | 2.5000 |
| W2 | b0_median_return_pct | 1.5747 | 1.7404 | 0.1657 |
| W2 | matched_random_p50_median_return_pct | 0.1788 | 0.1788 | 0.0000 |
| W2 | paired_spread_median_pct | 1.5435 | 1.6820 | 0.1385 |
| W2 | b0_mean_return_pct | 2.3847 | 2.6748 | 0.2901 |
| W2 | matched_random_p50_mean_return_pct | 0.2606 | 0.2606 | 0.0000 |
| W2 | paired_spread_mean_pct | 2.1240 | 2.4141 | 0.2901 |
| W2 | beat_random_p50_rate_pct | 57.5000 | 60.0000 | 2.5000 |
| W4 | b0_median_return_pct | 4.6030 | 4.6030 | 0.0000 |
| W4 | matched_random_p50_median_return_pct | 0.4693 | 0.4693 | 0.0000 |
| W4 | paired_spread_median_pct | 3.3309 | 3.6791 | 0.3482 |
| W4 | b0_mean_return_pct | 3.9783 | 4.9089 | 0.9306 |
| W4 | matched_random_p50_mean_return_pct | 1.3458 | 1.3458 | 0.0000 |
| W4 | paired_spread_mean_pct | 2.6325 | 3.5630 | 0.9305 |
| W4 | beat_random_p50_rate_pct | 63.1600 | 65.7900 | 2.6300 |

## Matched Random Percentile Old -> New

| Horizon | Metric | Old | New | Delta |
| :--- | :--- | ---: | ---: | ---: |
| W1 | median_percentile | 54.9750 | 57.5750 | 2.6000 |
| W1 | mean_percentile | 56.4872 | 57.8377 | 1.3505 |
| W1 | weeks_gt_p50_pct | 55.0000 | 57.5000 | 2.5000 |
| W2 | median_percentile | 64.0900 | 68.9000 | 4.8100 |
| W2 | mean_percentile | 57.4362 | 58.8220 | 1.3858 |
| W2 | weeks_gt_p50_pct | 57.5000 | 60.0000 | 2.5000 |
| W4 | median_percentile | 65.5200 | 72.5700 | 7.0500 |
| W4 | mean_percentile | 59.8313 | 62.0203 | 2.1890 |
| W4 | weeks_gt_p50_pct | 63.1600 | 65.7900 | 2.6300 |

## Rank Diagnostics and Top3 vs Top2 / MC3

| Horizon / segment | Metric | Old | New | Delta |
| :--- | :--- | ---: | ---: | ---: |
| W1 / All common-support weeks | hyp_a_r3_minus_r2_median_spread_pct | 2.7138 | 2.4121 | -0.3017 |
| W1 / All common-support weeks | hyp_a_r3_gt_r2_win_rate_pct | 68.0000 | 56.0000 | -12.0000 |
| W1 / All common-support weeks | hyp_a_wilcoxon_p | 0.0903 | 0.6338 | 0.5435 |
| W1 / All common-support weeks | mc3_median_pct | 0.1366 | 0.0992 | -0.0374 |
| W1 / All common-support weeks | mc3_mean_pct | 0.3980 | 0.1721 | -0.2259 |
| W1 / All common-support weeks | mc3_win_rate_pct | 64.0000 | 60.0000 | -4.0000 |
| W1 / All common-support weeks | mc3_wilcoxon_p | 0.4742 | 0.8532 | 0.3790 |
| W1 / Train-era weeks 1-30 | hyp_a_r3_minus_r2_median_spread_pct | 2.7138 | 2.6784 | -0.0354 |
| W1 / Train-era weeks 1-30 | hyp_a_r3_gt_r2_win_rate_pct | 73.3300 | 64.2900 | -9.0400 |
| W1 / Train-era weeks 1-30 | hyp_a_wilcoxon_p | 0.1354 | 0.3910 | 0.2556 |
| W1 / Train-era weeks 1-30 | mc3_median_pct | 0.5318 | 0.3262 | -0.2056 |
| W1 / Train-era weeks 1-30 | mc3_mean_pct | 0.6190 | 0.6374 | 0.0184 |
| W1 / Train-era weeks 1-30 | mc3_win_rate_pct | 66.6700 | 71.4300 | 4.7600 |
| W1 / Train-era weeks 1-30 | mc3_wilcoxon_p | 0.4887 | 0.4263 | -0.0624 |
| W1 / Contaminated validation weeks 31-40 | hyp_a_r3_minus_r2_median_spread_pct | 2.2735 | -0.4460 | -2.7195 |
| W1 / Contaminated validation weeks 31-40 | hyp_a_r3_gt_r2_win_rate_pct | 60.0000 | 45.4500 | -14.5500 |
| W1 / Contaminated validation weeks 31-40 | hyp_a_wilcoxon_p | 0.4922 | 0.7646 | 0.2724 |
| W1 / Contaminated validation weeks 31-40 | mc3_median_pct | 0.1074 | -0.3056 | -0.4130 |
| W1 / Contaminated validation weeks 31-40 | mc3_mean_pct | 0.0667 | -0.4201 | -0.4868 |
| W1 / Contaminated validation weeks 31-40 | mc3_win_rate_pct | 60.0000 | 45.4500 | -14.5500 |
| W1 / Contaminated validation weeks 31-40 | mc3_wilcoxon_p | 0.8457 | 0.4648 | -0.3809 |
| W1 / Early half | hyp_a_r3_minus_r2_median_spread_pct | 3.3126 | 2.7138 | -0.5988 |
| W1 / Early half | hyp_a_r3_gt_r2_win_rate_pct | 76.9200 | 69.2300 | -7.6900 |
| W1 / Early half | hyp_a_wilcoxon_p | 0.1677 | 0.4143 | 0.2466 |
| W1 / Early half | mc3_median_pct | 0.5318 | 0.5318 | 0.0000 |
| W1 / Early half | mc3_mean_pct | 0.6879 | 0.6809 | -0.0070 |
| W1 / Early half | mc3_win_rate_pct | 69.2300 | 69.2300 | 0.0000 |
| W1 / Early half | mc3_wilcoxon_p | 0.4973 | 0.4973 | 0.0000 |
| W1 / Late half | hyp_a_r3_minus_r2_median_spread_pct | 1.2154 | -0.8882 | -2.1036 |
| W1 / Late half | hyp_a_r3_gt_r2_win_rate_pct | 58.3300 | 41.6700 | -16.6600 |
| W1 / Late half | hyp_a_wilcoxon_p | 0.5186 | 0.6772 | 0.1586 |
| W1 / Late half | mc3_median_pct | 0.1074 | -0.1165 | -0.2239 |
| W1 / Late half | mc3_mean_pct | 0.0841 | -0.3790 | -0.4631 |
| W1 / Late half | mc3_win_rate_pct | 58.3300 | 50.0000 | -8.3300 |
| W1 / Late half | mc3_wilcoxon_p | 0.8501 | 0.5186 | -0.3315 |
| W2 / All common-support weeks | hyp_a_r3_minus_r2_median_spread_pct | 2.0812 | 0.0662 | -2.0150 |
| W2 / All common-support weeks | hyp_a_r3_gt_r2_win_rate_pct | 68.0000 | 52.0000 | -16.0000 |
| W2 / All common-support weeks | hyp_a_wilcoxon_p | 0.0236 | 0.2872 | 0.2636 |
| W2 / All common-support weeks | mc3_median_pct | 0.5042 | 0.0023 | -0.5019 |
| W2 / All common-support weeks | mc3_mean_pct | 0.9477 | 0.5918 | -0.3559 |
| W2 / All common-support weeks | mc3_win_rate_pct | 56.0000 | 52.0000 | -4.0000 |
| W2 / All common-support weeks | mc3_wilcoxon_p | 0.2411 | 0.5602 | 0.3191 |
| W2 / Train-era weeks 1-30 | hyp_a_r3_minus_r2_median_spread_pct | 3.4934 | 3.0928 | -0.4006 |
| W2 / Train-era weeks 1-30 | hyp_a_r3_gt_r2_win_rate_pct | 73.3300 | 64.2900 | -9.0400 |
| W2 / Train-era weeks 1-30 | hyp_a_wilcoxon_p | 0.1070 | 0.2412 | 0.1342 |
| W2 / Train-era weeks 1-30 | mc3_median_pct | 1.4730 | 1.5792 | 0.1062 |
| W2 / Train-era weeks 1-30 | mc3_mean_pct | 1.1094 | 1.1454 | 0.0360 |
| W2 / Train-era weeks 1-30 | mc3_win_rate_pct | 53.3300 | 57.1400 | 3.8100 |
| W2 / Train-era weeks 1-30 | mc3_wilcoxon_p | 0.3028 | 0.2958 | -0.0070 |
| W2 / Contaminated validation weeks 31-40 | hyp_a_r3_minus_r2_median_spread_pct | 0.0427 | -0.1764 | -0.2191 |
| W2 / Contaminated validation weeks 31-40 | hyp_a_r3_gt_r2_win_rate_pct | 60.0000 | 36.3600 | -23.6400 |
| W2 / Contaminated validation weeks 31-40 | hyp_a_wilcoxon_p | 0.2754 | 0.8311 | 0.5557 |
| W2 / Contaminated validation weeks 31-40 | mc3_median_pct | 0.2532 | -0.0503 | -0.3035 |
| W2 / Contaminated validation weeks 31-40 | mc3_mean_pct | 0.7052 | -0.1128 | -0.8180 |
| W2 / Contaminated validation weeks 31-40 | mc3_win_rate_pct | 60.0000 | 45.4500 | -14.5500 |
| W2 / Contaminated validation weeks 31-40 | mc3_wilcoxon_p | 0.4922 | 0.7002 | 0.2080 |
| W2 / Early half | hyp_a_r3_minus_r2_median_spread_pct | 3.9129 | 3.4934 | -0.4195 |
| W2 / Early half | hyp_a_r3_gt_r2_win_rate_pct | 76.9200 | 69.2300 | -7.6900 |
| W2 / Early half | hyp_a_wilcoxon_p | 0.1272 | 0.2163 | 0.0891 |
| W2 / Early half | mc3_median_pct | 2.3226 | 2.3226 | 0.0000 |
| W2 / Early half | mc3_mean_pct | 1.2362 | 1.1692 | -0.0670 |
| W2 / Early half | mc3_win_rate_pct | 53.8500 | 53.8500 | 0.0000 |
| W2 / Early half | mc3_wilcoxon_p | 0.2734 | 0.2734 | 0.0000 |
| W2 / Late half | hyp_a_r3_minus_r2_median_spread_pct | 0.0427 | -0.3532 | -0.3959 |
| W2 / Late half | hyp_a_r3_gt_r2_win_rate_pct | 58.3300 | 33.3300 | -25.0000 |
| W2 / Late half | hyp_a_wilcoxon_p | 0.3804 | 0.6772 | 0.2968 |
| W2 / Late half | mc3_median_pct | 0.2532 | -0.0240 | -0.2772 |
| W2 / Late half | mc3_mean_pct | 0.6352 | -0.0337 | -0.6689 |
| W2 / Late half | mc3_win_rate_pct | 58.3300 | 50.0000 | -8.3300 |
| W2 / Late half | mc3_wilcoxon_p | 0.4697 | 0.8501 | 0.3804 |
| W3 / All common-support weeks | hyp_a_r3_minus_r2_median_spread_pct | 3.0551 | 0.9510 | -2.1041 |
| W3 / All common-support weeks | hyp_a_r3_gt_r2_win_rate_pct | 66.6700 | 58.3300 | -8.3400 |
| W3 / All common-support weeks | hyp_a_wilcoxon_p | 0.0691 | 0.7257 | 0.6566 |
| W3 / All common-support weeks | mc3_median_pct | 0.9626 | 0.6576 | -0.3050 |
| W3 / All common-support weeks | mc3_mean_pct | 0.5725 | 0.1579 | -0.4146 |
| W3 / All common-support weeks | mc3_win_rate_pct | 58.3300 | 58.3300 | 0.0000 |
| W3 / All common-support weeks | mc3_wilcoxon_p | 0.4389 | 0.6840 | 0.2451 |
| W3 / Train-era weeks 1-30 | hyp_a_r3_minus_r2_median_spread_pct | 2.6704 | 0.8477 | -1.8227 |
| W3 / Train-era weeks 1-30 | hyp_a_r3_gt_r2_win_rate_pct | 73.3300 | 57.1400 | -16.1900 |
| W3 / Train-era weeks 1-30 | hyp_a_wilcoxon_p | 0.1514 | 0.9032 | 0.7518 |
| W3 / Train-era weeks 1-30 | mc3_median_pct | 1.4011 | 1.5882 | 0.1871 |
| W3 / Train-era weeks 1-30 | mc3_mean_pct | 0.7465 | 0.5275 | -0.2190 |
| W3 / Train-era weeks 1-30 | mc3_win_rate_pct | 66.6700 | 64.2900 | -2.3800 |
| W3 / Train-era weeks 1-30 | mc3_wilcoxon_p | 0.4212 | 0.5830 | 0.1618 |
| W3 / Contaminated validation weeks 31-40 | hyp_a_r3_minus_r2_median_spread_pct | 4.2626 | 1.1974 | -3.0652 |
| W3 / Contaminated validation weeks 31-40 | hyp_a_r3_gt_r2_win_rate_pct | 55.5600 | 60.0000 | 4.4400 |
| W3 / Contaminated validation weeks 31-40 | hyp_a_wilcoxon_p | 0.3008 | 0.8457 | 0.5449 |
| W3 / Contaminated validation weeks 31-40 | mc3_median_pct | -0.2388 | -0.0800 | 0.1588 |
| W3 / Contaminated validation weeks 31-40 | mc3_mean_pct | 0.2824 | -0.3595 | -0.6419 |
| W3 / Contaminated validation weeks 31-40 | mc3_win_rate_pct | 44.4400 | 50.0000 | 5.5600 |
| W3 / Contaminated validation weeks 31-40 | mc3_wilcoxon_p | 0.8203 | 0.8457 | 0.0254 |
| W3 / Early half | hyp_a_r3_minus_r2_median_spread_pct | 3.4397 | 0.8500 | -2.5897 |
| W3 / Early half | hyp_a_r3_gt_r2_win_rate_pct | 76.9200 | 61.5400 | -15.3800 |
| W3 / Early half | hyp_a_wilcoxon_p | 0.1272 | 0.8394 | 0.7122 |
| W3 / Early half | mc3_median_pct | 1.4011 | 1.4011 | 0.0000 |
| W3 / Early half | mc3_mean_pct | 0.7367 | 0.4052 | -0.3315 |
| W3 / Early half | mc3_win_rate_pct | 61.5400 | 61.5400 | 0.0000 |
| W3 / Early half | mc3_wilcoxon_p | 0.5417 | 0.6848 | 0.1431 |
| W3 / Late half | hyp_a_r3_minus_r2_median_spread_pct | 1.3428 | 1.0520 | -0.2908 |
| W3 / Late half | hyp_a_r3_gt_r2_win_rate_pct | 54.5500 | 54.5500 | 0.0000 |
| W3 / Late half | hyp_a_wilcoxon_p | 0.3652 | 0.9658 | 0.6006 |
| W3 / Late half | mc3_median_pct | 0.0789 | 0.0789 | 0.0000 |
| W3 / Late half | mc3_mean_pct | 0.3784 | -0.1344 | -0.5128 |
| W3 / Late half | mc3_win_rate_pct | 54.5500 | 54.5500 | 0.0000 |
| W3 / Late half | mc3_wilcoxon_p | 0.6377 | 0.9658 | 0.3281 |
| W4 / All common-support weeks | hyp_a_r3_minus_r2_median_spread_pct | 3.8469 | 3.0804 | -0.7665 |
| W4 / All common-support weeks | hyp_a_r3_gt_r2_win_rate_pct | 73.9100 | 60.8700 | -13.0400 |
| W4 / All common-support weeks | hyp_a_wilcoxon_p | 0.0522 | 0.8462 | 0.7940 |
| W4 / All common-support weeks | mc3_median_pct | 0.9683 | 0.1951 | -0.7732 |
| W4 / All common-support weeks | mc3_mean_pct | -0.1078 | -0.8874 | -0.7796 |
| W4 / All common-support weeks | mc3_win_rate_pct | 56.5200 | 52.1700 | -4.3500 |
| W4 / All common-support weeks | mc3_wilcoxon_p | 0.6869 | 0.8229 | 0.1360 |
| W4 / Train-era weeks 1-30 | hyp_a_r3_minus_r2_median_spread_pct | 3.8469 | 2.5286 | -1.3183 |
| W4 / Train-era weeks 1-30 | hyp_a_r3_gt_r2_win_rate_pct | 80.0000 | 57.1400 | -22.8600 |
| W4 / Train-era weeks 1-30 | hyp_a_wilcoxon_p | 0.0946 | 1.0000 | 0.9054 |
| W4 / Train-era weeks 1-30 | mc3_median_pct | 1.4361 | -0.3480 | -1.7841 |
| W4 / Train-era weeks 1-30 | mc3_mean_pct | 0.0495 | -1.1183 | -1.1678 |
| W4 / Train-era weeks 1-30 | mc3_win_rate_pct | 60.0000 | 50.0000 | -10.0000 |
| W4 / Train-era weeks 1-30 | mc3_wilcoxon_p | 0.6788 | 0.7609 | 0.0821 |
| W4 / Contaminated validation weeks 31-40 | hyp_a_r3_minus_r2_median_spread_pct | 3.1579 | 4.0126 | 0.8547 |
| W4 / Contaminated validation weeks 31-40 | hyp_a_r3_gt_r2_win_rate_pct | 62.5000 | 66.6700 | 4.1700 |
| W4 / Contaminated validation weeks 31-40 | hyp_a_wilcoxon_p | 0.4609 | 0.7344 | 0.2735 |
| W4 / Contaminated validation weeks 31-40 | mc3_median_pct | -0.1478 | 0.1951 | 0.3429 |
| W4 / Contaminated validation weeks 31-40 | mc3_mean_pct | -0.4028 | -0.5282 | -0.1254 |
| W4 / Contaminated validation weeks 31-40 | mc3_win_rate_pct | 50.0000 | 55.5600 | 5.5600 |
| W4 / Contaminated validation weeks 31-40 | mc3_wilcoxon_p | 0.9453 | 1.0000 | 0.0547 |
| W4 / Early half | hyp_a_r3_minus_r2_median_spread_pct | 3.8469 | 3.0804 | -0.7665 |
| W4 / Early half | hyp_a_r3_gt_r2_win_rate_pct | 76.9200 | 61.5400 | -15.3800 |
| W4 / Early half | hyp_a_wilcoxon_p | 0.1677 | 0.8926 | 0.7249 |
| W4 / Early half | mc3_median_pct | 0.9683 | -1.6643 | -2.6326 |
| W4 / Early half | mc3_mean_pct | -0.4485 | -1.3378 | -0.8893 |
| W4 / Early half | mc3_win_rate_pct | 53.8500 | 46.1500 | -7.7000 |
| W4 / Early half | mc3_wilcoxon_p | 0.8926 | 0.6848 | -0.2078 |
| W4 / Late half | hyp_a_r3_minus_r2_median_spread_pct | 2.9306 | 2.9306 | 0.0000 |
| W4 / Late half | hyp_a_r3_gt_r2_win_rate_pct | 70.0000 | 60.0000 | -10.0000 |
| W4 / Late half | hyp_a_wilcoxon_p | 0.2324 | 0.9219 | 0.6895 |
| W4 / Late half | mc3_median_pct | 1.0577 | 0.9652 | -0.0925 |
| W4 / Late half | mc3_mean_pct | 0.3350 | -0.3018 | -0.6368 |
| W4 / Late half | mc3_win_rate_pct | 60.0000 | 60.0000 | 0.0000 |
| W4 / Late half | mc3_wilcoxon_p | 0.6250 | 0.7695 | 0.1445 |

## Rank1 / Rank2 / Rank3 Median Return Old -> New

| Horizon | Rank | Old Median | New Median | Delta |
| :--- | :--- | ---: | ---: | ---: |
| W1 | Rank1 | 0.3201 | 0.3201 | 0.0000 |
| W1 | Rank2 | -0.0472 | 0.0459 | 0.0931 |
| W1 | Rank3 | 0.5562 | 0.3369 | -0.2193 |
| W2 | Rank1 | 0.4071 | 1.9559 | 1.5488 |
| W2 | Rank2 | -0.6604 | -0.6604 | 0.0000 |
| W2 | Rank3 | 0.5976 | 0.1996 | -0.3980 |
| W4 | Rank1 | 2.8094 | 1.5779 | -1.2315 |
| W4 | Rank2 | 0.9170 | 2.7660 | 1.8490 |
| W4 | Rank3 | 3.8111 | 3.7664 | -0.0447 |

## EPS25 Tightening Probe

| Horizon | Metric | Old | New | Delta |
| :--- | :--- | ---: | ---: | ---: |
| W1 | paired_median_spread_pct | 0.0000 | 0.0000 | 0.0000 |
| W1 | paired_mean_spread_pct | 0.1320 | -0.3463 | -0.4783 |
| W1 | win_rate_pct | 25.8100 | 20.0000 | -5.8100 |
| W1 | wilcoxon_p | 0.8986 | 0.5791 | -0.3195 |
| W1 | paired_weeks | 31.0000 | 30.0000 | -1.0000 |
| W2 | paired_median_spread_pct | 0.0000 | 0.0000 | 0.0000 |
| W2 | paired_mean_spread_pct | -0.0081 | -0.5045 | -0.4964 |
| W2 | win_rate_pct | 25.8100 | 20.0000 | -5.8100 |
| W2 | wilcoxon_p | 0.9323 | 0.3038 | -0.6285 |
| W2 | paired_weeks | 31.0000 | 30.0000 | -1.0000 |
| W3 | paired_median_spread_pct | 0.0000 | 0.0000 | 0.0000 |
| W3 | paired_mean_spread_pct | 0.0572 | -0.2986 | -0.3558 |
| W3 | win_rate_pct | 30.0000 | 27.5900 | -2.4100 |
| W3 | wilcoxon_p | 0.7436 | 0.5791 | -0.1645 |
| W3 | paired_weeks | 30.0000 | 29.0000 | -1.0000 |
| W4 | paired_median_spread_pct | 0.0000 | 0.0000 | 0.0000 |
| W4 | paired_mean_spread_pct | -0.1925 | 0.2209 | 0.4134 |
| W4 | win_rate_pct | 20.6900 | 21.4300 | 0.7400 |
| W4 | wilcoxon_p | 0.4332 | 0.9341 | 0.5009 |
| W4 | paired_weeks | 29.0000 | 28.0000 | -1.0000 |
- Verdict: MIXED / NOT YET DEMONSTRATED.

## Frozen outcome invariants

- signal_daily_prices.parquet SHA256: 1dfced4c23c639478dd42f0188648823f1c2cafea796fc1a043c56d87d55eb4f
- candidate_weekly_outcomes.parquet SHA256: 0d417cc276edf35293f4e2ea8a1dd723839c141d03ca006c276076c1ff006f83
- train_candidate_weekly_outcomes.parquet SHA256: 029f54e725d15046a5ab2902b2b009c6b6d43204823a35c32bcd2b03396dd35c

## Regenerated fixed research outputs

- B0 vs Matched-N Random
- Three-Tier decomposition
- Rank1/Rank2/Rank3 + TopK
- Layer-1 eligibility / industry / ranking decomposition
- Fixed-date contaminated historical validation

## Attribution caveat

All-historical comparisons isolate the EPS data revision most directly. Train/contaminated-validation old-to-new deltas are not attributable solely to EPS recalibration: the legacy baseline used positional week slicing while V2 uses the fixed calendar (Train 2025-10-10..2026-05-22; contaminated validation 2026-05-29..2026-08-07).

## Prior-conclusion restatement

| Prior conclusion | Verdict | Corrected interpretation |
| :--- | :--- | :--- |
| Pure Eligibility | RETAINED | Directionally positive; independent proof remains not demonstrated. |
| ACTIONABLE | RETAINED | Operationally critical gate. |
| Geometry | RETAINED | Quality/sanity filter; independent return alpha not demonstrated. |
| EPS Known | RETAINED | PIT data-quality/completeness gate; independent return alpha not demonstrated. |
| Industry Diversity | RETAINED | Portfolio construction only; robust independent advantage not demonstrated. |
| B0 W4 quality | STRENGTHENED | Promising historical medium-horizon signal, pending virgin forward validation. |
| Fine-rank monotonicity | RETAINED | Not demonstrated; B0 remains a non-monotonic top-bucket selector. |
| R3 vs R2 | WEAKENED | Prior statistical support largely disappeared after recalibration. |
| Top3 vs Top2 / MC3 | WEAKENED | Median W4 contribution remains directional but less stable. |
| Layer2 | RETAINED | No Layer2 rule qualifies. |

The audit intentionally does not search new rules, change selectors, or reselect champions.
