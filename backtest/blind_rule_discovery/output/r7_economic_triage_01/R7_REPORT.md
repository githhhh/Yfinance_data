# R7 Frozen Risk Economic Triage

Known-history retrospective research. NOT AN UNTOUCHED HOLDOUT.
RD-Agent calls: 0. No discovery, threshold search, champion selection or production change.
Terminal close-return attribution, not portfolio P&L and not realized stop-execution P&L.
W1/W2/W4 primary; W3 diagnostic only. No best-horizon selection.

## Input and Protocol

- Samples: 8983; usable-entry snapshot weeks: 182; tickers: 1140.
- Input SHA256: `8bfd4411a7751177955927d8b145f15bdd077f49d26c7ddc6f9d6ef44fa6e83e`.
- Protocol SHA256: `07c084dd7d778a829d3043d0ee71cc5386bda70f4c24744688d4ec8a931a9076`.
- Round-trip cost: 20.0 bps per admitted candidate, cash return zero.
- Return fields and report numbers are decimal returns (0.01 = 1 percentage point). Gross and net are both retained.
- Four existing PIT families retain q20/q80 past-only calibration with W3 purge after six calendar quarters.
- R6 Agent/simple arms replay their frozen decisions, not their names or narratives; source thresholds and aggregate facts must reproduce.
- All rules freeze before economic evaluation. All input files are read-only and hash checked.

## What This Run Can Decide

1. Whether flagged candidates have worse terminal distributions and higher W3 stop risk.
2. Whether avoided terminal losses exceed foregone terminal gains, including unresolved and ambiguous path classes.
3. Whether cash veto adds stock-specific value beyond exact same-week Matched-N random removal.
4. Whether direction survives ticker overlap control, fees, quarterly splits and removal of the best week.
5. Whether evidence is insufficient, mixed, or historically directionally positive. None is production approval.

## Decision Matrix

| policy | risk_direction | verdict | primary_horizons_positive | primary_horizons_cash_positive |
| --- | --- | --- | --- | --- |
| deep_pullback | DIRECTIONALLY_ELEVATED | MIXED_ECONOMIC_DIRECTION | 2 | 2 |
| deep_pullback_and_extended | DIRECTIONALLY_ELEVATED | HISTORICAL_ECONOMIC_DIRECTION | 3 | 3 |
| extended_vs_candidate | DIRECTIONALLY_ELEVATED | ECONOMIC_VALUE_NOT_DEMONSTRATED | 0 | 0 |
| high_pct_above_ceiling | DIRECTIONALLY_ELEVATED | MIXED_ECONOMIC_DIRECTION | 2 | 1 |
| r6_rdagent | DIRECTIONALLY_ELEVATED | MIXED_ECONOMIC_DIRECTION | 2 | 0 |
| r6_simple | DIRECTIONALLY_ELEVATED | MIXED_ECONOMIC_DIRECTION | 3 | 2 |

Verdicts are descriptive triage, not hypothesis-test passes. HISTORICAL_ECONOMIC_DIRECTION requires all three primary horizons to have positive mean cash delta and positive random-relative increment, at least three supported quarters each, and positive leave-one-quarter-out, leave-one-ticker-out and best-week-removed increments.
DIRECTIONALLY_ELEVATED risk requires at least three supported quarters, positive equal-week mean stop lift and positive stop lift in at least two thirds of supported quarters. The risk-only flag in decision_matrix.csv distinguishes this from consistent economic direction.
INSUFFICIENT_EVIDENCE includes inadequate exposure support. MIXED is not a reason to select a winning horizon.

## Complete Primary Economic Matrix

| policy | panel | horizon | supported_quarters | total_quarters | available_weeks | wholly_unknown_weeks | coverage | baseline_net | veto_cash_net | random_cash_net | incremental_mean | cash_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deep_pullback | all_entries | w1 | 8 | 9 | 104 | 0 | 0.113664 | 0.002048 | 0.002406 | 0.001669 | 0.000738 | 0.000359 |
| deep_pullback | all_entries | w2 | 8 | 9 | 104 | 0 | 0.113664 | 0.008434 | 0.008970 | 0.007667 | 0.001302 | 0.000536 |
| deep_pullback | all_entries | w4 | 8 | 9 | 104 | 0 | 0.113664 | 0.019221 | 0.016166 | 0.016869 | -0.000703 | -0.003055 |
| deep_pullback | nonoverlap_w4 | w1 | 8 | 9 | 104 | 0 | 0.129458 | 0.001849 | 0.002195 | 0.001597 | 0.000598 | 0.000346 |
| deep_pullback | nonoverlap_w4 | w2 | 8 | 9 | 104 | 0 | 0.129458 | 0.011037 | 0.011198 | 0.010179 | 0.001019 | 0.000161 |
| deep_pullback | nonoverlap_w4 | w4 | 8 | 9 | 104 | 0 | 0.129458 | 0.021613 | 0.018802 | 0.019007 | -0.000205 | -0.002811 |
| deep_pullback_and_extended | all_entries | w1 | 8 | 9 | 104 | 0 | 0.028961 | 0.002048 | 0.002619 | 0.002020 | 0.000599 | 0.000571 |
| deep_pullback_and_extended | all_entries | w2 | 8 | 9 | 104 | 0 | 0.028961 | 0.008434 | 0.009037 | 0.008401 | 0.000636 | 0.000603 |
| deep_pullback_and_extended | all_entries | w4 | 8 | 9 | 104 | 0 | 0.028961 | 0.019221 | 0.019937 | 0.018879 | 0.001058 | 0.000716 |
| deep_pullback_and_extended | nonoverlap_w4 | w1 | 8 | 9 | 104 | 0 | 0.035448 | 0.001849 | 0.002664 | 0.001919 | 0.000744 | 0.000815 |
| deep_pullback_and_extended | nonoverlap_w4 | w2 | 8 | 9 | 104 | 0 | 0.035448 | 0.011037 | 0.011789 | 0.011076 | 0.000712 | 0.000752 |
| deep_pullback_and_extended | nonoverlap_w4 | w4 | 8 | 9 | 104 | 0 | 0.035448 | 0.021613 | 0.022602 | 0.021323 | 0.001279 | 0.000989 |
| extended_vs_candidate | all_entries | w1 | 8 | 9 | 104 | 0 | 0.194706 | 0.002048 | 0.001428 | 0.001993 | -0.000565 | -0.000620 |
| extended_vs_candidate | all_entries | w2 | 8 | 9 | 104 | 0 | 0.194706 | 0.008434 | 0.005027 | 0.007448 | -0.002421 | -0.003407 |
| extended_vs_candidate | all_entries | w4 | 8 | 9 | 104 | 0 | 0.194706 | 0.019221 | 0.015395 | 0.016169 | -0.000773 | -0.003826 |
| extended_vs_candidate | nonoverlap_w4 | w1 | 8 | 9 | 104 | 0 | 0.201786 | 0.001849 | 0.001710 | 0.001926 | -0.000216 | -0.000139 |
| extended_vs_candidate | nonoverlap_w4 | w2 | 8 | 9 | 104 | 0 | 0.201786 | 0.011037 | 0.007117 | 0.009739 | -0.002622 | -0.003920 |
| extended_vs_candidate | nonoverlap_w4 | w4 | 8 | 9 | 104 | 0 | 0.201786 | 0.021613 | 0.017635 | 0.018263 | -0.000629 | -0.003978 |
| high_pct_above_ceiling | all_entries | w1 | 8 | 9 | 104 | 0 | 0.248782 | 0.002048 | 0.003039 | 0.001401 | 0.001638 | 0.000991 |
| high_pct_above_ceiling | all_entries | w2 | 8 | 9 | 104 | 0 | 0.248782 | 0.008434 | 0.008595 | 0.007016 | 0.001579 | 0.000161 |
| high_pct_above_ceiling | all_entries | w4 | 8 | 9 | 104 | 0 | 0.248782 | 0.019221 | 0.014002 | 0.015078 | -0.001076 | -0.005219 |
| high_pct_above_ceiling | nonoverlap_w4 | w1 | 8 | 9 | 104 | 0 | 0.254247 | 0.001849 | 0.003193 | 0.001286 | 0.001907 | 0.001344 |
| high_pct_above_ceiling | nonoverlap_w4 | w2 | 8 | 9 | 104 | 0 | 0.254247 | 0.011037 | 0.010574 | 0.008977 | 0.001597 | -0.000463 |
| high_pct_above_ceiling | nonoverlap_w4 | w4 | 8 | 9 | 104 | 0 | 0.254247 | 0.021613 | 0.015811 | 0.016697 | -0.000887 | -0.005802 |
| r6_rdagent | all_entries | w1 | 8 | 9 | 104 | 0 | 0.158489 | 0.002048 | 0.001639 | 0.001528 | 0.000111 | -0.000409 |
| r6_rdagent | all_entries | w2 | 8 | 9 | 104 | 0 | 0.158489 | 0.008434 | 0.007544 | 0.006744 | 0.000800 | -0.000890 |
| r6_rdagent | all_entries | w4 | 8 | 9 | 104 | 0 | 0.158489 | 0.019221 | 0.016130 | 0.015174 | 0.000956 | -0.003091 |
| r6_rdagent | nonoverlap_w4 | w1 | 8 | 9 | 104 | 0 | 0.158994 | 0.001849 | 0.001578 | 0.001645 | -0.000067 | -0.000270 |
| r6_rdagent | nonoverlap_w4 | w2 | 8 | 9 | 104 | 0 | 0.158994 | 0.011037 | 0.009701 | 0.009156 | 0.000544 | -0.001336 |
| r6_rdagent | nonoverlap_w4 | w4 | 8 | 9 | 104 | 0 | 0.158994 | 0.021613 | 0.018024 | 0.017540 | 0.000485 | -0.003588 |
| r6_simple | all_entries | w1 | 8 | 9 | 104 | 0 | 0.133331 | 0.002048 | 0.002178 | 0.001657 | 0.000522 | 0.000130 |
| r6_simple | all_entries | w2 | 8 | 9 | 104 | 0 | 0.133331 | 0.008434 | 0.008986 | 0.007855 | 0.001131 | 0.000553 |
| r6_simple | all_entries | w4 | 8 | 9 | 104 | 0 | 0.133331 | 0.019221 | 0.017266 | 0.016910 | 0.000357 | -0.001955 |
| r6_simple | nonoverlap_w4 | w1 | 8 | 9 | 104 | 0 | 0.143654 | 0.001849 | 0.002196 | 0.001691 | 0.000505 | 0.000347 |
| r6_simple | nonoverlap_w4 | w2 | 8 | 9 | 104 | 0 | 0.143654 | 0.011037 | 0.011703 | 0.010531 | 0.001172 | 0.000666 |
| r6_simple | nonoverlap_w4 | w4 | 8 | 9 | 104 | 0 | 0.143654 | 0.021613 | 0.019500 | 0.019235 | 0.000265 | -0.002113 |

## Loss, Gain and Cost Attribution

| policy | horizon | avoided_gross_loss | foregone_gross_gain | saved_cost | cash_delta_gross | break_even_cost_bps |
| --- | --- | --- | --- | --- | --- | --- |
| deep_pullback | w1 | 0.004426 | 0.004339 | 0.000259 | 0.000087 | -6.724060 |
| deep_pullback | w2 | 0.006145 | 0.006244 | 0.000259 | -0.000098 | 7.575983 |
| deep_pullback | w4 | 0.007075 | 0.010144 | 0.000259 | -0.003070 | 237.131496 |
| deep_pullback_and_extended | w1 | 0.001954 | 0.001210 | 0.000071 | 0.000744 | -209.893258 |
| deep_pullback_and_extended | w2 | 0.002157 | 0.001476 | 0.000071 | 0.000681 | -192.050275 |
| deep_pullback_and_extended | w4 | 0.002631 | 0.001712 | 0.000071 | 0.000918 | -259.089957 |
| extended_vs_candidate | w1 | 0.005705 | 0.006248 | 0.000404 | -0.000542 | 26.876018 |
| extended_vs_candidate | w2 | 0.007150 | 0.011474 | 0.000404 | -0.004323 | 214.259863 |
| extended_vs_candidate | w4 | 0.009394 | 0.013776 | 0.000404 | -0.004382 | 217.142279 |
| high_pct_above_ceiling | w1 | 0.007565 | 0.006729 | 0.000508 | 0.000835 | -32.857590 |
| high_pct_above_ceiling | w2 | 0.009945 | 0.010917 | 0.000508 | -0.000972 | 38.210996 |
| high_pct_above_ceiling | w4 | 0.012134 | 0.018445 | 0.000508 | -0.006311 | 248.210833 |
| r6_rdagent | w1 | 0.003567 | 0.004156 | 0.000318 | -0.000588 | 37.010983 |
| r6_rdagent | w2 | 0.005155 | 0.006810 | 0.000318 | -0.001654 | 104.051462 |
| r6_rdagent | w4 | 0.006248 | 0.010154 | 0.000318 | -0.003906 | 245.696764 |
| r6_simple | w1 | 0.003679 | 0.003619 | 0.000287 | 0.000060 | -4.183937 |
| r6_simple | w2 | 0.005685 | 0.005306 | 0.000287 | 0.000379 | -26.390485 |
| r6_simple | w4 | 0.006298 | 0.008698 | 0.000287 | -0.002400 | 167.098822 |

cash_delta = avoided_gross_loss - foregone_gross_gain + saved_cost. Components have the SAME initial candidate-slot denominator. Break-even cost is algebraic, not a suggested or fitted fee; negative means gross cash delta is already positive.
Costs cancel from incremental_vs_random because both policies remove exactly the same number each week. A cost-only improvement over baseline is not selection alpha.

## Time Stability and Concentration

| policy | horizon | positive_supported_quarters | block_ci_low | block_ci_high | leave_one_quarter_out_min | leave_one_ticker_out_min | without_best_week | without_worst_week |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deep_pullback | w1 | 4 | -0.000758 | 0.002262 | 0.000263 | 0.000416 | 0.000119 | 0.000773 |
| deep_pullback | w2 | 6 | -0.000498 | 0.003049 | 0.000410 | 0.000905 | 0.000693 | 0.001327 |
| deep_pullback | w4 | 5 | -0.002302 | 0.002393 | -0.000421 | -0.000430 | -0.000517 | 0.000655 |
| deep_pullback_and_extended | w1 | 5 | -0.000144 | 0.001951 | 0.000381 | 0.000461 | 0.000362 | 0.000855 |
| deep_pullback_and_extended | w2 | 6 | -0.000087 | 0.001534 | 0.000528 | 0.000558 | 0.000512 | 0.000981 |
| deep_pullback_and_extended | w4 | 8 | 0.000545 | 0.002243 | 0.001192 | 0.001074 | 0.001072 | 0.001599 |
| extended_vs_candidate | w1 | 3 | -0.001845 | 0.002012 | -0.000780 | -0.000725 | -0.000732 | -0.000018 |
| extended_vs_candidate | w2 | 2 | -0.007077 | 0.000866 | -0.003523 | -0.003244 | -0.003275 | -0.000963 |
| extended_vs_candidate | w4 | 2 | -0.003748 | 0.003615 | -0.002219 | -0.001494 | -0.001509 | -0.000203 |
| high_pct_above_ceiling | w1 | 7 | 0.000142 | 0.003634 | 0.001528 | 0.001544 | 0.001550 | 0.002053 |
| high_pct_above_ceiling | w2 | 3 | -0.001426 | 0.004497 | 0.000666 | 0.001176 | 0.000994 | 0.001914 |
| high_pct_above_ceiling | w4 | 2 | -0.004903 | 0.003616 | -0.002843 | -0.001450 | -0.001908 | -0.000113 |
| r6_rdagent | w1 | 3 | -0.000889 | 0.000683 | -0.000251 | -0.000172 | -0.000232 | 0.000035 |
| r6_rdagent | w2 | 6 | -0.000880 | 0.002335 | 0.000185 | 0.000396 | 0.000305 | 0.000779 |
| r6_rdagent | w4 | 3 | -0.001698 | 0.003859 | -0.001214 | -0.000035 | -0.000062 | 0.000839 |
| r6_simple | w1 | 5 | -0.000011 | 0.001418 | 0.000157 | 0.000358 | 0.000320 | 0.000679 |
| r6_simple | w2 | 6 | 0.000117 | 0.002824 | 0.000585 | 0.001060 | 0.000848 | 0.001482 |
| r6_simple | w4 | 4 | -0.001382 | 0.002941 | -0.000605 | -0.000244 | -0.000417 | 0.000602 |

Intervals resample contiguous eight-calendar-week blocks, 2000 draws, seed 42. Missing weeks are not zero returns. Intervals are unavailable below sixteen calendar weeks. These are descriptive uncertainty ranges, not multiplicity-adjusted significance or independent validation; repeat issuers and adaptive research history remain relevant.

## Quarterly Support and Direction

| quarter | policy | status | n | flagged_n | unknown_n | matched_weeks | stop_lift | incremental_vs_random | cash_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024Q2 | deep_pullback | SUPPORTED | 406 | 67 | 200 | 12 | 0.198992 | 0.000292 | -0.002581 |
| 2024Q2 | extended_vs_candidate | SUPPORTED | 406 | 71 | 0 | 11 | 0.023251 | 0.010504 | 0.006739 |
| 2024Q2 | deep_pullback_and_extended | SUPPORTED | 406 | 15 | 200 | 7 | -0.026758 | 0.000849 | 0.000110 |
| 2024Q2 | high_pct_above_ceiling | SUPPORTED | 406 | 149 | 0 | 12 | 0.090107 | 0.012809 | 0.007372 |
| 2024Q3 | deep_pullback | SUPPORTED | 659 | 108 | 365 | 11 | 0.180573 | 0.000940 | -0.000278 |
| 2024Q3 | extended_vs_candidate | SUPPORTED | 659 | 134 | 0 | 13 | 0.074826 | 0.000451 | -0.003174 |
| 2024Q3 | deep_pullback_and_extended | SUPPORTED | 659 | 25 | 365 | 9 | 0.136216 | 0.000527 | 0.000136 |
| 2024Q3 | high_pct_above_ceiling | SUPPORTED | 659 | 159 | 0 | 12 | 0.034459 | -0.001477 | -0.003797 |
| 2024Q4 | deep_pullback | SUPPORTED | 611 | 43 | 341 | 11 | 0.078128 | -0.001467 | -0.002401 |
| 2024Q4 | extended_vs_candidate | SUPPORTED | 611 | 141 | 0 | 12 | 0.116733 | -0.004189 | -0.002775 |
| 2024Q4 | deep_pullback_and_extended | SUPPORTED | 611 | 13 | 341 | 7 | 0.038385 | 0.000955 | 0.001317 |
| 2024Q4 | high_pct_above_ceiling | SUPPORTED | 611 | 159 | 0 | 13 | 0.012351 | -0.003225 | -0.008774 |
| 2025Q1 | deep_pullback | SUPPORTED | 467 | 87 | 223 | 10 | 0.182385 | 0.001309 | 0.006931 |
| 2025Q1 | extended_vs_candidate | SUPPORTED | 467 | 100 | 0 | 12 | 0.029466 | -0.001117 | 0.004965 |
| 2025Q1 | deep_pullback_and_extended | SUPPORTED | 467 | 26 | 223 | 8 | 0.120704 | 0.001498 | 0.004433 |
| 2025Q1 | high_pct_above_ceiling | SUPPORTED | 467 | 150 | 0 | 13 | 0.178652 | 0.002745 | 0.010113 |
| 2025Q2 | deep_pullback | SUPPORTED | 271 | 34 | 183 | 11 | 0.360253 | -0.002533 | -0.011556 |
| 2025Q2 | extended_vs_candidate | SUPPORTED | 271 | 65 | 0 | 11 | 0.097693 | -0.003018 | -0.014091 |
| 2025Q2 | deep_pullback_and_extended | SUPPORTED | 271 | 10 | 183 | 7 | 0.340476 | 0.001666 | -0.000515 |
| 2025Q2 | high_pct_above_ceiling | SUPPORTED | 271 | 42 | 0 | 12 | 0.143359 | -0.006354 | -0.016954 |
| 2025Q3 | deep_pullback | SUPPORTED | 540 | 57 | 322 | 13 | 0.255994 | 0.000543 | -0.001814 |
| 2025Q3 | extended_vs_candidate | SUPPORTED | 540 | 116 | 0 | 13 | 0.177429 | -0.003255 | -0.008365 |
| 2025Q3 | deep_pullback_and_extended | SUPPORTED | 540 | 21 | 322 | 11 | 0.250888 | 0.001070 | 0.000391 |
| 2025Q3 | high_pct_above_ceiling | SUPPORTED | 540 | 95 | 0 | 13 | 0.115544 | -0.004275 | -0.008983 |
| 2025Q4 | deep_pullback | SUPPORTED | 604 | 80 | 330 | 13 | 0.127858 | -0.001095 | -0.005595 |
| 2025Q4 | extended_vs_candidate | SUPPORTED | 604 | 128 | 0 | 13 | 0.209326 | -0.000531 | -0.005766 |
| 2025Q4 | deep_pullback_and_extended | SUPPORTED | 604 | 25 | 330 | 8 | 0.408158 | 0.001887 | 0.000987 |
| 2025Q4 | high_pct_above_ceiling | SUPPORTED | 604 | 153 | 0 | 13 | 0.219571 | -0.001627 | -0.008845 |
| 2026Q1 | deep_pullback | SUPPORTED | 666 | 62 | 385 | 13 | 0.237024 | 0.000371 | -0.005194 |
| 2026Q1 | extended_vs_candidate | SUPPORTED | 666 | 162 | 0 | 13 | 0.084896 | -0.003874 | -0.009357 |
| 2026Q1 | deep_pullback_and_extended | SUPPORTED | 666 | 15 | 385 | 7 | 0.364228 | 0.001779 | 0.001054 |
| 2026Q1 | high_pct_above_ceiling | SUPPORTED | 666 | 158 | 0 | 13 | 0.143712 | -0.005690 | -0.016549 |
| 2026Q2 | deep_pullback | EMPTY_TEST_QUARTER | 0 | 0 | 0 | 0 | N/A | N/A | N/A |
| 2026Q2 | extended_vs_candidate | EMPTY_TEST_QUARTER | 0 | 0 | 0 | 0 | N/A | N/A | N/A |
| 2026Q2 | deep_pullback_and_extended | EMPTY_TEST_QUARTER | 0 | 0 | 0 | 0 | N/A | N/A | N/A |
| 2026Q2 | high_pct_above_ceiling | EMPTY_TEST_QUARTER | 0 | 0 | 0 | 0 | N/A | N/A | N/A |
| 2024Q2 | r6_rdagent | SUPPORTED | 406 | 49 | 200 | 11 | 0.066301 | 0.000510 | 0.000056 |
| 2024Q2 | r6_simple | SUPPORTED | 406 | 64 | 200 | 12 | 0.015865 | -0.003318 | -0.005921 |
| 2024Q3 | r6_rdagent | SUPPORTED | 659 | 169 | 0 | 12 | 0.030117 | -0.001112 | -0.001281 |
| 2024Q3 | r6_simple | SUPPORTED | 659 | 169 | 0 | 12 | 0.030117 | -0.001112 | -0.001281 |
| 2024Q4 | r6_rdagent | SUPPORTED | 611 | 113 | 0 | 13 | -0.018664 | -0.001656 | -0.003948 |
| 2024Q4 | r6_simple | SUPPORTED | 611 | 43 | 341 | 11 | 0.078128 | -0.001467 | -0.002401 |
| 2025Q1 | r6_rdagent | SUPPORTED | 467 | 74 | 223 | 11 | 0.162256 | -0.000932 | 0.000230 |
| 2025Q1 | r6_simple | SUPPORTED | 467 | 87 | 223 | 10 | 0.182385 | 0.001309 | 0.006931 |
| 2025Q2 | r6_rdagent | SUPPORTED | 271 | 63 | 183 | 10 | -0.023889 | 0.012377 | -0.004938 |
| 2025Q2 | r6_simple | SUPPORTED | 271 | 44 | 51 | 11 | 0.006088 | 0.006358 | -0.004166 |
| 2025Q3 | r6_rdagent | SUPPORTED | 540 | 41 | 357 | 11 | 0.379192 | 0.000302 | -0.001420 |
| 2025Q3 | r6_simple | SUPPORTED | 540 | 57 | 322 | 13 | 0.255994 | 0.000543 | -0.001814 |
| 2025Q4 | r6_rdagent | SUPPORTED | 604 | 80 | 330 | 13 | 0.127858 | -0.001095 | -0.005595 |
| 2025Q4 | r6_simple | SUPPORTED | 604 | 80 | 330 | 13 | 0.127858 | -0.001095 | -0.005595 |
| 2026Q1 | r6_rdagent | SUPPORTED | 666 | 74 | 385 | 13 | 0.147351 | -0.004517 | -0.011811 |
| 2026Q1 | r6_simple | SUPPORTED | 666 | 70 | 388 | 12 | -0.045753 | 0.000902 | -0.002658 |
| 2026Q2 | r6_rdagent | EMPTY_TEST_QUARTER | 0 | 0 | 0 | 0 | N/A | N/A | N/A |
| 2026Q2 | r6_simple | EMPTY_TEST_QUARTER | 0 | 0 | 0 | 0 | N/A | N/A | N/A |

The quarterly table above is W4 for compactness; quarterly_summary.csv includes ALL horizons and both panels, including empty quarters. SUPPORTED means at least 10 flagged, 10 known retained and three matched snapshots; it is not a significance claim.

## Interpretation Boundaries

- Every week's baseline gets one equal initial slot per usable entry. Veto leaves removed slots in cash; retained slots never receive extra weight. Random is the exact expectation of uniform removal of the identical N. There is no shrinking or survivor averaging.
- This is conditional on executable entries and upstream maturity filtering, not the full replay universe. Missing/non-executable candidates cannot be recovered from this CSV. The inherited calendar retains empty edge quarters; they are no evidence, not failures.
- The nonoverlap_w4 panel admits each ticker's earliest entry and reserves it through W4 close, identically for ALL policies and horizons. Veto does not free a later admission. This controls duplicate exposure without claiming a live policy simulation.
- Initial slots across weeks are not one funded portfolio. No CAGR, Sharpe, portfolio drawdown, leverage or reinvestment claim is made.
- Unknown features remain visible and unflagged, not declared safe; known-only return and stop contrasts are exported separately. No imputation or EPS refresh occurs.
- Fully unknown-feature weeks remain in operational weekly accounting, but are absent (not zero-alpha) in evidence means, intervals and concentration diagnostics. TEST_FEATURE_UNAVAILABLE is distinct from observed NO_FLAGS; evidence_weeks and observed weeks remain explicit.
- W3 ambiguous labels remain in terminal-return accounting because their closing return is observed. They are excluded ONLY from path-order risk rates. unresolved samples remain throughout.
- Avoided losses are terminal close losses, not assumed -8% fills. Actual gap-aware stop P&L, stopped-path opportunity costs, and policy-dependent capital reuse are NOT_IDENTIFIABLE_FROM_TERMINAL_RETURN_CSV.
- A favorable W4 mark-to-market result can coexist with early stop risk. This run must not justify ignoring stops or changing exits.
- Group outcomes are candidate-weighted descriptive distributions; economic summaries are equal-week means. Neither is an annualized portfolio return.
- No winning family is selected. Even a positive matrix remains a prospective hypothesis; inspected quarters never become untouched OOS again.

## Deliverables

- economic_summary.csv: all 48 policy/panel/horizon cells, fee and stability accounting.
- quarterly_summary.csv / weekly_economics.csv: all denominators, known coverage, risk capture and paired contrasts.
- label_contributions.csv: four exhaustive path classes, including unresolved and ambiguous, with additive loss/gain/fee attribution.
- group_outcomes.csv: per-quarter baseline/flagged/retained/unknown distributions, p10/p90, positive rates and available MAE/MFE facts.
- event_flags.csv / frozen_rules.json: ticker-week membership, shared nonoverlap admission and exact frozen thresholds.
- ticker_concentration.csv: exact leave-one-ticker-out random-relative results without refitting rules, for every panel/horizon. This is a concentration diagnostic, not a ticker blacklist.
- decision_matrix.csv / input_manifest.json / COMPLETE.json: descriptive decisions, provenance and completion hashes.

KEEP PRODUCTION FROZEN
