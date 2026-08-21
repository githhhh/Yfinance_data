# 按周 Signal Oracle 推荐质量评估 - 无 EPS

- 周范围: 2026-01-02 至 2026-08-07；路径收益截至 2026-08-14
- Oracle universe: 每周所有 `signal == True` 且路径收益可计算的标的；winner/loser 在每周内排序，不跨周合并。
- Big winner: 当周 latest_return Top5；Opportunity winner: 当周 max_gain Top5；Big loser: 当周 latest_return Bottom5 或命中 -8% stop。

## Universe 覆盖

- Signal rows: 2500；valid path rows: 1967；weeks: 31

## Variant 总结

| eps_mode   | variant      |   weeks |   picks | median_week_avg_latest_return_pct   | avg_week_avg_latest_return_pct   | median_week_worst_pick_return_pct   | week_latest_top5_hit_rate   | week_gain_top5_hit_rate   | pick_latest_top5_rate   | week_bottom5_hit_rate   | pick_bottom5_rate   | week_stop_rate   | pick_stop_rate   |   score |
|:-----------|:-------------|--------:|--------:|:------------------------------------|:---------------------------------|:------------------------------------|:----------------------------|:--------------------------|:------------------------|:------------------------|:--------------------|:-----------------|:-----------------|--------:|
| no_eps     | v3_core_top3 |      31 |      86 | 12.62%                              | 21.65%                           | 9.48%                               | 25.8%                       | 32.3%                     | 11.6%                   | 32.3%                   | 14.0%               | 38.7%            | 17.4%            |   0.153 |

## Best Variant: `v3_core_top3`

| snapshot_date   |   picks | codes          | avg_latest_return_pct   | worst_latest_return_pct   |   hit_latest_top5_count |   hit_gain_top5_count |   hit_loss_bottom5_count |   stop_8pct_count |
|:----------------|--------:|:---------------|:------------------------|:--------------------------|------------------------:|----------------------:|-------------------------:|------------------:|
| 2026-01-02      |       3 | TSM,BIDU,ASML  | 50.70%                  | 38.82%                    |                       2 |                     2 |                        1 |                 0 |
| 2026-01-09      |       3 | VALE,NU,GH     | 49.71%                  | 49.71%                    |                       0 |                     0 |                        0 |                 1 |
| 2026-01-16      |       3 | BKR,SHEL,FE    | n/a                     | n/a                       |                       0 |                     0 |                        0 |                 0 |
| 2026-01-23      |       3 | MTB,MDT,NU     | 18.91%                  | 18.91%                    |                       1 |                     1 |                        1 |                 1 |
| 2026-01-30      |       3 | HON,AAOI,AAPL  | 18.15%                  | 18.15%                    |                       1 |                     1 |                        1 |                 0 |
| 2026-02-13      |       1 | WULF           | 7.75%                   | 7.75%                     |                       0 |                     1 |                        1 |                 1 |
| 2026-02-20      |       3 | FANG,HEI,UAL   | 14.47%                  | 9.48%                     |                       1 |                     1 |                        2 |                 1 |
| 2026-02-27      |       3 | EQIX,WULF,GMED | 7.75%                   | 7.75%                     |                       0 |                     1 |                        1 |                 1 |
| 2026-03-06      |       2 | FIGS,RTX       | 8.82%                   | 8.82%                     |                       1 |                     1 |                        1 |                 1 |
| 2026-03-13      |       2 | NYAX,DOCN      | 96.08%                  | 96.08%                    |                       1 |                     1 |                        1 |                 0 |
| 2026-03-20      |       1 | LMND           | n/a                     | n/a                       |                       0 |                     0 |                        0 |                 0 |
| 2026-03-27      |       2 | DELL,CASY      | 108.10%                 | 22.67%                    |                       2 |                     2 |                        2 |                 0 |
| 2026-04-02      |       3 | MRVL,PFIS,INVA | 74.33%                  | 32.47%                    |                       1 |                     1 |                        0 |                 0 |
| 2026-04-10      |       3 | SPIR,FRAF,HLIO | 15.57%                  | 15.29%                    |                       0 |                     0 |                        0 |                 0 |
| 2026-04-17      |       3 | AMLX,ALB,CVEO  | 17.92%                  | 12.16%                    |                       0 |                     0 |                        0 |                 1 |
| 2026-04-24      |       3 | MCRI,SANM,CSV  | 9.95%                   | 9.95%                     |                       0 |                     0 |                        0 |                 0 |
| 2026-05-01      |       3 | RBB,NVEC,ACHC  | 22.56%                  | 12.11%                    |                       0 |                     0 |                        0 |                 2 |
| 2026-05-08      |       3 | FTNT,HWBK,KRYS | 21.79%                  | 10.71%                    |                       0 |                     0 |                        0 |                 0 |
| 2026-05-15      |       3 | HLIO,CVS,WES   | 6.75%                   | 2.76%                     |                       0 |                     0 |                        0 |                 0 |
| 2026-05-22      |       3 | CMPR,PDLB,CLBK | -12.41%                 | -40.09%                   |                       0 |                     0 |                        1 |                 1 |
| 2026-05-29      |       3 | BWA,ILMN,FUSB  | 12.62%                  | 3.39%                     |                       0 |                     0 |                        0 |                 0 |
| 2026-06-05      |       3 | TBBB,FCCO,TWIN | 19.69%                  | 10.20%                    |                       0 |                     0 |                        0 |                 0 |
| 2026-06-12      |       3 | MCS,EXPD,CWBC  | 23.64%                  | 11.90%                    |                       0 |                     0 |                        0 |                 0 |
| 2026-06-18      |       3 | AXGN,TER,ADI   | 0.76%                   | -7.78%                    |                       0 |                     0 |                        0 |                 3 |
| 2026-06-26      |       3 | NWFL,BSVN,EGBN | 7.31%                   | 7.22%                     |                       0 |                     0 |                        0 |                 0 |
| 2026-07-02      |       3 | BSET,MPB,KYIV  | 2.79%                   | -4.26%                    |                       0 |                     0 |                        0 |                 1 |
| 2026-07-10      |       3 | BSVN,PHVS,RAPP | 7.57%                   | 2.12%                     |                       0 |                     0 |                        0 |                 1 |
| 2026-07-17      |       3 | WTFC,KARO,EQBK | 5.13%                   | 3.30%                     |                       0 |                     0 |                        0 |                 0 |
| 2026-07-24      |       3 | PKG,PCAR,BLFS  | 8.06%                   | 2.12%                     |                       0 |                     0 |                        0 |                 0 |
| 2026-07-31      |       3 | NBBK,SYBT,SHOO | 2.30%                   | 0.88%                     |                       0 |                     0 |                        0 |                 0 |
| 2026-08-07      |       3 | CFFI,MTUS,SWK  | 0.97%                   | -0.63%                    |                       0 |                     0 |                        0 |                 0 |
