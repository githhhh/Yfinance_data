# Replay Pool Data Source Audit

## 判定规则

- 字段列缺失: 不正常，必须修复。
- 核心价格/结构字段空值: 不正常，必须修复。
- signal 行的 `eps_yoy_growth` 空值: 单独隔离；只有 point-in-time EPS 源可安全补充，当前快照源不得回填。
- signal 行的 IBD candidate / entry 判断字段必须完整；非 signal 行对应空值视为正常。
- `industry` / `sector` 允许用 `Unknown` 作为 repairable fallback，但会单独计数。
- `price_52_week_high` / `dist_to_52w_high_pct` 是价格 as-of 派生字段，必须由已裁剪 daily pkl 重算且不得为空。
- pullback、dryness 等解释增强字段空值计为 optional gap，不阻断 pool 基准使用。

## 总览

- Weeks audited: 58
- Passed weeks including EPS-isolated weeks: 43
- Weeks requiring supplement/repair: 15
- Non-EPS abnormal empty values needing supplement/repair: 0
- Signal EPS gaps isolated pending point-in-time supplement: 2738
- Signal EPS gaps with current snapshot-only source: 2174
- Signal EPS gaps unresolved: 564
- Current snapshot EPS supplement sources are reported separately and are not point-in-time safe.

## 每周审计

| snapshot_date | status | rows | cols | signal | missing_fields | non_eps_abnormal | signal_eps_missing | eps_supp_available | eps_unresolved | repairable_fallback | optional_gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-07-04 | failed | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2025-07-11 | failed | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2025-07-18 | failed | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2025-07-25 | failed | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2025-08-01 | failed | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2025-08-08 | failed | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2025-08-15 | failed | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2025-08-22 | failed | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2025-08-29 | failed | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2025-09-05 | failed | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2025-09-12 | failed | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2025-09-19 | failed | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2025-09-26 | failed | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2025-10-03 | failed | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2025-10-10 | passed_except_eps | 112 | 47 | 12 | 0 | 0 | 12 | 7 | 5 | 104 | 160 |
| 2025-10-17 | failed | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2025-10-24 | passed_except_eps | 120 | 47 | 29 | 0 | 0 | 29 | 17 | 12 | 110 | 197 |
| 2025-10-31 | passed_except_eps | 124 | 47 | 28 | 0 | 0 | 28 | 18 | 10 | 112 | 209 |
| 2025-11-07 | passed_except_eps | 113 | 47 | 15 | 0 | 0 | 15 | 7 | 8 | 102 | 164 |
| 2025-11-14 | passed_except_eps | 103 | 47 | 18 | 0 | 0 | 18 | 10 | 8 | 92 | 136 |
| 2025-11-21 | passed_except_eps | 94 | 47 | 8 | 0 | 0 | 8 | 5 | 3 | 78 | 125 |
| 2025-11-28 | passed_except_eps | 100 | 47 | 36 | 0 | 0 | 36 | 20 | 16 | 82 | 161 |
| 2025-12-05 | passed_except_eps | 101 | 47 | 30 | 0 | 0 | 30 | 18 | 12 | 82 | 174 |
| 2025-12-12 | passed_except_eps | 99 | 47 | 26 | 0 | 0 | 26 | 17 | 9 | 74 | 173 |
| 2025-12-19 | passed_except_eps | 100 | 47 | 19 | 0 | 0 | 19 | 13 | 6 | 74 | 162 |
| 2025-12-26 | passed_except_eps | 94 | 47 | 17 | 0 | 0 | 17 | 14 | 3 | 64 | 135 |
| 2026-01-02 | passed_except_eps | 83 | 47 | 15 | 0 | 0 | 15 | 12 | 3 | 48 | 115 |
| 2026-01-09 | passed_except_eps | 91 | 47 | 40 | 0 | 0 | 40 | 26 | 14 | 60 | 131 |
| 2026-01-16 | passed_except_eps | 94 | 47 | 21 | 0 | 0 | 21 | 13 | 8 | 66 | 147 |
| 2026-01-23 | passed_except_eps | 100 | 47 | 14 | 0 | 0 | 14 | 6 | 8 | 70 | 165 |
| 2026-01-30 | passed_except_eps | 100 | 47 | 17 | 0 | 0 | 17 | 14 | 3 | 70 | 146 |
| 2026-02-06 | passed | 0 | 47 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2026-02-13 | passed_except_eps | 119 | 47 | 27 | 0 | 0 | 27 | 15 | 12 | 78 | 199 |
| 2026-02-20 | passed_except_eps | 119 | 47 | 18 | 0 | 0 | 18 | 11 | 7 | 80 | 175 |
| 2026-02-27 | passed_except_eps | 111 | 47 | 27 | 0 | 0 | 27 | 14 | 13 | 74 | 151 |
| 2026-03-06 | passed_except_eps | 106 | 47 | 9 | 0 | 0 | 9 | 6 | 3 | 64 | 138 |
| 2026-03-13 | passed_except_eps | 102 | 47 | 10 | 0 | 0 | 10 | 3 | 7 | 68 | 133 |
| 2026-03-20 | passed_except_eps | 90 | 47 | 8 | 0 | 0 | 8 | 4 | 4 | 62 | 114 |
| 2026-03-27 | passed_except_eps | 92 | 47 | 10 | 0 | 0 | 10 | 8 | 2 | 62 | 128 |
| 2026-04-02 | passed_except_eps | 141 | 47 | 45 | 0 | 0 | 45 | 33 | 12 | 80 | 234 |
| 2026-04-10 | passed_except_eps | 148 | 47 | 69 | 0 | 0 | 69 | 49 | 20 | 78 | 250 |
| 2026-04-17 | passed_except_eps | 254 | 47 | 96 | 0 | 0 | 96 | 75 | 21 | 96 | 550 |
| 2026-04-24 | passed_except_eps | 380 | 47 | 80 | 0 | 0 | 80 | 58 | 22 | 138 | 816 |
| 2026-05-01 | passed_except_eps | 451 | 47 | 105 | 0 | 0 | 105 | 80 | 25 | 152 | 823 |
| 2026-05-08 | passed_except_eps | 513 | 47 | 128 | 0 | 0 | 128 | 101 | 27 | 152 | 948 |
| 2026-05-15 | passed_except_eps | 515 | 47 | 66 | 0 | 0 | 66 | 46 | 20 | 158 | 823 |
| 2026-05-22 | passed_except_eps | 554 | 47 | 87 | 0 | 0 | 87 | 74 | 13 | 162 | 886 |
| 2026-05-29 | passed_except_eps | 570 | 47 | 112 | 0 | 0 | 112 | 88 | 24 | 166 | 911 |
| 2026-06-05 | passed_except_eps | 594 | 47 | 108 | 0 | 0 | 108 | 92 | 16 | 158 | 895 |
| 2026-06-12 | passed_except_eps | 674 | 47 | 268 | 0 | 0 | 268 | 240 | 28 | 158 | 1138 |
| 2026-06-18 | passed_except_eps | 670 | 47 | 145 | 0 | 0 | 145 | 116 | 29 | 150 | 985 |
| 2026-06-26 | passed_except_eps | 731 | 47 | 211 | 0 | 0 | 211 | 178 | 33 | 146 | 1162 |
| 2026-07-02 | passed_except_eps | 760 | 47 | 120 | 0 | 0 | 120 | 100 | 20 | 144 | 1197 |
| 2026-07-10 | passed_except_eps | 716 | 47 | 83 | 0 | 0 | 83 | 66 | 17 | 122 | 960 |
| 2026-07-17 | passed_except_eps | 754 | 47 | 174 | 0 | 0 | 174 | 155 | 19 | 112 | 990 |
| 2026-07-24 | passed_except_eps | 745 | 47 | 106 | 0 | 0 | 106 | 94 | 12 | 86 | 977 |
| 2026-07-31 | passed_except_eps | 742 | 47 | 114 | 0 | 0 | 114 | 102 | 12 | 56 | 928 |
| 2026-08-07 | passed_except_eps | 776 | 47 | 167 | 0 | 0 | 167 | 149 | 18 | 52 | 1030 |

## 每周明细

### 2025-07-04

- 状态: `failed`
- 需要补充/修复: -
- 正常空值: -
- repairable fallback: -
- optional gap: -
- signal EPS 缺失代码: -
- signal EPS 本地补源覆盖: 0; unresolved: 0

### 2025-07-11

- 状态: `failed`
- 需要补充/修复: -
- 正常空值: -
- repairable fallback: -
- optional gap: -
- signal EPS 缺失代码: -
- signal EPS 本地补源覆盖: 0; unresolved: 0

### 2025-07-18

- 状态: `failed`
- 需要补充/修复: -
- 正常空值: -
- repairable fallback: -
- optional gap: -
- signal EPS 缺失代码: -
- signal EPS 本地补源覆盖: 0; unresolved: 0

### 2025-07-25

- 状态: `failed`
- 需要补充/修复: -
- 正常空值: -
- repairable fallback: -
- optional gap: -
- signal EPS 缺失代码: -
- signal EPS 本地补源覆盖: 0; unresolved: 0

### 2025-08-01

- 状态: `failed`
- 需要补充/修复: -
- 正常空值: -
- repairable fallback: -
- optional gap: -
- signal EPS 缺失代码: -
- signal EPS 本地补源覆盖: 0; unresolved: 0

### 2025-08-08

- 状态: `failed`
- 需要补充/修复: -
- 正常空值: -
- repairable fallback: -
- optional gap: -
- signal EPS 缺失代码: -
- signal EPS 本地补源覆盖: 0; unresolved: 0

### 2025-08-15

- 状态: `failed`
- 需要补充/修复: -
- 正常空值: -
- repairable fallback: -
- optional gap: -
- signal EPS 缺失代码: -
- signal EPS 本地补源覆盖: 0; unresolved: 0

### 2025-08-22

- 状态: `failed`
- 需要补充/修复: -
- 正常空值: -
- repairable fallback: -
- optional gap: -
- signal EPS 缺失代码: -
- signal EPS 本地补源覆盖: 0; unresolved: 0

### 2025-08-29

- 状态: `failed`
- 需要补充/修复: -
- 正常空值: -
- repairable fallback: -
- optional gap: -
- signal EPS 缺失代码: -
- signal EPS 本地补源覆盖: 0; unresolved: 0

### 2025-09-05

- 状态: `failed`
- 需要补充/修复: -
- 正常空值: -
- repairable fallback: -
- optional gap: -
- signal EPS 缺失代码: -
- signal EPS 本地补源覆盖: 0; unresolved: 0

### 2025-09-12

- 状态: `failed`
- 需要补充/修复: -
- 正常空值: -
- repairable fallback: -
- optional gap: -
- signal EPS 缺失代码: -
- signal EPS 本地补源覆盖: 0; unresolved: 0

### 2025-09-19

- 状态: `failed`
- 需要补充/修复: -
- 正常空值: -
- repairable fallback: -
- optional gap: -
- signal EPS 缺失代码: -
- signal EPS 本地补源覆盖: 0; unresolved: 0

### 2025-09-26

- 状态: `failed`
- 需要补充/修复: -
- 正常空值: -
- repairable fallback: -
- optional gap: -
- signal EPS 缺失代码: -
- signal EPS 本地补源覆盖: 0; unresolved: 0

### 2025-10-03

- 状态: `failed`
- 需要补充/修复: -
- 正常空值: -
- repairable fallback: -
- optional gap: -
- signal EPS 缺失代码: -
- signal EPS 本地补源覆盖: 0; unresolved: 0

### 2025-10-10

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=12
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=107; ibd_entry_breakout_range_ratio_invalid_or_non_signal=105; ibd_entry_close_position_invalid_or_non_signal=105; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=105; ibd_entry_date_invalid_or_non_signal=105; ibd_entry_price_invalid_or_non_signal=105; ibd_entry_rule_invalid_or_non_signal=105; ibd_entry_volume_ratio_invalid_or_non_signal=105; ...+12
- repairable fallback: industry=52; sector=52
- optional gap: ibd_candidate_extra=104; pullback_v_is_dry=20; pullback_duration_weeks=12; pullback_pct=12; pullback_pct_off_peak=12
- signal EPS 缺失代码: ALM;ANET;ASTS;CORZ;GH;GSK;MP;OKLO;RIOT;SANM;TEM;VRT
- signal EPS 本地补源覆盖: 7; unresolved: 5

### 2025-10-17

- 状态: `failed`
- 需要补充/修复: -
- 正常空值: -
- repairable fallback: -
- optional gap: -
- signal EPS 缺失代码: -
- signal EPS 本地补源覆盖: 0; unresolved: 0

### 2025-10-24

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=29
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=114; ibd_entry_close_position_invalid_or_non_signal=114; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=114; ibd_entry_date_invalid_or_non_signal=114; ibd_entry_price_invalid_or_non_signal=114; ibd_entry_rule_invalid_or_non_signal=114; ibd_entry_volume_ratio_invalid_or_non_signal=114; ibd_trigger_price_invalid_or_non_signal=114; ...+12
- repairable fallback: industry=55; sector=55
- optional gap: ibd_candidate_extra=104; pullback_v_is_dry=27; pullback_duration_weeks=22; pullback_pct=22; pullback_pct_off_peak=22
- signal EPS 缺失代码: AAPL;AVGO;BKR;CDNS;COF;CRWD;FTAI;GH;GOOG;HWM;IBM;KLAC;MMM;NRG;NTRA;NU;NVDA;OPEN;PANW;RIOT;ROST;RTX;SEDG;SHEL;SHOP;VEEV;WELL;WFC;ZS
- signal EPS 本地补源覆盖: 17; unresolved: 12

### 2025-10-31

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=28
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=112; ibd_entry_breakout_range_ratio_invalid_or_non_signal=108; ibd_entry_close_position_invalid_or_non_signal=108; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=108; ibd_entry_date_invalid_or_non_signal=108; ibd_entry_price_invalid_or_non_signal=108; ibd_entry_rule_invalid_or_non_signal=108; ibd_entry_volume_ratio_invalid_or_non_signal=108; ...+12
- repairable fallback: industry=56; sector=56
- optional gap: ibd_candidate_extra=105; pullback_v_is_dry=32; pullback_duration_weeks=24; pullback_pct=24; pullback_pct_off_peak=24
- signal EPS 缺失代码: AMZN;C;CAH;CHRW;COE;CORZ;CPNG;CRDO;FUTU;GSK;HOOD;LKNCY;LSCC;MS;MTSR;NBIS;NET;NVDA;PLTR;ROKU;SOFI;STX;TSLA;VTR;W;WDC;WFC;WULF
- signal EPS 本地补源覆盖: 18; unresolved: 10

### 2025-11-07

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=15
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=106; ibd_entry_close_position_invalid_or_non_signal=106; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=106; ibd_entry_date_invalid_or_non_signal=106; ibd_entry_price_invalid_or_non_signal=106; ibd_entry_rule_invalid_or_non_signal=106; ibd_entry_volume_ratio_invalid_or_non_signal=106; ibd_trigger_price_invalid_or_non_signal=106; ...+12
- repairable fallback: industry=51; sector=51
- optional gap: ibd_candidate_extra=102; pullback_v_is_dry=23; pullback_duration_weeks=13; pullback_pct=13; pullback_pct_off_peak=13
- signal EPS 缺失代码: ARGX;CASY;COLL;EXPE;FIGS;GH;JPM;LYFT;NRG;SANM;SEDG;STX;VEEV;VTR;XYL
- signal EPS 本地补源覆盖: 7; unresolved: 8

### 2025-11-14

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=18
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=98; ibd_entry_close_position_invalid_or_non_signal=98; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=98; ibd_entry_date_invalid_or_non_signal=98; ibd_entry_price_invalid_or_non_signal=98; ibd_entry_rule_invalid_or_non_signal=98; ibd_entry_volume_ratio_invalid_or_non_signal=98; ibd_trigger_price_invalid_or_non_signal=98; ...+12
- repairable fallback: industry=46; sector=46
- optional gap: ibd_candidate_extra=90; pullback_v_is_dry=16; pullback_duration_weeks=10; pullback_pct=10; pullback_pct_off_peak=10
- signal EPS 缺失代码: ABBV;ALL;AU;AZN;CASY;COLL;EA;FE;GSK;JNJ;LH;LLY;LYFT;MDT;MNST;MS;NTRA;SHEL
- signal EPS 本地补源覆盖: 10; unresolved: 8

### 2025-11-21

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=8
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=90; ibd_entry_close_position_invalid_or_non_signal=90; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=90; ibd_entry_date_invalid_or_non_signal=90; ibd_entry_price_invalid_or_non_signal=90; ibd_entry_reject_reason_valid_or_non_signal=90; ibd_entry_rule_invalid_or_non_signal=90; ibd_entry_volume_ratio_invalid_or_non_signal=90; ...+12
- repairable fallback: industry=39; sector=39
- optional gap: ibd_candidate_extra=87; pullback_v_is_dry=11; pullback_duration_weeks=9; pullback_pct=9; pullback_pct_off_peak=9
- signal EPS 缺失代码: ABBV;BTSG;FE;GOOG;MDT;MNPR;ROST;WMT
- signal EPS 本地补源覆盖: 5; unresolved: 3

### 2025-11-28

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=36
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=94; ibd_entry_close_position_invalid_or_non_signal=94; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=94; ibd_entry_date_invalid_or_non_signal=94; ibd_entry_price_invalid_or_non_signal=94; ibd_entry_rule_invalid_or_non_signal=94; ibd_entry_volume_ratio_invalid_or_non_signal=94; ibd_trigger_price_invalid_or_non_signal=94; ...+12
- repairable fallback: industry=41; sector=41
- optional gap: ibd_candidate_extra=76; pullback_v_is_dry=22; pullback_duration_weeks=21; pullback_pct=21; pullback_pct_off_peak=21
- signal EPS 缺失代码: AAPL;APP;AU;AVGO;BIDU;BKR;BTSG;C;CASY;CEG;CHRW;CIEN;COHR;CRDO;EA;FE;HLT;IONQ;LYFT;MNST;MS;NRG;NU;OUST;ROST;SHOP;SOFI;STX;TTWO;UHS;VCYT;W;WES;WFC;WMT;WWD
- signal EPS 本地补源覆盖: 20; unresolved: 16

### 2025-12-05

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=30
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=97; ibd_entry_close_position_invalid_or_non_signal=97; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=97; ibd_entry_date_invalid_or_non_signal=97; ibd_entry_price_invalid_or_non_signal=97; ibd_entry_rule_invalid_or_non_signal=97; ibd_entry_volume_ratio_invalid_or_non_signal=97; ibd_trigger_price_invalid_or_non_signal=97; ...+12
- repairable fallback: industry=41; sector=41
- optional gap: ibd_candidate_extra=83; pullback_v_is_dry=28; pullback_duration_weeks=21; pullback_pct=21; pullback_pct_off_peak=21
- signal EPS 缺失代码: AMAT;APP;ASML;ASTS;BIDU;BLK;C;CAT;COF;CVNA;DASH;EA;EWBC;FN;JOBY;JPM;LSCC;LYFT;MS;PATH;SHOP;SMTC;TSLA;TSM;TTWO;VRT;WDC;WES;WFC;WWD
- signal EPS 本地补源覆盖: 18; unresolved: 12

### 2025-12-12

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=26
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=96; ibd_entry_close_position_invalid_or_non_signal=96; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=96; ibd_entry_date_invalid_or_non_signal=96; ibd_entry_price_invalid_or_non_signal=96; ibd_entry_rule_invalid_or_non_signal=96; ibd_entry_volume_ratio_invalid_or_non_signal=96; ibd_trigger_price_invalid_or_non_signal=96; ...+12
- repairable fallback: industry=37; sector=37
- optional gap: ibd_candidate_extra=86; pullback_v_is_dry=27; pullback_duration_weeks=20; pullback_pct=20; pullback_pct_off_peak=20
- signal EPS 缺失代码: AEO;ASTS;BLK;CIEN;COF;EXPE;FEIM;HLT;HWM;IBM;JPM;LH;NUTX;PAC;PFG;ROKU;ROST;RTX;STX;TKO;TPR;TSLA;UBS;URBN;ZION;ZM
- signal EPS 本地补源覆盖: 17; unresolved: 9

### 2025-12-19

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=19
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=97; ibd_entry_breakout_range_ratio_invalid_or_non_signal=84; ibd_entry_close_position_invalid_or_non_signal=84; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=84; ibd_entry_date_invalid_or_non_signal=84; ibd_entry_price_invalid_or_non_signal=84; ibd_entry_rule_invalid_or_non_signal=84; ibd_entry_volume_ratio_invalid_or_non_signal=84; ...+12
- repairable fallback: industry=37; sector=37
- optional gap: ibd_candidate_extra=87; pullback_v_is_dry=24; pullback_duration_weeks=17; pullback_pct=17; pullback_pct_off_peak=17
- signal EPS 缺失代码: ALM;CEG;CHRW;EXPE;HLT;HWM;LLY;MNST;MU;PAC;PLTR;RKLB;RTX;SCHW;SHOP;TKO;TSLA;TTWO;UAL
- signal EPS 本地补源覆盖: 13; unresolved: 6

### 2025-12-26

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=17
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=91; ibd_entry_close_position_invalid_or_non_signal=91; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=91; ibd_entry_date_invalid_or_non_signal=91; ibd_entry_price_invalid_or_non_signal=91; ibd_entry_rule_invalid_or_non_signal=91; ibd_entry_volume_ratio_invalid_or_non_signal=91; ibd_trigger_price_invalid_or_non_signal=91; ...+12
- repairable fallback: industry=32; sector=32
- optional gap: ibd_candidate_extra=79; pullback_v_is_dry=17; pullback_duration_weeks=13; pullback_pct=13; pullback_pct_off_peak=13
- signal EPS 缺失代码: ALM;AU;CAH;CASY;CIEN;FEIM;HEI;HUT;HWM;IBKR;JPM;NU;NVDA;SCHW;SMTC;SNDK;TSM
- signal EPS 本地补源覆盖: 14; unresolved: 3

### 2026-01-02

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=15
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=79; ibd_entry_close_position_invalid_or_non_signal=79; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=79; ibd_entry_date_invalid_or_non_signal=79; ibd_entry_price_invalid_or_non_signal=79; ibd_entry_rule_invalid_or_non_signal=79; ibd_entry_volume_ratio_invalid_or_non_signal=79; ibd_trigger_price_invalid_or_non_signal=79; ...+12
- repairable fallback: industry=24; sector=24
- optional gap: ibd_candidate_extra=75; pullback_duration_weeks=10; pullback_pct=10; pullback_pct_off_peak=10; pullback_v_is_dry=10
- signal EPS 缺失代码: ASML;BIDU;BKR;BTSG;FTAI;GEV;HEI;HWM;LSCC;NYAX;RKLB;SHEL;TSM;WES;ZION
- signal EPS 本地补源覆盖: 12; unresolved: 3

### 2026-01-09

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=40
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=80; ibd_entry_close_position_invalid_or_non_signal=80; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=80; ibd_entry_date_invalid_or_non_signal=80; ibd_entry_price_invalid_or_non_signal=80; ibd_entry_rule_invalid_or_non_signal=80; ibd_entry_volume_ratio_invalid_or_non_signal=80; ibd_trigger_price_invalid_or_non_signal=80; ...+12
- repairable fallback: industry=30; sector=30
- optional gap: ibd_candidate_extra=63; pullback_duration_weeks=17; pullback_pct=17; pullback_pct_off_peak=17; pullback_v_is_dry=17
- signal EPS 缺失代码: AMAT;AMZN;ASML;AU;AZN;BTU;CASY;CAT;CHRW;CVNA;EXPE;FCX;FTAI;GH;GMED;GOOG;GSK;HEI;HLT;HUT;HWM;IBM;LDOS;LGND;LSCC;NTRA;NU;NUTX;ROST;SMTC;SNDK;SOLV;STX;TROW;UAL;URBN;VALE;W;WDC;WES
- signal EPS 本地补源覆盖: 26; unresolved: 14

### 2026-01-16

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=21
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=85; ibd_entry_close_position_invalid_or_non_signal=85; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=85; ibd_entry_date_invalid_or_non_signal=85; ibd_entry_price_invalid_or_non_signal=85; ibd_entry_rule_invalid_or_non_signal=85; ibd_entry_volume_ratio_invalid_or_non_signal=85; ibd_trigger_price_invalid_or_non_signal=85; ...+12
- repairable fallback: industry=33; sector=33
- optional gap: ibd_candidate_extra=80; pullback_v_is_dry=19; pullback_duration_weeks=16; pullback_pct=16; pullback_pct_off_peak=16
- signal EPS 缺失代码: ASTS;BIDU;BKR;BTU;CAH;FE;FN;IBKR;JNJ;MNST;MRNA;RTX;SANM;SCHW;SHEL;SN;STX;TKO;WELL;WMT;WULF
- signal EPS 本地补源覆盖: 13; unresolved: 8

### 2026-01-23

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=14
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=93; ibd_entry_close_position_invalid_or_non_signal=93; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=93; ibd_entry_date_invalid_or_non_signal=93; ibd_entry_price_invalid_or_non_signal=93; ibd_entry_reject_reason_valid_or_non_signal=93; ibd_entry_rule_invalid_or_non_signal=93; ibd_entry_volume_ratio_invalid_or_non_signal=93; ...+12
- repairable fallback: industry=35; sector=35
- optional gap: ibd_candidate_extra=91; pullback_v_is_dry=23; pullback_duration_weeks=17; pullback_pct=17; pullback_pct_off_peak=17
- signal EPS 缺失代码: ALM;BIDU;BILI;BKR;COHR;CVNA;GH;KO;LMND;MDT;MTB;NU;PAC;ROKU
- signal EPS 本地补源覆盖: 6; unresolved: 8

### 2026-01-30

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=17
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=93; ibd_entry_breakout_range_ratio_invalid_or_non_signal=90; ibd_entry_close_position_invalid_or_non_signal=90; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=90; ibd_entry_date_invalid_or_non_signal=90; ibd_entry_price_invalid_or_non_signal=90; ibd_entry_rule_invalid_or_non_signal=90; ibd_entry_volume_ratio_invalid_or_non_signal=90; ...+12
- repairable fallback: industry=35; sector=35
- optional gap: ibd_candidate_extra=88; pullback_v_is_dry=22; pullback_duration_weeks=12; pullback_pct=12; pullback_pct_off_peak=12
- signal EPS 缺失代码: AAOI;AAPL;ALM;CAH;CIEN;GOOG;GSK;HON;IBM;JNJ;KO;LMT;MDT;PFG;SCHW;SHEL;ZM
- signal EPS 本地补源覆盖: 14; unresolved: 3

### 2026-02-06

- 状态: `passed`
- 需要补充/修复: -
- 正常空值: -
- repairable fallback: -
- optional gap: -
- signal EPS 缺失代码: -
- signal EPS 本地补源覆盖: 0; unresolved: 0

### 2026-02-13

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=27
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=107; ibd_entry_close_position_invalid_or_non_signal=107; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=107; ibd_entry_date_invalid_or_non_signal=107; ibd_entry_price_invalid_or_non_signal=107; ibd_entry_rule_invalid_or_non_signal=107; ibd_entry_volume_ratio_invalid_or_non_signal=107; ibd_trigger_price_invalid_or_non_signal=107; ...+12
- repairable fallback: industry=39; sector=39
- optional gap: ibd_candidate_extra=103; pullback_v_is_dry=30; pullback_duration_weeks=22; pullback_pct=22; pullback_pct_off_peak=22
- signal EPS 缺失代码: ADC;AEIS;AEO;AMAT;AU;AZN;DG;FE;GNRC;HWM;LSCC;MAS;MTZ;NATL;NEM;NYAX;OR;PFIS;ROST;SHEL;SN;TSM;VRT;WELL;WES;WLK;WULF
- signal EPS 本地补源覆盖: 15; unresolved: 12

### 2026-02-20

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=18
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=112; ibd_entry_close_position_invalid_or_non_signal=112; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=112; ibd_entry_date_invalid_or_non_signal=112; ibd_entry_price_invalid_or_non_signal=112; ibd_entry_rule_invalid_or_non_signal=112; ibd_entry_volume_ratio_invalid_or_non_signal=112; ibd_trigger_price_invalid_or_non_signal=112; ...+12
- repairable fallback: industry=40; sector=40
- optional gap: ibd_candidate_extra=106; pullback_v_is_dry=24; pullback_duration_weeks=15; pullback_pct=15; pullback_pct_off_peak=15
- signal EPS 缺失代码: AAOI;AS;AXTI;CDE;COHR;FANG;FN;HEI;HWM;MRNA;NRG;PAC;RTX;SHEL;TTMI;UAL;UBS;XPO
- signal EPS 本地补源覆盖: 11; unresolved: 7

### 2026-02-27

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=27
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=99; ibd_entry_breakout_range_ratio_invalid_or_non_signal=96; ibd_entry_close_position_invalid_or_non_signal=96; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=96; ibd_entry_date_invalid_or_non_signal=96; ibd_entry_price_invalid_or_non_signal=96; ibd_entry_rule_invalid_or_non_signal=96; ibd_entry_volume_ratio_invalid_or_non_signal=96; ...+12
- repairable fallback: industry=37; sector=37
- optional gap: ibd_candidate_extra=91; pullback_v_is_dry=18; pullback_duration_weeks=14; pullback_pct=14; pullback_pct_off_peak=14
- signal EPS 缺失代码: AMGN;APEI;AU;AZN;BMY;BTSG;CASY;CDE;DG;EQIX;EXPE;FCX;FIGS;GMED;JNJ;KO;LH;NATL;NEM;OR;PAAS;PHG;TIGO;TKO;VICR;VVX;WULF
- signal EPS 本地补源覆盖: 14; unresolved: 13

### 2026-03-06

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=9
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=104; ibd_entry_breakout_range_ratio_invalid_or_non_signal=99; ibd_entry_close_position_invalid_or_non_signal=99; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=99; ibd_entry_date_invalid_or_non_signal=99; ibd_entry_price_invalid_or_non_signal=99; ibd_entry_rule_invalid_or_non_signal=99; ibd_entry_volume_ratio_invalid_or_non_signal=99; ...+12
- repairable fallback: industry=32; sector=32
- optional gap: ibd_candidate_extra=101; pullback_v_is_dry=13; pullback_duration_weeks=8; pullback_pct=8; pullback_pct_off_peak=8
- signal EPS 缺失代码: CF;EXPE;FIGS;LMT;MPC;NYAX;PBF;RTX;VVX
- signal EPS 本地补源覆盖: 6; unresolved: 3

### 2026-03-13

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=10
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=99; ibd_entry_close_position_invalid_or_non_signal=99; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=99; ibd_entry_date_invalid_or_non_signal=99; ibd_entry_price_invalid_or_non_signal=99; ibd_entry_rule_invalid_or_non_signal=99; ibd_entry_volume_ratio_invalid_or_non_signal=99; ibd_trigger_price_invalid_or_non_signal=99; ...+12
- repairable fallback: industry=34; sector=34
- optional gap: ibd_candidate_extra=95; pullback_v_is_dry=11; pullback_duration_weeks=9; pullback_pct=9; pullback_pct_off_peak=9
- signal EPS 缺失代码: APEI;BTU;DOCN;EQIX;KR;MU;NFG;NYAX;SNDK;WLK
- signal EPS 本地补源覆盖: 3; unresolved: 7

### 2026-03-20

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=8
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=87; ibd_entry_breakout_range_ratio_invalid_or_non_signal=85; ibd_entry_close_position_invalid_or_non_signal=85; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=85; ibd_entry_date_invalid_or_non_signal=85; ibd_entry_price_invalid_or_non_signal=85; ibd_entry_rule_invalid_or_non_signal=85; ibd_entry_volume_ratio_invalid_or_non_signal=85; ...+12
- repairable fallback: industry=31; sector=31
- optional gap: ibd_candidate_extra=84; pullback_v_is_dry=9; pullback_duration_weeks=7; pullback_pct=7; pullback_pct_off_peak=7
- signal EPS 缺失代码: BTU;CIEN;DOCN;LMND;MTZ;SEDG;STX;WDC
- signal EPS 本地补源覆盖: 4; unresolved: 4

### 2026-03-27

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=10
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=88; ibd_entry_close_position_invalid_or_non_signal=88; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=88; ibd_entry_date_invalid_or_non_signal=88; ibd_entry_price_invalid_or_non_signal=88; ibd_entry_rule_invalid_or_non_signal=88; ibd_entry_volume_ratio_invalid_or_non_signal=88; ibd_trigger_price_invalid_or_non_signal=88; ...+12
- repairable fallback: industry=31; sector=31
- optional gap: ibd_candidate_extra=87; pullback_v_is_dry=11; pullback_duration_weeks=10; pullback_pct=10; pullback_pct_off_peak=10
- signal EPS 缺失代码: BKR;CASY;DELL;EQIX;FCX;GEV;NEM;NFG;PFIS;TIGO
- signal EPS 本地补源覆盖: 8; unresolved: 2

### 2026-04-02

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=45
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=131; ibd_entry_close_position_invalid_or_non_signal=131; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=131; ibd_entry_date_invalid_or_non_signal=131; ibd_entry_price_invalid_or_non_signal=131; ibd_entry_rule_invalid_or_non_signal=131; ibd_entry_volume_ratio_invalid_or_non_signal=131; ibd_trigger_price_invalid_or_non_signal=131; ...+12
- repairable fallback: industry=40; sector=40
- optional gap: ibd_candidate_extra=114; pullback_v_is_dry=36; pullback_duration_weeks=28; pullback_pct=28; pullback_pct_off_peak=28
- signal EPS 缺失代码: AA;AD;ADC;AROW;BBT;BMY;BTSG;BVFL;CLST;D;ECO;ELA;EQIX;FBNC;FCX;FN;FRO;GEV;HG;HON;INSW;INVA;LITE;MRVL;MSGS;NVST;NVT;PAHC;PFG;PFIS;PL;ROST;RSI;SATS;SMTC;SSRM;STX;TCBI;TDW;TKO;TMP;UI;VIK;VVX;XPO
- signal EPS 本地补源覆盖: 33; unresolved: 12

### 2026-04-10

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=69
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=128; ibd_entry_close_position_invalid_or_non_signal=128; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=128; ibd_entry_date_invalid_or_non_signal=128; ibd_entry_price_invalid_or_non_signal=128; ibd_entry_rule_invalid_or_non_signal=128; ibd_entry_volume_ratio_invalid_or_non_signal=128; ibd_trigger_price_invalid_or_non_signal=128; ...+12
- repairable fallback: industry=39; sector=39
- optional gap: ibd_candidate_extra=96; pullback_v_is_dry=40; pullback_duration_weeks=38; pullback_pct=38; pullback_pct_off_peak=38
- signal EPS 缺失代码: AA;AAOI;ACLS;ADC;AEHR;AEIS;AMAT;AROW;ASML;AU;BBT;BVFL;BWFG;C;CAT;CMC;COHR;CTRI;D;DAN;DBD;EWBC;FBNC;FE;FN;FRAF;GEF;GEV;GLW;GNRC;GSK;HLIO;HLT;HON;HUT;INTC;INVA;IVT;KEYS;LSCC;MKSI;MRVL;MTB;MYE;NBIS;NHC;NVT;OPY;PAHC;PFG;PFIS;SMTC;SNDK;SPIR;SSRM;STX;TCBI;TER;TTMI;UI;VALE;VRT;VTR;WDC;WMT;WULF;WWD;XPO;ZION
- signal EPS 本地补源覆盖: 49; unresolved: 20

### 2026-04-17

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=96
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=228; ibd_entry_close_position_invalid_or_non_signal=228; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=228; ibd_entry_date_invalid_or_non_signal=228; ibd_entry_price_invalid_or_non_signal=228; ibd_entry_rule_invalid_or_non_signal=228; ibd_entry_volume_ratio_invalid_or_non_signal=228; ibd_trigger_price_invalid_or_non_signal=228; ...+12
- repairable fallback: industry=48; sector=48
- optional gap: ibd_candidate_extra=182; pullback_v_is_dry=95; pullback_duration_weeks=91; pullback_pct=91; pullback_pct_off_peak=91
- signal EPS 缺失代码: ACLX;AD;ADI;ALB;ALL;ALNT;AMAL;AMKR;AMLX;ANRO;APEI;ARMK;AROW;AXTI;BFH;BMY;BOKF;BTSG;BURL;CAC;CAMT;CASH;CASS;CATY;CMI;CNA;CPF;CRUS;CSWC;CSX;CVEO;DG;DLX;EBAY;ENLT;ESEA;ESP;EZPW;FBP;FCFS;FCX;FDX;FIGS;FLEX;FRO;GATX;GCT;GHM;GNRC;GRMN;GSAT;HUT;IBKR;INVA;IVT;JAZZ;JBHT;LGND;LIND;MAR;MKSI;MSGS;MSM;MU;NATL;NBIS;NYAX;OPLN;OR;OSIS;OVBC;OVLY;PL;RPRX;RRBI;RSI;SATS;SHEN;SILC;SLAB;SMTC;STT;SXI;TPR;TRVI;TXN;ULS;UTMD;UVE;VALE;VICR;VIK;WCC;WELL;WULF;XPO
- signal EPS 本地补源覆盖: 75; unresolved: 21

### 2026-04-24

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=80
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=350; ibd_entry_breakout_range_ratio_invalid_or_non_signal=330; ibd_entry_close_position_invalid_or_non_signal=330; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=330; ibd_entry_date_invalid_or_non_signal=330; ibd_entry_price_invalid_or_non_signal=330; ibd_entry_rule_invalid_or_non_signal=330; ibd_entry_volume_ratio_invalid_or_non_signal=330; ...+12
- repairable fallback: industry=69; sector=69
- optional gap: ibd_candidate_extra=336; pullback_v_is_dry=159; pullback_duration_weeks=107; pullback_pct=107; pullback_pct_off_peak=107
- signal EPS 缺失代码: ALSN;AMPX;ANET;ARM;ASYS;AZZ;BATRA;BKR;CAMT;CHMG;COHU;CSV;CVGW;CVLG;DGII;DHIL;DOCN;ECO;ENTG;ERAS;ETN;EWBC;FAF;FPS;GL;GOOG;HCSG;HXL;IESC;INBX;KALU;LASR;LBRT;LPTH;LSBK;MCHP;MCRI;MSBI;NCSM;NEE;NHC;NPO;NSC;NUE;NVTS;NYAX;OBK;PAC;PFG;POET;PRA;PUMP;Q;R;REPX;RMBS;RNGR;RSI;SANM;SCI;SEDG;SEI;SLB;SRBK;STLD;STRL;TDW;TRT;TSM;TWST;TXN;UNP;UVE;VECO;VMI;VPG;VSH;WLAC;WSFS;WTTR
- signal EPS 本地补源覆盖: 58; unresolved: 22

### 2026-05-01

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=105
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=404; ibd_entry_breakout_range_ratio_invalid_or_non_signal=393; ibd_entry_close_position_invalid_or_non_signal=393; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=393; ibd_entry_date_invalid_or_non_signal=393; ibd_entry_price_invalid_or_non_signal=393; ibd_entry_rule_invalid_or_non_signal=393; ibd_entry_volume_ratio_invalid_or_non_signal=393; ...+12
- repairable fallback: industry=76; sector=76
- optional gap: ibd_candidate_extra=381; pullback_v_is_dry=142; pullback_duration_weeks=100; pullback_pct=100; pullback_pct_off_peak=100
- signal EPS 缺失代码: AAPL;ACHC;AEE;AEP;AIT;AKR;ALL;ALRS;ANAB;AROW;ATLO;AXSM;BAND;BEN;BOH;BWFG;CARE;CATY;CBOE;CE;CHEF;CNC;CNOB;COCO;CTO;DINO;DTM;EBAY;ECG;ECO;ESEA;ESOA;ESP;ET;ETON;ETR;FANG;FDX;FRO;FVR;GNRC;GPRE;GTX;GVA;HAL;HR;INDV;INSW;IRM;JAZZ;JCI;KALV;KEN;KNSA;KO;KRG;KRP;LAMR;LNT;MDU;MO;MPC;MRAM;MSBI;MSGE;MSGS;MTX;MUSA;MYFW;NHC;NPKI;NVEC;NVGS;OSW;PCB;PFIS;PHVS;PR;RBB;REPX;REX;RGCO;ROKU;RS;SBSI;SHIP;SIF;SNDA;STNG;SUN;TEN;THR;TPC;TRMK;TVTX;TWLO;VCTR;VLO;VMI;VTR;WCC;WELL;WERN;WILC;XOMA
- signal EPS 本地补源覆盖: 80; unresolved: 25

### 2026-05-08

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=128
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=468; ibd_entry_breakout_range_ratio_invalid_or_non_signal=430; ibd_entry_close_position_invalid_or_non_signal=430; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=430; ibd_entry_date_invalid_or_non_signal=430; ibd_entry_price_invalid_or_non_signal=430; ibd_entry_rule_invalid_or_non_signal=430; ibd_entry_volume_ratio_invalid_or_non_signal=430; ...+12
- repairable fallback: industry=76; sector=76
- optional gap: ibd_candidate_extra=433; pullback_v_is_dry=146; pullback_duration_weeks=123; pullback_pct=123; pullback_pct_off_peak=123
- signal EPS 缺失代码: AAON;AAPL;AD;AKAM;AKR;AMAT;AMN;APLD;ASML;ASRT;ASYS;AXTI;BELFA;BKSY;BLBD;BTSG;BVFL;CALY;CCSI;CECO;CELC;CGNX;CHEF;CNO;COCO;CON;CORZ;CPF;CTS;CYTK;CZWI;DCOM;DVA;EBAY;EGP;ESCA;ESEA;FEIM;FFIV;FISI;FRAF;FRO;FSEA;FSTR;FTNT;GLW;GTX;GWW;HCSG;HFBL;HR;HST;HWBK;HXL;HZO;IBKR;IRM;JAZZ;JLHL;KNSA;KRYS;LIFE;LINC;LIND;LIVN;LRCX;MANU;MATX;MCFT;MCRI;MDV;MEC;MITK;MKSI;MLI;MNST;MOD;MPTI;MRNA;MRX;MXL;NBIS;NHC;NMM;NPO;NVDA;NVEC;NWPX;OCS;ODC;OPLN;ORN;OSBC;OSS;PBT;PCB;PDFS;PEBK;PFIS;PL;PRA;QUIK;RBC;RFIL;RIOT;RKLB;ROAD;ROK;RPRX;SMBC;SMTC;SNDA;SNEX;SSRM;SYRE;TKR;TRS;TSAT;TSN;TWLO;UFCS;ULS;VTRS;VVX;WFCF;WLFC;WSR;WT
- signal EPS 本地补源覆盖: 101; unresolved: 27

### 2026-05-15

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=66
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=488; ibd_entry_breakout_range_ratio_invalid_or_non_signal=476; ibd_entry_close_position_invalid_or_non_signal=476; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=476; ibd_entry_date_invalid_or_non_signal=476; ibd_entry_price_invalid_or_non_signal=476; ibd_entry_rule_invalid_or_non_signal=476; ibd_entry_volume_ratio_invalid_or_non_signal=476; ...+12
- repairable fallback: industry=79; sector=79
- optional gap: ibd_candidate_extra=463; pullback_v_is_dry=132; pullback_duration_weeks=76; pullback_pct=76; pullback_pct_off_peak=76
- signal EPS 缺失代码: AIZ;AKAM;ALKS;ALL;ARMK;BUUU;BW;CAEP;CAPL;COHR;CON;CVEO;CVLG;CVS;DCO;DVA;DY;ELMD;ETON;FCX;FTNT;GL;HG;HLIO;JBHT;KNX;KRYS;LITE;LQDA;MO;MPC;MSGS;NGS;NPO;NXT;ORA;PAA;PL;PLGO;PM;PNRG;POET;PRM;PUMP;RKLB;SATS;SDRL;SEDG;STRZ;SUN;TFSL;TH;TRGP;TRT;TSEM;UNP;USAC;UTI;VAL;VIK;VIRT;VLGEA;WBI;WCC;WES;WMB
- signal EPS 本地补源覆盖: 46; unresolved: 20

### 2026-05-22

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=87
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=525; ibd_entry_close_position_invalid_or_non_signal=525; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=525; ibd_entry_date_invalid_or_non_signal=525; ibd_entry_price_invalid_or_non_signal=525; ibd_entry_rule_invalid_or_non_signal=525; ibd_entry_volume_ratio_invalid_or_non_signal=525; ibd_trigger_price_invalid_or_non_signal=525; ...+12
- repairable fallback: industry=81; sector=81
- optional gap: ibd_candidate_extra=505; pullback_v_is_dry=114; pullback_duration_weeks=89; pullback_pct=89; pullback_pct_off_peak=89
- signal EPS 缺失代码: ACMR;ACNB;AKR;ALAB;ALGM;ALRS;ARCB;ARM;ATRO;BE;BK;BKR;BKSY;BNL;BNY;BSRR;BURL;CEVA;CLBK;CMP;CMPR;CPBI;CRDO;CTBI;CTS;CYD;DAL;DCOM;ELMD;ENPH;FDX;FEIM;FISI;FRO;FSBC;GL;GS;HLT;IVT;KLAC;KRG;KRP;LIN;LIVN;LPG;LPTH;LQDA;LXFR;MANU;MCY;MOV;NMM;NOV;NTAP;NXPI;OBK;OSS;OUST;OVBC;PCB;PDLB;PFG;PFIS;PKBK;PLPC;PSMT;QCOM;R;RBCAA;ROIV;ROST;RXO;SBRA;SIRI;SKYT;SLB;SNDR;STBA;TFSL;TIGO;TMP;TRS;UNM;VOYG;VRSN;WBI;XHR
- signal EPS 本地补源覆盖: 74; unresolved: 13

### 2026-05-29

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=112
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=521; ibd_entry_breakout_range_ratio_invalid_or_non_signal=507; ibd_entry_close_position_invalid_or_non_signal=507; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=507; ibd_entry_date_invalid_or_non_signal=507; ibd_entry_price_invalid_or_non_signal=507; ibd_entry_rule_invalid_or_non_signal=507; ibd_entry_volume_ratio_invalid_or_non_signal=507; ...+12
- repairable fallback: industry=83; sector=83
- optional gap: ibd_candidate_extra=486; pullback_v_is_dry=131; pullback_duration_weeks=98; pullback_pct=98; pullback_pct_off_peak=98
- signal EPS 缺失代码: AA;ALGM;ALM;ALNT;AMAT;ARCB;ARM;ASTC;ATI;ATLC;AVGO;BFH;BJRI;BUSE;BUUU;BWA;CATY;CBL;CC;CENX;CLFD;CNOB;COHU;CRDO;CSTM;CTS;CURB;DAL;DCOM;DIOD;DY;ELA;ENLT;FDX;FEIM;FISI;FLEX;FUSB;GH;GHM;H;HLIO;HLT;ILMN;INGM;ITRN;JBHT;KLAC;KRG;LEA;LIND;LLY;LRCX;LSCC;LSTR;MAR;MEC;MITK;MKSI;MNST;MOD;MOG-A;MOV;MPTI;MRCY;MSCI;MSGE;NBIX;NTB;NVEC;NVRI;NWPX;NXT;NYAX;PDFS;PDLB;PFIS;PKOH;PL;PRAX;RHP;ROKU;RS;RVMD;SANM;SBLK;SEDG;SHBI;SHEN;SILA;SNDK;SNX;SPIR;SXI;THFF;TKR;TMP;TVTX;TWST;UBCP;UE;UMAC;VELO;VICR;VIK;VLGEA;VVX;WDC;WEYS;WLKP;WULF;XHR
- signal EPS 本地补源覆盖: 88; unresolved: 24

### 2026-06-05

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=108
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=556; ibd_entry_close_position_invalid_or_non_signal=556; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=556; ibd_entry_date_invalid_or_non_signal=556; ibd_entry_price_invalid_or_non_signal=556; ibd_entry_rule_invalid_or_non_signal=556; ibd_entry_volume_ratio_invalid_or_non_signal=556; ibd_trigger_price_invalid_or_non_signal=556; ...+12
- repairable fallback: industry=79; sector=79
- optional gap: ibd_candidate_extra=515; pullback_v_is_dry=125; pullback_duration_weeks=85; pullback_pct=85; pullback_pct_off_peak=85
- signal EPS 缺失代码: AFL;AIT;AKR;ALL;ALNT;AMG;AMN;APLE;AROW;ATLO;AVA;AVNS;AXGN;BFH;BHB;BNL;BPOP;BSRR;BWFG;C;CATY;CBK;CDP;CNC;CNOB;CPF;CSX;DINO;DOC;DOCN;DTM;EFSI;FBP;FCCO;FCF;FMBH;FRST;FSBC;FSEA;FTK;FVR;FXNC;GABC;GL;GRDN;GSBC;GWW;HNGE;IESC;IGIC;IVT;JLHL;KRP;LGND;LIN;MAC;MAR;MCBS;MCRI;MEC;MPC;MRX;MSBI;MYE;NBIX;NTB;NTRS;OBK;ODC;ODFL;OSBC;OVLY;PBFS;PCB;PEBK;PFIS;PGC;PLBC;PRA;PTRN;R;RBC;SMBC;SPG;SPHR;SRCE;STBA;STT;SXI;TBBB;TBN;TMP;TNK;TNL;TWIN;TWLO;UNFI;UNP;UTI;VLO;VMI;VOYA;VSXY;WABC;WLKP;WSBF;XOMA;XPO
- signal EPS 本地补源覆盖: 92; unresolved: 16

### 2026-06-12

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=268
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=570; ibd_entry_close_position_invalid_or_non_signal=570; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=570; ibd_entry_date_invalid_or_non_signal=570; ibd_entry_price_invalid_or_non_signal=570; ibd_entry_rule_invalid_or_non_signal=570; ibd_entry_volume_ratio_invalid_or_non_signal=570; ibd_trigger_price_invalid_or_non_signal=570; ...+12
- repairable fallback: industry=79; sector=79
- optional gap: ibd_candidate_extra=478; pullback_v_is_dry=171; pullback_duration_weeks=163; pullback_pct=163; pullback_pct_off_peak=163
- signal EPS 缺失代码: ABCB;ACLS;ACNB;AD;AEIS;AIP;AIR;AIT;AIZ;ALGM;ALL;AMAL;AMBQ;AMG;AMKR;AMRX;ARMK;AROW;ASB;ASH;ASML;ASRT;ASYS;ATLC;AVBH;AXSM;AZZ;BAP;BBT;BCML;BDL;BHB;BHE;BJRI;BLX;BNL;BOTJ;BPOP;BPRN;BRX;BSRR;BURL;BUSE;BY;C;CAC;CAKE;CALY;CASY;CBAN;CBFV;CCNE;CDP;CECO;CHEF;CLBK;CNOB;COCO;COHU;CON;COSO;CPF;CRDO;CSTM;CSX;CTBI;CTO;CTRN;CVS;CW;CWBC;CXW;CZWI;DAVE;DBD;DCO;DCOM;DOC;DVA;DXPE;EFSC;EGP;ELA;ELVN;ENVA;ESE;ESI;ETON;EWBC;EXEL;EXPD;FAC;FBIZ;FBP;FCBC;FCCO;FCF;FDSB;FFBC;FFIV;FMBH;FNLC;FNRN;FORM;FR;FRAF;FRST;FULT;FUNC;FVCB;FVR;FXNC;GABC;GRC;GRDN;HBCP;HBNC;HBT;HFWA;HLT;HMN;HR;HWBK;IBKR;IBOC;ICHR;IDA;IFS;INCY;IVT;JBL;KALU;KIM;KLAC;KLIC;KO;KRG;KRYS;LARK;LAUR;LIN;LIND;LIVN;LPG;LQDA;LYTS;M;MAC;MATX;MCS;MET;MITK;MOV;MRX;MTX;MUSA;MYE;MYFW;NEE;NHC;NMM;NNN;NPO;NSA;NTB;NTST;NUVL;NVMI;NWFL;OBK;OCC;OFG;OII;ONTO;OSBC;OSW;OVBC;OVLY;PCB;PDFS;PEB;PEBK;PEBO;PECO;PHIN;PKBK;PKE;PKOH;PLBC;PLGO;PLOW;PLPC;PLXS;PRM;PRSU;PSMT;PSTL;PTGX;QCRH;RBB;RBC;RBKB;RFIL;RL;RNST;ROIV;ROK;ROKU;ROST;RSI;SCHL;SENEA;SEPN;SFNC;SHBI;SI;SIF;SILC;SITM;SKT;SLAB;SMBC;SMBK;SMTC;SN;SNDK;SNEX;SPHR;SRBK;SRCE;STBA;STNG;SYRE;TBBB;TCBK;TGTX;THFF;TIGO;TJX;TMP;TNGX;TNK;TNL;TRMK;TRNO;TSM;TTMI;TVTX;UCTT;UFCS;UMBF;UVE;UVSP;VECO;VELO;VIRT;WBI;WBS;WEYS;WKC;WPC;WSBF;WSFS;WSM;WTS;WULF;WWD;ZION
- signal EPS 本地补源覆盖: 240; unresolved: 28

### 2026-06-18

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=145
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=618; ibd_entry_close_position_invalid_or_non_signal=618; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=618; ibd_entry_date_invalid_or_non_signal=618; ibd_entry_price_invalid_or_non_signal=618; ibd_entry_rule_invalid_or_non_signal=618; ibd_entry_volume_ratio_invalid_or_non_signal=618; ibd_trigger_price_invalid_or_non_signal=618; ...+12
- repairable fallback: industry=75; sector=75
- optional gap: ibd_candidate_extra=546; pullback_v_is_dry=166; pullback_duration_weeks=91; pullback_pct=91; pullback_pct_off_peak=91
- signal EPS 缺失代码: ACA;ACLS;ADI;AEHR;AEIS;AFBI;AIP;AIR;ALGM;ALOT;AMD;AMKR;AOSL;ARWR;ASH;ASX;AVT;AXGN;AZZ;BAP;BCAX;BE;BEN;BHE;BJRI;BLBD;BPRN;BVFL;CALY;CAT;CBNA;CCBG;CHRN;CIFR;CLDX;CMI;CR;CTRN;CW;D;DAL;DAVE;DIOD;DNLI;DXPE;ECBK;EIG;ENTG;ESE;ESI;ESTA;FCEL;FDSB;FLXS;FORM;FRO;FSTR;FTDR;FUSB;GE;GEV;GFS;GHM;GL;GLW;GNRC;GOLF;GTX;GVA;HFBL;HQ;HTB;ICHR;IFS;IMAX;INSW;INTC;IRM;ITT;KEYS;KLIC;KNSA;LION;LOCO;LSCC;LTH;LYTS;MCHP;MCY;MKSI;MOD;MOV;MRNA;MRX;MTSI;MYRG;NBIS;NPKI;NVRI;NVT;NXPI;ONTO;OPY;ORKA;ORN;OSS;OUST;PANW;PBFS;PTGX;QCOM;RBC;RGCO;RLAY;RMIX;ROIV;ROK;RVMD;SEI;SHAZ;SLAB;SN;SPHR;SSRM;STNG;SWBI;TER;TGTX;TH;TRS;TRV;TRVI;TSEM;TSM;TTMI;TXN;URGN;VCYT;VIK;VSXY;WAB;WSR;WULF;WWD;WYFI
- signal EPS 本地补源覆盖: 116; unresolved: 29

### 2026-06-26

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=211
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=686; ibd_entry_breakout_range_ratio_invalid_or_non_signal=565; ibd_entry_close_position_invalid_or_non_signal=565; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=565; ibd_entry_date_invalid_or_non_signal=565; ibd_entry_price_invalid_or_non_signal=565; ibd_entry_rule_invalid_or_non_signal=565; ibd_entry_volume_ratio_invalid_or_non_signal=565; ...+12
- repairable fallback: industry=73; sector=73
- optional gap: ibd_candidate_extra=591; pullback_v_is_dry=160; pullback_duration_weeks=137; pullback_pct=137; pullback_pct_off_peak=137
- signal EPS 缺失代码: AAL;ABCB;ACA;ACU;ADPT;AEE;AEP;AFL;AGX;AKTS;ALL;AMAL;AMN;AMTB;APGE;AROC;AROW;ASB;ASND;ATLO;AUPH;AVBP;BATRA;BCAL;BCAX;BCML;BFS;BFST;BIOA;BNL;BRX;BSVN;BUUU;BVFL;BWB;BWFG;CAC;CAH;CBFV;CBK;CBL;CBU;CCBG;CFR;CGEM;CHCO;CHMG;CINF;CLBK;CNP;CPBI;CPF;CSX;CTBI;CUBI;CVS;CWBC;CYTK;CZFS;CZWI;DFTX;DGII;DNTH;DOC;DTM;EA;EBMT;EFSC;EG;EGBN;EGP;EIG;ELMD;ELVN;EQBK;ERAS;ETON;EVRG;FBIZ;FBNC;FCBC;FCCO;FNLC;FRAF;FRME;FTNT;FULT;FXNC;GLW;HAFC;HBCP;HBNC;HCSG;HFWA;HG;HGV;HWBK;HZO;IBOC;IDA;ILMN;INTG;IRM;JMSB;KGS;KNSA;KRT;KYMR;LAMR;LBRX;LCNB;LINC;LLY;LNT;LQDT;MAC;MAMA;MANE;MBWM;MBX;MCB;MCBS;MCY;MIRM;MITK;MNSB;MO;MPC;MPLT;MSGE;MSGS;MYFW;NESR;NGS;NGVT;NHC;NIC;NJR;NTCT;NTRA;NTST;NVCT;NWFL;OII;OPLN;ORRF;OSBC;OUT;PECO;PFBC;PGC;PHVS;PINE;PKBK;PLGO;PLOW;PLPC;PLSE;PNW;PSTL;PTRN;QCRH;RBB;RCUS;RFIL;RIOT;RMBI;RPRX;SBFG;SBSI;SENEA;SFNC;SI;SMBK;SNDA;SPG;SRBK;SRRK;STEL;SWX;TCBK;THG;TMP;TRMK;TRV;TSBK;UAL;UBSI;UDR;UE;UFCS;UMBF;UNP;UNTY;USCB;UTMD;UVE;VABK;VLO;VTR;WELL;WHG;WMB;WSBC;WSBF;WSFS;WSR;WTBA;WTTR;XENE;ZION
- signal EPS 本地补源覆盖: 178; unresolved: 33

### 2026-07-02

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=120
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=723; ibd_entry_close_position_invalid_or_non_signal=723; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=723; ibd_entry_date_invalid_or_non_signal=723; ibd_entry_price_invalid_or_non_signal=723; ibd_entry_rule_invalid_or_non_signal=723; ibd_entry_volume_ratio_invalid_or_non_signal=723; ibd_trigger_price_invalid_or_non_signal=723; ...+12
- repairable fallback: industry=72; sector=72
- optional gap: ibd_candidate_extra=668; pullback_v_is_dry=178; pullback_duration_weeks=117; pullback_pct=117; pullback_pct_off_peak=117
- signal EPS 缺失代码: AAPL;ABBV;ACHC;AFL;AHR;ANAB;ARWR;ASIC;ASX;AWR;AXGN;AXS;BFC;BLZE;BMRC;BNY;BSET;CB;CBNA;CCXI;CDP;CLDX;CNS;CRL;CRWD;CSX;CW;DDOG;DGX;DINO;DK;EBAY;ECO;EG;EXEL;EZPW;FCEL;FFIV;FLYW;FROG;GAIN;GH;HCSG;HFWA;IDA;IGIC;INCY;JAZZ;JMSB;JNJ;KFRC;KO;KYIV;LAMR;LGND;LIN;LLYVA;LYV;MANU;MATX;MBWM;MD;MIRM;MPB;MPC;MPLT;MRCY;MUSA;MVBF;NBIX;NBTB;NIC;NREF;NTRA;NTRS;NTST;NUTX;NUVL;NVRI;NYAX;OFG;OOMA;OUST;PACS;PKE;PNTG;PRI;SBFG;SENEA;SIF;SIRI;SLDE;SNDA;SPNT;SRBK;SUN;SUNC;TCBK;TRIN;TROW;UDR;UHT;UNP;USFD;UTI;UVE;VLO;VOYA;VRTX;VSEC;VTR;WELL;WERN;WEYS;WFCF;WILC;WLY;XENE;XOMA;XZO
- signal EPS 本地补源覆盖: 100; unresolved: 20

### 2026-07-10

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=83
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=695; ibd_entry_close_position_invalid_or_non_signal=695; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=695; ibd_entry_date_invalid_or_non_signal=695; ibd_entry_price_invalid_or_non_signal=695; ibd_entry_rule_invalid_or_non_signal=695; ibd_entry_volume_ratio_invalid_or_non_signal=695; ibd_trigger_price_invalid_or_non_signal=695; ...+12
- repairable fallback: industry=61; sector=61
- optional gap: ibd_candidate_extra=654; pullback_v_is_dry=120; pullback_duration_weeks=62; pullback_pct=62; pullback_pct_off_peak=62
- signal EPS 缺失代码: AAMI;AAPL;ADM;AFBI;AMG;ANDG;ANET;BAND;BBIO;BSVN;BZH;CLMT;CNXN;DELL;DINO;DNTH;EA;EBAY;ECO;ECPG;EWTX;EXPD;FHI;FR;FTH;FTK;GPRE;GWW;HPE;ICHR;IGIC;INSW;LASR;LINC;LOB;MET;MNPR;MOD;MPC;MRX;MS;MUSA;NESR;NET;NJR;NMM;NREF;NSC;NTCT;OII;ONTO;PAA;PAG;PAGP;PBF;PBI;PENG;PFG;PHVS;PNTG;PSX;RAPP;RDWR;RGA;SAH;SBLK;SGHC;SION;SLAB;SRRK;STNG;SYRE;TEN;TIGO;TRAX;TRNO;TWLO;VCTR;VIRT;WBI;WCC;WT;WYFI
- signal EPS 本地补源覆盖: 66; unresolved: 17

### 2026-07-17

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=174
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=683; ibd_entry_close_position_invalid_or_non_signal=683; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=683; ibd_entry_date_invalid_or_non_signal=683; ibd_entry_price_invalid_or_non_signal=683; ibd_entry_rule_invalid_or_non_signal=683; ibd_entry_volume_ratio_invalid_or_non_signal=683; ibd_trigger_price_invalid_or_non_signal=683; ...+12
- repairable fallback: industry=56; sector=56
- optional gap: ibd_candidate_extra=619; pullback_v_is_dry=107; pullback_duration_weeks=88; pullback_pct=88; pullback_pct_off_peak=88
- signal EPS 缺失代码: ABCB;ACHC;ACT;ADM;AFL;AHR;AIT;AKR;ALRS;ALX;AMAL;ARCB;ASB;ASH;ATLO;BATRA;BIIB;BOH;BOTJ;BRX;BSRR;BUSE;BWFG;BY;CALY;CBL;CCK;CDNA;CFR;CHCO;CHMG;CHRN;CHRW;CLBK;CNO;CNS;CNXN;COAG;CRNX;CTBI;CTO;CTRE;CTRN;CUBE;CURB;CVEO;CVLG;CVS;CZFS;DINO;DKL;DNTH;EGP;EIX;EPR;EQBK;ESQ;EWBC;FA;FAF;FBIZ;FBP;FBRX;FCF;FEIM;FLYW;FMBH;FR;FRME;FRT;FULT;FVCB;GABC;GEF;GHRS;GL;GPRE;GTY;HBNC;HFWA;HIW;HR;HST;HTO;HXL;IOR;IRM;ITIC;IVT;JMSB;JXN;KARO;KELYA;KFY;KIM;KRG;LAMR;LCNB;LOB;LQDT;LTC;MATX;MDV;MO;MPC;MSBI;MSGS;MTB;NBTB;NET;NHC;NTB;NWBI;OBK;OHI;ONB;ORKA;OSBC;OZK;PAG;PARR;PEBO;PECO;PFG;PFIS;PINE;PKOH;PLBC;PLD;PLSE;PM;PSX;QCRH;RDN;REG;RGCO;RHI;RLYB;RNR;RUSHA;RXO;SAH;SEIC;SION;SKT;SMBC;SMBK;SNDR;SNEX;SPFI;SPG;SRBK;STAG;SUN;SUNC;TCBK;TRGP;TRMK;TRVI;UBSI;UE;UMBF;UNF;UNTY;VLGEA;VTRS;WABC;WERN;WSFS;WT;WTFC;WTTR;XHR;XMTR
- signal EPS 本地补源覆盖: 155; unresolved: 19

### 2026-07-24

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=106
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=703; ibd_entry_close_position_invalid_or_non_signal=703; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=703; ibd_entry_date_invalid_or_non_signal=703; ibd_entry_price_invalid_or_non_signal=703; ibd_entry_rule_invalid_or_non_signal=703; ibd_entry_volume_ratio_invalid_or_non_signal=703; ibd_trigger_price_invalid_or_non_signal=703; ...+12
- repairable fallback: industry=43; sector=43
- optional gap: ibd_candidate_extra=665; pullback_v_is_dry=111; pullback_duration_weeks=67; pullback_pct=67; pullback_pct_off_peak=67
- signal EPS 缺失代码: ACNB;ACU;AIT;ALL;ALOT;AMRX;ARWR;ATI;AVBC;AVT;BHB;BIP;BLFS;BPRN;CARE;CB;CBAN;CCBG;CCNE;CDP;CHMG;CR;CRS;CTRE;CZFS;CZNC;DAC;DBD;DELL;DGX;ECO;EDRY;EPR;ESEA;FBLA;FCBC;FEIM;FISI;FLXS;FXNC;GEF;GSL;HALO;HBCP;HG;HST;HWM;IBCP;IMAX;INSW;IOR;IRM;JAZZ;JNJ;JPM;LARK;LBRX;LH;LPG;LQDA;MMM;MOG-A;MRK;MSBI;NMM;NVEC;NWFL;OFG;OGE;OHI;ORRF;OVV;PAA;PAGP;PBFS;PBT;PCAR;PKG;PLSE;QCRH;RBCAA;RHP;RNR;ROST;RS;SAFT;SBLK;SBRA;SFNC;SHBI;SMBK;SON;SPG;STBA;SXT;THG;THRM;TMP;TRST;URI;USCB;VIK;VVX;WAB;WES;WRLD
- signal EPS 本地补源覆盖: 94; unresolved: 12

### 2026-07-31

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=114
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=688; ibd_entry_breakout_range_ratio_invalid_or_non_signal=682; ibd_entry_close_position_invalid_or_non_signal=682; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=682; ibd_entry_date_invalid_or_non_signal=682; ibd_entry_price_invalid_or_non_signal=682; ibd_entry_rule_invalid_or_non_signal=682; ibd_entry_volume_ratio_invalid_or_non_signal=682; ...+12
- repairable fallback: industry=28; sector=28
- optional gap: ibd_candidate_extra=650; pullback_v_is_dry=89; pullback_duration_weeks=63; pullback_pct=63; pullback_pct_off_peak=63
- signal EPS 缺失代码: AAMI;ACT;ACU;AMLX;AMTB;ANDG;ARXS;ASH;ATLC;AVBC;BFC;BFH;BHB;BIP;BPOP;BPRN;BWFG;CAC;CASY;CBNA;CBNK;CCNE;CHEF;CIVB;CLDX;CNK;CTRN;CZFS;CZNC;DAC;DCOM;EAT;ECBK;ECPG;ENVA;ESEA;ESQ;ETON;FCCO;FISI;FRBA;FSEA;FXNC;GABC;GH;GIII;GKOS;GRMN;GSAT;HBNC;HCSG;HTB;IBCP;IMAX;KNSA;KO;LH;LPG;LTH;MANU;MBIN;MBWM;MCHB;MCS;MMM;MPB;MSBI;MTX;MVBF;MYE;NBBK;NET;NEU;NMIH;NTAP;NUE;NWFL;OBK;ODC;OSBC;PFGC;PKOH;PLSE;PRK;PROV;PSMT;QCRH;RACC;RAPP;RCKY;RDVT;RNG;ROST;SAIH;SBLK;SHBI;SHIP;SHOO;SION;SNOW;SNX;SYBT;THFF;THG;TOWN;TRAX;TRMK;TXRH;USCB;UVE;VSXY;VTRS;WCC;WTBA
- signal EPS 本地补源覆盖: 102; unresolved: 12

### 2026-08-07

- 状态: `passed_except_eps`
- 需要补充/修复: eps_yoy_growth_signal=167
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=707; ibd_entry_close_position_invalid_or_non_signal=707; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=707; ibd_entry_date_invalid_or_non_signal=707; ibd_entry_price_invalid_or_non_signal=707; ibd_entry_rule_invalid_or_non_signal=707; ibd_entry_volume_ratio_invalid_or_non_signal=707; ibd_trigger_price_invalid_or_non_signal=707; ...+12
- repairable fallback: industry=26; sector=26
- optional gap: ibd_candidate_extra=649; pullback_v_is_dry=111; pullback_duration_weeks=90; pullback_pct=90; pullback_pct_off_peak=90
- signal EPS 缺失代码: ABNB;ACT;ADPT;AIR;AIRT;AIT;AIZ;ALH;ALNT;ALX;AME;AMGN;AMKR;AMN;ANET;ASH;ATEX;ATI;ATLO;ATRO;AVT;AXGN;BAC;BDL;BLZE;BOKF;BSET;BUUU;BVS;CATY;CBAN;CCXI;CFFI;CFG;CGEM;CIX;CLDX;CON;COSO;CPAY;CPBI;CRL;CRNX;CSWC;CUBI;DAL;DBX;DCO;DELL;DGII;DGX;DIOD;DXPE;EDRY;ELMD;ESCA;ESNT;ETN;EXPD;EXPE;FA;FAST;FBLA;FBNC;FET;FHI;FLXS;FNLC;FRD;FROG;FRST;FTDR;FTK;GD;GE;GFF;GRMN;HBB;HBCP;HEI;HNGE;HNVR;HPE;HVT;HWBK;IBTA;INCY;IOSP;IVZ;JCI;JHX;JXN;KALU;KEYS;LIN;LIND;LQDT;LYV;MBWM;MD;MET;MNPR;MPTI;MTRN;MTUS;NAVN;NDSN;NEO;NET;NEU;NKSH;NLY;NREF;NTB;NTRA;NUE;NVEC;NVT;PANW;PH;PKG;PLPC;PRAX;PRLB;PTGX;RAPP;RGA;RHI;ROIV;ROKU;RS;RVMD;SAFT;SENEA;SEPN;SHOO;SIF;SION;SNA;SNOW;SWK;SXI;SXT;TILE;TPC;TRIN;TSAT;TVTX;TWIN;TWLO;TWST;UFCS;UNM;URI;UVE;VABK;VAC;VIK;WBS;WCC;WEYS;WRLD;WSBF;WSM;WT;WTS;WTTR
- signal EPS 本地补源覆盖: 149; unresolved: 18
