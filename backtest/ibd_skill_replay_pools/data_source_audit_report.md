# Replay Pool Data Source Audit

## 判定规则

- 字段列缺失: 不正常，必须修复。
- 核心价格/结构字段空值: 不正常，必须修复。
- signal 行的 `eps_yoy_growth` 空值: 不正常，必须补充；非 signal 行 EPS 空值视为正常。
- signal 行的 IBD candidate / entry 判断字段必须完整；非 signal 行对应空值视为正常。
- `industry` / `sector` 允许用 `Unknown` 作为 repairable fallback，但会单独计数。
- pullback、52w high、dryness 等解释增强字段空值计为 optional gap，不阻断 pool 基准使用。

## 总览

- Weeks audited: 32
- Passed weeks: 0
- Weeks requiring supplement/repair: 32
- Abnormal empty values needing supplement/repair: 408
- Signal EPS gaps needing supplement: 408
- Signal EPS gaps with local supplement source: 43
- Signal EPS gaps unresolved: 365
- EPS supplement sources are reported separately and are not silently written back into historical replay pools.

## 每周审计

| snapshot_date | status | rows | cols | signal | missing_fields | abnormal_empty | signal_eps_missing | eps_supp_available | eps_unresolved | repairable_fallback | optional_gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-01-02 | failed | 306 | 47 | 29 | 0 | 1 | 1 | 0 | 1 | 18 | 433 |
| 2026-01-09 | failed | 346 | 47 | 149 | 0 | 18 | 18 | 1 | 17 | 24 | 607 |
| 2026-01-16 | failed | 380 | 47 | 145 | 0 | 15 | 15 | 2 | 13 | 24 | 795 |
| 2026-01-23 | failed | 376 | 47 | 64 | 0 | 6 | 6 | 0 | 6 | 26 | 750 |
| 2026-01-30 | failed | 407 | 47 | 106 | 0 | 9 | 9 | 2 | 7 | 28 | 831 |
| 2026-02-06 | failed | 482 | 47 | 265 | 0 | 15 | 15 | 3 | 12 | 18 | 1084 |
| 2026-02-13 | failed | 497 | 47 | 92 | 0 | 15 | 15 | 3 | 12 | 28 | 997 |
| 2026-02-20 | failed | 497 | 47 | 60 | 0 | 7 | 7 | 0 | 7 | 24 | 844 |
| 2026-02-27 | failed | 459 | 47 | 54 | 0 | 8 | 8 | 1 | 7 | 26 | 745 |
| 2026-03-06 | failed | 407 | 47 | 26 | 0 | 6 | 6 | 1 | 5 | 22 | 566 |
| 2026-03-13 | failed | 360 | 47 | 20 | 0 | 4 | 4 | 1 | 3 | 22 | 473 |
| 2026-03-20 | failed | 331 | 47 | 32 | 0 | 7 | 7 | 0 | 7 | 20 | 466 |
| 2026-03-27 | failed | 346 | 47 | 69 | 0 | 8 | 8 | 3 | 5 | 22 | 588 |
| 2026-04-02 | failed | 409 | 47 | 151 | 0 | 11 | 11 | 2 | 9 | 24 | 804 |
| 2026-04-10 | failed | 474 | 47 | 286 | 0 | 25 | 25 | 4 | 21 | 30 | 1011 |
| 2026-04-17 | failed | 452 | 47 | 174 | 0 | 18 | 18 | 2 | 16 | 30 | 1041 |
| 2026-04-24 | failed | 447 | 47 | 78 | 0 | 10 | 10 | 1 | 9 | 30 | 919 |
| 2026-05-01 | failed | 472 | 47 | 122 | 0 | 20 | 20 | 4 | 16 | 30 | 901 |
| 2026-05-08 | failed | 487 | 47 | 136 | 0 | 18 | 18 | 5 | 13 | 30 | 917 |
| 2026-05-15 | failed | 469 | 47 | 59 | 0 | 10 | 10 | 2 | 8 | 32 | 777 |
| 2026-05-22 | failed | 514 | 47 | 106 | 0 | 10 | 10 | 2 | 8 | 36 | 924 |
| 2026-05-29 | failed | 505 | 47 | 99 | 0 | 10 | 10 | 1 | 9 | 34 | 881 |
| 2026-06-05 | failed | 530 | 47 | 121 | 0 | 12 | 12 | 1 | 11 | 34 | 941 |
| 2026-06-12 | failed | 613 | 47 | 292 | 0 | 22 | 22 | 0 | 22 | 34 | 1180 |
| 2026-06-18 | failed | 591 | 47 | 119 | 0 | 17 | 17 | 0 | 17 | 36 | 941 |
| 2026-06-26 | failed | 669 | 47 | 239 | 0 | 25 | 25 | 0 | 25 | 42 | 1197 |
| 2026-07-02 | failed | 691 | 47 | 129 | 0 | 13 | 13 | 0 | 13 | 42 | 1221 |
| 2026-07-10 | failed | 683 | 47 | 85 | 0 | 15 | 15 | 0 | 15 | 48 | 1031 |
| 2026-07-17 | failed | 713 | 47 | 181 | 0 | 18 | 18 | 2 | 16 | 50 | 1020 |
| 2026-07-24 | failed | 731 | 47 | 114 | 0 | 11 | 11 | 0 | 11 | 48 | 1036 |
| 2026-07-31 | failed | 259 | 47 | 36 | 0 | 3 | 3 | 0 | 3 | 20 | 378 |
| 2026-08-07 | failed | 790 | 47 | 179 | 0 | 21 | 21 | 0 | 21 | 50 | 1112 |

## 每周明细

### 2026-01-02

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=1
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=300; ibd_entry_close_position_invalid_or_non_signal=300; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=300; ibd_entry_date_invalid_or_non_signal=300; ibd_entry_price_invalid_or_non_signal=300; ibd_entry_rule_invalid_or_non_signal=300; ibd_entry_volume_ratio_invalid_or_non_signal=300; ibd_trigger_price_invalid_or_non_signal=300; ...+12
- repairable fallback: industry=9; sector=9
- optional gap: ibd_candidate_extra=287; pullback_v_is_dry=47; pullback_duration_weeks=21; pullback_pct=21; pullback_pct_off_peak=21; dist_to_52w_high_pct=18; price_52_week_high=18
- signal EPS 缺失代码: NVMI
- signal EPS 本地补源覆盖: 0; unresolved: 1

### 2026-01-09

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=18
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=295; ibd_entry_close_position_invalid_or_non_signal=295; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=295; ibd_entry_date_invalid_or_non_signal=295; ibd_entry_price_invalid_or_non_signal=295; ibd_entry_rule_invalid_or_non_signal=295; ibd_entry_volume_ratio_invalid_or_non_signal=295; ibd_trigger_price_invalid_or_non_signal=295; ...+12
- repairable fallback: industry=12; sector=12
- optional gap: ibd_candidate_extra=258; pullback_v_is_dry=79; pullback_duration_weeks=76; pullback_pct=76; pullback_pct_off_peak=76; dist_to_52w_high_pct=21; price_52_week_high=21
- signal EPS 缺失代码: ASH;BE;BMRC;BVC;CTRE;H;IVZ;KMT;LINC;LQDA;NUTX;NVMI;NVRI;SNDK;SWBI;TWIN;VSXY;WYY
- signal EPS 本地补源覆盖: 1; unresolved: 17

### 2026-01-16

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=15
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=325; ibd_entry_close_position_invalid_or_non_signal=325; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=325; ibd_entry_date_invalid_or_non_signal=325; ibd_entry_price_invalid_or_non_signal=325; ibd_entry_rule_invalid_or_non_signal=325; ibd_entry_volume_ratio_invalid_or_non_signal=325; ibd_trigger_price_invalid_or_non_signal=325; ...+12
- repairable fallback: industry=12; sector=12
- optional gap: ibd_candidate_extra=296; pullback_v_is_dry=126; pullback_duration_weeks=111; pullback_pct=111; pullback_pct_off_peak=111; dist_to_52w_high_pct=20; price_52_week_high=20
- signal EPS 缺失代码: AEP;AMTM;AUGO;BE;CHMG;ELA;HZO;KLIC;LINC;LNT;PSNL;TECH;TWIN;WULF;WYY
- signal EPS 本地补源覆盖: 2; unresolved: 13

### 2026-01-23

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=6
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=354; ibd_entry_close_position_invalid_or_non_signal=354; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=354; ibd_entry_date_invalid_or_non_signal=354; ibd_entry_price_invalid_or_non_signal=354; ibd_entry_rule_invalid_or_non_signal=354; ibd_entry_volume_ratio_invalid_or_non_signal=354; ibd_trigger_price_invalid_or_non_signal=354; ...+12
- repairable fallback: industry=13; sector=13
- optional gap: ibd_candidate_extra=342; pullback_v_is_dry=122; pullback_duration_weeks=82; pullback_pct=82; pullback_pct_off_peak=82; dist_to_52w_high_pct=20; price_52_week_high=20
- signal EPS 缺失代码: ASND;CNOB;COHR;MOD;NVRI;RFIL
- signal EPS 本地补源覆盖: 0; unresolved: 6

### 2026-01-30

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=9
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=360; ibd_entry_breakout_range_ratio_invalid_or_non_signal=348; ibd_entry_close_position_invalid_or_non_signal=348; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=348; ibd_entry_date_invalid_or_non_signal=348; ibd_entry_price_invalid_or_non_signal=348; ibd_entry_rule_invalid_or_non_signal=348; ibd_entry_volume_ratio_invalid_or_non_signal=348; ...+12
- repairable fallback: industry=14; sector=14
- optional gap: ibd_candidate_extra=355; pullback_v_is_dry=127; pullback_duration_weeks=101; pullback_pct=101; pullback_pct_off_peak=101; dist_to_52w_high_pct=23; price_52_week_high=23
- signal EPS 缺失代码: BE;BUUU;CHMG;LMT;LQDA;MOD;NJR;SWBI;TSN
- signal EPS 本地补源覆盖: 2; unresolved: 7

### 2026-02-06

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=15
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=363; ibd_entry_breakout_range_ratio_invalid_or_non_signal=336; ibd_entry_close_position_invalid_or_non_signal=336; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=336; ibd_entry_date_invalid_or_non_signal=336; ibd_entry_price_invalid_or_non_signal=336; ibd_entry_rule_invalid_or_non_signal=336; ibd_entry_volume_ratio_invalid_or_non_signal=336; ...+12
- repairable fallback: industry=9; sector=9
- optional gap: ibd_candidate_extra=309; pullback_v_is_dry=191; pullback_duration_weeks=182; pullback_pct=182; pullback_pct_off_peak=182; dist_to_52w_high_pct=19; price_52_week_high=19
- signal EPS 缺失代码: ASH;CHMG;CNOB;CTRE;FSEA;GFF;GPRE;HSY;HZO;JBTM;KLIC;KMT;NVRI;RCBC;VSXY
- signal EPS 本地补源覆盖: 3; unresolved: 12

### 2026-02-13

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=15
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=454; ibd_entry_close_position_invalid_or_non_signal=454; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=454; ibd_entry_date_invalid_or_non_signal=454; ibd_entry_price_invalid_or_non_signal=454; ibd_entry_rule_invalid_or_non_signal=454; ibd_entry_volume_ratio_invalid_or_non_signal=454; ibd_trigger_price_invalid_or_non_signal=454; ...+12
- repairable fallback: industry=14; sector=14
- optional gap: ibd_candidate_extra=444; pullback_v_is_dry=203; pullback_duration_weeks=100; pullback_pct=100; pullback_pct_off_peak=100; dist_to_52w_high_pct=25; price_52_week_high=25
- signal EPS 缺失代码: AEP;AUGO;CRC;HCSG;ICHR;KLIC;LINC;LNT;MGRT;MO;RCBC;SR;TWIN;UHT;WULF
- signal EPS 本地补源覆盖: 3; unresolved: 12

### 2026-02-20

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=7
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=478; ibd_entry_close_position_invalid_or_non_signal=478; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=478; ibd_entry_date_invalid_or_non_signal=478; ibd_entry_price_invalid_or_non_signal=478; ibd_entry_rule_invalid_or_non_signal=478; ibd_entry_volume_ratio_invalid_or_non_signal=478; ibd_trigger_price_invalid_or_non_signal=478; ...+12
- repairable fallback: industry=12; sector=12
- optional gap: ibd_candidate_extra=462; pullback_v_is_dry=120; pullback_duration_weeks=72; pullback_pct=72; pullback_pct_off_peak=72; dist_to_52w_high_pct=23; price_52_week_high=23
- signal EPS 缺失代码: ASH;COHR;CTO;H;MSGS;UCTT;VSXY
- signal EPS 本地补源覆盖: 0; unresolved: 7

### 2026-02-27

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=8
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=438; ibd_entry_breakout_range_ratio_invalid_or_non_signal=426; ibd_entry_close_position_invalid_or_non_signal=426; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=426; ibd_entry_date_invalid_or_non_signal=426; ibd_entry_price_invalid_or_non_signal=426; ibd_entry_rule_invalid_or_non_signal=426; ibd_entry_volume_ratio_invalid_or_non_signal=426; ...+12
- repairable fallback: industry=13; sector=13
- optional gap: ibd_candidate_extra=425; pullback_v_is_dry=83; pullback_duration_weeks=63; pullback_pct=63; pullback_pct_off_peak=63; dist_to_52w_high_pct=24; price_52_week_high=24
- signal EPS 缺失代码: BE;HCSG;HSY;JAZZ;LNT;RFIL;TWIN;WULF
- signal EPS 本地补源覆盖: 1; unresolved: 7

### 2026-03-06

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=6
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=399; ibd_entry_breakout_range_ratio_invalid_or_non_signal=389; ibd_entry_close_position_invalid_or_non_signal=389; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=389; ibd_entry_date_invalid_or_non_signal=389; ibd_entry_price_invalid_or_non_signal=389; ibd_entry_rule_invalid_or_non_signal=389; ibd_entry_volume_ratio_invalid_or_non_signal=389; ...+12
- repairable fallback: industry=11; sector=11
- optional gap: ibd_candidate_extra=392; pullback_v_is_dry=57; pullback_duration_weeks=25; pullback_pct=25; pullback_pct_off_peak=25; dist_to_52w_high_pct=21; price_52_week_high=21
- signal EPS 缺失代码: ASND;GPRE;LMT;PBF;RLYB;SWBI
- signal EPS 本地补源覆盖: 1; unresolved: 5

### 2026-03-13

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=4
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=352; ibd_entry_breakout_range_ratio_invalid_or_non_signal=348; ibd_entry_close_position_invalid_or_non_signal=348; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=348; ibd_entry_date_invalid_or_non_signal=348; ibd_entry_price_invalid_or_non_signal=348; ibd_entry_rule_invalid_or_non_signal=348; ibd_entry_volume_ratio_invalid_or_non_signal=348; ...+12
- repairable fallback: industry=11; sector=11
- optional gap: ibd_candidate_extra=349; pullback_v_is_dry=27; dist_to_52w_high_pct=20; price_52_week_high=20; pullback_duration_weeks=19; pullback_pct=19; pullback_pct_off_peak=19
- signal EPS 缺失代码: BE;BVC;SNDK;SR
- signal EPS 本地补源覆盖: 1; unresolved: 3

### 2026-03-20

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=7
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=323; ibd_entry_breakout_range_ratio_invalid_or_non_signal=307; ibd_entry_close_position_invalid_or_non_signal=307; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=307; ibd_entry_date_invalid_or_non_signal=307; ibd_entry_price_invalid_or_non_signal=307; ibd_entry_rule_invalid_or_non_signal=307; ibd_entry_volume_ratio_invalid_or_non_signal=307; ...+12
- repairable fallback: industry=10; sector=10
- optional gap: ibd_candidate_extra=317; pullback_v_is_dry=32; pullback_duration_weeks=27; pullback_pct=27; pullback_pct_off_peak=27; dist_to_52w_high_pct=18; price_52_week_high=18
- signal EPS 缺失代码: ANDG;DK;ELA;ETON;ICHR;PTGX;UCTT
- signal EPS 本地补源覆盖: 0; unresolved: 7

### 2026-03-27

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=8
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=321; ibd_entry_close_position_invalid_or_non_signal=321; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=321; ibd_entry_date_invalid_or_non_signal=321; ibd_entry_price_invalid_or_non_signal=321; ibd_entry_rule_invalid_or_non_signal=321; ibd_entry_volume_ratio_invalid_or_non_signal=321; ibd_trigger_price_invalid_or_non_signal=321; ...+12
- repairable fallback: industry=11; sector=11
- optional gap: ibd_candidate_extra=316; pullback_v_is_dry=63; pullback_duration_weeks=57; pullback_pct=57; pullback_pct_off_peak=57; dist_to_52w_high_pct=19; price_52_week_high=19
- signal EPS 缺失代码: CRC;GPRE;LNT;MO;NTCT;PLAB;TSN;UNFI
- signal EPS 本地补源覆盖: 3; unresolved: 5

### 2026-04-02

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=11
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=377; ibd_entry_close_position_invalid_or_non_signal=377; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=377; ibd_entry_date_invalid_or_non_signal=377; ibd_entry_price_invalid_or_non_signal=377; ibd_entry_rule_invalid_or_non_signal=377; ibd_entry_volume_ratio_invalid_or_non_signal=377; ibd_trigger_price_invalid_or_non_signal=377; ...+12
- repairable fallback: industry=12; sector=12
- optional gap: ibd_candidate_extra=323; pullback_v_is_dry=121; pullback_duration_weeks=106; pullback_pct=106; pullback_pct_off_peak=106; dist_to_52w_high_pct=21; price_52_week_high=21
- signal EPS 缺失代码: ANDG;AUGO;ELA;ICHR;JAZZ;LNT;MGRT;MSGS;NJR;NVRI;TSN
- signal EPS 本地补源覆盖: 2; unresolved: 9

### 2026-04-10

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=25
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=396; ibd_entry_close_position_invalid_or_non_signal=396; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=396; ibd_entry_date_invalid_or_non_signal=396; ibd_entry_price_invalid_or_non_signal=396; ibd_entry_rule_invalid_or_non_signal=396; ibd_entry_volume_ratio_invalid_or_non_signal=396; ibd_trigger_price_invalid_or_non_signal=396; ...+12
- repairable fallback: industry=15; sector=15
- optional gap: ibd_candidate_extra=269; pullback_v_is_dry=181; pullback_duration_weeks=171; pullback_pct=171; pullback_pct_off_peak=171; dist_to_52w_high_pct=24; price_52_week_high=24
- signal EPS 缺失代码: AEHR;AEP;ASH;ASND;AUGO;BE;CHMG;CNOB;COHR;CTO;ICHR;KLIC;LNT;MGRT;MO;MOD;NVMI;PLAB;RFIL;SNDK;SR;TSN;UCTT;UHT;WULF
- signal EPS 本地补源覆盖: 4; unresolved: 21

### 2026-04-17

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=18
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=392; ibd_entry_close_position_invalid_or_non_signal=392; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=392; ibd_entry_date_invalid_or_non_signal=392; ibd_entry_price_invalid_or_non_signal=392; ibd_entry_rule_invalid_or_non_signal=392; ibd_entry_volume_ratio_invalid_or_non_signal=392; ibd_trigger_price_invalid_or_non_signal=392; ...+12
- repairable fallback: industry=15; sector=15
- optional gap: ibd_candidate_extra=327; pullback_v_is_dry=173; pullback_duration_weeks=163; pullback_pct=163; pullback_pct_off_peak=163; dist_to_52w_high_pct=26; price_52_week_high=26
- signal EPS 缺失代码: AKR;ASH;BE;CNOB;CTO;H;HPE;HZO;JAZZ;KLIC;LION;MANE;MSGS;OOMA;RFIL;ULS;UNFI;WULF
- signal EPS 本地补源覆盖: 2; unresolved: 16

### 2026-04-24

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=10
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=411; ibd_entry_close_position_invalid_or_non_signal=411; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=411; ibd_entry_date_invalid_or_non_signal=411; ibd_entry_price_invalid_or_non_signal=411; ibd_entry_rule_invalid_or_non_signal=411; ibd_entry_volume_ratio_invalid_or_non_signal=411; ibd_trigger_price_invalid_or_non_signal=411; ...+12
- repairable fallback: industry=15; sector=15
- optional gap: ibd_candidate_extra=399; pullback_v_is_dry=173; pullback_duration_weeks=101; pullback_pct=101; pullback_pct_off_peak=101; dist_to_52w_high_pct=22; price_52_week_high=22
- signal EPS 缺失代码: AEP;CHMG;FPS;HCSG;MDV;MO;MOD;OCC;OOMA;SWBI
- signal EPS 本地补源覆盖: 1; unresolved: 9

### 2026-05-01

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=20
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=411; ibd_entry_close_position_invalid_or_non_signal=411; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=411; ibd_entry_date_invalid_or_non_signal=411; ibd_entry_price_invalid_or_non_signal=411; ibd_entry_reject_reason_valid_or_non_signal=411; ibd_entry_rule_invalid_or_non_signal=411; ibd_entry_volume_ratio_invalid_or_non_signal=411; ...+12
- repairable fallback: industry=15; sector=15
- optional gap: ibd_candidate_extra=396; pullback_v_is_dry=146; pullback_duration_weeks=103; pullback_pct=103; pullback_pct_off_peak=103; dist_to_52w_high_pct=25; price_52_week_high=25
- signal EPS 缺失代码: AEP;AKR;BUUU;CNOB;CTO;DK;ETON;FPS;FVR;GPRE;IRM;JAZZ;LION;LNT;MO;MSGS;NVRI;PBF;TXNM;ULS
- signal EPS 本地补源覆盖: 4; unresolved: 16

### 2026-05-08

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=18
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=423; ibd_entry_breakout_range_ratio_invalid_or_non_signal=415; ibd_entry_close_position_invalid_or_non_signal=415; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=415; ibd_entry_date_invalid_or_non_signal=415; ibd_entry_price_invalid_or_non_signal=415; ibd_entry_rule_invalid_or_non_signal=415; ibd_entry_volume_ratio_invalid_or_non_signal=415; ...+12
- repairable fallback: industry=15; sector=15
- optional gap: ibd_candidate_extra=393; pullback_v_is_dry=137; pullback_duration_weeks=113; pullback_pct=113; pullback_pct_off_peak=113; dist_to_52w_high_pct=24; price_52_week_high=24
- signal EPS 缺失代码: AAON;AKR;AMN;APC;CTRE;FSEA;H;HCSG;HZO;IRM;JAZZ;LINC;MDV;MOD;RFIL;RLYB;TSN;ULS
- signal EPS 本地补源覆盖: 5; unresolved: 13

### 2026-05-15

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=10
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=441; ibd_entry_breakout_range_ratio_invalid_or_non_signal=438; ibd_entry_close_position_invalid_or_non_signal=438; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=438; ibd_entry_date_invalid_or_non_signal=438; ibd_entry_price_invalid_or_non_signal=438; ibd_entry_rule_invalid_or_non_signal=438; ibd_entry_volume_ratio_invalid_or_non_signal=438; ...+12
- repairable fallback: industry=16; sector=16
- optional gap: ibd_candidate_extra=427; pullback_v_is_dry=113; pullback_duration_weeks=63; pullback_pct=63; pullback_pct_off_peak=63; dist_to_52w_high_pct=24; price_52_week_high=24
- signal EPS 缺失代码: APC;BUUU;COHR;ETON;LQDA;MO;MSGS;OCC;PBF;TXNM
- signal EPS 本地补源覆盖: 2; unresolved: 8

### 2026-05-22

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=10
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=486; ibd_entry_close_position_invalid_or_non_signal=486; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=486; ibd_entry_date_invalid_or_non_signal=486; ibd_entry_price_invalid_or_non_signal=486; ibd_entry_rule_invalid_or_non_signal=486; ibd_entry_volume_ratio_invalid_or_non_signal=486; ibd_trigger_price_invalid_or_non_signal=486; ...+12
- repairable fallback: industry=18; sector=18
- optional gap: ibd_candidate_extra=462; pullback_v_is_dry=117; pullback_duration_weeks=97; pullback_pct=97; pullback_pct_off_peak=97; dist_to_52w_high_pct=27; price_52_week_high=27
- signal EPS 缺失代码: AKR;BE;CLBK;CRC;EDRY;H;LION;LQDA;NVRI;UHT
- signal EPS 本地补源覆盖: 2; unresolved: 8

### 2026-05-29

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=10
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=463; ibd_entry_close_position_invalid_or_non_signal=463; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=463; ibd_entry_date_invalid_or_non_signal=463; ibd_entry_price_invalid_or_non_signal=463; ibd_entry_rule_invalid_or_non_signal=463; ibd_entry_volume_ratio_invalid_or_non_signal=463; ibd_trigger_price_invalid_or_non_signal=463; ...+12
- repairable fallback: industry=17; sector=17
- optional gap: ibd_candidate_extra=435; pullback_v_is_dry=126; pullback_duration_weeks=90; pullback_pct=90; pullback_pct_off_peak=90; dist_to_52w_high_pct=25; price_52_week_high=25
- signal EPS 缺失代码: AAON;BUUU;CNOB;ELA;H;MOD;MOG-A;OCC;SNDK;WULF
- signal EPS 本地补源覆盖: 1; unresolved: 9

### 2026-06-05

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=12
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=495; ibd_entry_close_position_invalid_or_non_signal=495; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=495; ibd_entry_date_invalid_or_non_signal=495; ibd_entry_price_invalid_or_non_signal=495; ibd_entry_rule_invalid_or_non_signal=495; ibd_entry_volume_ratio_invalid_or_non_signal=495; ibd_trigger_price_invalid_or_non_signal=495; ...+12
- repairable fallback: industry=17; sector=17
- optional gap: ibd_candidate_extra=453; pullback_v_is_dry=130; pullback_duration_weeks=104; pullback_pct=104; pullback_pct_off_peak=104; dist_to_52w_high_pct=23; price_52_week_high=23
- signal EPS 缺失代码: AKR;AMN;CNOB;DK;FSEA;FVR;HNGE;NODK;OSCR;TWIN;UNFI;VSXY
- signal EPS 本地补源覆盖: 1; unresolved: 11

### 2026-06-12

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=22
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=513; ibd_entry_close_position_invalid_or_non_signal=513; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=513; ibd_entry_date_invalid_or_non_signal=513; ibd_entry_price_invalid_or_non_signal=513; ibd_entry_rule_invalid_or_non_signal=513; ibd_entry_volume_ratio_invalid_or_non_signal=513; ibd_trigger_price_invalid_or_non_signal=513; ...+12
- repairable fallback: industry=17; sector=17
- optional gap: ibd_candidate_extra=408; pullback_v_is_dry=188; pullback_duration_weeks=180; pullback_pct=180; pullback_pct_off_peak=180; dist_to_52w_high_pct=22; price_52_week_high=22
- signal EPS 缺失代码: ASH;CLBK;CNOB;CTO;ELA;ETON;FVR;HCSG;ICHR;KLIC;LQDA;NVMI;NVRI;OCC;PTGX;RFIL;SHOO;SNDK;TWIN;UCTT;WKC;WULF
- signal EPS 本地补源覆盖: 0; unresolved: 22

### 2026-06-18

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=17
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=549; ibd_entry_breakout_range_ratio_invalid_or_non_signal=514; ibd_entry_close_position_invalid_or_non_signal=514; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=514; ibd_entry_date_invalid_or_non_signal=514; ibd_entry_price_invalid_or_non_signal=514; ibd_entry_rule_invalid_or_non_signal=514; ibd_entry_volume_ratio_invalid_or_non_signal=514; ...+12
- repairable fallback: industry=18; sector=18
- optional gap: ibd_candidate_extra=487; pullback_v_is_dry=164; pullback_duration_weeks=82; pullback_pct=82; pullback_pct_off_peak=82; dist_to_52w_high_pct=22; price_52_week_high=22
- signal EPS 缺失代码: AEHR;ALH;ALOT;ASH;BE;FLXS;HCSG;ICHR;IRM;KLIC;LION;MOD;NVRI;PTGX;SWBI;VSXY;WULF
- signal EPS 本地补源覆盖: 0; unresolved: 17

### 2026-06-26

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=25
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=636; ibd_entry_breakout_range_ratio_invalid_or_non_signal=463; ibd_entry_close_position_invalid_or_non_signal=463; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=463; ibd_entry_date_invalid_or_non_signal=463; ibd_entry_price_invalid_or_non_signal=463; ibd_entry_rule_invalid_or_non_signal=463; ibd_entry_volume_ratio_invalid_or_non_signal=463; ...+12
- repairable fallback: industry=21; sector=21
- optional gap: ibd_candidate_extra=523; pullback_v_is_dry=170; pullback_duration_weeks=152; pullback_pct=152; pullback_pct_off_peak=152; dist_to_52w_high_pct=24; price_52_week_high=24
- signal EPS 缺失代码: AEP;ALMR;AMN;ASND;BUUU;CCXI;CLBK;DK;ETON;GFF;HCSG;HZO;IRM;LINC;LNT;MANE;MO;MSGS;NJR;NTCT;PK;PSNL;RCUS;RFIL;UHT
- signal EPS 本地补源覆盖: 0; unresolved: 25

### 2026-07-02

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=13
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=656; ibd_entry_close_position_invalid_or_non_signal=656; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=656; ibd_entry_date_invalid_or_non_signal=656; ibd_entry_price_invalid_or_non_signal=656; ibd_entry_rule_invalid_or_non_signal=656; ibd_entry_volume_ratio_invalid_or_non_signal=656; ibd_trigger_price_invalid_or_non_signal=656; ...+12
- repairable fallback: industry=21; sector=21
- optional gap: ibd_candidate_extra=601; pullback_v_is_dry=184; pullback_duration_weeks=130; pullback_pct=130; pullback_pct_off_peak=130; dist_to_52w_high_pct=23; price_52_week_high=23
- signal EPS 缺失代码: ASH;BMRC;CCXI;CRWD;CTRE;DK;HCSG;JAN;JAZZ;NUTX;NVRI;OOMA;UHT
- signal EPS 本地补源覆盖: 0; unresolved: 13

### 2026-07-10

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=15
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=664; ibd_entry_close_position_invalid_or_non_signal=664; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=664; ibd_entry_date_invalid_or_non_signal=664; ibd_entry_price_invalid_or_non_signal=664; ibd_entry_rule_invalid_or_non_signal=664; ibd_entry_volume_ratio_invalid_or_non_signal=664; ibd_trigger_price_invalid_or_non_signal=664; ...+12
- repairable fallback: industry=24; sector=24
- optional gap: ibd_candidate_extra=623; pullback_v_is_dry=139; pullback_duration_weeks=73; pullback_pct=73; pullback_pct_off_peak=73; dist_to_52w_high_pct=25; price_52_week_high=25
- signal EPS 缺失代码: ALH;ALMR;ANDG;COAG;GPRE;HPE;ICHR;LINC;MOD;NJR;NTCT;PBF;PENG;SION;TRAX
- signal EPS 本地补源覆盖: 0; unresolved: 15

### 2026-07-17

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=18
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=647; ibd_entry_close_position_invalid_or_non_signal=647; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=647; ibd_entry_date_invalid_or_non_signal=647; ibd_entry_price_invalid_or_non_signal=647; ibd_entry_rule_invalid_or_non_signal=647; ibd_entry_volume_ratio_invalid_or_non_signal=647; ibd_trigger_price_invalid_or_non_signal=647; ...+12
- repairable fallback: industry=25; sector=25
- optional gap: ibd_candidate_extra=575; pullback_v_is_dry=112; pullback_duration_weeks=93; pullback_pct=93; pullback_pct_off_peak=93; dist_to_52w_high_pct=27; price_52_week_high=27
- signal EPS 缺失代码: AKR;APC;ASH;CDNA;CHMG;CLBK;COAG;CTO;CTRE;GPRE;IRM;IVZ;MDV;MO;MSGS;PK;RLYB;SION
- signal EPS 本地补源覆盖: 2; unresolved: 16

### 2026-07-24

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=11
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=696; ibd_entry_close_position_invalid_or_non_signal=696; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=696; ibd_entry_date_invalid_or_non_signal=696; ibd_entry_price_invalid_or_non_signal=696; ibd_entry_rule_invalid_or_non_signal=696; ibd_entry_volume_ratio_invalid_or_non_signal=696; ibd_trigger_price_invalid_or_non_signal=696; ...+12
- repairable fallback: industry=24; sector=24
- optional gap: ibd_candidate_extra=647; pullback_v_is_dry=115; pullback_duration_weeks=74; pullback_pct=74; pullback_pct_off_peak=74; dist_to_52w_high_pct=26; price_52_week_high=26
- signal EPS 缺失代码: ALOT;BLFS;CHMG;CTRE;EDRY;FLXS;IRM;JAZZ;LQDA;MOG-A;RCBC
- signal EPS 本地补源覆盖: 0; unresolved: 11

### 2026-07-31

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=3
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=242; ibd_entry_breakout_range_ratio_invalid_or_non_signal=240; ibd_entry_close_position_invalid_or_non_signal=240; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=240; ibd_entry_date_invalid_or_non_signal=240; ibd_entry_price_invalid_or_non_signal=240; ibd_entry_rule_invalid_or_non_signal=240; ibd_entry_volume_ratio_invalid_or_non_signal=240; ...+12
- repairable fallback: industry=10; sector=10
- optional gap: ibd_candidate_extra=233; pullback_v_is_dry=41; pullback_duration_weeks=28; pullback_pct=28; pullback_pct_off_peak=28; dist_to_52w_high_pct=10; price_52_week_high=10
- signal EPS 缺失代码: ARXS;ETON;SION
- signal EPS 本地补源覆盖: 0; unresolved: 3

### 2026-08-07

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=21
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=716; ibd_entry_close_position_invalid_or_non_signal=716; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=716; ibd_entry_date_invalid_or_non_signal=716; ibd_entry_price_invalid_or_non_signal=716; ibd_entry_rule_invalid_or_non_signal=716; ibd_entry_volume_ratio_invalid_or_non_signal=716; ibd_trigger_price_invalid_or_non_signal=716; ...+12
- repairable fallback: industry=25; sector=25
- optional gap: ibd_candidate_extra=655; pullback_v_is_dry=122; pullback_duration_weeks=95; pullback_pct=95; pullback_pct_off_peak=95; dist_to_52w_high_pct=25; price_52_week_high=25
- signal EPS 缺失代码: AIRT;ALH;AMN;ASH;BUUU;BVS;CCXI;EDRY;FLXS;FSEA;GFF;HNGE;HPE;IVZ;NEO;PTGX;SHOO;SION;TECH;TWIN;VSXY
- signal EPS 本地补源覆盖: 0; unresolved: 21
