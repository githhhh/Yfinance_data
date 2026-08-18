# Replay Pool Data Source Audit

## 判定规则

- 字段列缺失: 不正常，必须修复。
- 核心价格/结构字段空值: 不正常，必须修复。
- signal 行的 `eps_yoy_growth` 空值: 不正常，必须补充；非 signal 行 EPS 空值视为正常。
- signal 行的 IBD candidate / entry 判断字段必须完整；非 signal 行对应空值视为正常。
- `industry` / `sector` 允许用 `Unknown` 作为 repairable fallback，但会单独计数。
- `price_52_week_high` / `dist_to_52w_high_pct` 是价格 as-of 派生字段，必须由已裁剪 daily pkl 重算且不得为空。
- pullback、dryness 等解释增强字段空值计为 optional gap，不阻断 pool 基准使用。

## 总览

- Weeks audited: 32
- Passed weeks: 0
- Weeks requiring supplement/repair: 32
- Abnormal empty values needing supplement/repair: 3817
- Signal EPS gaps needing supplement: 3817
- Signal EPS gaps with current snapshot-only source: 3452
- Signal EPS gaps unresolved: 365
- Current snapshot EPS supplement sources are reported separately and are not point-in-time safe.

## 每周审计

| snapshot_date | status | rows | cols | signal | missing_fields | abnormal_empty | signal_eps_missing | eps_supp_available | eps_unresolved | repairable_fallback | optional_gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-01-02 | failed | 306 | 47 | 29 | 0 | 29 | 29 | 28 | 1 | 18 | 397 |
| 2026-01-09 | failed | 346 | 47 | 149 | 0 | 149 | 149 | 132 | 17 | 24 | 565 |
| 2026-01-16 | failed | 380 | 47 | 145 | 0 | 145 | 145 | 132 | 13 | 24 | 755 |
| 2026-01-23 | failed | 376 | 47 | 64 | 0 | 64 | 64 | 58 | 6 | 26 | 710 |
| 2026-01-30 | failed | 407 | 47 | 106 | 0 | 106 | 106 | 99 | 7 | 28 | 785 |
| 2026-02-06 | failed | 482 | 47 | 265 | 0 | 265 | 265 | 253 | 12 | 18 | 1046 |
| 2026-02-13 | failed | 497 | 47 | 92 | 0 | 92 | 92 | 80 | 12 | 28 | 947 |
| 2026-02-20 | failed | 497 | 47 | 60 | 0 | 60 | 60 | 53 | 7 | 24 | 798 |
| 2026-02-27 | failed | 459 | 47 | 54 | 0 | 54 | 54 | 47 | 7 | 26 | 697 |
| 2026-03-06 | failed | 407 | 47 | 26 | 0 | 26 | 26 | 21 | 5 | 22 | 524 |
| 2026-03-13 | failed | 360 | 47 | 20 | 0 | 20 | 20 | 17 | 3 | 22 | 433 |
| 2026-03-20 | failed | 331 | 47 | 32 | 0 | 32 | 32 | 25 | 7 | 20 | 430 |
| 2026-03-27 | failed | 346 | 47 | 69 | 0 | 69 | 69 | 64 | 5 | 22 | 550 |
| 2026-04-02 | failed | 409 | 47 | 151 | 0 | 151 | 151 | 142 | 9 | 24 | 762 |
| 2026-04-10 | failed | 474 | 47 | 286 | 0 | 286 | 286 | 265 | 21 | 30 | 963 |
| 2026-04-17 | failed | 452 | 47 | 174 | 0 | 174 | 174 | 158 | 16 | 30 | 989 |
| 2026-04-24 | failed | 447 | 47 | 78 | 0 | 78 | 78 | 69 | 9 | 30 | 875 |
| 2026-05-01 | failed | 472 | 47 | 122 | 0 | 122 | 122 | 106 | 16 | 30 | 851 |
| 2026-05-08 | failed | 487 | 47 | 136 | 0 | 136 | 136 | 123 | 13 | 30 | 869 |
| 2026-05-15 | failed | 469 | 47 | 59 | 0 | 59 | 59 | 51 | 8 | 32 | 729 |
| 2026-05-22 | failed | 514 | 47 | 106 | 0 | 106 | 106 | 98 | 8 | 36 | 870 |
| 2026-05-29 | failed | 505 | 47 | 99 | 0 | 99 | 99 | 90 | 9 | 34 | 831 |
| 2026-06-05 | failed | 530 | 47 | 121 | 0 | 121 | 121 | 110 | 11 | 34 | 895 |
| 2026-06-12 | failed | 613 | 47 | 292 | 0 | 292 | 292 | 270 | 22 | 34 | 1136 |
| 2026-06-18 | failed | 591 | 47 | 119 | 0 | 119 | 119 | 102 | 17 | 36 | 897 |
| 2026-06-26 | failed | 669 | 47 | 239 | 0 | 239 | 239 | 214 | 25 | 42 | 1149 |
| 2026-07-02 | failed | 691 | 47 | 129 | 0 | 129 | 129 | 116 | 13 | 42 | 1175 |
| 2026-07-10 | failed | 683 | 47 | 85 | 0 | 85 | 85 | 70 | 15 | 48 | 981 |
| 2026-07-17 | failed | 713 | 47 | 181 | 0 | 181 | 181 | 165 | 16 | 50 | 966 |
| 2026-07-24 | failed | 731 | 47 | 114 | 0 | 114 | 114 | 103 | 11 | 48 | 984 |
| 2026-07-31 | failed | 259 | 47 | 36 | 0 | 36 | 36 | 33 | 3 | 20 | 358 |
| 2026-08-07 | failed | 790 | 47 | 179 | 0 | 179 | 179 | 158 | 21 | 50 | 1062 |

## 每周明细

### 2026-01-02

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=29
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=300; ibd_entry_close_position_invalid_or_non_signal=300; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=300; ibd_entry_date_invalid_or_non_signal=300; ibd_entry_price_invalid_or_non_signal=300; ibd_entry_rule_invalid_or_non_signal=300; ibd_entry_volume_ratio_invalid_or_non_signal=300; ibd_trigger_price_invalid_or_non_signal=300; ...+12
- repairable fallback: industry=9; sector=9
- optional gap: ibd_candidate_extra=287; pullback_v_is_dry=47; pullback_duration_weeks=21; pullback_pct=21; pullback_pct_off_peak=21
- signal EPS 缺失代码: ACMR;AEE;AMKR;ANET;ASML;ASX;ATRO;AXSM;BTSG;CGON;CHRN;CRS;DKL;FTI;FUSB;GE;HFBL;IBKR;LSCC;MKSI;NGS;NVMI;OHI;PHVS;TBBB;TER;TSM;UAN;WES
- signal EPS 本地补源覆盖: 28; unresolved: 1

### 2026-01-09

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=149
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=295; ibd_entry_close_position_invalid_or_non_signal=295; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=295; ibd_entry_date_invalid_or_non_signal=295; ibd_entry_price_invalid_or_non_signal=295; ibd_entry_rule_invalid_or_non_signal=295; ibd_entry_volume_ratio_invalid_or_non_signal=295; ibd_trigger_price_invalid_or_non_signal=295; ...+12
- repairable fallback: industry=12; sector=12
- optional gap: ibd_candidate_extra=258; pullback_v_is_dry=79; pullback_duration_weeks=76; pullback_pct=76; pullback_pct_off_peak=76
- signal EPS 缺失代码: ABCB;ACA;ACT;ADI;AIR;ALNT;ALRS;AMAT;AMKR;AMRX;AMZN;ARCB;ASH;ASML;AUB;AVBH;AZZ;BE;BFH;BH;BMRC;BOKF;BSRR;BVC;BWFG;CASY;CAT;CFBK;CFG;CGON;CR;CSWC;CTBI;CTRE;CUBI;CW;DAC;DAL;DCO;DHT;DJCO;DKL;DOCN;ECO;EGP;ELAN;ENS;ESQ;EXPD;EXPE;EZPW;FCCO;FDX;FHB;FITB;FORM;FR;FRAF;FUSB;GEF;GH;GOOG;GRC;GSL;GTX;H;HBNC;HEI;HLT;HVT;HWC;HWM;HXL;IFS;INCY;ITRN;IVZ;JBHT;JXN;KLAC;KMT;LGND;LINC;LIND;LIVN;LQDA;LSCC;MBX;MG;MOV;MTRN;MTX;NTRA;NTRS;NUTX;NVGS;NVMI;NVRI;NWPX;PCAR;PFBC;PKE;PLOW;PRM;PROV;PSMT;PSTL;PSX;RBC;RDVT;RF;ROK;ROST;RPRX;RVMD;SBCF;SHBI;SHC;SMBK;SMTC;SNDK;SRBK;ST;STX;SUNC;SWBI;TFSL;THRM;TKR;TRIN;TROW;TWIN;UBSI;UVSP;VLO;VOYA;VSAT;VSEC;VSXY;WAB;WBS;WCC;WDC;WERN;WES;WSBF;WTS;WYY;XHR
- signal EPS 本地补源覆盖: 132; unresolved: 17

### 2026-01-16

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=145
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=325; ibd_entry_close_position_invalid_or_non_signal=325; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=325; ibd_entry_date_invalid_or_non_signal=325; ibd_entry_price_invalid_or_non_signal=325; ibd_entry_rule_invalid_or_non_signal=325; ibd_entry_volume_ratio_invalid_or_non_signal=325; ibd_trigger_price_invalid_or_non_signal=325; ...+12
- repairable fallback: industry=12; sector=12
- optional gap: ibd_candidate_extra=296; pullback_v_is_dry=126; pullback_duration_weeks=111; pullback_pct=111; pullback_pct_off_peak=111
- signal EPS 缺失代码: AAMI;ACA;ADM;ADPT;AEE;AEP;AIR;AIT;AME;AMTM;ANDE;APGE;AUB;AUGO;AVBC;AZZ;BCAL;BE;BELFA;BH;BNL;BOKF;BSRR;CBAN;CBK;CCBG;CCNE;CFBK;CHMG;CR;CTBI;CURB;CVX;CW;CZNC;D;DAC;DGII;DGX;ECBK;ECO;EFSI;EGP;ELA;ELAN;EMR;EPD;ESI;EVRG;FBIZ;FBK;FBNC;FBRX;FHI;FNRN;FNWD;FRST;FSBC;GD;GHM;GLW;HAFC;HBCP;HBNC;HBT;HVT;HZO;IBKR;INSW;ITRN;JBL;JCI;JNJ;KEYS;KLIC;KN;KRYS;LAMR;LFUS;LINC;LNT;LSBK;LTC;LXFR;MBBC;MBWM;MBX;MNST;MOV;MPB;MPLX;MRNA;MSGE;MTSI;MTX;MYRG;NHC;NPO;NWPX;ORKA;OSBC;PKBK;PLD;PLOW;PLPC;PLXS;PSNL;PSTL;ROIV;RTX;RUSHA;SBCF;SCHW;SENEA;SMBK;SN;STX;SUNC;SXI;TECH;TFSL;TIGO;TILE;TMP;TPC;TRMK;TRST;TSBK;TSEM;TTMI;TWIN;UMAC;UNTY;USFD;VIAV;VSEC;VTRS;WCC;WELL;WPC;WSBF;WT;WTS;WULF;WYY
- signal EPS 本地补源覆盖: 132; unresolved: 13

### 2026-01-23

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=64
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=354; ibd_entry_close_position_invalid_or_non_signal=354; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=354; ibd_entry_date_invalid_or_non_signal=354; ibd_entry_price_invalid_or_non_signal=354; ibd_entry_rule_invalid_or_non_signal=354; ibd_entry_volume_ratio_invalid_or_non_signal=354; ibd_trigger_price_invalid_or_non_signal=354; ...+12
- repairable fallback: industry=13; sector=13
- optional gap: ibd_candidate_extra=342; pullback_v_is_dry=122; pullback_duration_weeks=82; pullback_pct=82; pullback_pct_off_peak=82
- signal EPS 缺失代码: ADPT;AMAL;AMGN;ASND;ATLO;AVBH;AXSM;BFC;BVFL;CBNA;CGEM;CNOB;COHR;CSX;CTBI;CVS;DAC;DEA;DHT;DNTH;DTM;EPD;ESQ;FCAP;FMBH;FNWD;FSBC;FTK;FUSB;FVCB;GFS;GH;GRDN;HBT;HFWA;IFS;KO;LBRX;LCNB;LSBK;MBWM;MOD;MTB;MVBF;NGL;NGS;NHC;NIC;NVGS;NVRI;OBK;ORKA;PEBO;PHVS;PROV;PSX;RFIL;ROK;ROKU;RPRX;SNA;STLD;VVX;XHR
- signal EPS 本地补源覆盖: 58; unresolved: 6

### 2026-01-30

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=106
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=360; ibd_entry_breakout_range_ratio_invalid_or_non_signal=348; ibd_entry_close_position_invalid_or_non_signal=348; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=348; ibd_entry_date_invalid_or_non_signal=348; ibd_entry_price_invalid_or_non_signal=348; ibd_entry_rule_invalid_or_non_signal=348; ibd_entry_volume_ratio_invalid_or_non_signal=348; ...+12
- repairable fallback: industry=14; sector=14
- optional gap: ibd_candidate_extra=355; pullback_v_is_dry=127; pullback_duration_weeks=101; pullback_pct=101; pullback_pct_off_peak=101
- signal EPS 缺失代码: AAPL;ACNB;ACT;ALRS;AMRX;AMTB;AROC;AROW;ASB;AVBC;AVBH;AXGN;BE;BFC;BHE;BLX;BOTJ;BPOP;BUSE;BUUU;BY;CAC;CARE;CATY;CB;CBAN;CBK;CBNA;CHMG;CIEN;CINF;CNO;CPF;CSX;CUBI;CZNC;DAC;DHT;DIOD;DKL;DOCN;ECO;ENVA;EPD;ESE;FANG;FCAP;FDX;FFBC;FISI;FRO;FSTR;FUSB;FXNC;GLW;GOOG;GSL;GTX;HFBL;HTB;HWBK;JNJ;JXN;KNSA;KO;LFUS;LIVN;LMT;LQDA;MOD;MPB;NBBK;NDSN;NJR;NPO;NTB;ONB;OVBC;PAA;PAGP;PBFS;PCB;PEBO;PFG;PFS;PLBC;PLPC;PLXS;PNW;SCHW;SPG;SRCE;STT;SWBI;TBBB;TDW;THFF;TMP;TRMK;TSBK;TSN;UMBF;UNTY;VIAV;WSFS;WTBA
- signal EPS 本地补源覆盖: 99; unresolved: 7

### 2026-02-06

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=265
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=363; ibd_entry_breakout_range_ratio_invalid_or_non_signal=336; ibd_entry_close_position_invalid_or_non_signal=336; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=336; ibd_entry_date_invalid_or_non_signal=336; ibd_entry_price_invalid_or_non_signal=336; ibd_entry_rule_invalid_or_non_signal=336; ibd_entry_volume_ratio_invalid_or_non_signal=336; ...+12
- repairable fallback: industry=9; sector=9
- optional gap: ibd_candidate_extra=309; pullback_v_is_dry=191; pullback_duration_weeks=182; pullback_pct=182; pullback_pct_off_peak=182
- signal EPS 缺失代码: ABCB;ACA;ACNB;ACT;AFL;AIR;AIT;AIZ;ALRS;ALX;AMGN;AMTB;ARCB;AROW;ASB;ASH;ASX;ATI;AUB;AVT;AZZ;BBT;BCAL;BFH;BFST;BH;BMY;BNL;BNY;BOH;BPOP;BSRR;BUSE;BVFL;BWFG;C;CARE;CASY;CATY;CB;CBK;CCBG;CCK;CCNE;CDP;CFR;CHMG;CHRN;CIEN;CINF;CLMT;CLYM;CNO;CNOB;COCO;COSO;CPF;CR;CRS;CSCO;CSTM;CTBI;CTRE;CURB;CZWI;D;DAL;DCO;DCOM;DEA;DGII;DINO;DTM;DXPE;ECBK;ECPG;EGP;ELAN;EMR;ESI;ESQ;EWBC;EXPD;EZPW;FANG;FBIZ;FBK;FBLA;FBNC;FBP;FCCO;FCF;FFBC;FISI;FIVE;FMBH;FNB;FNLC;FNRN;FNWD;FORM;FRST;FSBC;FSEA;FSTR;FULT;FUSB;FXNC;GD;GE;GEF;GFF;GFS;GHM;GL;GPRE;GRDN;GTX;GWW;HALO;HBCP;HBNC;HBT;HFWA;HG;HIFS;HLIO;HLT;HST;HSY;HTB;HVT;HWC;HWM;HZO;INCY;ITIC;ITRN;IVT;JBHT;JBL;JBTM;JCI;JPM;KALU;KEYS;KGS;KLIC;KMT;KN;KYMR;LAMR;LTC;MCY;MG;MGYR;MMM;MNSB;MOV;MPB;MSM;MTRN;MTSI;MTX;MVBF;MYRG;NBBK;NDSN;NGL;NGS;NHC;NIC;NKSH;NPO;NSC;NTB;NVRI;NWPX;OBT;OII;ONB;OSBC;OUT;OVBC;PBAM;PBFS;PCAR;PCB;PEBK;PEBO;PFIS;PFS;PH;PLD;PM;PNC;PNW;PR;PRLB;PSMT;PSX;QCRH;R;RBC;RCBC;REG;RF;RMR;RNR;RNST;ROIV;ROST;RPRX;RS;SBCF;SFNC;SHBI;SLAB;SMBC;SMBK;SMTC;SNX;SRBK;SRCE;STBA;STLD;STRL;SUN;SUNC;SXI;TBBB;TCBX;TFSL;THFF;TILE;TKR;TMP;TOWN;TRGP;TRMK;TRST;TRV;TXN;UNP;UNTY;USCB;USFD;UVSP;VIK;VLO;VSXY;VTR;WAB;WASH;WBS;WPC;WSBC;WSFS;WSM;WTBA;WTS;XHR;XMTR;XPO;ZION
- signal EPS 本地补源覆盖: 253; unresolved: 12

### 2026-02-13

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=92
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=454; ibd_entry_close_position_invalid_or_non_signal=454; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=454; ibd_entry_date_invalid_or_non_signal=454; ibd_entry_price_invalid_or_non_signal=454; ibd_entry_rule_invalid_or_non_signal=454; ibd_entry_volume_ratio_invalid_or_non_signal=454; ibd_trigger_price_invalid_or_non_signal=454; ...+12
- repairable fallback: industry=14; sector=14
- optional gap: ibd_candidate_extra=444; pullback_v_is_dry=203; pullback_duration_weeks=100; pullback_pct=100; pullback_pct_off_peak=100
- signal EPS 缺失代码: AAMI;ACA;ADM;AEE;AEP;AGX;AHR;ALNT;AMAT;AMG;ATI;AUGO;BRX;CAPL;CLYM;CMI;CR;CRC;CRS;CSX;CURB;CVEO;CW;DGII;DGX;ESI;EVRG;FBRX;FHI;FIVE;FORM;FRBA;GFS;GSL;GTX;HCSG;HST;HWM;HXL;ICHR;IDA;IESC;KLAC;KLIC;KNSA;KYMR;LINC;LNT;LSCC;LTC;MGRT;MO;MPLX;MTRN;MTSI;MYRG;NSC;NTST;NVGS;OHI;PFIS;PKG;PROV;RCBC;ROST;SBRA;SENEA;SEPN;SIF;SN;SPHR;SR;ST;STRL;TEN;THC;TSM;TWIN;UFCS;UHT;UNF;UNP;UTMD;VAL;VCTR;VSAT;WBI;WELL;WES;WT;WULF;XHR
- signal EPS 本地补源覆盖: 80; unresolved: 12

### 2026-02-20

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=60
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=478; ibd_entry_close_position_invalid_or_non_signal=478; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=478; ibd_entry_date_invalid_or_non_signal=478; ibd_entry_price_invalid_or_non_signal=478; ibd_entry_rule_invalid_or_non_signal=478; ibd_entry_volume_ratio_invalid_or_non_signal=478; ibd_trigger_price_invalid_or_non_signal=478; ...+12
- repairable fallback: industry=12; sector=12
- optional gap: ibd_candidate_extra=462; pullback_v_is_dry=120; pullback_duration_weeks=72; pullback_pct=72; pullback_pct_off_peak=72
- signal EPS 缺失代码: ANRO;ASH;AXGN;BLBD;CDP;CGEM;CGON;CHEF;CNA;COHR;CSCO;CTO;CVLG;FANG;FBP;FHI;FR;FTH;GE;GSL;H;HBCP;HEI;HIFS;HWM;HYNE;JBL;KN;KNSA;KNX;LAMR;LIVN;MATX;MNSB;MRNA;MSGE;MSGS;MYE;NMM;PECO;PRAX;R;RAPP;RTX;SBLK;SENEA;SHIP;SIF;SMBC;SNDA;SPG;STBA;TNK;TRV;TTMI;UCTT;VSEC;VSXY;XPO;ZIM
- signal EPS 本地补源覆盖: 53; unresolved: 7

### 2026-02-27

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=54
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=438; ibd_entry_breakout_range_ratio_invalid_or_non_signal=426; ibd_entry_close_position_invalid_or_non_signal=426; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=426; ibd_entry_date_invalid_or_non_signal=426; ibd_entry_price_invalid_or_non_signal=426; ibd_entry_rule_invalid_or_non_signal=426; ibd_entry_volume_ratio_invalid_or_non_signal=426; ...+12
- repairable fallback: industry=13; sector=13
- optional gap: ibd_candidate_extra=425; pullback_v_is_dry=83; pullback_duration_weeks=63; pullback_pct=63; pullback_pct_off_peak=63
- signal EPS 缺失代码: AAMI;AGX;AIZ;AMGN;ARTNA;BE;BMY;BTSG;CASY;CLDX;CON;DNTH;ECPG;EGP;ELAN;EPR;ESEA;EXPE;EZPW;FAF;HCSG;HSY;IMAX;JAZZ;JNJ;KLAC;KO;LAMR;LH;LNT;MBBC;MG;MTRN;NVT;NWPX;ODC;OHI;OUT;PKE;RCKY;REG;RFIL;RPRX;SBRA;SKT;SNDA;SUN;SUNC;THC;TIGO;TWIN;VLO;VVX;WULF
- signal EPS 本地补源覆盖: 47; unresolved: 7

### 2026-03-06

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=26
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=399; ibd_entry_breakout_range_ratio_invalid_or_non_signal=389; ibd_entry_close_position_invalid_or_non_signal=389; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=389; ibd_entry_date_invalid_or_non_signal=389; ibd_entry_price_invalid_or_non_signal=389; ibd_entry_rule_invalid_or_non_signal=389; ibd_entry_volume_ratio_invalid_or_non_signal=389; ...+12
- repairable fallback: industry=11; sector=11
- optional gap: ibd_candidate_extra=392; pullback_v_is_dry=57; pullback_duration_weeks=25; pullback_pct=25; pullback_pct_off_peak=25
- signal EPS 缺失代码: ASND;CAPL;CLMT;CVEO;DCO;DINO;DNTH;EPD;EXPE;FTK;GD;GPRE;HFBL;LMT;MPC;PARR;PBF;PBT;PSX;RLYB;RTX;SWBI;TNGX;UAN;UNF;VVX
- signal EPS 本地补源覆盖: 21; unresolved: 5

### 2026-03-13

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=20
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=352; ibd_entry_breakout_range_ratio_invalid_or_non_signal=348; ibd_entry_close_position_invalid_or_non_signal=348; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=348; ibd_entry_date_invalid_or_non_signal=348; ibd_entry_price_invalid_or_non_signal=348; ibd_entry_rule_invalid_or_non_signal=348; ibd_entry_volume_ratio_invalid_or_non_signal=348; ...+12
- repairable fallback: industry=11; sector=11
- optional gap: ibd_candidate_extra=349; pullback_v_is_dry=27; pullback_duration_weeks=19; pullback_pct=19; pullback_pct_off_peak=19
- signal EPS 缺失代码: ADM;AMKR;ANDE;ANRO;BE;BOTJ;BVC;CQP;DOCN;ESQ;LBRX;MU;ORKA;PLPC;SNDK;SR;ST;UAN;UMAC;XENE
- signal EPS 本地补源覆盖: 17; unresolved: 3

### 2026-03-20

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=32
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=323; ibd_entry_breakout_range_ratio_invalid_or_non_signal=307; ibd_entry_close_position_invalid_or_non_signal=307; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=307; ibd_entry_date_invalid_or_non_signal=307; ibd_entry_price_invalid_or_non_signal=307; ibd_entry_rule_invalid_or_non_signal=307; ibd_entry_volume_ratio_invalid_or_non_signal=307; ...+12
- repairable fallback: industry=10; sector=10
- optional gap: ibd_candidate_extra=317; pullback_v_is_dry=32; pullback_duration_weeks=27; pullback_pct=27; pullback_pct_off_peak=27
- signal EPS 缺失代码: ANDG;ARCB;CIEN;CLDX;DINO;DIOD;DK;DOCN;ELA;ELAN;ESEA;ETON;FIVE;GFS;GRDN;HYNE;ICHR;KRP;OBT;PBT;PTGX;REPX;SLAB;STX;TEN;TNGX;TSAT;TSEM;UCTT;UTMD;WDC;WTTR
- signal EPS 本地补源覆盖: 25; unresolved: 7

### 2026-03-27

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=69
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=321; ibd_entry_close_position_invalid_or_non_signal=321; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=321; ibd_entry_date_invalid_or_non_signal=321; ibd_entry_price_invalid_or_non_signal=321; ibd_entry_rule_invalid_or_non_signal=321; ibd_entry_volume_ratio_invalid_or_non_signal=321; ibd_trigger_price_invalid_or_non_signal=321; ...+12
- repairable fallback: industry=11; sector=11
- optional gap: ibd_candidate_extra=316; pullback_v_is_dry=63; pullback_duration_weeks=57; pullback_pct=57; pullback_pct_off_peak=57
- signal EPS 缺失代码: AAMI;ACNB;ADM;AGX;ANDE;AROW;ATEN;ATLO;AZZ;CAC;CARE;CASY;CBK;CLMT;COSO;CRC;CSCO;CSTM;CZWI;DELL;DXPE;ELVN;EPD;ESE;ET;FNRN;FTI;FXNC;GPRE;HBT;KN;LNT;MBWM;MGYR;MO;MPB;MTRN;MTX;NGS;NIC;NTCT;NVGS;OII;PAA;PAGP;PBAM;PCB;PFIS;PLAB;PLPC;PRLB;R;RUSHA;SENEA;SILC;SMBK;ST;SUNC;TDW;TIGO;TMP;TNK;TPC;TRGP;TSN;UNFI;VAL;WBI;WTBA
- signal EPS 本地补源覆盖: 64; unresolved: 5

### 2026-04-02

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=151
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=377; ibd_entry_close_position_invalid_or_non_signal=377; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=377; ibd_entry_date_invalid_or_non_signal=377; ibd_entry_price_invalid_or_non_signal=377; ibd_entry_rule_invalid_or_non_signal=377; ibd_entry_volume_ratio_invalid_or_non_signal=377; ibd_trigger_price_invalid_or_non_signal=377; ...+12
- repairable fallback: industry=12; sector=12
- optional gap: ibd_candidate_extra=323; pullback_v_is_dry=121; pullback_duration_weeks=106; pullback_pct=106; pullback_pct_off_peak=106
- signal EPS 缺失代码: AAMI;AMAL;AMTB;ANDG;ANRO;AROW;ARW;ATLO;AUB;AUGO;AVBC;AXGN;BBT;BCAX;BHE;BLX;BMY;BNL;BNY;BOTJ;BPOP;BRX;BTSG;BUSE;BVFL;BY;CAC;CARE;CBK;CBL;CCNE;CDP;CGEM;CINF;CLMT;CNO;COSO;CPF;CSTM;CTBI;D;DAC;DFTX;ECO;ELA;ESE;ESEA;ESQ;EZPW;FBIZ;FBLA;FBNC;FHI;FISI;FITB;FIVE;FMBH;FNB;FORM;FRO;FULT;FXNC;HBNC;HFWA;HG;HST;HWC;HYNE;ICHR;IFS;IMAX;INGM;INSW;ITRN;JAZZ;JBHT;KNSA;LIN;LNT;MATX;MBWM;MGRT;MPB;MRVL;MSGE;MSGS;MTRN;MYRG;NGL;NJR;NMM;NTB;NTST;NVRI;NVT;OBK;ODC;OSBC;OSW;OVBC;PCAR;PCB;PEBO;PFG;PFIS;PFS;PHVS;PLPC;PNC;RAPP;RCKY;REG;ROK;ROST;RPRX;RSI;SBCF;SBLK;SHIP;SILC;SLAB;SMBK;SMTC;SNA;SNX;SPHR;SRBK;SRCE;STX;TBBB;TDW;TMP;TNK;TRIN;TSBK;TSEM;TSN;UFCS;USB;UVSP;VIK;VIRT;VSAT;VSEC;VVX;WERN;WPC;WSFS;WTBA;XHR;XPO
- signal EPS 本地补源覆盖: 142; unresolved: 9

### 2026-04-10

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=286
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=396; ibd_entry_close_position_invalid_or_non_signal=396; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=396; ibd_entry_date_invalid_or_non_signal=396; ibd_entry_price_invalid_or_non_signal=396; ibd_entry_rule_invalid_or_non_signal=396; ibd_entry_volume_ratio_invalid_or_non_signal=396; ibd_trigger_price_invalid_or_non_signal=396; ...+12
- repairable fallback: industry=15; sector=15
- optional gap: ibd_candidate_extra=269; pullback_v_is_dry=181; pullback_duration_weeks=171; pullback_pct=171; pullback_pct_off_peak=171
- signal EPS 缺失代码: AAMI;ACA;ACLS;ACMR;AEE;AEHR;AEP;AGX;AIP;AIR;AIT;ALNT;ALRS;ALX;AMAL;AMAT;AMG;AMKR;AMTB;APGE;ARCB;AROC;AROW;ARW;ASB;ASH;ASML;ASND;ASX;AUB;AUGO;AVBC;AVBH;AZZ;BBT;BE;BELFA;BFC;BHE;BLBD;BNL;BNY;BOH;BOTJ;BPOP;BRX;BUSE;BVFL;BWFG;BY;C;CASS;CAT;CATY;CBAN;CBK;CCBG;CCNE;CDP;CFR;CHMG;CINF;CLDX;CLYM;CMI;CNO;CNOB;COHR;CPF;CRS;CSCO;CSX;CTBI;CTO;CURB;CVEO;CW;CZNC;D;DCO;DCOM;DEA;DFTX;DGII;DIOD;DXPE;ECBK;ECPG;EGP;ELVN;ENS;ESE;ESI;ETN;EVRG;EWBC;FBIZ;FBLA;FBNC;FBP;FBRX;FCBC;FCCO;FCF;FDX;FFBC;FHB;FISI;FITB;FLEX;FMBH;FNB;FNLC;FORM;FRAF;FRST;FSTR;FULT;FUSB;FXNC;GEF;GHM;GL;GLW;GRC;GRDN;GRMN;GSBC;HBCP;HBNC;HBT;HFWA;HLIO;HLT;HST;HWC;ICHR;IDA;IESC;INGM;INTC;IVT;JBL;JCI;KALU;KEYS;KLAC;KLIC;KN;KNX;LAMR;LCNB;LFUS;LIND;LNT;LRCX;LSBK;LSCC;LTC;MAC;MATX;MG;MGRT;MGYR;MKSI;MNSB;MO;MOD;MOV;MRVL;MRX;MSGE;MSM;MTB;MTSI;MTX;MVBF;MYE;MYRG;NBBK;NBIS;NDSN;NESR;NHC;NIC;NPO;NTB;NTRS;NTST;NVMI;NVT;NWPX;OBK;OBT;ODC;OHI;ONTO;OSBC;OSW;OUT;OVBC;PCAR;PCB;PDFS;PEBK;PEBO;PECO;PFG;PFIS;PFS;PKBK;PKE;PLAB;PLBC;PLOW;PLPC;PLXS;PNC;PNW;PROV;PSMT;R;RAPP;RBC;RCKY;REG;RF;RFIL;RMR;RNR;SBCF;SHBI;SHIP;SIF;SKT;SMBC;SMTC;SNA;SNDK;SNX;SPNT;SR;SRBK;SRCE;ST;STBA;STRL;STT;STX;SXI;SYRE;TBBB;TCBX;TER;TFSL;THFF;TPC;TRMK;TRST;TRV;TSAT;TSN;TTMI;UBSI;UCTT;UFCS;UHT;UNP;UNTY;UTMD;UVSP;VIAV;VSEC;VTR;WAB;WDC;WERN;WEYS;WPC;WSBC;WSBF;WT;WULF;XHR;XPO;ZION
- signal EPS 本地补源覆盖: 265; unresolved: 21

### 2026-04-17

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=174
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=392; ibd_entry_close_position_invalid_or_non_signal=392; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=392; ibd_entry_date_invalid_or_non_signal=392; ibd_entry_price_invalid_or_non_signal=392; ibd_entry_rule_invalid_or_non_signal=392; ibd_entry_volume_ratio_invalid_or_non_signal=392; ibd_trigger_price_invalid_or_non_signal=392; ...+12
- repairable fallback: industry=15; sector=15
- optional gap: ibd_candidate_extra=327; pullback_v_is_dry=173; pullback_duration_weeks=163; pullback_pct=163; pullback_pct_off_peak=163
- signal EPS 缺失代码: ACA;ACMR;ADI;AKR;ALL;ALNT;ALRS;ALX;AMAL;AMD;AME;AMKR;AMLX;ANRO;ARCB;ARMK;AROW;ARW;ASH;ASX;ATI;AVBC;AVBP;AVT;AXGN;BATRA;BCAX;BE;BELFA;BFH;BMY;BOH;BOKF;BRX;BTSG;CAC;CASS;CATY;CCK;CDP;CGEM;CLDX;CMI;CNA;CNOB;COSO;CPF;CSCO;CSWC;CSX;CTBI;CTO;CTRN;CVEO;CVLG;DCOM;DGX;EBAY;EGP;ESEA;ESQ;EZPW;FBP;FDX;FISI;FIVE;FLEX;FR;FRAF;FRO;FRT;FSBC;FUSB;GFS;GHM;GL;GRMN;GTX;H;HFBL;HG;HPE;HR;HST;HZO;IBKR;IBTA;IVT;JAZZ;JBHT;KIM;KLAC;KLIC;KNX;KRG;LAMR;LBRX;LGND;LIND;LION;LIVN;MAC;MANE;MCHB;MCY;MITK;MKSI;MNSB;MSGS;MSM;MTRN;MU;NBIS;NDSN;NSC;NVGS;OHI;ON;ONTO;OOMA;OPLN;OUT;OVBC;PBAM;PBI;PDFS;PEBO;PECO;PHVS;PLD;PLGO;PRAX;PRLB;PSMT;PSTL;R;RFIL;RNR;ROIV;RPRX;RSI;RUSHA;RVMD;SILC;SIRI;SKT;SLAB;SMTC;SNDA;SPG;STRL;STRZ;STT;SXI;TRVI;TXN;UAN;ULS;UNF;UNFI;UTMD;UVE;VCTR;VIK;VSH;VTRS;WBS;WCC;WELL;WT;WULF;XHR;XPO;ZIM
- signal EPS 本地补源覆盖: 158; unresolved: 16

### 2026-04-24

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=78
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=411; ibd_entry_close_position_invalid_or_non_signal=411; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=411; ibd_entry_date_invalid_or_non_signal=411; ibd_entry_price_invalid_or_non_signal=411; ibd_entry_rule_invalid_or_non_signal=411; ibd_entry_volume_ratio_invalid_or_non_signal=411; ibd_trigger_price_invalid_or_non_signal=411; ...+12
- repairable fallback: industry=15; sector=15
- optional gap: ibd_candidate_extra=399; pullback_v_is_dry=173; pullback_duration_weeks=101; pullback_pct=101; pullback_pct_off_peak=101
- signal EPS 缺失代码: ACT;AEP;ANDE;ANET;ARM;AROC;ARWR;ASYS;ATEX;AZZ;BATRA;BLBD;BRUN;CHMG;COHU;CSCO;CVLG;DGII;DINO;DOCN;ECO;ETN;FAF;FCEL;FPS;GHRS;GL;GOOG;HAFC;HCSG;HR;HXL;IESC;IIPR;KALU;KRP;LIN;LSBK;MBBC;MCRI;MDV;MO;MOD;MSGE;NGL;NHC;NPO;NSC;NUE;NWPX;OBK;OCC;OOMA;PBAM;PFG;PRM;R;REPX;RMBI;RSI;RXO;SRBK;STLD;STRL;SWBI;TDW;TSM;TWST;TXN;UNP;UVE;VECO;VPG;WAB;WBI;WERN;WSFS;WTTR
- signal EPS 本地补源覆盖: 69; unresolved: 9

### 2026-05-01

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=122
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=411; ibd_entry_close_position_invalid_or_non_signal=411; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=411; ibd_entry_date_invalid_or_non_signal=411; ibd_entry_price_invalid_or_non_signal=411; ibd_entry_reject_reason_valid_or_non_signal=411; ibd_entry_rule_invalid_or_non_signal=411; ibd_entry_volume_ratio_invalid_or_non_signal=411; ...+12
- repairable fallback: industry=15; sector=15
- optional gap: ibd_candidate_extra=396; pullback_v_is_dry=146; pullback_duration_weeks=103; pullback_pct=103; pullback_pct_off_peak=103
- signal EPS 缺失代码: AAPL;ACHC;ADM;AEE;AEP;AIT;AKR;ALL;ALRS;AROW;ATLO;AXSM;BAND;BEN;BOH;BPOP;BUUU;BWFG;CARE;CATY;CHEF;CNOB;COCO;CQP;CTO;DAC;DHT;DINO;DK;DTM;EBAY;ECO;ENVA;ESEA;ESQ;ET;ETON;EWBC;FANG;FCEL;FDX;FHB;FNRN;FPS;FRO;FRST;FTI;FUSB;FVR;GPRE;GSL;GTX;HR;HTB;IBKR;INDV;INSW;IRM;JAZZ;JCI;JMSB;KNSA;KO;KRG;KRP;LAMR;LION;LNT;MO;MPC;MSBI;MSGE;MSGS;MTX;MYFW;NGS;NHC;NMM;NREF;NVEC;NVGS;NVRI;NWBI;OSW;PAA;PBF;PCB;PEBO;PECO;PFIS;PGC;PHVS;PR;RBB;REPX;RMR;ROKU;RS;SHIP;SIF;SNDA;STNG;SUN;SUNC;TEN;THG;TNK;TPC;TRGP;TRMK;TRST;TVTX;TWLO;TXNM;ULS;VCTR;VLO;VTR;WBI;WCC;WELL;WERN
- signal EPS 本地补源覆盖: 106; unresolved: 16

### 2026-05-08

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=136
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=423; ibd_entry_breakout_range_ratio_invalid_or_non_signal=415; ibd_entry_close_position_invalid_or_non_signal=415; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=415; ibd_entry_date_invalid_or_non_signal=415; ibd_entry_price_invalid_or_non_signal=415; ibd_entry_rule_invalid_or_non_signal=415; ibd_entry_volume_ratio_invalid_or_non_signal=415; ...+12
- repairable fallback: industry=15; sector=15
- optional gap: ibd_candidate_extra=393; pullback_v_is_dry=137; pullback_duration_weeks=113; pullback_pct=113; pullback_pct_off_peak=113
- signal EPS 缺失代码: AAON;AAPL;ABCB;ACNB;ACT;ADM;AKR;ALX;AMAT;AMN;AMTB;APC;ASML;ASYS;BELFA;BLBD;BTSG;BVFL;CALY;CDP;CHEF;CHRN;CLYM;CNO;COCO;CON;CPF;CTRE;CVS;CW;CYTK;CZWI;DCOM;DVA;EBAY;ECBK;EGP;ESCA;ESEA;FAF;FBLA;FFIV;FISI;FRAF;FRO;FSBC;FSEA;FSTR;FTNT;GLW;GSL;GTX;GWW;H;HBCP;HBNC;HCSG;HFBL;HR;HST;HTB;HWBK;HWM;HXL;HZO;IBKR;IIPR;IRM;JAZZ;KNSA;KRYS;LIFE;LINC;LIND;LIVN;LRCX;MANU;MATX;MCRI;MCY;MDV;MEC;MITK;MKSI;MKTW;MNST;MOD;MPTI;MRNA;MRX;MS;NBIS;NHC;NMM;NPO;NVDA;NVEC;NWPX;ODC;OPLN;OSBC;PBT;PCB;PDFS;PEBK;PFIS;PLD;PRLB;RAPP;RBC;RBCAA;RFIL;RLYB;ROK;RPRX;SBLK;SMBC;SMTC;SNDA;SYRE;TKR;TMP;TRIN;TRVI;TSAT;TSN;TWLO;UFCS;ULS;VTRS;VVX;WABC;WES;WPC;WSBF;WT
- signal EPS 本地补源覆盖: 123; unresolved: 13

### 2026-05-15

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=59
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=441; ibd_entry_breakout_range_ratio_invalid_or_non_signal=438; ibd_entry_close_position_invalid_or_non_signal=438; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=438; ibd_entry_date_invalid_or_non_signal=438; ibd_entry_price_invalid_or_non_signal=438; ibd_entry_rule_invalid_or_non_signal=438; ibd_entry_volume_ratio_invalid_or_non_signal=438; ...+12
- repairable fallback: industry=16; sector=16
- optional gap: ibd_candidate_extra=427; pullback_v_is_dry=113; pullback_duration_weeks=63; pullback_pct=63; pullback_pct_off_peak=63
- signal EPS 缺失代码: AIZ;ALKS;ALL;APC;ARMK;BUUU;CAPL;COHR;CON;CQP;CVEO;CVLG;DCO;DVA;ELMD;EPD;ET;ETON;EXEL;FTNT;GL;HG;HLIO;JBHT;KNX;KRYS;LQDA;LTH;MO;MPC;MSGS;NGS;NPO;OCC;PAA;PAGP;PBF;PLGO;PM;PRM;REPX;ROIV;SBRA;STRZ;TFSL;THG;TRGP;TSEM;TXNM;UAN;UNM;UNP;VAL;VIK;VIRT;WBI;WCC;WES;WEYS
- signal EPS 本地补源覆盖: 51; unresolved: 8

### 2026-05-22

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=106
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=486; ibd_entry_close_position_invalid_or_non_signal=486; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=486; ibd_entry_date_invalid_or_non_signal=486; ibd_entry_price_invalid_or_non_signal=486; ibd_entry_rule_invalid_or_non_signal=486; ibd_entry_volume_ratio_invalid_or_non_signal=486; ibd_trigger_price_invalid_or_non_signal=486; ...+12
- repairable fallback: industry=18; sector=18
- optional gap: ibd_candidate_extra=462; pullback_v_is_dry=117; pullback_duration_weeks=97; pullback_pct=97; pullback_pct_off_peak=97
- signal EPS 缺失代码: ACMR;ACNB;AKR;ALAB;ALRS;ALX;ARCB;ARM;ATRO;BE;BNL;BNY;BOH;BPOP;BSRR;CIVB;CLBK;CLMT;CPBI;CRC;CRDO;CTBI;CURB;D;DAL;DCOM;DEA;ECBK;EDRY;ELMD;EVRG;EWBC;FAF;FCBC;FDX;FFBC;FHB;FISI;FRAF;FRO;FSBC;GL;GRDN;GS;H;HAFC;HBCP;HLT;HMN;HTB;IIPR;IVT;KLAC;KO;KRG;KRP;LIN;LION;LIVN;LPG;LQDA;LXFR;MANU;MCY;MITK;MNSB;MOV;NMM;NTAP;NVRI;NWBI;OBK;OFG;OVBC;PCB;PDLB;PEBO;PECO;PFG;PFIS;PKBK;PLD;PLPC;PLSE;PRI;PSMT;R;RBCAA;REPX;ROIV;ROST;RXO;SIRI;SNDR;STBA;TFSL;TIGO;TMP;UHT;UNM;UTMD;WABC;WBI;WPC;WSFS;XHR
- signal EPS 本地补源覆盖: 98; unresolved: 8

### 2026-05-29

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=99
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=463; ibd_entry_close_position_invalid_or_non_signal=463; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=463; ibd_entry_date_invalid_or_non_signal=463; ibd_entry_price_invalid_or_non_signal=463; ibd_entry_rule_invalid_or_non_signal=463; ibd_entry_volume_ratio_invalid_or_non_signal=463; ibd_trigger_price_invalid_or_non_signal=463; ...+12
- repairable fallback: industry=17; sector=17
- optional gap: ibd_candidate_extra=435; pullback_v_is_dry=126; pullback_duration_weeks=90; pullback_pct=90; pullback_pct_off_peak=90
- signal EPS 缺失代码: AAON;ALNT;AMAT;ARCB;ARM;ATI;ATLC;AVBP;BFH;BJRI;BUSE;BUUU;CATY;CBL;CNOB;COHU;CRDO;CRS;CSTM;CSWC;CURB;DAL;DCOM;DFTX;DIOD;DNTH;ELA;FCF;FDX;FISI;FLEX;FRST;FSBC;FUSB;GH;GHM;H;HBNC;HFBL;HLIO;HLT;HWBK;IIPR;ILMN;ITRN;JBHT;KFRC;KRG;LIND;LLY;LOCO;LRCX;LSCC;LTH;MEC;MITK;MKSI;MNST;MOD;MOG-A;MOV;MPTI;MSGE;NBIX;NVEC;NWPX;OCC;PDFS;PDLB;PFIS;PKOH;PRAX;RAPP;RHP;ROKU;RS;RVMD;SBLK;SHBI;SNDK;SNX;SXI;THFF;TKR;TMP;TVTX;TWST;UMAC;URI;UTMD;VABK;VIK;VVX;WAFD;WDC;WEYS;WSBF;WULF;XHR
- signal EPS 本地补源覆盖: 90; unresolved: 9

### 2026-06-05

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=121
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=495; ibd_entry_close_position_invalid_or_non_signal=495; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=495; ibd_entry_date_invalid_or_non_signal=495; ibd_entry_price_invalid_or_non_signal=495; ibd_entry_rule_invalid_or_non_signal=495; ibd_entry_volume_ratio_invalid_or_non_signal=495; ibd_trigger_price_invalid_or_non_signal=495; ...+12
- repairable fallback: industry=17; sector=17
- optional gap: ibd_candidate_extra=453; pullback_v_is_dry=130; pullback_duration_weeks=104; pullback_pct=104; pullback_pct_off_peak=104
- signal EPS 缺失代码: AFL;AIT;AKR;ALL;ALNT;ALX;AMG;AMN;APGE;APLE;AROW;ATLO;AXGN;BFH;BHB;BNL;BPOP;BRX;BSRR;BVFL;BWFG;BY;C;CASS;CBK;CDP;CIVB;CNOB;CNS;COSO;CPF;CSX;DINO;DK;DOCN;DTM;EFSI;EIG;ENVA;ET;FBP;FCCO;FCF;FFBC;FHI;FMBH;FRD;FRST;FSEA;FTK;FVR;FXNC;GABC;GL;GRDN;GSBC;GWW;HMN;HNGE;HYNE;IESC;INDV;INGM;IVT;JMSB;KFRC;KIM;KRP;LGND;LIN;LQDT;MAC;MCBS;MCHB;MCRI;MEC;MPC;MRX;MSBI;MYE;NBIX;NODK;NTB;NTRS;OBK;ODC;OSBC;OSCR;PBFS;PCB;PEBK;PFIS;PGC;PLSE;PTRN;R;RBC;REPX;RMBI;RMR;RNR;SEPN;SPG;SPHR;SRCE;STBA;STT;SXI;TBBB;TMP;TWIN;TWLO;UNFI;UNP;URI;VLO;VOYA;VSXY;WABC;WSBF;XPO
- signal EPS 本地补源覆盖: 110; unresolved: 11

### 2026-06-12

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=292
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=513; ibd_entry_close_position_invalid_or_non_signal=513; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=513; ibd_entry_date_invalid_or_non_signal=513; ibd_entry_price_invalid_or_non_signal=513; ibd_entry_rule_invalid_or_non_signal=513; ibd_entry_volume_ratio_invalid_or_non_signal=513; ibd_trigger_price_invalid_or_non_signal=513; ...+12
- repairable fallback: industry=17; sector=17
- optional gap: ibd_candidate_extra=408; pullback_v_is_dry=188; pullback_duration_weeks=180; pullback_pct=180; pullback_pct_off_peak=180
- signal EPS 缺失代码: ABCB;ACLS;ACNB;AIP;AIR;AIT;AIZ;AMAL;AMBQ;AMG;AMKR;AMRX;AMTB;ANRO;ARMK;AROW;ASB;ASH;ASML;ASYS;ATLC;AVBC;AVBH;AXSM;AZZ;BAP;BATRA;BBT;BDL;BHB;BHE;BJRI;BLX;BNL;BOH;BOTJ;BPOP;BPRN;BRX;BSRR;BUSE;BY;C;CAC;CAKE;CALY;CASS;CASY;CATY;CBAN;CCNE;CDP;CFR;CHCO;CHEF;CLBK;CLDX;CNOB;CNS;COCO;COHU;CON;COSO;CPF;CRDO;CSTM;CSX;CTBI;CTO;CTRN;CVS;CW;CXW;D;DAVE;DCO;DCOM;DVA;DXPE;ECBK;EFSC;EGP;ELA;ELVN;ENVA;ESE;ESI;ETON;EWBC;EXEL;EXPD;FBIZ;FBLA;FBP;FCBC;FCCO;FCF;FDSB;FFBC;FFIV;FHB;FHI;FMBH;FNLC;FNRN;FORM;FR;FRAF;FRO;FRST;FSBC;FULT;FUNC;FVCB;FVR;FXNC;GABC;GRC;GRDN;HAFC;HBCP;HBNC;HBT;HCSG;HFWA;HLT;HMN;HR;HWBK;HWM;HXL;IBKR;ICHR;IDA;IFS;INCY;INSW;IVT;JBL;KALU;KIM;KLAC;KLIC;KO;KRG;KRYS;LARK;LAUR;LCNB;LIN;LIND;LIVN;LOCO;LPG;LQDA;LSBK;LTH;MAC;MATX;MBWM;MCHB;MCS;MET;MITK;MNSB;MOV;MRX;MTX;MYE;MYFW;NBBK;NHC;NMM;NNN;NPO;NVMI;NVRI;NWBI;NWFL;OBK;OBT;OCC;OFG;OII;ONTO;ORRF;OSBC;OSW;OVBC;PBAM;PCB;PDFS;PEB;PEBK;PEBO;PECO;PFBC;PFIS;PFS;PKBK;PKE;PKOH;PLBC;PLD;PLGO;PLOW;PLPC;PLXS;PNW;PRI;PRM;PRSU;PSMT;PSTL;PTGX;QCRH;RBB;RBC;REG;RFIL;RMBI;RNR;RNST;ROIV;ROK;ROKU;ROST;RSI;SENEA;SEPN;SFNC;SHBI;SHOO;SI;SIF;SILC;SKT;SLAB;SMBC;SMBK;SMTC;SN;SNA;SNDK;SPHR;SPNT;SRBK;SRCE;STBA;STNG;SXT;SYRE;TBBB;TCBK;TGTX;THFF;THG;TIGO;TMP;TNGX;TRMK;TRNO;TROW;TRVI;TSBK;TSM;TTMI;TVTX;TWIN;UBSI;UCTT;UFCS;UMBF;UVE;UVSP;VABK;VECO;VIRT;VTRS;WAFD;WBI;WBS;WELL;WEYS;WKC;WPC;WSBF;WSFS;WSM;WTS;WULF;ZION
- signal EPS 本地补源覆盖: 270; unresolved: 22

### 2026-06-18

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=119
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=549; ibd_entry_breakout_range_ratio_invalid_or_non_signal=514; ibd_entry_close_position_invalid_or_non_signal=514; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=514; ibd_entry_date_invalid_or_non_signal=514; ibd_entry_price_invalid_or_non_signal=514; ibd_entry_rule_invalid_or_non_signal=514; ibd_entry_volume_ratio_invalid_or_non_signal=514; ...+12
- repairable fallback: industry=18; sector=18
- optional gap: ibd_candidate_extra=487; pullback_v_is_dry=164; pullback_duration_weeks=82; pullback_pct=82; pullback_pct_off_peak=82
- signal EPS 缺失代码: ACA;ACLS;ADI;AEHR;AIP;AIR;ALH;ALOT;AMD;AMKR;ANET;ARWR;ASH;ASX;ATRO;AVT;AXGN;AZZ;BAP;BCAX;BE;BEN;BHE;BJRI;BLBD;BPRN;BVFL;CALY;CAT;CBNA;CCBG;CHRN;CLDX;CMI;CR;CTRN;CW;CYTK;D;DAL;DAVE;DIOD;DNLI;DXPE;ECBK;ECPG;EIG;ESE;ESI;ETN;FCEL;FDSB;FLXS;FLYW;FORM;FRO;FSTR;FTDR;FUSB;GE;GFS;GHM;GL;GLW;GTX;HCSG;HFBL;HTB;HYNE;ICHR;IFS;IMAX;INTC;IRM;KEYS;KLIC;KNSA;LION;LOCO;LSCC;LTH;MCY;MKSI;MOD;MRNA;MRX;MTSI;MYRG;NBIS;NVRI;NVT;ONTO;ORKA;PANW;PBFS;PLSE;PTGX;RBC;ROIV;ROK;RVMD;SLAB;SN;SPHR;STNG;SWBI;TER;TGTX;TRV;TRVI;TSEM;TSM;TTMI;TXN;URGN;VIK;VSXY;WAB;WULF
- signal EPS 本地补源覆盖: 102; unresolved: 17

### 2026-06-26

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=239
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=636; ibd_entry_breakout_range_ratio_invalid_or_non_signal=463; ibd_entry_close_position_invalid_or_non_signal=463; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=463; ibd_entry_date_invalid_or_non_signal=463; ibd_entry_price_invalid_or_non_signal=463; ibd_entry_rule_invalid_or_non_signal=463; ibd_entry_volume_ratio_invalid_or_non_signal=463; ...+12
- repairable fallback: industry=21; sector=21
- optional gap: ibd_candidate_extra=523; pullback_v_is_dry=170; pullback_duration_weeks=152; pullback_pct=152; pullback_pct_off_peak=152
- signal EPS 缺失代码: ABCB;ACA;ACT;ACU;ADPT;AEE;AEP;AFL;AGX;ALL;ALMR;ALX;AMAL;AMN;AMTB;APGE;AROC;AROW;ASB;ASND;ATLO;AUB;AVBP;BAC;BATRA;BCAL;BCAX;BFS;BFST;BLZE;BNL;BOH;BOKF;BRX;BSVN;BUUU;BVFL;BWB;BWFG;BZH;CAC;CASS;CB;CBK;CBL;CBNK;CCBG;CCXI;CFG;CFR;CGEM;CHCO;CINF;CLBK;CLYM;CNA;CPBI;CPF;CSX;CTBI;CUBI;CVS;CYTK;CZFS;CZWI;DEA;DFTX;DGII;DK;DNTH;DOC;DTM;EG;EGP;EIG;ELMD;ELVN;EQBK;ETON;ETSY;EVRG;EWTX;FBIZ;FBNC;FCBC;FCCO;FHB;FITB;FLYW;FNB;FNLC;FRAF;FRME;FTNT;FULT;FXNC;GABC;GFF;GHRS;GLW;GTES;HAFC;HBCP;HBNC;HCSG;HFWA;HG;HWBK;HWC;HZO;IDA;IIPR;ILMN;INCY;INDV;IRM;JMSB;JNJ;KGS;KNSA;KO;KRT;KYMR;LBRX;LCNB;LIFE;LINC;LLY;LNT;LQDT;LYV;MAC;MANE;MBWM;MBX;MCBS;MCHB;MCY;MITK;MNSB;MO;MPC;MRK;MSGE;MSGS;MTB;MYFW;NAVN;NDSN;NESR;NHC;NIC;NJR;NNN;NTCT;NTRA;NTST;NWFL;OII;ONB;OPLN;ORRF;OSBC;OUT;PECO;PFBC;PFGC;PFS;PGC;PHVS;PK;PKBK;PLGO;PLOW;PLPC;PLSE;PNC;PNW;PRI;PSNL;PSTL;PTRN;QCRH;RAPP;RBB;RCUS;RDVT;REG;RFIL;RMBI;RNR;RPRX;SBFG;SBRA;SENEA;SFNC;SI;SMBK;SNA;SNDA;SPG;SPNT;SRBK;SXT;TCBK;THG;TILE;TMP;TRMK;TRV;TSBK;UBSI;UFCS;UHT;UMBF;UNP;UNTY;URI;USB;USCB;UTMD;UVE;VABK;VAC;VLO;VTR;VTRS;WAFD;WASH;WELL;WHG;WRLD;WSBC;WSBF;WSFS;WTBA;WTTR;XENE;ZION
- signal EPS 本地补源覆盖: 214; unresolved: 25

### 2026-07-02

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=129
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=656; ibd_entry_close_position_invalid_or_non_signal=656; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=656; ibd_entry_date_invalid_or_non_signal=656; ibd_entry_price_invalid_or_non_signal=656; ibd_entry_rule_invalid_or_non_signal=656; ibd_entry_volume_ratio_invalid_or_non_signal=656; ibd_trigger_price_invalid_or_non_signal=656; ...+12
- repairable fallback: industry=21; sector=21
- optional gap: ibd_candidate_extra=601; pullback_v_is_dry=184; pullback_duration_weeks=130; pullback_pct=130; pullback_pct_off_peak=130
- signal EPS 缺失代码: AAPL;ABBV;ACHC;AFL;AHR;ARWR;ASH;ASX;AWR;AXGN;BFC;BLZE;BMRC;BNY;BSET;CB;CBNA;CCXI;CDP;CHE;CLDX;CLMT;CNA;CNS;CORT;CRL;CRWD;CSX;CTRE;CW;DDOG;DEA;DGX;DINO;DK;EBAY;ECO;EG;ESQ;ET;EXEL;EXPD;EZPW;FAF;FCEL;FFIV;FLYW;FROG;GD;GH;GOOG;HCSG;HEI;HFWA;IBTA;IDA;INCY;ITIC;JAN;JAZZ;JMSB;JNJ;KFRC;KO;LAMR;LGND;LIN;LLYVA;LYV;MANU;MATX;MBWM;MCHB;MD;MET;MKTW;MPB;MPC;MRK;MVBF;NBIX;NBTB;NIC;NNN;NREF;NSC;NTRA;NTRS;NTST;NUTX;NVRI;OFG;OHI;OOMA;PACS;PKE;PNTG;PRI;SBFG;SBRA;SENEA;SEZL;SIF;SIRI;SLDE;SNDA;SPNT;SRBK;SUN;SUNC;SXT;TCBK;TRIN;TRNO;TROW;UHT;UNP;USFD;UVE;VLO;VOYA;VSEC;VTR;VTRS;WELL;WERN;WEYS;WT;XENE
- signal EPS 本地补源覆盖: 116; unresolved: 13

### 2026-07-10

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=85
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=664; ibd_entry_close_position_invalid_or_non_signal=664; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=664; ibd_entry_date_invalid_or_non_signal=664; ibd_entry_price_invalid_or_non_signal=664; ibd_entry_rule_invalid_or_non_signal=664; ibd_entry_volume_ratio_invalid_or_non_signal=664; ibd_trigger_price_invalid_or_non_signal=664; ...+12
- repairable fallback: industry=24; sector=24
- optional gap: ibd_candidate_extra=623; pullback_v_is_dry=139; pullback_duration_weeks=73; pullback_pct=73; pullback_pct_off_peak=73
- signal EPS 缺失代码: AAMI;AAPL;ADM;ALH;ALMR;AMG;ANDG;ANET;BAND;BBIO;BSVN;BZH;CGON;CLMT;CNXN;COAG;DELL;DINO;DNTH;EBAY;ECO;ECPG;ET;EWTX;EXPD;FHI;FR;FTH;FTK;GD;GPRE;GWW;HPE;ICHR;INSW;JPM;LIFE;LINC;MET;MKTW;MNPR;MOD;MPC;MRX;MS;NESR;NET;NJR;NLY;NMM;NREF;NSC;NTCT;OII;ONTO;PAA;PAG;PAGP;PBF;PBI;PENG;PFG;PHVS;PNTG;PSX;RAPP;RF;RGA;SBLK;SEIC;SION;SLAB;STNG;SYRE;TEN;TIGO;TNK;TRAX;TRNO;TWLO;VCTR;VIRT;WBI;WCC;WT
- signal EPS 本地补源覆盖: 70; unresolved: 15

### 2026-07-17

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=181
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=647; ibd_entry_close_position_invalid_or_non_signal=647; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=647; ibd_entry_date_invalid_or_non_signal=647; ibd_entry_price_invalid_or_non_signal=647; ibd_entry_rule_invalid_or_non_signal=647; ibd_entry_volume_ratio_invalid_or_non_signal=647; ibd_trigger_price_invalid_or_non_signal=647; ...+12
- repairable fallback: industry=25; sector=25
- optional gap: ibd_candidate_extra=575; pullback_v_is_dry=112; pullback_duration_weeks=93; pullback_pct=93; pullback_pct_off_peak=93
- signal EPS 缺失代码: ABCB;ACHC;ACT;ADM;ADPT;AFL;AGM;AHR;AIT;AKR;ALRS;ALX;AMAL;APC;ARCB;ASB;ASH;ATLO;BATRA;BOH;BOKF;BOTJ;BRX;BSRR;BUSE;BWFG;BY;CALY;CBL;CCK;CDNA;CFR;CHCO;CHMG;CHRN;CLBK;CNO;CNS;CNXN;COAG;CRNX;CSWC;CTBI;CTO;CTRE;CTRN;CURB;CVEO;CVLG;CVS;CZFS;DINO;DKL;DNTH;EAT;EGP;EPD;EPR;EQBK;ESNT;ESQ;ET;EWBC;FA;FAF;FBIZ;FBP;FBRX;FCF;FLYW;FMBH;FR;FRME;FRT;FULT;FVCB;GABC;GEF;GHRS;GL;GPRE;HBNC;HFWA;HIW;HR;HST;HWC;HXL;IOR;IRM;ITIC;IVT;IVZ;JMSB;JXN;KARO;KELYA;KFY;KIM;KRG;LAMR;LCNB;LQDT;LTC;MATX;MCS;MDV;MO;MPB;MPC;MSBI;MSGS;MTB;MVBF;NBTB;NET;NHC;NTB;NWBI;OBK;OHI;ONB;ORKA;OSBC;PAG;PARR;PBAM;PEBO;PECO;PFG;PFIS;PFS;PK;PKOH;PLBC;PLD;PLSE;PM;PSX;QCRH;REG;RHI;RLYB;RNR;RNST;RUSHA;RXO;SCSC;SEIC;SION;SKT;SMBC;SMBK;SNDR;SPG;SRBK;SUN;SUNC;TCBK;TOWN;TRGP;TRMK;TRVI;UBSI;UMBF;UNF;UNTY;USCB;VAC;VTRS;VVX;WABC;WASH;WERN;WES;WPC;WSFS;WT;WTTR;XHR;XMTR
- signal EPS 本地补源覆盖: 165; unresolved: 16

### 2026-07-24

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=114
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=696; ibd_entry_close_position_invalid_or_non_signal=696; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=696; ibd_entry_date_invalid_or_non_signal=696; ibd_entry_price_invalid_or_non_signal=696; ibd_entry_rule_invalid_or_non_signal=696; ibd_entry_volume_ratio_invalid_or_non_signal=696; ibd_trigger_price_invalid_or_non_signal=696; ...+12
- repairable fallback: industry=24; sector=24
- optional gap: ibd_candidate_extra=647; pullback_v_is_dry=115; pullback_duration_weeks=74; pullback_pct=74; pullback_pct_off_peak=74
- signal EPS 缺失代码: ACNB;ACU;AIT;ALL;ALOT;AME;AMRX;ARWR;ATI;AVBC;AVT;BFST;BHB;BLFS;BMY;BPRN;BUSE;CARE;CB;CBAN;CCBG;CCNE;CDP;CFBK;CHMG;CR;CRS;CTRE;CZFS;CZNC;DAC;DELL;DGX;ECO;EDRY;EPD;EPR;ESEA;FBLA;FCBC;FISI;FLXS;FMNB;FRBA;FXNC;GEF;GSL;HALO;HBCP;HG;HST;HWM;IBCP;IMAX;INSW;IOR;IRM;JAZZ;JNJ;JPM;LARK;LBRX;LH;LPG;LQDA;MCS;MMM;MOG-A;MPLX;MRK;MSBI;MTG;NMM;NVEC;NWFL;OFG;OHI;ORRF;PAA;PAGP;PBFS;PBT;PCAR;PKG;PLSE;QCRH;RBCAA;RCBC;RHP;RNR;ROST;RS;RYZ;SAFT;SBLK;SBRA;SFNC;SHBI;SMBK;SPG;STBA;SXT;THG;THRM;TMP;TRST;URI;USCB;VIK;VVX;WAB;WES;WPC;WRLD
- signal EPS 本地补源覆盖: 103; unresolved: 11

### 2026-07-31

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=36
- 正常空值: ibd_entry_reject_reason_valid_or_non_signal=242; ibd_entry_breakout_range_ratio_invalid_or_non_signal=240; ibd_entry_close_position_invalid_or_non_signal=240; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=240; ibd_entry_date_invalid_or_non_signal=240; ibd_entry_price_invalid_or_non_signal=240; ibd_entry_rule_invalid_or_non_signal=240; ibd_entry_volume_ratio_invalid_or_non_signal=240; ...+12
- repairable fallback: industry=10; sector=10
- optional gap: ibd_candidate_extra=233; pullback_v_is_dry=41; pullback_duration_weeks=28; pullback_pct=28; pullback_pct_off_peak=28
- signal EPS 缺失代码: ANET;ARXS;ATLC;CCNE;CNK;DYN;ECPG;ESEA;ETON;FISI;INCY;KO;LLYVA;LPG;MMM;MPB;NET;NTAP;NUE;OSBC;PESI;PKOH;RDVT;RNG;ROST;RTX;SCHW;SHBI;SHIP;SION;SNOW;THFF;TOWN;UVE;VTRS;WTBA
- signal EPS 本地补源覆盖: 33; unresolved: 3

### 2026-08-07

- 状态: `failed`
- 需要补充/修复: eps_yoy_growth_signal=179
- 正常空值: ibd_entry_breakout_range_ratio_invalid_or_non_signal=716; ibd_entry_close_position_invalid_or_non_signal=716; ibd_entry_close_vs_trigger_pct_invalid_or_non_signal=716; ibd_entry_date_invalid_or_non_signal=716; ibd_entry_price_invalid_or_non_signal=716; ibd_entry_rule_invalid_or_non_signal=716; ibd_entry_volume_ratio_invalid_or_non_signal=716; ibd_trigger_price_invalid_or_non_signal=716; ...+12
- repairable fallback: industry=25; sector=25
- optional gap: ibd_candidate_extra=655; pullback_v_is_dry=122; pullback_duration_weeks=95; pullback_pct=95; pullback_pct_off_peak=95
- signal EPS 缺失代码: AAMI;ABNB;ACHC;AIR;AIRT;AIT;ALH;ALNT;ALX;AME;AMGN;AMKR;AMN;ANET;ARTNA;ASH;ATEX;ATI;ATLO;ATRO;AVT;AXGN;BAC;BDL;BFC;BHB;BLZE;BOKF;BUUU;BVS;CATY;CBAN;CCXI;CFFI;CFG;CGON;CIX;CLDX;CNO;CON;COSO;CPAY;CPBI;CRL;CRNX;CSWC;CUBI;DAL;DBX;DCO;DELL;DGII;DGX;DIOD;DXPE;ECBK;EDRY;ESCA;ESNT;ETN;EXPD;EXPE;FA;FAST;FBLA;FBNC;FET;FHI;FLXS;FNLC;FRD;FROG;FRST;FSEA;FTDR;FTK;GE;GFF;GH;GRMN;HBB;HBCP;HEI;HNGE;HPE;HVT;HWBK;IBTA;IESC;IOSP;IVZ;JCI;JHX;JMSB;JXN;KALU;KEYS;LFUS;LIN;LIND;LQDT;MD;MET;MPTI;MTRN;MTUS;MTX;NAVN;NDSN;NEO;NESR;NET;NEU;NKSH;NLY;NREF;NTRA;NUE;NVEC;NVT;PANW;PH;PKG;PLPC;PRAX;PRLB;PTGX;RAPP;RGA;RHI;RJF;ROIV;ROKU;RS;RVMD;SAFT;SENEA;SEPN;SHOO;SIF;SION;SNA;SNOW;SSB;SWK;SXI;SXT;TCBX;TECH;TILE;TPC;TRIN;TSAT;TVTX;TWIN;TWLO;TWST;UFCS;UNM;URI;USFD;UTMD;VABK;VAC;VCTR;VIK;VSXY;WBS;WCC;WEYS;WRLD;WSBF;WSFS;WSM;WT;WTBA;WTS;WTTR;ZION
- signal EPS 本地补源覆盖: 158; unresolved: 21
