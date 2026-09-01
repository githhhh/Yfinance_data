# Track B: Breaking B0 Ranking + Top3 Selection - Rigorous Research Report

## 1. Research Protocol & Infrastructure Integrity
- **Protocol Integrity**: Fully decoupled 3-phase execution (`train` -> `lock` -> `validate`).
- **B0 Production Reproduction**: **100.0%** exact match (42/42 snapshot dates identical to `dashboard/skill_industry_eps_known.py`).
- **Sealed Validation**: Exactly 5 shortlisted challengers were locked prior to evaluating validation period.
- **Code & Panel Hashes**:
  - Code Hash: `d596cddc1c46e6dad1b74fadcca4a6745a4d17b38d0ff93f8415345e0da5e4a3`
  - Panel Hash: `6886c8d721b808476aa8491bb2140e28a7f96fb07ad6e01f313aec06f5e17c24`
  - Git SHA: `aea015989601c1d7ade5a310a51521c08bafddf9`

## 2. Dry-Policy & Top3 Selector Controlled Experiment (Train Period)

### Train-Only Dry-Policy Outcome Matrix
| dry_policy | selector | challenger_id | horizon | train_picks_count | mature_weeks | mean_return_pct | median_return_pct | cvar10_pct | stop8_rate_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| symmetric | distinct_1 | B0_ORIGINAL__distinct_1 | W4 | 64 | 14 | 5.664 | 2.711 | -18.931 | 31.25 |
| symmetric | pure_top3 | B0_ORIGINAL__pure_top3 | W4 | 66 | 15 | 4.786 | 2.085 | -21.308 | 33.33 |
| symmetric | max_2_per_ind | B0_ORIGINAL__max_2_per_ind | W4 | 66 | 15 | 4.786 | 2.085 | -21.308 | 33.33 |
| reward_only | distinct_1 | B0_DRY_REWARD_ONLY__distinct_1 | W4 | 64 | 14 | 5.664 | 2.711 | -18.931 | 31.25 |
| reward_only | pure_top3 | B0_DRY_REWARD_ONLY__pure_top3 | W4 | 66 | 15 | 4.786 | 2.085 | -21.308 | 33.33 |
| reward_only | max_2_per_ind | B0_DRY_REWARD_ONLY__max_2_per_ind | W4 | 66 | 15 | 4.786 | 2.085 | -21.308 | 33.33 |
| ignored | distinct_1 | B0_DRY_IGNORED__distinct_1 | W4 | 64 | 14 | 5.343 | 2.711 | -18.931 | 31.25 |
| ignored | pure_top3 | B0_DRY_IGNORED__pure_top3 | W4 | 66 | 15 | 4.475 | 2.085 | -21.308 | 33.33 |
| ignored | max_2_per_ind | B0_DRY_IGNORED__max_2_per_ind | W4 | 66 | 15 | 4.475 | 2.085 | -21.308 | 33.33 |


### Case Study: CRWD (Snapshot: 2026-07-02, Rule: pivot, pullback_v_is_dry: False)
| Metric / Attribute | B0_ORIGINAL (Symmetric Penalty) | B0_DRY_REWARD_ONLY (Reward Only) |
| :--- | :--- | :--- |
| **Reason Codes** | `geometry_caution_not_failure, volume_confirms_breakout, eps_acceleration_support, near_52w_high, pullback_structure` | `geometry_caution_not_failure, volume_confirms_breakout, eps_acceleration_support, near_52w_high, pullback_structure` |
| **Risk Codes** | `non_actionable_radar_only, extended_from_buy_point, pullback_not_dry` | `non_actionable_radar_only, extended_from_buy_point` |
| **Pullback Penalty Applied** | `pullback_not_dry` in risk_codes | **None** (neutral) |
| **Sort Key Prefix (Failure, Lane, Status, -(Ev-Risk), Risk)** | `(0, 3, 1, 0, 3)` | `(0, 3, 1, -1, 2)` |


## 3. Locked Challengers Evaluation & Champion Classification

### Champion Classification Matrix
| selector_id | classification | train_w4_med_spread | train_w4_mean_spread | train_w4_cvar_delta | train_w4_stop_delta | val_w4_med_spread | val_w4_mean_spread | val_w4_cvar_delta | val_w4_stop_delta | val_support_weeks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0_DRY_REWARD_ONLY__distinct_1 | DOMINATES B0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 9 |
| signal_f1_lgbm_w4_distinct_industry | UNSTABLE | 10.266633333333337 | 13.905671428571427 | -0.6344666666666665 | 19.047619047619047 | -22.4117 | -16.917096296296297 | -16.80763333333333 | 62.96296296296296 | 9 |
| actionable_f1_lgbm_w4_distinct_industry | UNSTABLE | 4.2655666666666665 | 2.5287952380952388 | 6.972766666666667 | 4.761904761904762 | -7.224866666666666 | -4.809351851851852 | -11.329233333333336 | 33.33333333333333 | 9 |
| signal_agent_ridge_w1_portfolio_aware | UNSTABLE | 8.0312 | 8.541414285714286 | 2.9119 | 14.285714285714285 | -2.147616666666668 | -7.297662499999999 | -14.264733333333329 | 29.166666666666664 | 8 |
| actionable_agent_lgbm_w4_distinct_industry | UNSTABLE | 4.2655666666666665 | 2.0246476190476193 | 4.047133333333334 | 4.761904761904762 | -6.833966666666669 | -8.128277777777777 | -6.833966666666669 | 37.03703703703704 | 9 |


### Validation Paired Tail Metrics vs B0 (Identical Common Support)
| selector_id | segment | horizon | support_weeks | challenger_mean | b0_mean | mean_spread | challenger_median | b0_median | median_spread | challenger_cvar10 | b0_cvar10 | cvar_delta | challenger_p10 | b0_p10 | challenger_top10_mean | b0_top10_mean | challenger_tail_ratio10 | b0_tail_ratio10 | challenger_stop_rate_pct | b0_stop_rate_pct | stop_delta_pct | challenger_one_pick_ruins_pct | b0_one_pick_ruins_pct | one_pick_ruins_delta_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0_DRY_REWARD_ONLY__distinct_1 | contaminated_validation | W4 | 9 | 2.0072925925925924 | 2.0072925925925924 | 0.0 | 3.7727333333333335 | 3.7727333333333335 | 0.0 | -16.656766666666666 | -16.656766666666666 | 0.0 | -6.465726666666667 | -6.465726666666667 | 14.597166666666666 | 14.597166666666666 | 0.8763505522280235 | 0.8763505522280235 | 18.51851851851852 | 18.51851851851852 | 0.0 | 33.333333333333336 | 33.333333333333336 | 0.0 |
| signal_f1_lgbm_w4_distinct_industry | contaminated_validation | W4 | 9 | -14.909803703703703 | 2.0072925925925924 | -16.917096296296297 | -19.0267 | 3.7727333333333335 | -22.4117 | -33.4644 | -16.656766666666666 | -16.80763333333333 | -31.55309333333333 | -6.465726666666667 | 27.895833333333332 | 14.597166666666666 | 0.8335972954343521 | 0.8763505522280235 | 81.48148148148148 | 18.51851851851852 | 62.96296296296296 | 0.0 | 33.333333333333336 | -33.33333333333333 |
| actionable_f1_lgbm_w4_distinct_industry | contaminated_validation | W4 | 9 | -2.802059259259259 | 2.0072925925925924 | -4.809351851851852 | -4.900033333333334 | 3.7727333333333335 | -7.224866666666666 | -27.986 | -16.656766666666666 | -11.329233333333336 | -20.341146666666667 | -6.465726666666667 | 26.991733333333332 | 14.597166666666666 | 0.9644727125467496 | 0.8763505522280235 | 51.85185185185185 | 18.51851851851852 | 33.33333333333333 | 0.0 | 33.333333333333336 | -33.33333333333333 |
| signal_agent_ridge_w1_portfolio_aware | contaminated_validation | W4 | 8 | -5.511049999999999 | 1.7866124999999995 | -7.297662499999999 | -1.4655333333333338 | 2.8647 | -2.147616666666668 | -30.921499999999995 | -16.656766666666666 | -14.264733333333329 | -23.08094 | -7.739606666666667 | 16.79296666666667 | 14.597166666666666 | 0.5430838305601822 | 0.8763505522280235 | 50.0 | 20.833333333333332 | 29.166666666666664 | 20.0 | 33.333333333333336 | -13.33333333333333 |
| actionable_agent_lgbm_w4_distinct_industry | contaminated_validation | W4 | 9 | -6.120985185185185 | 2.0072925925925924 | -8.128277777777777 | -9.457433333333334 | 3.7727333333333335 | -6.833966666666669 | -23.490733333333335 | -16.656766666666666 | -6.833966666666669 | -19.442093333333336 | -6.465726666666667 | 24.2399 | 14.597166666666666 | 1.0318920084799397 | 0.8763505522280235 | 55.55555555555555 | 18.51851851851852 | 37.03703703703704 | 0.0 | 33.333333333333336 | -33.33333333333333 |


## 4. Provenance Details

```json
{
  "code_hash": "d596cddc1c46e6dad1b74fadcca4a6745a4d17b38d0ff93f8415345e0da5e4a3",
  "panel_hash": "6886c8d721b808476aa8491bb2140e28a7f96fb07ad6e01f313aec06f5e17c24",
  "git_sha": "aea015989601c1d7ade5a310a51521c08bafddf9",
  "locked_challenger_ids": [
    "B0_DRY_REWARD_ONLY__distinct_1",
    "signal_f1_lgbm_w4_distinct_industry",
    "actionable_f1_lgbm_w4_distinct_industry",
    "signal_agent_ridge_w1_portfolio_aware",
    "actionable_agent_lgbm_w4_distinct_industry"
  ]
}
```
