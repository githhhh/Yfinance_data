# Track B: Breaking B0 Ranking + Top3 Selection - Rigorous Research Report

## 1. Research Protocol & Infrastructure Integrity
- **Protocol Integrity**: Decoupled 3-phase execution (`train` -> `lock` -> `validate`).
- **B0 Production Reproduction**: **100.0%** exact match (42/42 snapshot dates identical between current production replay and historical panel `is_b0`). Audit log saved to `output/b0_historical_reproduction.csv`.
- **Sealed Validation**: Exactly 5 shortlisted challengers were locked prior to evaluating validation period.
- **Code & Panel Hashes**:
  - Codebase Hash: `6c97726eb83500bb5d9b5a3c8d4c7595c2390f88a2b1732ac68cd5b9689a60fd`
  - Dependency Hashes:
    - Challenge Package: `7456aace115d3a162a224419b07166939d4ebcf96d6de3bdee681097ae4ca279`
    - Production B0: `115387c9861f7202c0f6b3c89fe2d2ff594544de93264901ee6d2f72e930c477`
  - Panel Hash: `6886c8d721b808476aa8491bb2140e28a7f96fb07ad6e01f313aec06f5e17c24`
  - Git SHA: `dcbf7eb250f90880f65cf4769b20bed7d0862cdd` (git_dirty: `True`, code_dirty: `False`)

## 2. Dry-Policy & Top3 Selector Controlled Experiment (Train Period)

### Train-Only Dry-Policy Outcome Matrix (Selection-First Mature Portfolio Weeks)
| dry_policy | selector | challenger_id | horizon | train_picks_count | mature_weeks | mean_return_pct | median_return_pct | cvar10_pct | stop8_rate_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| symmetric | distinct_1 | B0_ORIGINAL__distinct_1 | W4 | 64 | 14 | 5.4448 | 3.5174 | -8.8733 | 30.95 |
| symmetric | pure_top3 | B0_ORIGINAL__pure_top3 | W4 | 66 | 15 | 5.2692 | 5.6993 | -8.8733 | 31.11 |
| symmetric | max_2_per_ind | B0_ORIGINAL__max_2_per_ind | W4 | 66 | 15 | 5.2692 | 5.6993 | -8.8733 | 31.11 |
| reward_only | distinct_1 | B0_DRY_REWARD_ONLY__distinct_1 | W4 | 64 | 14 | 5.4448 | 3.5174 | -8.8733 | 30.95 |
| reward_only | pure_top3 | B0_DRY_REWARD_ONLY__pure_top3 | W4 | 66 | 15 | 5.2692 | 5.6993 | -8.8733 | 31.11 |
| reward_only | max_2_per_ind | B0_DRY_REWARD_ONLY__max_2_per_ind | W4 | 66 | 15 | 5.2692 | 5.6993 | -8.8733 | 31.11 |
| ignored | distinct_1 | B0_DRY_IGNORED__distinct_1 | W4 | 64 | 14 | 4.9563 | 1.0602 | -8.8733 | 30.95 |
| ignored | pure_top3 | B0_DRY_IGNORED__pure_top3 | W4 | 66 | 15 | 4.8133 | 1.3355 | -8.8733 | 31.11 |
| ignored | max_2_per_ind | B0_DRY_IGNORED__max_2_per_ind | W4 | 66 | 15 | 4.8133 | 1.3355 | -8.8733 | 31.11 |


### Case Studies & Behavioral Impact of False Penalty
#### Case Study 1: Individual Rank Change (CRWD, Snapshot: 2026-07-02)
- **Status**: `EXTENDED` | **Rule**: `pivot` | **pullback_v_is_dry**: `False`
- **B0_ORIGINAL (Symmetric)**: Risk codes = `non_actionable_radar_only, extended_from_buy_point, pullback_not_dry` | Risk count = `3` | **Raw Rank = #96**
- **B0_DRY_REWARD_ONLY**: Risk codes = `non_actionable_radar_only, extended_from_buy_point` | Risk count = `2` | **Raw Rank = #86**
- **Top3 Impact**: CRWD is non-actionable radar, so it did not enter Top3 in either policy. Removing the False penalty improved raw rank from #96 to #86.

#### Case Study 2: Dynamic Behavioral Impact on Top3
Across all 42 historical snapshot dates:
- In **916 candidate instances** (out of 1152 `pullback_v_is_dry == False` records), candidate raw rank improved under `reward_only`.
- In **2 snapshot dates**, internal Top3 rank order swapped between qualified candidates:
  - Snapshot `2026-02-20`: `['FANG', 'HEI']` -> `['HEI', 'FANG']`
  - Snapshot `2026-07-10`: `['BSVN', 'RAPP', 'LASR']` -> `['RAPP', 'BSVN', 'LASR']`
- In **0 candidate instances**, the set of 3 selected Top3 stocks changed (both sets contained the exact same 3 stocks in all snapshots).
- **Empirical Takeaway**: In the observed historical sample, the False penalty altered sorting keys and candidate ranks, but was behaviorally redundant regarding final Top3 membership.


## 3. Locked Challengers Evaluation & Champion Classification

### Champion Classification Matrix
| selector_id | classification | train_w4_med_spread | train_w4_mean_spread | train_w4_cvar_delta | train_w4_stop_delta | val_w4_med_spread | val_w4_mean_spread | val_w4_cvar_delta | val_w4_stop_delta | val_support_weeks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0_DRY_REWARD_ONLY__distinct_1 | EQUIVALENT TO B0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 9 |
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


## 4. Rigorous Scientific Conclusions

1. **A. False Penalty**:
   - In the observed Train and Validation periods, removing the `pullback_not_dry` penalty (`reward_only`) improved candidate raw ranks (916 instances across 1152 records) and swapped internal Top3 rank order in 2 snapshots, but **did not change final Top3 membership** in any snapshot.
   - The False penalty is behaviorally redundant in the observed sample; there is no empirical evidence that it harmed portfolio-level Top3 performance.

2. **B. True Reward**:
   - Retaining `dry_pullback` showed a modest Train-only positive indication: paired W4 mean advantage $\approx +0.49\%$ (+0.4885%) versus `ignored` (5.4448% vs 4.9563%). However, paired median spread = 0.0%, CVaR delta = 0.0%, and stop delta = 0.0%, and `ignored` was not evaluated on a fresh sealed holdout.

3. **C. Ignored Policy**:
   - `Ignored` underperformed `reward_only` by $\approx 0.49\%$ in paired Train W4 mean, but there was no paired median, downside, or stop rate improvement evidence.

4. **D. Industry Concentration Constraint (`distinct_1`)**:
   - Modest Train-side support for keeping the hard industry diversity constraint (`distinct_1` achieved 5.4448% W4 mean and 30.95% stop rate vs 5.2692% and 31.11% for `pure_top3`).

5. **E. Overall Champion Finding**:
   - **NO ROBUST REPLACEMENT FOR B0 FOUND**.
   - `B0_DRY_REWARD_ONLY__distinct_1` is classified as **`EQUIVALENT TO B0`** (zero return/downside spread on identical support).
   - All complex ML models suffered severe out-of-sample degradation on sealed validation and were classified as **`UNSTABLE`**.

## 5. Provenance Details

```json
{
  "code_hash": "6c97726eb83500bb5d9b5a3c8d4c7595c2390f88a2b1732ac68cd5b9689a60fd",
  "dependency_hashes": {
    "codebase_hash": "6c97726eb83500bb5d9b5a3c8d4c7595c2390f88a2b1732ac68cd5b9689a60fd",
    "challenge_package_hash": "7456aace115d3a162a224419b07166939d4ebcf96d6de3bdee681097ae4ca279",
    "production_b0_hash": "115387c9861f7202c0f6b3c89fe2d2ff594544de93264901ee6d2f72e930c477"
  },
  "panel_hash": "6886c8d721b808476aa8491bb2140e28a7f96fb07ad6e01f313aec06f5e17c24",
  "git_sha": "dcbf7eb250f90880f65cf4769b20bed7d0862cdd",
  "git_dirty": true,
  "code_dirty": false,
  "locked_challenger_ids": [
    "B0_DRY_REWARD_ONLY__distinct_1",
    "signal_f1_lgbm_w4_distinct_industry",
    "actionable_f1_lgbm_w4_distinct_industry",
    "signal_agent_ridge_w1_portfolio_aware",
    "actionable_agent_lgbm_w4_distinct_industry"
  ]
}
```
