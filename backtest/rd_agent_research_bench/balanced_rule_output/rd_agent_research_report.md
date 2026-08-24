# RD-Agent Balanced Rule Research Report

## Scope And Source Isolation

- B0 source: Git repository only, `.agents/skills/ibd-candidate-prescreen/SKILL.md` at HEAD `9e8e8bb` and base commit `7925b22` via `git show`/`git diff`.
- Environment-registered `ibd-candidate-prescreen` was not invoked or loaded through the skill mechanism.
- Replay weeks loaded: 42 (`2025-10-10` to `2026-08-07`); sealed holdout: 2026-07-02, 2026-07-10, 2026-07-17, 2026-07-24, 2026-07-31, 2026-08-07.

## Why Current Backtests Are Not Full Quant Backtests

- Existing Backtrader integration is a weekly rebalance model: positions are sold when not re-selected. That violates the requested IBD lifecycle and can confound selector quality with forced turnover.
- This run adds a daily OHLC path state machine with next-open entry, protective stop, ordinary profit zone, 8-week lock, repeated-signal ignore, and censored mark-to-market. It is still not a broker-grade simulator because daily OHLC cannot know intraday path; conservative priority is pre-registered.
- Benchmarks are fair only inside this evaluator because they share replay pools, next-open entry, costs/slippage assumptions, missing-path handling, and the same state machine.

## Data Sufficiency

- The replay sample is short and ticker exposures repeat. Results are exploratory unless a rule improves OOS, does not worsen downside, and is not concentrated in one ticker/week/source.
- Current data is not enough to claim a confirmed structural replacement for B0. More complete weekly PIT snapshots are needed, especially beyond one market regime and with uncensored 8-week outcomes.

## Top Observed Variants

| variant               |   weeks_with_picks |   selection_count |   avg_picks_per_week |   median_forward_8w_return_pct |   mean_forward_8w_return_pct |   worst_forward_8w_return_pct |   mfe_median_pct |   mae_median_pct |   profit_zone_rate |   stop_first_rate |   trade_count |   trade_expectancy_pct |   trade_win_rate |   trade_median_return_pct |   censored_trades |   power_trigger_count |   max_single_ticker_pick_share |   max_single_week_pick_share | top1_mode   |
|:----------------------|-------------------:|------------------:|---------------------:|-------------------------------:|-----------------------------:|------------------------------:|-----------------:|-----------------:|-------------------:|------------------:|--------------:|-----------------------:|-----------------:|--------------------------:|------------------:|----------------------:|-------------------------------:|-----------------------------:|:------------|
| raw_pool_c_continuous |                 42 |               126 |                3     |                        6.68911 |                      12.6541 |                      -32.1292 |          32.0087 |        -10.0413  |           0.166667 |          0.357143 |            71 |                6.64806 |         0.295775 |                  -7.5     |                19 |                    19 |                      0.031746  |                    0.0238095 | False       |
| b0_repository_skill   |                 40 |                97 |                2.425 |                        6.23688 |                      10.3399 |                      -30.8398 |          20.0333 |         -4.75992 |           0.195876 |          0.206186 |            55 |               13.7745  |         0.545455 |                   6.22305 |                23 |                     4 |                      0.0206186 |                    0.0309278 | False       |
| eps_drop              |                 40 |               101 |                2.525 |                        4.45731 |                       9.6385 |                      -19.9204 |          16.5214 |         -6.59722 |           0.178218 |          0.277228 |            61 |                8.24315 |         0.508197 |                   1.46214 |                23 |                     4 |                      0.019802  |                    0.029703  | False       |
| allow_zero_strict     |                 40 |               100 |                2.5   |                        4.39971 |                      11.9172 |                      -19.9204 |          20.2636 |         -6.07977 |           0.2      |          0.24     |            57 |               13.5786  |         0.526316 |                   3.52772 |                23 |                     5 |                      0.02      |                    0.03      | False       |
| base_pullback_context |                 40 |               101 |                2.525 |                        4.39971 |                      11.9172 |                      -19.9204 |          20.2636 |         -6.07977 |           0.19802  |          0.237624 |            57 |               13.5786  |         0.526316 |                   3.52772 |                23 |                     5 |                      0.019802  |                    0.029703  | False       |
| close_trigger_soft    |                 40 |               101 |                2.525 |                        4.39971 |                      11.9172 |                      -19.9204 |          20.2636 |         -6.07977 |           0.19802  |          0.237624 |            57 |               13.5786  |         0.526316 |                   3.52772 |                23 |                     5 |                      0.019802  |                    0.029703  | False       |
| fresh_continuous      |                 40 |               101 |                2.525 |                        4.39971 |                      11.9172 |                      -19.9204 |          20.2636 |         -6.07977 |           0.19802  |          0.237624 |            57 |               13.5786  |         0.526316 |                   3.52772 |                23 |                     5 |                      0.019802  |                    0.029703  | False       |
| ignore_entry_valid    |                 40 |               101 |                2.525 |                        4.39971 |                      11.9172 |                      -19.9204 |          20.2636 |         -6.07977 |           0.19802  |          0.237624 |            57 |               13.5786  |         0.526316 |                   3.52772 |                23 |                     5 |                      0.019802  |                    0.029703  | False       |

## B0 Rule Verdicts

| Current rule                     | Current level         | Experimental evidence                                                   | OOS impact                         | Stability                                  | New handling           |
|:---------------------------------|:----------------------|:------------------------------------------------------------------------|:-----------------------------------|:-------------------------------------------|:-----------------------|
| `ibd_entry_status == ACTIONABLE` | Hard Eligibility      | status_soft/status_all variants did not clear Pareto bar                | No stable OOS replacement          | Exploratory; repeated ticker/week exposure | Hard Eligibility       |
| `ibd_entry_valid`                | Encoded in status     | ignore_entry_valid largely overlaps B0/actionable status                | Insufficient independent increment | Unclear independent sample                 | Context Only           |
| Close above trigger / candidate  | Hard Eligibility      | close_trigger_soft adds candidates but not stable median improvement    | Promising but not confirmed        | Boundary sensitive                         | Continuous/Major Score |
| Fresh Zone 0-5 / 0-2             | Hard + tie-break      | fresh_continuous similar to balanced variants                           | Promising but not confirmed        | Needs more weeks near boundaries           | Continuous/Major Score |
| Entry volume >=1.5               | Critical hard         | volume_soft/signal_specific did not dominate B0                         | Promising but not confirmed        | Route-specific evidence thin               | Continuous/Major Score |
| Geometry failure                 | Critical hard         | geometry_drop/soft can raise observed returns but worsens semantic risk | No support to relax clear FAIL     | Some picks have clear failure flags        | Hard Eligibility       |
| Base depth/duration/mbox         | Context               | base_pullback_context similar to other balanced variants                | Insufficient evidence              | Likely redundant with C_continuous/source  | Context Only           |
| Pullback depth/duration          | Context               | No stable independent uplift                                            | Insufficient evidence              | Route sample small                         | Context Only           |
| `pullback_v_is_dry`              | Major FAIL when false | hard variant not proven; minor/drop comparison supports softer handling | Hard gate not confirmed            | Only applicable to pullback routes         | Minor Bonus            |
| `eps_yoy_growth >=25`            | Auxiliary             | eps_pass_hard/known_hard/drop do not beat B0 robustly                   | Hard gate not confirmed            | PIT coverage good but sample short         | Minor Bonus            |
| EPS UNKNOWN                      | Info missing          | PIT audit has 0 future-date violations; missing paths retained          | Confirmed not FAIL                 | Missingness still material                 | Manual Review          |
| Industry                         | Coverage only         | coverage affects list composition, not raw score                        | No ranking evidence                | Not causal quality signal                  | Context Only           |
| Top1                             | Diagnostic only       | top1_diagnostic concentration and sensitivity remain high               | Cannot replace Top3                | High rank-error sensitivity                | Context Only           |

## Exit Policy Sensitivity

| selector            | exit_policy             |   trade_count |   expectancy_pct |   median_return_pct |   win_rate |   censored_trades |   power_trigger_count | note                                      |
|:--------------------|:------------------------|--------------:|-----------------:|--------------------:|-----------:|------------------:|----------------------:|:------------------------------------------|
| b0_repository_skill | fixed_forward_8w_mark   |            97 |         10.3399  |             6.23688 |   0.381443 |                41 |                     0 | fixed label only, no intra-path exits     |
| b0_repository_skill | ibd_8w_default_7p5_22p5 |            55 |         13.7745  |             6.22305 |   0.545455 |                23 |                     4 | selector frozen; only exit policy changes |
| b0_repository_skill | no_8week_rule_7p5_22p5  |            55 |          6.54381 |             6.22305 |   0.545455 |                19 |                     0 | selector frozen; only exit policy changes |
| b0_repository_skill | stop7_profit20          |            55 |         13.0184  |             5.67521 |   0.527273 |                21 |                     4 | selector frozen; only exit policy changes |
| b0_repository_skill | stop8_profit25          |            54 |         15.4785  |            10.1883  |   0.611111 |                28 |                     4 | selector frozen; only exit policy changes |
| b0_repository_skill | post_lock_resume_profit |            55 |          6.54381 |             6.22305 |   0.545455 |                19 |                     4 | selector frozen; only exit policy changes |

## Required Answers

- ACTIONABLE as hard boundary: Promising to test as soft/status-wide, but not confirmed as a replacement. Keep formal Top Review ACTIONABLE-only for now; keep all-signal shadow as audit.
- Volume and Geometry: both show useful diagnostic value, but evidence does not support stronger global hardening. Volume is better studied as saturated/route-specific evidence; Geometry failures remain risk flags/hard exclusions only when clearly failed.
- `pullback_v_is_dry`: do not use as global hard gate. The balanced recommendation is Minor Bonus/Risk Flag for pullback-applicable routes; NOT_APPLICABLE for base breakout.
- EPS: keep PIT-known status for formal information completeness; `EPS >= 25` is a soft auxiliary score/risk context, not a hard gate. EPS UNKNOWN remains UNKNOWN/Manual Review, not FAIL.
- Unused schema fields: `base_mbox_count`, `base_duration_weeks`, `base_depth_abs`, `pullback_duration_weeks`, `pullback_pct_off_peak`, `C_continuous`, `rank_C_continuous`, and source labels deserve continued context-only or audit-layer tests.
- New derived dimensions: `trigger_pos`, fresh distance decay, saturated volume score, base/pullback context score. None has enough OOS evidence to enter production hard rules.
- Best observed vs most balanced: the highest observed return variant is not automatically recommended. The most balanced conclusion is to keep B0 and move only low-risk clarifications into future research.
- Top1: useful diagnostic, not a Skill replacement due concentration and rank-error sensitivity.
- Improvement source: current evidence primarily tests selector changes; exit policy is held constant in the main comparison. IBD 8-week rule improves rule fidelity but not yet proven to improve results robustly.

## Conclusion

Verdict: Promising but not confirmed. Keep B0 production rules unchanged except for future low-risk documentation/schema fixes. Do not update the repository Skill from these exploratory results.
