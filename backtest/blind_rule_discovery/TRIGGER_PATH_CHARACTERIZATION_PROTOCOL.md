# R4 Trigger-Path Winner / Loser Characterization Protocol

This protocol freezes the next retrospective research stage after R1/R2/R3.

R4 is **known-history characterization and robustness research**. It is not a new
sealed holdout, and no R1/R2/R3 period regains unseen status.

## 1. Research question

At the moment a BreakoutFollow candidate reaches an executable trigger entry, which
point-in-time stock/setup features and causal execution features are associated with:

1. a fast favorable path;
2. an early stop-first path;
3. better or worse W1-W4 return trajectories;
4. better or worse 3w/4w MAE/MFE;
5. stable cross-sectional separation after controlling market state.

The purpose is to characterize **winner stocks and loser stocks**, not to rediscover
another market-timing rule.

## 2. Frozen execution entry

The only approved R4 execution command is:

```bash
python -m backtest.blind_rule_discovery.trigger_path_characterization_r4_runner ...
```

The following modules are implementation/development layers and are not approved as
standalone research commands:

- `trigger_path_characterization.py`
- `trigger_path_characterization_r4.py`

An executor such as Gemini may run the frozen runner and read outputs. It may not
edit, patch, tune, or replace any implementation during the run.

## 3. Entry semantics

Entry causality is inherited from the established BF research path:

- start from the historical signal snapshot;
- use `ibd_trigger_price`, falling back to `ibd_candidate_price` only when needed;
- inspect only the next 5 trading sessions;
- accept an open/gap entry only up to +5% above trigger;
- otherwise enter when intraday High first reaches trigger;
- do not use a later hindsight dip entry.

Stock/setup predictors are values already known by the signal snapshot. R4 additionally
allows the following execution facts because they are known at the moment the order is
actually triggered:

- `entry_delay_sessions`;
- `entry_extension_pct`;
- `entry_is_gap_or_open`.

No post-entry bar close/volume feature may be used as an entry predictor.

## 4. Primary 3-week first-passage outcome

The primary horizon is 15 trading sessions including the entry session.

For an executable entry:

- `fast_winner_3w`: +20% is reached before -8% within 15 sessions;
- `stop_first_3w`: -8% is reached before +20% within 15 sessions;
- `unresolved_3w`: neither boundary is reached within 15 sessions;
- `ambiguous_3w`: intrabar order cannot be established causally.

### Probability denominator

The primary probabilities are:

```text
P(fast_winner_3w)
P(stop_first_3w)
P(unresolved_3w)
```

using **all non-ambiguous executable entries** as the denominator.

`unresolved_3w` remains in the denominator. It must never be removed merely to make
winner/stop rates look larger.

`ambiguous_3w` is excluded from those three probabilities and reported separately.

## 5. Stop-first recovery split

A stop-first trade is not automatically treated as the same economic failure as a
setup that never recovers.

R4 therefore separately reports:

- `stop_first_then_winner_12w`: stop first within 3w, but the canonical 12w path later
  reaches +20%;
- `persistent_stop_first`: stop first within 3w without that canonical 12w recovery.

This allows later work to distinguish bad setup selection from entry/stop-placement
problems.

## 6. W1-W4 path and risk/reward

For every usable entry, report stock return and SPY excess return at:

- W1 = 5 sessions;
- W2 = 10 sessions;
- W3 = 15 sessions;
- W4 = 20 sessions.

For every W1-W4 return/excess series report at minimum:

```text
p25 / p50 / p75
```

Also report:

- 3w MAE p50;
- 3w MFE p50;
- 4w MAE p50;
- 4w MFE p50;
- `MFE_3w_p50 / abs(MAE_3w_p50)` where defined.

Fast-winner probability alone is not sufficient evidence of a good feature.

## 7. Market timing must be separated from stock selection

### 7.1 Frozen R3 favorable-regime scope

R4 may condition descriptively on the already-known R3 historical regime:

```text
M_8w_drawdown <= -0.04719988
AND
M_dist_52w_high >= -0.05692191
```

This is frozen historical context only. It is not a newly validated market rule and
cannot be called OOS evidence.

### 7.2 Same-snapshot cross-sectional comparison

The stronger stock-selection control is identical `snapshot_date`.

For each stock/setup feature, compare `fast_winner_3w` stocks with `stop_first_3w`
stocks occurring in the same snapshot. Because broad-market state is identical within
the snapshot, this directly asks why one candidate worked while another failed.

Report at minimum:

- matched snapshot count;
- pairwise probability that the winner feature value exceeds the stop-first value;
- equal-weight snapshot AUC median;
- median within-snapshot feature difference;
- sign-consistency fraction across matched snapshots.

## 8. Single-feature characterization

Use deterministic historical quantile bins. For every stock/execution feature, report
both the full-history scope and the frozen favorable-regime scope.

Each bin must contain:

- sample/evaluable/ambiguous counts;
- fast-winner rate and lift;
- stop-first rate and reduction/lift;
- unresolved rate;
- stop-first recovery split;
- W1-W4 p25/p50/p75 return and excess paths;
- 3w/4w MAE/MFE;
- quarter path-edge stability.

Also produce explicit `feature_extremes.csv` identifying, with support constraints:

- the bin most associated with fast-winner behavior;
- the bin most associated with stop-first behavior.

These are retrospective characterizations, not production thresholds.

## 9. Stock interaction search

The interaction search may use only:

- the explicit stock/setup feature allowlist;
- causal execution features listed in section 3.

**Every `M_*` feature is forbidden as a stock-condition input.**

Generate deterministic q20/q40/q60/q80 threshold conditions and search:

- every supported single condition;
- every distinct-feature two-condition AND pair.

No LLM chooses thresholds or rules.

For every rule report:

- fast-winner lift;
- stop-first reduction;
- `path_edge = fast-winner lift + stop-first reduction`;
- W1-W4 path metrics;
- MAE/MFE;
- evaluated-quarter count;
- positive path-edge quarter fraction;
- median/worst quarter path edge.

The scalar `quality_score` is only an ordering aid. All underlying metrics remain the
primary evidence.

Produce separate views from the same frozen search:

- `winner_interactions_*`: rules most associated with fast winners;
- `stop_risk_interactions_*`: rules most associated with stop-first risk.

Do not run a second tuned search for either view.

## 10. Rolling robustness

Run expanding-window re-search separately for:

- `all` market history;
- frozen `r3_favorable` scope.

For every chronological fold:

1. thresholds are regenerated from past training quarters only;
2. the stock rule is selected from past data only;
3. the rule is frozen;
4. exactly the next entry quarter is evaluated.

Every fold after the minimum training window must remain in the output.

If a scope has insufficient training support, the fold is marked
`train_insufficient=1`; it is **not dropped**. It remains a zero-selection/stability
failure in the all-fold denominator.

Report both:

- positive path-edge fraction among evaluable folds;
- positive path-edge fraction among all folds;
- zero-selection fraction;
- insufficient-training fraction.

Also keep two different pooled baselines:

1. baseline across all chronological test folds;
2. baseline only across folds in which the stock rule selected at least one trade.

The second is the relevant pooled baseline for residual stock-selection edge; the
first also measures opportunity/timing availability.

## 11. Interpretation discipline

R4 can support statements such as:

- a stock feature is consistently higher/lower in same-snapshot fast winners;
- a feature bin has lower stop-first probability and better W3 MAE/MFE historically;
- a stock-only interaction retains positive edge in multiple rolling folds;
- a favorable market regime plus a stock feature has better historical path quality.

R4 cannot by itself support:

- "this is production Alpha";
- "replace B0 immediately";
- "this threshold will generalize";
- "R3 favorable regime is validated OOS".

## 12. Execution discipline

Gemini/execution agents are strictly read/run only.

They may:

- verify branch and HEAD;
- run the committed tests;
- execute the frozen R4 runner once;
- read and report generated outputs.

They may not:

- edit source/tests/docs/config;
- create helper scripts or alternate runners;
- change any default threshold/support/search parameter;
- call DeepSeek, another LLM, or RD-agent;
- rerun with different parameters after seeing results;
- overwrite historical R1/R2/R3 outputs;
- commit anything.

If tests or the frozen run fail, stop and report the exact failure. Code repair belongs
in a separate audited development turn before any new execution.
