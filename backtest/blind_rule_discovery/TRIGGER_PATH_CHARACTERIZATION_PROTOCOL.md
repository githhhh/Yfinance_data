# R4 Trigger-Path Winner / Loser Characterization Protocol

R4 is the next retrospective research stage after R1/R2/R3. It is **known-history
characterization and robustness research**, not a new sealed holdout. No previously
consumed period regains unseen status.

## 1. Research question

At an executable BreakoutFollow trigger entry, which point-in-time stock/setup
features and causal execution facts are associated with:

1. +20% before -8% within 3 trading weeks;
2. -8% before +20% within 3 trading weeks;
3. persistent stop-first risk versus later 12w recovery;
4. W1/W2/W3/W4 return and excess-return paths;
5. 3w/4w MAE/MFE;
6. stable same-market cross-sectional separation between winners and losers.

The purpose is to characterize **winner stocks and loser stocks**. It is explicitly
not another search for a broad-market timing rule.

## 2. Only approved execution entry

The only approved R4 command is:

```bash
python -m backtest.blind_rule_discovery.trigger_path_characterization_r4_final_runner ...
```

The following are implementation/development layers only and must not be executed as
standalone R4 studies:

- `trigger_path_characterization.py`
- `trigger_path_characterization_r4.py`
- `trigger_path_characterization_r4_runner.py`
- `trigger_path_characterization_r4_search.py`

Gemini/execution agents may run the final runner and read outputs only. They may not
edit or patch any implementation during execution.

## 3. Entry semantics

Entry causality inherits the established BF research path:

- start from the historical signal snapshot;
- use `ibd_trigger_price`, falling back to `ibd_candidate_price` only when required;
- inspect only the next 5 trading sessions;
- an open/gap entry is accepted only up to +5% above trigger;
- otherwise enter when intraday High first reaches trigger;
- never use a later hindsight dip entry.

Stock/setup predictors are fields already known by the signal snapshot. R4 also allows
three execution facts because they are known when the entry actually fires:

- `entry_delay_sessions`;
- `entry_extension_pct`;
- `entry_is_gap_or_open`.

No post-entry close/volume information may be used as a predictor.

## 4. Primary 3-week first-passage outcome

The primary horizon is 15 trading sessions including the entry session.

- `fast_winner_3w`: +20% before -8% within 15 sessions;
- `stop_first_3w`: -8% before +20% within 15 sessions;
- `unresolved_3w`: neither boundary within 15 sessions;
- `ambiguous_3w`: causal intrabar order cannot be determined.

The primary probability denominator is **all non-ambiguous executable entries**.
`unresolved_3w` remains in the denominator. It must never be removed just to increase
reported winner/stop rates. Ambiguous paths are excluded from these three rates and
reported separately.

## 5. Stop-first recovery split

R4 separately reports:

- `stop_first_then_winner_12w`: stop first within 3w, but canonical 12w path later
  reaches +20%;
- `persistent_stop_first`: stop first within 3w without that later canonical recovery.

This distinction is required because a bad setup and an entry/stop-placement problem
are not the same economic failure.

## 6. W1-W4 and risk/reward path

For every usable entry:

- W1 = 5 sessions;
- W2 = 10 sessions;
- W3 = 15 sessions;
- W4 = 20 sessions.

Report stock return and SPY excess return at every horizon with:

```text
p25 / p50 / p75
```

Also report:

- 3w MAE p50;
- 3w MFE p50;
- 4w MAE p50;
- 4w MFE p50;
- `MFE_3w_p50 / abs(MAE_3w_p50)` when defined.

A high fast-winner probability without acceptable path/risk metrics is not sufficient.

## 7. Market timing and stock selection must remain separated

### 7.1 Frozen R3 favorable-regime context

R4 may condition descriptively on the already-known R3 historical regime:

```text
M_8w_drawdown <= -0.04719988
AND
M_dist_52w_high >= -0.05692191
```

This is frozen retrospective context only. It is not newly validated OOS evidence.

### 7.2 Same-snapshot cross-sectional control

The stronger stock-selection control is identical `snapshot_date`.

For every stock/execution feature compare `fast_winner_3w` stocks and
`stop_first_3w` stocks from the same snapshot. Because broad-market state is identical
within a snapshot, this directly tests stock-level separation.

Report at minimum:

- matched snapshot count;
- pairwise probability winner feature value > stop-first value;
- equal-weight snapshot AUC median;
- median within-snapshot feature difference;
- sign-consistency fraction.

## 8. Single-feature characterization

Use deterministic historical q20/q40/q60/q80 bins. Apply the same full-history bin
boundaries to both scopes:

- `all`;
- frozen `r3_favorable`.

Every bin must report:

- selected/evaluable/ambiguous support;
- fast-winner rate and lift;
- stop-first rate and reduction/lift;
- unresolved rate;
- persistent/recovered stop-first split;
- W1-W4 p25/p50/p75 return and excess paths;
- 3w/4w MAE/MFE;
- evaluated-quarter count;
- positive path-edge quarter fraction;
- median/worst quarter path edge;
- higher-stop-risk quarter fraction.

`feature_extremes.csv` must separately identify the supported bin most associated
with fast winners and the supported bin most associated with stop-first risk.
These are retrospective characterizations, not production thresholds.

## 9. Stock-only interaction search

The search may use only:

- the explicit stock/setup feature allowlist;
- causal execution features from section 3.

**Every `M_*` feature is forbidden as a stock-condition input.**

Generate deterministic q20/q40/q60/q80 conditions and test:

- every supported single condition;
- every distinct-feature two-condition AND pair.

There is no pair pruning or beam search in R4. No LLM chooses thresholds/rules.

For ranking every candidate rule, the vectorized search computes only metrics required
by the frozen score:

- fast-winner lift;
- stop-first reduction;
- `path_edge = fast-winner lift + stop-first reduction`;
- W3 excess p50;
- 3w MAE/MFE and their ratio;
- evaluated-quarter count;
- positive/median/worst quarter path edge.

This compact evaluation is a performance optimization only. It does **not** reduce the
search space or change ranking semantics. After ranking, the final runner enriches
reported top rules with full W1-W4 p25/p50/p75 path distributions.

Frozen reporting limits:

- quality-ranked rules: top 500;
- winner-oriented view: top 200;
- stop-risk view: top 200.

The complete candidate count and pair count remain in metadata/search audit.

The scalar `quality_score` is only an ordering aid; all underlying metrics are the
primary evidence.

## 10. Separate winner and loser interaction views

From the **same frozen search**, produce:

- `winner_interactions_*`: highest fast-winner lift, then lower stop risk / better W3;
- `stop_risk_interactions_*`: highest stop-first lift, then weaker fast-winner/W3 path.

Do not run a second tuned search for either view.

## 11. Rolling robustness

Run expanding-window re-search separately for:

- `all`;
- frozen `r3_favorable`.

For every chronological fold:

1. regenerate stock thresholds from past training quarters only;
2. search/select the stock rule from past data only;
3. freeze the rule;
4. evaluate exactly the next entry quarter.

Every fold after the minimum training window must remain in output.

If training support is insufficient, record `train_insufficient=1`, selected=0, and
keep the fold in the all-fold denominator. Never silently drop difficult folds.

Rolling reporting must include:

- all-fold positive path-edge fraction;
- evaluable-fold positive path-edge fraction;
- zero-selection count/fraction;
- insufficient-training count/fraction;
- same-snapshot matched edge where available;
- pooled selected fast-winner/stop-first rates;
- pooled baseline across all folds;
- pooled baseline restricted to folds where the rule actually selected trades.

The selected-fold baseline is the key pooled comparison for residual stock-selection
edge. The all-fold baseline also reflects opportunity/timing availability.

## 12. Interpretation boundary

R4 can support claims such as:

- a stock feature is repeatedly higher/lower in same-snapshot fast winners;
- a feature bin historically lowers stop-first risk and improves W3/MAE/MFE;
- a stock-only interaction retains path edge across multiple chronological folds;
- favorable-market context plus a stock feature improves historical path quality.

R4 alone cannot support:

- "production Alpha";
- immediate replacement of B0;
- guaranteed threshold generalization;
- treating the R3 favorable regime as new OOS evidence.

## 13. Executor discipline

Gemini/execution agents are strictly read/run only.

Allowed:

- verify branch/HEAD/status;
- read this protocol;
- run committed tests;
- execute the final frozen R4 runner once;
- read generated outputs and report facts.

Forbidden:

- edit source/tests/docs/config;
- create helper scripts/runners;
- change default search/support thresholds;
- change output limits;
- call DeepSeek, another LLM, or RD-agent;
- rerun with different parameters after seeing results;
- overwrite R1/R2/R3 output;
- commit anything.

If a test or frozen command fails, stop and return the exact failure. Code repair must
happen in a separate audited development turn before any new execution.
