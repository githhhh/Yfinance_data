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
python -m backtest.blind_rule_discovery.trigger_path_characterization_r4_causal_runner ...
```

All other R4 modules are implementation/development history and must not be executed
as standalone research commands:

- `trigger_path_characterization.py`
- `trigger_path_characterization_r4.py`
- `trigger_path_characterization_r4_runner.py`
- `trigger_path_characterization_r4_search.py`
- `trigger_path_characterization_r4_final_runner.py`

Gemini/execution agents may run the causal runner and read outputs only. They may not
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

Report stock return and SPY excess return at every horizon with `p25 / p50 / p75`.
Also report 3w/4w MAE and MFE p50 and `MFE_3w_p50 / abs(MAE_3w_p50)` where defined.
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

For every stock/execution feature compare `fast_winner_3w` and `stop_first_3w` stocks
from the same snapshot. Because broad-market state is identical within a snapshot,
this directly tests stock-level separation.

Report matched snapshot count, pairwise probability winner value > stop-first value,
equal-weight snapshot AUC median, median within-snapshot difference, and sign
consistency.

## 8. Single-feature characterization

Use deterministic historical q20/q40/q60/q80 bins. Apply the same full-history bin
boundaries to both scopes: `all` and frozen `r3_favorable`.

Every bin must report support; fast-winner/stop-first/unresolved probabilities;
persistent/recovered stop split; W1-W4 p25/p50/p75 return and excess; 3w/4w MAE/MFE;
and quarter stability. `feature_extremes.csv` separately identifies the supported bin
most associated with fast winners and the one most associated with stop-first risk.
These are retrospective characterizations, not production thresholds.

## 9. Stock-only interaction search

The search may use only the explicit stock/setup feature allowlist plus the causal
execution features in section 3. **Every `M_*` feature is forbidden as a stock-condition
input.**

Generate deterministic q20/q40/q60/q80 conditions and test every supported single
condition and every distinct-feature two-condition AND pair. There is no pair pruning,
beam search, or LLM rule generation.

Candidate ranking uses fast-winner lift, stop-first reduction, path edge, W3 excess,
3w MAE/MFE, and quarter stability. The vectorized candidate evaluator is only a
performance optimization: it does not reduce the search space or change the frozen
ranking semantics. Reported top rules are subsequently enriched with full W1-W4
p25/p50/p75 distributions.

Frozen output limits are top 500 quality-ranked, top 200 winner-oriented, and top 200
stop-risk rules. Complete candidate/pair counts remain in metadata.

## 10. Separate winner and loser interaction views

From the **same frozen search**, produce `winner_interactions_*` and
`stop_risk_interactions_*`. Do not run a second tuned search for either view.

## 11. Causal rolling robustness and label purge

Run expanding-window re-search separately for `all` and frozen `r3_favorable`.

For every chronological fold:

1. define the next `entry_quarter` as the test quarter;
2. take only earlier entry quarters as candidate training rows;
3. **purge every training row whose `exit_date_w3` is missing or is on/after the first
   calendar day of the test quarter**;
4. regenerate stock thresholds from the remaining purged training data only;
5. search/select the stock rule from that purged past data only;
6. freeze the rule and evaluate exactly the next entry quarter.

This purge is mandatory because an entry in the final weeks of a training quarter can
have its 15-session outcome inside the next quarter. Merely grouping by entry quarter
is not sufficient causal isolation.

Every fold must record:

- `test_quarter_start`;
- `train_rows_before_purge`;
- `train_rows_after_purge`;
- `train_rows_purged_for_w3_overlap`;
- `train_max_exit_date_w3`;
- `w3_label_overlap_after_purge` (must always be false).

Every chronological fold after the minimum training window remains in output. If
post-purge training support is insufficient, record `train_insufficient=1`, select
zero trades, and keep the fold in the all-fold denominator.

Rolling reporting must include all-fold and evaluable-fold positive path-edge
fractions, zero-selection and insufficient-training fractions, same-snapshot matched
edge, and pooled baselines both across all folds and across selected/traded folds.
The selected-fold baseline is the key pooled comparison for residual stock-selection
edge.

## 12. Interpretation boundary

R4 may support claims about historical same-snapshot winner/loser feature separation,
stop-risk characteristics, and stock-only interactions that persist across multiple
causally purged chronological folds. R4 alone cannot establish production Alpha,
justify immediate replacement of B0, guarantee threshold generalization, or turn the
R3 favorable regime into OOS evidence.

## 13. Executor discipline

Gemini/execution agents are strictly read/run only.

Allowed: verify branch/HEAD/status, read this protocol, run committed tests, execute
the causal frozen R4 runner once, and read/report outputs.

Forbidden: edit source/tests/docs/config; create helper scripts/runners; change default
search/support thresholds or output limits; call DeepSeek/LLM/RD-agent; rerun with
different parameters after seeing results; overwrite R1/R2/R3 output; commit anything.

If a test or frozen command fails, stop and return the exact failure. Code repair must
happen in a separate audited development turn before any new execution.
