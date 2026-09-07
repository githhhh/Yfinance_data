# R5 Causal Stop-Risk Validation Protocol

R5 is the final narrow robustness study after R4.

It exists because R4 found much stronger retrospective loser/stop-risk structure than
winner structure, but R4 chronological rolling selected the best **quality-ranked
positive stock rule** in each training fold. Therefore R4 did not actually test the
claim that stop-risk information itself may be more chronologically learnable.

R5 answers only that missing question.

R5 is retrospective known-history research. All R1-R4 periods remain known history.
It is **not** a sealed holdout, not a new blind experiment, and not production Alpha
certification.

## 1. Research questions

R5 asks two questions:

1. If every expanding training window searches the same frozen R4 stock-only
   single/pair grid but selects the strongest **stop-risk** rule, does that rule
   identify higher stop-first and persistent-stop risk in the next quarter?
2. Do the specific semantic risk families discovered after R4 retain the same risk
   direction when their thresholds are regenerated from past-only training data?

R5 does **not** ask for another winner rule or another Top3 ranking formula.

## 2. Only approved execution entry

```bash
python -m backtest.blind_rule_discovery.stop_risk_validation_r5 ...
```

Gemini/execution agents are read/run only. They may not edit source, tests, docs,
configuration, thresholds, scoring, or outputs during execution.

## 3. Population and outcome semantics

Population and executable-entry causality are inherited unchanged from R4:

- candidate must reach an executable BF trigger entry within the existing 5-session
  buy-zone window;
- +20% before -8% within 15 trading sessions = `fast_winner_3w`;
- -8% before +20% within 15 sessions = `stop_first_3w`;
- neither = `unresolved_3w`;
- ambiguous same-bar ordering is excluded from the primary probability denominator;
- unresolved remains in the denominator.

R5 focuses on:

- `stop_first_rate`;
- `persistent_stop_first_rate`;
- stop-first then 12w recovery;
- W1-W4 return/excess path;
- 3w/4w MAE/MFE;
- same-snapshot stop-risk contrast where available.

## 4. Market separation

R5 runs both scopes independently:

- `all`;
- frozen `r3_favorable`.

`M_*` features remain forbidden in stock-rule search.

The favorable scope is retrospective context only. It is not an independently
validated market rule.

## 5. Causal rolling purge

For every test quarter:

1. training quarters are strictly earlier entry quarters;
2. before any search or threshold generation, remove every training row whose
   `exit_date_w3` is missing or not strictly earlier than the first calendar day of
   the test quarter;
3. assert that no surviving W3 label overlaps the test quarter;
4. preserve every chronological fold, including empty/zero-selection folds.

Every rolling output row must record:

- `test_quarter_start`;
- `train_rows_before_purge`;
- `train_rows_after_purge`;
- `train_rows_purged_for_w3_overlap`;
- `train_max_exit_date_w3`;
- `w3_label_overlap_after_purge`;
- `test_quarter_in_train`.

Any overlap is a protocol failure.

## 6. Risk re-search rolling

Use exactly the R4 stock interaction search space:

- deterministic q20/q40/q60/q80 conditions;
- every supported single condition;
- every supported distinct-feature two-condition AND pair;
- no `M_*` features;
- no LLM;
- no pair pruning;
- no third condition.

Support thresholds stay inherited from R4 rolling:

- `all`: min selected 60, min evaluable 45, min evaluated quarters 4;
- `r3_favorable`: min selected 30, min evaluable 20, min evaluated quarters 2;
- per-quarter support uses `max(5, min_quarter_n // 2)`.

### 6.1 Risk-rule selection order

R5 does not invent a tunable scalar risk score.

All supported candidates are ordered lexicographically by:

1. `higher_stop_risk_quarter_fraction` descending;
2. `median_quarter_stop_first_lift` descending;
3. aggregate `stop_first_lift` descending;
4. aggregate `persistent_stop_first_lift` descending;
5. `evaluable_n` descending.

The first rule is frozen and applied to exactly the next test quarter.

This tests whether **consistently elevated stop risk** is learnable from past history.

## 7. Post-R4 semantic family rolling

R4 produced the following risk hypotheses:

1. deep pullback;
2. extended vs candidate/buy point;
3. deep pullback + extended;
4. high percentage above ceiling;
5. high actual entry extension.

R5 freezes these semantic directions as:

```text
deep_pullback:
    pullback_pct <= training q20

extended_vs_candidate:
    current_vs_ibd_candidate_pct >= training q80

deep_pullback_and_extended:
    pullback_pct <= training q20
    AND current_vs_ibd_candidate_pct >= training q80

high_pct_above_ceiling:
    pct_above_ceiling >= training q80

high_entry_extension:
    entry_extension_pct >= training q80
```

Thresholds are regenerated independently from each purged training fold only.

These families were chosen **after seeing R4 known history**. Therefore their rolling
results are post-hoc chronological robustness evidence, not independent OOS evidence.
No additional family may be added after seeing R5 results.

## 8. Primary validation metrics

For the searched stop-risk rule and each semantic family report:

- test selected/evaluable N;
- test stop-first rate;
- contemporaneous test baseline stop-first rate;
- `stop_first_lift = selected - baseline`;
- persistent-stop rate;
- contemporaneous persistent-stop baseline;
- persistent-stop lift;
- stop-first then 12w recovery rate;
- fast-winner rate;
- unresolved rate;
- W1-W4 return/excess p25/p50/p75;
- 3w/4w MAE/MFE;
- same-snapshot median stop-risk lift where available.

A useful risk detector should have **positive** stop-first lift in future folds.

## 9. Rolling summary discipline

For each scope report:

- all fold count;
- zero-selection count/fraction;
- evaluable stop-lift folds;
- positive stop-lift fraction among all folds;
- positive stop-lift fraction among evaluable folds;
- median stop-first lift among evaluable folds;
- positive persistent-stop lift fraction;
- median persistent-stop lift;
- same-snapshot positive stop-lift fraction;
- pooled selected stop-first rate;
- pooled baseline stop-first rate on the same selected/traded folds;
- pooled stop-first lift;
- pooled selected persistent-stop rate;
- pooled baseline persistent-stop rate on selected/traded folds;
- pooled persistent-stop lift.

Zero-selection and empty-scope folds remain in all-fold denominators.

## 10. Interpretation

R5 may support:

- stop-risk information is chronologically learnable;
- a post-R4 semantic risk family has or lacks directional chronological robustness;
- risk-side evidence appears stronger/weaker than R4 positive-rule rolling.

R5 may not support:

- production hard reject thresholds;
- replacing B0 immediately;
- untouched holdout claims;
- new winner Alpha claims;
- adding new risk families after seeing results.

## 11. Output isolation

R1-R4 outputs are immutable.

R5 uses only:

```text
backtest/blind_rule_discovery/output/stop_risk_validation_r5
```

Expected files:

- `stop_risk_metadata.json`
- `stop_risk_rolling.csv`
- `risk_family_rolling.csv`

## 12. Executor discipline

Gemini may:

- verify branch and exact HEAD;
- read this protocol;
- run committed tests;
- execute the approved R5 entry exactly once;
- read and report outputs.

Gemini may not:

- edit anything;
- tune thresholds or support;
- add or remove families;
- rerun with different options;
- call DeepSeek/RD-agent/another LLM;
- overwrite R1-R4 outputs;
- commit anything.

If tests or execution fail, stop and return the exact failure to ChatGPT for audit and
repair.
