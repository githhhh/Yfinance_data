# R7 Frozen Risk Economic Triage

## Decision and Scope

R6 found historical stop-risk association without consistent evidence that a veto
preserves enough winners. R7 asks whether existing risk information has economic
value under explicitly limited terminal-return accounting. **No additional
RD-Agent calls are needed.** More proposals are not a substitute for measuring
the payoff of the proposals and families already available.

One run delivers the full comparison, not a sequence of single-metric probes:

- Six fixed arms: `deep_pullback`, `extended_vs_candidate`,
  `deep_pullback_and_extended`, `high_pct_above_ceiling`, `r6_rdagent`, `r6_simple`.
- Two panels: all usable entries and a common W4 nonoverlapping-ticker schedule.
- W1/W2/W4 primary and W3 diagnostic only: **48 economic summary cells**.
- Same-week baseline, cash veto and exact Matched-N random cash-veto expectation.
- Gross/net means, distributions, loss/gain attribution, path labels, support,
  quarterly direction, time-block uncertainty and issuer/week concentration.
- A descriptive decision matrix, including risk-only versus economic direction.

No feature discovery, threshold optimization, champion selection, B0 rerun,
provider call, price download, EPS refresh or production mutation occurs. The
four families already exist in R5; R7 excludes `high_entry_extension`, which is
not known at the snapshot. The R6 arms replay **frozen expressions**, not their
occasionally misleading natural-language names. They are not regenerated.

All periods are known history. Nested/walk-forward calculations do not restore
an untouched holdout. No feature or policy gets production approval from R7.

## Bound Inputs

Use precisely the R4 `trigger_path_samples.csv` and matching
`trigger_path_metadata.json` used in the completed R6 run. Supply its directory
containing `input_manifest.json`, `frozen_rules.json` and `summary.json`.

The loader verifies CSV and metadata SHA256, replay hash declaration, exact row
count, ticker-week uniqueness, exclusive/exhaustive W3 labels, the inherited
calendar, weekly dates, terminal horizon ordering and complete finite returns.
It then reproduces each R6 threshold from its original purged training surface
and checks original aggregate selection facts. Changed input, malformed fields,
wrong frozen rules, or mismatched calendars fail closed. Missing numeric PIT
features are allowed, counted and unflagged; they are never called safe.

The source replay SHA is declared provenance, not independently recomputed from
the pool files. R7 does not require those pool files on the analysis machine.
R6's population has usable executable entries and upstream maturity selection.
It is not all original replay weeks or all signals. Empty calendar quarters stay
in the output with `EMPTY_TEST_QUARTER`, never a synthetic return of zero.

`return_w1` through `return_w4` must be decimal entry-to-terminal-close returns
(0.01 = 1%), with `exit_date_w1` through `exit_date_w4`. Existing R4 generates
exactly those fields. Missing returns stop the run; R7 does not average only
surviving/mature subgroups. Optional existing MAE/MFE fields are descriptive only.

## Frozen Rules and Timing

- Family thresholds use the same R5 q20/q80 definitions, not a new grid.
- Start after six consecutive snapshot-calendar quarters. For each quarter,
  calibration uses only `snapshot_date < quarter_start` and
  `exit_date_w3 < quarter_start`, preserving the R6 purge convention.
- No outcome, including W4, is used to fit a family threshold. W4 maturity need
  not determine feature calibration because that outcome is never used in it.
- Freeze all arms before computing economic evaluation. R6 arms additionally
  verify their original training-surface SHA and fitted threshold.
- The inherited calendar can include an empty entry-quarter edge such as
  2026Q2. Keep its calendar denominator separate from evaluable-quarter counts.
- No family is ranked or selected based on R7 outputs.

## Economic Accounting

For one snapshot and horizon, let `n` be admitted candidates, `m` flagged, `r_i`
the frozen terminal gross return and `c` the declared round-trip cost in decimal
units. One initial candidate slot is `1/n` of that cohort's notional allocation.

```
baseline_net   = sum(r_i - c) / n
veto_cash_net  = sum_unflagged(r_i - c) / n
random_cash_net = (n - m) / n * baseline_net
incremental_vs_random = veto_cash_net - random_cash_net
cash_delta = veto_cash_net - baseline_net
           = avoided_gross_loss - foregone_gross_gain + saved_cost
```

Removed slots earn zero cash return; retained slots are NOT reweighted. All
policies use the same initial denominator. Random removal is uniform over the
same snapshot's candidates and removes **exactly m**, so its expectation is
analytical. There are no random draws, no N shrink and no survivor averaging.
All-flagged and zero-flag snapshots retain explicit zero-exposure/no-action
accounting, but cannot masquerade as supported selection evidence.

The same per-candidate cost applies to both matched-removal policies, so it
cancels from their incremental comparison. R7 reports gross cash delta, net cash
delta, saved costs and the **algebraic** break-even cost; it never chooses a cost
assumption to make a rule pass. Supply one fee-plus-slippage assumption before
running, justified by the intended implementation. It is not a measured fill.

Loss/gain attribution is by actual terminal returns, split into all four W3 path
classes: `stop_first_3w`, `fast_winner_3w`, `unresolved_3w`, `ambiguous_3w`.
Ambiguous first-passage order does not make an observed closing return unknown;
these rows stay in economic accounting and leave only path-order risk rates.
An eventual positive terminal return after Stop First remains positive here.
**No label is substituted with an assumed -8% or +20% execution fill.**

## Dependence and Stability

The `nonoverlap_w4` panel admits the earliest executable entry per ticker and
reserves that ticker through its W4 close. Reentry on that same closing date is
not allowed. The schedule starts with no open positions at the first evaluated
cohort and is identical across all arms and horizons. A veto does not create a
new admission or free capital for another policy. The raw panel remains visible.

Economic summaries average snapshots equally, not candidates. Group-return
distributions are explicitly candidate-weighted and kept in a separate output.
Known-feature-only within-week return and stop contrasts isolate missing-field
composition from observed feature associations. Unknown coverage is explicit.

Support: at least ten flagged, ten known retained and three matched snapshots
in a quarter. These inherited descriptive support floors do not imply adequate
power for rare winners. Zero-action, empty, missing-training-feature and
insufficient-support quarters have separate statuses.

Stability output includes:

- Every calendar quarter's sample, membership, return and risk direction.
- 2000 moving-block bootstrap draws, eight calendar weeks per block, seed 42;
  gaps are missing, not zero. Fewer than sixteen weeks gives no interval.
- Leave-one-quarter-out and best/worst-week-removed incremental means.
- Exact leave-one-ticker-out incremental means, with frozen thresholds and
  admissions. Singleton-week removal updates the denominator. This is a
  concentration check, not a blacklist or a new ticker selector.

These are descriptive post-research sensitivities, not independent confirmation,
multiplicity-adjusted tests, causal effects or guarantees of future stability.
No horizon/family is chosen because its interval or metric looks best.

Holiday snapshots retain their original last-trading-session dates. Weekly
identity for bootstrap spacing is `W-FRI`, not an assumption that every snapshot
is a Friday. PIT cutoffs, quarterly assignment and R6 hashes use the original
dates. Fully unknown-feature weeks remain in operational accounting but not as
zero-alpha observations in evidence means, intervals or concentration checks.
`TEST_FEATURE_UNAVAILABLE` is distinct from observed `NO_FLAGS`; all observed
and evidence denominators remain explicit.

## What the Conclusion Means

The decision matrix separates stop-risk direction from economic direction.
`HISTORICAL_ECONOMIC_DIRECTION` requires positive cash delta and random-relative
increment across all W1/W2/W4, at least three supported quarters each, and
positive worst leave-one-quarter-out, worst leave-one-ticker-out and
best-week-removed increments. It is deliberately **not** a production pass.
`MIXED_ECONOMIC_DIRECTION`, `ECONOMIC_VALUE_NOT_DEMONSTRATED` and
`INSUFFICIENT_EVIDENCE` remain legitimate outcomes. Do not respond to these by
changing thresholds, fees or selecting the best horizon.

This run cannot establish gap-aware realized stop P&L, actual funding/capacity,
replacement trades, live execution costs, portfolio drawdown/CAGR/Sharpe or
virgin-forward persistence. Frozen terminal facts do not contain those answers.
If mark-to-market economics are promising, actual stopped-execution economics
remain a separate approval requirement, not an assumption baked into this run.

## Execute on the Data Machine

Use Conda `quant_env`. No RD-Agent distribution, credentials or network is needed.
Use a new output directory. The example `20` bps is an explicit illustrative
round-trip assumption, not a calibrated or recommended trading cost.

```bash
conda run -n quant_env python -m pytest -q tests/test_blind_rule_discovery_r7.py

conda run -n quant_env python -m backtest.blind_rule_discovery.r7_runner \
  --samples backtest/blind_rule_discovery/output/trigger_path_characterization_r4/trigger_path_samples.csv \
  --metadata backtest/blind_rule_discovery/output/trigger_path_characterization_r4/trigger_path_metadata.json \
  --r6-dir backtest/blind_rule_discovery/output/r6_risk_features_06 \
  --output-root backtest/blind_rule_discovery/output/r7_economic_triage_01 \
  --round-trip-cost-bps 20 \
  --preflight
```

After preflight passes, execute the identical command without `--preflight`.
Adjust only the input paths to the actual matching artifacts. Do not pair R4
samples with metadata from another run. Preflight writes nothing and does not
call a model. A failed run must not be reported as successful merely because
some files exist: **`COMPLETE.json` must exist and all output hashes must match.**

Return the entire output directory for audit: report, input manifest, completion
hashes, frozen rules, economic/quarterly/weekly summaries, decision matrix,
label contributions, group outcomes, ticker concentration and event flags.
Do not submit only favorable tables. Inputs are rehashed before publication;
outputs never overwrite R4/R5/R6 data or reports.

**KEEP PRODUCTION FROZEN.**
