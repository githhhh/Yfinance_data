# R8 Winner / Stop Stable Feature Atlas — Split Execution V2

## Goal

R1-R7 found that stable winner alpha is difficult to identify, while several PIT
features describe stop-risk pockets. R8 does not search for another ranking
champion. Its primary deliverable is the missing Winner/Stop feature atlas:

- which snapshot-PIT features differ consistently between `fast_winner_3w` and
  `stop_first_3w`;
- how large the separation is and whether it survives calendar-quarter and
  repeated-ticker controls;
- whether q20/q80 tails or fixed quintile surfaces reveal nonlinear structure;
- whether bounded two-feature mechanisms recur when proposed from past-only facts.

`unresolved_3w` and `ambiguous_3w` are never relabeled. R8 is known-history
retrospective research, not untouched OOS, causal proof, or a production selector.

## Why R8 is now split

The original combined runner allowed a transport failure in the optional RD-Agent
layer to prevent publication of the deterministic atlas. Two formal attempts
failed before the first successful model response because the configured API
proxy terminated the connection at about 60 seconds before the reasoning model
returned its first byte.

That transport limitation must not invalidate deterministic research that needs no
model at all. R8 therefore has two independently auditable stages:

1. **R8A deterministic atlas** — fully offline after the bound CSVs are present;
   no API key, RD-Agent call, request ledger or cache is required.
2. **R8B optional Agent interactions** — runs only after binding to a completed
   R8A directory. R8B failure writes its own `AGENT_FAILED` artifact and never
   invalidates R8A.

No model, outcome definition, stability threshold or production rule was changed
to rescue historical results.

## Bound inputs

Both stages use exactly the R4 `trigger_path_samples.csv` and matching metadata
bound by the completed R6 `input_manifest.json`. They reproduce the R6 sample SHA,
metadata SHA, feature list, calendar, row count, snapshot-week count and ticker
count. The completed R6 frozen-rule hash and research mode are also checked.

The population is the existing usable executable entries only, not every original
signal or listing. No price download, EPS refresh, B0 rank, market field,
execution-time fact or future field is added.

# R8A — Deterministic full feature atlas

Every R6-allowlisted PIT feature is published. There is no top-N prefilter and no
feature selection before publication.

## Fixed panels

Every atlas table is generated for both:

- `all_entries` — every usable executable entry;
- `nonoverlap_w3` — an outcome-independent issuer schedule: admit a ticker's
  earliest entry, reserve the ticker through `exit_date_w3`, forbid same-close-date
  re-entry, then allow a later entry only after the reservation has ended.

`nonoverlap_w3` is the primary report panel. `all_entries` remains a sensitivity
panel. The admission schedule never reads a label or feature value.

## Four-class raw profiles

For every calendar quarter, feature, panel and W3 path class, export:

- class N / known N / missing fraction;
- q10 / q25 / median / q75 / q90;
- finite mean.

Winner, Stop, Unresolved and Ambiguous all remain visible.

## Winner-vs-Stop chronological contrasts

Starting after six consecutive calendar quarters, each test quarter uses a purged
past surface (`snapshot_date < quarter_start` and `exit_date_w3 < quarter_start`).
For every feature and panel export:

- raw Cliff's delta, Winner minus Stop;
- empirical past-percentile Winner/Stop means and gap;
- equal-snapshot Winner-minus-Stop percentile gap and direction;
- Winner and Stop missing fractions;
- past-only q20/q80 thresholds;
- low/high tail Winner-vs-Stop log-odds relative to feature-known primary rows.

A quarter is supported only with at least 8 known Winners, 8 known Stops and 3
matched snapshot weeks containing both classes. Unsupported and empty quarters
remain explicit.

## Predeclared descriptive stability

Within a panel, a feature is `CONSISTENT_WINNER_HIGH` or
`CONSISTENT_STOP_HIGH` only when:

- at least 6 supported outer quarters;
- at least 75% of supported quarters have the same matched-week direction, with
  zero-direction quarters counting against consistency;
- `|median Cliff's delta| >= 0.10`;
- median matched percentile gap and median Cliff's delta have the same sign;
- deleting any one supported quarter does not flip the median matched-gap sign.

Everything else is `MIXED_OR_WEAK` or `INSUFFICIENT_EVIDENCE`. These are
**descriptive labels**, not p-value passes, alpha estimates or production weights.

## Fixed quintile surface

For each outer quarter, feature and panel, q20/q40/q60/q80 boundaries are fitted
from that panel's purged past only. Every test row is placed into a fixed bin and
all four W3 class rates plus Winner share within Winner/Stop are exported. No bin
is selected or promoted. Degenerate bins remain visible.

## R8A outputs

A successful R8A directory contains:

- `class_profiles.csv`
- `quarter_feature_contrasts.csv`
- `feature_stability.csv`
- `quintile_surfaces.csv`
- `input_manifest.json`
- `R8_ATLAS_REPORT.md`
- `COMPLETE.json` with `status=ATLAS_COMPLETE` and `rdagent_calls=0`

Every published output is SHA256-bound by `COMPLETE.json`.

# R8B — Optional compact RD-Agent interactions

R8B requires a completed R8A directory and verifies every R8A output hash plus the
R4/R6 bindings before making any model call. Its output root is separate.

## Dependency-control panel

Interaction discovery and outer evaluation use `nonoverlap_w3` only. This is
predeclared and reduces repeated-issuer amplification. R8A retains the full-entry
sensitivity atlas separately.

## Compact prompt contract

The previous verbose prompt is replaced by `R8_COMPACT_INTERACTION_V1`. Every
allowlisted PIT feature still appears; there is **no feature prefilter**. Each
feature is represented by a short fixed row containing:

`feature, Winner-known N, Stop-known N, Cliff delta, supported-quarter count,
Winner-high quarter fraction, Winner-minus-Stop missingness gap`.

Later-round feedback returns only aggregate inner evidence. Full inner-quarter
evidence remains preserved in `interaction_frozen.json`; it is simply not echoed
back into the model prompt. `discovery_trace.jsonl` records `prompt_chars` for
transport audit.

This reduces token and reasoning burden without hiding weak features or choosing a
historical top-N subset.

## Bounded interaction contract

Each proposal has exactly:

`name, hypothesis, expression, target, tail, quantile`

- `target`: `winner` or `stop`;
- only `low/q20` or `high/q80` extreme tails;
- expression leaves: `raw` or `train_percentile` of an allowlisted PIT feature;
- binary nodes: `difference`, `product`, `minimum`, `maximum`;
- maximum expression depth 2;
- exactly two distinct PIT features.

No Python, arbitrary threshold, date, ticker, market field or future fact is
allowed. Thresholds are fitted from the corresponding purged past surface only.

## Discovery and freeze

- 3 formal rounds maximum per non-empty outer fold;
- at most 2 proposals per round;
- patience 2;
- at most 3 qualifying interactions frozen per fold;
- qualification requires >=3 supported inner quarters, >=2/3 positive target
  direction across evaluated inner quarters, positive median equal-snapshot target
  lift and positive median `target_capture - other_class_loss`;
- **all fold rules freeze before any outer evaluation**;
- outer results never enter proposal feedback.

The cache identity includes the exact model identifier. Streaming, 8192 output
tokens, 240-second client timeout, three metered transport retries and disabled
reasoning auto-continue remain unchanged. These settings cannot override an
upstream proxy first-byte timeout, which is why R8B is optional and isolated.

The existing R8 ledger/cache must be reused after a failed attempt. The two prior
failed attempts therefore remain counted; they are not reset or rewritten.

## R8B outputs

On success:

- `discovery_trace.jsonl`
- `interaction_frozen.json`
- `interaction_outer.csv`
- `interaction_stability.csv`
- `interaction_feature_recurrence.csv`
- `input_manifest.json`
- `R8_INTERACTION_REPORT.md`
- `COMPLETE.json` with `status=AGENT_COMPLETE`

On transport/model failure, the R8B output root writes `FAILED.json` with
`status=AGENT_FAILED` and `atlas_invalidated=false`. The completed R8A directory
remains valid and publishable.

## Interpretation

The most valuable R8 outcome may be stable Winner/Stop descriptive features with
no ranking alpha, nonlinear tail structure, an issuer-overlap-sensitive effect,
recurring two-feature mechanisms, or no robust separation at all.

Do not convert a descriptive feature or Agent recurrence into a B0 bonus/penalty
automatically. Any future production hypothesis must be frozen before genuinely
future observations arrive.

**KEEP PRODUCTION FROZEN.**
