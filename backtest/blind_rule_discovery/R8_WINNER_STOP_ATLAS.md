# R8 Winner / Stop Stable Feature Atlas

## Goal

R1-R7 found that stable winner alpha is difficult to identify, while several PIT
features describe stop-risk pockets. R8 changes the question. It does **not** ask
for another ranking winner. It builds the missing descriptive atlas:

- Which snapshot-PIT features consistently differ between `fast_winner_3w` and
  `stop_first_3w` across calendar quarters?
- How large is the separation, how much do the class distributions overlap, and
  is the direction stable after matching within snapshot weeks?
- Do fixed past-only q20/q80 tails or quintile surfaces reveal monotone,
  threshold-like, or U-shaped behavior that a single score would hide?
- Can a small number of bounded two-feature interactions improve class
  separation on the next quarter without being promoted as alpha?

The primary binary contrast is **Fast Winner vs Stop First**. `unresolved_3w` and
`ambiguous_3w` are never relabeled; they remain visible in class profiles and
quintile surfaces. R8 is known-history retrospective research, not untouched OOS,
not causal proof, and not a production selector.

## Bound Inputs

Use exactly the R4 `trigger_path_samples.csv` and matching metadata bound by the
completed R6 `input_manifest.json`. R8 must reproduce the R6 sample SHA, metadata
SHA, feature list, calendar, row count, snapshot-week count and ticker count.
Only the R6 snapshot-PIT feature allowlist is profiled or exposed to the Agent.
No price download, EPS refresh, B0 rank, market field, ticker identity, date,
execution-time fact or future field may enter a feature expression.

The source population is the existing 8,983 usable executable entries, not all
listings or all original signals. That conditioning remains an explicit limit.

## Layer A — Fixed Full Feature Atlas

Every allowlisted PIT feature is evaluated. There is no top-N prefilter and no
feature selection before publication.

### 1. Four-class raw profile

For every calendar quarter, feature and W3 path class, export:

- class N / known N / missing fraction;
- q10 / q25 / median / q75 / q90;
- raw mean where finite.

This table includes Winner, Stop, Unresolved and Ambiguous classes.

### 2. Winner-vs-Stop chronological contrasts

Starting only after six consecutive calendar quarters, each test quarter uses a
purged past surface (`snapshot_date < quarter_start` and `exit_date_w3 <
quarter_start`). For every feature:

- raw Cliff's delta, Winner minus Stop;
- Winner/Stop values mapped to the empirical percentile of the purged past;
- mean and median percentile gap;
- equal-snapshot Winner-minus-Stop percentile gap, its median and sign fraction;
- Winner and Stop missing fractions;
- fixed past q20/q80 thresholds and class-enrichment log-odds in each tail,
  relative to feature-known Winner/Stop rows only so missingness cannot create a
  false enrichment signal.

A quarter is supported only with at least 8 known Winners, 8 known Stops and 3
matched snapshot weeks containing both classes. Unsupported and empty quarters
remain explicit.

### 3. Predeclared descriptive stability label

A feature is `CONSISTENT_WINNER_HIGH` or `CONSISTENT_STOP_HIGH` only when:

- at least 6 supported outer quarters;
- at least 75% of supported quarters have the same matched-week direction (zero
  direction counts against consistency);
- median absolute Cliff's delta is at least 0.10;
- the median matched percentile gap and median Cliff's delta have the same sign;
- deleting any one supported quarter does not flip the sign of the median matched
  percentile gap.

Otherwise it is `MIXED_OR_WEAK`; insufficient support is separate. These are
**descriptive labels**, not hypothesis-test passes or production weights. No
p-value or multiplicity-adjusted significance claim is made.

### 4. Fixed quintile surface

For every outer quarter and feature, q20/q40/q60/q80 boundaries are fitted from
purged past only. Each test row is placed into one of five bins. Export all four
W3 class rates and Winner-vs-Stop share per bin. No bin is chosen or promoted.
Degenerate bins for discrete/binary features remain visible rather than being
silently redefined. This is specifically intended to expose non-monotone
structure that rank scores can hide.

## Layer B — RD-Agent Interaction Discovery

R8 may use the remaining provider allowance, but Agent output is subordinate to
the fixed atlas.

### Prompt boundary

For each outer quarter after the first six calendar quarters, the Agent sees only:

- compact aggregates computed from purged past;
- per-feature Winner/Stop counts, Cliff's delta, missingness and quarter-direction
  consistency;
- inner-quarter feedback for interaction proposals from the same past surface.

It never sees the outer-quarter outcomes before rules for that quarter are frozen.
Later folds may use earlier labels after they have matured under the same W3 purge.

### Bounded expression contract

Each proposal has exactly:

`name, hypothesis, expression, target, tail, quantile`

- `target`: `winner` or `stop`;
- only the extreme tail pairs are legal: `tail=low, quantile=0.2` or
  `tail=high, quantile=0.8`;
- expression leaves: `raw` or `train_percentile` of an allowlisted PIT feature;
- binary nodes: `difference`, `product`, `minimum`, `maximum`;
- max expression depth 2;
- **exactly two distinct PIT features** must occur in the expression.

No Python, arbitrary thresholds, generated code or inferred earnings trajectory is
executed. A threshold is always fitted from the corresponding past training
surface.

### Fixed discovery budget

- maximum 3 rounds per outer fold;
- maximum 2 proposals per round;
- patience 2 rounds;
- at most 3 qualifying interactions frozen per fold;
- no Agent call for an empty test quarter;
- run provider-attempt cap 80;
- global provider budget remains 1000 using the completed R6 accounted usage as a
  lower-bound prior. Provider-account usage remains explicitly unverified.

The request cache identity includes the exact model identifier in addition to the
system and prompt hashes, preventing the R6 alias/cache ambiguity from recurring.
All retries are metered. Streaming, 8192 output tokens, 240-second provider timeout,
three transport retries and disabled reasoning auto-continue retain the audited R6
compatibility behavior.

### Inner qualification

For a proposed target pocket, each inner quarter compares selected vs complement
only among Fast Winner and Stop First rows. Qualification requires:

- at least 3 supported inner quarters;
- positive target enrichment in at least two thirds of all evaluated inner
  quarters;
- positive median equal-snapshot target-rate lift;
- positive median `target_capture - other_class_loss`.

Every qualifying interaction is ranked by this fixed evidence tuple only to cap
output size; up to three are frozen. This is not champion selection. Outer results
are evaluated only after all fold rules have been frozen and are never copied into
proposal feedback.

## Outputs

A successful run publishes the complete directory, including:

- `class_profiles.csv`
- `quarter_feature_contrasts.csv`
- `feature_stability.csv`
- `quintile_surfaces.csv`
- `discovery_trace.jsonl`
- `interaction_frozen.json`
- `interaction_outer.csv`
- `interaction_stability.csv`
- `interaction_feature_recurrence.csv`
- `input_manifest.json`
- `R8_REPORT.md`
- `COMPLETE.json`

`COMPLETE.json` hashes every published output. A failed run writes `FAILED.json`
and is never reported as complete. Use a fresh output root for every attempt and
reuse the same R8 request ledger/cache after failure.

## Interpretation

The most valuable R8 outcome can legitimately be:

- a small set of consistently Winner-high or Stop-high **descriptive** features;
- stable tail/nonlinear structure without a profitable ranking alpha;
- no stable univariate feature but recurring interaction mechanisms;
- or no robust separation at all.

Do not change thresholds after seeing R8, do not convert a stable descriptive
feature into a B0 penalty/bonus automatically, and do not treat Agent recurrence
as independent validation. Any prospective production hypothesis must be frozen
before genuinely future observations arrive.

**KEEP PRODUCTION FROZEN.**
