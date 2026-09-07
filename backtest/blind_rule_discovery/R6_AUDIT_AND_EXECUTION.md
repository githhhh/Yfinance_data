# R1-R5 Code Audit and R6 Execution Handoff

Audited remote baseline: `cfbf98f` (2026-09-07).
Scope: source and committed reports. The long-history samples and raw R4/R5
outputs are unavailable here; no empirical result has been reproduced.
No price, EPS, pool, production selector, or historical outcome file is modified.

## Audit Findings

### P1: R5 selection used a label beyond its purge horizon

`stop_risk_validation_r5.rank_stop_risk_rules` previously broke ties using
`persistent_stop_first_lift`. `PathSearchContext` derives it from
`stop_first_then_winner_12w`, but rolling purges only `exit_date_w3`.
A row's W3 label may be known before the cutoff while its recovery is unknowable.
A synthetic regression changing only that metric reproduces a different rule.

The fix removes this metric from selection. W3 risk evidence, support, and
canonical rule JSON determine order. Persistent Stop remains descriptive and
never reaches R6 discovery. R5 metadata identifies the corrected procedure.
The original R5 strict-causal claim requires corrected execution. The actual
affected rules/folds are unknown here. Fixed semantic-family thresholds do not
use this tie breaker, so this finding alone does not invalidate their associations.
Original R1-R5 outputs/numbers are preserved, not silently restated.

### P2: pooled lift includes quarterly composition

Legacy pooled selected rates weight quarters by selected counts; baseline rates
weight them by universe counts. A synthetic case with zero lift in each quarter
has +32pp pooled lift. Original fields remain for compatibility. R5 now also emits
`selection_weighted_baseline_stop_first_rate`, `selection_weighted_stop_first_lift`
and corresponding Persistent Stop fields, using selected evaluable counts for
both arms. This controls quarterly composition only. R6 primarily reports
equal-snapshot selected-versus-complement contrasts.

### Population discrepancy remains unresolved

The committed final report records different R4/R5 usable-entry counts without
matching replay hashes. This audit cannot identify the cause without those files.
R6 requires one explicit sample CSV and its matching metadata: supplied samples
SHA256, exact row count, unique keys, exclusive/exhaustive labels, ordered dates,
and matching entry-quarter inventory. The replay SHA is declared provenance, not
independently recomputed here. Never pair R4 samples with unrelated R5 metadata.

## Research Judgment

- Retain R1's failed holdout result and consumed status.
- Retain the descriptive market-versus-stock distinction.
- Retain deep-pullback/above-ceiling as retrospective risk hypotheses.
- Downgrade strict causal-purge claims for R5 searched rules pending correction.
- R4's failed quality-score search does not prove current PIT fields are useless.
- Stop First detection does not imply profitable removal; report winner loss.
- Further historical research may prioritize a prospective hypothesis, but cannot
  restore untouched evidence to inspected quarters.

## R6 Contract

R6 is **known-history adaptive retrospective research**: a custom local loop using
the official RD-Agent LiteLLM backend for proposals. It is **not** `fin_factor`,
CoSTEER, or canonical R1. Deterministic R2/R3 protocols stay unchanged.

This machine has no usable RD-Agent distribution metadata. Live integration has
not run here; backend wiring and request accounting are tested with stubs. The
data machine must pass backend preflight. No dependency was installed and no
substitute model client is allowed.

### Objective and Expressions

Identify W3 Stop First risk while measuring Fast Winner removal cost; compare
against single-feature baselines. Inputs use the current snapshot-PIT allowlist.
Exclude market fields, B0 membership/rank, identifiers as predictors, and
entry-delay/extension facts unknown at snapshot time.

Population is **usable executable entries**, conditional on reconstructed universe
and prior maturity filtering. It is not all signals/listings or portfolio P&L;
CDFs fitted here are not ready-to-deploy all-signal probabilities.

- Leaves: raw feature or training empirical percentile.
- Combinations: difference/product/minimum/maximum; tree depth <=2.
- High/low tail; threshold quantile only 0.2/0.4/0.6/0.8.
- Up to 3 proposals per round, including mechanisms and falsifying observations.
- Weak single features may enter interactions; no univariate pre-pruning.
- JSON interpreter only; never execute model-generated Python.
- Do not infer EPS trajectories or earnings surprises from one YoY value.

### Time and Selection

1. Consecutive **snapshot-quarter** calendar, including empty quarters. This
   differs from legacy entry-quarter reporting and must not be conflated.
2. Begin outer evaluation after six quarters. Purge unknown/crossing W3 exit dates
   for each cutoff. No W4/12w labels enter discovery.
3. Expanding inner folds fit transformations/quantiles using preceding quarters
   only. Their feedback is adaptively reused, not independent validation.
4. Inner support: >=10 flagged evaluable, >=10 complementary, >=3 matched snapshots
   per quarter. Qualification: >=3 supported inner quarters; positive stop lift in
   >=2/3 of all inner quarters; positive median matched stop lift and positive
   median stop-capture minus winner-loss. These are research support gates, not
   production thresholds or statistical significance.
5. Freeze qualifying Agent candidate and simple comparator separately. Abstain
   when none qualifies; no forced fallback.
6. Freeze **all outer rules before evaluating any outer fold**. Later folds may
   use earlier rows only after labels mature; outer results are never copied into
   proposal feedback. This is retrospective walk-forward, not untouched OOS.

### Budget and Stopping

Default: six rounds/fold, <=3 proposals/round. Stop after three rounds without
improvement in fixed inner ranking or an explicit empty proposal list.
`42/1000` is user-reported usage, not API-verified balance. Every underlying
completion, including backend retries/continuations, reserves a ledger unit before
sending. SDK retries are disabled; failures consume units. Accounted ceiling: 1000
including prior usage. Default invocation ceiling: 120 attempts, one-hour request
admission ceiling, 90-second request timeout. Cached responses do not spend again.

Always reuse the same ledger/cache paths; never reset them or run concurrent
writers. Other clients' provider usage is not observed: reconcile it before a run.
Caps are not targets. Backend/JSON/budget failure writes `FAILED.json`, preserves
paid cache, and publishes no successful report. Another attempt needs a fresh
output root. Do not change research settings to improve disappointing results.

## Run on the Data Machine

From repository root in `quant_env`, use the existing R4 CSV and **its matching**
`trigger_path_metadata.json`. No pool/price/EPS reconstruction is performed.

```bash
conda activate quant_env
python -m pytest -q tests/test_blind_rule_discovery*.py

R6_SAMPLES=backtest/blind_rule_discovery/output/trigger_path_characterization_r4/trigger_path_samples.csv
R6_META=backtest/blind_rule_discovery/output/trigger_path_characterization_r4/trigger_path_metadata.json
R6_SHA=$(openssl dgst -sha256 "$R6_SAMPLES" | awk '{print $NF}')

python -m backtest.blind_rule_discovery.r6_runner \
  --samples "$R6_SAMPLES" --metadata "$R6_META" --samples-sha256 "$R6_SHA" \
  --output-root backtest/blind_rule_discovery/output/r6_risk_features_01 \
  --ledger backtest/blind_rule_discovery/work/r6_requests.json \
  --cache-dir backtest/blind_rule_discovery/work/r6_response_cache \
  --prior-used 42 --run-call-cap 120 --rounds 6 --preflight
```

After preflight passes, run the identical command with `--preflight` removed.
Use existing `RD_AGENT_MODEL` or `CHAT_MODEL`, `DEEPSEEK_API_KEY`, and
`DEEPSEEK_API_BASE`. Credentials are not printed. Backend incompatibility is a
setup failure, not permission for the executor to patch an ad-hoc agent.

## Return These Artifacts

- `input_manifest.json`: hashes, counts/calendar, provenance limitations, execution
  HEAD/configuration, backend/version and requests.
- `discovery_trace.jsonl`: proposals, rejections, prompt digests and accepted IDs.
- `frozen_rules.json`: train-surface hashes, thresholds, inner evidence, abstentions.
- `outer_quarters.csv`: all folds, coverage, ambiguity/unresolved counts, risk
  capture, winner loss and matched-snapshot lift.
- `agent_vs_simple.csv`: common-supported-quarter deltas, including coverage.
- `feature_stability.csv`: recurrence and supported directional folds.
- `summary.json`, `R6_REPORT.md`: mechanically generated results and limitations.

Do not hand-copy counts or infer significance from effect size. Recurrence is not
independent attribution. Same-snapshot contrasts leave stock-level confounding;
repeat tickers and overlapping paths remain dependent. Positive findings require
prospective evidence before a production decision.

**KEEP PRODUCTION FROZEN.**
