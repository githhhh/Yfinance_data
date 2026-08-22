# 2026-08-22 Main EPS Baseline Audit

## Goal

Minimal production convergence on `main`:

- keep formal strategy data on `main`;
- restore `quant_env` compatibility for `futu-api`;
- add a small EPS supplement layer for signal rows;
- preserve the current `skill_industry_eps_known` and `clean_eps_pass_no_dry_no_geom_caution` baselines as research reference data;
- do not merge RD/qlib/experimental skill logic into production.

## Execution Log

1. Confirmed production failure was dependency-related, not data update or Futu_OpenD availability:
   - `futu-api 9.1.5108`
   - broken state: `protobuf 6.33.6`
   - import failure: `TypeError: Descriptors cannot be created directly`
2. Repaired `quant_env` for production Futu compatibility:
   - command: `conda run --no-capture-output -n quant_env python -m pip install "protobuf==3.20.3"`
   - verified: `import futu` succeeds with `protobuf 3.20.3`
   - note: this conflicts with research dependencies such as `opentelemetry-proto` and `databricks-sdk`; research and production environments should be split later.
3. Fast-forwarded local `main` to `origin/main`.
   - `results_pkl/stock_data_220826_1d.pkl`
   - `results_pkl/stock_data_220826_1wk.pkl`
   - current `us/` sources were already present in `origin/main`.
4. Added lookup-only EPS supplement package:
   - `eps_pit/__init__.py`
   - `eps_pit/lookup.py`
5. Added production hook:
   - `BreakoutFollowPoolRun.save_snapshot()` now enriches missing `eps_yoy_growth` before writing pool CSV.
   - Enrichment is signal-only.
   - Source priority is point-in-time signal CSV first, then same-run Stage2 whitelist.
6. Replayed the same enrichment on current `us/breakout_follow_pool.csv`:
   - signal EPS missing before: 20
   - signal EPS missing after: 17
   - repaired codes: `AKR`, `MOG-A`, `TWIN`
   - unresolved rows remain blank rather than being filled from unsafe future/current-only sources.

## Baseline Artifacts

Reference outputs copied from the research branch without bringing in experiment code:

- `backtest/rd_agent_research_bench/output/backtrader_summary.csv`
- `backtest/rd_agent_research_bench/output/backtrader_decision_matrix.csv`

Key Backtrader baseline with initial capital `10000` and `-8%` stop:

| Mode | Variant | Final Value | Return | Max DD | Stops | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| with EPS | `skill_industry_eps_known` | 15470.50 | 54.71% | -16.90% | 5 | formal baseline |
| with EPS | `clean_eps_pass_no_dry_no_geom_caution` | 18198.79 | 81.99% | -6.61% | 2 | high-confidence candidate, not replacement |
| no EPS | `v3_core_top3` | 13354.29 | 33.54% | -19.92% | 7 | no-EPS baseline |

`clean_eps_pass_no_dry_no_geom_caution` is not promoted to production replacement because the decision matrix marks it as `high_confidence_candidate: failed coverage`:

- rebalance coverage ratio: `0.95`
- pick coverage ratio: `0.8712871287`

## Guardrails

- No experimental RD/qlib optimizer logic was merged into production.
- No rule change promotes `clean_eps_pass_no_dry_no_geom_caution`.
- EPS supplement does not rank or recommend tickers; it only repairs missing feature values for signal rows where an approved source exists.
- Remaining EPS blanks are treated as unresolved source gaps, not silently backfilled.
