"""Final frozen execution runner for R4 trigger-path characterization.

This runner combines the audited full-path characterization layer with the
vectorized exact stock-interaction search. Search space is unchanged: every
supported single condition and every distinct-feature two-condition pair is tested.
Only expensive W1-W4 distribution enrichment is deferred to reported top rules.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .dataset import load_replay_candidates
from .outcomes import OutcomeConfig, load_price_pickle, restrict_to_mature_outcome_quarters
from .pipeline_contract import validate_replay_preflight
from .trigger_path_characterization import (
    EXECUTION_FEATURES,
    R3_FAVORABLE_REGIME,
    SEARCH_QUANTILES,
    _scope_frame,
    build_trigger_path_frame,
    stock_feature_columns,
    within_snapshot_feature_contrasts,
)
from .trigger_path_characterization_r4 import (
    add_matched_metrics,
    feature_bin_characterization,
    interaction_views,
    matched_selected_vs_unselected,
    parse_args,
    rule_mask,
    summarize_path,
)
from .trigger_path_characterization_r4_search import enrich_full_metrics, search_stock_interactions_fast

QUALITY_OUTPUT_LIMIT = 500
WINNER_OUTPUT_LIMIT = 200
STOP_RISK_OUTPUT_LIMIT = 200


def _empty_fold_row(
    *,
    scope: str,
    test_quarter: str,
    train_quarters: Sequence[str],
    test: pd.DataFrame,
    train_insufficient: bool,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scope": scope,
        "test_quarter": test_quarter,
        "train_start_quarter": train_quarters[0],
        "train_end_quarter": train_quarters[-1],
        "train_quarter_count": len(train_quarters),
        "test_quarter_in_train": False,
        "train_insufficient": int(train_insufficient),
        "rule_json": None,
        "train_quality_score": None,
        "train_path_edge": None,
        "test_scope_n": int(len(test)),
        "test_selected_n": 0,
        "test_evaluable_n": 0,
        **{f"test_baseline_{key}": value for key, value in baseline.items()},
    }


def rolling_scope_search(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    scope: str,
    min_train_quarters: int,
    min_quarter_n: int,
) -> pd.DataFrame:
    quarters = sorted(frame["entry_quarter"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    for test_index in range(min_train_quarters, len(quarters)):
        train_quarters = quarters[:test_index]
        test_quarter = quarters[test_index]
        train_all = frame.loc[frame["entry_quarter"].astype(str).isin(train_quarters)].reset_index(drop=True)
        test_all = frame.loc[frame["entry_quarter"].astype(str) == test_quarter].reset_index(drop=True)
        train = _scope_frame(train_all, scope)
        test = _scope_frame(test_all, scope)
        baseline = summarize_path(test)
        if len(train) < 80:
            rows.append(
                _empty_fold_row(
                    scope=scope,
                    test_quarter=test_quarter,
                    train_quarters=train_quarters,
                    test=test,
                    train_insufficient=True,
                    baseline=baseline,
                )
            )
            continue
        scored, _ = search_stock_interactions_fast(
            train,
            features,
            min_selected=60 if scope == "all" else 30,
            min_evaluable=45 if scope == "all" else 20,
            min_quarter_n=max(5, min_quarter_n // 2),
            min_evaluated_quarters=4 if scope == "all" else 2,
        )
        if scored.empty:
            rows.append(
                _empty_fold_row(
                    scope=scope,
                    test_quarter=test_quarter,
                    train_quarters=train_quarters,
                    test=test,
                    train_insufficient=False,
                    baseline=baseline,
                )
            )
            continue
        best = scored.iloc[0]
        rule_json = str(best["rule_json"])
        test_metrics = summarize_path(test, rule_mask(test, rule_json))
        fast_lift = None
        stop_reduction = None
        path_edge = None
        if (
            test_metrics["fast_winner_rate"] is not None
            and baseline["fast_winner_rate"] is not None
            and test_metrics["stop_first_rate"] is not None
            and baseline["stop_first_rate"] is not None
        ):
            fast_lift = float(test_metrics["fast_winner_rate"] - baseline["fast_winner_rate"])
            stop_reduction = float(baseline["stop_first_rate"] - test_metrics["stop_first_rate"])
            path_edge = fast_lift + stop_reduction
        matched = matched_selected_vs_unselected(test, rule_json) if not test.empty else {}
        rows.append(
            {
                "scope": scope,
                "test_quarter": test_quarter,
                "train_start_quarter": train_quarters[0],
                "train_end_quarter": train_quarters[-1],
                "train_quarter_count": len(train_quarters),
                "test_quarter_in_train": test_quarter in train_quarters,
                "train_insufficient": 0,
                "rule_json": rule_json,
                "train_quality_score": best["quality_score"],
                "train_path_edge": best["path_edge"],
                "test_scope_n": int(len(test)),
                **{f"test_{key}": value for key, value in test_metrics.items()},
                **{f"test_baseline_{key}": value for key, value in baseline.items()},
                "test_fast_winner_lift": fast_lift,
                "test_stop_first_reduction": stop_reduction,
                "test_path_edge": path_edge,
                **{f"test_{key}": value for key, value in matched.items()},
            }
        )
    return pd.DataFrame(rows)


def rolling_stock_selection(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    min_train_quarters: int,
    min_quarter_n: int,
) -> pd.DataFrame:
    parts = [
        rolling_scope_search(
            frame,
            features,
            scope=scope,
            min_train_quarters=min_train_quarters,
            min_quarter_n=min_quarter_n,
        )
        for scope in ("all", "r3_favorable")
    ]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summarize_rolling(rolling: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if rolling.empty:
        return summary
    for scope, group in rolling.groupby("scope", sort=True):
        def numeric(column: str) -> pd.Series:
            if column not in group.columns:
                return pd.Series(np.nan, index=group.index, dtype=float)
            return pd.to_numeric(group[column], errors="coerce")

        selected = numeric("test_selected_n").fillna(0)
        evaluable = numeric("test_evaluable_n").fillna(0)
        fast = numeric("test_fast_winner_n").fillna(0)
        stop = numeric("test_stop_first_n").fillna(0)
        base_eval = numeric("test_baseline_evaluable_n").fillna(0)
        base_fast = numeric("test_baseline_fast_winner_n").fillna(0)
        base_stop = numeric("test_baseline_stop_first_n").fillna(0)
        edge = numeric("test_path_edge")
        matched_edge = numeric("test_matched_path_edge_p50")
        insufficient = numeric("train_insufficient").fillna(0).astype(int)

        selected_eval_total = float(evaluable.sum())
        selected_fast_rate = float(fast.sum() / selected_eval_total) if selected_eval_total else None
        selected_stop_rate = float(stop.sum() / selected_eval_total) if selected_eval_total else None

        baseline_eval_all = float(base_eval.sum())
        baseline_fast_all = float(base_fast.sum() / baseline_eval_all) if baseline_eval_all else None
        baseline_stop_all = float(base_stop.sum() / baseline_eval_all) if baseline_eval_all else None

        traded = selected > 0
        baseline_eval_traded = float(base_eval.loc[traded].sum())
        baseline_fast_traded = (
            float(base_fast.loc[traded].sum() / baseline_eval_traded) if baseline_eval_traded else None
        )
        baseline_stop_traded = (
            float(base_stop.loc[traded].sum() / baseline_eval_traded) if baseline_eval_traded else None
        )
        matched_pooled_edge = (
            (selected_fast_rate - baseline_fast_traded) + (baseline_stop_traded - selected_stop_rate)
            if None not in {selected_fast_rate, selected_stop_rate, baseline_fast_traded, baseline_stop_traded}
            else None
        )
        summary[scope] = {
            "folds": int(len(group)),
            "insufficient_training_folds": int(insufficient.sum()),
            "insufficient_training_fraction": float(insufficient.mean()),
            "zero_selection_folds": int((selected == 0).sum()),
            "zero_selection_fraction": float((selected == 0).mean()),
            "evaluable_path_edge_folds": int(edge.notna().sum()),
            "positive_path_edge_all_fold_fraction": float((edge.fillna(-np.inf) > 0).mean()),
            "positive_path_edge_evaluable_fraction": float((edge.dropna() > 0).mean()) if edge.notna().any() else None,
            "path_edge_p50_evaluable": float(edge.dropna().median()) if edge.notna().any() else None,
            "matched_positive_edge_fold_fraction": float((matched_edge.dropna() > 0).mean()) if matched_edge.notna().any() else None,
            "selected_n_total": int(selected.sum()),
            "evaluable_n_total": int(evaluable.sum()),
            "pooled_selected_fast_winner_rate": selected_fast_rate,
            "pooled_selected_stop_first_rate": selected_stop_rate,
            "pooled_baseline_fast_winner_rate_all_folds": baseline_fast_all,
            "pooled_baseline_stop_first_rate_all_folds": baseline_stop_all,
            "pooled_baseline_fast_winner_rate_selected_folds": baseline_fast_traded,
            "pooled_baseline_stop_first_rate_selected_folds": baseline_stop_traded,
            "pooled_matched_path_edge_selected_folds": matched_pooled_edge,
        }
    return summary


def _write_scope_outputs(
    output_root: Path,
    scope: str,
    scoped: pd.DataFrame,
    compact: pd.DataFrame,
) -> None:
    quality = enrich_full_metrics(scoped, compact, top_n=QUALITY_OUTPUT_LIMIT)
    quality = add_matched_metrics(scoped, quality, top_n=100)
    winner_compact, stop_compact = interaction_views(compact)
    winner = enrich_full_metrics(scoped, winner_compact, top_n=WINNER_OUTPUT_LIMIT)
    winner = add_matched_metrics(scoped, winner, top_n=100)
    stop = enrich_full_metrics(scoped, stop_compact, top_n=STOP_RISK_OUTPUT_LIMIT)
    stop = add_matched_metrics(scoped, stop, top_n=100)
    quality.to_csv(output_root / f"stock_interactions_{scope}.csv", index=False)
    winner.to_csv(output_root / f"winner_interactions_{scope}.csv", index=False)
    stop.to_csv(output_root / f"stop_risk_interactions_{scope}.csv", index=False)


def main() -> int:
    args = parse_args()
    immutable = {
        Path("backtest/blind_rule_discovery/output/rd_agent_run_01").resolve(),
        Path("backtest/blind_rule_discovery/output/retrospective_ceiling_r2").resolve(),
        Path("backtest/blind_rule_discovery/output/retrospective_ceiling_r3").resolve(),
    }
    if args.output_root.resolve() in immutable:
        raise RuntimeError("historical R1/R2/R3 output is immutable and may not be reused")
    args.output_root.mkdir(parents=True, exist_ok=True)

    provenance = validate_replay_preflight(args.replay_root, daily_pkl=args.daily_pkl, required_quarters=12)
    prices = load_price_pickle(args.daily_pkl, require_adjusted=True)
    if args.spy_code not in prices:
        raise KeyError(f"benchmark {args.spy_code!r} missing from daily price bundle")
    candidates_all = load_replay_candidates(args.replay_root)
    config = OutcomeConfig()
    candidates, immature, maturity_cutoff = restrict_to_mature_outcome_quarters(
        candidates_all,
        prices[args.spy_code],
        minimum_sessions=config.minimum_sessions + config.entry_window_sessions,
    )
    frame, reviewer = build_trigger_path_frame(candidates, prices, prices[args.spy_code], config=config)
    features = stock_feature_columns(frame)
    if not features:
        raise ValueError("no stock/execution features available for R4")
    if any(feature.startswith("M_") for feature in features):
        raise AssertionError("market feature leaked into R4 stock feature list")

    frame.to_csv(args.output_root / "trigger_path_samples.csv", index=False)
    bins, extremes = feature_bin_characterization(frame, features, min_quarter_n=args.min_quarter_n)
    bins.to_csv(args.output_root / "feature_bin_characterization.csv", index=False)
    extremes.to_csv(args.output_root / "feature_extremes.csv", index=False)
    within_snapshot_feature_contrasts(frame, features).to_csv(
        args.output_root / "within_snapshot_feature_contrasts.csv", index=False
    )

    search_audit: dict[str, Any] = {}
    for scope in ("all", "r3_favorable"):
        scoped = _scope_frame(frame, scope)
        compact, audit = search_stock_interactions_fast(
            scoped,
            features,
            min_selected=args.min_selected_all if scope == "all" else args.min_selected_favorable,
            min_evaluable=args.min_evaluable_all if scope == "all" else args.min_evaluable_favorable,
            min_quarter_n=args.min_quarter_n,
            min_evaluated_quarters=(
                args.min_evaluated_quarters_all if scope == "all" else args.min_evaluated_quarters_favorable
            ),
        )
        _write_scope_outputs(args.output_root, scope, scoped, compact)
        search_audit[scope] = audit

    rolling = rolling_stock_selection(
        frame,
        features,
        min_train_quarters=args.rolling_min_train_quarters,
        min_quarter_n=args.min_quarter_n,
    )
    rolling.to_csv(args.output_root / "rolling_stock_selection.csv", index=False)

    censor_reasons = (
        reviewer.loc[~reviewer["usable"].fillna(False), "reason"].fillna("unknown").value_counts().to_dict()
        if not reviewer.empty
        else {}
    )
    metadata = {
        "research_mode": "r4_trigger_path_winner_loser_characterization_final_frozen",
        "canonical_blind_experiment": False,
        "unseen_holdout_claim_allowed": False,
        "llm_used": False,
        "primary_outcome": "+20% before -8% within 15 trading sessions from executable entry",
        "primary_denominator": "all full-path non-ambiguous entries; unresolved remains in denominator",
        "ambiguous_handling": "excluded from path probability denominator and reported separately",
        "secondary_outcomes": [
            "-8% before +20% within 15 sessions",
            "persistent stop-first vs stop-first-then-12w-recovery",
            "W1/W2/W3/W4 return and excess return p25/p50/p75",
            "3w/4w MAE and MFE",
        ],
        "market_control": {
            "frozen_r3_favorable_regime": R3_FAVORABLE_REGIME,
            "favorable_regime_is_retrospective_context_only": True,
            "within_snapshot_comparison": "same snapshot_date fast-winner vs stop-first stock feature contrast",
            "market_features_forbidden_in_stock_condition_search": True,
        },
        "search_contract": {
            "search_space": "all supported singles plus all distinct-feature q20/q40/q60/q80 pairs",
            "pair_pruning": False,
            "candidate_metrics": "compact vectorized search metrics",
            "reported_rule_enrichment": "full W1-W4 p25/p50/p75 only after ranking",
            "quality_output_limit": QUALITY_OUTPUT_LIMIT,
            "winner_output_limit": WINNER_OUTPUT_LIMIT,
            "stop_risk_output_limit": STOP_RISK_OUTPUT_LIMIT,
        },
        "rolling_contract": {
            "all_chronological_folds_preserved": True,
            "insufficient_training_fold_is_not_dropped": True,
            "selected_fold_baseline_reported_separately": True,
            "thresholds_regenerated_from_past_only": True,
        },
        "replay_dataset_sha256": provenance.get("replay_dataset_sha256"),
        "candidate_rows_before_maturity_filter": int(len(candidates_all)),
        "candidate_rows": int(len(candidates)),
        "excluded_immature_rows": int(len(immature)),
        "outcome_maturity_cutoff": str(maturity_cutoff.date()),
        "usable_trigger_entries": int(len(frame)),
        "censored_rows": int((~reviewer["usable"].fillna(False)).sum()) if not reviewer.empty else 0,
        "censor_reasons": censor_reasons,
        "entry_quarters": sorted(frame["entry_quarter"].astype(str).unique()),
        "stock_features": features,
        "stock_feature_count": len(features),
        "execution_features": list(EXECUTION_FEATURES),
        "favorable_regime_rows": int(frame["r3_favorable_regime"].astype(int).sum()),
        "baseline_all": summarize_path(frame),
        "baseline_r3_favorable": summarize_path(_scope_frame(frame, "r3_favorable")),
        "search_quantiles": list(SEARCH_QUANTILES),
        "search_audit": search_audit,
        "rolling_summary": summarize_rolling(rolling),
        "outputs": [
            "trigger_path_samples.csv",
            "feature_bin_characterization.csv",
            "feature_extremes.csv",
            "within_snapshot_feature_contrasts.csv",
            "stock_interactions_all.csv",
            "winner_interactions_all.csv",
            "stop_risk_interactions_all.csv",
            "stock_interactions_r3_favorable.csv",
            "winner_interactions_r3_favorable.csv",
            "stop_risk_interactions_r3_favorable.csv",
            "rolling_stock_selection.csv",
        ],
    }
    (args.output_root / "trigger_path_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=float) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
