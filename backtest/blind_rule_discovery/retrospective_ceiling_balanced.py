"""Feature-balanced retrospective ceiling search.

This is the execution entry point for the non-canonical retrospective ceiling study.
It deliberately preserves threshold candidates from every feature before testing
interactions, so a feature that is weak alone cannot disappear before pair search.

This module uses already-known history. It is NOT an unseen holdout experiment.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .dataset import DISCOVERY_FEATURE_ALLOWLIST, build_blind_dataset, load_replay_candidates
from .outcomes import OutcomeConfig, load_price_pickle, restrict_to_mature_outcome_quarters
from .pipeline_contract import validate_replay_preflight
from .retrospective_ceiling import (
    DEFAULT_QUANTILES,
    ROLLING_QUANTILES,
    PARETO_METRICS,
    SCORE_WEIGHTS,
    _apply_rule_to_frame,
    _canonical_rule,
    _condition_key,
    _condition_mask,
    _conditions_compatible,
    _dedupe_records,
    _eligible,
    _public_table,
    _record_for_rule,
    _rule_conditions,
    _rule_key,
    _rule_mask,
    _write_best_rule,
    evaluate_mask,
    generate_conditions,
    leave_one_quarter_out,
    pareto_frontier,
    score_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--daily-pkl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spy-code", default="SPY")
    parser.add_argument("--conditions-per-feature", type=int, default=6)
    parser.add_argument("--beam-width", type=int, default=40)
    parser.add_argument("--max-conditions", type=int, default=3)
    parser.add_argument("--max-clauses", type=int, default=3)
    parser.add_argument("--dnf-clause-pool", type=int, default=20)
    parser.add_argument("--min-selected", type=int, default=40)
    parser.add_argument("--min-resolved", type=int, default=30)
    parser.add_argument("--min-active-quarters", type=int, default=5)
    parser.add_argument("--min-resolved-per-quarter", type=int, default=5)
    parser.add_argument("--rolling-min-train-quarters", type=int, default=6)
    parser.add_argument("--rolling-conditions-per-feature", type=int, default=4)
    parser.add_argument("--rolling-beam-width", type=int, default=24)
    parser.add_argument("--skip-rolling", action="store_true")
    return parser.parse_args()


def _balanced_condition_pool(
    single_scored: pd.DataFrame,
    *,
    conditions_per_feature: int,
) -> list[dict[str, Any]]:
    """Keep strong thresholds from every feature, not only globally strong singles."""
    if conditions_per_feature < 1:
        raise ValueError("conditions_per_feature must be >= 1")
    chosen: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float]] = set()
    counts: dict[str, int] = {}
    for _, row in single_scored.iterrows():
        conditions = _rule_conditions(row["_rule"])
        if len(conditions) != 1:
            continue
        condition = conditions[0]
        feature = str(condition["feature"])
        if counts.get(feature, 0) >= conditions_per_feature:
            continue
        key = _condition_key(condition)
        if key in seen:
            continue
        seen.add(key)
        counts[feature] = counts.get(feature, 0) + 1
        chosen.append(condition)
    return chosen


def _score_all_single_conditions(
    frame: pd.DataFrame,
    conditions: Sequence[Mapping[str, Any]],
    condition_masks: Mapping[tuple[str, str, float], np.ndarray],
    *,
    min_selected: int,
    min_resolved: int,
    min_active_quarters: int,
    min_resolved_per_quarter: int,
) -> pd.DataFrame:
    """Score every eligible single condition without cross-feature mask deduplication.

    Two different features can select exactly the same rows historically. Collapsing
    those masks before interaction search arbitrarily deletes one feature and can hide
    a later interaction. Feature identity is therefore preserved until the balanced
    condition pool has selected representatives from every searchable feature.
    """
    records: list[dict[str, Any]] = []
    for condition in conditions:
        rule = _canonical_rule([[condition]])
        mask = condition_masks[_condition_key(condition)]
        metrics = evaluate_mask(frame, mask, min_resolved_per_quarter=min_resolved_per_quarter)
        if _eligible(
            metrics,
            min_selected=min_selected,
            min_resolved=min_resolved,
            min_active_quarters=min_active_quarters,
        ):
            records.append(_record_for_rule(rule, mask, metrics))
    return score_records(records)


def balanced_search_rules(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    conditions_per_feature: int = 6,
    beam_width: int = 40,
    max_conditions: int = 3,
    max_clauses: int = 3,
    dnf_clause_pool: int = 20,
    min_selected: int = 40,
    min_resolved: int = 30,
    min_active_quarters: int = 5,
    min_resolved_per_quarter: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Search singles, exhaustive balanced pairs, beam triples, then compact DNF."""
    if frame.empty:
        raise ValueError("retrospective search frame is empty")
    if not 1 <= max_conditions <= 3:
        raise ValueError("max_conditions must be within 1..3")
    if not 1 <= max_clauses <= 3:
        raise ValueError("max_clauses must be within 1..3")

    conditions = generate_conditions(frame, feature_columns, quantiles=quantiles)
    if not conditions:
        raise ValueError("no usable search conditions")
    condition_masks = {_condition_key(c): _condition_mask(frame, c) for c in conditions}

    single_scored = _score_all_single_conditions(
        frame,
        conditions,
        condition_masks,
        min_selected=min_selected,
        min_resolved=min_resolved,
        min_active_quarters=min_active_quarters,
        min_resolved_per_quarter=min_resolved_per_quarter,
    )
    if single_scored.empty:
        raise ValueError("no single-condition rule satisfies support constraints")

    all_records: list[dict[str, Any]] = single_scored.head(max(100, beam_width * 4)).to_dict(orient="records")
    interaction_conditions = _balanced_condition_pool(
        single_scored,
        conditions_per_feature=conditions_per_feature,
    )
    represented_features = {str(condition["feature"]) for condition in interaction_conditions}
    searchable_features = {str(condition["feature"]) for condition in conditions}
    missing_features = sorted(searchable_features - represented_features)
    if missing_features:
        raise ValueError(
            "feature-balanced interaction pool lost searchable features: " + ",".join(missing_features)
        )

    pair_scored = pd.DataFrame()
    if max_conditions >= 2:
        pair_records: list[dict[str, Any]] = []
        for left, right in itertools.combinations(interaction_conditions, 2):
            merged = [left, right]
            if not _conditions_compatible(merged):
                continue
            rule = _canonical_rule([merged])
            mask = _rule_mask(rule, condition_masks, len(frame))
            metrics = evaluate_mask(frame, mask, min_resolved_per_quarter=min_resolved_per_quarter)
            if not _eligible(
                metrics,
                min_selected=min_selected,
                min_resolved=min_resolved,
                min_active_quarters=min_active_quarters,
            ):
                continue
            pair_records.append(_record_for_rule(rule, mask, metrics))
        pair_scored = score_records(_dedupe_records(pair_records))
        all_records.extend(pair_scored.head(max(160, beam_width * 4)).to_dict(orient="records"))

    triple_scored = pd.DataFrame()
    if max_conditions >= 3 and not pair_scored.empty:
        triple_records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for _, parent in pair_scored.head(beam_width).iterrows():
            parent_conditions = _rule_conditions(parent["_rule"])
            for condition in interaction_conditions:
                merged = [*parent_conditions, condition]
                if not _conditions_compatible(merged):
                    continue
                rule = _canonical_rule([merged])
                key = _rule_key(rule)
                if key in seen:
                    continue
                seen.add(key)
                mask = _rule_mask(rule, condition_masks, len(frame))
                metrics = evaluate_mask(frame, mask, min_resolved_per_quarter=min_resolved_per_quarter)
                if not _eligible(
                    metrics,
                    min_selected=min_selected,
                    min_resolved=min_resolved,
                    min_active_quarters=min_active_quarters,
                ):
                    continue
                triple_records.append(_record_for_rule(rule, mask, metrics))
        triple_scored = score_records(_dedupe_records(triple_records))
        all_records.extend(triple_scored.head(max(160, beam_width * 4)).to_dict(orient="records"))

    conjunctions = score_records(_dedupe_records(all_records))
    if conjunctions.empty:
        raise ValueError("balanced conjunction search produced no eligible rule")

    if max_clauses > 1 and dnf_clause_pool >= 2:
        clause_pool = conjunctions.loc[conjunctions["clause_count"] == 1].head(dnf_clause_pool)
        clauses = [row["_rule"]["clauses"][0]["all"] for _, row in clause_pool.iterrows()]
        dnf_records: list[dict[str, Any]] = []
        for clause_count in range(2, max_clauses + 1):
            for combo in itertools.combinations(clauses, clause_count):
                if sum(len(clause) for clause in combo) > 6:
                    continue
                rule = _canonical_rule(combo)
                mask = _rule_mask(rule, condition_masks, len(frame))
                metrics = evaluate_mask(frame, mask, min_resolved_per_quarter=min_resolved_per_quarter)
                if not _eligible(
                    metrics,
                    min_selected=min_selected,
                    min_resolved=min_resolved,
                    min_active_quarters=min_active_quarters,
                ):
                    continue
                dnf_records.append(_record_for_rule(rule, mask, metrics))
        all_records.extend(dnf_records)

    scored = score_records(_dedupe_records(all_records))
    pareto = pareto_frontier(scored)
    return scored, pareto


def balanced_rolling_walk_forward(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    min_train_quarters: int = 6,
    conditions_per_feature: int = 4,
    beam_width: int = 24,
    min_selected: int = 30,
    min_resolved: int = 20,
    min_resolved_per_quarter: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pure expanding-window re-search: each fold derives thresholds from past data only."""
    quarters = sorted(frame["period_quarter"].astype(str).unique())
    if len(quarters) <= min_train_quarters:
        raise ValueError("not enough quarters for rolling walk-forward")
    rows: list[dict[str, Any]] = []
    usage: dict[str, int] = {}
    for test_index in range(min_train_quarters, len(quarters)):
        train_quarters = quarters[:test_index]
        test_quarter = quarters[test_index]
        train = frame.loc[frame["period_quarter"].astype(str).isin(train_quarters)].reset_index(drop=True)
        test = frame.loc[frame["period_quarter"].astype(str) == test_quarter].reset_index(drop=True)
        min_active = min(4, max(2, len(train_quarters) // 2))
        scored, _ = balanced_search_rules(
            train,
            feature_columns,
            quantiles=ROLLING_QUANTILES,
            conditions_per_feature=conditions_per_feature,
            beam_width=beam_width,
            max_conditions=2,
            max_clauses=1,
            dnf_clause_pool=0,
            min_selected=min_selected,
            min_resolved=min_resolved,
            min_active_quarters=min_active,
            min_resolved_per_quarter=min_resolved_per_quarter,
        )
        best = scored.iloc[0]
        rule = best["_rule"]
        for condition in _rule_conditions(rule):
            feature = str(condition["feature"])
            usage[feature] = usage.get(feature, 0) + 1
        test_metrics = evaluate_mask(
            test,
            _apply_rule_to_frame(rule, test),
            min_resolved_per_quarter=min_resolved_per_quarter,
        )
        rows.append(
            {
                "test_quarter": test_quarter,
                "train_start_quarter": train_quarters[0],
                "train_end_quarter": train_quarters[-1],
                "train_quarter_count": len(train_quarters),
                "rule_json": best["rule_json"],
                "train_robust_score": best["robust_score"],
                "train_winner_rate_lift": best["winner_rate_lift"],
                "train_excess_12w_p50": best["excess_12w_p50"],
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )
    usage_frame = pd.DataFrame(
        [
            {"feature": feature, "rolling_fold_count": count, "fraction": count / len(rows)}
            for feature, count in usage.items()
        ]
    ).sort_values(["rolling_fold_count", "feature"], ascending=[False, True]).reset_index(drop=True)
    return pd.DataFrame(rows), usage_frame


def _summarize_rolling(rolling: pd.DataFrame) -> dict[str, Any]:
    if rolling.empty:
        return {}
    selected = pd.to_numeric(rolling["test_selected_n"], errors="coerce").fillna(0)
    resolved = pd.to_numeric(rolling["test_resolved_n"], errors="coerce").fillna(0)
    winners = pd.to_numeric(rolling["test_winner_n"], errors="coerce").fillna(0)
    resolved_total = float(resolved.sum())
    winner_total = float(winners.sum())
    lift = pd.to_numeric(rolling["test_winner_rate_lift"], errors="coerce").dropna()
    excess = pd.to_numeric(rolling["test_excess_12w_p50"], errors="coerce").dropna()
    return {
        "folds": int(len(rolling)),
        "selected_n_total": int(selected.sum()),
        "resolved_n_total": int(resolved_total),
        "winner_n_total": int(winner_total),
        "pooled_resolved_winner_rate": (winner_total / resolved_total) if resolved_total else None,
        "positive_lift_fold_fraction": float((lift > 0).mean()) if not lift.empty else None,
        "winner_rate_lift_p50": float(lift.median()) if not lift.empty else None,
        "winner_rate_lift_min": float(lift.min()) if not lift.empty else None,
        "excess_12w_p50_across_folds": float(excess.median()) if not excess.empty else None,
    }


def main() -> int:
    args = parse_args()
    r1_root = Path("backtest/blind_rule_discovery/output/rd_agent_run_01").resolve()
    if args.output_root.resolve() == r1_root:
        raise RuntimeError("R1 output is immutable and may not be reused")
    args.output_root.mkdir(parents=True, exist_ok=True)

    provenance = validate_replay_preflight(
        args.replay_root,
        daily_pkl=args.daily_pkl,
        required_quarters=12,
    )
    prices = load_price_pickle(args.daily_pkl, require_adjusted=True)
    if args.spy_code not in prices:
        raise KeyError(f"benchmark {args.spy_code!r} missing from daily price bundle")

    candidates_all = load_replay_candidates(args.replay_root)
    config = OutcomeConfig()
    future_sessions_required = config.minimum_sessions + config.entry_window_sessions
    candidates, immature_rows, maturity_cutoff = restrict_to_mature_outcome_quarters(
        candidates_all,
        prices[args.spy_code],
        minimum_sessions=future_sessions_required,
    )
    agent_df, feature_map, reviewer = build_blind_dataset(
        candidates,
        prices,
        prices[args.spy_code],
        config=config,
    )
    frame = agent_df.rename(columns=feature_map).copy()
    feature_columns = [
        column
        for column in [*DISCOVERY_FEATURE_ALLOWLIST, *sorted(c for c in frame.columns if c.startswith("M_"))]
        if column in frame.columns
    ]
    if not feature_columns:
        raise ValueError("no retrospective search features available")

    scored, pareto = balanced_search_rules(
        frame,
        feature_columns,
        conditions_per_feature=args.conditions_per_feature,
        beam_width=args.beam_width,
        max_conditions=args.max_conditions,
        max_clauses=args.max_clauses,
        dnf_clause_pool=args.dnf_clause_pool,
        min_selected=args.min_selected,
        min_resolved=args.min_resolved,
        min_active_quarters=args.min_active_quarters,
        min_resolved_per_quarter=args.min_resolved_per_quarter,
    )
    _public_table(scored).to_csv(args.output_root / "candidate_rules.csv", index=False)
    _public_table(pareto).to_csv(args.output_root / "pareto_frontier.csv", index=False)
    _write_best_rule(args.output_root / "best_rule.json", scored.iloc[0])

    loqo = leave_one_quarter_out(
        frame,
        scored,
        top_n=20,
        min_resolved_per_quarter=args.min_resolved_per_quarter,
    )
    loqo.to_csv(args.output_root / "leave_one_quarter_out.csv", index=False)

    rolling = pd.DataFrame()
    rolling_usage = pd.DataFrame()
    if not args.skip_rolling:
        rolling, rolling_usage = balanced_rolling_walk_forward(
            frame,
            feature_columns,
            min_train_quarters=args.rolling_min_train_quarters,
            conditions_per_feature=args.rolling_conditions_per_feature,
            beam_width=args.rolling_beam_width,
            min_selected=max(20, min(args.min_selected, 30)),
            min_resolved=max(15, min(args.min_resolved, 20)),
            min_resolved_per_quarter=max(3, min(args.min_resolved_per_quarter, 4)),
        )
        rolling.to_csv(args.output_root / "rolling_walk_forward.csv", index=False)
        rolling_usage.to_csv(args.output_root / "rolling_feature_frequency.csv", index=False)

    baseline = evaluate_mask(
        frame,
        np.ones(len(frame), dtype=bool),
        min_resolved_per_quarter=args.min_resolved_per_quarter,
    )
    censor_reasons = (
        reviewer.loc[~reviewer["usable"].fillna(False), "reason"]
        .fillna("unknown")
        .value_counts()
        .to_dict()
        if not reviewer.empty
        else {}
    )
    metadata = {
        "research_mode": "retrospective_empirical_ceiling_feature_balanced",
        "canonical_blind_experiment": False,
        "unseen_holdout_claim_allowed": False,
        "r1_consumed_periods_are_known_history": True,
        "purpose": "estimate historical interaction ceiling and stability, not prove future generalization",
        "search_semantics": {
            "single_conditions": "exhaustive over dense empirical quantiles",
            "pair_conditions": "exhaustive over a feature-balanced threshold pool",
            "triple_conditions": "beam expansion from strongest pairs",
            "dnf": "OR combinations from strongest conjunction clauses",
            "llm_used": False,
        },
        "replay_provenance_verified": True,
        "replay_dataset_sha256": provenance.get("replay_dataset_sha256"),
        "candidate_rows_before_maturity_filter": int(len(candidates_all)),
        "candidate_rows": int(len(candidates)),
        "excluded_immature_rows": int(len(immature_rows)),
        "outcome_maturity_cutoff": str(maturity_cutoff.date()),
        "usable_rows": int(len(frame)),
        "entry_quarters": sorted(frame["period_quarter"].astype(str).unique()),
        "censored_rows": int((~reviewer["usable"].fillna(False)).sum()) if not reviewer.empty else 0,
        "censor_reasons": censor_reasons,
        "features": feature_columns,
        "feature_count": len(feature_columns),
        "quantiles": list(DEFAULT_QUANTILES),
        "conditions_per_feature": args.conditions_per_feature,
        "beam_width": args.beam_width,
        "max_conditions": args.max_conditions,
        "max_clauses": args.max_clauses,
        "dnf_clause_pool": args.dnf_clause_pool,
        "support_constraints": {
            "min_selected": args.min_selected,
            "min_resolved": args.min_resolved,
            "min_active_quarters": args.min_active_quarters,
            "min_resolved_per_quarter": args.min_resolved_per_quarter,
        },
        "score_weights": SCORE_WEIGHTS,
        "pareto_metrics": list(PARETO_METRICS),
        "candidate_rule_count": int(len(scored)),
        "pareto_rule_count": int(len(pareto)),
        "baseline": baseline,
        "best_rule_file": "best_rule.json",
        "rolling_enabled": not args.skip_rolling,
        "rolling_summary": _summarize_rolling(rolling),
    }
    (args.output_root / "retrospective_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
