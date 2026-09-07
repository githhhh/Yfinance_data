"""Retrospective empirical-ceiling search over already-known Blind Discovery history.

This module is intentionally NOT a sealed-holdout or canonical blind experiment.
It may use every mature historical quarter, including quarters previously consumed by
R1, to answer a different question: how much signal is present in the current
feature set if we systematically search simple coupled rules?

The search is deterministic and model-free:
- dense quantile threshold grid for every numeric feature;
- exhaustive single-condition search;
- beam search for 2/3-condition conjunctions;
- OR/DNF combinations of the strongest conjunctions;
- multi-objective robust score plus a Pareto frontier;
- quarter stability and leave-one-quarter-out sensitivity;
- expanding-window rolling re-search using only past quarters for each fold.

Outputs from this module MUST NOT be described as unseen holdout evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .dataset import DISCOVERY_FEATURE_ALLOWLIST, build_blind_dataset, load_replay_candidates
from .outcomes import OutcomeConfig, load_price_pickle, restrict_to_mature_outcome_quarters
from .pipeline_contract import validate_replay_preflight

DEFAULT_QUANTILES = tuple(i / 20 for i in range(1, 20))  # 5%..95%
ROLLING_QUANTILES = tuple(i / 10 for i in range(1, 10))  # 10%..90%
PARETO_METRICS = (
    "winner_rate_lift",
    "excess_12w_p50",
    "excess_12w_p25",
    "mae_12w_p50",
    "quarter_outperform_fraction",
    "median_quarter_lift",
    "coverage",
)
SCORE_WEIGHTS = {
    "winner_rate_lift": 0.25,
    "excess_12w_p50": 0.15,
    "excess_12w_p25": 0.10,
    "mae_12w_p50": 0.10,
    "mfe_12w_p50": 0.05,
    "quarter_outperform_fraction": 0.15,
    "median_quarter_lift": 0.10,
    "worst_quarter_lift": 0.05,
    "coverage": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--daily-pkl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spy-code", default="SPY")
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--max-conditions", type=int, default=3)
    parser.add_argument("--max-clauses", type=int, default=3)
    parser.add_argument("--dnf-clause-pool", type=int, default=18)
    parser.add_argument("--min-selected", type=int, default=40)
    parser.add_argument("--min-resolved", type=int, default=30)
    parser.add_argument("--min-active-quarters", type=int, default=5)
    parser.add_argument("--min-resolved-per-quarter", type=int, default=5)
    parser.add_argument("--rolling-min-train-quarters", type=int, default=6)
    parser.add_argument("--rolling-beam-width", type=int, default=20)
    parser.add_argument("--skip-rolling", action="store_true")
    return parser.parse_args()


def _finite_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _quantile_thresholds(series: pd.Series, quantiles: Sequence[float]) -> list[float]:
    numeric = _finite_numeric(series).dropna()
    if numeric.empty:
        return []
    values = numeric.quantile(list(quantiles)).to_numpy(dtype=float)
    rounded = sorted({float(np.round(v, 8)) for v in values if np.isfinite(v)})
    return rounded


def generate_conditions(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> list[dict[str, Any]]:
    """Generate deterministic >= / <= threshold conditions from empirical quantiles."""
    conditions: list[dict[str, Any]] = []
    for feature in sorted(feature_columns):
        numeric = _finite_numeric(frame[feature])
        observed = numeric.dropna()
        if observed.empty:
            continue
        thresholds = _quantile_thresholds(observed, quantiles)
        if not thresholds:
            continue
        lo = float(observed.min())
        hi = float(observed.max())
        for threshold in thresholds:
            # Skip tautological edges. They add search volume without information.
            if threshold > lo:
                conditions.append({"feature": feature, "op": ">=", "threshold": threshold})
            if threshold < hi:
                conditions.append({"feature": feature, "op": "<=", "threshold": threshold})
    return conditions


def _condition_key(condition: Mapping[str, Any]) -> tuple[str, str, float]:
    return (str(condition["feature"]), str(condition["op"]), float(condition["threshold"]))


def _condition_mask(frame: pd.DataFrame, condition: Mapping[str, Any]) -> np.ndarray:
    values = _finite_numeric(frame[str(condition["feature"])]).to_numpy(dtype=float)
    threshold = float(condition["threshold"])
    op = str(condition["op"])
    if op == ">=":
        return np.greater_equal(values, threshold) & np.isfinite(values)
    if op == "<=":
        return np.less_equal(values, threshold) & np.isfinite(values)
    raise ValueError(f"retrospective ceiling supports only >= / <= conditions, got {op!r}")


def _conditions_compatible(conditions: Sequence[Mapping[str, Any]]) -> bool:
    """Allow a feature once, or twice only as a non-empty lower/upper interval."""
    by_feature: dict[str, list[Mapping[str, Any]]] = {}
    for condition in conditions:
        by_feature.setdefault(str(condition["feature"]), []).append(condition)
    for group in by_feature.values():
        if len(group) > 2:
            return False
        if len(group) == 2:
            ops = {str(item["op"]) for item in group}
            if ops != {">=", "<="}:
                return False
            lower = max(float(item["threshold"]) for item in group if item["op"] == ">=")
            upper = min(float(item["threshold"]) for item in group if item["op"] == "<=")
            if lower > upper:
                return False
    return True


def _canonical_clause(conditions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"feature": feature, "op": op, "threshold": threshold}
        for feature, op, threshold in sorted(_condition_key(c) for c in conditions)
    ]


def _canonical_rule(clauses: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    normalized = [_canonical_clause(clause) for clause in clauses]
    normalized.sort(key=lambda clause: json.dumps(clause, sort_keys=True, separators=(",", ":")))
    return {"version": 1, "clauses": [{"all": clause} for clause in normalized]}


def _rule_key(rule: Mapping[str, Any]) -> str:
    return json.dumps(rule, sort_keys=True, separators=(",", ":"))


def _rule_conditions(rule: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(condition) for clause in rule["clauses"] for condition in clause["all"]]


def _rule_mask(
    rule: Mapping[str, Any],
    condition_masks: Mapping[tuple[str, str, float], np.ndarray],
    row_count: int,
) -> np.ndarray:
    selected = np.zeros(row_count, dtype=bool)
    for clause in rule["clauses"]:
        clause_mask = np.ones(row_count, dtype=bool)
        for condition in clause["all"]:
            clause_mask &= condition_masks[_condition_key(condition)]
        selected |= clause_mask
    return selected


def _mask_hash(mask: np.ndarray) -> str:
    packed = np.packbits(mask.astype(np.uint8)).tobytes()
    return hashlib.sha256(packed).hexdigest()


def _safe_quantile(values: np.ndarray, q: float) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.quantile(finite, q))


def _baseline_winner_rate(frame: pd.DataFrame) -> float | None:
    primary = frame["Y_primary"].astype(str).to_numpy()
    resolved = np.isin(primary, ["winner", "loser"])
    if not resolved.any():
        return None
    return float(np.mean(primary[resolved] == "winner"))


def evaluate_mask(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    min_resolved_per_quarter: int = 5,
) -> dict[str, Any]:
    """Evaluate a selected subset on multiple outcome/stability dimensions."""
    if len(mask) != len(frame):
        raise ValueError("mask length differs from frame")
    selected_n = int(mask.sum())
    coverage = selected_n / len(frame) if len(frame) else 0.0

    primary = frame["Y_primary"].astype(str).to_numpy()
    resolved_all = np.isin(primary, ["winner", "loser"])
    resolved_selected = mask & resolved_all
    resolved_n = int(resolved_selected.sum())
    winner_n = int(np.sum(resolved_selected & (primary == "winner")))
    winner_rate = (winner_n / resolved_n) if resolved_n else None
    universe_winner_rate = (
        float(np.mean(primary[resolved_all] == "winner")) if resolved_all.any() else None
    )
    winner_rate_lift = (
        winner_rate - universe_winner_rate
        if winner_rate is not None and universe_winner_rate is not None
        else None
    )

    def selected_numeric(column: str) -> np.ndarray:
        return _finite_numeric(frame[column]).to_numpy(dtype=float)[mask]

    excess = selected_numeric("Y_12w_excess")
    mae = selected_numeric("Y_mae_12w")
    mfe = selected_numeric("Y_mfe_12w")

    quarter_values = frame["period_quarter"].astype(str).to_numpy()
    quarter_lifts: list[float] = []
    active_quarters = 0
    evaluated_quarters = 0
    outperform_quarters = 0
    for quarter in sorted(pd.unique(quarter_values)):
        q_mask = quarter_values == quarter
        if np.any(mask & q_mask):
            active_quarters += 1
        q_selected_resolved = mask & q_mask & resolved_all
        q_universe_resolved = q_mask & resolved_all
        q_selected_n = int(q_selected_resolved.sum())
        q_universe_n = int(q_universe_resolved.sum())
        if q_selected_n < min_resolved_per_quarter or q_universe_n == 0:
            continue
        evaluated_quarters += 1
        selected_wr = float(np.mean(primary[q_selected_resolved] == "winner"))
        universe_wr = float(np.mean(primary[q_universe_resolved] == "winner"))
        lift = selected_wr - universe_wr
        quarter_lifts.append(lift)
        if lift > 0:
            outperform_quarters += 1

    return {
        "selected_n": selected_n,
        "coverage": coverage,
        "resolved_n": resolved_n,
        "winner_n": winner_n,
        "resolved_winner_rate": winner_rate,
        "universe_resolved_winner_rate": universe_winner_rate,
        "winner_rate_lift": winner_rate_lift,
        "excess_12w_p25": _safe_quantile(excess, 0.25),
        "excess_12w_p50": _safe_quantile(excess, 0.50),
        "excess_12w_p75": _safe_quantile(excess, 0.75),
        "mae_12w_p50": _safe_quantile(mae, 0.50),
        "mfe_12w_p50": _safe_quantile(mfe, 0.50),
        "active_quarters": active_quarters,
        "evaluated_quarters": evaluated_quarters,
        "quarter_outperform_fraction": (
            outperform_quarters / evaluated_quarters if evaluated_quarters else None
        ),
        "median_quarter_lift": (
            float(np.median(quarter_lifts)) if quarter_lifts else None
        ),
        "worst_quarter_lift": min(quarter_lifts) if quarter_lifts else None,
    }


def _eligible(
    metrics: Mapping[str, Any],
    *,
    min_selected: int,
    min_resolved: int,
    min_active_quarters: int,
) -> bool:
    return (
        int(metrics["selected_n"]) >= min_selected
        and int(metrics["resolved_n"]) >= min_resolved
        and int(metrics["active_quarters"]) >= min_active_quarters
    )


def _record_for_rule(
    rule: dict[str, Any],
    mask: np.ndarray,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    condition_count = sum(len(clause["all"]) for clause in rule["clauses"])
    return {
        "_rule": rule,
        "_mask_hash": _mask_hash(mask),
        "rule_json": _rule_key(rule),
        "clause_count": len(rule["clauses"]),
        "condition_count": condition_count,
        **dict(metrics),
    }


def score_records(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Rank rules by a transparent multi-objective percentile score."""
    if not records:
        return pd.DataFrame()
    scored = pd.DataFrame(records)
    score = pd.Series(0.0, index=scored.index, dtype=float)
    for metric, weight in SCORE_WEIGHTS.items():
        numeric = pd.to_numeric(scored[metric], errors="coerce")
        rank = numeric.rank(pct=True, method="average", na_option="bottom")
        score += weight * rank.fillna(0.0)
    complexity_penalty = (
        0.0125 * (pd.to_numeric(scored["condition_count"]) - 1).clip(lower=0)
        + 0.0100 * (pd.to_numeric(scored["clause_count"]) - 1).clip(lower=0)
    )
    scored["robust_score"] = score - complexity_penalty
    return scored.sort_values(
        ["robust_score", "winner_rate_lift", "excess_12w_p50", "coverage"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)


def _dedupe_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep the simplest representation of each identical selected subset."""
    best: dict[str, dict[str, Any]] = {}
    for raw in records:
        record = dict(raw)
        key = str(record["_mask_hash"])
        prior = best.get(key)
        if prior is None:
            best[key] = record
            continue
        current_complexity = (int(record["condition_count"]), int(record["clause_count"]))
        prior_complexity = (int(prior["condition_count"]), int(prior["clause_count"]))
        if current_complexity < prior_complexity:
            best[key] = record
    return list(best.values())


def pareto_frontier(scored: pd.DataFrame, *, limit: int = 1000) -> pd.DataFrame:
    """Return non-dominated rules on the main reward/stability dimensions."""
    if scored.empty:
        return scored.copy()
    work = scored.head(limit).copy().reset_index(drop=True)
    values = work[list(PARETO_METRICS)].apply(pd.to_numeric, errors="coerce").fillna(-np.inf).to_numpy()
    keep = np.ones(len(work), dtype=bool)
    for i in range(len(work)):
        if not keep[i]:
            continue
        dominates_i = np.all(values >= values[i], axis=1) & np.any(values > values[i], axis=1)
        dominates_i[i] = False
        if np.any(dominates_i):
            keep[i] = False
    return work.loc[keep].sort_values("robust_score", ascending=False).reset_index(drop=True)


def search_rules(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    beam_width: int = 32,
    max_conditions: int = 3,
    max_clauses: int = 3,
    dnf_clause_pool: int = 18,
    min_selected: int = 40,
    min_resolved: int = 30,
    min_active_quarters: int = 5,
    min_resolved_per_quarter: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Systematically search compact coupled rules and return scored/Pareto tables."""
    if frame.empty:
        raise ValueError("retrospective search frame is empty")
    if max_conditions < 1 or max_conditions > 3:
        raise ValueError("max_conditions must be within 1..3")
    if max_clauses < 1 or max_clauses > 3:
        raise ValueError("max_clauses must be within 1..3")
    conditions = generate_conditions(frame, feature_columns, quantiles=quantiles)
    if not conditions:
        raise ValueError("no usable search conditions")
    condition_masks = {_condition_key(c): _condition_mask(frame, c) for c in conditions}

    all_records: list[dict[str, Any]] = []
    single_records: list[dict[str, Any]] = []
    for condition in conditions:
        rule = _canonical_rule([[condition]])
        mask = condition_masks[_condition_key(condition)]
        metrics = evaluate_mask(
            frame, mask, min_resolved_per_quarter=min_resolved_per_quarter
        )
        if _eligible(
            metrics,
            min_selected=min_selected,
            min_resolved=min_resolved,
            min_active_quarters=min_active_quarters,
        ):
            single_records.append(_record_for_rule(rule, mask, metrics))
    single_scored = score_records(_dedupe_records(single_records))
    if single_scored.empty:
        raise ValueError("no single-condition rule satisfies support constraints")
    beam = single_scored.head(beam_width)
    all_records.extend(beam.head(max(beam_width * 4, 100)).to_dict(orient="records"))

    for depth in range(2, max_conditions + 1):
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for _, parent in beam.iterrows():
            parent_rule = parent["_rule"]
            parent_conditions = _rule_conditions(parent_rule)
            for condition in conditions:
                merged = [*parent_conditions, condition]
                if not _conditions_compatible(merged):
                    continue
                rule = _canonical_rule([merged])
                key = _rule_key(rule)
                if key in seen:
                    continue
                seen.add(key)
                mask = _rule_mask(rule, condition_masks, len(frame))
                metrics = evaluate_mask(
                    frame, mask, min_resolved_per_quarter=min_resolved_per_quarter
                )
                if not _eligible(
                    metrics,
                    min_selected=min_selected,
                    min_resolved=min_resolved,
                    min_active_quarters=min_active_quarters,
                ):
                    continue
                candidates.append(_record_for_rule(rule, mask, metrics))
        depth_scored = score_records(_dedupe_records(candidates))
        if depth_scored.empty:
            break
        beam = depth_scored.head(beam_width)
        all_records.extend(beam.head(max(beam_width * 4, 100)).to_dict(orient="records"))

    conjunctions = score_records(_dedupe_records(all_records))
    if conjunctions.empty:
        raise ValueError("conjunction search produced no eligible rules")

    if max_clauses > 1 and dnf_clause_pool >= 2:
        clause_pool = conjunctions.loc[conjunctions["clause_count"] == 1].head(dnf_clause_pool)
        clauses = [row["_rule"]["clauses"][0]["all"] for _, row in clause_pool.iterrows()]
        dnf_records: list[dict[str, Any]] = []
        for clause_count in range(2, max_clauses + 1):
            for combo in itertools.combinations(clauses, clause_count):
                total_conditions = sum(len(clause) for clause in combo)
                if total_conditions > 6:
                    continue
                rule = _canonical_rule(combo)
                mask = _rule_mask(rule, condition_masks, len(frame))
                metrics = evaluate_mask(
                    frame, mask, min_resolved_per_quarter=min_resolved_per_quarter
                )
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
    return scored, pareto, condition_masks


def _apply_rule_to_frame(rule: Mapping[str, Any], frame: pd.DataFrame) -> np.ndarray:
    masks: dict[tuple[str, str, float], np.ndarray] = {}
    for condition in _rule_conditions(rule):
        key = _condition_key(condition)
        if key not in masks:
            masks[key] = _condition_mask(frame, condition)
    return _rule_mask(rule, masks, len(frame))


def leave_one_quarter_out(
    frame: pd.DataFrame,
    rules: pd.DataFrame,
    *,
    top_n: int = 20,
    min_resolved_per_quarter: int = 5,
) -> pd.DataFrame:
    """Sensitivity check: re-score fixed top rules after dropping each quarter."""
    rows: list[dict[str, Any]] = []
    quarters = sorted(frame["period_quarter"].astype(str).unique())
    for rank, (_, record) in enumerate(rules.head(top_n).iterrows(), start=1):
        rule = record["_rule"]
        lifts: list[float] = []
        excess: list[float] = []
        for quarter in quarters:
            subset = frame.loc[frame["period_quarter"].astype(str) != quarter].reset_index(drop=True)
            mask = _apply_rule_to_frame(rule, subset)
            metrics = evaluate_mask(
                subset, mask, min_resolved_per_quarter=min_resolved_per_quarter
            )
            if metrics["winner_rate_lift"] is not None:
                lifts.append(float(metrics["winner_rate_lift"]))
            if metrics["excess_12w_p50"] is not None:
                excess.append(float(metrics["excess_12w_p50"]))
        rows.append(
            {
                "global_rank": rank,
                "rule_json": record["rule_json"],
                "loqo_folds": len(quarters),
                "winner_rate_lift_min": min(lifts) if lifts else None,
                "winner_rate_lift_p50": float(np.median(lifts)) if lifts else None,
                "winner_rate_lift_max": max(lifts) if lifts else None,
                "excess_12w_p50_min": min(excess) if excess else None,
                "excess_12w_p50_p50": float(np.median(excess)) if excess else None,
            }
        )
    return pd.DataFrame(rows)


def rolling_walk_forward(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    min_train_quarters: int = 6,
    beam_width: int = 20,
    min_selected: int = 30,
    min_resolved: int = 20,
    min_resolved_per_quarter: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window re-search on past quarters, then evaluate the next quarter."""
    quarters = sorted(frame["period_quarter"].astype(str).unique())
    if len(quarters) <= min_train_quarters:
        raise ValueError("not enough quarters for rolling walk-forward")
    rows: list[dict[str, Any]] = []
    feature_usage: dict[str, int] = {}
    for test_index in range(min_train_quarters, len(quarters)):
        train_quarters = quarters[:test_index]
        test_quarter = quarters[test_index]
        train = frame.loc[frame["period_quarter"].astype(str).isin(train_quarters)].reset_index(drop=True)
        test = frame.loc[frame["period_quarter"].astype(str) == test_quarter].reset_index(drop=True)
        min_active = min(4, max(2, len(train_quarters) // 2))
        scored, _, _ = search_rules(
            train,
            feature_columns,
            quantiles=ROLLING_QUANTILES,
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
            feature_usage[feature] = feature_usage.get(feature, 0) + 1
        test_mask = _apply_rule_to_frame(rule, test)
        test_metrics = evaluate_mask(
            test, test_mask, min_resolved_per_quarter=min_resolved_per_quarter
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
    usage = pd.DataFrame(
        [
            {"feature": feature, "rolling_fold_count": count, "fraction": count / len(rows)}
            for feature, count in feature_usage.items()
        ]
    ).sort_values(["rolling_fold_count", "feature"], ascending=[False, True]).reset_index(drop=True)
    return pd.DataFrame(rows), usage


def _public_table(scored: pd.DataFrame) -> pd.DataFrame:
    return scored.drop(columns=["_rule", "_mask_hash"], errors="ignore")


def _write_best_rule(path: Path, best: pd.Series) -> None:
    payload = {
        "research_mode": "retrospective_empirical_ceiling",
        "not_unseen_holdout": True,
        "rule": best["_rule"],
        "metrics": {
            key: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
            for key, value in best.items()
            if key not in {"_rule", "_mask_hash", "rule_json"}
        },
    }
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.output_root.resolve() == (Path("backtest/blind_rule_discovery/output/rd_agent_run_01").resolve()):
        raise RuntimeError("R1 output is immutable and may not be reused for retrospective search")
    args.output_root.mkdir(parents=True, exist_ok=True)

    provenance = validate_replay_preflight(
        args.replay_root,
        daily_pkl=args.daily_pkl,
        required_quarters=12,
    )
    prices = load_price_pickle(args.daily_pkl)
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
    # This is explicitly retrospective/non-blind: restore real feature names.
    frame = agent_df.rename(columns=feature_map).copy()
    feature_columns = [
        column
        for column in [*DISCOVERY_FEATURE_ALLOWLIST, *sorted(c for c in frame.columns if c.startswith("M_"))]
        if column in frame.columns
    ]
    if not feature_columns:
        raise ValueError("no retrospective search features available")

    scored, pareto, _ = search_rules(
        frame,
        feature_columns,
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

    rolling_rows = pd.DataFrame()
    rolling_usage = pd.DataFrame()
    if not args.skip_rolling:
        rolling_rows, rolling_usage = rolling_walk_forward(
            frame,
            feature_columns,
            min_train_quarters=args.rolling_min_train_quarters,
            beam_width=args.rolling_beam_width,
            min_selected=max(20, min(args.min_selected, 30)),
            min_resolved=max(15, min(args.min_resolved, 20)),
            min_resolved_per_quarter=max(3, min(args.min_resolved_per_quarter, 4)),
        )
        rolling_rows.to_csv(args.output_root / "rolling_walk_forward.csv", index=False)
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
        "research_mode": "retrospective_empirical_ceiling",
        "canonical_blind_experiment": False,
        "unseen_holdout_claim_allowed": False,
        "r1_consumed_periods_may_be_used_as_known_history": True,
        "purpose": "estimate historical feature-interaction ceiling and robustness, not prove future generalization",
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
        "rolling_fold_count": int(len(rolling_rows)),
    }
    (args.output_root / "retrospective_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
