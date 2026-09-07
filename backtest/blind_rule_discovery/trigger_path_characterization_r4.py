"""Frozen R4 trigger-path winner/loser characterization study.

R4 is retrospective known-history research, not a sealed holdout. Its primary unit
is an executable BF trigger entry. The main outcome is +20% before -8% within 15
trading sessions; unresolved entries remain in the probability denominator and
ambiguous paths are reported separately.

This final entry hardens the development implementation by adding:
- W1/W2/W3/W4 p25/p50/p75 return and excess-return paths;
- explicit persistent-stop vs stop-then-recover accounting;
- quarter stability for every feature bin and stock rule;
- separate winner and stop-risk interaction views;
- pooled rolling stock-selection baselines and zero-selection accounting.

Market M_* fields are context only. They are forbidden in stock-condition search.
The frozen R3 favorable regime may be used as a retrospective conditioning scope,
while within-snapshot comparisons provide the direct market-matched stock contrast.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .dataset import load_replay_candidates
from .outcomes import OutcomeConfig, load_price_pickle, restrict_to_mature_outcome_quarters
from .pipeline_contract import validate_replay_preflight
from .trigger_path_characterization import (
    EXECUTION_FEATURES,
    R3_FAVORABLE_REGIME,
    SEARCH_QUANTILES,
    Condition,
    _bin_definitions,
    _feature_bin_masks,
    _scope_frame,
    build_trigger_path_frame,
    condition_mask,
    generate_stock_conditions,
    stock_feature_columns,
    within_snapshot_feature_contrasts,
)

ROLLING_MIN_TRAIN_QUARTERS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--daily-pkl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spy-code", default="SPY")
    parser.add_argument("--min-selected-all", type=int, default=80)
    parser.add_argument("--min-selected-favorable", type=int, default=40)
    parser.add_argument("--min-evaluable-all", type=int, default=60)
    parser.add_argument("--min-evaluable-favorable", type=int, default=30)
    parser.add_argument("--min-quarter-n", type=int, default=10)
    parser.add_argument("--min-evaluated-quarters-all", type=int, default=5)
    parser.add_argument("--min-evaluated-quarters-favorable", type=int, default=3)
    parser.add_argument("--rolling-min-train-quarters", type=int, default=ROLLING_MIN_TRAIN_QUARTERS)
    return parser.parse_args()


def _finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def summarize_path(frame: pd.DataFrame, mask: np.ndarray | None = None) -> dict[str, Any]:
    """Summarize 3w first-passage and W1-W4 path for non-ambiguous entries.

    Unresolved entries remain in the denominator by design. Only ambiguous 3-week
    paths are excluded from fast-winner / stop-first / unresolved probabilities.
    """
    if mask is None:
        selected = frame.copy()
    else:
        if len(mask) != len(frame):
            raise ValueError("mask length differs from frame")
        selected = frame.loc[np.asarray(mask, dtype=bool)].copy()
    selected_n = int(len(selected))
    if selected_n == 0:
        ambiguous = pd.Series([], dtype=bool)
        evaluable = selected.copy()
    else:
        ambiguous = selected["ambiguous_3w"].astype(int) == 1
        evaluable = selected.loc[~ambiguous].copy()
    evaluable_n = int(len(evaluable))
    fast_n = int(evaluable["fast_winner_3w"].sum()) if evaluable_n else 0
    stop_n = int(evaluable["stop_first_3w"].sum()) if evaluable_n else 0
    unresolved_n = int(evaluable["unresolved_3w"].sum()) if evaluable_n else 0
    recovered_n = int(evaluable["stop_first_then_winner_12w"].sum()) if evaluable_n else 0
    persistent_stop_n = max(0, stop_n - recovered_n)

    def quantile(column: str, q: float) -> float | None:
        if column not in evaluable.columns:
            return None
        values = _finite(evaluable[column]).dropna()
        return float(values.quantile(q)) if not values.empty else None

    mae3 = quantile("mae_3w", 0.50)
    mfe3 = quantile("mfe_3w", 0.50)
    rr3 = None if mae3 is None or mfe3 is None or mae3 >= 0 or abs(mae3) < 1e-12 else mfe3 / abs(mae3)
    result: dict[str, Any] = {
        "selected_n": selected_n,
        "evaluable_n": evaluable_n,
        "ambiguous_n": int(ambiguous.sum()) if selected_n else 0,
        "fast_winner_n": fast_n,
        "stop_first_n": stop_n,
        "unresolved_n": unresolved_n,
        "stop_first_then_winner_12w_n": recovered_n,
        "persistent_stop_first_n": persistent_stop_n,
        "fast_winner_rate": fast_n / evaluable_n if evaluable_n else None,
        "stop_first_rate": stop_n / evaluable_n if evaluable_n else None,
        "unresolved_rate": unresolved_n / evaluable_n if evaluable_n else None,
        "stop_first_then_winner_12w_rate": recovered_n / stop_n if stop_n else None,
        "persistent_stop_first_rate": persistent_stop_n / evaluable_n if evaluable_n else None,
        "persistent_share_of_stop_first": persistent_stop_n / stop_n if stop_n else None,
        "mae_3w_p50": mae3,
        "mfe_3w_p50": mfe3,
        "mae_4w_p50": quantile("mae_4w", 0.50),
        "mfe_4w_p50": quantile("mfe_4w", 0.50),
        "mfe_mae_ratio_3w": rr3,
    }
    for prefix in ("return", "excess"):
        for week in ("w1", "w2", "w3", "w4"):
            for label, q in (("p25", 0.25), ("p50", 0.50), ("p75", 0.75)):
                result[f"{prefix}_{week}_{label}"] = quantile(f"{prefix}_{week}", q)
    return result


def _path_edge(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> tuple[float | None, float | None, float | None]:
    if (
        metrics.get("fast_winner_rate") is None
        or baseline.get("fast_winner_rate") is None
        or metrics.get("stop_first_rate") is None
        or baseline.get("stop_first_rate") is None
    ):
        return None, None, None
    fast_lift = float(metrics["fast_winner_rate"]) - float(baseline["fast_winner_rate"])
    stop_reduction = float(baseline["stop_first_rate"]) - float(metrics["stop_first_rate"])
    return fast_lift, stop_reduction, fast_lift + stop_reduction


def quarter_stability(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    min_quarter_n: int,
) -> dict[str, Any]:
    """Compare the selected subset with its contemporaneous scope baseline by entry quarter."""
    path_edges: list[float] = []
    stop_lifts: list[float] = []
    quarters = frame["entry_quarter"].astype(str).to_numpy()
    for quarter in sorted(pd.unique(quarters)):
        q_mask = quarters == quarter
        q_frame = frame.loc[q_mask].reset_index(drop=True)
        q_selected = frame.loc[q_mask & mask].reset_index(drop=True)
        metrics = summarize_path(q_selected)
        if int(metrics["evaluable_n"]) < min_quarter_n:
            continue
        baseline = summarize_path(q_frame)
        fast_lift, stop_reduction, edge = _path_edge(metrics, baseline)
        if edge is None or fast_lift is None or stop_reduction is None:
            continue
        path_edges.append(float(edge))
        stop_lifts.append(float(-stop_reduction))  # positive means higher stop-first risk than baseline.
    edges = np.asarray(path_edges, dtype=float)
    stop = np.asarray(stop_lifts, dtype=float)
    return {
        "evaluated_quarters": int(len(edges)),
        "positive_path_edge_quarter_fraction": float(np.mean(edges > 0)) if edges.size else None,
        "median_quarter_path_edge": float(np.median(edges)) if edges.size else None,
        "worst_quarter_path_edge": float(np.min(edges)) if edges.size else None,
        "higher_stop_risk_quarter_fraction": float(np.mean(stop > 0)) if stop.size else None,
        "median_quarter_stop_first_lift": float(np.median(stop)) if stop.size else None,
    }


def feature_bin_characterization(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    min_quarter_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Characterize winner and loser feature bins with quarter-stability diagnostics."""
    definitions = _bin_definitions(frame, features)
    rows: list[dict[str, Any]] = []
    for scope in ("all", "r3_favorable"):
        scoped = _scope_frame(frame, scope)
        baseline = summarize_path(scoped)
        for feature in features:
            cuts = definitions.get(feature)
            if not cuts or feature not in scoped.columns:
                continue
            for ordinal, (label, mask, low, high) in enumerate(_feature_bin_masks(scoped, feature, cuts)):
                metrics = summarize_path(scoped, mask)
                if int(metrics["selected_n"]) == 0:
                    continue
                fast_lift, stop_reduction, edge = _path_edge(metrics, baseline)
                stability = quarter_stability(scoped, mask, min_quarter_n=min_quarter_n)
                rows.append(
                    {
                        "scope": scope,
                        "feature": feature,
                        "bin_ordinal": ordinal,
                        "bin_label": label,
                        "low_exclusive": low,
                        "high_inclusive": high,
                        **metrics,
                        "baseline_fast_winner_rate": baseline["fast_winner_rate"],
                        "baseline_stop_first_rate": baseline["stop_first_rate"],
                        "fast_winner_lift": fast_lift,
                        "stop_first_reduction": stop_reduction,
                        "path_edge": edge,
                        **stability,
                    }
                )
    bins = pd.DataFrame(rows)
    extremes: list[dict[str, Any]] = []
    if bins.empty:
        return bins, pd.DataFrame()
    for (scope, feature), group in bins.groupby(["scope", "feature"], sort=True):
        supported = group.loc[(group["evaluable_n"] >= 20) & (group["evaluated_quarters"] >= 2)].copy()
        if supported.empty:
            continue
        winner = supported.sort_values(
            ["fast_winner_lift", "stop_first_reduction", "excess_w3_p50"],
            ascending=[False, False, False],
        ).iloc[0]
        loser = supported.assign(
            stop_first_lift=-pd.to_numeric(supported["stop_first_reduction"], errors="coerce")
        ).sort_values(
            ["stop_first_lift", "fast_winner_lift", "excess_w3_p50"],
            ascending=[False, True, True],
        ).iloc[0]
        extremes.append(
            {
                "scope": scope,
                "feature": feature,
                "winner_bin": winner["bin_label"],
                "winner_evaluable_n": int(winner["evaluable_n"]),
                "winner_fast_winner_rate": winner["fast_winner_rate"],
                "winner_fast_lift": winner["fast_winner_lift"],
                "winner_stop_first_rate": winner["stop_first_rate"],
                "winner_path_edge": winner["path_edge"],
                "winner_positive_edge_quarter_fraction": winner["positive_path_edge_quarter_fraction"],
                "winner_excess_w3_p25": winner["excess_w3_p25"],
                "winner_excess_w3_p50": winner["excess_w3_p50"],
                "winner_excess_w3_p75": winner["excess_w3_p75"],
                "winner_mae_3w_p50": winner["mae_3w_p50"],
                "winner_mfe_3w_p50": winner["mfe_3w_p50"],
                "loser_bin": loser["bin_label"],
                "loser_evaluable_n": int(loser["evaluable_n"]),
                "loser_stop_first_rate": loser["stop_first_rate"],
                "loser_stop_first_lift": -float(loser["stop_first_reduction"]),
                "loser_fast_winner_rate": loser["fast_winner_rate"],
                "loser_higher_stop_risk_quarter_fraction": loser["higher_stop_risk_quarter_fraction"],
                "loser_excess_w3_p25": loser["excess_w3_p25"],
                "loser_excess_w3_p50": loser["excess_w3_p50"],
                "loser_excess_w3_p75": loser["excess_w3_p75"],
                "loser_mae_3w_p50": loser["mae_3w_p50"],
                "loser_mfe_3w_p50": loser["mfe_3w_p50"],
            }
        )
    return bins, pd.DataFrame(extremes)


def _rule_json(conditions: Sequence[Condition]) -> str:
    payload = {"all": [c.as_dict() for c in sorted(conditions, key=lambda item: item.key())]}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _record_rule(
    frame: pd.DataFrame,
    mask: np.ndarray,
    conditions: Sequence[Condition],
    baseline: Mapping[str, Any],
    *,
    min_quarter_n: int,
) -> dict[str, Any]:
    metrics = summarize_path(frame, mask)
    fast_lift, stop_reduction, edge = _path_edge(metrics, baseline)
    return {
        "rule_json": _rule_json(conditions),
        "condition_count": len(conditions),
        **metrics,
        "baseline_fast_winner_rate": baseline["fast_winner_rate"],
        "baseline_stop_first_rate": baseline["stop_first_rate"],
        "fast_winner_lift": fast_lift,
        "stop_first_reduction": stop_reduction,
        "path_edge": edge,
        **quarter_stability(frame, mask, min_quarter_n=min_quarter_n),
    }


def _score_interactions(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    scored = pd.DataFrame(records).drop_duplicates(subset=["rule_json"]).reset_index(drop=True)
    weights = {
        "path_edge": 0.30,
        "fast_winner_lift": 0.15,
        "stop_first_reduction": 0.15,
        "excess_w3_p50": 0.10,
        "mae_3w_p50": 0.10,
        "mfe_3w_p50": 0.05,
        "mfe_mae_ratio_3w": 0.05,
        "positive_path_edge_quarter_fraction": 0.05,
        "worst_quarter_path_edge": 0.05,
    }
    score = pd.Series(0.0, index=scored.index, dtype=float)
    for metric, weight in weights.items():
        numeric = pd.to_numeric(scored[metric], errors="coerce")
        score += weight * numeric.rank(pct=True, method="average", na_option="bottom").fillna(0.0)
    score -= 0.01 * (pd.to_numeric(scored["condition_count"], errors="coerce") - 1).clip(lower=0)
    scored["quality_score"] = score
    return scored.sort_values(
        ["quality_score", "path_edge", "excess_w3_p50", "evaluable_n"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)


def search_stock_interactions(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    min_selected: int,
    min_evaluable: int,
    min_quarter_n: int,
    min_evaluated_quarters: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Exact singles + exact distinct-feature two-condition stock rules."""
    if any(str(feature).startswith("M_") for feature in features):
        raise ValueError("M_* market features are forbidden in R4 stock-condition search")
    baseline = summarize_path(frame)
    conditions = generate_stock_conditions(frame, features)
    masks = {condition.key(): condition_mask(frame, condition) for condition in conditions}
    records: list[dict[str, Any]] = []
    pair_tested = 0
    supported_single = 0
    supported_pair = 0

    def eligible(record: Mapping[str, Any]) -> bool:
        return (
            int(record["selected_n"]) >= min_selected
            and int(record["evaluable_n"]) >= min_evaluable
            and int(record["evaluated_quarters"]) >= min_evaluated_quarters
        )

    for condition in conditions:
        record = _record_rule(frame, masks[condition.key()], [condition], baseline, min_quarter_n=min_quarter_n)
        if eligible(record):
            records.append(record)
            supported_single += 1

    for i, left in enumerate(conditions):
        for right in conditions[i + 1 :]:
            if left.feature == right.feature:
                continue
            pair_tested += 1
            mask = masks[left.key()] & masks[right.key()]
            if int(np.count_nonzero(mask)) < min_selected:
                continue
            record = _record_rule(frame, mask, [left, right], baseline, min_quarter_n=min_quarter_n)
            if eligible(record):
                records.append(record)
                supported_pair += 1

    scored = _score_interactions(records)
    return scored, {
        "generated_stock_condition_count": len(conditions),
        "supported_single_count": supported_single,
        "distinct_feature_pair_count_tested": pair_tested,
        "supported_pair_count": supported_pair,
        "candidate_rule_count": int(len(scored)),
    }


def _conditions_from_json(rule_json: str) -> list[Condition]:
    payload = json.loads(rule_json)
    return [Condition(str(c["feature"]), str(c["op"]), float(c["threshold"])) for c in payload["all"]]


def rule_mask(frame: pd.DataFrame, rule_json: str) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for condition in _conditions_from_json(rule_json):
        mask &= condition_mask(frame, condition)
    return mask


def matched_selected_vs_unselected(frame: pd.DataFrame, rule_json: str) -> dict[str, Any]:
    """Equal-snapshot selected-vs-unselected path lift, controlling identical M_* context."""
    selected_mask = rule_mask(frame, rule_json)
    fast_lifts: list[float] = []
    stop_reductions: list[float] = []
    for _, snapshot in frame.assign(_selected=selected_mask).groupby("snapshot_date", sort=True):
        selected = snapshot.loc[snapshot["_selected"]].reset_index(drop=True)
        unselected = snapshot.loc[~snapshot["_selected"]].reset_index(drop=True)
        if selected.empty or unselected.empty:
            continue
        sm = summarize_path(selected)
        um = summarize_path(unselected)
        if min(int(sm["evaluable_n"]), int(um["evaluable_n"])) < 1:
            continue
        fast_lifts.append(float(sm["fast_winner_rate"] - um["fast_winner_rate"]))
        stop_reductions.append(float(um["stop_first_rate"] - sm["stop_first_rate"]))
    edge = np.asarray(fast_lifts, dtype=float) + np.asarray(stop_reductions, dtype=float)
    return {
        "matched_snapshot_count": int(len(fast_lifts)),
        "matched_fast_winner_lift_p50": float(np.median(fast_lifts)) if fast_lifts else None,
        "matched_stop_first_reduction_p50": float(np.median(stop_reductions)) if stop_reductions else None,
        "matched_path_edge_p50": float(np.median(edge)) if edge.size else None,
        "matched_positive_path_edge_fraction": float(np.mean(edge > 0)) if edge.size else None,
    }


def add_matched_metrics(frame: pd.DataFrame, scored: pd.DataFrame, *, top_n: int = 100) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    out = scored.copy()
    columns = (
        "matched_snapshot_count",
        "matched_fast_winner_lift_p50",
        "matched_stop_first_reduction_p50",
        "matched_path_edge_p50",
        "matched_positive_path_edge_fraction",
    )
    for column in columns:
        out[column] = np.nan
    for index in out.head(top_n).index:
        matched = matched_selected_vs_unselected(frame, str(out.at[index, "rule_json"]))
        for key, value in matched.items():
            out.at[index, key] = value
    return out


def interaction_views(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return explicit winner-seeking and stop-risk views from the same frozen search."""
    if scored.empty:
        return scored.copy(), scored.copy()
    winner = scored.sort_values(
        ["fast_winner_lift", "stop_first_reduction", "excess_w3_p50", "positive_path_edge_quarter_fraction"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    loser = scored.assign(
        stop_first_lift=-pd.to_numeric(scored["stop_first_reduction"], errors="coerce")
    ).sort_values(
        ["stop_first_lift", "fast_winner_lift", "excess_w3_p50", "higher_stop_risk_quarter_fraction"],
        ascending=[False, True, True, False],
        na_position="last",
    ).reset_index(drop=True)
    return winner, loser


def _rolling_scope_search(
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
        if len(train) < 80:
            continue
        scored, _ = search_stock_interactions(
            train,
            features,
            min_selected=60 if scope == "all" else 30,
            min_evaluable=45 if scope == "all" else 20,
            min_quarter_n=max(5, min_quarter_n // 2),
            min_evaluated_quarters=4 if scope == "all" else 2,
        )
        test_baseline = summarize_path(test)
        if scored.empty:
            rows.append(
                {
                    "scope": scope,
                    "test_quarter": test_quarter,
                    "train_start_quarter": train_quarters[0],
                    "train_end_quarter": train_quarters[-1],
                    "train_quarter_count": len(train_quarters),
                    "test_quarter_in_train": False,
                    "rule_json": None,
                    "train_quality_score": None,
                    "test_scope_n": int(len(test)),
                    "test_selected_n": 0,
                    "test_evaluable_n": 0,
                    **{f"test_baseline_{k}": v for k, v in test_baseline.items()},
                }
            )
            continue
        best = scored.iloc[0]
        rule_json = str(best["rule_json"])
        mask = rule_mask(test, rule_json)
        test_metrics = summarize_path(test, mask)
        fast_lift, stop_reduction, edge = _path_edge(test_metrics, test_baseline)
        matched = matched_selected_vs_unselected(test, rule_json) if not test.empty else {}
        rows.append(
            {
                "scope": scope,
                "test_quarter": test_quarter,
                "train_start_quarter": train_quarters[0],
                "train_end_quarter": train_quarters[-1],
                "train_quarter_count": len(train_quarters),
                "test_quarter_in_train": test_quarter in train_quarters,
                "rule_json": rule_json,
                "train_quality_score": best["quality_score"],
                "train_path_edge": best["path_edge"],
                "test_scope_n": int(len(test)),
                **{f"test_{k}": v for k, v in test_metrics.items()},
                **{f"test_baseline_{k}": v for k, v in test_baseline.items()},
                "test_fast_winner_lift": fast_lift,
                "test_stop_first_reduction": stop_reduction,
                "test_path_edge": edge,
                **{f"test_{k}": v for k, v in matched.items()},
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
        _rolling_scope_search(
            frame,
            features,
            scope=scope,
            min_train_quarters=min_train_quarters,
            min_quarter_n=min_quarter_n,
        )
        for scope in ("all", "r3_favorable")
    ]
    parts = [part for part in parts if not part.empty]
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
        selected_eval = float(evaluable.sum())
        baseline_eval = float(base_eval.sum())
        selected_fast_rate = float(fast.sum() / selected_eval) if selected_eval else None
        selected_stop_rate = float(stop.sum() / selected_eval) if selected_eval else None
        baseline_fast_rate = float(base_fast.sum() / baseline_eval) if baseline_eval else None
        baseline_stop_rate = float(base_stop.sum() / baseline_eval) if baseline_eval else None
        pooled_edge = (
            (selected_fast_rate - baseline_fast_rate) + (baseline_stop_rate - selected_stop_rate)
            if None not in {selected_fast_rate, selected_stop_rate, baseline_fast_rate, baseline_stop_rate}
            else None
        )
        summary[scope] = {
            "folds": int(len(group)),
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
            "pooled_baseline_fast_winner_rate": baseline_fast_rate,
            "pooled_baseline_stop_first_rate": baseline_stop_rate,
            "pooled_path_edge": pooled_edge,
        }
    return summary


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
        scored, audit = search_stock_interactions(
            scoped,
            features,
            min_selected=args.min_selected_all if scope == "all" else args.min_selected_favorable,
            min_evaluable=args.min_evaluable_all if scope == "all" else args.min_evaluable_favorable,
            min_quarter_n=args.min_quarter_n,
            min_evaluated_quarters=(
                args.min_evaluated_quarters_all if scope == "all" else args.min_evaluated_quarters_favorable
            ),
        )
        scored = add_matched_metrics(scoped, scored, top_n=100)
        winner_view, loser_view = interaction_views(scored)
        scored.to_csv(args.output_root / f"stock_interactions_{scope}.csv", index=False)
        winner_view.to_csv(args.output_root / f"winner_interactions_{scope}.csv", index=False)
        loser_view.to_csv(args.output_root / f"stop_risk_interactions_{scope}.csv", index=False)
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
        "research_mode": "r4_trigger_path_winner_loser_characterization_final",
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
