"""Causal R5 stop-risk validation for BreakoutFollow trigger entries.

R5 is a narrow retrospective follow-up to R4. R4 established that historical
stop-risk / loser pockets are easier to find than stable winner rules, but its
chronological rolling selector still chose the quality-ranked positive stock rule.
Therefore R4 did not answer whether stop-risk information itself is chronologically
learnable.

R5 answers only that missing question. It does not search for a new winner Alpha.
For each expanding-window fold it:

1. purges training rows whose W3 label overlaps the test quarter;
2. searches the already-frozen R4 q20/q40/q60/q80 stock-only single/pair grid;
3. selects the most consistently elevated stop-risk rule from training data only;
4. freezes that rule and evaluates exactly the next quarter;
5. separately evaluates a small set of post-R4 semantic risk families whose
   thresholds are regenerated from training data only.

All R1-R4 periods are already-known history. R5 is robustness research, not an
unseen holdout and not production Alpha certification.
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
from .trigger_path_characterization import _scope_frame, build_trigger_path_frame, stock_feature_columns
from .trigger_path_characterization_r4 import matched_selected_vs_unselected, rule_mask, summarize_path
from .trigger_path_characterization_r4_causal_runner import purge_training_rows_for_w3
from .trigger_path_characterization_r4_search import search_stock_interactions_fast

ROLLING_MIN_TRAIN_QUARTERS = 6
RISK_FAMILIES = (
    "deep_pullback",
    "extended_vs_candidate",
    "deep_pullback_and_extended",
    "high_pct_above_ceiling",
    "high_entry_extension",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--daily-pkl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spy-code", default="SPY")
    parser.add_argument("--rolling-min-train-quarters", type=int, default=ROLLING_MIN_TRAIN_QUARTERS)
    parser.add_argument("--min-quarter-n", type=int, default=10)
    return parser.parse_args()


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _baseline_persistent_rate(frame: pd.DataFrame) -> float | None:
    return summarize_path(frame).get("persistent_stop_first_rate")


def rank_stop_risk_rules(scored: pd.DataFrame, *, baseline_persistent_rate: float | None) -> pd.DataFrame:
    """Order the same frozen search candidates only by stop-risk evidence.

    Selection is deliberately lexicographic rather than a newly tuned scalar score:

    1. fraction of evaluated training quarters with higher stop-first risk;
    2. median quarter stop-first lift;
    3. aggregate stop-first lift;
    4. support, then canonical rule JSON for deterministic ties.

    Persistent-stop lift uses 12-week recovery and is descriptive only: the
    rolling training purge covers W3, so it must never affect rule selection.
    """
    if scored.empty:
        return scored.copy()
    out = scored.copy()
    out["stop_first_lift"] = -pd.to_numeric(out["stop_first_reduction"], errors="coerce")
    persistent = pd.to_numeric(out["persistent_stop_first_rate"], errors="coerce")
    out["baseline_persistent_stop_first_rate"] = baseline_persistent_rate
    out["persistent_stop_first_lift"] = (
        persistent - float(baseline_persistent_rate)
        if baseline_persistent_rate is not None
        else np.nan
    )
    return out.sort_values(
        [
            "higher_stop_risk_quarter_fraction",
            "median_quarter_stop_first_lift",
            "stop_first_lift",
            "evaluable_n",
            "rule_json",
        ],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def _test_metrics(frame: pd.DataFrame, mask: np.ndarray, baseline: Mapping[str, Any]) -> dict[str, Any]:
    metrics = summarize_path(frame, mask)
    stop_lift = None
    persistent_lift = None
    if metrics.get("stop_first_rate") is not None and baseline.get("stop_first_rate") is not None:
        stop_lift = float(metrics["stop_first_rate"] - baseline["stop_first_rate"])
    if (
        metrics.get("persistent_stop_first_rate") is not None
        and baseline.get("persistent_stop_first_rate") is not None
    ):
        persistent_lift = float(
            metrics["persistent_stop_first_rate"] - baseline["persistent_stop_first_rate"]
        )
    return {
        **metrics,
        "stop_first_lift": stop_lift,
        "persistent_stop_first_lift": persistent_lift,
    }


def matched_stop_risk_from_mask(frame: pd.DataFrame, selected_mask: np.ndarray) -> dict[str, Any]:
    """Equal-snapshot selected-vs-unselected stop-risk contrast for arbitrary masks."""
    if len(selected_mask) != len(frame):
        raise ValueError("selected mask length differs from frame")
    stop_lifts: list[float] = []
    persistent_lifts: list[float] = []
    for _, snapshot in frame.assign(_selected=np.asarray(selected_mask, dtype=bool)).groupby(
        "snapshot_date", sort=True
    ):
        selected = snapshot.loc[snapshot["_selected"]].reset_index(drop=True)
        unselected = snapshot.loc[~snapshot["_selected"]].reset_index(drop=True)
        if selected.empty or unselected.empty:
            continue
        sm = summarize_path(selected)
        um = summarize_path(unselected)
        if min(int(sm["evaluable_n"]), int(um["evaluable_n"])) < 1:
            continue
        if sm["stop_first_rate"] is None or um["stop_first_rate"] is None:
            continue
        stop_lifts.append(float(sm["stop_first_rate"] - um["stop_first_rate"]))
        if sm["persistent_stop_first_rate"] is not None and um["persistent_stop_first_rate"] is not None:
            persistent_lifts.append(
                float(sm["persistent_stop_first_rate"] - um["persistent_stop_first_rate"])
            )
    stop_array = np.asarray(stop_lifts, dtype=float)
    persistent_array = np.asarray(persistent_lifts, dtype=float)
    return {
        "matched_snapshot_count": int(stop_array.size),
        "matched_stop_first_lift_p50": float(np.median(stop_array)) if stop_array.size else None,
        "matched_positive_stop_lift_fraction": float(np.mean(stop_array > 0)) if stop_array.size else None,
        "matched_persistent_stop_lift_p50": (
            float(np.median(persistent_array)) if persistent_array.size else None
        ),
        "matched_positive_persistent_lift_fraction": (
            float(np.mean(persistent_array > 0)) if persistent_array.size else None
        ),
    }


def _empty_risk_fold(
    *,
    scope: str,
    test_quarter: str,
    train_quarters: Sequence[str],
    test: pd.DataFrame,
    baseline: Mapping[str, Any],
    purge_audit: Mapping[str, Any],
    train_insufficient: bool,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "test_quarter": test_quarter,
        "train_start_quarter": train_quarters[0],
        "train_end_quarter": train_quarters[-1],
        "train_quarter_count": len(train_quarters),
        "test_quarter_in_train": False,
        "train_insufficient": int(train_insufficient),
        **dict(purge_audit),
        "rule_json": None,
        "train_higher_stop_risk_quarter_fraction": None,
        "train_median_quarter_stop_first_lift": None,
        "train_stop_first_lift": None,
        "train_persistent_stop_first_lift": None,
        "test_scope_n": int(len(test)),
        "test_selected_n": 0,
        "test_evaluable_n": 0,
        **{f"test_baseline_{key}": value for key, value in baseline.items()},
    }


def rolling_stop_risk_search(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    scope: str,
    min_train_quarters: int,
    min_quarter_n: int,
) -> pd.DataFrame:
    """Past-only exact stock-rule re-search, selecting the strongest stop-risk rule."""
    if any(str(feature).startswith("M_") for feature in features):
        raise ValueError("M_* features are forbidden in R5 stop-risk rule search")
    quarters = sorted(frame["entry_quarter"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    for test_index in range(min_train_quarters, len(quarters)):
        train_quarters = quarters[:test_index]
        test_quarter = quarters[test_index]
        raw_train = frame.loc[frame["entry_quarter"].astype(str).isin(train_quarters)].reset_index(drop=True)
        train_all, purge_audit = purge_training_rows_for_w3(raw_train, test_quarter=test_quarter)
        test_all = frame.loc[frame["entry_quarter"].astype(str) == test_quarter].reset_index(drop=True)
        train = _scope_frame(train_all, scope)
        test = _scope_frame(test_all, scope)
        baseline = summarize_path(test)
        scope_audit = {**purge_audit, "train_scope_rows_after_purge": int(len(train))}
        if len(train) < 80:
            rows.append(
                _empty_risk_fold(
                    scope=scope,
                    test_quarter=test_quarter,
                    train_quarters=train_quarters,
                    test=test,
                    baseline=baseline,
                    purge_audit=scope_audit,
                    train_insufficient=True,
                )
            )
            continue
        scored, audit = search_stock_interactions_fast(
            train,
            features,
            min_selected=60 if scope == "all" else 30,
            min_evaluable=45 if scope == "all" else 20,
            min_quarter_n=max(5, min_quarter_n // 2),
            min_evaluated_quarters=4 if scope == "all" else 2,
        )
        ranked = rank_stop_risk_rules(scored, baseline_persistent_rate=_baseline_persistent_rate(train))
        if ranked.empty:
            row = _empty_risk_fold(
                scope=scope,
                test_quarter=test_quarter,
                train_quarters=train_quarters,
                test=test,
                baseline=baseline,
                purge_audit=scope_audit,
                train_insufficient=False,
            )
            row.update({f"search_{key}": value for key, value in audit.items()})
            rows.append(row)
            continue
        best = ranked.iloc[0]
        rule_json = str(best["rule_json"])
        test_metrics = _test_metrics(test, rule_mask(test, rule_json), baseline)
        matched = matched_selected_vs_unselected(test, rule_json) if not test.empty else {}
        rows.append(
            {
                "scope": scope,
                "test_quarter": test_quarter,
                "train_start_quarter": train_quarters[0],
                "train_end_quarter": train_quarters[-1],
                "train_quarter_count": len(train_quarters),
                "test_quarter_in_train": False,
                "train_insufficient": 0,
                **scope_audit,
                "rule_json": rule_json,
                "train_higher_stop_risk_quarter_fraction": best["higher_stop_risk_quarter_fraction"],
                "train_median_quarter_stop_first_lift": best["median_quarter_stop_first_lift"],
                "train_stop_first_lift": best["stop_first_lift"],
                "train_persistent_stop_first_lift": best["persistent_stop_first_lift"],
                "train_evaluable_n": best["evaluable_n"],
                "test_scope_n": int(len(test)),
                **{f"test_{key}": value for key, value in test_metrics.items()},
                **{f"test_baseline_{key}": value for key, value in baseline.items()},
                "test_matched_stop_first_lift_p50": (
                    -matched.get("matched_stop_first_reduction_p50")
                    if matched.get("matched_stop_first_reduction_p50") is not None
                    else None
                ),
                "test_matched_snapshot_count": matched.get("matched_snapshot_count"),
                **{f"search_{key}": value for key, value in audit.items()},
            }
        )
    return pd.DataFrame(rows)


def rolling_stop_risk_research(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    min_train_quarters: int,
    min_quarter_n: int,
) -> pd.DataFrame:
    parts = [
        rolling_stop_risk_search(
            frame,
            features,
            scope=scope,
            min_train_quarters=min_train_quarters,
            min_quarter_n=min_quarter_n,
        )
        for scope in ("all", "r3_favorable")
    ]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _train_quantile(frame: pd.DataFrame, feature: str, q: float) -> float | None:
    if feature not in frame.columns:
        return None
    values = _numeric(frame[feature]).dropna()
    return float(values.quantile(q)) if not values.empty else None


def build_risk_family_rule(frame: pd.DataFrame, family: str) -> dict[str, Any] | None:
    """Build one post-R4 semantic risk family using training-only quantiles."""
    if family == "deep_pullback":
        threshold = _train_quantile(frame, "pullback_pct", 0.20)
        if threshold is None:
            return None
        return {"family": family, "conditions": [("pullback_pct", "<=", threshold)]}
    if family == "extended_vs_candidate":
        threshold = _train_quantile(frame, "current_vs_ibd_candidate_pct", 0.80)
        if threshold is None:
            return None
        return {"family": family, "conditions": [("current_vs_ibd_candidate_pct", ">=", threshold)]}
    if family == "deep_pullback_and_extended":
        pullback = _train_quantile(frame, "pullback_pct", 0.20)
        extension = _train_quantile(frame, "current_vs_ibd_candidate_pct", 0.80)
        if pullback is None or extension is None:
            return None
        return {
            "family": family,
            "conditions": [
                ("pullback_pct", "<=", pullback),
                ("current_vs_ibd_candidate_pct", ">=", extension),
            ],
        }
    if family == "high_pct_above_ceiling":
        threshold = _train_quantile(frame, "pct_above_ceiling", 0.80)
        if threshold is None:
            return None
        return {"family": family, "conditions": [("pct_above_ceiling", ">=", threshold)]}
    if family == "high_entry_extension":
        threshold = _train_quantile(frame, "entry_extension_pct", 0.80)
        if threshold is None:
            return None
        return {"family": family, "conditions": [("entry_extension_pct", ">=", threshold)]}
    raise ValueError(f"unknown R5 risk family: {family}")


def risk_family_mask(frame: pd.DataFrame, rule: Mapping[str, Any]) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for feature, op, threshold in rule["conditions"]:
        values = _numeric(frame[str(feature)]).to_numpy(dtype=float)
        if op == "<=":
            mask &= np.isfinite(values) & (values <= float(threshold))
        elif op == ">=":
            mask &= np.isfinite(values) & (values >= float(threshold))
        else:
            raise ValueError(f"unsupported family op: {op}")
    return mask


def rolling_risk_families(
    frame: pd.DataFrame,
    *,
    min_train_quarters: int,
) -> pd.DataFrame:
    """Chronological robustness of post-R4 semantic families; no family search."""
    quarters = sorted(frame["entry_quarter"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    for scope in ("all", "r3_favorable"):
        for test_index in range(min_train_quarters, len(quarters)):
            train_quarters = quarters[:test_index]
            test_quarter = quarters[test_index]
            raw_train = frame.loc[frame["entry_quarter"].astype(str).isin(train_quarters)].reset_index(drop=True)
            train_all, purge_audit = purge_training_rows_for_w3(raw_train, test_quarter=test_quarter)
            test_all = frame.loc[frame["entry_quarter"].astype(str) == test_quarter].reset_index(drop=True)
            train = _scope_frame(train_all, scope)
            test = _scope_frame(test_all, scope)
            baseline = summarize_path(test)
            for family in RISK_FAMILIES:
                rule = build_risk_family_rule(train, family)
                if rule is None:
                    mask = np.zeros(len(test), dtype=bool)
                    selected_metrics = summarize_path(test, mask)
                    selected_metrics = {
                        **selected_metrics,
                        "stop_first_lift": None,
                        "persistent_stop_first_lift": None,
                    }
                    rule_json = None
                    matched = {}
                else:
                    mask = risk_family_mask(test, rule)
                    selected_metrics = _test_metrics(test, mask, baseline)
                    rule_json = json.dumps(rule, sort_keys=True, separators=(",", ":"))
                    matched = matched_stop_risk_from_mask(test, mask) if not test.empty else {}
                rows.append(
                    {
                        "scope": scope,
                        "family": family,
                        "test_quarter": test_quarter,
                        "train_start_quarter": train_quarters[0],
                        "train_end_quarter": train_quarters[-1],
                        "train_quarter_count": len(train_quarters),
                        "test_quarter_in_train": False,
                        **purge_audit,
                        "train_scope_rows_after_purge": int(len(train)),
                        "family_rule_json": rule_json,
                        "test_scope_n": int(len(test)),
                        **{f"test_{key}": value for key, value in selected_metrics.items()},
                        **{f"test_baseline_{key}": value for key, value in baseline.items()},
                        **{f"test_{key}": value for key, value in matched.items()},
                    }
                )
    return pd.DataFrame(rows)


def _pooled_rate(numerator: pd.Series, denominator: pd.Series) -> float | None:
    d = float(pd.to_numeric(denominator, errors="coerce").fillna(0).sum())
    if d <= 0:
        return None
    return float(pd.to_numeric(numerator, errors="coerce").fillna(0).sum() / d)


def selection_weighted_baseline(group: pd.DataFrame) -> dict[str, Any]:
    """Match quarter weights to selected evaluable rows, not universe size.

    This controls quarterly composition only. Equal-snapshot complementary
    contrasts remain the stock-selection diagnostic.
    """
    weights = pd.to_numeric(group["test_evaluable_n"], errors="coerce").fillna(0)
    baseline_n = pd.to_numeric(group["test_baseline_evaluable_n"], errors="coerce")
    result = {}
    for label, numerator in (("stop_first", "stop_first_n"), ("persistent_stop", "persistent_stop_first_n")):
        baseline_counts = pd.to_numeric(group[f"test_baseline_{numerator}"], errors="coerce")
        selected_counts = pd.to_numeric(group[f"test_{numerator}"], errors="coerce")
        usable = (weights > 0) & (baseline_n > 0) & baseline_counts.notna() & selected_counts.notna()
        total = weights.loc[usable].sum()
        expected = ((baseline_counts.loc[usable] / baseline_n.loc[usable]) * weights.loc[usable]).sum()
        result[f"selection_weighted_baseline_{label}_rate"] = float(expected / total) if total > 0 else None
        result[f"selection_weighted_{label}_lift"] = (
            float((selected_counts.loc[usable].sum() - expected) / total) if total > 0 else None
        )
    return result


def summarize_risk_rolling(rolling: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if rolling.empty:
        return summary
    for scope, group in rolling.groupby("scope", sort=True):
        selected = pd.to_numeric(group["test_selected_n"], errors="coerce").fillna(0)
        evaluable = pd.to_numeric(group["test_evaluable_n"], errors="coerce").fillna(0)
        stop_lift = pd.to_numeric(group["test_stop_first_lift"], errors="coerce")
        persistent_lift = pd.to_numeric(group["test_persistent_stop_first_lift"], errors="coerce")
        matched = pd.to_numeric(group["test_matched_stop_first_lift_p50"], errors="coerce")
        overlap = pd.to_numeric(group["w3_label_overlap_after_purge"], errors="coerce").fillna(0).astype(bool)
        if overlap.any():
            raise AssertionError("R5 summary observed post-purge W3 overlap")
        traded = selected > 0
        selected_stop_rate = _pooled_rate(group["test_stop_first_n"], evaluable)
        selected_persistent_rate = _pooled_rate(group["test_persistent_stop_first_n"], evaluable)
        baseline_eval_traded = pd.to_numeric(group["test_baseline_evaluable_n"], errors="coerce").fillna(0).where(traded, 0)
        baseline_stop_traded = _pooled_rate(
            pd.to_numeric(group["test_baseline_stop_first_n"], errors="coerce").fillna(0).where(traded, 0),
            baseline_eval_traded,
        )
        baseline_persistent_traded = _pooled_rate(
            pd.to_numeric(group["test_baseline_persistent_stop_first_n"], errors="coerce").fillna(0).where(traded, 0),
            baseline_eval_traded,
        )
        summary[scope] = {
            **selection_weighted_baseline(group),
            "folds": int(len(group)),
            "w3_overlap_purge_rows_total": int(pd.to_numeric(group["train_rows_purged_for_w3_overlap"], errors="coerce").fillna(0).sum()),
            "post_purge_overlap_fold_count": int(overlap.sum()),
            "zero_selection_folds": int((selected == 0).sum()),
            "zero_selection_fraction": float((selected == 0).mean()),
            "evaluable_stop_lift_folds": int(stop_lift.notna().sum()),
            "positive_stop_lift_all_fold_fraction": float((stop_lift.fillna(-np.inf) > 0).mean()),
            "positive_stop_lift_evaluable_fraction": float((stop_lift.dropna() > 0).mean()) if stop_lift.notna().any() else None,
            "stop_first_lift_p50_evaluable": float(stop_lift.dropna().median()) if stop_lift.notna().any() else None,
            "positive_persistent_lift_all_fold_fraction": float((persistent_lift.fillna(-np.inf) > 0).mean()),
            "persistent_stop_lift_p50_evaluable": float(persistent_lift.dropna().median()) if persistent_lift.notna().any() else None,
            "matched_positive_stop_lift_fraction": float((matched.dropna() > 0).mean()) if matched.notna().any() else None,
            "selected_n_total": int(selected.sum()),
            "evaluable_n_total": int(evaluable.sum()),
            "pooled_selected_stop_first_rate": selected_stop_rate,
            "pooled_baseline_stop_first_rate_selected_folds": baseline_stop_traded,
            "pooled_selected_stop_first_lift": (
                selected_stop_rate - baseline_stop_traded
                if selected_stop_rate is not None and baseline_stop_traded is not None
                else None
            ),
            "pooled_selected_persistent_stop_rate": selected_persistent_rate,
            "pooled_baseline_persistent_stop_rate_selected_folds": baseline_persistent_traded,
            "pooled_selected_persistent_stop_lift": (
                selected_persistent_rate - baseline_persistent_traded
                if selected_persistent_rate is not None and baseline_persistent_traded is not None
                else None
            ),
        }
    return summary


def summarize_family_rolling(families: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if families.empty:
        return rows
    for (scope, family), group in families.groupby(["scope", "family"], sort=True):
        selected = pd.to_numeric(group["test_selected_n"], errors="coerce").fillna(0)
        evaluable = pd.to_numeric(group["test_evaluable_n"], errors="coerce").fillna(0)
        stop_lift = pd.to_numeric(group["test_stop_first_lift"], errors="coerce")
        persistent_lift = pd.to_numeric(group["test_persistent_stop_first_lift"], errors="coerce")
        matched_stop = pd.to_numeric(group.get("test_matched_stop_first_lift_p50"), errors="coerce")
        matched_persistent = pd.to_numeric(group.get("test_matched_persistent_stop_lift_p50"), errors="coerce")
        traded = selected > 0
        selected_stop_rate = _pooled_rate(group["test_stop_first_n"], evaluable)
        selected_persistent_rate = _pooled_rate(group["test_persistent_stop_first_n"], evaluable)
        baseline_eval = pd.to_numeric(group["test_baseline_evaluable_n"], errors="coerce").fillna(0).where(traded, 0)
        baseline_stop_rate = _pooled_rate(
            pd.to_numeric(group["test_baseline_stop_first_n"], errors="coerce").fillna(0).where(traded, 0),
            baseline_eval,
        )
        baseline_persistent_rate = _pooled_rate(
            pd.to_numeric(group["test_baseline_persistent_stop_first_n"], errors="coerce").fillna(0).where(traded, 0),
            baseline_eval,
        )
        rows.append(
            {
                "scope": scope,
                "family": family,
                **selection_weighted_baseline(group),
                "folds": int(len(group)),
                "zero_selection_folds": int((selected == 0).sum()),
                "evaluable_stop_lift_folds": int(stop_lift.notna().sum()),
                "positive_stop_lift_all_fold_fraction": float((stop_lift.fillna(-np.inf) > 0).mean()),
                "positive_stop_lift_evaluable_fraction": float((stop_lift.dropna() > 0).mean()) if stop_lift.notna().any() else None,
                "stop_first_lift_p50_evaluable": float(stop_lift.dropna().median()) if stop_lift.notna().any() else None,
                "positive_persistent_lift_all_fold_fraction": float((persistent_lift.fillna(-np.inf) > 0).mean()),
                "persistent_stop_lift_p50_evaluable": float(persistent_lift.dropna().median()) if persistent_lift.notna().any() else None,
                "matched_positive_stop_lift_fraction": float((matched_stop.dropna() > 0).mean()) if matched_stop.notna().any() else None,
                "matched_stop_lift_p50": float(matched_stop.dropna().median()) if matched_stop.notna().any() else None,
                "matched_positive_persistent_lift_fraction": (
                    float((matched_persistent.dropna() > 0).mean()) if matched_persistent.notna().any() else None
                ),
                "matched_persistent_lift_p50": (
                    float(matched_persistent.dropna().median()) if matched_persistent.notna().any() else None
                ),
                "selected_n_total": int(selected.sum()),
                "evaluable_n_total": int(evaluable.sum()),
                "pooled_selected_stop_first_rate": selected_stop_rate,
                "pooled_baseline_stop_first_rate_selected_folds": baseline_stop_rate,
                "pooled_selected_stop_first_lift": (
                    selected_stop_rate - baseline_stop_rate
                    if selected_stop_rate is not None and baseline_stop_rate is not None
                    else None
                ),
                "pooled_selected_persistent_stop_rate": selected_persistent_rate,
                "pooled_baseline_persistent_stop_rate_selected_folds": baseline_persistent_rate,
                "pooled_selected_persistent_stop_lift": (
                    selected_persistent_rate - baseline_persistent_rate
                    if selected_persistent_rate is not None and baseline_persistent_rate is not None
                    else None
                ),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    immutable = {
        Path("backtest/blind_rule_discovery/output/rd_agent_run_01").resolve(),
        Path("backtest/blind_rule_discovery/output/retrospective_ceiling_r2").resolve(),
        Path("backtest/blind_rule_discovery/output/retrospective_ceiling_r3").resolve(),
        Path("backtest/blind_rule_discovery/output/trigger_path_characterization_r4").resolve(),
    }
    if args.output_root.resolve() in immutable:
        raise RuntimeError("R1-R4 outputs are immutable and may not be reused by R5")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError("Corrected R5 requires a fresh output root; preserve prior R5 evidence")
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
        raise ValueError("no stock/execution features available for R5")
    if any(feature.startswith("M_") for feature in features):
        raise AssertionError("market feature leaked into R5 stock feature list")

    risk_rolling = rolling_stop_risk_research(
        frame,
        features,
        min_train_quarters=args.rolling_min_train_quarters,
        min_quarter_n=args.min_quarter_n,
    )
    family_rolling = rolling_risk_families(frame, min_train_quarters=args.rolling_min_train_quarters)
    risk_rolling.to_csv(args.output_root / "stop_risk_rolling.csv", index=False)
    family_rolling.to_csv(args.output_root / "risk_family_rolling.csv", index=False)

    metadata = {
        "research_mode": "r5_w3_only_selection_audit_corrected",
        "selection_revision": "remove_unpurged_12w_tiebreaker",
        "pooled_lift_semantics": "legacy pooled lifts are composition-unadjusted; also report selection-weighted lifts",
        "canonical_blind_experiment": False,
        "unseen_holdout_claim_allowed": False,
        "llm_used": False,
        "purpose": "validate whether R4 historical stop-risk information is chronologically learnable",
        "winner_alpha_search": False,
        "r1_r4_periods_are_known_history": True,
        "stock_features": features,
        "market_features_forbidden_in_rule_search": True,
        "risk_rule_selection": {
            "search_space": "same frozen R4 q20/q40/q60/q80 singles plus distinct-feature pairs",
            "selection_order": [
                "higher_stop_risk_quarter_fraction desc",
                "median_quarter_stop_first_lift desc",
                "stop_first_lift desc",
                "evaluable_n desc",
                "rule_json asc",
            ],
            "persistent_stop_training_metric": "descriptive_only_not_used_in_selection",
            "weighted_risk_score_used": False,
        },
        "post_r4_semantic_families": list(RISK_FAMILIES),
        "family_threshold_semantics": "q20/q80 regenerated from each purged past-only training fold",
        "rolling_label_purge": "exit_date_w3 must be strictly before test quarter start",
        "replay_dataset_sha256": provenance.get("replay_dataset_sha256"),
        "candidate_rows_before_maturity_filter": int(len(candidates_all)),
        "candidate_rows": int(len(candidates)),
        "excluded_immature_rows": int(len(immature)),
        "outcome_maturity_cutoff": str(maturity_cutoff.date()),
        "usable_trigger_entries": int(len(frame)),
        "censored_rows": int((~reviewer["usable"].fillna(False)).sum()) if not reviewer.empty else 0,
        "entry_quarters": sorted(frame["entry_quarter"].astype(str).unique()),
        "risk_rolling_summary": summarize_risk_rolling(risk_rolling),
        "risk_family_summary": summarize_family_rolling(family_rolling),
    }
    (args.output_root / "stop_risk_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
