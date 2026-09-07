"""R4 trigger-entry winner/loser characterization and stock-selection diagnostics.

This is an explicitly retrospective study over known history. It does not create a
new unseen holdout and does not use an LLM. The primary question is no longer
"which global rule has the highest historical winner rate?". Instead, after a BF
candidate reaches an executable trigger entry, R4 asks which point-in-time stock
and execution features are associated with:

- +20% before -8% within 15 trading sessions (fast winner);
- -8% before +20% within 15 trading sessions (stop first);
- unresolved / ambiguous 3-week paths;
- W1/W2/W3/W4 return and excess-return paths;
- 3-week and 4-week MAE/MFE.

Market timing and stock selection are separated in two ways:

1. report the frozen R3 favorable-market regime as a historical conditioning scope;
2. compare fast winners and stop-first losers within the same snapshot_date, where
   all candidates share the same broad-market M_* state.

Stock interactions use stock/execution features only. M_* features are never
allowed into the stock-condition search. Rolling folds regenerate stock thresholds
from past quarters only and evaluate exactly the next quarter.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .dataset import DISCOVERY_FEATURE_ALLOWLIST, load_replay_candidates
from .outcomes import (
    OutcomeConfig,
    _benchmark_return,
    _resolve_executable_entry,
    evaluate_candidate_path,
    load_price_pickle,
    point_in_time_market_features,
    restrict_to_mature_outcome_quarters,
)
from .pipeline_contract import validate_replay_preflight

R3_FAVORABLE_REGIME = {
    "M_8w_drawdown_max": -0.04719988,
    "M_dist_52w_high_min": -0.05692191,
}

WEEK_SESSIONS = {"w1": 5, "w2": 10, "w3": 15, "w4": 20}
EXECUTION_FEATURES = (
    "entry_delay_sessions",
    "entry_extension_pct",
    "entry_is_gap_or_open",
)
BOOLEAN_FEATURES = {"pullback_v_is_dry", "entry_is_gap_or_open"}
SEARCH_QUANTILES = (0.20, 0.40, 0.60, 0.80)
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


def _num(value: Any) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def _feature_value(name: str, value: Any) -> float:
    if name == "pullback_v_is_dry":
        if pd.isna(value):
            return float("nan")
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "t", "yes"}:
            return 1.0
        if normalized in {"0", "false", "f", "no"}:
            return 0.0
        return float("nan")
    return _num(value)


def _first_hit_date(window: pd.DataFrame, *, entry_price: float, threshold: float, up: bool) -> pd.Timestamp | None:
    if up:
        rows = window.loc[window["High"] >= entry_price * (1.0 + threshold), "date"]
    else:
        rows = window.loc[window["Low"] <= entry_price * (1.0 + threshold), "date"]
    return None if rows.empty else pd.Timestamp(rows.iloc[0])


def _path_3w_label(
    window: pd.DataFrame,
    *,
    entry_price: float,
    entry_method: str,
    stop_loss: float,
    winner_gain: float,
) -> tuple[str, str, pd.Timestamp | None, pd.Timestamp | None]:
    target = _first_hit_date(window, entry_price=entry_price, threshold=winner_gain, up=True)
    stop = _first_hit_date(window, entry_price=entry_price, threshold=stop_loss, up=False)
    first_bar = window.iloc[0]
    if entry_method == "intraday_trigger" and float(first_bar["Low"]) <= entry_price * (1.0 + stop_loss):
        return "ambiguous_3w", "entry_day_stop_order_unknown", target, stop
    if target is not None and stop is not None and target == stop:
        return "ambiguous_3w", "same_bar_target_stop_order_unknown", target, stop
    if target is not None and (stop is None or target < stop):
        return "fast_winner_3w", "", target, stop
    if stop is not None and (target is None or stop < target):
        return "stop_first_3w", "", target, stop
    return "unresolved_3w", "", target, stop


def evaluate_trigger_path(
    prices: pd.DataFrame,
    signal_date: str | pd.Timestamp,
    *,
    trigger_price: float,
    spy_prices: pd.DataFrame,
    config: OutcomeConfig = OutcomeConfig(),
) -> dict[str, Any]:
    """Evaluate the causal entry then 3-week first-passage and W1-W4 path."""
    sig = pd.Timestamp(signal_date).tz_localize(None).normalize()
    canonical = evaluate_candidate_path(
        prices,
        sig,
        trigger_price=trigger_price,
        spy_prices=spy_prices,
        config=config,
    )
    if canonical.get("label") == "censored":
        return {"usable": False, "reason": canonical.get("reason", "censored")}

    entry = _resolve_executable_entry(prices, sig, float(trigger_price), config)
    if "entry_date" not in entry:
        return {"usable": False, "reason": entry.get("reason", "no_executable_entry")}
    entry_date = pd.Timestamp(entry["entry_date"])
    entry_price = float(entry["entry_price"])
    post = prices.loc[prices["date"] >= entry_date].reset_index(drop=True)
    if len(post) < config.minimum_sessions:
        return {"usable": False, "reason": "insufficient_future_sessions"}

    window_3w = post.iloc[: WEEK_SESSIONS["w3"]].copy()
    window_4w = post.iloc[: WEEK_SESSIONS["w4"]].copy()
    path_label, path_reason, target_3w, stop_3w = _path_3w_label(
        window_3w,
        entry_price=entry_price,
        entry_method=str(entry["entry_method"]),
        stop_loss=config.stop_loss,
        winner_gain=config.winner_gain,
    )

    result: dict[str, Any] = {
        "usable": True,
        "reason": "",
        "entry_date": entry_date,
        "entry_price": entry_price,
        "entry_method": str(entry["entry_method"]),
        "entry_delay_sessions": int(entry["entry_index"]) + 1,
        "entry_extension_pct": entry_price / float(entry["trigger_price_adjusted"]) - 1.0,
        "entry_is_gap_or_open": int(str(entry["entry_method"]) == "gap_or_open"),
        "trigger_price_adjusted": float(entry["trigger_price_adjusted"]),
        "path_3w": path_label,
        "path_3w_reason": path_reason,
        "target_date_3w": target_3w,
        "stop_date_3w": stop_3w,
        "fast_winner_3w": int(path_label == "fast_winner_3w"),
        "stop_first_3w": int(path_label == "stop_first_3w"),
        "unresolved_3w": int(path_label == "unresolved_3w"),
        "ambiguous_3w": int(path_label == "ambiguous_3w"),
        "stop_first_then_winner_12w": int(
            path_label == "stop_first_3w" and canonical.get("label") == "stop_out_then_winner"
        ),
        "canonical_12w_label": canonical.get("label"),
        "mae_3w": float(window_3w["Low"].min() / entry_price - 1.0),
        "mfe_3w": float(window_3w["High"].max() / entry_price - 1.0),
        "mae_4w": float(window_4w["Low"].min() / entry_price - 1.0),
        "mfe_4w": float(window_4w["High"].max() / entry_price - 1.0),
    }
    for name, sessions in WEEK_SESSIONS.items():
        row = post.iloc[sessions - 1]
        exit_date = pd.Timestamp(row["date"])
        stock_return = float(row["Close"] / entry_price - 1.0)
        benchmark = _benchmark_return(spy_prices, entry_date, exit_date)
        result[f"return_{name}"] = stock_return
        result[f"excess_{name}"] = None if benchmark is None else stock_return - benchmark
        result[f"exit_date_{name}"] = exit_date
    return result


def build_trigger_path_frame(
    candidates: pd.DataFrame,
    price_map: Mapping[str, pd.DataFrame],
    spy_prices: pd.DataFrame,
    *,
    config: OutcomeConfig = OutcomeConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the R4 row surface and a censor/reviewer table."""
    rows: list[dict[str, Any]] = []
    reviewer: list[dict[str, Any]] = []
    available_stock_features = [f for f in DISCOVERY_FEATURE_ALLOWLIST if f in candidates.columns]
    for source in candidates.itertuples(index=False):
        data = source._asdict()
        code = str(data["code"])
        signal_date = pd.Timestamp(data["snapshot_date"]).normalize()
        prices = price_map.get(code)
        if prices is None or prices.empty:
            reviewer.append({"code": code, "snapshot_date": signal_date, "usable": False, "reason": "missing_price_data"})
            continue
        trigger = _num(data.get("ibd_trigger_price"))
        if not np.isfinite(trigger):
            trigger = _num(data.get("ibd_candidate_price"))
        if not np.isfinite(trigger):
            reviewer.append({"code": code, "snapshot_date": signal_date, "usable": False, "reason": "missing_trigger_price"})
            continue
        path = evaluate_trigger_path(
            prices,
            signal_date,
            trigger_price=trigger,
            spy_prices=spy_prices,
            config=config,
        )
        if not bool(path.get("usable")):
            reviewer.append({"code": code, "snapshot_date": signal_date, **path})
            continue
        market = point_in_time_market_features(spy_prices, signal_date)
        favorable = (
            market.get("M_8w_drawdown") is not None
            and market.get("M_dist_52w_high") is not None
            and float(market["M_8w_drawdown"]) <= R3_FAVORABLE_REGIME["M_8w_drawdown_max"]
            and float(market["M_dist_52w_high"]) >= R3_FAVORABLE_REGIME["M_dist_52w_high_min"]
        )
        row: dict[str, Any] = {
            "code": code,
            "snapshot_date": signal_date,
            "signal_quarter": str(signal_date.to_period("Q")),
            "entry_quarter": str(pd.Timestamp(path["entry_date"]).to_period("Q")),
            "r3_favorable_regime": int(favorable),
            **market,
            **path,
        }
        for feature in available_stock_features:
            row[feature] = _feature_value(feature, data.get(feature))
        rows.append(row)
        reviewer.append({"code": code, "snapshot_date": signal_date, "usable": True, "reason": ""})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("R4 trigger-path frame is empty")
    return frame.sort_values(["snapshot_date", "code"]).reset_index(drop=True), pd.DataFrame(reviewer)


def stock_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        feature
        for feature in [*DISCOVERY_FEATURE_ALLOWLIST, *EXECUTION_FEATURES]
        if feature in frame.columns
    ]


def _finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def summarize_path(frame: pd.DataFrame, mask: np.ndarray | None = None) -> dict[str, Any]:
    """Summarize path probabilities using all non-ambiguous entries as denominator."""
    if mask is None:
        selected = frame.copy()
    else:
        if len(mask) != len(frame):
            raise ValueError("mask length differs from frame")
        selected = frame.loc[np.asarray(mask, dtype=bool)].copy()
    selected_n = int(len(selected))
    ambiguous = selected["ambiguous_3w"].astype(int) == 1
    evaluable = selected.loc[~ambiguous].copy()
    evaluable_n = int(len(evaluable))
    fast_n = int(evaluable["fast_winner_3w"].sum()) if evaluable_n else 0
    stop_n = int(evaluable["stop_first_3w"].sum()) if evaluable_n else 0
    unresolved_n = int(evaluable["unresolved_3w"].sum()) if evaluable_n else 0
    recovered_n = int(evaluable["stop_first_then_winner_12w"].sum()) if evaluable_n else 0

    def q(column: str, quantile: float = 0.5) -> float | None:
        values = _finite(evaluable[column]).dropna()
        return float(values.quantile(quantile)) if not values.empty else None

    mae3 = q("mae_3w")
    mfe3 = q("mfe_3w")
    rr3 = None if mae3 is None or mfe3 is None or mae3 >= 0 or abs(mae3) < 1e-12 else mfe3 / abs(mae3)
    return {
        "selected_n": selected_n,
        "evaluable_n": evaluable_n,
        "ambiguous_n": int(ambiguous.sum()),
        "fast_winner_n": fast_n,
        "stop_first_n": stop_n,
        "unresolved_n": unresolved_n,
        "stop_first_then_winner_12w_n": recovered_n,
        "fast_winner_rate": fast_n / evaluable_n if evaluable_n else None,
        "stop_first_rate": stop_n / evaluable_n if evaluable_n else None,
        "unresolved_rate": unresolved_n / evaluable_n if evaluable_n else None,
        "stop_first_then_winner_12w_rate": recovered_n / stop_n if stop_n else None,
        "return_w1_p50": q("return_w1"),
        "return_w2_p50": q("return_w2"),
        "return_w3_p50": q("return_w3"),
        "return_w4_p50": q("return_w4"),
        "excess_w1_p50": q("excess_w1"),
        "excess_w2_p50": q("excess_w2"),
        "excess_w3_p50": q("excess_w3"),
        "excess_w4_p50": q("excess_w4"),
        "mae_3w_p50": mae3,
        "mfe_3w_p50": mfe3,
        "mae_4w_p50": q("mae_4w"),
        "mfe_4w_p50": q("mfe_4w"),
        "mfe_mae_ratio_3w": rr3,
    }


def _scope_frame(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "all":
        return frame.reset_index(drop=True)
    if scope == "r3_favorable":
        return frame.loc[frame["r3_favorable_regime"].astype(int) == 1].reset_index(drop=True)
    raise ValueError(f"unknown scope: {scope}")


def _bin_definitions(frame: pd.DataFrame, features: Sequence[str]) -> dict[str, list[float]]:
    definitions: dict[str, list[float]] = {}
    for feature in features:
        values = _finite(frame[feature]).dropna()
        if values.empty:
            continue
        if feature in BOOLEAN_FEATURES or values.nunique() <= 2:
            definitions[feature] = sorted(float(v) for v in values.unique())
            continue
        cuts = sorted({float(values.quantile(q)) for q in SEARCH_QUANTILES})
        if cuts:
            definitions[feature] = cuts
    return definitions


def _feature_bin_masks(frame: pd.DataFrame, feature: str, cuts: Sequence[float]) -> list[tuple[str, np.ndarray, float | None, float | None]]:
    values = _finite(frame[feature]).to_numpy(dtype=float)
    finite = np.isfinite(values)
    if feature in BOOLEAN_FEATURES or len(cuts) <= 2 and set(cuts).issubset({0.0, 1.0}):
        return [
            (f"={value:g}", finite & np.isclose(values, value), value, value)
            for value in cuts
        ]
    edges = [-np.inf, *sorted(set(float(c) for c in cuts)), np.inf]
    out: list[tuple[str, np.ndarray, float | None, float | None]] = []
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        if index == 0:
            mask = finite & (values <= high)
        elif index == len(edges) - 2:
            mask = finite & (values > low)
        else:
            mask = finite & (values > low) & (values <= high)
        label = f"({low:.6g},{high:.6g}]"
        out.append((label, mask, None if not np.isfinite(low) else low, None if not np.isfinite(high) else high))
    return out


def feature_bin_characterization(
    frame: pd.DataFrame,
    features: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe every feature bin in all-history and frozen favorable-regime scopes."""
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
                if metrics["selected_n"] == 0:
                    continue
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
                        "fast_winner_lift": (
                            metrics["fast_winner_rate"] - baseline["fast_winner_rate"]
                            if metrics["fast_winner_rate"] is not None and baseline["fast_winner_rate"] is not None
                            else None
                        ),
                        "stop_first_reduction": (
                            baseline["stop_first_rate"] - metrics["stop_first_rate"]
                            if metrics["stop_first_rate"] is not None and baseline["stop_first_rate"] is not None
                            else None
                        ),
                    }
                )
    bins = pd.DataFrame(rows)
    extreme_rows: list[dict[str, Any]] = []
    if not bins.empty:
        for (scope, feature), group in bins.groupby(["scope", "feature"], sort=True):
            supported = group.loc[group["evaluable_n"] >= 20].copy()
            if supported.empty:
                continue
            winner_row = supported.sort_values(["fast_winner_rate", "stop_first_rate"], ascending=[False, True]).iloc[0]
            loser_row = supported.sort_values(["stop_first_rate", "fast_winner_rate"], ascending=[False, True]).iloc[0]
            extreme_rows.append(
                {
                    "scope": scope,
                    "feature": feature,
                    "winner_bin": winner_row["bin_label"],
                    "winner_evaluable_n": int(winner_row["evaluable_n"]),
                    "winner_fast_winner_rate": winner_row["fast_winner_rate"],
                    "winner_fast_lift": winner_row["fast_winner_lift"],
                    "winner_stop_first_rate": winner_row["stop_first_rate"],
                    "winner_excess_w3_p50": winner_row["excess_w3_p50"],
                    "winner_mae_3w_p50": winner_row["mae_3w_p50"],
                    "winner_mfe_3w_p50": winner_row["mfe_3w_p50"],
                    "loser_bin": loser_row["bin_label"],
                    "loser_evaluable_n": int(loser_row["evaluable_n"]),
                    "loser_stop_first_rate": loser_row["stop_first_rate"],
                    "loser_stop_lift": (
                        loser_row["stop_first_rate"] - loser_row["baseline_stop_first_rate"]
                        if pd.notna(loser_row["stop_first_rate"]) and pd.notna(loser_row["baseline_stop_first_rate"])
                        else None
                    ),
                    "loser_fast_winner_rate": loser_row["fast_winner_rate"],
                    "loser_excess_w3_p50": loser_row["excess_w3_p50"],
                    "loser_mae_3w_p50": loser_row["mae_3w_p50"],
                    "loser_mfe_3w_p50": loser_row["mfe_3w_p50"],
                }
            )
    return bins, pd.DataFrame(extreme_rows)


def _pairwise_probability_greater(left: np.ndarray, right: np.ndarray) -> tuple[float | None, int]:
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size == 0 or right.size == 0:
        return None, 0
    comparisons = (left[:, None] > right[None, :]).astype(float)
    comparisons += 0.5 * (left[:, None] == right[None, :])
    return float(comparisons.mean()), int(comparisons.size)


def within_snapshot_feature_contrasts(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Compare fast winners with stop-first losers within identical signal snapshots."""
    rows: list[dict[str, Any]] = []
    for scope in ("all", "r3_favorable"):
        scoped = _scope_frame(frame, scope)
        for feature in features:
            snapshot_diffs: list[float] = []
            snapshot_aucs: list[float] = []
            weighted_auc_num = 0.0
            weighted_auc_den = 0
            for _, snapshot in scoped.groupby("snapshot_date", sort=True):
                fast = _finite(snapshot.loc[snapshot["fast_winner_3w"].astype(int) == 1, feature]).dropna().to_numpy(dtype=float)
                stop = _finite(snapshot.loc[snapshot["stop_first_3w"].astype(int) == 1, feature]).dropna().to_numpy(dtype=float)
                if fast.size == 0 or stop.size == 0:
                    continue
                auc, pairs = _pairwise_probability_greater(fast, stop)
                if auc is None:
                    continue
                snapshot_aucs.append(auc)
                snapshot_diffs.append(float(np.median(fast) - np.median(stop)))
                weighted_auc_num += auc * pairs
                weighted_auc_den += pairs
            if not snapshot_aucs:
                continue
            diffs = np.asarray(snapshot_diffs, dtype=float)
            aucs = np.asarray(snapshot_aucs, dtype=float)
            rows.append(
                {
                    "scope": scope,
                    "feature": feature,
                    "matched_snapshot_count": int(len(snapshot_aucs)),
                    "winner_gt_stop_pair_probability": weighted_auc_num / weighted_auc_den if weighted_auc_den else None,
                    "equal_weight_snapshot_auc_p50": float(np.median(aucs)),
                    "snapshot_median_difference_p50": float(np.median(diffs)),
                    "positive_difference_snapshot_fraction": float(np.mean(diffs > 0)),
                    "pair_count": int(weighted_auc_den),
                }
            )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    threshold: float

    def key(self) -> tuple[str, str, float]:
        return self.feature, self.op, round(float(self.threshold), 12)

    def as_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "op": self.op, "threshold": float(self.threshold)}


def generate_stock_conditions(frame: pd.DataFrame, features: Sequence[str]) -> list[Condition]:
    conditions: list[Condition] = []
    seen: set[tuple[str, str, float]] = set()
    for feature in features:
        values = _finite(frame[feature]).dropna()
        if values.empty:
            continue
        if feature in BOOLEAN_FEATURES or values.nunique() <= 2:
            for value in sorted(float(v) for v in values.unique()):
                condition = Condition(feature, "==", value)
                if condition.key() not in seen:
                    seen.add(condition.key())
                    conditions.append(condition)
            continue
        for q in SEARCH_QUANTILES:
            threshold = float(values.quantile(q))
            for op in (">=", "<="):
                condition = Condition(feature, op, threshold)
                if condition.key() not in seen:
                    seen.add(condition.key())
                    conditions.append(condition)
    return conditions


def condition_mask(frame: pd.DataFrame, condition: Condition) -> np.ndarray:
    values = _finite(frame[condition.feature]).to_numpy(dtype=float)
    finite = np.isfinite(values)
    if condition.op == ">=":
        return finite & (values >= condition.threshold)
    if condition.op == "<=":
        return finite & (values <= condition.threshold)
    if condition.op == "==":
        return finite & np.isclose(values, condition.threshold)
    raise ValueError(condition.op)


def _rule_json(conditions: Sequence[Condition]) -> str:
    payload = {"all": [condition.as_dict() for condition in sorted(conditions, key=lambda c: c.key())]}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _quarter_stability(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    min_quarter_n: int,
) -> dict[str, Any]:
    edges: list[float] = []
    evaluated = 0
    positive = 0
    quarters = frame["entry_quarter"].astype(str).to_numpy()
    for quarter in sorted(pd.unique(quarters)):
        q_mask = quarters == quarter
        selected = q_mask & mask
        q_frame = frame.loc[q_mask].reset_index(drop=True)
        q_selected = frame.loc[selected].reset_index(drop=True)
        baseline = summarize_path(q_frame)
        metrics = summarize_path(q_selected)
        if int(metrics["evaluable_n"]) < min_quarter_n:
            continue
        if baseline["fast_winner_rate"] is None or baseline["stop_first_rate"] is None:
            continue
        evaluated += 1
        edge = (
            (metrics["fast_winner_rate"] - baseline["fast_winner_rate"])
            + (baseline["stop_first_rate"] - metrics["stop_first_rate"])
        )
        edges.append(float(edge))
        if edge > 0:
            positive += 1
    return {
        "evaluated_quarters": evaluated,
        "positive_edge_quarter_fraction": positive / evaluated if evaluated else None,
        "median_quarter_path_edge": float(np.median(edges)) if edges else None,
        "worst_quarter_path_edge": min(edges) if edges else None,
    }


def _record_rule(
    frame: pd.DataFrame,
    mask: np.ndarray,
    conditions: Sequence[Condition],
    baseline: Mapping[str, Any],
    *,
    min_quarter_n: int,
) -> dict[str, Any]:
    metrics = summarize_path(frame, mask)
    fast_lift = (
        metrics["fast_winner_rate"] - baseline["fast_winner_rate"]
        if metrics["fast_winner_rate"] is not None and baseline["fast_winner_rate"] is not None
        else None
    )
    stop_reduction = (
        baseline["stop_first_rate"] - metrics["stop_first_rate"]
        if metrics["stop_first_rate"] is not None and baseline["stop_first_rate"] is not None
        else None
    )
    stability = _quarter_stability(frame, mask, min_quarter_n=min_quarter_n)
    return {
        "rule_json": _rule_json(conditions),
        "condition_count": len(conditions),
        **metrics,
        "baseline_fast_winner_rate": baseline["fast_winner_rate"],
        "baseline_stop_first_rate": baseline["stop_first_rate"],
        "fast_winner_lift": fast_lift,
        "stop_first_reduction": stop_reduction,
        "path_edge": fast_lift + stop_reduction if fast_lift is not None and stop_reduction is not None else None,
        **stability,
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
        "positive_edge_quarter_fraction": 0.05,
        "worst_quarter_path_edge": 0.05,
    }
    score = pd.Series(0.0, index=scored.index)
    for metric, weight in weights.items():
        numeric = pd.to_numeric(scored[metric], errors="coerce")
        score += weight * numeric.rank(pct=True, method="average", na_option="bottom").fillna(0.0)
    score -= 0.01 * (pd.to_numeric(scored["condition_count"]) - 1).clip(lower=0)
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
    """Exhaustively search stock-only singles and distinct-feature two-way AND rules."""
    if any(str(feature).startswith("M_") for feature in features):
        raise ValueError("M_* market features are forbidden in R4 stock-condition search")
    baseline = summarize_path(frame)
    conditions = generate_stock_conditions(frame, features)
    masks = {condition.key(): condition_mask(frame, condition) for condition in conditions}
    records: list[dict[str, Any]] = []
    single_supported = 0
    pair_tested = 0
    pair_supported = 0

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
            single_supported += 1

    for left, right in itertools.combinations(conditions, 2):
        if left.feature == right.feature:
            continue
        pair_tested += 1
        mask = masks[left.key()] & masks[right.key()]
        if int(np.count_nonzero(mask)) < min_selected:
            continue
        record = _record_rule(frame, mask, [left, right], baseline, min_quarter_n=min_quarter_n)
        if eligible(record):
            records.append(record)
            pair_supported += 1

    scored = _score_interactions(records)
    audit = {
        "generated_stock_condition_count": len(conditions),
        "supported_single_count": single_supported,
        "distinct_feature_pair_count_tested": pair_tested,
        "supported_pair_count": pair_supported,
        "candidate_rule_count": int(len(scored)),
    }
    return scored, audit


def matched_selected_vs_unselected(frame: pd.DataFrame, rule_json: str) -> dict[str, Any]:
    payload = json.loads(rule_json)
    conditions = [Condition(str(c["feature"]), str(c["op"]), float(c["threshold"])) for c in payload["all"]]
    mask = np.ones(len(frame), dtype=bool)
    for condition in conditions:
        mask &= condition_mask(frame, condition)
    fast_lifts: list[float] = []
    stop_reductions: list[float] = []
    matched = 0
    for _, snapshot in frame.assign(_selected=mask).groupby("snapshot_date", sort=True):
        selected = snapshot.loc[snapshot["_selected"]].reset_index(drop=True)
        unselected = snapshot.loc[~snapshot["_selected"]].reset_index(drop=True)
        if selected.empty or unselected.empty:
            continue
        selected_metrics = summarize_path(selected)
        unselected_metrics = summarize_path(unselected)
        if min(int(selected_metrics["evaluable_n"]), int(unselected_metrics["evaluable_n"])) < 1:
            continue
        matched += 1
        fast_lifts.append(float(selected_metrics["fast_winner_rate"] - unselected_metrics["fast_winner_rate"]))
        stop_reductions.append(float(unselected_metrics["stop_first_rate"] - selected_metrics["stop_first_rate"]))
    return {
        "matched_snapshot_count": matched,
        "matched_fast_winner_lift_p50": float(np.median(fast_lifts)) if fast_lifts else None,
        "matched_stop_first_reduction_p50": float(np.median(stop_reductions)) if stop_reductions else None,
        "matched_path_edge_p50": (
            float(np.median(np.asarray(fast_lifts) + np.asarray(stop_reductions)))
            if fast_lifts and stop_reductions
            else None
        ),
        "matched_positive_path_edge_fraction": (
            float(np.mean((np.asarray(fast_lifts) + np.asarray(stop_reductions)) > 0))
            if fast_lifts and stop_reductions
            else None
        ),
    }


def add_matched_metrics(frame: pd.DataFrame, scored: pd.DataFrame, *, top_n: int = 100) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    out = scored.copy()
    for column in (
        "matched_snapshot_count",
        "matched_fast_winner_lift_p50",
        "matched_stop_first_reduction_p50",
        "matched_path_edge_p50",
        "matched_positive_path_edge_fraction",
    ):
        out[column] = np.nan
    for index in out.head(top_n).index:
        metrics = matched_selected_vs_unselected(frame, str(out.at[index, "rule_json"]))
        for key, value in metrics.items():
            out.at[index, key] = value
    return out


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
        min_selected = 60 if scope == "all" else 30
        min_evaluable = 45 if scope == "all" else 20
        min_eval_q = 4 if scope == "all" else 2
        scored, _ = search_stock_interactions(
            train,
            features,
            min_selected=min_selected,
            min_evaluable=min_evaluable,
            min_quarter_n=max(5, min_quarter_n // 2),
            min_evaluated_quarters=min_eval_q,
        )
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
                }
            )
            continue
        best = scored.iloc[0]
        rule_json = str(best["rule_json"])
        payload = json.loads(rule_json)
        conditions = [Condition(str(c["feature"]), str(c["op"]), float(c["threshold"])) for c in payload["all"]]
        mask = np.ones(len(test), dtype=bool)
        for condition in conditions:
            mask &= condition_mask(test, condition)
        test_baseline = summarize_path(test)
        test_metrics = summarize_path(test, mask)
        fast_lift = (
            test_metrics["fast_winner_rate"] - test_baseline["fast_winner_rate"]
            if test_metrics["fast_winner_rate"] is not None and test_baseline["fast_winner_rate"] is not None
            else None
        )
        stop_reduction = (
            test_baseline["stop_first_rate"] - test_metrics["stop_first_rate"]
            if test_metrics["stop_first_rate"] is not None and test_baseline["stop_first_rate"] is not None
            else None
        )
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
                **{f"test_{key}": value for key, value in test_metrics.items()},
                "test_baseline_fast_winner_rate": test_baseline["fast_winner_rate"],
                "test_baseline_stop_first_rate": test_baseline["stop_first_rate"],
                "test_fast_winner_lift": fast_lift,
                "test_stop_first_reduction": stop_reduction,
                "test_path_edge": fast_lift + stop_reduction if fast_lift is not None and stop_reduction is not None else None,
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
        _rolling_scope_search(
            frame,
            features,
            scope=scope,
            min_train_quarters=min_train_quarters,
            min_quarter_n=min_quarter_n,
        )
        for scope in ("all", "r3_favorable")
    ]
    nonempty = [part for part in parts if not part.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def summarize_rolling(rolling: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if rolling.empty:
        return summary
    for scope, group in rolling.groupby("scope", sort=True):
        path_edge = pd.to_numeric(group.get("test_path_edge"), errors="coerce")
        selected = pd.to_numeric(group.get("test_selected_n"), errors="coerce").fillna(0)
        evaluable = pd.to_numeric(group.get("test_evaluable_n"), errors="coerce").fillna(0)
        summary[scope] = {
            "folds": int(len(group)),
            "zero_selection_folds": int((selected == 0).sum()),
            "zero_selection_fraction": float((selected == 0).mean()),
            "evaluable_path_edge_folds": int(path_edge.notna().sum()),
            "positive_path_edge_all_fold_fraction": float((path_edge.fillna(-np.inf) > 0).mean()),
            "positive_path_edge_evaluable_fraction": (
                float((path_edge.dropna() > 0).mean()) if path_edge.notna().any() else None
            ),
            "path_edge_p50_evaluable": float(path_edge.dropna().median()) if path_edge.notna().any() else None,
            "selected_n_total": int(selected.sum()),
            "evaluable_n_total": int(evaluable.sum()),
        }
    return summary


def main() -> int:
    args = parse_args()
    r1 = Path("backtest/blind_rule_discovery/output/rd_agent_run_01").resolve()
    r2 = Path("backtest/blind_rule_discovery/output/retrospective_ceiling_r2").resolve()
    r3 = Path("backtest/blind_rule_discovery/output/retrospective_ceiling_r3").resolve()
    if args.output_root.resolve() in {r1, r2, r3}:
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

    frame.to_csv(args.output_root / "trigger_path_samples.csv", index=False)
    bins, extremes = feature_bin_characterization(frame, features)
    bins.to_csv(args.output_root / "feature_bin_characterization.csv", index=False)
    extremes.to_csv(args.output_root / "feature_extremes.csv", index=False)
    contrasts = within_snapshot_feature_contrasts(frame, features)
    contrasts.to_csv(args.output_root / "within_snapshot_feature_contrasts.csv", index=False)

    search_outputs: dict[str, pd.DataFrame] = {}
    search_audit: dict[str, Any] = {}
    for scope in ("all", "r3_favorable"):
        scoped = _scope_frame(frame, scope)
        if scope == "all":
            min_selected = args.min_selected_all
            min_evaluable = args.min_evaluable_all
            min_eval_q = args.min_evaluated_quarters_all
        else:
            min_selected = args.min_selected_favorable
            min_evaluable = args.min_evaluable_favorable
            min_eval_q = args.min_evaluated_quarters_favorable
        scored, audit = search_stock_interactions(
            scoped,
            features,
            min_selected=min_selected,
            min_evaluable=min_evaluable,
            min_quarter_n=args.min_quarter_n,
            min_evaluated_quarters=min_eval_q,
        )
        scored = add_matched_metrics(scoped, scored, top_n=100)
        scored.to_csv(args.output_root / f"stock_interactions_{scope}.csv", index=False)
        search_outputs[scope] = scored
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
        "research_mode": "r4_trigger_path_winner_loser_characterization",
        "canonical_blind_experiment": False,
        "unseen_holdout_claim_allowed": False,
        "llm_used": False,
        "primary_outcome": "+20% before -8% within 15 trading sessions from executable entry",
        "primary_denominator": "all full-path non-ambiguous entries; unresolved remains in denominator",
        "secondary_outcomes": [
            "-8% before +20% within 15 sessions",
            "W1/W2/W3/W4 return and excess return",
            "3w/4w MAE and MFE",
            "stop-first then +20% within canonical 12w path",
        ],
        "market_control": {
            "frozen_r3_favorable_regime": R3_FAVORABLE_REGIME,
            "favorable_regime_is_retrospective_context_only": True,
            "within_snapshot_comparison": "fast winners vs stop-first losers at identical snapshot_date",
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
            "stock_interactions_r3_favorable.csv",
            "rolling_stock_selection.csv",
        ],
    }
    (args.output_root / "trigger_path_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
