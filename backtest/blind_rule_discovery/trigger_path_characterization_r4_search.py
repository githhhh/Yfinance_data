"""Vectorized stock-only interaction search for frozen R4 characterization.

The search space and quality-score semantics match R4, but candidate evaluation is
kept compact: only metrics required for search/ranking are computed for every rule.
W1-W4 full distribution metrics are enriched only for reported top rules by the
frozen runner. This avoids millions of redundant quantile calculations in rolling
re-search without pruning any single or distinct-feature pair condition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .trigger_path_characterization import Condition, condition_mask, generate_stock_conditions
from .trigger_path_characterization_r4 import _rule_json, rule_mask, summarize_path


def _values(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)


@dataclass(frozen=True)
class PathSearchContext:
    row_count: int
    evaluable: np.ndarray
    fast: np.ndarray
    stop: np.ndarray
    unresolved: np.ndarray
    recovered: np.ndarray
    excess_w3: np.ndarray
    mae_3w: np.ndarray
    mfe_3w: np.ndarray
    quarters: tuple[str, ...]
    quarter_masks: tuple[np.ndarray, ...]
    baseline_fast_rate: float | None
    baseline_stop_rate: float | None
    quarter_baseline_fast: tuple[float | None, ...]
    quarter_baseline_stop: tuple[float | None, ...]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "PathSearchContext":
        ambiguous = frame["ambiguous_3w"].astype(int).to_numpy() == 1
        evaluable = ~ambiguous
        fast = frame["fast_winner_3w"].astype(int).to_numpy() == 1
        stop = frame["stop_first_3w"].astype(int).to_numpy() == 1
        unresolved = frame["unresolved_3w"].astype(int).to_numpy() == 1
        recovered = frame["stop_first_then_winner_12w"].astype(int).to_numpy() == 1
        quarter_values = frame["entry_quarter"].astype(str).to_numpy()
        quarters = tuple(sorted(pd.unique(quarter_values)))
        quarter_masks = tuple(quarter_values == quarter for quarter in quarters)
        eval_n = int(np.count_nonzero(evaluable))
        baseline_fast = float(np.count_nonzero(evaluable & fast) / eval_n) if eval_n else None
        baseline_stop = float(np.count_nonzero(evaluable & stop) / eval_n) if eval_n else None
        q_fast: list[float | None] = []
        q_stop: list[float | None] = []
        for q_mask in quarter_masks:
            q_eval = q_mask & evaluable
            q_n = int(np.count_nonzero(q_eval))
            q_fast.append(float(np.count_nonzero(q_eval & fast) / q_n) if q_n else None)
            q_stop.append(float(np.count_nonzero(q_eval & stop) / q_n) if q_n else None)
        return cls(
            row_count=len(frame),
            evaluable=evaluable,
            fast=fast,
            stop=stop,
            unresolved=unresolved,
            recovered=recovered,
            excess_w3=_values(frame, "excess_w3"),
            mae_3w=_values(frame, "mae_3w"),
            mfe_3w=_values(frame, "mfe_3w"),
            quarters=quarters,
            quarter_masks=quarter_masks,
            baseline_fast_rate=baseline_fast,
            baseline_stop_rate=baseline_stop,
            quarter_baseline_fast=tuple(q_fast),
            quarter_baseline_stop=tuple(q_stop),
        )

    @staticmethod
    def _q(values: np.ndarray, mask: np.ndarray, q: float) -> float | None:
        selected = values[mask]
        selected = selected[np.isfinite(selected)]
        return float(np.quantile(selected, q)) if selected.size else None

    def evaluate(self, mask: np.ndarray, *, min_quarter_n: int) -> dict[str, Any]:
        if len(mask) != self.row_count:
            raise ValueError("mask length differs from R4 search context")
        selected_n = int(np.count_nonzero(mask))
        selected_eval = mask & self.evaluable
        eval_n = int(np.count_nonzero(selected_eval))
        fast_n = int(np.count_nonzero(selected_eval & self.fast))
        stop_n = int(np.count_nonzero(selected_eval & self.stop))
        unresolved_n = int(np.count_nonzero(selected_eval & self.unresolved))
        recovered_n = int(np.count_nonzero(selected_eval & self.recovered))
        persistent_n = max(0, stop_n - recovered_n)
        fast_rate = fast_n / eval_n if eval_n else None
        stop_rate = stop_n / eval_n if eval_n else None
        fast_lift = (
            fast_rate - self.baseline_fast_rate
            if fast_rate is not None and self.baseline_fast_rate is not None
            else None
        )
        stop_reduction = (
            self.baseline_stop_rate - stop_rate
            if stop_rate is not None and self.baseline_stop_rate is not None
            else None
        )
        edges: list[float] = []
        stop_lifts: list[float] = []
        for q_mask, base_fast, base_stop in zip(
            self.quarter_masks, self.quarter_baseline_fast, self.quarter_baseline_stop
        ):
            q_selected = selected_eval & q_mask
            q_n = int(np.count_nonzero(q_selected))
            if q_n < min_quarter_n or base_fast is None or base_stop is None:
                continue
            q_fast = float(np.count_nonzero(q_selected & self.fast) / q_n)
            q_stop = float(np.count_nonzero(q_selected & self.stop) / q_n)
            edges.append((q_fast - base_fast) + (base_stop - q_stop))
            stop_lifts.append(q_stop - base_stop)
        edge_array = np.asarray(edges, dtype=float)
        stop_array = np.asarray(stop_lifts, dtype=float)
        mae = self._q(self.mae_3w, selected_eval, 0.50)
        mfe = self._q(self.mfe_3w, selected_eval, 0.50)
        rr = None if mae is None or mfe is None or mae >= 0 or abs(mae) < 1e-12 else mfe / abs(mae)
        return {
            "selected_n": selected_n,
            "evaluable_n": eval_n,
            "ambiguous_n": selected_n - int(np.count_nonzero(mask & self.evaluable)),
            "fast_winner_n": fast_n,
            "stop_first_n": stop_n,
            "unresolved_n": unresolved_n,
            "stop_first_then_winner_12w_n": recovered_n,
            "persistent_stop_first_n": persistent_n,
            "fast_winner_rate": fast_rate,
            "stop_first_rate": stop_rate,
            "unresolved_rate": unresolved_n / eval_n if eval_n else None,
            "stop_first_then_winner_12w_rate": recovered_n / stop_n if stop_n else None,
            "persistent_stop_first_rate": persistent_n / eval_n if eval_n else None,
            "persistent_share_of_stop_first": persistent_n / stop_n if stop_n else None,
            "baseline_fast_winner_rate": self.baseline_fast_rate,
            "baseline_stop_first_rate": self.baseline_stop_rate,
            "fast_winner_lift": fast_lift,
            "stop_first_reduction": stop_reduction,
            "path_edge": fast_lift + stop_reduction if fast_lift is not None and stop_reduction is not None else None,
            "excess_w3_p50": self._q(self.excess_w3, selected_eval, 0.50),
            "mae_3w_p50": mae,
            "mfe_3w_p50": mfe,
            "mfe_mae_ratio_3w": rr,
            "evaluated_quarters": int(len(edge_array)),
            "positive_path_edge_quarter_fraction": float(np.mean(edge_array > 0)) if edge_array.size else None,
            "median_quarter_path_edge": float(np.median(edge_array)) if edge_array.size else None,
            "worst_quarter_path_edge": float(np.min(edge_array)) if edge_array.size else None,
            "higher_stop_risk_quarter_fraction": float(np.mean(stop_array > 0)) if stop_array.size else None,
            "median_quarter_stop_first_lift": float(np.median(stop_array)) if stop_array.size else None,
        }


def _score(records: list[dict[str, Any]]) -> pd.DataFrame:
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
    total = pd.Series(0.0, index=scored.index, dtype=float)
    for metric, weight in weights.items():
        numeric = pd.to_numeric(scored[metric], errors="coerce")
        total += weight * numeric.rank(pct=True, method="average", na_option="bottom").fillna(0.0)
    total -= 0.01 * (pd.to_numeric(scored["condition_count"], errors="coerce") - 1).clip(lower=0)
    scored["quality_score"] = total
    return scored.sort_values(
        ["quality_score", "path_edge", "excess_w3_p50", "evaluable_n"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)


def search_stock_interactions_fast(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    min_selected: int,
    min_evaluable: int,
    min_quarter_n: int,
    min_evaluated_quarters: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Exact singles + exact distinct-feature pair search with compact metrics."""
    if any(str(feature).startswith("M_") for feature in features):
        raise ValueError("M_* market features are forbidden in R4 stock-condition search")
    context = PathSearchContext.from_frame(frame)
    conditions = generate_stock_conditions(frame, features)
    masks = {condition.key(): condition_mask(frame, condition) for condition in conditions}
    records: list[dict[str, Any]] = []
    supported_single = 0
    pair_tested = 0
    supported_pair = 0

    def eligible(metrics: Mapping[str, Any]) -> bool:
        return (
            int(metrics["selected_n"]) >= min_selected
            and int(metrics["evaluable_n"]) >= min_evaluable
            and int(metrics["evaluated_quarters"]) >= min_evaluated_quarters
        )

    for condition in conditions:
        metrics = context.evaluate(masks[condition.key()], min_quarter_n=min_quarter_n)
        if eligible(metrics):
            records.append({"rule_json": _rule_json([condition]), "condition_count": 1, **metrics})
            supported_single += 1

    for i, left in enumerate(conditions):
        for right in conditions[i + 1 :]:
            if left.feature == right.feature:
                continue
            pair_tested += 1
            mask = masks[left.key()] & masks[right.key()]
            if int(np.count_nonzero(mask)) < min_selected:
                continue
            metrics = context.evaluate(mask, min_quarter_n=min_quarter_n)
            if eligible(metrics):
                records.append({"rule_json": _rule_json([left, right]), "condition_count": 2, **metrics})
                supported_pair += 1
    scored = _score(records)
    return scored, {
        "generated_stock_condition_count": len(conditions),
        "supported_single_count": supported_single,
        "distinct_feature_pair_count_tested": pair_tested,
        "supported_pair_count": supported_pair,
        "candidate_rule_count": int(len(scored)),
        "candidate_metric_mode": "compact_vectorized_then_top_rule_full_enrichment",
    }


def enrich_full_metrics(frame: pd.DataFrame, scored: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    """Add full W1-W4 path metrics only for the reported top slice."""
    if scored.empty:
        return scored.copy()
    out = scored.head(top_n).copy().reset_index(drop=True)
    for index, item in out.iterrows():
        metrics = summarize_path(frame, rule_mask(frame, str(item["rule_json"])))
        for key, value in metrics.items():
            out.at[index, key] = value
    return out
