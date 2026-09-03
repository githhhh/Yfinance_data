from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import compare_metrics, evaluate_selection
from .utils import ZERO_TOL


KEYS = ["snapshot_date", "code"]


def setup_balanced_sensitivity(
    panel: pd.DataFrame,
    baseline_selected: pd.DataFrame,
    candidate_selected: pd.DataFrame,
    *,
    variant: str,
    min_complete_1w_rows: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal-weight setup sensitivity to reduce setup coverage dominance.

    Winner/loser Oracle flags remain defined in the full weekly universe. The
    sensitivity only changes how setup strata contribute to the summary.
    """
    if panel.empty or "ibd_candidate_rule" not in panel.columns:
        return pd.DataFrame(), pd.DataFrame()

    work = panel.copy()
    work["_setup_stratum"] = (
        work["ibd_candidate_rule"].fillna("<MISSING>").astype(str)
    )
    rows = []
    for setup, group in work.groupby("_setup_stratum", sort=True):
        complete_1w = int(group["forward_1w_censored"].eq(False).sum())
        total = len(group)
        base = _filter_selected(baseline_selected, group)
        candidate = _filter_selected(candidate_selected, group)
        base_metrics = evaluate_selection(
            group, base, variant="B0_ACTIONABLE_ONLY"
        )
        candidate_metrics = compare_metrics(
            evaluate_selection(group, candidate, variant=variant),
            base_metrics,
        )
        candidate_metrics.update(
            {
                "setup": setup,
                "rows": total,
                "complete_1w_rows": complete_1w,
                "complete_1w_rate": (
                    complete_1w / total if total else np.nan
                ),
                "eligible_for_balanced_summary": (
                    complete_1w >= min_complete_1w_rows
                ),
            }
        )
        rows.append(candidate_metrics)

    detail = pd.DataFrame(rows)
    eligible = detail[
        detail["eligible_for_balanced_summary"].eq(True)
    ].copy()
    metrics = [
        "opportunity_recall_1w_delta_vs_b0",
        "tradable_winner_capture_lift_mean_2_4w_delta_vs_b0",
        "tradable_loser_capture_lift_mean_2_4w_delta_vs_b0",
        "opp_severe_loser_exposure_mean_2_4w_delta_vs_b0",
        "avg_watchlist_size_delta_vs_b0",
        "incremental_opportunities_per_added_review",
    ]
    summary: dict[str, object] = {
        "variant": variant,
        "eligible_setups": int(len(eligible)),
        "min_complete_1w_rows": min_complete_1w_rows,
    }
    for metric in metrics:
        values = pd.to_numeric(eligible[metric], errors="coerce").dropna()
        summary[f"setup_balanced_mean_{metric}"] = (
            float(values.mean()) if len(values) else np.nan
        )
        summary[f"setup_balanced_median_{metric}"] = (
            float(values.median()) if len(values) else np.nan
        )
        if metric.endswith("_delta_vs_b0"):
            if "loser_capture_lift" in metric or "severe_loser" in metric:
                summary[f"setup_nonworse_rate_{metric}"] = (
                    float((values <= ZERO_TOL).mean())
                    if len(values)
                    else np.nan
                )
            else:
                summary[f"setup_nonnegative_rate_{metric}"] = (
                    float((values >= -ZERO_TOL).mean())
                    if len(values)
                    else np.nan
                )
    return detail, pd.DataFrame([summary])


def _filter_selected(
    selected: pd.DataFrame,
    panel_slice: pd.DataFrame,
) -> pd.DataFrame:
    if selected.empty or panel_slice.empty:
        return selected.iloc[0:0].copy()
    keys = panel_slice[KEYS].copy()
    keys["snapshot_date"] = keys["snapshot_date"].astype(str)
    keys["code"] = keys["code"].astype(str)
    work = selected.copy()
    work["snapshot_date"] = work["snapshot_date"].astype(str)
    work["code"] = work["code"].astype(str)
    return work.merge(keys.drop_duplicates(), on=KEYS, how="inner")
