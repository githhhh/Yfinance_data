from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .labels import HORIZONS


KEYS = ["snapshot_date", "code"]


def evaluate_selection(
    panel: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    variant: str,
) -> dict[str, Any]:
    chosen = _selected_panel(panel, selected)
    weeks = sorted(panel["snapshot_date"].astype(str).unique())
    counts = (
        selected.groupby("snapshot_date").size()
        if not selected.empty
        else pd.Series(dtype=float)
    ).reindex(weeks, fill_value=0)

    row: dict[str, Any] = {
        "variant": variant,
        "weeks": len(weeks),
        "picks": int(len(selected)),
        "avg_watchlist_size": float(counts.mean()) if len(counts) else np.nan,
        "median_watchlist_size": float(counts.median()) if len(counts) else np.nan,
        "p95_watchlist_size": float(counts.quantile(0.95)) if len(counts) else np.nan,
    }

    complete_1w = panel[panel["forward_1w_censored"].eq(False)]
    selected_1w = chosen[chosen["forward_1w_censored"].eq(False)]
    opportunities = complete_1w[complete_1w["review_opportunity_1w"].eq(True)]
    selected_opportunities = selected_1w[selected_1w["review_opportunity_1w"].eq(True)]
    non_actionable_opportunities = opportunities[
        opportunities["ibd_entry_status"].fillna("").astype(str).str.upper().ne("ACTIONABLE")
    ]
    selected_non_actionable_opportunities = selected_opportunities[
        selected_opportunities["ibd_entry_status"].fillna("").astype(str).str.upper().ne("ACTIONABLE")
    ]
    row.update(
        {
            "opportunity_recall_1w": _safe_ratio(
                len(selected_opportunities), len(opportunities)
            ),
            "non_actionable_opportunity_recall_1w": _safe_ratio(
                len(selected_non_actionable_opportunities),
                len(non_actionable_opportunities),
            ),
            "opportunities_per_review": _safe_ratio(
                len(selected_opportunities), len(selected_1w)
            ),
        }
    )

    for horizon in HORIZONS:
        complete = panel[panel[f"forward_{horizon}_censored"].eq(False)]
        chosen_complete = chosen[chosen[f"forward_{horizon}_censored"].eq(False)]

        winner_return = complete[complete[f"winner_return_top5_{horizon}"].eq(True)]
        winner_mfe = complete[complete[f"winner_mfe_top5_{horizon}"].eq(True)]
        winner_any = complete[complete[f"big_winner_any_{horizon}"].eq(True)]
        loser_any = complete[complete[f"big_loser_any_{horizon}"].eq(True)]
        return_top10 = complete[complete[f"winner_return_top10pct_{horizon}"].eq(True)]
        return_bottom10 = complete[complete[f"loser_return_bottom10pct_{horizon}"].eq(True)]

        selected_winner_return = chosen_complete[
            chosen_complete[f"winner_return_top5_{horizon}"].eq(True)
        ]
        selected_winner_mfe = chosen_complete[
            chosen_complete[f"winner_mfe_top5_{horizon}"].eq(True)
        ]
        selected_winner_any = chosen_complete[
            chosen_complete[f"big_winner_any_{horizon}"].eq(True)
        ]
        selected_loser_any = chosen_complete[
            chosen_complete[f"big_loser_any_{horizon}"].eq(True)
        ]
        selected_severe = chosen_complete[
            chosen_complete[f"severe_loser_{horizon}"].eq(True)
        ]
        selected_return_top10 = chosen_complete[
            chosen_complete[f"winner_return_top10pct_{horizon}"].eq(True)
        ]
        selected_return_bottom10 = chosen_complete[
            chosen_complete[f"loser_return_bottom10pct_{horizon}"].eq(True)
        ]

        loser_inclusion = _safe_ratio(len(selected_loser_any), len(loser_any))
        row.update(
            {
                f"winner_return_recall_{horizon}": _safe_ratio(
                    len(selected_winner_return), len(winner_return)
                ),
                f"winner_mfe_recall_{horizon}": _safe_ratio(
                    len(selected_winner_mfe), len(winner_mfe)
                ),
                f"big_winner_recall_{horizon}": _safe_ratio(
                    len(selected_winner_any), len(winner_any)
                ),
                f"big_loser_inclusion_{horizon}": loser_inclusion,
                f"big_loser_exclusion_{horizon}": (
                    1.0 - loser_inclusion if np.isfinite(loser_inclusion) else np.nan
                ),
                f"big_loser_density_{horizon}": _safe_ratio(
                    len(selected_loser_any), len(chosen_complete)
                ),
                f"severe_loser_exposure_{horizon}": _safe_ratio(
                    len(selected_severe), len(chosen_complete)
                ),
                f"winner_return_top10pct_recall_{horizon}": _safe_ratio(
                    len(selected_return_top10), len(return_top10)
                ),
                f"loser_return_bottom10pct_inclusion_{horizon}": _safe_ratio(
                    len(selected_return_bottom10), len(return_bottom10)
                ),
                f"median_return_{horizon}_pct": _median(
                    chosen_complete, f"forward_{horizon}_return_pct"
                ),
                f"median_mfe_{horizon}_pct": _median(
                    chosen_complete, f"mfe_{horizon}_pct"
                ),
                f"median_mae_{horizon}_pct": _median(
                    chosen_complete, f"mae_{horizon}_pct"
                ),
            }
        )

    row["big_winner_recall_mean_2_4w"] = _mean_metrics(
        row, [f"big_winner_recall_{h}" for h in ("2w", "3w", "4w")]
    )
    row["big_loser_exclusion_mean_2_4w"] = _mean_metrics(
        row, [f"big_loser_exclusion_{h}" for h in ("2w", "3w", "4w")]
    )
    row["big_loser_density_mean_2_4w"] = _mean_metrics(
        row, [f"big_loser_density_{h}" for h in ("2w", "3w", "4w")]
    )
    row["severe_loser_exposure_mean_2_4w"] = _mean_metrics(
        row, [f"severe_loser_exposure_{h}" for h in ("2w", "3w", "4w")]
    )
    return row


def compare_metrics(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    out = dict(candidate)
    for metric in (
        "opportunity_recall_1w",
        "non_actionable_opportunity_recall_1w",
        "big_winner_recall_mean_2_4w",
        "big_loser_exclusion_mean_2_4w",
        "big_loser_density_mean_2_4w",
        "severe_loser_exposure_mean_2_4w",
        "avg_watchlist_size",
    ):
        out[f"{metric}_delta_vs_b0"] = _delta(
            candidate.get(metric), baseline.get(metric)
        )
    return out


def missed_big_winners(panel: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    selected_keys = _key_set(selected)
    rows: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        winners = panel[
            panel[f"forward_{horizon}_censored"].eq(False)
            & panel[f"big_winner_any_{horizon}"].eq(True)
        ].copy()
        missed = winners[
            ~winners.apply(
                lambda row: (str(row["snapshot_date"]), str(row["code"])) in selected_keys,
                axis=1,
            )
        ].copy()
        if not missed.empty:
            missed.insert(0, "horizon", horizon)
            rows.append(_case_projection(missed))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def included_big_losers(panel: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    selected_keys = _key_set(selected)
    rows: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        losers = panel[
            panel[f"forward_{horizon}_censored"].eq(False)
            & panel[f"big_loser_any_{horizon}"].eq(True)
        ].copy()
        included = losers[
            losers.apply(
                lambda row: (str(row["snapshot_date"]), str(row["code"])) in selected_keys,
                axis=1,
            )
        ].copy()
        if not included.empty:
            included.insert(0, "horizon", horizon)
            rows.append(_case_projection(included))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _case_projection(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "horizon",
        "snapshot_date",
        "code",
        "ibd_entry_status",
        "ibd_candidate_rule",
        "current_vs_ibd_candidate_pct",
        "review_opportunity_1w",
        "forward_1w_return_pct",
        "forward_2w_return_pct",
        "forward_3w_return_pct",
        "forward_4w_return_pct",
        "mfe_1w_pct",
        "mfe_2w_pct",
        "mfe_3w_pct",
        "mfe_4w_pct",
        "mae_1w_pct",
        "mae_2w_pct",
        "mae_3w_pct",
        "mae_4w_pct",
    ]
    return frame[[column for column in columns if column in frame.columns]].copy()


def _selected_panel(panel: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if panel.empty or selected.empty:
        return panel.iloc[0:0].copy()
    keys = selected[KEYS].drop_duplicates().copy()
    keys["snapshot_date"] = keys["snapshot_date"].astype(str)
    keys["code"] = keys["code"].astype(str)
    work = panel.copy()
    work["snapshot_date"] = work["snapshot_date"].astype(str)
    work["code"] = work["code"].astype(str)
    return work.merge(keys, on=KEYS, how="inner").copy()


def _key_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    if frame.empty:
        return set()
    return set(zip(frame["snapshot_date"].astype(str), frame["code"].astype(str)))


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator in {0, 0.0}:
        return np.nan
    return float(numerator) / float(denominator)


def _median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if len(values) else np.nan


def _mean_metrics(row: dict[str, Any], columns: list[str]) -> float:
    values = [
        float(row[column])
        for column in columns
        if column in row and _finite(row[column])
    ]
    return float(np.mean(values)) if values else np.nan


def _delta(value: Any, baseline: Any) -> float:
    if not _finite(value) or not _finite(baseline):
        return np.nan
    return float(value) - float(baseline)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
