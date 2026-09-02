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
    """Evaluate review-list quality with raw recall and capacity-normalized lift."""
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
    selected_opportunities = selected_1w[
        selected_1w["review_opportunity_1w"].eq(True)
    ]
    non_actionable_opportunities = opportunities[
        opportunities["ibd_entry_status"]
        .fillna("")
        .astype(str)
        .str.upper()
        .ne("ACTIONABLE")
    ]
    selected_non_actionable_opportunities = selected_opportunities[
        selected_opportunities["ibd_entry_status"]
        .fillna("")
        .astype(str)
        .str.upper()
        .ne("ACTIONABLE")
    ]

    row.update(
        {
            "evaluable_picks_1w": int(len(selected_1w)),
            "opportunities_available_1w": int(len(opportunities)),
            "opportunities_captured_1w": int(len(selected_opportunities)),
            "selection_coverage_1w": _safe_ratio(
                len(selected_1w), len(complete_1w)
            ),
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
        _add_snapshot_clock_metrics(row, panel, chosen, horizon)
        _add_opportunity_clock_metrics(row, panel, chosen, horizon)

    row["big_winner_recall_mean_2_4w"] = _mean_metrics(
        row, [f"big_winner_recall_{h}" for h in ("2w", "3w", "4w")]
    )
    row["snapshot_winner_capture_lift_mean_2_4w"] = _mean_metrics(
        row, [f"winner_capture_lift_{h}" for h in ("2w", "3w", "4w")]
    )
    row["snapshot_loser_capture_lift_mean_2_4w"] = _mean_metrics(
        row, [f"loser_capture_lift_{h}" for h in ("2w", "3w", "4w")]
    )

    row["tradable_big_winner_recall_mean_2_4w"] = _mean_metrics(
        row, [f"tradable_big_winner_recall_{h}" for h in ("2w", "3w", "4w")]
    )
    row["tradable_winner_capture_lift_mean_2_4w"] = _mean_metrics(
        row, [f"tradable_winner_capture_lift_{h}" for h in ("2w", "3w", "4w")]
    )
    row["tradable_loser_capture_lift_mean_2_4w"] = _mean_metrics(
        row, [f"tradable_loser_capture_lift_{h}" for h in ("2w", "3w", "4w")]
    )
    row["opp_severe_loser_exposure_mean_2_4w"] = _mean_metrics(
        row, [f"opp_severe_loser_exposure_{h}" for h in ("2w", "3w", "4w")]
    )
    return row


def compare_metrics(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    out = dict(candidate)
    delta_metrics = (
        "opportunity_recall_1w",
        "non_actionable_opportunity_recall_1w",
        "selection_coverage_1w",
        "big_winner_recall_mean_2_4w",
        "snapshot_winner_capture_lift_mean_2_4w",
        "snapshot_loser_capture_lift_mean_2_4w",
        "tradable_big_winner_recall_mean_2_4w",
        "tradable_winner_capture_lift_mean_2_4w",
        "tradable_loser_capture_lift_mean_2_4w",
        "opp_severe_loser_exposure_mean_2_4w",
        "avg_watchlist_size",
    )
    for metric in delta_metrics:
        out[f"{metric}_delta_vs_b0"] = _delta(
            candidate.get(metric), baseline.get(metric)
        )

    added_reviews = _delta(
        candidate.get("evaluable_picks_1w"),
        baseline.get("evaluable_picks_1w"),
    )
    added_opportunities = _delta(
        candidate.get("opportunities_captured_1w"),
        baseline.get("opportunities_captured_1w"),
    )
    out["added_evaluable_reviews_vs_b0"] = added_reviews
    out["incremental_opportunities_vs_b0"] = added_opportunities
    out["incremental_opportunities_per_added_review"] = (
        added_opportunities / added_reviews
        if _finite(added_reviews) and added_reviews > 0
        else np.nan
    )
    base_size = baseline.get("avg_watchlist_size")
    out["attention_multiplier_vs_b0"] = (
        float(candidate["avg_watchlist_size"]) / float(base_size)
        if _finite(candidate.get("avg_watchlist_size"))
        and _finite(base_size)
        and float(base_size) > 0
        else np.nan
    )
    return out


def weekly_macro_table(
    panel: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    variant: str,
) -> pd.DataFrame:
    rows = []
    for snapshot, week in panel.groupby("snapshot_date", sort=True):
        picked = selected[
            selected["snapshot_date"].astype(str).eq(str(snapshot))
        ].copy()
        row = evaluate_selection(week, picked, variant=variant)
        row["snapshot_date"] = str(snapshot)
        rows.append(row)
    return pd.DataFrame(rows)


def macro_average_summary(weekly: pd.DataFrame) -> pd.DataFrame:
    if weekly.empty:
        return pd.DataFrame()
    metrics = [
        "opportunity_recall_1w",
        "selection_coverage_1w",
        "opportunities_per_review",
        "tradable_big_winner_recall_mean_2_4w",
        "tradable_winner_capture_lift_mean_2_4w",
        "tradable_loser_capture_lift_mean_2_4w",
        "opp_severe_loser_exposure_mean_2_4w",
        "avg_watchlist_size",
    ]
    rows = []
    for variant, group in weekly.groupby("variant", sort=True):
        row = {"variant": variant, "weeks": int(group["snapshot_date"].nunique())}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"macro_mean_{metric}"] = (
                float(values.mean()) if len(values) else np.nan
            )
            row[f"macro_median_{metric}"] = (
                float(values.median()) if len(values) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def moving_block_bootstrap_delta(
    baseline_weekly: pd.DataFrame,
    candidate_weekly: pd.DataFrame,
    *,
    metrics: list[str],
    block_size: int = 4,
    draws: int = 2000,
    seed: int = 20260902,
) -> pd.DataFrame:
    """Paired moving-block bootstrap over chronological weeks."""
    base = baseline_weekly.set_index("snapshot_date")
    cand = candidate_weekly.set_index("snapshot_date")
    common = sorted(set(base.index) & set(cand.index))
    if not common:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows = []
    n = len(common)
    block = max(1, min(block_size, n))
    starts = np.arange(0, max(1, n - block + 1))

    for metric in metrics:
        base_values = pd.to_numeric(base.loc[common, metric], errors="coerce").to_numpy()
        cand_values = pd.to_numeric(cand.loc[common, metric], errors="coerce").to_numpy()
        delta = cand_values - base_values
        observed = delta[np.isfinite(delta)]
        if not len(observed):
            continue

        samples = []
        for _ in range(draws):
            idx: list[int] = []
            while len(idx) < n:
                start = int(rng.choice(starts))
                idx.extend(range(start, min(start + block, n)))
            sampled = delta[np.array(idx[:n], dtype=int)]
            sampled = sampled[np.isfinite(sampled)]
            if len(sampled):
                samples.append(float(sampled.mean()))
        if not samples:
            continue
        arr = np.array(samples, dtype=float)
        rows.append(
            {
                "metric": metric,
                "weeks": n,
                "block_size": block,
                "observed_macro_delta": float(observed.mean()),
                "bootstrap_ci_2_5": float(np.quantile(arr, 0.025)),
                "bootstrap_ci_97_5": float(np.quantile(arr, 0.975)),
                "prob_delta_gt_0": float((arr > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def missed_big_winners(panel: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    return _case_audit(panel, selected, winner=True)


def included_big_losers(panel: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    return _case_audit(panel, selected, winner=False)


def _add_snapshot_clock_metrics(
    row: dict[str, Any],
    panel: pd.DataFrame,
    chosen: pd.DataFrame,
    horizon: str,
) -> None:
    complete = panel[panel[f"forward_{horizon}_censored"].eq(False)]
    chosen_complete = chosen[chosen[f"forward_{horizon}_censored"].eq(False)]
    coverage = _safe_ratio(len(chosen_complete), len(complete))

    winners = complete[complete[f"big_winner_any_{horizon}"].eq(True)]
    losers = complete[complete[f"big_loser_any_{horizon}"].eq(True)]
    selected_winners = chosen_complete[
        chosen_complete[f"big_winner_any_{horizon}"].eq(True)
    ]
    selected_losers = chosen_complete[
        chosen_complete[f"big_loser_any_{horizon}"].eq(True)
    ]

    winner_recall = _safe_ratio(len(selected_winners), len(winners))
    loser_inclusion = _safe_ratio(len(selected_losers), len(losers))
    row.update(
        {
            f"selection_coverage_{horizon}": coverage,
            f"big_winner_recall_{horizon}": winner_recall,
            f"big_loser_inclusion_{horizon}": loser_inclusion,
            f"winner_capture_lift_{horizon}": _lift(winner_recall, coverage),
            f"loser_capture_lift_{horizon}": _lift(loser_inclusion, coverage),
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


def _add_opportunity_clock_metrics(
    row: dict[str, Any],
    panel: pd.DataFrame,
    chosen: pd.DataFrame,
    horizon: str,
) -> None:
    eligible = panel[
        panel["review_opportunity_1w"].eq(True)
        & panel[f"opp_forward_{horizon}_censored"].eq(False)
    ]
    chosen_eligible = chosen[
        chosen["review_opportunity_1w"].eq(True)
        & chosen[f"opp_forward_{horizon}_censored"].eq(False)
    ]
    coverage = _safe_ratio(len(chosen_eligible), len(eligible))

    winners = eligible[eligible[f"opp_big_winner_any_{horizon}"].eq(True)]
    losers = eligible[eligible[f"opp_big_loser_any_{horizon}"].eq(True)]
    selected_winners = chosen_eligible[
        chosen_eligible[f"opp_big_winner_any_{horizon}"].eq(True)
    ]
    selected_losers = chosen_eligible[
        chosen_eligible[f"opp_big_loser_any_{horizon}"].eq(True)
    ]
    severe = chosen_eligible[
        chosen_eligible[f"opp_severe_loser_{horizon}"].eq(True)
    ]

    winner_recall = _safe_ratio(len(selected_winners), len(winners))
    loser_inclusion = _safe_ratio(len(selected_losers), len(losers))
    row.update(
        {
            f"tradable_selection_coverage_{horizon}": coverage,
            f"tradable_big_winner_recall_{horizon}": winner_recall,
            f"tradable_big_loser_inclusion_{horizon}": loser_inclusion,
            f"tradable_winner_capture_lift_{horizon}": _lift(
                winner_recall, coverage
            ),
            f"tradable_loser_capture_lift_{horizon}": _lift(
                loser_inclusion, coverage
            ),
            f"opp_severe_loser_exposure_{horizon}": _safe_ratio(
                len(severe), len(chosen_eligible)
            ),
            f"median_opp_return_{horizon}_pct": _median(
                chosen_eligible, f"opp_forward_{horizon}_return_pct"
            ),
            f"median_opp_mfe_{horizon}_pct": _median(
                chosen_eligible, f"opp_mfe_{horizon}_pct"
            ),
            f"median_opp_mae_{horizon}_pct": _median(
                chosen_eligible, f"opp_mae_{horizon}_pct"
            ),
        }
    )


def _case_audit(
    panel: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    winner: bool,
) -> pd.DataFrame:
    selected_keys = _key_set(selected)
    rows: list[pd.DataFrame] = []
    for clock, prefix in (("snapshot", ""), ("opportunity", "opp_")):
        for horizon in HORIZONS:
            flag = (
                f"{prefix}big_winner_any_{horizon}"
                if winner
                else f"{prefix}big_loser_any_{horizon}"
            )
            candidates = panel[panel[flag].eq(True)].copy()
            if winner:
                cases = candidates[
                    ~candidates.apply(
                        lambda r: (
                            str(r["snapshot_date"]),
                            str(r["code"]),
                        )
                        in selected_keys,
                        axis=1,
                    )
                ].copy()
            else:
                cases = candidates[
                    candidates.apply(
                        lambda r: (
                            str(r["snapshot_date"]),
                            str(r["code"]),
                        )
                        in selected_keys,
                        axis=1,
                    )
                ].copy()
            if cases.empty:
                continue
            cases.insert(0, "clock", clock)
            cases.insert(1, "horizon", horizon)
            rows.append(_case_projection(cases))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _case_projection(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "clock",
        "horizon",
        "snapshot_date",
        "code",
        "ibd_entry_status",
        "ibd_candidate_rule",
        "current_vs_ibd_candidate_pct",
        "review_opportunity_1w",
        "opportunity_type_1w",
        "opportunity_delay_sessions",
        "forward_1w_return_pct",
        "forward_2w_return_pct",
        "forward_3w_return_pct",
        "forward_4w_return_pct",
        "opp_forward_1w_return_pct",
        "opp_forward_2w_return_pct",
        "opp_forward_3w_return_pct",
        "opp_forward_4w_return_pct",
        "opp_mfe_4w_pct",
        "opp_mae_4w_pct",
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


def _lift(capture: Any, coverage: Any) -> float:
    if not _finite(capture) or not _finite(coverage) or float(coverage) <= 0:
        return np.nan
    return float(capture) / float(coverage)


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
