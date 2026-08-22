from __future__ import annotations

import pandas as pd


def rank_weighted_week_return(picks: pd.DataFrame) -> float:
    """Top-3 rank-sensitive return: rank 1 carries 3x, rank 2 carries 2x, rank 3 carries 1x."""
    if picks.empty or "latest_return_pct" not in picks.columns:
        return 0.0
    frame = picks.copy()
    frame["pick_order"] = pd.to_numeric(frame.get("pick_order"), errors="coerce")
    frame["latest_return_pct"] = pd.to_numeric(frame["latest_return_pct"], errors="coerce")
    frame = frame[frame["pick_order"].between(1, 3) & frame["latest_return_pct"].notna()]
    if frame.empty:
        return 0.0
    frame["weight"] = 4 - frame["pick_order"]
    return float((frame["latest_return_pct"] * frame["weight"]).sum() / frame["weight"].sum())


def robust_weekly_summary(weekly: pd.DataFrame, picks: pd.DataFrame) -> dict[str, float | int]:
    valid_weekly = weekly.copy()
    valid_weekly["avg_latest_return_pct"] = pd.to_numeric(valid_weekly.get("avg_latest_return_pct"), errors="coerce")
    valid_weekly["worst_latest_return_pct"] = pd.to_numeric(valid_weekly.get("worst_latest_return_pct"), errors="coerce")
    valid_weekly = valid_weekly[valid_weekly["avg_latest_return_pct"].notna()]
    pick_count = int(len(picks))
    return {
        "weeks": int(len(weekly)),
        "valid_weeks": int(len(valid_weekly)),
        "picks": pick_count,
        "median_week_avg_latest_return_pct": _float(valid_weekly["avg_latest_return_pct"].median()),
        "p25_week_avg_latest_return_pct": _float(valid_weekly["avg_latest_return_pct"].quantile(0.25)),
        "p75_week_avg_latest_return_pct": _float(valid_weekly["avg_latest_return_pct"].quantile(0.75)),
        "min_week_avg_latest_return_pct": _float(valid_weekly["avg_latest_return_pct"].min()),
        "max_week_avg_latest_return_pct": _float(valid_weekly["avg_latest_return_pct"].max()),
        "median_worst_pick_return_pct": _float(valid_weekly["worst_latest_return_pct"].median()),
        "min_worst_pick_return_pct": _float(valid_weekly["worst_latest_return_pct"].min()),
        "pick_top5_precision": _rate(picks.get("hit_latest_top5")),
        "pick_bottom5_precision_bad": _rate(picks.get("hit_loss_bottom5")),
        "pick_stop_rate": _rate(picks.get("hit_stop_8pct")),
    }


def coverage_rates(universe: pd.DataFrame, picks: pd.DataFrame) -> dict[str, float]:
    valid = universe[universe.get("valid_path", True).astype(bool)].copy()
    return {
        "top3_coverage": _coverage(picks.get("hit_latest_top3"), valid["latest_rank"].le(3)),
        "top5_coverage": _coverage(picks.get("hit_latest_top5"), valid["latest_rank"].le(5)),
        "gain5_coverage": _coverage(picks.get("hit_gain_top5"), valid["gain_rank"].le(5)),
        "bottom3_exposure_vs_slots": _coverage(picks.get("hit_loss_bottom3"), valid["loss_rank"].le(3)),
        "bottom5_exposure_vs_slots": _coverage(picks.get("hit_loss_bottom5"), valid["loss_rank"].le(5)),
        "stop_exposure_vs_all_stops": _coverage(picks.get("hit_stop_8pct"), valid["hit_stop_8pct"].astype(bool)),
        "pick_top5_precision": _rate(picks.get("hit_latest_top5")),
        "pick_bottom5_precision_bad": _rate(picks.get("hit_loss_bottom5")),
        "pick_stop_rate": _rate(picks.get("hit_stop_8pct")),
    }


def _coverage(pick_hits: pd.Series | None, denominator_mask: pd.Series) -> float:
    denominator = int(denominator_mask.sum())
    if denominator == 0 or pick_hits is None:
        return 0.0
    return float(pd.Series(pick_hits).fillna(False).astype(bool).sum() / denominator)


def _rate(series: pd.Series | None) -> float:
    if series is None or len(series) == 0:
        return 0.0
    return float(pd.Series(series).fillna(False).astype(bool).mean())


def _float(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)
