"""Random weekly Review Universe Top3 control baseline and distribution benchmarking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_random_top3_for_snapshot(
    snapshot_date: str,
    snapshot_events_df: pd.DataFrame,
    weekly_outcomes_df: pd.DataFrame,
    b0_snapshot_events: pd.DataFrame | None = None,
    n_draws: int = 1000,
    seed: int = 42,
    pick_limit: int = 3,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run 1,000 random Top3 draws for a single snapshot week and compute percentile distribution.
    
    Returns:
        (snapshot_summary_dict, detailed_draws_df)
    """
    rng = np.random.default_rng(seed)
    n_candidates = len(snapshot_events_df)
    actual_picks_count = min(pick_limit, n_candidates)

    if n_candidates == 0:
        return {"snapshot_date": snapshot_date, "status": "EMPTY_POOL"}, pd.DataFrame()

    candidate_codes = snapshot_events_df["code"].tolist()
    event_lookup = snapshot_events_df.set_index("code").to_dict(orient="index")

    # Map candidate weekly outcomes for fast lookup
    # key: (code, holding_week_index) -> dict
    weekly_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    if not weekly_outcomes_df.empty:
        snap_w = weekly_outcomes_df[weekly_outcomes_df["snapshot_date"] == snapshot_date]
        for _, w_row in snap_w.iterrows():
            weekly_lookup[(str(w_row["code"]), int(w_row["holding_week_index"]))] = w_row.to_dict()

    draw_records: list[dict[str, Any]] = []

    for draw_idx in range(n_draws):
        sampled_codes = rng.choice(candidate_codes, size=actual_picks_count, replace=False).tolist()

        # Collect candidate outcome records
        sampled_events = [event_lookup[c] for c in sampled_codes]

        # Valid prices check
        valid_events = [
            ev for ev in sampled_events
            if ev.get("entry_status") == "ENTRY_OK"
            and ev.get("entry_open") is not None
            and not pd.isna(ev.get("entry_open"))
        ]
        is_valid_draw = (len(valid_events) == actual_picks_count)

        # Compute As-Of Metrics
        current_rets = [float(ev["current_return_to_asof_pct"]) for ev in valid_events if ev.get("current_return_to_asof_pct") is not None]
        exec_rets = [float(ev["executed_return_to_asof_pct"]) for ev in valid_events if ev.get("executed_return_to_asof_pct") is not None]
        max_gains = [float(ev["max_gain_to_asof_pct"]) for ev in valid_events if ev.get("max_gain_to_asof_pct") is not None]
        stops_hit = sum(1 for ev in valid_events if ev.get("stop_8_hit_ever") is True)
        profits20_hit = sum(1 for ev in valid_events if ev.get("profit20_hit") is True)

        # Weekly Horizon Metrics for Week 1, 2, 3, 4
        w_rets: dict[int, list[float]] = {1: [], 2: [], 3: [], 4: []}
        w_max_gains: dict[int, list[float]] = {1: [], 2: [], 3: [], 4: []}
        w_stops: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}

        for c in sampled_codes:
            for w_idx in [1, 2, 3, 4]:
                w_info = weekly_lookup.get((c, w_idx))
                if w_info:
                    ret = w_info.get("week_close_return_from_entry_pct")
                    if ret is not None and not pd.isna(ret):
                        w_rets[w_idx].append(float(ret))
                    mg = w_info.get("week_max_gain_from_entry_pct")
                    if mg is not None and not pd.isna(mg):
                        w_max_gains[w_idx].append(float(mg))
                    if w_info.get("stop_8_hit_by_week_end") is True:
                        w_stops[w_idx] += 1

        draw_dict = {
            "snapshot_date": snapshot_date,
            "draw_index": draw_idx,
            "is_valid_draw": is_valid_draw,
            "valid_candidates_count": len(valid_events),
            "sampled_codes": ",".join(sampled_codes),
            "asof_mean_return_pct": np.mean(current_rets) if current_rets else np.nan,
            "asof_mean_exec_return_pct": np.mean(exec_rets) if exec_rets else np.nan,
            "asof_mean_max_gain_pct": np.mean(max_gains) if max_gains else np.nan,
            "asof_worst_return_pct": np.min(current_rets) if current_rets else np.nan,
            "asof_stop8_count": stops_hit,
            "asof_has_profit20": bool(profits20_hit > 0),
            "asof_all_stopped": bool(stops_hit == len(valid_events) and len(valid_events) > 0),
        }

        for w_idx in [1, 2, 3, 4]:
            r_list = w_rets[w_idx]
            mg_list = w_max_gains[w_idx]
            draw_dict[f"w{w_idx}_mean_return_pct"] = np.mean(r_list) if r_list else np.nan
            draw_dict[f"w{w_idx}_mean_max_gain_pct"] = np.mean(mg_list) if mg_list else np.nan
            draw_dict[f"w{w_idx}_worst_return_pct"] = np.min(r_list) if r_list else np.nan
            draw_dict[f"w{w_idx}_stop8_count"] = w_stops[w_idx]

        draw_records.append(draw_dict)

    draws_df = pd.DataFrame(draw_records)

    # Compute Quantiles across the 1000 draws
    summary: dict[str, Any] = {
        "snapshot_date": snapshot_date,
        "total_candidates": n_candidates,
        "n_draws": n_draws,
        "seed": seed,
        "valid_draw_pct": round(float((draws_df["is_valid_draw"].sum() / n_draws) * 100.0), 2),
    }

    metrics_to_quantiles = [
        "w1_mean_return_pct",
        "w1_mean_max_gain_pct",
        "w2_mean_return_pct",
        "w3_mean_return_pct",
        "w4_mean_return_pct",
        "asof_mean_return_pct",
        "asof_mean_exec_return_pct",
        "asof_mean_max_gain_pct",
        "asof_worst_return_pct",
        "asof_stop8_count",
    ]

    for metric in metrics_to_quantiles:
        s = draws_df[metric].dropna()
        if not s.empty:
            summary[f"{metric}_p05"] = round(float(np.percentile(s, 5)), 4)
            summary[f"{metric}_p25"] = round(float(np.percentile(s, 25)), 4)
            summary[f"{metric}_p50"] = round(float(np.percentile(s, 50)), 4)
            summary[f"{metric}_p75"] = round(float(np.percentile(s, 75)), 4)
            summary[f"{metric}_p95"] = round(float(np.percentile(s, 95)), 4)
        else:
            for q in ["p05", "p25", "p50", "p75", "p95"]:
                summary[f"{metric}_{q}"] = np.nan

    # Calculate B0 Actual Performance & Percentile for this snapshot
    if b0_snapshot_events is not None and not b0_snapshot_events.empty:
        b0_valid = b0_snapshot_events[b0_snapshot_events["entry_open"].notna()]
        b0_ret = b0_valid["current_return_to_asof_pct"].mean()
        b0_exec_ret = b0_valid["executed_return_to_asof_pct"].mean()
        b0_mg = b0_valid["max_gain_to_asof_pct"].mean()
        b0_w1_ret = b0_valid["week1_close_return_pct"].mean()
        b0_stops = (b0_valid["stop_8_hit_ever"] == True).sum()

        summary["b0_actual_asof_mean_return_pct"] = round(float(b0_ret), 4) if pd.notna(b0_ret) else np.nan
        summary["b0_actual_asof_mean_exec_return_pct"] = round(float(b0_exec_ret), 4) if pd.notna(b0_exec_ret) else np.nan
        summary["b0_actual_asof_mean_max_gain_pct"] = round(float(b0_mg), 4) if pd.notna(b0_mg) else np.nan
        summary["b0_actual_w1_mean_return_pct"] = round(float(b0_w1_ret), 4) if pd.notna(b0_w1_ret) else np.nan
        summary["b0_actual_stop8_count"] = int(b0_stops)

        # Percentile Rank
        if pd.notna(b0_ret) and not draws_df["asof_mean_return_pct"].dropna().empty:
            dist = draws_df["asof_mean_return_pct"].dropna().values
            summary["b0_asof_return_percentile"] = round(float((np.sum(dist <= b0_ret) / len(dist)) * 100.0), 2)
        else:
            summary["b0_asof_return_percentile"] = np.nan

        if pd.notna(b0_w1_ret) and not draws_df["w1_mean_return_pct"].dropna().empty:
            dist_w1 = draws_df["w1_mean_return_pct"].dropna().values
            summary["b0_w1_return_percentile"] = round(float((np.sum(dist_w1 <= b0_w1_ret) / len(dist_w1)) * 100.0), 2)
        else:
            summary["b0_w1_return_percentile"] = np.nan

    return summary, draws_df


def run_random_top3_benchmark(
    event_outcomes_df: pd.DataFrame,
    weekly_outcomes_df: pd.DataFrame,
    b0_events_df: pd.DataFrame | None = None,
    n_draws_per_week: int = 1000,
    seed: int = 42,
    output_distribution_csv: Path | str | None = "backtest/b0_top3_quality_audit/output/random_signal_top3_distribution.csv",
) -> pd.DataFrame:
    """Run weekly 1,000-draw random Top3 benchmark across all snapshot weeks."""
    snapshots = sorted(event_outcomes_df["snapshot_date"].unique().tolist())
    summary_rows: list[dict[str, Any]] = []

    logger.info(f"Running random Top3 benchmark for {len(snapshots)} snapshots ({n_draws_per_week} draws/week, seed={seed})...")

    for snap in snapshots:
        snap_events = event_outcomes_df[event_outcomes_df["snapshot_date"] == snap]
        snap_b0 = None
        if b0_events_df is not None and not b0_events_df.empty:
            snap_b0 = b0_events_df[b0_events_df["snapshot_date"] == snap]

        # Use deterministic week-specific seed derived from base seed
        week_seed = seed + int(pd.Timestamp(snap).strftime("%y%m%d")) % 100000

        snap_sum, _ = run_random_top3_for_snapshot(
            snapshot_date=snap,
            snapshot_events_df=snap_events,
            weekly_outcomes_df=weekly_outcomes_df,
            b0_snapshot_events=snap_b0,
            n_draws=n_draws_per_week,
            seed=week_seed,
            pick_limit=3,
        )
        summary_rows.append(snap_sum)

    dist_df = pd.DataFrame(summary_rows)

    if output_distribution_csv is not None:
        out_p = Path(output_distribution_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        dist_df.to_csv(out_p, index=False, encoding="utf-8-sig")
        logger.info(f"Saved random Top3 benchmark distribution to {out_p}")

    return dist_df
