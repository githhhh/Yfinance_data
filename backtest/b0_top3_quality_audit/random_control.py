"""Random weekly Review Universe control baseline and distribution benchmarking."""

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
    benchmark_mode: str = "MATCHED_N",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run 1,000 random draws for a single snapshot week and compute percentile distribution.
    
    benchmark_mode:
        - "MATCHED_N": sample exact same count of stocks as B0 actual picks this week (Primary Selection Quality Benchmark).
        - "FIXED_TOP3": mechanically sample min(pick_limit, n_candidates) stocks (Fixed Exposure Benchmark).
    
    Returns:
        (snapshot_summary_dict, detailed_draws_df)
    """
    rng = np.random.default_rng(seed)
    n_candidates = len(snapshot_events_df)

    b0_codes: list[str] = []
    if b0_snapshot_events is not None and not b0_snapshot_events.empty:
        if "code" in b0_snapshot_events.columns:
            b0_codes = b0_snapshot_events["code"].dropna().astype(str).tolist()
    b0_selected_count = len(b0_codes)

    if benchmark_mode == "MATCHED_N":
        if b0_snapshot_events is not None:
            actual_picks_count = min(b0_selected_count, n_candidates)
        else:
            actual_picks_count = min(pick_limit, n_candidates)
    else:
        actual_picks_count = min(pick_limit, n_candidates)

    if n_candidates == 0 or actual_picks_count == 0:
        summary: dict[str, Any] = {
            "snapshot_date": snapshot_date,
            "benchmark_mode": benchmark_mode,
            "total_candidates": n_candidates,
            "actual_picks_count": actual_picks_count,
            "b0_selected_count": b0_selected_count,
            "b0_valid_entry_count": 0,
            "b0_is_asof_valid": False,
            "b0_is_w1_valid": False,
            "b0_is_w2_valid": False,
            "b0_is_w3_valid": False,
            "b0_is_w4_valid": False,
            "b0_w1_valid_count": 0,
            "b0_w2_valid_count": 0,
            "b0_w3_valid_count": 0,
            "b0_w4_valid_count": 0,
            "status": "ZERO_PICKS_MATCHED" if actual_picks_count == 0 else "EMPTY_POOL",
            "valid_draw_pct": 100.0 if actual_picks_count == 0 else 0.0,
            "w1_valid_draw_pct": 100.0 if actual_picks_count == 0 else 0.0,
            "w2_valid_draw_pct": 100.0 if actual_picks_count == 0 else 0.0,
            "w3_valid_draw_pct": 100.0 if actual_picks_count == 0 else 0.0,
            "w4_valid_draw_pct": 100.0 if actual_picks_count == 0 else 0.0,
        }
        return summary, pd.DataFrame()

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
        current_rets = [float(ev["current_return_to_asof_pct"]) for ev in valid_events if ev.get("current_return_to_asof_pct") is not None and not pd.isna(ev.get("current_return_to_asof_pct"))]
        exec_rets = [float(ev["executed_return_to_asof_pct"]) for ev in valid_events if ev.get("executed_return_to_asof_pct") is not None and not pd.isna(ev.get("executed_return_to_asof_pct"))]
        max_gains = [float(ev["max_gain_to_asof_pct"]) for ev in valid_events if ev.get("max_gain_to_asof_pct") is not None and not pd.isna(ev.get("max_gain_to_asof_pct"))]
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

        is_asof_valid = bool(
            is_valid_draw
            and len(current_rets) == actual_picks_count
            and len(exec_rets) == actual_picks_count
            and len(max_gains) == actual_picks_count
        )

        draw_dict = {
            "snapshot_date": snapshot_date,
            "draw_index": draw_idx,
            "is_valid_draw": is_valid_draw,
            "valid_candidates_count": len(valid_events),
            "sampled_codes": ",".join(sampled_codes),
            "asof_mean_return_pct": np.mean(current_rets) if is_asof_valid else np.nan,
            "asof_mean_exec_return_pct": np.mean(exec_rets) if is_asof_valid else np.nan,
            "asof_mean_max_gain_pct": np.mean(max_gains) if is_asof_valid else np.nan,
            "asof_worst_return_pct": np.min(current_rets) if is_asof_valid else np.nan,
            "asof_stop8_count": stops_hit if is_asof_valid else np.nan,
            "asof_has_profit20": bool(profits20_hit > 0) if is_asof_valid else False,
            "asof_all_stopped": bool(stops_hit == len(valid_events) and len(valid_events) > 0) if is_asof_valid else False,
        }

        for w_idx in [1, 2, 3, 4]:
            r_list = w_rets[w_idx]
            mg_list = w_max_gains[w_idx]
            is_w_valid = bool(
                is_valid_draw
                and len(r_list) == actual_picks_count
                and len(mg_list) == actual_picks_count
            )
            draw_dict[f"w{w_idx}_mean_return_pct"] = np.mean(r_list) if is_w_valid else np.nan
            draw_dict[f"w{w_idx}_mean_max_gain_pct"] = np.mean(mg_list) if is_w_valid else np.nan
            draw_dict[f"w{w_idx}_worst_return_pct"] = np.min(r_list) if is_w_valid else np.nan
            draw_dict[f"w{w_idx}_stop8_count"] = w_stops[w_idx] if is_w_valid else np.nan

        draw_records.append(draw_dict)

    draws_df = pd.DataFrame(draw_records)
    valid_draws_df = draws_df[draws_df["is_valid_draw"]]

    # Compute Quantiles strictly across valid draws
    summary = {
        "snapshot_date": snapshot_date,
        "benchmark_mode": benchmark_mode,
        "total_candidates": n_candidates,
        "actual_picks_count": actual_picks_count,
        "b0_selected_count": b0_selected_count,
        "n_draws": n_draws,
        "seed": seed,
        "valid_draw_pct": round(float((len(valid_draws_df) / n_draws) * 100.0), 2),
    }

    # Explicit Horizon-Specific Coverage
    for w_idx in [1, 2, 3, 4]:
        s_w = valid_draws_df[f"w{w_idx}_mean_return_pct"].dropna() if not valid_draws_df.empty else pd.Series(dtype=float)
        summary[f"w{w_idx}_valid_draw_pct"] = round(float((len(s_w) / n_draws) * 100.0), 2)

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
        s = valid_draws_df[metric].dropna() if not valid_draws_df.empty else pd.Series(dtype=float)
        if not s.empty:
            summary[f"{metric}_p05"] = round(float(np.percentile(s, 5)), 4)
            summary[f"{metric}_p25"] = round(float(np.percentile(s, 25)), 4)
            summary[f"{metric}_p50"] = round(float(np.percentile(s, 50)), 4)
            summary[f"{metric}_p75"] = round(float(np.percentile(s, 75)), 4)
            summary[f"{metric}_p95"] = round(float(np.percentile(s, 95)), 4)
        else:
            for q in ["p05", "p25", "p50", "p75", "p95"]:
                summary[f"{metric}_{q}"] = np.nan

    # Calculate B0 Actual Performance & Percentile under STRICT IDENTICAL CENSORING (No Survivor Reweighting)
    b0_valid_entry_count = 0
    b0_is_asof_valid = False
    b0_is_w_valid: dict[int, bool] = {1: False, 2: False, 3: False, 4: False}
    b0_w_valid_counts: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}

    if b0_selected_count > 0:
        b0_event_records = [event_lookup.get(c) for c in b0_codes if c in event_lookup]
        b0_valid_entries = [
            ev for ev in b0_event_records
            if ev is not None
            and ev.get("entry_status") == "ENTRY_OK"
            and ev.get("entry_open") is not None
            and not pd.isna(ev.get("entry_open"))
        ]
        b0_valid_entry_count = len(b0_valid_entries)
        b0_is_entry_valid = (b0_valid_entry_count == b0_selected_count)

        # As-Of evaluation
        b0_asof_cur = [float(ev["current_return_to_asof_pct"]) for ev in b0_valid_entries if ev.get("current_return_to_asof_pct") is not None and not pd.isna(ev.get("current_return_to_asof_pct"))]
        b0_asof_exc = [float(ev["executed_return_to_asof_pct"]) for ev in b0_valid_entries if ev.get("executed_return_to_asof_pct") is not None and not pd.isna(ev.get("executed_return_to_asof_pct"))]
        b0_asof_mg = [float(ev["max_gain_to_asof_pct"]) for ev in b0_valid_entries if ev.get("max_gain_to_asof_pct") is not None and not pd.isna(ev.get("max_gain_to_asof_pct"))]
        b0_asof_stops = sum(1 for ev in b0_valid_entries if ev.get("stop_8_hit_ever") is True)

        b0_is_asof_valid = bool(
            b0_is_entry_valid
            and len(b0_asof_cur) == b0_selected_count
            and len(b0_asof_exc) == b0_selected_count
            and len(b0_asof_mg) == b0_selected_count
        )

        if b0_is_asof_valid:
            summary["b0_actual_asof_mean_return_pct"] = round(float(np.mean(b0_asof_cur)), 4)
            summary["b0_actual_asof_mean_exec_return_pct"] = round(float(np.mean(b0_asof_exc)), 4)
            summary["b0_actual_asof_mean_max_gain_pct"] = round(float(np.mean(b0_asof_mg)), 4)
            summary["b0_actual_stop8_count"] = int(b0_asof_stops)

            if not valid_draws_df.empty and not valid_draws_df["asof_mean_return_pct"].dropna().empty:
                dist = valid_draws_df["asof_mean_return_pct"].dropna().values
                summary["b0_asof_return_percentile"] = round(float((np.sum(dist <= np.mean(b0_asof_cur)) / len(dist)) * 100.0), 2)
            else:
                summary["b0_asof_return_percentile"] = np.nan

            if not valid_draws_df.empty and not valid_draws_df["asof_mean_exec_return_pct"].dropna().empty:
                dist_exec = valid_draws_df["asof_mean_exec_return_pct"].dropna().values
                summary["b0_asof_exec_return_percentile"] = round(float((np.sum(dist_exec <= np.mean(b0_asof_exc)) / len(dist_exec)) * 100.0), 2)
            else:
                summary["b0_asof_exec_return_percentile"] = np.nan
        else:
            summary["b0_actual_asof_mean_return_pct"] = np.nan
            summary["b0_actual_asof_mean_exec_return_pct"] = np.nan
            summary["b0_actual_asof_mean_max_gain_pct"] = np.nan
            summary["b0_actual_stop8_count"] = int(b0_asof_stops) if b0_is_entry_valid else np.nan
            summary["b0_asof_return_percentile"] = np.nan
            summary["b0_asof_exec_return_percentile"] = np.nan

        # Horizons 1..4
        for w_idx in [1, 2, 3, 4]:
            b0_w_r = []
            b0_w_mg = []
            b0_w_st = 0
            for c in b0_codes:
                w_info = weekly_lookup.get((c, w_idx))
                if w_info:
                    r = w_info.get("week_close_return_from_entry_pct")
                    if r is not None and not pd.isna(r):
                        b0_w_r.append(float(r))
                    mg = w_info.get("week_max_gain_from_entry_pct")
                    if mg is not None and not pd.isna(mg):
                        b0_w_mg.append(float(mg))
                    if w_info.get("stop_8_hit_by_week_end") is True:
                        b0_w_st += 1

            b0_w_valid_counts[w_idx] = len(b0_w_r)
            is_w_valid = bool(
                b0_is_entry_valid
                and len(b0_w_r) == b0_selected_count
                and len(b0_w_mg) == b0_selected_count
            )
            b0_is_w_valid[w_idx] = is_w_valid

            if is_w_valid:
                b0_mean_r = float(np.mean(b0_w_r))
                b0_mean_mg = float(np.mean(b0_w_mg))
                summary[f"b0_actual_w{w_idx}_mean_return_pct"] = round(b0_mean_r, 4)
                summary[f"b0_actual_w{w_idx}_mean_max_gain_pct"] = round(b0_mean_mg, 4)
                summary[f"b0_actual_w{w_idx}_stop8_count"] = int(b0_w_st)

                if not valid_draws_df.empty and not valid_draws_df[f"w{w_idx}_mean_return_pct"].dropna().empty:
                    dist_w = valid_draws_df[f"w{w_idx}_mean_return_pct"].dropna().values
                    summary[f"b0_w{w_idx}_return_percentile"] = round(float((np.sum(dist_w <= b0_mean_r) / len(dist_w)) * 100.0), 2)
                else:
                    summary[f"b0_w{w_idx}_return_percentile"] = np.nan
            else:
                summary[f"b0_actual_w{w_idx}_mean_return_pct"] = np.nan
                summary[f"b0_actual_w{w_idx}_mean_max_gain_pct"] = np.nan
                summary[f"b0_actual_w{w_idx}_stop8_count"] = np.nan
                summary[f"b0_w{w_idx}_return_percentile"] = np.nan
    else:
        summary["b0_actual_asof_mean_return_pct"] = np.nan
        summary["b0_actual_asof_mean_exec_return_pct"] = np.nan
        summary["b0_actual_asof_mean_max_gain_pct"] = np.nan
        summary["b0_actual_stop8_count"] = np.nan
        summary["b0_asof_return_percentile"] = np.nan
        summary["b0_asof_exec_return_percentile"] = np.nan
        for w_idx in [1, 2, 3, 4]:
            summary[f"b0_actual_w{w_idx}_mean_return_pct"] = np.nan
            summary[f"b0_actual_w{w_idx}_mean_max_gain_pct"] = np.nan
            summary[f"b0_actual_w{w_idx}_stop8_count"] = np.nan
            summary[f"b0_w{w_idx}_return_percentile"] = np.nan

    summary["b0_valid_entry_count"] = b0_valid_entry_count
    summary["b0_is_asof_valid"] = b0_is_asof_valid
    for w_idx in [1, 2, 3, 4]:
        summary[f"b0_is_w{w_idx}_valid"] = b0_is_w_valid[w_idx]
        summary[f"b0_w{w_idx}_valid_count"] = b0_w_valid_counts[w_idx]

    return summary, draws_df


def run_random_top3_benchmark(
    event_outcomes_df: pd.DataFrame,
    weekly_outcomes_df: pd.DataFrame,
    b0_events_df: pd.DataFrame | None = None,
    n_draws_per_week: int = 1000,
    seed: int = 42,
    pick_limit: int = 3,
    benchmark_mode: str = "MATCHED_N",
    output_distribution_csv: Path | str | None = "backtest/b0_top3_quality_audit/output/random_signal_top3_distribution.csv",
) -> pd.DataFrame:
    """Run weekly 1,000-draw random benchmark across all snapshot weeks (default: MATCHED_N)."""
    snapshots = sorted(event_outcomes_df["snapshot_date"].unique().tolist())
    summary_rows: list[dict[str, Any]] = []

    logger.info(
        f"Running random benchmark ({benchmark_mode}) for {len(snapshots)} snapshots "
        f"({n_draws_per_week} draws/week, seed={seed})..."
    )

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
            pick_limit=pick_limit,
            benchmark_mode=benchmark_mode,
        )
        summary_rows.append(snap_sum)

    dist_df = pd.DataFrame(summary_rows)

    if output_distribution_csv is not None:
        out_p = Path(output_distribution_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        dist_df.to_csv(out_p, index=False, encoding="utf-8-sig")
        logger.info(f"Saved random benchmark distribution ({benchmark_mode}) to {out_p}")

    return dist_df

