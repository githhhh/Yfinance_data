"""Generate the B0 Quality vs Matched-N Random research report.

This module is intentionally report-only. It consumes frozen Phase 1/2 audit
artifacts and does not modify production selector or frozen rule logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.b0_top3_quality_audit.random_control import run_random_top3_for_snapshot


HORIZONS = (1, 2, 4)
REPORT_NAME = "B0_QUALITY_VS_MATCHED_RANDOM_REPORT.md"


@dataclass(frozen=True)
class ReportPaths:
    root_dir: Path
    output_dir: Path
    events_path: Path
    weekly_path: Path
    b0_events_path: Path
    random_summary_path: Path
    three_tier_weekly_path: Path
    three_tier_summary_path: Path


def default_paths() -> ReportPaths:
    root_dir = Path(__file__).resolve().parent
    output_dir = root_dir / "output"
    return ReportPaths(
        root_dir=root_dir,
        output_dir=output_dir,
        events_path=root_dir / "data" / "candidate_event_outcomes.parquet",
        weekly_path=root_dir / "data" / "candidate_weekly_outcomes.parquet",
        b0_events_path=output_dir / "b0_selection_events.csv",
        random_summary_path=output_dir / "random_signal_top3_distribution.csv",
        three_tier_weekly_path=output_dir / "three_tier_weekly_comparison.csv",
        three_tier_summary_path=output_dir / "three_tier_alpha_summary.csv",
    )


def _valid_horizon_frame(random_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    valid_col = f"b0_is_w{horizon}_valid"
    b0_col = f"b0_actual_w{horizon}_mean_return_pct"
    random_col = f"w{horizon}_mean_return_pct_p50"
    percentile_col = f"b0_w{horizon}_return_percentile"
    mask = (
        (random_df["actual_picks_count"] > 0)
        & (random_df[valid_col] == True)  # noqa: E712
        & random_df[b0_col].notna()
        & random_df[random_col].notna()
        & random_df[percentile_col].notna()
    )
    return random_df.loc[mask].copy()


def _pct(value: float | int | np.floating | None) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return round(float(value), 4)


def _rate(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return np.nan
    return round(float(numerator) / float(denominator) * 100.0, 2)


def _bool_mean_pct(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    return round(float(clean.astype(bool).mean() * 100.0), 2)


def summarize_horizon_comparison(random_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize B0 vs Matched-N Random P50 by horizon.

    Zero-pick weeks and common-censored/immature B0 weeks are excluded from
    each horizon denominator.
    """
    zero_pick_weeks = int((random_df["actual_picks_count"] == 0).sum())
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        df = _valid_horizon_frame(random_df, horizon)
        b0_col = f"b0_actual_w{horizon}_mean_return_pct"
        random_col = f"w{horizon}_mean_return_pct_p50"
        spread = df[b0_col] - df[random_col]
        rows.append(
            {
                "horizon": f"W{horizon}",
                "valid_weeks": int(len(df)),
                "zero_pick_weeks_excluded": zero_pick_weeks,
                "b0_median_return_pct": _pct(df[b0_col].median()),
                "matched_random_p50_median_return_pct": _pct(df[random_col].median()),
                "paired_spread_median_pct": _pct(spread.median()),
                "b0_mean_return_pct": _pct(df[b0_col].mean()),
                "matched_random_p50_mean_return_pct": _pct(df[random_col].mean()),
                "paired_spread_mean_pct": _pct(spread.mean()),
                "beat_random_p50_rate_pct": _rate((spread > 0).sum(), len(spread)),
                "mature_denominator_note": (
                    "actual_picks_count>0, B0 common-censored valid, "
                    "Matched-N P50 available"
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_percentile_quality(random_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        df = _valid_horizon_frame(random_df, horizon)
        percentile_col = f"b0_w{horizon}_return_percentile"
        p = df[percentile_col].dropna()
        rows.append(
            {
                "horizon": f"W{horizon}",
                "valid_percentile_weeks": int(len(p)),
                "median_percentile": _pct(p.median()),
                "mean_percentile": _pct(p.mean()),
                "weeks_gt_p50_pct": _rate((p > 50.0).sum(), len(p)),
                "weeks_gt_p75_pct": _rate((p > 75.0).sum(), len(p)),
                "weeks_gt_p90_pct": _rate((p > 90.0).sum(), len(p)),
            }
        )
    return pd.DataFrame(rows)


def summarize_stability(
    random_df: pd.DataFrame,
    three_tier_weekly_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize beat-random stability across time segments."""
    rows: list[dict[str, Any]] = []
    train_dates: set[str] = set()
    contaminated_dates: set[str] = set()
    if three_tier_weekly_df is not None and not three_tier_weekly_df.empty:
        all_eval_dates = sorted(three_tier_weekly_df["snapshot_date"].astype(str).unique())
        train_dates = set(all_eval_dates[:30])
        contaminated_dates = set(all_eval_dates[30:40])

    for horizon in HORIZONS:
        df = _valid_horizon_frame(random_df, horizon).sort_values("snapshot_date").copy()
        b0_col = f"b0_actual_w{horizon}_mean_return_pct"
        random_col = f"w{horizon}_mean_return_pct_p50"
        spread = df[b0_col] - df[random_col]
        df["spread"] = spread

        midpoint = int(np.ceil(len(df) / 2.0))
        segments: list[tuple[str, pd.DataFrame]] = [
            ("All valid weeks", df),
            ("Early half", df.iloc[:midpoint]),
            ("Late half", df.iloc[midpoint:]),
        ]
        if train_dates:
            segments.append(("Train-era weeks 1-30", df[df["snapshot_date"].astype(str).isin(train_dates)]))
        if contaminated_dates:
            segments.append(
                (
                    "Contaminated historical validation weeks 31-40",
                    df[df["snapshot_date"].astype(str).isin(contaminated_dates)],
                )
            )

        for segment, seg_df in segments:
            rows.append(
                {
                    "horizon": f"W{horizon}",
                    "segment": segment,
                    "valid_weeks": int(len(seg_df)),
                    "beat_random_p50_rate_pct": _rate((seg_df["spread"] > 0).sum(), len(seg_df)),
                    "paired_spread_median_pct": _pct(seg_df["spread"].median()),
                    "paired_spread_mean_pct": _pct(seg_df["spread"].mean()),
                    "median_percentile": _pct(seg_df[f"b0_w{horizon}_return_percentile"].median()),
                }
            )
    return pd.DataFrame(rows)


def _week_seed(base_seed: int, snapshot_date: str) -> int:
    return base_seed + int(pd.Timestamp(snapshot_date).strftime("%y%m%d")) % 100000


def _is_entry_valid(record: dict[str, Any] | None) -> bool:
    if record is None:
        return False
    entry_open = record.get("entry_open")
    return bool(
        record.get("entry_status") == "ENTRY_OK"
        and entry_open is not None
        and not pd.isna(entry_open)
    )


def _mean_bool(records: list[dict[str, Any]], field: str) -> float:
    vals = [r.get(field) for r in records if r.get(field) is not None and not pd.isna(r.get(field))]
    if not vals:
        return np.nan
    return float(np.mean([1.0 if bool(v) else 0.0 for v in vals]))


def _mean_numeric(records: list[dict[str, Any]], field: str) -> float:
    vals = [float(r[field]) for r in records if r.get(field) is not None and not pd.isna(r.get(field))]
    if not vals:
        return np.nan
    return float(np.mean(vals))


def _min_numeric(records: list[dict[str, Any]], field: str) -> float:
    vals = [float(r[field]) for r in records if r.get(field) is not None and not pd.isna(r.get(field))]
    if not vals:
        return np.nan
    return float(np.min(vals))


def _draw_event_rates(
    draws_df: pd.DataFrame,
    event_lookup: dict[str, dict[str, Any]],
    actual_picks_count: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if draws_df.empty or actual_picks_count == 0:
        return pd.DataFrame(records)

    valid_draws = draws_df[
        (draws_df["is_valid_draw"] == True)  # noqa: E712
        & draws_df["asof_mean_return_pct"].notna()
    ].copy()
    for _, row in valid_draws.iterrows():
        codes = [c for c in str(row["sampled_codes"]).split(",") if c]
        sampled_records = [event_lookup[c] for c in codes if c in event_lookup]
        if len(sampled_records) != actual_picks_count:
            continue
        records.append(
            {
                "stop8_before_profit20_rate_pct": _mean_bool(sampled_records, "stop8_before_profit20") * 100.0,
                "stop8_ever_rate_pct": _mean_bool(sampled_records, "stop_8_hit_ever") * 100.0,
                "gap_stop_rate_pct": _mean_bool(sampled_records, "gap_stop") * 100.0,
                "profit20_rate_pct": _mean_bool(sampled_records, "profit20_hit") * 100.0,
                "has_profit20_winner": bool(any(bool(r.get("profit20_hit")) for r in sampled_records)),
                "all_stopped_ever": bool(all(bool(r.get("stop_8_hit_ever")) for r in sampled_records)),
                "worst_asof_return_pct": _min_numeric(sampled_records, "current_return_to_asof_pct"),
                "mean_max_gain_to_asof_pct": _mean_numeric(sampled_records, "max_gain_to_asof_pct"),
            }
        )
    return pd.DataFrame(records)


def build_weekly_quality_detail(
    events_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    b0_events_df: pd.DataFrame,
    random_summary_df: pd.DataFrame,
    n_draws: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Build weekly downside/upside details using Matched-N random draws."""
    rows: list[dict[str, Any]] = []
    for _, summary_row in random_summary_df.sort_values("snapshot_date").iterrows():
        snap = str(summary_row["snapshot_date"])
        snap_events = events_df[events_df["snapshot_date"].astype(str) == snap].copy()
        snap_b0 = b0_events_df[b0_events_df["snapshot_date"].astype(str) == snap].copy()
        actual_picks_count = int(summary_row.get("actual_picks_count", 0) or 0)
        event_lookup = {str(r["code"]): r.to_dict() for _, r in snap_events.iterrows()}
        b0_codes = snap_b0["code"].dropna().astype(str).tolist() if not snap_b0.empty else []
        b0_records = [event_lookup.get(code) for code in b0_codes]
        b0_valid_records = [r for r in b0_records if _is_entry_valid(r)]
        b0_asof_valid = bool(
            actual_picks_count > 0
            and len(b0_codes) == actual_picks_count
            and len(b0_valid_records) == actual_picks_count
        )

        row: dict[str, Any] = {
            "snapshot_date": snap,
            "actual_picks_count": actual_picks_count,
            "b0_selected_count": int(summary_row.get("b0_selected_count", len(b0_codes)) or 0),
            "b0_is_asof_valid": bool(summary_row.get("b0_is_asof_valid", False)),
            "b0_is_w1_valid": bool(summary_row.get("b0_is_w1_valid", False)),
            "b0_is_w2_valid": bool(summary_row.get("b0_is_w2_valid", False)),
            "b0_is_w4_valid": bool(summary_row.get("b0_is_w4_valid", False)),
        }

        if b0_asof_valid:
            row.update(
                {
                    "b0_stop8_before_profit20_rate_pct": _pct(_mean_bool(b0_valid_records, "stop8_before_profit20") * 100.0),
                    "b0_stop8_ever_rate_pct": _pct(_mean_bool(b0_valid_records, "stop_8_hit_ever") * 100.0),
                    "b0_gap_stop_rate_pct": _pct(_mean_bool(b0_valid_records, "gap_stop") * 100.0),
                    "b0_profit20_rate_pct": _pct(_mean_bool(b0_valid_records, "profit20_hit") * 100.0),
                    "b0_has_profit20_winner": bool(any(bool(r.get("profit20_hit")) for r in b0_valid_records)),
                    "b0_all_stopped_ever": bool(all(bool(r.get("stop_8_hit_ever")) for r in b0_valid_records)),
                    "b0_worst_asof_return_pct": _pct(_min_numeric(b0_valid_records, "current_return_to_asof_pct")),
                    "b0_mean_max_gain_to_asof_pct": _pct(_mean_numeric(b0_valid_records, "max_gain_to_asof_pct")),
                }
            )
        else:
            row.update(
                {
                    "b0_stop8_before_profit20_rate_pct": np.nan,
                    "b0_stop8_ever_rate_pct": np.nan,
                    "b0_gap_stop_rate_pct": np.nan,
                    "b0_profit20_rate_pct": np.nan,
                    "b0_has_profit20_winner": np.nan,
                    "b0_all_stopped_ever": np.nan,
                    "b0_worst_asof_return_pct": np.nan,
                    "b0_mean_max_gain_to_asof_pct": np.nan,
                }
            )

        for horizon in HORIZONS:
            row[f"b0_w{horizon}_mean_max_gain_pct"] = summary_row.get(
                f"b0_actual_w{horizon}_mean_max_gain_pct", np.nan
            )

        if actual_picks_count > 0 and not snap_events.empty:
            _, draws_df = run_random_top3_for_snapshot(
                snapshot_date=snap,
                snapshot_events_df=snap_events,
                weekly_outcomes_df=weekly_df,
                b0_snapshot_events=snap_b0,
                n_draws=n_draws,
                seed=_week_seed(seed, snap),
                benchmark_mode="MATCHED_N",
            )
            draw_rates = _draw_event_rates(draws_df, event_lookup, actual_picks_count)
            if not draw_rates.empty:
                row.update(
                    {
                        "random_stop8_before_profit20_rate_p50_pct": _pct(draw_rates["stop8_before_profit20_rate_pct"].median()),
                        "random_stop8_ever_rate_p50_pct": _pct(draw_rates["stop8_ever_rate_pct"].median()),
                        "random_gap_stop_rate_p50_pct": _pct(draw_rates["gap_stop_rate_pct"].median()),
                        "random_profit20_rate_p50_pct": _pct(draw_rates["profit20_rate_pct"].median()),
                        "random_has_profit20_winner_rate_pct": _bool_mean_pct(draw_rates["has_profit20_winner"]),
                        "random_all_stopped_ever_rate_pct": _bool_mean_pct(draw_rates["all_stopped_ever"]),
                        "random_worst_asof_return_p50_pct": _pct(draw_rates["worst_asof_return_pct"].median()),
                        "random_mean_max_gain_to_asof_p50_pct": _pct(draw_rates["mean_max_gain_to_asof_pct"].median()),
                    }
                )
                for horizon in HORIZONS:
                    col = f"w{horizon}_mean_max_gain_pct"
                    row[f"random_w{horizon}_mean_max_gain_p50_pct"] = _pct(draws_df[col].dropna().median())
            else:
                for col in [
                    "random_stop8_before_profit20_rate_p50_pct",
                    "random_stop8_ever_rate_p50_pct",
                    "random_gap_stop_rate_p50_pct",
                    "random_profit20_rate_p50_pct",
                    "random_has_profit20_winner_rate_pct",
                    "random_all_stopped_ever_rate_pct",
                    "random_worst_asof_return_p50_pct",
                    "random_mean_max_gain_to_asof_p50_pct",
                ]:
                    row[col] = np.nan
                for horizon in HORIZONS:
                    row[f"random_w{horizon}_mean_max_gain_p50_pct"] = np.nan
        else:
            for col in [
                "random_stop8_before_profit20_rate_p50_pct",
                "random_stop8_ever_rate_p50_pct",
                "random_gap_stop_rate_p50_pct",
                "random_profit20_rate_p50_pct",
                "random_has_profit20_winner_rate_pct",
                "random_all_stopped_ever_rate_pct",
                "random_worst_asof_return_p50_pct",
                "random_mean_max_gain_to_asof_p50_pct",
            ]:
                row[col] = np.nan
            for horizon in HORIZONS:
                row[f"random_w{horizon}_mean_max_gain_p50_pct"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_downside_upside(weekly_detail_df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        (
            "Downside",
            "Stop8 before Profit20 rate",
            "b0_stop8_before_profit20_rate_pct",
            "random_stop8_before_profit20_rate_p50_pct",
            False,
        ),
        ("Downside", "Stop8 ever rate", "b0_stop8_ever_rate_pct", "random_stop8_ever_rate_p50_pct", False),
        ("Downside", "Gap-stop rate", "b0_gap_stop_rate_pct", "random_gap_stop_rate_p50_pct", False),
        (
            "Downside",
            "All picks stopped week rate",
            "b0_all_stopped_ever",
            "random_all_stopped_ever_rate_pct",
            False,
        ),
        (
            "Downside",
            "Worst pick as-of return",
            "b0_worst_asof_return_pct",
            "random_worst_asof_return_p50_pct",
            True,
        ),
        ("Upside", "Profit20 pick rate", "b0_profit20_rate_pct", "random_profit20_rate_p50_pct", True),
        (
            "Upside",
            "Weeks with >=1 Profit20 winner",
            "b0_has_profit20_winner",
            "random_has_profit20_winner_rate_pct",
            True,
        ),
        (
            "Upside",
            "As-of mean max gain",
            "b0_mean_max_gain_to_asof_pct",
            "random_mean_max_gain_to_asof_p50_pct",
            True,
        ),
        ("Upside", "W1 mean max gain", "b0_w1_mean_max_gain_pct", "random_w1_mean_max_gain_p50_pct", True),
        ("Upside", "W2 mean max gain", "b0_w2_mean_max_gain_pct", "random_w2_mean_max_gain_p50_pct", True),
        ("Upside", "W4 mean max gain", "b0_w4_mean_max_gain_pct", "random_w4_mean_max_gain_p50_pct", True),
    ]

    rows: list[dict[str, Any]] = []
    for category, metric, b0_col, random_col, higher_is_better in specs:
        df = weekly_detail_df[
            (weekly_detail_df["actual_picks_count"] > 0)
            & weekly_detail_df[b0_col].notna()
            & weekly_detail_df[random_col].notna()
        ].copy()
        if b0_col in {"b0_all_stopped_ever", "b0_has_profit20_winner"}:
            b0_values = df[b0_col].astype(bool).astype(float) * 100.0
        else:
            b0_values = df[b0_col].astype(float)
        random_values = df[random_col].astype(float)
        spread = b0_values - random_values
        if higher_is_better:
            better = b0_values > random_values
        else:
            better = b0_values < random_values
        rows.append(
            {
                "category": category,
                "metric": metric,
                "valid_weeks": int(len(df)),
                "b0_median": _pct(b0_values.median()),
                "matched_random_p50_median": _pct(random_values.median()),
                "paired_spread_median": _pct(spread.median()),
                "b0_mean": _pct(b0_values.mean()),
                "matched_random_p50_mean": _pct(random_values.mean()),
                "paired_spread_mean": _pct(spread.mean()),
                "b0_better_than_random_p50_rate_pct": _rate(better.sum(), len(df)),
                "higher_is_better": higher_is_better,
            }
        )
    return pd.DataFrame(rows)


def summarize_alpha_decomposition(three_tier_summary_df: pd.DataFrame) -> pd.DataFrame:
    keep_metrics = {
        "Week 1 Executed Return": "W1",
        "Week 2 Executed Return": "W2",
        "Week 4 Executed Return": "W4",
        "Executed Return (to As-Of Secondary)": "AsOf executed",
    }
    rows: list[dict[str, Any]] = []
    for _, row in three_tier_summary_df.iterrows():
        metric = str(row["metric"])
        if metric not in keep_metrics:
            continue
        horizon = keep_metrics[metric]
        if horizon == "W1":
            interpretation = "W1 ranking alpha not proven; screening/total lift is visible but weak."
        elif horizon == "W4":
            interpretation = "W4 ranking alpha is a promising historical signal, pending forward shadow validation."
        else:
            interpretation = "Supportive context; not the primary ranking-alpha claim."
        rows.append(
            {
                "horizon": horizon,
                "mature_eval_weeks": int(row["mature_eval_weeks"]),
                "l0_signal_median_pct": _pct(row["l0_signal_median"]),
                "l1_screened_median_pct": _pct(row["l1_eligible_median"]),
                "b0_l2_median_pct": _pct(row["l2_b0_median"]),
                "screening_alpha_weekly_spread_median_pct": _pct(row["weekly_spread_screening_pct"]),
                "ranking_alpha_weekly_spread_median_pct": _pct(row["weekly_spread_ranking_pct"]),
                "total_alpha_weekly_spread_median_pct": _pct(row["weekly_spread_total_pct"]),
                "active_rank_weeks_count": int(row["active_rank_weeks_count"]),
                "active_rank_spread_ranking_pct": _pct(row["active_rank_spread_ranking_pct"]),
                "active_rank_win_rate_b0_vs_l1_pct": _pct(row["active_rank_win_rate_l2_vs_l1_pct"]),
                "win_rate_b0_vs_l0_pct": _pct(row["win_rate_l2_vs_l0_pct"]),
                "win_rate_b0_vs_l1_pct": _pct(row["win_rate_l2_vs_l1_pct"]),
                "win_rate_l1_vs_l0_pct": _pct(row["win_rate_l1_vs_l0_pct"]),
                "p_val_ranking_wilcoxon": _pct(row["p_val_ranking_wilcoxon"]),
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns is not None:
        df = df[columns].copy()
    if df.empty:
        return "_No rows._"
    return df.to_markdown(index=False)


def _fmt(value: Any, suffix: str = "%") -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):+.2f}{suffix}"


def generate_report_markdown(
    horizon_df: pd.DataFrame,
    percentile_df: pd.DataFrame,
    downside_upside_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    random_df: pd.DataFrame,
) -> str:
    w1 = horizon_df[horizon_df["horizon"] == "W1"].iloc[0]
    w4 = horizon_df[horizon_df["horizon"] == "W4"].iloc[0]
    p_w4 = percentile_df[percentile_df["horizon"] == "W4"].iloc[0]
    zero_pick_weeks = int((random_df["actual_picks_count"] == 0).sum())
    active_weeks = int((random_df["actual_picks_count"] > 0).sum())
    total_weeks = int(len(random_df))

    return f"""# B0 Quality vs Matched-N Random Report

**Frozen baseline commit**: `7cc3a439c89dd7e52135c7418590563b58b3c97e`  
**Primary benchmark**: Matched-N Random, 1,000 weekly draws, same holding count as B0 each week  
**Scope**: Phase 1/2 frozen historical audit artifacts under `backtest/b0_top3_quality_audit`  

## Executive Conclusion

B0 shows a positive but uneven selection-quality edge against equal-position Matched-N Random. The strongest direct evidence is W4: B0's median weekly portfolio return is {_fmt(w4['b0_median_return_pct'])} versus Matched-N Random P50 {_fmt(w4['matched_random_p50_median_return_pct'])}, with a paired median spread of {_fmt(w4['paired_spread_median_pct'])}. W4 percentile quality is also better than W1, with median weekly random percentile {p_w4['median_percentile']:.2f}.

W1 should remain conservatively worded. B0's W1 paired median spread is {_fmt(w1['paired_spread_median_pct'])}, and the existing three-tier decomposition does **not** prove W1 ranking alpha. The cleaner interpretation is: screening alpha is visible, W1 ranking alpha is unconfirmed, and W4 ranking alpha is a promising historical signal that must be validated by the forward shadow ledger starting 2026-08-28.

## 1. B0 vs Matched-N Random P50

Zero-pick weeks are excluded from return and percentile denominators. Each horizon also applies common censoring: if any B0 pick lacks a valid outcome for that horizon, that week is excluded for that horizon.

{_markdown_table(horizon_df, [
    'horizon',
    'valid_weeks',
    'b0_median_return_pct',
    'matched_random_p50_median_return_pct',
    'paired_spread_median_pct',
    'b0_mean_return_pct',
    'matched_random_p50_mean_return_pct',
    'paired_spread_mean_pct',
    'beat_random_p50_rate_pct',
])}

## 2. Weekly Random Percentile Quality

{_markdown_table(percentile_df, [
    'horizon',
    'valid_percentile_weeks',
    'median_percentile',
    'mean_percentile',
    'weeks_gt_p50_pct',
    'weeks_gt_p75_pct',
    'weeks_gt_p90_pct',
])}

## 3. Downside Quality and Upside Capture

Downside rows are better when lower, except worst-pick return where less negative is better. Upside rows are better when higher. Random columns use each week's Matched-N random P50, except winner/all-stopped week rates, which use the random probability of that weekly event.

{_markdown_table(downside_upside_df, [
    'category',
    'metric',
    'valid_weeks',
    'b0_median',
    'matched_random_p50_median',
    'paired_spread_median',
    'b0_mean',
    'matched_random_p50_mean',
    'b0_better_than_random_p50_rate_pct',
])}

## 4. Stability

{_markdown_table(stability_df, [
    'horizon',
    'segment',
    'valid_weeks',
    'beat_random_p50_rate_pct',
    'paired_spread_median_pct',
    'paired_spread_mean_pct',
    'median_percentile',
])}

## 5. Alpha Decomposition: L0 / L1 / B0

L0 is blind random from the signal pool. L1 is random after production-like eligibility screening and industry de-duplication. B0 is the deterministic production ranking/selection layer. Therefore:

- **Screening alpha** = L1 minus L0.
- **Ranking alpha** = B0/L2 minus L1.
- **Total alpha** = B0/L2 minus L0.

{_markdown_table(alpha_df, [
    'horizon',
    'mature_eval_weeks',
    'l0_signal_median_pct',
    'l1_screened_median_pct',
    'b0_l2_median_pct',
    'screening_alpha_weekly_spread_median_pct',
    'ranking_alpha_weekly_spread_median_pct',
    'active_rank_weeks_count',
    'active_rank_spread_ranking_pct',
    'active_rank_win_rate_b0_vs_l1_pct',
    'p_val_ranking_wilcoxon',
    'interpretation',
])}

## 6. Methodology Notes

- **Matched-N Random**: each random portfolio samples the same number of names B0 actually selected that week. A 1-pick B0 week is compared with random 1-pick portfolios; a 3-pick B0 week is compared with random 3-pick portfolios.
- **Common censoring**: B0 and random portfolios require all sampled picks to have valid entry and horizon outcomes. No survivor reweighting is allowed.
- **Maturity**: W1/W2/W4 denominators are horizon-specific and exclude immature or missing-outcome weeks.
- **0-pick rule**: {zero_pick_weeks} zero-pick week(s) out of {total_weeks} calendar benchmark rows are marked not applicable and excluded from return, percentile, downside and upside denominators. Active Matched-N rows: {active_weeks}.
- **Historical validation caveat**: Weeks 31-40 are contaminated historical validation because prior research already touched that period. They are useful for one-way audit reporting, not pure out-of-sample proof.
- **Forward shadow start**: unbiased validation starts from the pre-registered 2026-08-28 forward shadow ledger for B0, Pure Freshness, and Pure Close Position.

## 7. Reproducibility

Generated by:

```bash
PYTHONPATH=. python backtest/b0_top3_quality_audit/generate_b0_quality_vs_matched_random_report.py
```

Generated companion CSVs:

- `b0_quality_vs_matched_random_horizon_summary.csv`
- `b0_quality_vs_matched_random_percentile_summary.csv`
- `b0_quality_vs_matched_random_downside_upside_summary.csv`
- `b0_quality_vs_matched_random_stability_summary.csv`
- `b0_quality_vs_matched_random_alpha_decomposition.csv`
- `b0_quality_vs_matched_random_weekly_detail.csv`
"""


def run_report(paths: ReportPaths | None = None) -> dict[str, Path]:
    paths = paths or default_paths()
    events_df = pd.read_parquet(paths.events_path)
    weekly_df = pd.read_parquet(paths.weekly_path)
    b0_events_df = pd.read_csv(paths.b0_events_path)
    random_df = pd.read_csv(paths.random_summary_path)
    three_tier_weekly_df = pd.read_csv(paths.three_tier_weekly_path)
    three_tier_summary_df = pd.read_csv(paths.three_tier_summary_path)

    horizon_df = summarize_horizon_comparison(random_df)
    percentile_df = summarize_percentile_quality(random_df)
    stability_df = summarize_stability(random_df, three_tier_weekly_df)
    weekly_detail_df = build_weekly_quality_detail(events_df, weekly_df, b0_events_df, random_df)
    downside_upside_df = summarize_downside_upside(weekly_detail_df)
    alpha_df = summarize_alpha_decomposition(three_tier_summary_df)
    report = generate_report_markdown(
        horizon_df,
        percentile_df,
        downside_upside_df,
        stability_df,
        alpha_df,
        random_df,
    )

    outputs = {
        "report": paths.output_dir / REPORT_NAME,
        "horizon_summary": paths.output_dir / "b0_quality_vs_matched_random_horizon_summary.csv",
        "percentile_summary": paths.output_dir / "b0_quality_vs_matched_random_percentile_summary.csv",
        "downside_upside_summary": paths.output_dir / "b0_quality_vs_matched_random_downside_upside_summary.csv",
        "stability_summary": paths.output_dir / "b0_quality_vs_matched_random_stability_summary.csv",
        "alpha_decomposition": paths.output_dir / "b0_quality_vs_matched_random_alpha_decomposition.csv",
        "weekly_detail": paths.output_dir / "b0_quality_vs_matched_random_weekly_detail.csv",
    }
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    horizon_df.to_csv(outputs["horizon_summary"], index=False, encoding="utf-8-sig")
    percentile_df.to_csv(outputs["percentile_summary"], index=False, encoding="utf-8-sig")
    downside_upside_df.to_csv(outputs["downside_upside_summary"], index=False, encoding="utf-8-sig")
    stability_df.to_csv(outputs["stability_summary"], index=False, encoding="utf-8-sig")
    alpha_df.to_csv(outputs["alpha_decomposition"], index=False, encoding="utf-8-sig")
    weekly_detail_df.to_csv(outputs["weekly_detail"], index=False, encoding="utf-8-sig")
    outputs["report"].write_text(report, encoding="utf-8")
    return outputs


if __name__ == "__main__":
    written = run_report()
    print("Generated B0 Quality vs Matched-N Random report outputs:")
    for name, path in written.items():
        print(f"- {name}: {path}")
