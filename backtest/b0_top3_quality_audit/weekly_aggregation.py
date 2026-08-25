"""Natural-week calendar slicing and weekly candidate path aggregation."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def get_calendar_week_bounds(dt: date | datetime) -> tuple[date, date, str]:
    """Get Monday (start) and Friday (end) of the calendar week for given date.
    
    Returns:
        (monday_date, friday_date, iso_week_str e.g. '2025-W41')
    """
    if isinstance(dt, datetime):
        d = dt.date()
    else:
        d = dt
    monday = d - timedelta(days=d.weekday())
    friday = monday + timedelta(days=4)
    iso_year, iso_week, _ = d.isocalendar()
    week_str = f"{iso_year}-W{iso_week:02d}"
    return monday, friday, week_str


def slice_into_natural_weeks(
    bars: pd.DataFrame,
    entry_date_str: str,
    as_of_date: str = "2026-08-25",
) -> list[dict[str, Any]]:
    """Slice daily bars after entry_date into natural trading weeks (Mon-Fri).
    
    The natural week containing entry_date is assigned holding_week_index = 1.
    """
    if bars.empty:
        return []

    df = bars.copy()
    df["dt"] = pd.to_datetime(df["date"])
    df = df[df["dt"] >= pd.Timestamp(entry_date_str)].sort_values("dt").reset_index(drop=True)

    if df.empty:
        return []

    # Assign calendar week group
    records: list[dict[str, Any]] = []
    # Identify unique calendar weeks in order
    entry_dt = pd.Timestamp(entry_date_str).date()
    entry_mon, _, _ = get_calendar_week_bounds(entry_dt)
    as_of_dt = pd.Timestamp(as_of_date).date()

    # Group daily bars by (iso_year, iso_week)
    grouped_weeks: dict[tuple[int, int], pd.DataFrame] = {}
    for idx, row in df.iterrows():
        d = row["dt"].date()
        iso_y, iso_w, _ = d.isocalendar()
        key = (iso_y, iso_w)
        if key not in grouped_weeks:
            grouped_weeks[key] = []
        grouped_weeks[key].append(row)

    # Sort weeks chronologically
    sorted_week_keys = sorted(grouped_weeks.keys())

    for week_idx, (iso_y, iso_w) in enumerate(sorted_week_keys, 1):
        week_rows = pd.DataFrame(grouped_weeks[(iso_y, iso_w)])
        first_d = week_rows["dt"].iloc[0].date()
        last_d = week_rows["dt"].iloc[-1].date()
        mon, fri, cal_week_str = get_calendar_week_bounds(first_d)

        # A week is complete if friday is before as_of_dt
        is_complete = bool(fri <= as_of_dt)

        week_open = float(week_rows["open"].iloc[0])
        week_close = float(week_rows["close"].iloc[-1])
        week_high = float(week_rows["high"].max())
        week_low = float(week_rows["low"].min())

        records.append({
            "holding_week_index": week_idx,
            "calendar_week": cal_week_str,
            "week_start": mon.strftime("%Y-%m-%d"),
            "week_end": fri.strftime("%Y-%m-%d"),
            "week_first_trade_date": first_d.strftime("%Y-%m-%d"),
            "week_last_trade_date": last_d.strftime("%Y-%m-%d"),
            "week_trading_sessions": len(week_rows),
            "is_complete_week": is_complete,
            "week_open": week_open,
            "week_high": week_high,
            "week_low": week_low,
            "week_close": week_close,
            "daily_bars": week_rows,
        })

    return records


def compute_weekly_outcomes_for_candidate(
    snapshot_date: str,
    code: str,
    entry_date: str,
    entry_open: float,
    bars: pd.DataFrame,
    as_of_date: str = "2026-08-25",
) -> list[dict[str, Any]]:
    """Compute weekly path metrics and returns relative to entry_open."""
    if entry_open <= 0 or bars.empty:
        return []

    weekly_slices = slice_into_natural_weeks(bars, entry_date, as_of_date=as_of_date)
    stop_level = entry_open * 0.92

    results: list[dict[str, Any]] = []
    cum_max_high = entry_open
    cum_min_low = entry_open
    already_stopped = False

    for w in weekly_slices:
        w_idx = w["holding_week_index"]
        w_close = w["week_close"]
        w_high = w["week_high"]
        w_low = w["week_low"]

        cum_max_high = max(cum_max_high, w_high)
        cum_min_low = min(cum_min_low, w_low)

        # Check if stop hit during this week
        stop_hit_during = False
        for _, day_row in w["daily_bars"].iterrows():
            d_open = float(day_row["open"])
            d_low = float(day_row["low"])
            if d_open <= stop_level or d_low <= stop_level:
                stop_hit_during = True
                break

        if stop_hit_during:
            already_stopped = True

        res_row = {
            "snapshot_date": snapshot_date,
            "code": code,
            "entry_date": entry_date,
            "holding_week_index": w_idx,
            "calendar_week": w["calendar_week"],
            "week_start": w["week_start"],
            "week_end": w["week_end"],
            "week_trading_sessions": w["week_trading_sessions"],
            "is_complete_week": w["is_complete_week"],
            "week_open": w["week_open"],
            "week_high": w_high,
            "week_low": w_low,
            "week_close": w_close,
            "week_close_return_from_entry_pct": round((w_close / entry_open - 1.0) * 100.0, 4),
            "week_max_gain_from_entry_pct": round((w_high / entry_open - 1.0) * 100.0, 4),
            "week_max_drawdown_from_entry_pct": round((w_low / entry_open - 1.0) * 100.0, 4),
            "cumulative_max_gain_pct": round((cum_max_high / entry_open - 1.0) * 100.0, 4),
            "cumulative_max_drawdown_pct": round((cum_min_low / entry_open - 1.0) * 100.0, 4),
            "stop_8_hit_during_week": stop_hit_during,
            "stop_8_hit_by_week_end": already_stopped,
        }
        results.append(res_row)

    return results
