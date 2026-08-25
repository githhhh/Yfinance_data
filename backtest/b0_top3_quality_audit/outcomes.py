"""Per-candidate full path calculation, stop execution, and event outcome extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.b0_top3_quality_audit.price_cache import DailyPriceCache
from backtest.b0_top3_quality_audit.weekly_aggregation import (
    compute_weekly_outcomes_for_candidate,
)

logger = logging.getLogger(__name__)


def compute_single_candidate_outcome(
    event_row: pd.Series | dict[str, Any],
    price_cache: DailyPriceCache,
    as_of_date: str = "2026-08-25",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute comprehensive path outcomes and weekly breakdown for a single review universe event.
    
    Returns:
        (event_outcome_dict, weekly_outcomes_list)
    """
    code = str(event_row.get("code", "")).strip().upper()
    snapshot_date = str(event_row.get("snapshot_date", "")).strip()

    base_record: dict[str, Any] = {k: event_row.get(k) for k in event_row.keys() if k != "daily_bars"}
    base_record["code"] = code
    base_record["snapshot_date"] = snapshot_date
    base_record["is_valid_entry"] = False

    bars = price_cache.get_prices_for_ticker(code)
    if bars.empty:
        base_record.update({
            "entry_status": "NO_PRICE_DATA",
            "entry_date": "",
            "entry_open": None,
            "sessions_from_snapshot_to_entry": None,
            "entry_unavailable_reason": "no_bars_in_cache",
            "latest_observation_date": "",
            "latest_close": None,
            "current_return_to_asof_pct": None,
            "executed_return_to_asof_pct": None,
            "stop_8_hit_ever": False,
            "stop_8_date": "",
            "trading_days_to_stop": None,
            "calendar_weeks_to_stop": None,
            "gap_stop": False,
            "same_day_path_ambiguous": False,
            "max_gain_to_asof_pct": None,
            "max_gain_date": "",
            "max_gain_before_stop_pct": None,
            "max_gain_before_stop_date": "",
            "max_drawdown_to_asof_pct": None,
            "profit20_hit": False,
            "profit20_date": "",
            "profit20_before_stop8": False,
            "stop8_before_profit20": False,
            "observation_trading_days": 0,
            "observation_calendar_weeks": 0,
            "outcome_censored": True,
            "terminal_data_status": "NO_PRICE_DATA",
            "week1_close_return_pct": None,
            "week1_max_gain_pct": None,
            "week1_max_drawdown_pct": None,
            "week1_stop8_hit": False,
            "week1_profit20_hit": False,
        })
        return base_record, []

    # Filter bars on or after snapshot_date
    future_bars = bars[bars["date"] > snapshot_date].sort_values("date").reset_index(drop=True)
    if future_bars.empty:
        base_record.update({
            "entry_status": "NO_FUTURE_BARS",
            "entry_date": "",
            "entry_open": None,
            "sessions_from_snapshot_to_entry": None,
            "entry_unavailable_reason": "no_bars_after_snapshot",
            "latest_observation_date": "",
            "latest_close": None,
            "current_return_to_asof_pct": None,
            "executed_return_to_asof_pct": None,
            "stop_8_hit_ever": False,
            "stop_8_date": "",
            "trading_days_to_stop": None,
            "calendar_weeks_to_stop": None,
            "gap_stop": False,
            "same_day_path_ambiguous": False,
            "max_gain_to_asof_pct": None,
            "max_gain_date": "",
            "max_gain_before_stop_pct": None,
            "max_gain_before_stop_date": "",
            "max_drawdown_to_asof_pct": None,
            "profit20_hit": False,
            "profit20_date": "",
            "profit20_before_stop8": False,
            "stop8_before_profit20": False,
            "observation_trading_days": 0,
            "observation_calendar_weeks": 0,
            "outcome_censored": True,
            "terminal_data_status": "NO_FUTURE_DATA",
            "week1_close_return_pct": None,
            "week1_max_gain_pct": None,
            "week1_max_drawdown_pct": None,
            "week1_stop8_hit": False,
            "week1_profit20_hit": False,
        })
        return base_record, []

    entry_bar = future_bars.iloc[0]
    entry_date = str(entry_bar["date"])
    entry_open = float(entry_bar["open"])

    # Compute entry calendar delay
    snap_dt = pd.Timestamp(snapshot_date)
    entry_dt = pd.Timestamp(entry_date)
    cal_days = (entry_dt - snap_dt).days

    if cal_days > 7:
        # If entry is delayed beyond 7 calendar days (> 1 week gap), mark as stale/invalid
        base_record.update({
            "entry_date": entry_date,
            "entry_open": entry_open,
            "is_valid_entry": False,
            "entry_status": "ENTRY_STALE_EXPIRED",
            "sessions_from_snapshot_to_entry": cal_days,
            "entry_unavailable_reason": f"stale_entry_delayed_{cal_days}d",
            "latest_observation_date": entry_date,
            "latest_close": entry_open,
            "current_return_to_asof_pct": None,
            "executed_return_to_asof_pct": None,
            "stop_8_hit_ever": False,
            "stop_8_date": "",
            "trading_days_to_stop": None,
            "calendar_weeks_to_stop": None,
            "gap_stop": False,
            "same_day_path_ambiguous": False,
            "max_gain_to_asof_pct": None,
            "max_gain_date": "",
            "max_gain_before_stop_pct": None,
            "max_gain_before_stop_date": "",
            "max_drawdown_to_asof_pct": None,
            "profit20_hit": False,
            "profit20_date": "",
            "profit20_before_stop8": False,
            "stop8_before_profit20": False,
            "observation_trading_days": 0,
            "observation_calendar_weeks": 0,
            "outcome_censored": True,
            "terminal_data_status": "STALE_ENTRY_ABANDONED",
            "week1_close_return_pct": None,
            "week1_max_gain_pct": None,
            "week1_max_drawdown_pct": None,
            "week1_stop8_hit": False,
            "week1_profit20_hit": False,
        })
        return base_record, []

    entry_status = "ENTRY_OK"
    is_valid_entry = True

    # Daily trajectory analysis
    stop_level = entry_open * 0.92
    profit20_level = entry_open * 1.20

    stop_hit = False
    stop_date = ""
    gap_stop = False
    stop_exit_price = None
    trading_days_to_stop = None
    cal_weeks_to_stop = None

    profit20_hit = False
    profit20_date = ""
    same_day_ambiguity = False

    max_high = entry_open
    max_high_date = entry_date
    min_low = entry_open

    max_gain_before_stop = entry_open
    max_gain_before_stop_date = entry_date

    holding_bars = future_bars[future_bars["date"] <= as_of_date].copy()
    if holding_bars.empty:
        holding_bars = future_bars.iloc[:1].copy()

    for idx, row in holding_bars.iterrows():
        d_str = str(row["date"])
        d_open = float(row["open"])
        d_high = float(row["high"])
        d_low = float(row["low"])
        d_close = float(row["close"])

        # Track global highest high and lowest low
        if d_high > max_high:
            max_high = d_high
            max_high_date = d_str
        if d_low < min_low:
            min_low = d_low

        # Track gain before stop
        if not stop_hit:
            if d_high > max_gain_before_stop:
                max_gain_before_stop = d_high
                max_gain_before_stop_date = d_str

        # Check profit 20%
        hit_20_today = bool(d_high >= profit20_level)
        if hit_20_today and not profit20_hit:
            profit20_hit = True
            profit20_date = d_str

        # Check stop 8%
        hit_stop_today = bool(d_open <= stop_level or d_low <= stop_level)
        if hit_stop_today and not stop_hit:
            stop_hit = True
            stop_date = d_str
            trading_days_to_stop = idx + 1
            cal_weeks_to_stop = int((pd.Timestamp(d_str) - entry_dt).days // 7) + 1

            if d_open <= stop_level:
                gap_stop = True
                stop_exit_price = d_open
            else:
                gap_stop = False
                stop_exit_price = stop_level

            # Check same day ambiguity
            if hit_20_today:
                same_day_ambiguity = True

    latest_bar = holding_bars.iloc[-1]
    latest_obs_date = str(latest_bar["date"])
    latest_close = float(latest_bar["close"])

    current_return_to_asof_pct = round((latest_close / entry_open - 1.0) * 100.0, 4)

    if stop_hit and stop_exit_price is not None:
        executed_return_to_asof_pct = round((stop_exit_price / entry_open - 1.0) * 100.0, 4)
    else:
        executed_return_to_asof_pct = current_return_to_asof_pct

    max_gain_to_asof_pct = round((max_high / entry_open - 1.0) * 100.0, 4)
    max_gain_before_stop_pct = round((max_gain_before_stop / entry_open - 1.0) * 100.0, 4)
    max_drawdown_to_asof_pct = round((min_low / entry_open - 1.0) * 100.0, 4)

    obs_trading_days = len(holding_bars)
    obs_cal_weeks = max(1, int((pd.Timestamp(latest_obs_date) - entry_dt).days // 7) + 1)

    profit20_before_stop8 = bool(profit20_hit and (not stop_hit or profit20_date < stop_date))
    stop8_before_profit20 = bool(stop_hit and (not profit20_hit or stop_date < profit20_date or same_day_ambiguity))

    # Weekly outcomes calculation
    weekly_list = compute_weekly_outcomes_for_candidate(
        snapshot_date=snapshot_date,
        code=code,
        entry_date=entry_date,
        entry_open=entry_open,
        bars=holding_bars,
        as_of_date=as_of_date,
    )

    # First natural week metrics
    w1 = weekly_list[0] if weekly_list else None
    week1_close_ret = w1["week_close_return_from_entry_pct"] if w1 else None
    week1_max_gain = w1["week_max_gain_from_entry_pct"] if w1 else None
    week1_max_dd = w1["week_max_drawdown_from_entry_pct"] if w1 else None
    week1_stop8 = w1["stop_8_hit_during_week"] if w1 else False
    week1_profit20 = bool(w1 and (w1["week_high"] >= profit20_level))

    base_record.update({
        "is_valid_entry": True,
        "entry_status": entry_status,
        "entry_date": entry_date,
        "entry_open": entry_open,
        "sessions_from_snapshot_to_entry": cal_days,
        "entry_unavailable_reason": "",
        "latest_observation_date": latest_obs_date,
        "latest_close": latest_close,
        "current_return_to_asof_pct": current_return_to_asof_pct,
        "executed_return_to_asof_pct": executed_return_to_asof_pct,
        "stop_8_hit_ever": stop_hit,
        "stop_8_date": stop_date,
        "trading_days_to_stop": trading_days_to_stop,
        "calendar_weeks_to_stop": cal_weeks_to_stop,
        "gap_stop": gap_stop,
        "same_day_path_ambiguous": same_day_ambiguity,
        "max_gain_to_asof_pct": max_gain_to_asof_pct,
        "max_gain_date": max_high_date,
        "max_gain_before_stop_pct": max_gain_before_stop_pct,
        "max_gain_before_stop_date": max_gain_before_stop_date,
        "max_drawdown_to_asof_pct": max_drawdown_to_asof_pct,
        "profit20_hit": profit20_hit,
        "profit20_date": profit20_date,
        "profit20_before_stop8": profit20_before_stop8,
        "stop8_before_profit20": stop8_before_profit20,
        "observation_trading_days": obs_trading_days,
        "observation_calendar_weeks": obs_cal_weeks,
        "outcome_censored": bool(not stop_hit and obs_cal_weeks < 8),
        "terminal_data_status": "OK" if latest_obs_date >= "2026-08-15" else "PARTIAL_HISTORY",
        "week1_close_return_pct": week1_close_ret,
        "week1_max_gain_pct": week1_max_gain,
        "week1_max_drawdown_pct": week1_max_dd,
        "week1_stop8_hit": week1_stop8,
        "week1_profit20_hit": week1_profit20,
    })

    return base_record, weekly_list


def compute_all_candidate_outcomes(
    events_df: pd.DataFrame,
    price_cache: DailyPriceCache,
    output_event_outcomes_parquet: Path | str | None = "backtest/b0_top3_quality_audit/data/candidate_event_outcomes.parquet",
    output_weekly_outcomes_parquet: Path | str | None = "backtest/b0_top3_quality_audit/data/candidate_weekly_outcomes.parquet",
    as_of_date: str = "2026-08-25",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute outcomes for all Review Universe events across all weeks.
    
    Returns:
        (event_outcomes_df, weekly_outcomes_df)
    """
    event_records: list[dict[str, Any]] = []
    weekly_records: list[dict[str, Any]] = []

    logger.info(f"Computing outcomes for {len(events_df)} review universe events...")
    for idx, row in events_df.iterrows():
        ev_res, w_res = compute_single_candidate_outcome(row, price_cache, as_of_date=as_of_date)
        event_records.append(ev_res)
        weekly_records.extend(w_res)

    event_outcomes_df = pd.DataFrame(event_records)
    weekly_outcomes_df = pd.DataFrame(weekly_records)

    if output_event_outcomes_parquet is not None:
        out_p = Path(output_event_outcomes_parquet)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        event_outcomes_df.to_parquet(out_p, index=False, engine="pyarrow")
        logger.info(f"Saved {len(event_outcomes_df)} candidate event outcomes to {out_p}")

    if output_weekly_outcomes_parquet is not None:
        out_w = Path(output_weekly_outcomes_parquet)
        out_w.parent.mkdir(parents=True, exist_ok=True)
        weekly_outcomes_df.to_parquet(out_w, index=False, engine="pyarrow")
        logger.info(f"Saved {len(weekly_outcomes_df)} weekly outcomes to {out_w}")

    return event_outcomes_df, weekly_outcomes_df
