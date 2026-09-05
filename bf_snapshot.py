"""Authoritative BreakoutFollow snapshot/calendar semantics.

``snapshot_date`` always means the latest actual market-data date. Pool kind is
separate: a Thursday snapshot can be a COMPLETE snapshot when Friday was a
market holiday and the data file arrived in the Sat/Sun/Mon recovery window.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any


class BreakoutFollowPoolKind(str, Enum):
    MIDWEEK = "midweek"
    COMPLETE = "complete"


def _coerce_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if hasattr(value, "date") and callable(getattr(value, "date")):
        return value.date()
    text = str(value).strip()
    if len(text) >= 10:
        text = text[:10]
    return datetime.strptime(text, "%Y-%m-%d").date()


def _observed_fixed_holiday(
    year: int,
    month: int,
    day: int,
    *,
    observe_saturday: bool = True,
) -> date:
    actual = date(year, month, day)
    if observe_saturday and actual.weekday() == 5:
        return actual - timedelta(days=1)
    if actual.weekday() == 6:
        return actual + timedelta(days=1)
    return actual


def _nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    current = date(year, month, 1)
    offset = (weekday - current.weekday()) % 7
    return current + timedelta(days=offset + 7 * (nth - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    current = date(year, month + 1, 1) - timedelta(days=1)
    return current - timedelta(days=(current.weekday() - weekday) % 7)


def _easter_sunday(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _market_holidays_for_year(year: int) -> set[date]:
    holidays = {
        _observed_fixed_holiday(year, 1, 1, observe_saturday=False),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter_sunday(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed_fixed_holiday(year, 7, 4),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed_fixed_holiday(year, 12, 25),
    }
    if year >= 2022:
        holidays.add(_observed_fixed_holiday(year, 6, 19))
    return holidays


def _is_market_trading_day(value: date) -> bool:
    if value.weekday() >= 5:
        return False
    holidays: set[date] = set()
    for year in range(value.year - 1, value.year + 2):
        holidays.update(_market_holidays_for_year(year))
    return value not in holidays


def _complete_market_data_date(value: date) -> bool:
    if value.weekday() not in {3, 4} or not _is_market_trading_day(value):
        return False
    week_start = monday_of_week(value)
    trading_days = [
        current
        for current in (week_start + timedelta(days=offset) for offset in range(5))
        if _is_market_trading_day(current)
    ]
    return bool(trading_days) and value == trading_days[-1]


def _complete_artifact_window(value: date) -> tuple[date, date]:
    week_start = monday_of_week(value)
    friday = week_start + timedelta(days=4)
    start = friday if value == friday else friday + timedelta(days=1)
    end = week_start + timedelta(days=7)
    return start, end


def classify_breakout_follow_pool(
    data_or_run_date: Any,
    *,
    file_date: Any | None = None,
) -> BreakoutFollowPoolKind:
    """Classify a BF run using the repository's single calendar contract.

    Without ``file_date``, ``data_or_run_date`` is the Asia/Shanghai scheduler
    date: Tue-Fri are midweek windows; Sat/Sun/Mon are complete/recovery
    windows.

    With ``file_date``, ``data_or_run_date`` is the latest actual US market
    data date. Friday data is complete only in its immediate recovery window.
    Thursday data becomes complete only when the following Friday is a market
    holiday and the pkl/file date is in the Sat/Sun/Mon recovery window. All
    other Mon-Thu market-data dates remain midweek.
    """
    value = _coerce_date(data_or_run_date)
    if file_date is None:
        return (
            BreakoutFollowPoolKind.COMPLETE
            if value.weekday() in {5, 6, 0}
            else BreakoutFollowPoolKind.MIDWEEK
        )

    if not _complete_market_data_date(value):
        return BreakoutFollowPoolKind.MIDWEEK

    artifact_date = _coerce_date(file_date)
    start, end = _complete_artifact_window(value)
    if start <= artifact_date <= end:
        return BreakoutFollowPoolKind.COMPLETE
    return BreakoutFollowPoolKind.MIDWEEK


def monday_of_week(value: date) -> date:
    return value - timedelta(days=value.weekday())


def complete_snapshot_week(complete_date: date) -> date:
    """Return the canonical Friday label for a COMPLETE market snapshot."""
    value = _coerce_date(complete_date)
    if not _complete_market_data_date(value):
        raise ValueError(f"Invalid complete snapshot date: {value.isoformat()}")
    return monday_of_week(value) + timedelta(days=4)


def complete_target_week(complete_date: date) -> date:
    """Map an already-identified COMPLETE snapshot to its review-week Monday.

    ``complete_date`` must be the actual last market-data date, not a
    scheduler/run date. Friday is the normal final date; Thursday is accepted
    only when the following Friday is a market holiday.
    """
    complete_week = complete_snapshot_week(complete_date)
    return complete_week + timedelta(days=3)


def is_valid_complete_baseline(complete_date: date, midweek_date: date) -> bool:
    """Return whether a complete snapshot is the baseline for a midweek one."""
    try:
        complete = _coerce_date(complete_date)
        midweek = _coerce_date(midweek_date)
        return (
            monday_of_week(midweek) == complete_target_week(complete)
            and midweek > complete
        )
    except (TypeError, ValueError):
        return False


__all__ = [
    "BreakoutFollowPoolKind",
    "classify_breakout_follow_pool",
    "complete_snapshot_week",
    "complete_target_week",
    "is_valid_complete_baseline",
    "monday_of_week",
]
