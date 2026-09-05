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
    data date. Friday data is complete. Thursday data becomes complete only
    when the pkl/file date is Sat/Sun/Mon, which is the Friday-holiday recovery
    case. All other Mon-Thu market-data dates remain midweek.
    """
    value = _coerce_date(data_or_run_date)
    if file_date is None:
        return (
            BreakoutFollowPoolKind.COMPLETE
            if value.weekday() in {5, 6, 0}
            else BreakoutFollowPoolKind.MIDWEEK
        )

    artifact_date = _coerce_date(file_date)
    if value.weekday() == 4:
        return BreakoutFollowPoolKind.COMPLETE
    if value.weekday() == 3 and artifact_date.weekday() in {5, 6, 0}:
        return BreakoutFollowPoolKind.COMPLETE
    return BreakoutFollowPoolKind.MIDWEEK


def monday_of_week(value: date) -> date:
    return value - timedelta(days=value.weekday())


def complete_target_week(complete_date: date) -> date:
    """Map an already-identified COMPLETE snapshot to its review-week Monday.

    File identity / the classifier establishes that the snapshot is COMPLETE;
    this function therefore accepts Thursday as the last actual trading day of
    a Friday-holiday week. Tuesday/Wednesday remain invalid complete dates.
    """
    value = _coerce_date(complete_date)
    weekday = value.weekday()
    if weekday == 0:
        return value
    if weekday in {3, 4, 5, 6}:
        return value + timedelta(days=7 - weekday)
    raise ValueError(f"Invalid complete snapshot date: {value.isoformat()}")


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
    "complete_target_week",
    "is_valid_complete_baseline",
    "monday_of_week",
]
