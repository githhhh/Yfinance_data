from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)


REQUIRED_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
MARKET_REFERENCE_SYMBOLS = ("^GSPC", "^IXIC", "^DJI")
US_EASTERN = ZoneInfo("America/New_York")


class DataIntegrityError(ValueError):
    """Raised when downloaded market data is incomplete or internally invalid."""


class _NYSEHolidayCalendar(AbstractHolidayCalendar):
    """Regular full-day NYSE holidays used to determine the latest completed session."""

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "Juneteenth National Independence Day",
            month=6,
            day=19,
            start_date="2022-01-01",
            observance=nearest_workday,
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def _format_rows(mask: pd.Series, index: pd.Index, limit: int = 5) -> str:
    bad_index = index[mask.to_numpy()]
    values = [str(value) for value in bad_index[:limit]]
    suffix = "" if len(bad_index) <= limit else f", ... (+{len(bad_index) - limit})"
    return ", ".join(values) + suffix


def _nyse_holidays_around(day) -> set:
    start = pd.Timestamp(day) - pd.Timedelta(days=370)
    end = pd.Timestamp(day) + pd.Timedelta(days=370)
    return {
        timestamp.date()
        for timestamp in _NYSEHolidayCalendar().holidays(start=start, end=end)
    }


def _is_regular_us_equity_session(day) -> bool:
    return day.weekday() < 5 and day not in _nyse_holidays_around(day)


def expected_latest_us_session(
    now: Optional[datetime] = None,
    *,
    close_buffer_minutes: int = 30,
):
    """Return the latest regular US equity session that should be fully closed.

    A small post-close buffer prevents publishing a daily bar while Yahoo may still
    be finalizing the just-closed session.
    """
    current = now or datetime.now(tz=US_EASTERN)
    if current.tzinfo is None:
        current = current.replace(tzinfo=US_EASTERN)
    else:
        current = current.astimezone(US_EASTERN)

    candidate = current.date()
    cutoff_minutes = 16 * 60 + close_buffer_minutes
    current_minutes = current.hour * 60 + current.minute

    if not _is_regular_us_equity_session(candidate) or current_minutes < cutoff_minutes:
        candidate -= timedelta(days=1)

    while not _is_regular_us_equity_session(candidate):
        candidate -= timedelta(days=1)
    return candidate


def validate_ohlcv_frame(symbol: str, data: pd.DataFrame) -> None:
    """Fail closed unless every stored OHLCV row is complete and numerically sane."""
    if not isinstance(data, pd.DataFrame):
        raise DataIntegrityError(f"{symbol}: expected DataFrame, got {type(data).__name__}")
    if data.empty:
        raise DataIntegrityError(f"{symbol}: empty price history")

    missing_columns = [col for col in REQUIRED_OHLCV_COLUMNS if col not in data.columns]
    if missing_columns:
        raise DataIntegrityError(f"{symbol}: missing required columns {missing_columns}")

    if data.index.has_duplicates:
        duplicates = data.index[data.index.duplicated()].unique()
        preview = ", ".join(str(value) for value in duplicates[:5])
        raise DataIntegrityError(f"{symbol}: duplicate timestamps: {preview}")

    parsed_index = pd.to_datetime(data.index, errors="coerce")
    if pd.isna(parsed_index).any():
        raise DataIntegrityError(f"{symbol}: invalid/NaT timestamps in price history")
    if not parsed_index.is_monotonic_increasing:
        raise DataIntegrityError(f"{symbol}: timestamps are not monotonic increasing")

    required = data.loc[:, list(REQUIRED_OHLCV_COLUMNS)]
    numeric = required.apply(pd.to_numeric, errors="coerce")

    null_mask = numeric.isna().any(axis=1)
    if null_mask.any():
        raise DataIntegrityError(
            f"{symbol}: null/non-numeric OHLCV rows at {_format_rows(null_mask, data.index)}"
        )

    finite_mask = pd.Series(
        np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1),
        index=data.index,
    )
    if not finite_mask.all():
        bad_mask = ~finite_mask
        raise DataIntegrityError(
            f"{symbol}: non-finite OHLCV rows at {_format_rows(bad_mask, data.index)}"
        )

    price_columns = ["Open", "High", "Low", "Close"]
    nonpositive_price = (numeric[price_columns] <= 0).any(axis=1)
    if nonpositive_price.any():
        raise DataIntegrityError(
            f"{symbol}: non-positive price rows at "
            f"{_format_rows(nonpositive_price, data.index)}"
        )

    negative_volume = numeric["Volume"] < 0
    if negative_volume.any():
        raise DataIntegrityError(
            f"{symbol}: negative volume rows at {_format_rows(negative_volume, data.index)}"
        )

    inconsistent = (
        (numeric["High"] < numeric["Low"])
        | (numeric["High"] < numeric["Open"])
        | (numeric["High"] < numeric["Close"])
        | (numeric["Low"] > numeric["Open"])
        | (numeric["Low"] > numeric["Close"])
    )
    if inconsistent.any():
        raise DataIntegrityError(
            f"{symbol}: inconsistent OHLC rows at {_format_rows(inconsistent, data.index)}"
        )


def latest_bar_date(data: pd.DataFrame):
    """Return the calendar date encoded by the newest bar without changing its timezone."""
    value = pd.Timestamp(data.index[-1])
    if pd.isna(value):
        raise DataIntegrityError("latest timestamp is NaT")
    return value.date()


def reference_latest_dates(
    stock_data: Mapping[str, pd.DataFrame],
    reference_symbols: Sequence[str] = MARKET_REFERENCE_SYMBOLS,
) -> Dict[str, object]:
    if not all(symbol in stock_data for symbol in reference_symbols):
        return {}
    return {
        symbol: latest_bar_date(stock_data[symbol]) for symbol in reference_symbols
    }


def find_latest_bar_mismatches(
    stock_data: Mapping[str, pd.DataFrame],
    reference_symbols: Sequence[str] = MARKET_REFERENCE_SYMBOLS,
) -> Dict[str, object]:
    """Return symbols lagging a consensus latest market-reference bar.

    If references are missing or disagree, there is no trustworthy target date;
    callers must resolve/fail the reference set first instead of retrying the world.
    """
    reference_dates = reference_latest_dates(stock_data, reference_symbols)
    if not reference_dates or len(set(reference_dates.values())) != 1:
        return {}

    expected_date = next(iter(reference_dates.values()))
    return {
        symbol: latest_bar_date(data)
        for symbol, data in stock_data.items()
        if latest_bar_date(data) != expected_date
    }


def validate_download_batch(
    stock_data: Mapping[str, pd.DataFrame],
    *,
    expected_symbols: Optional[Iterable[str]] = None,
    interval: Optional[str] = None,
    reference_symbols: Sequence[str] = MARKET_REFERENCE_SYMBOLS,
    now: Optional[datetime] = None,
) -> None:
    """Validate completeness of a batch before it is allowed to reach a PKL."""
    if not isinstance(stock_data, Mapping) or not stock_data:
        raise DataIntegrityError("download batch is empty")

    expected = (
        list(dict.fromkeys(expected_symbols))
        if expected_symbols is not None
        else []
    )
    if expected:
        expected_set = set(expected)
        actual_set = set(stock_data)
        missing = sorted(expected_set - actual_set)
        unexpected = sorted(actual_set - expected_set)
        if missing:
            raise DataIntegrityError(
                f"download batch missing {len(missing)} expected symbols: {missing[:20]}"
            )
        if unexpected:
            raise DataIntegrityError(
                f"download batch contains unexpected symbols: {unexpected[:20]}"
            )

    frame_errors = []
    for symbol, data in stock_data.items():
        try:
            validate_ohlcv_frame(symbol, data)
        except DataIntegrityError as exc:
            frame_errors.append(str(exc))

    if frame_errors:
        preview = "; ".join(frame_errors[:10])
        extra = "" if len(frame_errors) <= 10 else f"; ... (+{len(frame_errors) - 10})"
        raise DataIntegrityError(f"invalid OHLCV data: {preview}{extra}")

    if interval in {"1d", "1wk"}:
        reference_dates = reference_latest_dates(stock_data, reference_symbols)
        if reference_dates:
            if len(set(reference_dates.values())) != 1:
                raise DataIntegrityError(
                    f"market reference latest-bar dates disagree: {reference_dates}"
                )

            reference_date = next(iter(reference_dates.values()))
            if interval == "1d":
                expected_session = expected_latest_us_session(now)
                if reference_date != expected_session:
                    raise DataIntegrityError(
                        "market references do not reach the latest completed US "
                        f"session {expected_session}: {reference_dates}"
                    )

            mismatches = find_latest_bar_mismatches(stock_data, reference_symbols)
            if mismatches:
                preview = list(mismatches.items())[:20]
                raise DataIntegrityError(
                    f"{len(mismatches)} symbols do not reach latest market bar "
                    f"{reference_date}: {preview}"
                )
