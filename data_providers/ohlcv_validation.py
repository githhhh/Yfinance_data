from __future__ import annotations

from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd


REQUIRED_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
MARKET_REFERENCE_SYMBOLS = ("^GSPC", "^IXIC", "^DJI")


class DataIntegrityError(ValueError):
    """Raised when downloaded market data is incomplete or internally invalid."""


class OHLCInconsistencyError(DataIntegrityError):
    """Raised only for a bar whose four prices violate OHLC ordering."""


def _format_rows(mask: pd.Series, index: pd.Index, limit: int = 5) -> str:
    bad_index = index[mask.to_numpy()]
    values = [str(value) for value in bad_index[:limit]]
    suffix = "" if len(bad_index) <= limit else f", ... (+{len(bad_index) - limit})"
    return ", ".join(values) + suffix


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
        examples = []
        for timestamp, row in numeric.loc[inconsistent, price_columns].head(5).iterrows():
            relations = [
                name
                for name, violated in (
                    ("High<Low", row["High"] < row["Low"]),
                    ("High<Open", row["High"] < row["Open"]),
                    ("High<Close", row["High"] < row["Close"]),
                    ("Low>Open", row["Low"] > row["Open"]),
                    ("Low>Close", row["Low"] > row["Close"]),
                )
                if violated
            ]
            values = ", ".join(f"{column}={row[column]}" for column in price_columns)
            examples.append(f"{timestamp} ({values}; {', '.join(relations)})")
        remaining = int(inconsistent.sum()) - len(examples)
        suffix = f"; ... (+{remaining} rows)" if remaining else ""
        raise OHLCInconsistencyError(
            f"{symbol}: inconsistent OHLC rows at {'; '.join(examples)}{suffix}"
        )


def validate_download_batch(
    stock_data: Mapping[str, pd.DataFrame],
    *,
    expected_symbols: Optional[Iterable[str]] = None,
) -> None:
    """Validate batch coverage and OHLCV integrity, not market-date freshness.

    Valid older histories remain available for analysis-stage sanitization.
    """
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
