from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def to_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "<na>", "nat"}:
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def to_bool(value: object) -> bool | None:
    if value is True or value is False:
        return bool(value)
    numeric = to_float(value)
    if numeric == 1.0:
        return True
    if numeric == 0.0:
        return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def pct(value: float | None, base: float | None) -> float | None:
    if value is None or base is None or base == 0:
        return None
    return round((value / base - 1.0) * 100.0, 6)


def parse_date(value: object) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "nat"}:
        return None
    try:
        ts = pd.Timestamp(text)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.tz_localize(None) if getattr(ts, "tz", None) is not None else ts


def fmt_date(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(pd.Timestamp(value).date())


def normalize_bars(frame: pd.DataFrame | dict[str, Any] | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    if isinstance(frame, dict) and {"index", "columns", "data"}.issubset(frame):
        frame = pd.DataFrame(frame["data"], columns=frame["columns"], index=frame["index"])
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    bars = frame.copy()
    if "Date" in bars.columns:
        index = pd.to_datetime(bars["Date"])
        bars = bars.drop(columns=["Date"])
    else:
        index = pd.to_datetime(bars.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    bars.index = index
    bars = bars.rename(columns={column: str(column).title() for column in bars.columns})
    for column in ["Open", "High", "Low", "Close"]:
        if column not in bars.columns:
            return pd.DataFrame()
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    return bars.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()


def next_bar_after(bars: pd.DataFrame, snapshot_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Series] | None:
    if bars.empty:
        return None
    window = bars[bars.index > snapshot_date]
    if window.empty:
        return None
    return pd.Timestamp(window.index[0]), window.iloc[0]


def week_start(value: pd.Timestamp) -> pd.Timestamp:
    value = pd.Timestamp(value).tz_localize(None) if getattr(pd.Timestamp(value), "tz", None) is not None else pd.Timestamp(value)
    return value.normalize() - pd.Timedelta(days=value.weekday())


def breakout_week_friday(anchor: pd.Timestamp, week_number: int) -> pd.Timestamp:
    return week_start(anchor) + pd.Timedelta(days=(week_number - 1) * 7 + 4)


def content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
