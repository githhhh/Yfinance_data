from __future__ import annotations

import hashlib
import math
import pickle
from pathlib import Path

import pandas as pd


ZERO_TOL = 1e-12


def to_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def is_positive(value: object, *, tol: float = ZERO_TOL) -> bool:
    number = to_float(value)
    return number is not None and number > tol


def is_nonnegative(value: object, *, tol: float = ZERO_TOL) -> bool:
    number = to_float(value)
    return number is not None and number >= -tol


def is_nonpositive(value: object, *, tol: float = ZERO_TOL) -> bool:
    number = to_float(value)
    return number is not None and number <= tol


def to_bool(value: object) -> bool | None:
    if value is True or value is False:
        return bool(value)
    numeric = to_float(value)
    if numeric == 1.0:
        return True
    if numeric == 0.0:
        return False
    text = str(value).strip().lower()
    if text in {"true", "yes", "y"}:
        return True
    if text in {"false", "no", "n"}:
        return False
    return None


def pct(value: float | None, base: float | None) -> float | None:
    if value is None or base is None or base == 0:
        return None
    return (value / base - 1.0) * 100.0


def parse_date(value: object) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.tz_localize(None) if getattr(ts, "tz", None) is not None else ts


def fmt_date(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(pd.Timestamp(value).date())


def normalize_bars(frame: pd.DataFrame | dict | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    if isinstance(frame, dict) and {"index", "columns", "data"}.issubset(frame):
        frame = pd.DataFrame(frame["data"], columns=frame["columns"], index=frame["index"])
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    bars = frame.copy()
    if "Date" in bars.columns:
        index = pd.to_datetime(bars.pop("Date"))
    else:
        index = pd.to_datetime(bars.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    bars.index = index
    bars = bars.rename(columns={column: str(column).title() for column in bars.columns})
    required = ["Open", "High", "Low", "Close"]
    if any(column not in bars.columns for column in required):
        return pd.DataFrame()
    for column in required:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    return bars.dropna(subset=required).sort_index()


def load_price_cache(path: Path) -> dict[str, pd.DataFrame]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    return {str(code): normalize_bars(value) for code, value in raw.items()}


def load_pools(pool_root: Path) -> list[tuple[str, pd.DataFrame, Path]]:
    pools = []
    for path in sorted(pool_root.glob("*/breakout_follow_pool.csv")):
        snapshot = path.parent.name
        frame = pd.read_csv(path)
        frame["snapshot_date"] = snapshot
        pools.append((snapshot, frame, path))
    return pools


def normalize_eps_pit(frame: pd.DataFrame) -> pd.DataFrame:
    """Use only resolved records that were effective on/before the snapshot."""
    rows = []
    for _, source in frame.iterrows():
        row = source.to_dict()
        snapshot = parse_date(row.get("snapshot_date"))
        effective = parse_date(row.get("effective_date"))
        status = str(row.get("status", "") or "").strip().lower()
        value = to_float(row.get("eps_yoy_growth"))
        verified = (
            status == "resolved"
            and value is not None
            and snapshot is not None
            and effective is not None
            and effective.normalize() <= snapshot.normalize()
        )
        row["pit_eps_state"] = "VERIFIED" if verified else "UNKNOWN"
        row["pit_eps_yoy_growth"] = value if verified else None
        rows.append(row)
    return pd.DataFrame(rows)


def content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
