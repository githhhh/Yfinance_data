"""Explicit inputs for the market-data download universe.

These are strategy inputs that must be present in ``results_pkl``.  Published
artifacts such as EPS PIT caches are deliberately not inferred from ``us/``.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd


DOWNLOAD_UNIVERSE_SOURCE_FILES = (
    "us/52wk_new_high_results.csv",
    "us/breakout_follow_pool.csv",
    "us/breakout_follow_pool_midweek.csv",
    "us/eps_growth_screener_results.csv",
    "us/weekly_vol_screener_results.csv",
)


def _normalize_code(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    code = str(value).strip()
    if not code or code.lower() == "nan":
        return None
    return code.replace(".", "-")


def build_download_universe(*, data_root: str | Path = ".") -> list[str]:
    """Return the deduplicated, deterministic market-data input universe."""
    root = Path(data_root)
    tickers: set[str] = set()

    for relative_path in DOWNLOAD_UNIVERSE_SOURCE_FILES:
        source_path = root / relative_path
        if not source_path.exists():
            logging.warning("Download universe source missing: %s", source_path)
            continue
        try:
            source = pd.read_csv(source_path, dtype={"code": str})
        except Exception as exc:
            logging.warning("Download universe source unreadable: %s (%s)", source_path, exc)
            continue
        if "code" not in source.columns:
            logging.warning("Download universe source has no code column: %s", source_path)
            continue

        source_codes = {
            code
            for code in (_normalize_code(value) for value in source["code"])
            if code is not None
        }
        tickers.update(source_codes)
        logging.info("Download universe source %s: %s codes", relative_path, len(source_codes))

    result = sorted(tickers)
    logging.info("Download universe total: %s codes", len(result))
    return result


def pick_active_weekend_signal_codes(
    *, data_root: str | Path = ".", review_date: date | None = None
) -> list[str]:
    """Protect complete-week signals during their *next* review week only.

    This is a final download eligibility pick, not a scan, provider retry or
    data-quality policy. The ordinary five-source universe is unchanged.
    """
    from bf_snapshot import complete_target_week, monday_of_week

    today = review_date or datetime.now(ZoneInfo("Asia/Shanghai")).date()
    if today.weekday() > 4:
        return []
    path = Path(data_root) / "us" / "breakout_follow_pool.csv"
    try:
        frame = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig")
        required = {"code", "signal", "snapshot_date"}
        if not required.issubset(frame.columns):
            raise ValueError(f"missing columns: {sorted(required - set(frame.columns))}")
        dates = pd.to_datetime(frame["snapshot_date"], errors="coerce").dt.date.dropna().unique()
        if len(dates) != 1 or complete_target_week(dates[0]) != monday_of_week(today):
            return []
        mask = frame["signal"].astype(str).str.strip().str.lower().isin(("true", "1"))
        return sorted({
            code for code in ( _normalize_code(v) for v in frame.loc[mask, "code"] )
            if code is not None
        })
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logging.warning("Weekend Signal Final Pick unavailable: %s", exc)
        return []
