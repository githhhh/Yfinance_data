"""Explicit inputs for the market-data download universe.

These are strategy inputs that must be present in ``results_pkl``.  Published
artifacts such as EPS PIT caches are deliberately not inferred from ``us/``.
"""

from __future__ import annotations

import logging
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
