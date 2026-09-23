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

IBD_DOUBLE_BOTTOM_TRACKING_SOURCE_FILES = (
    "us/ibd_double_bottom_snapshot.csv",
    "us/ibd_double_bottom_snapshot_midweek.csv",
)


def _normalize_code(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    code = str(value).strip()
    if not code or code.lower() == "nan":
        return None
    return code.replace(".", "-")


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _double_bottom_tracking_codes(source: pd.DataFrame) -> set[str]:
    """Project only Double Bottom states that still require future OHLCV.

    Path A has no persisted terminal row in the shared snapshot contract, so
    any row that still contains the Path A detector remains tracked. Path B
    explicitly records Retired terminal state and must stop contributing to
    the download universe once that state is reached.
    """
    required = {"code", "detection_path", "signal_type"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(
            f"IBD Double Bottom snapshot missing columns: {sorted(missing)}"
        )

    codes: set[str] = set()
    for _, row in source.iterrows():
        code = _normalize_code(row.get("code"))
        if code is None:
            continue

        paths = {
            item.strip()
            for item in _text(row.get("detection_path")).split("+")
            if item.strip()
        }
        if "double_bottom" in paths:
            codes.add(code)
            continue
        if "ibd_double_bottom" not in paths:
            continue

        path_b_signal = (
            _text(row.get("ibd_double_bottom_signal_type"))
            or _text(row.get("signal_type"))
        )
        retired_reason = (
            _text(row.get("ibd_double_bottom_retired_reason"))
            or _text(row.get("retired_reason"))
        )
        if path_b_signal == "Retired" or retired_reason:
            continue
        codes.add(code)

    return codes


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

    for relative_path in IBD_DOUBLE_BOTTOM_TRACKING_SOURCE_FILES:
        source_path = root / relative_path
        if not source_path.exists():
            logging.warning("Download universe source missing: %s", source_path)
            continue
        try:
            source = pd.read_csv(source_path, dtype={"code": str})
            source_codes = _double_bottom_tracking_codes(source)
        except Exception as exc:
            logging.warning(
                "Double Bottom tracking source unreadable: %s (%s)",
                source_path,
                exc,
            )
            continue
        tickers.update(source_codes)
        logging.info(
            "Download universe Double Bottom source %s: %s active codes",
            relative_path,
            len(source_codes),
        )

    result = sorted(tickers)
    logging.info("Download universe total: %s codes", len(result))
    return result
