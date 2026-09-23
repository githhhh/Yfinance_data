"""Explicit inputs for the market-data download universe.

These are strategy inputs that must be present in ``results_pkl``.  Published
artifacts such as EPS PIT caches are deliberately not inferred from ``us/``.
"""

from __future__ import annotations

import json
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
IBD_DOUBLE_BOTTOM_TRACKING_STATE_FILE = "us/ibd_double_bottom_snapshot_state.json"


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


def _load_current_double_bottom_tracking_source(root: Path) -> pd.DataFrame | None:
    state_path = root / IBD_DOUBLE_BOTTOM_TRACKING_STATE_FILE
    snapshot_paths = {
        "complete": root / IBD_DOUBLE_BOTTOM_TRACKING_SOURCE_FILES[0],
        "midweek": root / IBD_DOUBLE_BOTTOM_TRACKING_SOURCE_FILES[1],
    }
    existing_snapshots = [path for path in snapshot_paths.values() if path.exists()]

    if not state_path.exists():
        if existing_snapshots:
            raise RuntimeError(
                "IBD Double Bottom snapshot state missing while snapshot CSV exists"
            )
        return None

    try:
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    except Exception as exc:
        raise RuntimeError(
            f"IBD Double Bottom snapshot state unreadable: {state_path}"
        ) from exc

    kind = _text(state.get("kind"))
    snapshot_date = _text(state.get("snapshot_date"))
    if kind not in snapshot_paths:
        raise ValueError(f"IBD Double Bottom snapshot state kind invalid: {kind}")
    try:
        normalized_date = pd.Timestamp(snapshot_date).strftime("%Y-%m-%d")
    except Exception as exc:
        raise ValueError(
            f"IBD Double Bottom snapshot state date invalid: {snapshot_date}"
        ) from exc
    if normalized_date != snapshot_date:
        raise ValueError(
            f"IBD Double Bottom snapshot state date invalid: {snapshot_date}"
        )

    source_path = snapshot_paths[kind]
    if not source_path.exists():
        raise RuntimeError(
            f"IBD Double Bottom current snapshot missing: {source_path}"
        )
    try:
        source = pd.read_csv(source_path, dtype={"code": str})
    except Exception as exc:
        raise RuntimeError(
            f"IBD Double Bottom current snapshot unreadable: {source_path}"
        ) from exc

    required = {"code", "snapshot_date", "detection_path", "signal_type"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(
            f"IBD Double Bottom current snapshot missing columns: {sorted(missing)}"
        )
    if not source.empty:
        dates = source["snapshot_date"].dropna().astype(str).str.strip().str[:10]
        dates = dates[dates.ne("")]
        if (
            len(dates) != len(source)
            or dates.nunique() != 1
            or dates.iloc[0] != snapshot_date
        ):
            raise ValueError(
                "IBD Double Bottom current snapshot date does not match state"
            )

    logging.info(
        "Download universe Double Bottom current snapshot %s: %s",
        kind,
        snapshot_date,
    )
    return source


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

    double_bottom_source = _load_current_double_bottom_tracking_source(root)
    if double_bottom_source is not None:
        source_codes = _double_bottom_tracking_codes(double_bottom_source)
        tickers.update(source_codes)
        logging.info(
            "Download universe Double Bottom active tracking: %s codes",
            len(source_codes),
        )

    result = sorted(tickers)
    logging.info("Download universe total: %s codes", len(result))
    return result
