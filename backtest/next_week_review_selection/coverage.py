from __future__ import annotations

import pandas as pd

from .labels import HORIZONS


def build_price_path_audits(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Produce explicit coverage diagnostics for the historical price cache."""
    return {
        "price_path_coverage_summary.csv": _summary(panel),
        "price_path_coverage_by_week.csv": _grouped(panel, "snapshot_date"),
        "price_path_coverage_by_status.csv": _grouped(panel, "ibd_entry_status"),
        "price_path_coverage_by_setup.csv": _grouped(panel, "ibd_candidate_rule"),
        "price_path_coverage_by_signal_source.csv": _grouped(panel, "signal_source"),
        "price_path_coverage_by_ticker.csv": _grouped(panel, "code"),
        "price_path_missing_1w_cases.csv": _missing_1w_cases(panel),
    }


def _summary(panel: pd.DataFrame) -> pd.DataFrame:
    row = _coverage_row(panel)
    states = (
        panel["price_path_state"].fillna("").astype(str).value_counts()
        if "price_path_state" in panel.columns
        else pd.Series(dtype=int)
    )
    for state, count in states.items():
        if state:
            row[f"state_{state}"] = int(count)
    return pd.DataFrame([row])


def _grouped(panel: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in panel.columns:
        return pd.DataFrame()
    rows = []
    work = panel.copy()
    work[column] = work[column].fillna("<MISSING>").astype(str)
    for value, group in work.groupby(column, dropna=False, sort=True):
        row = {column: value}
        row.update(_coverage_row(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _coverage_row(frame: pd.DataFrame) -> dict[str, float | int]:
    total = len(frame)
    row: dict[str, float | int] = {
        "rows": total,
        "symbol_found_count": int(
            frame.get("price_cache_symbol_found", pd.Series(False, index=frame.index))
            .fillna(False)
            .astype(bool)
            .sum()
        ),
    }
    row["symbol_found_rate"] = (
        row["symbol_found_count"] / total if total else float("nan")
    )
    for horizon in HORIZONS:
        complete = int(frame[f"forward_{horizon}_censored"].eq(False).sum())
        row[f"complete_{horizon}_count"] = complete
        row[f"complete_{horizon}_rate"] = (
            complete / total if total else float("nan")
        )
    return row


def _missing_1w_cases(panel: pd.DataFrame) -> pd.DataFrame:
    missing = panel[panel["forward_1w_censored"].eq(True)].copy()
    columns = [
        "snapshot_date",
        "code",
        "ibd_entry_status",
        "ibd_candidate_rule",
        "signal_source",
        "price_cache_symbol_found",
        "price_path_state",
        "price_path_available_sessions",
        "price_cache_first_date",
        "price_cache_last_date",
        "price_path_first_forward_date",
        "price_path_last_forward_date",
    ]
    return missing[
        [column for column in columns if column in missing.columns]
    ].sort_values(["snapshot_date", "code"], kind="mergesort")
