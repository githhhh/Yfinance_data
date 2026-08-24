#!/usr/bin/env python
"""Rebuild historical replay signal EPS using the reviewed SEC/Yahoo logic."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eps_pit.providers.sec_yahoo_provider import SECYahooEPSProvider, calculate_latest_eps_yoy


EPS_COLUMNS = [
    "eps_yoy_growth",
    "eps_yoy_growth_source",
]

PIT_COLUMNS = [
    "snapshot_date",
    "code",
    "eps_yoy_growth",
    "source",
    "effective_date",
    "current_eps",
    "prior_year_eps",
    "current_period",
    "prior_year_period",
    "status",
]


def _is_truthy(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y"}


def _normalize_code(value: object) -> str:
    return str(value or "").strip().upper().replace(".", "-")


def _snapshot_from_pool(path: Path, df: pd.DataFrame) -> str:
    if "snapshot_date" in df.columns:
        dates = df["snapshot_date"].dropna().astype(str).str.strip().str[:10]
        dates = dates[dates.ne("")]
        if not dates.empty:
            return str(dates.iloc[0])
    return path.parent.name


def _pool_files(pool_dir: Path) -> list[Path]:
    return sorted(pool_dir.glob("????-??-??/breakout_follow_pool.csv"))


def _pit_row(snapshot_date: str, code: str, record: dict[str, Any] | None) -> dict[str, Any]:
    row = {column: pd.NA for column in PIT_COLUMNS}
    row["snapshot_date"] = snapshot_date
    row["code"] = code
    if record is None:
        row["status"] = "unresolved"
        return row
    row.update(
        {
            "eps_yoy_growth": record.get("eps_yoy_growth"),
            "source": record.get("source"),
            "effective_date": record.get("effective_date"),
            "current_eps": record.get("current_eps"),
            "prior_year_eps": record.get("prior_year_eps"),
            "current_period": record.get("current_period"),
            "prior_year_period": record.get("prior_year_period"),
            "status": "filled",
        }
    )
    return row


def rebuild_replay_signal_eps(
    pool_dir: str | Path = "backtest/ibd_skill_replay_pools",
    *,
    provider: Any | None = None,
    workers: int = 8,
) -> dict[str, Any]:
    """Force-rebuild signal-row EPS values in historical replay pools."""
    root = Path(pool_dir)
    if provider is None:
        provider = SECYahooEPSProvider()

    pools: list[dict[str, Any]] = []
    signal_entries: list[dict[str, Any]] = []
    snapshots_by_code: dict[str, set[str]] = defaultdict(set)

    for pool_path in _pool_files(root):
        df = pd.read_csv(pool_path, dtype={"code": str}, encoding="utf-8-sig")
        snapshot_date = _snapshot_from_pool(pool_path, df)
        for column in EPS_COLUMNS:
            if column not in df.columns:
                df[column] = pd.NA
        df["eps_yoy_growth_source"] = df["eps_yoy_growth_source"].astype("object")

        pool_info = {
            "path": pool_path,
            "df": df,
            "snapshot_date": snapshot_date,
            "signal_indices": [],
        }
        pools.append(pool_info)

        if "signal" not in df.columns or "code" not in df.columns:
            continue

        signal_mask = df["signal"].map(_is_truthy)
        for idx in list(df.index[signal_mask]):
            code = _normalize_code(df.at[idx, "code"])
            pool_info["signal_indices"].append(idx)
            signal_entries.append(
                {
                    "snapshot_date": snapshot_date,
                    "code": code,
                    "pool_info": pool_info,
                    "idx": idx,
                }
            )
            if code:
                snapshots_by_code[code].add(snapshot_date)

    records_by_key = _fetch_eps_records(provider, snapshots_by_code, workers=workers)
    pit_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for entry in signal_entries:
        snapshot_date = entry["snapshot_date"]
        code = entry["code"]
        pool_info = entry["pool_info"]
        idx = entry["idx"]
        df = pool_info["df"]
        record = records_by_key.get((snapshot_date, code))
        pit_rows.append(_pit_row(snapshot_date, code, record))

        if record is None:
            df.at[idx, "eps_yoy_growth"] = pd.NA
            df.at[idx, "eps_yoy_growth_source"] = pd.NA
            continue

        df.at[idx, "eps_yoy_growth"] = record.get("eps_yoy_growth")
        df.at[idx, "eps_yoy_growth_source"] = record.get("source")

    for pool_info in pools:
        df = pool_info["df"]
        pool_path = pool_info["path"]
        snapshot_date = pool_info["snapshot_date"]
        signal_indices = pool_info["signal_indices"]
        if "signal" not in df.columns or "code" not in df.columns:
            summary_rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "pool_path": str(pool_path),
                    "signal_rows": 0,
                    "filled_rows": 0,
                    "unresolved_rows": 0,
                    "status": "skipped_missing_signal_or_code",
                }
            )
            continue

        signal_eps = pd.to_numeric(df.loc[signal_indices, "eps_yoy_growth"], errors="coerce")
        filled = int(signal_eps.notna().sum())
        unresolved = int(signal_eps.isna().sum())

        df.to_csv(pool_path, index=False, encoding="utf-8-sig")
        summary_rows.append(
            {
                "snapshot_date": snapshot_date,
                "pool_path": str(pool_path),
                "signal_rows": len(signal_indices),
                "filled_rows": filled,
                "unresolved_rows": unresolved,
                "status": "rebuilt",
            }
        )

    pit_df = pd.DataFrame(pit_rows, columns=PIT_COLUMNS)
    pit_df.to_csv(root / "signal_eps_pit.csv", index=False)

    audit_dir = root / "eps_signal_refresh_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(audit_dir / "summary.csv", index=False)

    summary = {
        "pool_dir": str(root),
        "pool_files": len(summary_rows),
        "signal_rows": int(summary_df["signal_rows"].sum()) if not summary_df.empty else 0,
        "filled_rows": int(summary_df["filled_rows"].sum()) if not summary_df.empty else 0,
        "unresolved_rows": int(summary_df["unresolved_rows"].sum()) if not summary_df.empty else 0,
        "signal_eps_pit_path": str(root / "signal_eps_pit.csv"),
        "audit_summary_path": str(audit_dir / "summary.csv"),
    }
    (audit_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _fetch_eps_records(
    provider: Any,
    snapshots_by_code: dict[str, set[str]],
    *,
    workers: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    if isinstance(provider, SECYahooEPSProvider):
        return _fetch_sec_yahoo_records(provider, snapshots_by_code, workers=workers)

    records: dict[tuple[str, str], dict[str, Any]] = {}
    for code, snapshots in sorted(snapshots_by_code.items()):
        for snapshot_date in sorted(snapshots):
            record = provider.fetch_eps_yoy(code, snapshot_date)
            if record is not None:
                records[(snapshot_date, code)] = record
    return records


def _fetch_sec_yahoo_records(
    provider: SECYahooEPSProvider,
    snapshots_by_code: dict[str, set[str]],
    *,
    workers: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}

    def fetch_code(code: str) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        try:
            sec_records = provider.sec.fetch_quarterly_history(code)
        except Exception:
            sec_records = []
        try:
            yahoo_records = provider.yahoo.fetch_quarterly_history(code)
        except Exception:
            yahoo_records = []
        return code, sec_records, yahoo_records

    max_workers = max(1, int(workers or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_code, code): code for code in sorted(snapshots_by_code)}
        for future in as_completed(futures):
            code, sec_records, yahoo_records = future.result()
            for snapshot_date in sorted(snapshots_by_code[code]):
                record = calculate_latest_eps_yoy(sec_records, snapshot_date)
                if record is None:
                    record = calculate_latest_eps_yoy(yahoo_records, snapshot_date)
                if record is not None:
                    records[(snapshot_date, code)] = record
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild historical replay signal EPS with SEC/Yahoo data.")
    parser.add_argument("--pool-dir", default="backtest/ibd_skill_replay_pools")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    summary = rebuild_replay_signal_eps(args.pool_dir, workers=args.workers)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
