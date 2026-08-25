"""Review Universe extraction and event table construction across historical Replay Pools."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def is_signal_true(value: object) -> bool:
    """Check if signal is explicitly True (supports bool, 1, 'true', '1.0')."""
    if value is None:
        return False
    if isinstance(value, (bool, int)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"true", "1", "1.0"}


def is_non_empty_rule(value: object) -> bool:
    """Check if ibd_candidate_rule is non-empty string."""
    if value is None or pd.isna(value):
        return False
    return bool(str(value).strip())


def scan_replay_pools(
    pools_root: Path | str = "backtest/ibd_skill_replay_pools",
    pattern: str = "*/breakout_follow_pool.csv",
) -> list[Path]:
    """Find all replay pool CSVs sorted by snapshot date."""
    root = Path(pools_root)
    pool_paths = sorted(
        root.glob(pattern),
        key=lambda p: p.parent.name,
    )
    return pool_paths


def build_review_universe_events(
    pool_paths: Sequence[Path] | None = None,
    pools_root: Path | str = "backtest/ibd_skill_replay_pools",
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Scan all historical pools, extract Review Universe events, and build master event table.
    
    Review Universe condition:
        signal == True AND ibd_candidate_rule is non-empty.
    """
    if pool_paths is None:
        pool_paths = scan_replay_pools(pools_root)

    events_list: list[pd.DataFrame] = []
    for path in pool_paths:
        snapshot_date = path.parent.name
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")
            continue

        if "code" not in df.columns:
            logger.warning(f"Missing 'code' column in {path}")
            continue

        # Standardize snapshot_date column
        df["snapshot_date"] = snapshot_date
        df["code"] = df["code"].astype(str).str.strip().str.upper()
        df["original_row_idx"] = df.index.astype(int)

        # Filter Review Universe
        signal_mask = df["signal"].apply(is_signal_true)
        rule_mask = df["ibd_candidate_rule"].apply(is_non_empty_rule)
        review_df = df[signal_mask & rule_mask].copy()

        if not review_df.empty:
            review_df["event_id"] = (
                review_df["snapshot_date"] + "_" + review_df["code"] + "_" + review_df["original_row_idx"].astype(str)
            )
            events_list.append(review_df)

    if not events_list:
        empty_df = pd.DataFrame()
        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            empty_df.to_parquet(out_p, index=False)
        return empty_df

    events_df = pd.concat(events_list, ignore_index=True)
    # Ensure event_id is unique
    events_df = events_df.drop_duplicates(subset=["event_id"]).reset_index(drop=True)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Convert object columns to string where appropriate for clean parquet serialization
        events_df.to_parquet(out_p, index=False, engine="pyarrow")
        logger.info(f"Saved {len(events_df)} review universe events to {out_p}")

    return events_df


def summarize_universe(events_df: pd.DataFrame) -> dict:
    """Generate summary dictionary for review universe events."""
    if events_df.empty:
        return {
            "total_events": 0,
            "unique_snapshots": 0,
            "unique_codes": 0,
            "earliest_snapshot": "",
            "latest_snapshot": "",
            "events_per_snapshot": {},
        }

    snapshots = sorted(events_df["snapshot_date"].unique().tolist())
    unique_codes = sorted(events_df["code"].unique().tolist())
    counts_by_snap = events_df.groupby("snapshot_date").size().to_dict()

    return {
        "total_events": int(len(events_df)),
        "unique_snapshots": int(len(snapshots)),
        "unique_codes": int(len(unique_codes)),
        "earliest_snapshot": snapshots[0] if snapshots else "",
        "latest_snapshot": snapshots[-1] if snapshots else "",
        "events_per_snapshot": counts_by_snap,
    }
