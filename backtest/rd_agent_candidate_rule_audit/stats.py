from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RollingSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_rolling_splits(
    frame: pd.DataFrame,
    *,
    test_weeks: int = 4,
    embargo_weeks: int = 8,
    min_train_weeks: int = 8,
) -> list[RollingSplit]:
    weeks = sorted(pd.to_datetime(frame["snapshot_date"]).dropna().dt.normalize().unique())
    splits: list[RollingSplit] = []
    start = min_train_weeks + embargo_weeks
    for test_start_idx in range(start, len(weeks), test_weeks):
        test_end_idx = min(test_start_idx + test_weeks - 1, len(weeks) - 1)
        train_end_idx = test_start_idx - embargo_weeks - 1
        if train_end_idx < min_train_weeks - 1:
            continue
        splits.append(
            RollingSplit(
                train_start=pd.Timestamp(weeks[0]),
                train_end=pd.Timestamp(weeks[train_end_idx]),
                test_start=pd.Timestamp(weeks[test_start_idx]),
                test_end=pd.Timestamp(weeks[test_end_idx]),
            )
        )
    return splits


def week_block_bootstrap(
    frame: pd.DataFrame,
    *,
    value_col: str,
    seed: int,
    iterations: int = 1000,
) -> list[float]:
    if frame.empty or value_col not in frame.columns:
        return []
    rng = np.random.default_rng(seed)
    groups = [(week, group[value_col].dropna().to_numpy(dtype=float)) for week, group in frame.groupby("snapshot_date")]
    groups = [(week, values) for week, values in groups if len(values)]
    if not groups:
        return []
    samples: list[float] = []
    for _ in range(iterations):
        values = []
        picks = rng.integers(0, len(groups), size=len(groups))
        for idx in picks:
            values.extend(groups[int(idx)][1])
        samples.append(float(np.mean(values)) if values else float("nan"))
    return samples


def paired_week_route_bootstrap(
    frame: pd.DataFrame,
    *,
    treated_col: str,
    control_col: str,
    seed: int,
    iterations: int = 1000,
) -> list[float]:
    if frame.empty or treated_col not in frame.columns or control_col not in frame.columns:
        return []
    work = frame.copy()
    work["_diff"] = pd.to_numeric(work[treated_col], errors="coerce") - pd.to_numeric(work[control_col], errors="coerce")
    work = work.dropna(subset=["_diff"])
    if work.empty:
        return []
    rng = np.random.default_rng(seed)
    week_groups = [
        group["_diff"].to_numpy(dtype=float)
        for _, group in work.groupby("snapshot_date", sort=True)
        if not group["_diff"].dropna().empty
    ]
    if not week_groups:
        return []
    samples: list[float] = []
    for _ in range(iterations):
        values = []
        picks = rng.integers(0, len(week_groups), size=len(week_groups))
        for idx in picks:
            values.extend(week_groups[int(idx)])
        samples.append(float(np.mean(values)) if values else float("nan"))
    return samples


def ci_from_samples(samples: list[float]) -> tuple[float | None, float | None]:
    clean = [value for value in samples if np.isfinite(value)]
    if not clean:
        return None, None
    return float(np.percentile(clean, 2.5)), float(np.percentile(clean, 97.5))
