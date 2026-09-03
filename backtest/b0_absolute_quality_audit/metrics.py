from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    BLOCK_BOOTSTRAP_LEN,
    BLOCK_BOOTSTRAP_ROUNDS,
    RANDOM_SEED,
)


def cvar10(values: np.ndarray) -> float | None:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    n_tail = max(1, int(math.ceil(len(x) * 0.10)))
    return float(np.mean(np.sort(x)[:n_tail]))


def basic_distribution_stats(values: list[float] | np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "cvar10": None,
            "positive_rate": None,
            "worst": None,
            "best": None,
        }
    return {
        "n": int(len(x)),
        "mean": round(float(np.mean(x)), 6),
        "median": round(float(np.median(x)), 6),
        "p10": round(float(np.percentile(x, 10)), 6),
        "p25": round(float(np.percentile(x, 25)), 6),
        "p75": round(float(np.percentile(x, 75)), 6),
        "p90": round(float(np.percentile(x, 90)), 6),
        "cvar10": round(float(cvar10(x)), 6),
        "positive_rate": round(float(np.mean(x > 0.0)), 6),
        "worst": round(float(np.min(x)), 6),
        "best": round(float(np.max(x)), 6),
    }


def moving_block_bootstrap_ci(
    values: list[float] | np.ndarray,
    *,
    block_len: int = BLOCK_BOOTSTRAP_LEN,
    rounds: int = BLOCK_BOOTSTRAP_ROUNDS,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {
            "n": 0,
            "mean_ci_low": None,
            "mean_ci_high": None,
            "median_ci_low": None,
            "median_ci_high": None,
        }
    if n < block_len:
        m = float(np.mean(x))
        med = float(np.median(x))
        return {
            "n": n,
            "mean_ci_low": m,
            "mean_ci_high": m,
            "median_ci_low": med,
            "median_ci_high": med,
        }

    rng = np.random.default_rng(seed)
    starts = np.arange(0, n - block_len + 1)
    n_blocks = int(math.ceil(n / float(block_len)))
    means = np.empty(rounds, dtype=float)
    medians = np.empty(rounds, dtype=float)

    for i in range(rounds):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        sample = np.concatenate([x[s : s + block_len] for s in chosen])[:n]
        means[i] = float(np.mean(sample))
        medians[i] = float(np.median(sample))

    return {
        "n": n,
        "mean_ci_low": round(float(np.percentile(means, 2.5)), 6),
        "mean_ci_high": round(float(np.percentile(means, 97.5)), 6),
        "median_ci_low": round(float(np.percentile(medians, 2.5)), 6),
        "median_ci_high": round(float(np.percentile(medians, 97.5)), 6),
    }


def paired_edge_summary(frame: pd.DataFrame, value_col: str, benchmark_col: str) -> dict[str, Any]:
    sub = frame[[value_col, benchmark_col]].dropna().copy()
    if sub.empty:
        return {
            "support_weeks": 0,
            "value": basic_distribution_stats([]),
            "benchmark": basic_distribution_stats([]),
            "spread": basic_distribution_stats([]),
            "spread_block_bootstrap": moving_block_bootstrap_ci([]),
            "beat_rate": None,
        }
    spread = sub[value_col].astype(float).to_numpy() - sub[benchmark_col].astype(float).to_numpy()
    return {
        "support_weeks": int(len(sub)),
        "value": basic_distribution_stats(sub[value_col].astype(float).to_numpy()),
        "benchmark": basic_distribution_stats(sub[benchmark_col].astype(float).to_numpy()),
        "spread": basic_distribution_stats(spread),
        "spread_block_bootstrap": moving_block_bootstrap_ci(spread),
        "beat_rate": round(float(np.mean(spread > 0.0)), 6),
    }


def four_offset_nonoverlap(
    frame: pd.DataFrame,
    *,
    value_col: str,
    benchmark_col: str | None = None,
) -> pd.DataFrame:
    work = frame.sort_values("snapshot_date").copy()
    if benchmark_col:
        work = work[["snapshot_date", value_col, benchmark_col]].dropna()
    else:
        work = work[["snapshot_date", value_col]].dropna()

    rows: list[dict[str, Any]] = []
    for offset in range(4):
        part = work.iloc[offset::4].copy()
        values = part[value_col].astype(float).to_numpy()
        row: dict[str, Any] = {
            "offset": offset,
            "weeks": int(len(part)),
            "value_mean": None if len(values) == 0 else round(float(np.mean(values)), 6),
            "value_median": None if len(values) == 0 else round(float(np.median(values)), 6),
            "value_positive_rate": None if len(values) == 0 else round(float(np.mean(values > 0.0)), 6),
        }
        if benchmark_col:
            bench = part[benchmark_col].astype(float).to_numpy()
            spread = values - bench
            row.update({
                "benchmark_mean": None if len(bench) == 0 else round(float(np.mean(bench)), 6),
                "spread_mean": None if len(spread) == 0 else round(float(np.mean(spread)), 6),
                "spread_median": None if len(spread) == 0 else round(float(np.median(spread)), 6),
                "spread_positive_rate": None if len(spread) == 0 else round(float(np.mean(spread > 0.0)), 6),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_oracle_capture(frame: pd.DataFrame) -> dict[str, Any]:
    sub = frame[
        ["b0_return", "random_mean", "oracle", "oracle_capture_ratio"]
    ].dropna(subset=["b0_return", "random_mean", "oracle"]).copy()
    if sub.empty:
        return {
            "support_weeks": 0,
            "aggregate_capture_ratio": None,
            "weekly_capture_mean": None,
            "weekly_capture_median": None,
            "weekly_capture_positive_rate": None,
        }

    numerator = float(sub["b0_return"].mean() - sub["random_mean"].mean())
    denominator = float(sub["oracle"].mean() - sub["random_mean"].mean())
    agg = None if abs(denominator) <= 1e-9 else numerator / denominator
    weekly = pd.to_numeric(frame["oracle_capture_ratio"], errors="coerce").dropna()

    return {
        "support_weeks": int(len(sub)),
        "aggregate_capture_ratio": None if agg is None else round(float(agg), 6),
        "weekly_capture_mean": None if weekly.empty else round(float(weekly.mean()), 6),
        "weekly_capture_median": None if weekly.empty else round(float(weekly.median()), 6),
        "weekly_capture_positive_rate": None if weekly.empty else round(float((weekly > 0).mean()), 6),
    }


def safe_spearman(rank_values: pd.Series, returns: pd.Series) -> float | None:
    x = pd.to_numeric(rank_values, errors="coerce")
    y = pd.to_numeric(returns, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 3:
        return None
    # Lower B0 rank is better; negate rank so positive correlation means correct ordering.
    xr = (-x[mask]).rank(method="average")
    yr = y[mask].rank(method="average")
    if xr.nunique() <= 1 or yr.nunique() <= 1:
        return None
    corr = xr.corr(yr, method="pearson")
    return None if pd.isna(corr) else float(corr)
