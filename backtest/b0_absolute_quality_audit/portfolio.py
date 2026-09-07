from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    MAX_EXACT_COMBINATIONS,
    RANDOM_SEED,
    RAW_MC_DRAWS,
    TOP_N,
)


def industry_key(row: pd.Series | dict[str, Any]) -> str:
    raw = str(row.get("industry", "") or "").strip().lower()
    if raw and raw not in {"nan", "none", "<na>"}:
        return raw
    return f"__unknown__{str(row.get('code', '')).strip().upper()}"


def capital_adjusted_return(values: list[float], *, top_n: int = TOP_N) -> float:
    return float(np.sum(values) / float(top_n))


def portfolio_from_codes(
    snapshot_df: pd.DataFrame,
    codes: list[str],
    *,
    return_col: str,
    stop_col: str | None = None,
) -> dict[str, Any]:
    lookup = snapshot_df.set_index("code", drop=False)
    rows: list[pd.Series] = []
    for code in codes:
        if code not in lookup.index:
            return {
                "mature": False,
                "pick_count": len(codes),
                "selection_quality_return": np.nan,
                "capital_adjusted_return": np.nan,
                "capital_adjusted_stop8": np.nan,
                "one_pick_ruined": False,
            }
        row = lookup.loc[code]
        if isinstance(row, pd.DataFrame):
            raise RuntimeError(f"Duplicate code {code} within snapshot")
        rows.append(row)

    if not rows:
        return {
            "mature": True,
            "pick_count": 0,
            "selection_quality_return": np.nan,
            "capital_adjusted_return": 0.0,
            "capital_adjusted_stop8": 0.0,
            "one_pick_ruined": False,
        }

    rets = [pd.to_numeric(pd.Series([r.get(return_col)]), errors="coerce").iloc[0] for r in rows]
    if any(pd.isna(x) for x in rets):
        return {
            "mature": False,
            "pick_count": len(codes),
            "selection_quality_return": np.nan,
            "capital_adjusted_return": np.nan,
            "capital_adjusted_stop8": np.nan,
            "one_pick_ruined": False,
        }

    ret_vals = [float(x) for x in rets]
    stop_vals: list[bool] = []
    if stop_col:
        for r in rows:
            v = r.get(stop_col)
            if v is None or pd.isna(v):
                return {
                    "mature": False,
                    "pick_count": len(codes),
                    "selection_quality_return": np.nan,
                    "capital_adjusted_return": np.nan,
                    "capital_adjusted_stop8": np.nan,
                    "one_pick_ruined": False,
                }
            stop_vals.append(bool(v))
    else:
        stop_vals = [False] * len(rows)

    return {
        "mature": True,
        "pick_count": len(codes),
        "selection_quality_return": float(np.mean(ret_vals)),
        "capital_adjusted_return": capital_adjusted_return(ret_vals),
        "capital_adjusted_stop8": float(np.sum(stop_vals) / float(TOP_N) * 100.0),
        "one_pick_ruined": bool(any(v <= -8.0 for v in ret_vals) or any(stop_vals)),
    }


def greedy_oracle_codes(
    candidates: pd.DataFrame,
    *,
    k: int,
    return_col: str,
    distinct_industry: bool,
) -> list[str]:
    if k <= 0 or candidates.empty:
        return []

    work = candidates.copy()
    work["_ret"] = pd.to_numeric(work[return_col], errors="coerce")
    work = work[work["_ret"].notna()].sort_values(
        ["_ret", "code"], ascending=[False, True], kind="stable"
    )

    selected: list[str] = []
    used: set[str] = set()
    for _, row in work.iterrows():
        if len(selected) >= k:
            break
        ind = industry_key(row)
        if distinct_industry and ind in used:
            continue
        selected.append(str(row["code"]))
        used.add(ind)
    return selected


def _valid_combo(
    candidate_rows: list[pd.Series],
    combo: tuple[int, ...],
    *,
    distinct_industry: bool,
) -> bool:
    if not distinct_industry:
        return True
    inds = [industry_key(candidate_rows[i]) for i in combo]
    return len(inds) == len(set(inds))


def exact_portfolio_distribution(
    candidates: pd.DataFrame,
    *,
    k: int,
    return_col: str,
    distinct_industry: bool,
) -> np.ndarray:
    if k <= 0:
        return np.array([0.0], dtype=float)

    work = candidates.copy()
    work["_ret"] = pd.to_numeric(work[return_col], errors="coerce")
    work = work[work["_ret"].notna()].reset_index(drop=True)
    if len(work) < k:
        return np.array([], dtype=float)

    rows = [row for _, row in work.iterrows()]
    vals = work["_ret"].to_numpy(dtype=float)
    out: list[float] = []
    for combo in itertools.combinations(range(len(work)), k):
        if not _valid_combo(rows, combo, distinct_industry=distinct_industry):
            continue
        out.append(float(np.sum(vals[list(combo)]) / float(TOP_N)))
    return np.asarray(out, dtype=float)


def _industry_ids(work: pd.DataFrame) -> np.ndarray:
    keys = [industry_key(row) for _, row in work.iterrows()]
    mapping: dict[str, int] = {}
    ids: list[int] = []
    for key in keys:
        if key not in mapping:
            mapping[key] = len(mapping)
        ids.append(mapping[key])
    return np.asarray(ids, dtype=int)


def monte_carlo_portfolio_distribution(
    candidates: pd.DataFrame,
    *,
    k: int,
    return_col: str,
    distinct_industry: bool,
    n_draws: int = RAW_MC_DRAWS,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    if k <= 0:
        return np.array([0.0], dtype=float)

    work = candidates.copy()
    work["_ret"] = pd.to_numeric(work[return_col], errors="coerce")
    work = work[work["_ret"].notna()].reset_index(drop=True)
    n = len(work)
    if n < k:
        return np.array([], dtype=float)

    vals = work["_ret"].to_numpy(dtype=float)
    ind_ids = _industry_ids(work)
    rng = np.random.default_rng(seed)

    accepted: list[np.ndarray] = []
    have = 0
    attempts = 0
    max_attempts = max(n_draws * 20, 100_000)
    batch = min(max(20_000, n_draws // 4), 100_000)

    while have < n_draws and attempts < max_attempts:
        size = min(batch, max_attempts - attempts)
        idx = rng.integers(0, n, size=(size, k), endpoint=False)
        attempts += size

        unique_candidate = np.ones(size, dtype=bool)
        for a in range(k):
            for b in range(a + 1, k):
                unique_candidate &= idx[:, a] != idx[:, b]

        if distinct_industry:
            unique_ind = np.ones(size, dtype=bool)
            inds = ind_ids[idx]
            for a in range(k):
                for b in range(a + 1, k):
                    unique_ind &= inds[:, a] != inds[:, b]
            keep = unique_candidate & unique_ind
        else:
            keep = unique_candidate

        idx = idx[keep]
        if len(idx) == 0:
            continue

        sums = vals[idx].sum(axis=1) / float(TOP_N)
        needed = n_draws - have
        if len(sums) > needed:
            sums = sums[:needed]
        accepted.append(sums.astype(float))
        have += len(sums)

    if have < n_draws:
        raise RuntimeError(
            f"Could not draw enough feasible portfolios: requested={n_draws}, "
            f"accepted={have}, n={n}, k={k}, distinct={distinct_industry}"
        )
    return np.concatenate(accepted)


def portfolio_distribution(
    candidates: pd.DataFrame,
    *,
    k: int,
    return_col: str,
    distinct_industry: bool,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = int(pd.to_numeric(candidates[return_col], errors="coerce").notna().sum())
    theoretical = math.comb(n, k) if n >= k and k >= 0 else 0

    if theoretical <= MAX_EXACT_COMBINATIONS:
        dist = exact_portfolio_distribution(
            candidates,
            k=k,
            return_col=return_col,
            distinct_industry=distinct_industry,
        )
        return dist, {
            "method": "exact",
            "theoretical_combinations_before_constraints": int(theoretical),
            "evaluated_portfolios": int(len(dist)),
            "mc_percentile_se_max_pp": 0.0,
        }

    dist = monte_carlo_portfolio_distribution(
        candidates,
        k=k,
        return_col=return_col,
        distinct_industry=distinct_industry,
        seed=seed,
    )
    # Worst-case SE for a percentile probability occurs at p=.5.
    se_pp = math.sqrt(0.25 / float(len(dist))) * 100.0
    return dist, {
        "method": "monte_carlo",
        "theoretical_combinations_before_constraints": int(theoretical),
        "evaluated_portfolios": int(len(dist)),
        "mc_percentile_se_max_pp": round(float(se_pp), 4),
    }


def percentile_rank(value: float, distribution: np.ndarray) -> float | None:
    if distribution.size == 0 or not math.isfinite(value):
        return None
    return float(np.mean(distribution <= value) * 100.0)


def distribution_summary(
    value: float,
    distribution: np.ndarray,
    oracle_value: float | None,
) -> dict[str, Any]:
    if distribution.size == 0:
        return {
            "random_mean": None,
            "random_median": None,
            "random_p25": None,
            "random_p75": None,
            "b0_percentile": None,
            "oracle": oracle_value,
            "edge_vs_random_mean": None,
            "oracle_capture_ratio": None,
        }

    random_mean = float(np.mean(distribution))
    random_median = float(np.median(distribution))
    edge = float(value - random_mean)
    capture = None
    if oracle_value is not None:
        denom = float(oracle_value - random_mean)
        if abs(denom) > 1e-9:
            capture = edge / denom

    return {
        "random_mean": round(random_mean, 6),
        "random_median": round(random_median, 6),
        "random_p25": round(float(np.percentile(distribution, 25)), 6),
        "random_p75": round(float(np.percentile(distribution, 75)), 6),
        "b0_percentile": round(float(percentile_rank(value, distribution)), 4),
        "oracle": None if oracle_value is None else round(float(oracle_value), 6),
        "edge_vs_random_mean": round(edge, 6),
        "oracle_capture_ratio": None if capture is None else round(float(capture), 6),
    }
