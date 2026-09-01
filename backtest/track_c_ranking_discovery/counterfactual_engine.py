from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MC_PATHS, RANDOM_SEED, TOP_N
from .protocol import WeeklyPortfolioOutcome


@dataclass
class CounterfactualMatrixResult:
    """2x2 counterfactual result on pathwise paired common support."""
    null_model: str
    support_weeks: int
    valid_paths: int
    median_common_support_weeks: float
    mean_A_random_ind_random_stock: float
    mean_B_random_ind_b0_best_stock: float
    mean_C_b0_ind_random_stock: float
    mean_D_b0_native: float
    b0_induced_industry_allocation_effect: float
    conditional_stock_selection_effect: float
    interaction_effect: float
    b0_percentile_vs_5000_paths_mean: float
    b0_percentile_vs_5000_paths_median: float
    b0_percentile_vs_5000_paths_cvar: float
    b0_percentile_vs_5000_paths_stop: float


def _safe_numeric(value: object) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def _tail_cvar(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    n_tail = max(1, int(vals.size * 0.1))
    return float(np.mean(np.sort(vals)[:n_tail]))


def run_counterfactual_monte_carlo(
    panel_df: pd.DataFrame,
    b0_weekly_outcomes: list[WeeklyPortfolioOutcome],
    b0_scored_by_snapshot: dict[str, pd.DataFrame],
    horizon: str = "W4",
    n_paths: int = MC_PATHS,
    seed: int = RANDOM_SEED,
    null_model: str = "Null1_Uniform_Industry",
) -> tuple[CounterfactualMatrixResult, pd.DataFrame]:
    """Run k-matched paths with selection-first / maturity-second paired support.

    Missing selected outcomes are never converted to cash and never redrawn.
    Each path is compared with B0 only on the exact weeks where the required
    branches and B0 are jointly mature.
    """
    if null_model not in {
        "Null1_Uniform_Industry",
        "Null2_Candidate_Conditioned_Distinct",
    }:
        raise ValueError(f"Unsupported null model: {null_model}")

    rng = np.random.default_rng(seed)
    ret_col = f"{horizon.lower()}_return_pct"
    stop_col = f"{horizon.lower()}_stop8"

    snaps = sorted(b0_scored_by_snapshot.keys())
    b0_map = {o.snapshot_date: o for o in b0_weekly_outcomes}
    snap_indices = {s: i for i, s in enumerate(snaps)}
    n_snaps = len(snaps)

    # NaN means "selected but outcome unavailable / branch unavailable".
    # k=0 mature cash weeks are explicitly written as 0 below.
    path_A_ret = np.full((n_paths, n_snaps), np.nan, dtype=float)
    path_B_ret = np.full((n_paths, n_snaps), np.nan, dtype=float)
    path_C_ret = np.full((n_paths, n_snaps), np.nan, dtype=float)
    path_A_stop = np.full((n_paths, n_snaps), np.nan, dtype=float)

    b0_ret = np.full(n_snaps, np.nan, dtype=float)
    b0_stop = np.full(n_snaps, np.nan, dtype=float)
    active_attr_mask = np.zeros(n_snaps, dtype=bool)

    for s in snaps:
        idx = snap_indices[s]
        b0_out = b0_map.get(s)
        if b0_out is None or not b0_out.is_mature:
            continue

        b0_ret[idx] = _safe_numeric(b0_out.capital_adjusted_return)
        b0_stop[idx] = _safe_numeric(b0_out.capital_adjusted_stop8)
        k = int(b0_out.pick_count)

        if k == 0:
            # Full-path performance includes the same cash week for every branch.
            path_A_ret[:, idx] = 0.0
            path_B_ret[:, idx] = 0.0
            path_C_ret[:, idx] = 0.0
            path_A_stop[:, idx] = 0.0
            continue

        active_attr_mask[idx] = True
        df_s = b0_scored_by_snapshot[s]
        p_sub = panel_df[
            panel_df.snapshot_date.astype(str) == str(s)
        ][["code", ret_col, stop_col]]
        merged = df_s.merge(p_sub, on="code", how="left")

        eligible = merged[
            (merged.is_actionable == 1)
            & (merged.has_geom_failure == 0)
            & (merged.below_buy_point == 0)
            & (merged.has_known_eps == 1)
            & (merged.has_valid_industry == 1)
        ].copy()
        if eligible.empty or len(eligible) < k:
            continue

        ind_to_codes: dict[str, list[str]] = {}
        ind_to_best_code: dict[str, str] = {}
        for ind, g in eligible.groupby("industry_key"):
            key = str(ind)
            ind_to_codes[key] = g["code"].astype(str).tolist()
            ind_to_best_code[key] = str(g.sort_values("raw_rank")["code"].iloc[0])

        industries = list(ind_to_codes)
        if len(industries) < k:
            continue

        code_to_ret = {
            str(c): _safe_numeric(r)
            for c, r in zip(eligible["code"], pd.to_numeric(eligible[ret_col], errors="coerce"))
        }
        code_to_stop = {
            str(c): _safe_numeric(v)
            for c, v in zip(eligible["code"], pd.to_numeric(eligible[stop_col], errors="coerce"))
        }
        cand_list = eligible[["code", "industry_key"]].astype(str).to_dict(orient="records")

        b0_codes = [str(x) for x in b0_out.selected_codes]
        b0_eligible = eligible[eligible.code.astype(str).isin(b0_codes)]
        b0_industries = b0_eligible["industry_key"].astype(str).unique().tolist()
        if len(b0_industries) != k:
            # The primary B0 is distinct_1; if this invariant is violated,
            # attribution for the snapshot is undefined.
            continue

        def branch_metrics(codes: list[str]) -> tuple[float, float]:
            if len(codes) != k:
                return np.nan, np.nan
            rets = np.array([code_to_ret.get(c, np.nan) for c in codes], dtype=float)
            if not np.isfinite(rets).all():
                return np.nan, np.nan
            stops = np.array([code_to_stop.get(c, np.nan) for c in codes], dtype=float)
            cap_ret = float(np.sum(rets) / float(TOP_N))
            cap_stop = (
                float(np.sum(stops) / float(TOP_N) * 100.0)
                if np.isfinite(stops).all()
                else np.nan
            )
            return cap_ret, cap_stop

        for p in range(n_paths):
            if null_model == "Null1_Uniform_Industry":
                sampled_inds = rng.choice(industries, size=k, replace=False).tolist()
            else:
                sampled_inds: list[str] = []
                for perm_idx in rng.permutation(len(cand_list)):
                    ind = cand_list[int(perm_idx)]["industry_key"]
                    if ind not in sampled_inds:
                        sampled_inds.append(ind)
                    if len(sampled_inds) == k:
                        break
                if len(sampled_inds) != k:
                    continue

            a_codes = [
                ind_to_codes[ind][int(rng.integers(0, len(ind_to_codes[ind])))]
                for ind in sampled_inds
            ]
            b_codes = [ind_to_best_code[ind] for ind in sampled_inds]
            c_codes = [
                ind_to_codes[ind][int(rng.integers(0, len(ind_to_codes[ind])))]
                for ind in b0_industries
            ]

            a_ret, a_stop = branch_metrics(a_codes)
            b_ret, _ = branch_metrics(b_codes)
            c_ret, _ = branch_metrics(c_codes)
            path_A_ret[p, idx] = a_ret
            path_B_ret[p, idx] = b_ret
            path_C_ret[p, idx] = c_ret
            path_A_stop[p, idx] = a_stop

    path_stats: list[tuple[float, float, float, float, int]] = []
    mean_wins: list[bool] = []
    median_wins: list[bool] = []
    cvar_wins: list[bool] = []
    stop_wins: list[bool] = []

    for p in range(n_paths):
        # 2x2 attribution excludes k=0 weeks and requires A/B/C/D on identical support.
        attr_mask = (
            active_attr_mask
            & np.isfinite(path_A_ret[p])
            & np.isfinite(path_B_ret[p])
            & np.isfinite(path_C_ret[p])
            & np.isfinite(b0_ret)
        )
        if np.any(attr_mask):
            path_stats.append(
                (
                    float(np.mean(path_A_ret[p, attr_mask])),
                    float(np.mean(path_B_ret[p, attr_mask])),
                    float(np.mean(path_C_ret[p, attr_mask])),
                    float(np.mean(b0_ret[attr_mask])),
                    int(np.sum(attr_mask)),
                )
            )

        # Full-path percentile is a paired comparison. A missing random week is
        # removed from both A and B0 for that path; k=0 cash weeks remain included.
        perf_mask = np.isfinite(path_A_ret[p]) & np.isfinite(b0_ret)
        if np.any(perf_mask):
            a_vals = path_A_ret[p, perf_mask]
            d_vals = b0_ret[perf_mask]
            mean_wins.append(float(np.mean(d_vals)) > float(np.mean(a_vals)))
            median_wins.append(float(np.median(d_vals)) > float(np.median(a_vals)))
            cvar_wins.append(_tail_cvar(d_vals) > _tail_cvar(a_vals))

            stop_mask = perf_mask & np.isfinite(path_A_stop[p]) & np.isfinite(b0_stop)
            if np.any(stop_mask):
                stop_wins.append(
                    float(np.mean(b0_stop[stop_mask])) < float(np.mean(path_A_stop[p, stop_mask]))
                )

    if not path_stats:
        raise RuntimeError("Counterfactual simulation produced no path with paired A/B/C/D mature support.")

    stats = np.asarray(path_stats, dtype=float)
    mean_A = float(np.mean(stats[:, 0]))
    mean_B = float(np.mean(stats[:, 1]))
    mean_C = float(np.mean(stats[:, 2]))
    mean_D = float(np.mean(stats[:, 3]))
    support_counts = stats[:, 4]

    ind_alloc_effect = mean_D - mean_B
    stock_sel_effect = mean_D - mean_C
    interaction_effect = (mean_D - mean_B) - (mean_C - mean_A)

    pct_mean = float(np.mean(mean_wins) * 100.0) if mean_wins else np.nan
    pct_median = float(np.mean(median_wins) * 100.0) if median_wins else np.nan
    pct_cvar = float(np.mean(cvar_wins) * 100.0) if cvar_wins else np.nan
    pct_stop = float(np.mean(stop_wins) * 100.0) if stop_wins else np.nan

    result = CounterfactualMatrixResult(
        null_model=null_model,
        support_weeks=int(np.sum(active_attr_mask & np.isfinite(b0_ret))),
        valid_paths=int(len(path_stats)),
        median_common_support_weeks=float(np.median(support_counts)),
        mean_A_random_ind_random_stock=round(mean_A, 4),
        mean_B_random_ind_b0_best_stock=round(mean_B, 4),
        mean_C_b0_ind_random_stock=round(mean_C, 4),
        mean_D_b0_native=round(mean_D, 4),
        b0_induced_industry_allocation_effect=round(ind_alloc_effect, 4),
        conditional_stock_selection_effect=round(stock_sel_effect, 4),
        interaction_effect=round(interaction_effect, 4),
        b0_percentile_vs_5000_paths_mean=round(pct_mean, 2),
        b0_percentile_vs_5000_paths_median=round(pct_median, 2),
        b0_percentile_vs_5000_paths_cvar=round(pct_cvar, 2),
        b0_percentile_vs_5000_paths_stop=round(pct_stop, 2),
    )

    df = pd.DataFrame(
        [
            {
                "null_model": null_model,
                "A_random_ind_random_stock": mean_A,
                "B_random_ind_b0_best_stock": mean_B,
                "C_b0_ind_random_stock": mean_C,
                "D_b0_native": mean_D,
                "industry_allocation_effect (D-B)": ind_alloc_effect,
                "stock_selection_effect (D-C)": stock_sel_effect,
                "interaction_effect": interaction_effect,
                "b0_percentile_mean": pct_mean,
                "b0_percentile_median": pct_median,
                "b0_percentile_cvar": pct_cvar,
                "b0_percentile_stop": pct_stop,
                "valid_paths": int(len(path_stats)),
                "median_common_support_weeks": float(np.median(support_counts)),
                "paired_common_support": True,
            }
        ]
    )
    return result, df
