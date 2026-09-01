from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from .config import MC_PATHS, RANDOM_SEED, TOP_N, PRIMARY_HORIZON
from .protocol import compute_3slot_portfolio_weekly, WeeklyPortfolioOutcome


@dataclass
class CounterfactualMatrixResult:
    """Standardized 2x2 Counterfactual Decomposition Result."""
    null_model: str  # 'Null1_Uniform_Industry' or 'Null2_Candidate_Conditioned_Distinct'
    support_weeks: int
    mean_A_random_ind_random_stock: float
    mean_B_random_ind_b0_best_stock: float
    mean_C_b0_ind_random_stock: float
    mean_D_b0_native: float
    b0_induced_industry_allocation_effect: float  # D - B (and C - A)
    conditional_stock_selection_effect: float  # D - C (and B - A)
    interaction_effect: float  # (D - B) - (C - A)
    b0_percentile_vs_5000_paths_mean: float
    b0_percentile_vs_5000_paths_median: float
    b0_percentile_vs_5000_paths_cvar: float
    b0_percentile_vs_5000_paths_stop: float


def run_counterfactual_monte_carlo(
    panel_df: pd.DataFrame,
    b0_weekly_outcomes: list[WeeklyPortfolioOutcome],
    b0_scored_by_snapshot: dict[str, pd.DataFrame],
    horizon: str = "W4",
    n_paths: int = MC_PATHS,
    seed: int = RANDOM_SEED,
    null_model: str = "Null1_Uniform_Industry",
) -> tuple[CounterfactualMatrixResult, pd.DataFrame]:
    """Run 5,000 full historical simulation paths and 2x2 counterfactual decomposition using fast vectorization."""
    rng = np.random.default_rng(seed)
    ret_col = f"{horizon.lower()}_return_pct"
    stop_col = f"{horizon.lower()}_stop8"

    snaps = sorted(b0_scored_by_snapshot.keys())
    b0_map = {o.snapshot_date: o for o in b0_weekly_outcomes}

    # Pre-generate 5,000 paths for Branch A, B, C
    path_A_cap_rets = np.zeros((n_paths, len(snaps)), dtype=np.float64)
    path_B_cap_rets = np.zeros((n_paths, len(snaps)), dtype=np.float64)
    path_C_cap_rets = np.zeros((n_paths, len(snaps)), dtype=np.float64)

    mature_snaps = [s for s in snaps if b0_map[s].is_mature]
    snap_indices = {s: i for i, s in enumerate(snaps)}

    for s in snaps:
        s_idx = snap_indices[s]
        b0_out = b0_map[s]
        k = b0_out.pick_count

        if k == 0:
            continue

        df_s = b0_scored_by_snapshot[s]
        p_sub = panel_df[panel_df.snapshot_date.astype(str) == str(s)][["code", ret_col, stop_col]]
        merged = df_s.merge(p_sub, on="code", how="left")
        el = merged[
            (merged.is_actionable == 1) &
            (merged.has_geom_failure == 0) &
            (merged.below_buy_point == 0) &
            (merged.has_known_eps == 1) &
            (merged.has_valid_industry == 1)
        ].copy()

        if el.empty or len(el) < k:
            continue

        # Fast lookup structures
        ind_to_codes = {}
        ind_to_best_code = {}
        for ind, g in el.groupby("industry_key"):
            ind_to_codes[ind] = g["code"].tolist()
            ind_to_best_code[ind] = g.sort_values("raw_rank")["code"].iloc[0]

        industries = list(ind_to_codes.keys())
        # DO NOT fillna(0.0) - preserve exact raw outcome (selection-first, maturity-second)
        code_to_ret = dict(zip(el["code"], pd.to_numeric(el[ret_col], errors="coerce")))
        cand_list = el[["code", "industry_key"]].to_dict(orient="records")

        b0_codes = b0_out.selected_codes
        b0_el = el[el.code.isin(b0_codes)]
        b0_industries = b0_el["industry_key"].unique().tolist()

        for p in range(n_paths):
            # 1. Sample k industries for Branch A & B
            if null_model == "Null1_Uniform_Industry":
                if len(industries) >= k:
                    sampled_inds = rng.choice(industries, size=k, replace=False).tolist()
                else:
                    sampled_inds = industries
            else:
                # Null 2: Candidate-conditioned distinct
                perm_idx = rng.permutation(len(cand_list))
                sampled_inds = []
                for idx in perm_idx:
                    ind = cand_list[idx]["industry_key"]
                    if ind not in sampled_inds:
                        sampled_inds.append(ind)
                    if len(sampled_inds) >= k:
                        break

            # Branch A: Random ind x Random stock
            sum_ret_a = 0.0
            has_nan_a = False
            for ind in sampled_inds:
                c_pool = ind_to_codes.get(ind)
                if c_pool:
                    chosen = c_pool[rng.integers(0, len(c_pool))]
                    r = code_to_ret.get(chosen, np.nan)
                    if np.isnan(r):
                        has_nan_a = True
                        break
                    sum_ret_a += r
            path_A_cap_rets[p, s_idx] = np.nan if has_nan_a else (sum_ret_a / float(TOP_N))

            # Branch B: Random ind x B0-best stock in ind
            sum_ret_b = 0.0
            has_nan_b = False
            for ind in sampled_inds:
                best_c = ind_to_best_code.get(ind)
                if best_c:
                    r = code_to_ret.get(best_c, np.nan)
                    if np.isnan(r):
                        has_nan_b = True
                        break
                    sum_ret_b += r
            path_B_cap_rets[p, s_idx] = np.nan if has_nan_b else (sum_ret_b / float(TOP_N))

            # Branch C: B0 ind x Random stock in ind
            sum_ret_c = 0.0
            has_nan_c = False
            for ind in b0_industries:
                c_pool = ind_to_codes.get(ind)
                if c_pool:
                    chosen = c_pool[rng.integers(0, len(c_pool))]
                    r = code_to_ret.get(chosen, np.nan)
                    if np.isnan(r):
                        has_nan_c = True
                        break
                    sum_ret_c += r
            path_C_cap_rets[p, s_idx] = np.nan if has_nan_c else (sum_ret_c / float(TOP_N))

    # Compute path statistics over mature weeks (using nan-safe functions)
    mature_indices = [snap_indices[s] for s in mature_snaps]
    path_A_sub = path_A_cap_rets[:, mature_indices]
    path_B_sub = path_B_cap_rets[:, mature_indices]
    path_C_sub = path_C_cap_rets[:, mature_indices]

    path_A_mature_means = np.nanmean(path_A_sub, axis=1)
    path_B_mature_means = np.nanmean(path_B_sub, axis=1)
    path_C_mature_means = np.nanmean(path_C_sub, axis=1)

    b0_mature_cap_rets = np.array([b0_map[s].capital_adjusted_return for s in mature_snaps])
    b0_mean = float(np.mean(b0_mature_cap_rets))
    b0_med = float(np.median(b0_mature_cap_rets))
    b0_cvar = float(np.mean(np.sort(b0_mature_cap_rets)[:max(1, int(len(b0_mature_cap_rets) * 0.1))]))
    b0_stop = float(np.mean([b0_map[s].capital_adjusted_stop8 for s in mature_snaps]))

    # Percentile of B0 vs Path A distribution (pure baseline)
    pct_mean = float((path_A_mature_means < b0_mean).mean() * 100.0)
    path_A_medians = np.nanmedian(path_A_sub, axis=1)
    pct_med = float((path_A_medians < b0_med).mean() * 100.0)

    # 2x2 Mean values across 5,000 paths
    mean_A = float(np.mean(path_A_mature_means))
    mean_B = float(np.mean(path_B_mature_means))
    mean_C = float(np.mean(path_C_mature_means))
    mean_D = b0_mean

    # Decomposed effects
    ind_alloc_effect = mean_D - mean_B
    stock_sel_effect = mean_D - mean_C
    interact_effect = (mean_D - mean_B) - (mean_C - mean_A)

    res = CounterfactualMatrixResult(
        null_model=null_model,
        support_weeks=len(mature_snaps),
        mean_A_random_ind_random_stock=round(mean_A, 4),
        mean_B_random_ind_b0_best_stock=round(mean_B, 4),
        mean_C_b0_ind_random_stock=round(mean_C, 4),
        mean_D_b0_native=round(mean_D, 4),
        b0_induced_industry_allocation_effect=round(ind_alloc_effect, 4),
        conditional_stock_selection_effect=round(stock_sel_effect, 4),
        interaction_effect=round(interact_effect, 4),
        b0_percentile_vs_5000_paths_mean=round(pct_mean, 2),
        b0_percentile_vs_5000_paths_median=round(pct_med, 2),
        b0_percentile_vs_5000_paths_cvar=round(float((path_A_mature_means < b0_cvar).mean() * 100.0), 2),
        b0_percentile_vs_5000_paths_stop=round(float((path_A_mature_means < b0_stop).mean() * 100.0), 2),
    )

    df_decomp = pd.DataFrame([{
        "null_model": null_model,
        "A_random_ind_random_stock": mean_A,
        "B_random_ind_b0_best_stock": mean_B,
        "C_b0_ind_random_stock": mean_C,
        "D_b0_native": mean_D,
        "industry_allocation_effect (D-B)": ind_alloc_effect,
        "stock_selection_effect (D-C)": stock_sel_effect,
        "interaction_effect": interact_effect,
        "b0_percentile_mean": pct_mean,
        "b0_percentile_median": pct_med,
    }])

    return res, df_decomp
