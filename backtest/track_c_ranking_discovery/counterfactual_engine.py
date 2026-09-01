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
    """Run 5,000 full historical simulation paths and 2x2 counterfactual decomposition."""
    rng = np.random.default_rng(seed)
    ret_col = f"{horizon.lower()}_return_pct"
    stop_col = f"{horizon.lower()}_stop8"

    # Map snapshot to eligible dataframe
    snaps = sorted(b0_scored_by_snapshot.keys())
    eligible_by_snap = {}
    for s in snaps:
        df_s = b0_scored_by_snapshot[s]
        # Merge outcome columns from panel_df for evaluation
        p_sub = panel_df[panel_df.snapshot_date.astype(str) == str(s)][["code", ret_col, stop_col]]
        merged = df_s.merge(p_sub, on="code", how="left")
        el = merged[
            (merged.is_actionable == 1) &
            (merged.has_geom_failure == 0) &
            (merged.below_buy_point == 0)
        ].copy()
        eligible_by_snap[s] = el

    # Map B0 outcomes by snapshot
    b0_map = {o.snapshot_date: o for o in b0_weekly_outcomes}

    # Pre-generate 5,000 paths for Branch A, B, C
    # Path outcomes storage: path_idx -> list of weekly capital returns
    path_A_cap_rets = np.zeros((n_paths, len(snaps)), dtype=np.float64)
    path_B_cap_rets = np.zeros((n_paths, len(snaps)), dtype=np.float64)
    path_C_cap_rets = np.zeros((n_paths, len(snaps)), dtype=np.float64)

    # For average 2x2 table across all mature weeks
    mature_snaps = [s for s in snaps if b0_map[s].is_mature]
    snap_indices = {s: i for i, s in enumerate(snaps)}

    for s in snaps:
        s_idx = snap_indices[s]
        b0_out = b0_map[s]
        k = b0_out.pick_count
        el = eligible_by_snap[s]

        if k == 0 or el.empty:
            # k=0 week: Capital return is 0.0 for all branches
            continue

        industries = el["industry"].dropna().unique().tolist()
        b0_codes = b0_out.selected_codes
        b0_el = el[el.code.isin(b0_codes)]
        b0_industries = b0_el["industry"].unique().tolist()

        # Generate N draws for this snapshot
        for p in range(n_paths):
            # 1. Sample k industries for Branch A & B
            if null_model == "Null1_Uniform_Industry":
                # Uniform random k industries from available
                if len(industries) >= k:
                    sampled_inds = rng.choice(industries, size=k, replace=False).tolist()
                else:
                    sampled_inds = industries
            else:
                # Null 2: Candidate-conditioned distinct
                # Sample k distinct candidates from el with distinct industry constraint
                shuffled_el = el.sample(frac=1.0, random_state=rng.integers(0, 1000000))
                sampled_inds = []
                for _, r in shuffled_el.iterrows():
                    ind = str(r["industry"])
                    if ind not in sampled_inds:
                        sampled_inds.append(ind)
                    if len(sampled_inds) >= k:
                        break

            # Branch A: Random ind x Random stock
            a_codes = []
            for ind in sampled_inds:
                ind_stocks = el[el.industry == ind]
                if not ind_stocks.empty:
                    chosen_code = ind_stocks.sample(n=1, random_state=rng.integers(0, 1000000))["code"].iloc[0]
                    a_codes.append(chosen_code)
            a_rets = el[el.code.isin(a_codes)][ret_col].values
            a_cap_ret = float(np.sum(np.nan_to_num(a_rets, nan=0.0)) / float(TOP_N))
            path_A_cap_rets[p, s_idx] = a_cap_ret

            # Branch B: Random ind x B0-best stock in ind
            b_codes = []
            for ind in sampled_inds:
                ind_stocks = el[el.industry == ind].sort_values("raw_rank")
                if not ind_stocks.empty:
                    b_codes.append(ind_stocks.iloc[0]["code"])
            b_rets = el[el.code.isin(b_codes)][ret_col].values
            b_cap_ret = float(np.sum(np.nan_to_num(b_rets, nan=0.0)) / float(TOP_N))
            path_B_cap_rets[p, s_idx] = b_cap_ret

            # Branch C: B0 ind x Random stock in ind
            c_codes = []
            for ind in b0_industries:
                ind_stocks = el[el.industry == ind]
                if not ind_stocks.empty:
                    chosen_code = ind_stocks.sample(n=1, random_state=rng.integers(0, 1000000))["code"].iloc[0]
                    c_codes.append(chosen_code)
            c_rets = el[el.code.isin(c_codes)][ret_col].values
            c_cap_ret = float(np.sum(np.nan_to_num(c_rets, nan=0.0)) / float(TOP_N))
            path_C_cap_rets[p, s_idx] = c_cap_ret

    # Compute path statistics over mature weeks
    mature_indices = [snap_indices[s] for s in mature_snaps]
    path_A_mature_means = path_A_cap_rets[:, mature_indices].mean(axis=1)
    path_B_mature_means = path_B_cap_rets[:, mature_indices].mean(axis=1)
    path_C_mature_means = path_C_cap_rets[:, mature_indices].mean(axis=1)

    b0_mature_cap_rets = np.array([b0_map[s].capital_adjusted_return for s in mature_snaps])
    b0_mean = float(np.mean(b0_mature_cap_rets))
    b0_med = float(np.median(b0_mature_cap_rets))
    b0_cvar = float(np.mean(np.sort(b0_mature_cap_rets)[:max(1, int(len(b0_mature_cap_rets) * 0.1))]))
    b0_stop = float(np.mean([b0_map[s].capital_adjusted_stop8 for s in mature_snaps]))

    # Percentile of B0 vs Path A distribution (pure baseline)
    pct_mean = float((path_A_mature_means < b0_mean).mean() * 100.0)
    path_A_medians = np.median(path_A_cap_rets[:, mature_indices], axis=1)
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

    # Summary table
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
