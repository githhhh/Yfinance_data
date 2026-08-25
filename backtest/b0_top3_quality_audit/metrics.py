"""Quality evaluation metrics computation for B0 picks, weekly Top3, and random control comparisons."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def _winsorize_series(s: pd.Series, limits: tuple[float, float] = (0.05, 0.05)) -> pd.Series:
    """Winsorize a pandas series to truncate extreme 5% tails."""
    if s.empty or len(s) < 5:
        return s
    low_val = np.percentile(s, limits[0] * 100)
    high_val = np.percentile(s, (1.0 - limits[1]) * 100)
    return s.clip(lower=low_val, upper=high_val)


def compute_pick_level_quality(
    b0_outcomes_df: pd.DataFrame,
    output_csv: Path | str | None = "backtest/b0_top3_quality_audit/output/b0_pick_quality.csv",
) -> pd.DataFrame:
    """Compute performance metrics grouped by Pick Order (Pick 1, Pick 2, Pick 3, and Overall)."""
    if b0_outcomes_df.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    pick_groups = [("Pick_1", b0_outcomes_df[b0_outcomes_df["pick_order"] == 1]),
                   ("Pick_2", b0_outcomes_df[b0_outcomes_df["pick_order"] == 2]),
                   ("Pick_3", b0_outcomes_df[b0_outcomes_df["pick_order"] == 3]),
                   ("Overall_Top3", b0_outcomes_df)]

    for label, group in pick_groups:
        valid = group[group["entry_open"].notna()]
        n_total = len(group)
        n_valid = len(valid)

        w1_ret = valid["week1_close_return_pct"].dropna()
        w1_mg = valid["week1_max_gain_pct"].dropna()
        asof_ret = valid["current_return_to_asof_pct"].dropna()
        asof_exec = valid["executed_return_to_asof_pct"].dropna()
        asof_mg = valid["max_gain_to_asof_pct"].dropna()
        mg_before_stop = valid["max_gain_before_stop_pct"].dropna()

        w1_winz = _winsorize_series(w1_ret)
        asof_winz = _winsorize_series(asof_ret)
        asof_exec_winz = _winsorize_series(asof_exec)

        stops_count = (valid["stop_8_hit_ever"] == True).sum()
        profits20_count = (valid["profit20_hit"] == True).sum()
        p20_before_stop_count = (valid["profit20_before_stop8"] == True).sum()
        gap_stops_count = (valid["gap_stop"] == True).sum()

        rec = {
            "group": label,
            "total_picks": n_total,
            "valid_picks": n_valid,
            "w1_mean_close_return_pct": round(float(w1_ret.mean()), 4) if not w1_ret.empty else np.nan,
            "w1_median_close_return_pct": round(float(w1_ret.median()), 4) if not w1_ret.empty else np.nan,
            "w1_winsorized_mean_pct": round(float(w1_winz.mean()), 4) if not w1_winz.empty else np.nan,
            "w1_std_close_return_pct": round(float(w1_ret.std()), 4) if len(w1_ret) > 1 else np.nan,
            "w1_min_close_return_pct": round(float(w1_ret.min()), 4) if not w1_ret.empty else np.nan,
            "w1_max_close_return_pct": round(float(w1_ret.max()), 4) if not w1_ret.empty else np.nan,
            "w1_p25_close_return_pct": round(float(np.percentile(w1_ret, 25)), 4) if not w1_ret.empty else np.nan,
            "w1_p75_close_return_pct": round(float(np.percentile(w1_ret, 75)), 4) if not w1_ret.empty else np.nan,
            "w1_win_rate_pct": round(float((w1_ret > 0).mean() * 100.0), 2) if not w1_ret.empty else np.nan,
            "w1_mean_max_gain_pct": round(float(w1_mg.mean()), 4) if not w1_mg.empty else np.nan,
            "asof_mean_current_return_pct": round(float(asof_ret.mean()), 4) if not asof_ret.empty else np.nan,
            "asof_median_current_return_pct": round(float(asof_ret.median()), 4) if not asof_ret.empty else np.nan,
            "asof_winsorized_mean_pct": round(float(asof_winz.mean()), 4) if not asof_winz.empty else np.nan,
            "asof_std_current_return_pct": round(float(asof_ret.std()), 4) if len(asof_ret) > 1 else np.nan,
            "asof_min_current_return_pct": round(float(asof_ret.min()), 4) if not asof_ret.empty else np.nan,
            "asof_max_current_return_pct": round(float(asof_ret.max()), 4) if not asof_ret.empty else np.nan,
            "asof_p25_current_return_pct": round(float(np.percentile(asof_ret, 25)), 4) if not asof_ret.empty else np.nan,
            "asof_p75_current_return_pct": round(float(np.percentile(asof_ret, 75)), 4) if not asof_ret.empty else np.nan,
            "asof_mean_exec_return_pct": round(float(asof_exec.mean()), 4) if not asof_exec.empty else np.nan,
            "asof_median_exec_return_pct": round(float(asof_exec.median()), 4) if not asof_exec.empty else np.nan,
            "asof_winsorized_exec_mean_pct": round(float(asof_exec_winz.mean()), 4) if not asof_exec_winz.empty else np.nan,
            "asof_mean_max_gain_pct": round(float(asof_mg.mean()), 4) if not asof_mg.empty else np.nan,
            "asof_stop8_rate_pct": round(float((stops_count / n_valid) * 100.0), 2) if n_valid > 0 else np.nan,
            "gap_stop_rate_pct": round(float((gap_stops_count / n_valid) * 100.0), 2) if n_valid > 0 else np.nan,
            "mean_max_gain_before_stop_pct": round(float(mg_before_stop.mean()), 4) if not mg_before_stop.empty else np.nan,
            "profit20_rate_pct": round(float((profits20_count / n_valid) * 100.0), 2) if n_valid > 0 else np.nan,
            "profit20_before_stop8_rate_pct": round(float((p20_before_stop_count / n_valid) * 100.0), 2) if n_valid > 0 else np.nan,
        }
        records.append(rec)

    df_out = pd.DataFrame(records)
    if output_csv is not None:
        out_p = Path(output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_p, index=False, encoding="utf-8-sig")

    return df_out


def compute_paired_pick_comparison(
    b0_outcomes_df: pd.DataFrame,
    output_csv: Path | str | None = "backtest/b0_top3_quality_audit/output/b0_paired_pick_comparison.csv",
) -> pd.DataFrame:
    """Compute pairwise paired difference metrics across the common snapshot weeks having all 3 picks."""
    if b0_outcomes_df.empty:
        return pd.DataFrame()

    piv_w1 = b0_outcomes_df.pivot(index="snapshot_date", columns="pick_order", values="week1_close_return_pct").dropna()
    piv_asof = b0_outcomes_df.pivot(index="snapshot_date", columns="pick_order", values="current_return_to_asof_pct").dropna()
    piv_exec = b0_outcomes_df.pivot(index="snapshot_date", columns="pick_order", values="executed_return_to_asof_pct").dropna()

    common_weeks = piv_w1.index.intersection(piv_asof.index).intersection(piv_exec.index)
    n_common = len(common_weeks)
    if n_common == 0:
        return pd.DataFrame()

    pairs = [
        ("Pick_1_vs_Pick_2", 1, 2),
        ("Pick_1_vs_Pick_3", 1, 3),
        ("Pick_2_vs_Pick_3", 2, 3),
    ]

    records: list[dict[str, Any]] = []
    for label, p_a, p_b in pairs:
        # Week 1 diff
        diff_w1 = piv_w1.loc[common_weeks, p_a] - piv_w1.loc[common_weeks, p_b]
        w1_win_rate = (diff_w1 > 0).mean() * 100.0
        w1_t_stat, w1_p_val = stats.ttest_1samp(diff_w1, 0.0) if len(diff_w1) > 1 else (np.nan, np.nan)
        try:
            w1_wilcoxon_stat, w1_wilcoxon_p = stats.wilcoxon(diff_w1)
        except Exception:
            w1_wilcoxon_p = np.nan

        # As-of diff
        diff_asof = piv_asof.loc[common_weeks, p_a] - piv_asof.loc[common_weeks, p_b]
        asof_win_rate = (diff_asof > 0).mean() * 100.0
        asof_t_stat, asof_p_val = stats.ttest_1samp(diff_asof, 0.0) if len(diff_asof) > 1 else (np.nan, np.nan)
        try:
            asof_wilcoxon_stat, asof_wilcoxon_p = stats.wilcoxon(diff_asof)
        except Exception:
            asof_wilcoxon_p = np.nan

        # Exec diff
        diff_exec = piv_exec.loc[common_weeks, p_a] - piv_exec.loc[common_weeks, p_b]
        exec_win_rate = (diff_exec > 0).mean() * 100.0

        records.append({
            "comparison_pair": label,
            "common_weeks_count": n_common,
            "w1_diff_mean_pct": round(float(diff_w1.mean()), 4),
            "w1_diff_median_pct": round(float(diff_w1.median()), 4),
            "w1_win_rate_a_over_b_pct": round(float(w1_win_rate), 2),
            "w1_paired_ttest_pvalue": round(float(w1_p_val), 4) if pd.notna(w1_p_val) else np.nan,
            "w1_wilcoxon_pvalue": round(float(w1_wilcoxon_p), 4) if pd.notna(w1_wilcoxon_p) else np.nan,
            "asof_diff_mean_pct": round(float(diff_asof.mean()), 4),
            "asof_diff_median_pct": round(float(diff_asof.median()), 4),
            "asof_win_rate_a_over_b_pct": round(float(asof_win_rate), 2),
            "asof_paired_ttest_pvalue": round(float(asof_p_val), 4) if pd.notna(asof_p_val) else np.nan,
            "asof_wilcoxon_pvalue": round(float(asof_wilcoxon_p), 4) if pd.notna(asof_wilcoxon_p) else np.nan,
            "exec_diff_mean_pct": round(float(diff_exec.mean()), 4),
            "exec_diff_median_pct": round(float(diff_exec.median()), 4),
            "exec_win_rate_a_over_b_pct": round(float(exec_win_rate), 2),
        })

    df_out = pd.DataFrame(records)
    if output_csv is not None:
        out_p = Path(output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_p, index=False, encoding="utf-8-sig")

    return df_out


def compute_weekly_top3_quality(
    b0_outcomes_df: pd.DataFrame,
    weekly_outcomes_df: pd.DataFrame | None = None,
    output_csv: Path | str | None = "backtest/b0_top3_quality_audit/output/b0_weekly_top3_quality.csv",
) -> pd.DataFrame:
    """Compute performance metrics aggregated per snapshot week for the B0 Top3 portfolio."""
    if b0_outcomes_df.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for s_date, group in b0_outcomes_df.groupby("snapshot_date"):
        valid = group[group["entry_open"].notna()]
        n_picks = len(group)
        n_valid = len(valid)
        codes = group["code"].tolist()

        w1_ret = valid["week1_close_return_pct"].dropna()
        w1_mg = valid["week1_max_gain_pct"].dropna()
        asof_ret = valid["current_return_to_asof_pct"].dropna()
        asof_exec = valid["executed_return_to_asof_pct"].dropna()
        asof_mg = valid["max_gain_to_asof_pct"].dropna()

        stops_count = (valid["stop_8_hit_ever"] == True).sum()
        p20_count = (valid["profit20_hit"] == True).sum()

        # Compute week 2, 3, 4 returns if weekly_outcomes_df is provided
        w_rets: dict[int, list[float]] = {2: [], 3: [], 4: []}
        if weekly_outcomes_df is not None and not weekly_outcomes_df.empty:
            for c in codes:
                for w_idx in [2, 3, 4]:
                    w_match = weekly_outcomes_df[
                        (weekly_outcomes_df["snapshot_date"] == str(s_date))
                        & (weekly_outcomes_df["code"] == c)
                        & (weekly_outcomes_df["holding_week_index"] == w_idx)
                    ]
                    if not w_match.empty:
                        r = w_match.iloc[0].get("week_close_return_from_entry_pct")
                        if r is not None and pd.notna(r):
                            w_rets[w_idx].append(float(r))

        rec = {
            "snapshot_date": str(s_date),
            "total_picks": n_picks,
            "valid_picks": n_valid,
            "picked_codes": ",".join(codes),
            "w1_mean_close_return_pct": round(float(w1_ret.mean()), 4) if not w1_ret.empty else np.nan,
            "w1_median_close_return_pct": round(float(w1_ret.median()), 4) if not w1_ret.empty else np.nan,
            "w1_portfolio_win_rate_pct": round(float((w1_ret > 0).mean() * 100.0), 2) if not w1_ret.empty else np.nan,
            "w1_mean_max_gain_pct": round(float(w1_mg.mean()), 4) if not w1_mg.empty else np.nan,
            "w2_mean_close_return_pct": round(float(np.mean(w_rets[2])), 4) if w_rets[2] else np.nan,
            "w3_mean_close_return_pct": round(float(np.mean(w_rets[3])), 4) if w_rets[3] else np.nan,
            "w4_mean_close_return_pct": round(float(np.mean(w_rets[4])), 4) if w_rets[4] else np.nan,
            "asof_mean_current_return_pct": round(float(asof_ret.mean()), 4) if not asof_ret.empty else np.nan,
            "asof_median_current_return_pct": round(float(asof_ret.median()), 4) if not asof_ret.empty else np.nan,
            "asof_mean_exec_return_pct": round(float(asof_exec.mean()), 4) if not asof_exec.empty else np.nan,
            "asof_median_exec_return_pct": round(float(asof_exec.median()), 4) if not asof_exec.empty else np.nan,
            "asof_mean_max_gain_pct": round(float(asof_mg.mean()), 4) if not asof_mg.empty else np.nan,
            "asof_stop8_count": stops_count,
            "has_profit20_winner": bool(p20_count > 0),
            "all_three_stopped": bool(stops_count == len(valid) and len(valid) >= 3),
        }
        records.append(rec)

    df_out = pd.DataFrame(records)
    if output_csv is not None:
        out_p = Path(output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_p, index=False, encoding="utf-8-sig")

    return df_out


def compute_b0_vs_random_summary(
    b0_weekly_quality_df: pd.DataFrame,
    random_dist_df: pd.DataFrame,
    output_csv: Path | str | None = "backtest/b0_top3_quality_audit/output/b0_vs_random_summary.csv",
) -> pd.DataFrame:
    """Compare B0 performance against Random Top3 baseline distributions with statistical tests."""
    if b0_weekly_quality_df.empty or random_dist_df.empty:
        return pd.DataFrame()

    merged = pd.merge(b0_weekly_quality_df, random_dist_df, on="snapshot_date", suffixes=("_b0", "_rnd"))

    # Compare key metrics
    w1_b0 = merged["w1_mean_close_return_pct"].dropna()
    w1_rnd_p50 = merged["w1_mean_return_pct_p50"].dropna()
    asof_b0 = merged["asof_mean_current_return_pct"].dropna()
    asof_rnd_p50 = merged["asof_mean_return_pct_p50"].dropna()
    asof_exec_b0 = merged["asof_mean_exec_return_pct"].dropna()
    asof_exec_rnd_p50 = merged["asof_mean_exec_return_pct_p50"].dropna()

    w1_spread = w1_b0 - w1_rnd_p50
    asof_spread = asof_b0 - asof_rnd_p50
    asof_exec_spread = asof_exec_b0 - asof_exec_rnd_p50

    w1_beat_count = (w1_spread > 0).sum()
    asof_beat_count = (asof_spread > 0).sum()
    asof_exec_beat_count = (asof_exec_spread > 0).sum()

    n_weeks = len(merged)

    # Statistical significance tests
    w1_t_stat, w1_p_val = stats.ttest_1samp(w1_spread, 0.0) if len(w1_spread) > 1 else (np.nan, np.nan)
    try:
        _, w1_wilcoxon_p = stats.wilcoxon(w1_spread)
    except Exception:
        w1_wilcoxon_p = np.nan

    asof_t_stat, asof_p_val = stats.ttest_1samp(asof_spread, 0.0) if len(asof_spread) > 1 else (np.nan, np.nan)
    try:
        _, asof_wilcoxon_p = stats.wilcoxon(asof_spread)
    except Exception:
        asof_wilcoxon_p = np.nan

    asof_exec_t_stat, asof_exec_p_val = stats.ttest_1samp(asof_exec_spread, 0.0) if len(asof_exec_spread) > 1 else (np.nan, np.nan)
    try:
        _, asof_exec_wilcoxon_p = stats.wilcoxon(asof_exec_spread)
    except Exception:
        asof_exec_wilcoxon_p = np.nan

    records = [
        {
            "comparison_dimension": "Week_1_Mean_Close_Return",
            "b0_mean": round(float(w1_b0.mean()), 4),
            "random_p50_mean": round(float(w1_rnd_p50.mean()), 4),
            "b0_spread": round(float(w1_spread.mean()), 4),
            "b0_win_rate_vs_random_median_pct": round(float((w1_beat_count / n_weeks) * 100.0), 2),
            "average_percentile_rank": round(float(merged["b0_w1_return_percentile"].dropna().mean()), 2),
            "paired_ttest_pvalue": round(float(w1_p_val), 4) if pd.notna(w1_p_val) else np.nan,
            "wilcoxon_pvalue": round(float(w1_wilcoxon_p), 4) if pd.notna(w1_wilcoxon_p) else np.nan,
        },
        {
            "comparison_dimension": "As_Of_Current_Return",
            "b0_mean": round(float(asof_b0.mean()), 4),
            "random_p50_mean": round(float(asof_rnd_p50.mean()), 4),
            "b0_spread": round(float(asof_spread.mean()), 4),
            "b0_win_rate_vs_random_median_pct": round(float((asof_beat_count / n_weeks) * 100.0), 2),
            "average_percentile_rank": round(float(merged["b0_asof_return_percentile"].dropna().mean()), 2),
            "paired_ttest_pvalue": round(float(asof_p_val), 4) if pd.notna(asof_p_val) else np.nan,
            "wilcoxon_pvalue": round(float(asof_wilcoxon_p), 4) if pd.notna(asof_wilcoxon_p) else np.nan,
        },
        {
            "comparison_dimension": "As_Of_Executed_Return_With_Stop8",
            "b0_mean": round(float(asof_exec_b0.mean()), 4),
            "random_p50_mean": round(float(asof_exec_rnd_p50.mean()), 4),
            "b0_spread": round(float(asof_exec_spread.mean()), 4),
            "b0_win_rate_vs_random_median_pct": round(float((asof_exec_beat_count / n_weeks) * 100.0), 2),
            "average_percentile_rank": round(float(merged["b0_asof_exec_return_percentile"].dropna().mean()), 2) if "b0_asof_exec_return_percentile" in merged.columns else round(float(merged["b0_asof_return_percentile"].dropna().mean()), 2),
            "paired_ttest_pvalue": round(float(asof_exec_p_val), 4) if pd.notna(asof_exec_p_val) else np.nan,
            "wilcoxon_pvalue": round(float(asof_exec_wilcoxon_p), 4) if pd.notna(asof_exec_wilcoxon_p) else np.nan,
        },
    ]

    df_out = pd.DataFrame(records)
    if output_csv is not None:
        out_p = Path(output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_p, index=False, encoding="utf-8-sig")

    return df_out
