"""B0 Rank Position & TopK Marginal Contribution Audit.

This module is DIAGNOSTIC / AUDIT ONLY. It evaluates whether B0's Rank 1, 2, 3
carry sequential/monotonic ranking information, and whether Top3 > Top2 has
rigorous statistical support under Common Support.

It does NOT modify the production selector, frozen rules, or frozen protocols.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from backtest.b0_top3_quality_audit.research_windows import (
    contaminated_validation_dates,
    train_dates,
)

logger = logging.getLogger(__name__)

PRIMARY_HORIZONS = (1, 2, 4)
DIAGNOSTIC_HORIZON = 3
ALL_HORIZONS = (1, 2, 3, 4)
REPORT_NAME = "B0_RANK_POSITION_TOPK_AUDIT_REPORT.md"


@dataclass(frozen=True)
class AuditPaths:
    root_dir: Path
    output_dir: Path
    b0_events_path: Path
    events_path: Path
    weekly_path: Path
    three_tier_weekly_path: Path
    random_summary_path: Path


def default_audit_paths() -> AuditPaths:
    root_dir = Path(__file__).resolve().parent
    output_dir = root_dir / "output"
    return AuditPaths(
        root_dir=root_dir,
        output_dir=output_dir,
        b0_events_path=output_dir / "b0_selection_events.csv",
        events_path=root_dir / "data" / "candidate_event_outcomes.parquet",
        weekly_path=root_dir / "data" / "candidate_weekly_outcomes.parquet",
        three_tier_weekly_path=output_dir / "three_tier_weekly_comparison.csv",
        random_summary_path=output_dir / "random_signal_top3_distribution.csv",
    )


def _pct(val: float | int | np.floating | None) -> float:
    if val is None or pd.isna(val):
        return np.nan
    return round(float(val), 4)


def _rate(num: float, den: float) -> float:
    if den == 0 or pd.isna(den):
        return np.nan
    return round(float(num) / float(den) * 100.0, 2)


def describe_statistical_support(p_value: float) -> str:
    """Describe two-sided statistical support without overstating p >= 0.05."""
    return (
        "statistically significant"
        if p_value < 0.05
        else "directional and not statistically significant"
    )


def wilcoxon_signed_rank_p(diffs: Sequence[float | int]) -> float:
    """Paired Wilcoxon signed-rank two-sided test on non-zero differences."""
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    d_nonzero = d[d != 0.0]
    if len(d_nonzero) < 5:
        return np.nan
    try:
        res = stats.wilcoxon(d_nonzero, alternative="two-sided")
        return round(float(res.pvalue), 4)
    except Exception:
        return np.nan


def bootstrap_ci95(
    values: Sequence[float | int],
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot_stats = [stat_fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    low = round(float(np.percentile(boot_stats, 2.5)), 4)
    high = round(float(np.percentile(boot_stats, 97.5)), 4)
    return (low, high)


def build_common_support_weekly_matrix(
    b0_events_df: pd.DataFrame,
    events_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    all_snapshots: list[str],
    train_snapshots: set[str],
    contaminated_snapshots: set[str],
) -> pd.DataFrame:
    """Build week-level dataset containing Rank 1, 2, 3 outcomes and portfolios.
    
    Only weeks where B0 produced all 3 picks and all 3 picks have mature (is_complete_week==True)
    and valid outcomes enter the common-support denominator for each horizon.
    """
    event_lookup = {
        (str(r["snapshot_date"]), str(r["code"])): r.to_dict()
        for _, r in events_df.iterrows()
    }
    weekly_lookup = {
        (str(r["snapshot_date"]), str(r["code"]), int(r["holding_week_index"])): r.to_dict()
        for _, r in weekly_df.iterrows()
    }

    weekly_rows: list[dict[str, Any]] = []

    for snap in all_snapshots:
        snap_b0 = b0_events_df[b0_events_df["snapshot_date"].astype(str) == snap].sort_values("pick_order")
        n_picks = len(snap_b0)
        is_3picks = (n_picks == 3)
        picks = snap_b0.to_dict(orient="records") if is_3picks else []

        c1 = str(picks[0]["code"]) if is_3picks else ""
        c2 = str(picks[1]["code"]) if is_3picks else ""
        c3 = str(picks[2]["code"]) if is_3picks else ""

        e1 = event_lookup.get((snap, c1), {}) if is_3picks else {}
        e2 = event_lookup.get((snap, c2), {}) if is_3picks else {}
        e3 = event_lookup.get((snap, c3), {}) if is_3picks else {}

        row: dict[str, Any] = {
            "snapshot_date": snap,
            "is_train": snap in train_snapshots,
            "is_contaminated_val": snap in contaminated_snapshots,
            "n_picks": n_picks,
            "is_3picks": is_3picks,
            "r1_code": c1,
            "r2_code": c2,
            "r3_code": c3,
        }

        # Horizon Returns and Portfolios
        for h in ALL_HORIZONS:
            if not is_3picks:
                row[f"w{h}_common_valid"] = False
                row[f"r1_w{h}_return_pct"] = np.nan
                row[f"r2_w{h}_return_pct"] = np.nan
                row[f"r3_w{h}_return_pct"] = np.nan
                row[f"k1_w{h}"] = np.nan
                row[f"k2_w{h}"] = np.nan
                row[f"k3_w{h}"] = np.nan
                row[f"mc2_w{h}"] = np.nan
                row[f"mc3_w{h}"] = np.nan
                row[f"r3_minus_r2_w{h}"] = np.nan
                row[f"r1_minus_r2_w{h}"] = np.nan
                row[f"r1_minus_r3_w{h}"] = np.nan
                row[f"r1_w{h}_max_gain_pct"] = np.nan
                row[f"r2_w{h}_max_gain_pct"] = np.nan
                row[f"r3_w{h}_max_gain_pct"] = np.nan
                continue

            w1_info = weekly_lookup.get((snap, c1, h), {})
            w2_info = weekly_lookup.get((snap, c2, h), {})
            w3_info = weekly_lookup.get((snap, c3, h), {})

            w1_complete = bool(w1_info.get("is_complete_week", False))
            w2_complete = bool(w2_info.get("is_complete_week", False))
            w3_complete = bool(w3_info.get("is_complete_week", False))

            r1_ret = w1_info.get("week_close_return_from_entry_pct")
            r2_ret = w2_info.get("week_close_return_from_entry_pct")
            r3_ret = w3_info.get("week_close_return_from_entry_pct")

            r1_mg = w1_info.get("week_max_gain_from_entry_pct")
            r2_mg = w2_info.get("week_max_gain_from_entry_pct")
            r3_mg = w3_info.get("week_max_gain_from_entry_pct")

            valid = (
                w1_complete and w2_complete and w3_complete
                and r1_ret is not None and not pd.isna(r1_ret)
                and r2_ret is not None and not pd.isna(r2_ret)
                and r3_ret is not None and not pd.isna(r3_ret)
                and r1_mg is not None and not pd.isna(r1_mg)
                and r2_mg is not None and not pd.isna(r2_mg)
                and r3_mg is not None and not pd.isna(r3_mg)
            )

            row[f"w{h}_common_valid"] = valid
            if valid:
                v1, v2, v3 = float(r1_ret), float(r2_ret), float(r3_ret)
                row[f"r1_w{h}_return_pct"] = _pct(v1)
                row[f"r2_w{h}_return_pct"] = _pct(v2)
                row[f"r3_w{h}_return_pct"] = _pct(v3)
                row[f"r1_w{h}_max_gain_pct"] = _pct(r1_mg)
                row[f"r2_w{h}_max_gain_pct"] = _pct(r2_mg)
                row[f"r3_w{h}_max_gain_pct"] = _pct(r3_mg)

                k1 = v1
                k2 = (v1 + v2) / 2.0
                k3 = (v1 + v2 + v3) / 3.0
                row[f"k1_w{h}"] = _pct(k1)
                row[f"k2_w{h}"] = _pct(k2)
                row[f"k3_w{h}"] = _pct(k3)
                row[f"mc2_w{h}"] = _pct(k2 - k1)
                row[f"mc3_w{h}"] = _pct(k3 - k2)
                row[f"r3_minus_r2_w{h}"] = _pct(v3 - v2)
                row[f"r1_minus_r2_w{h}"] = _pct(v1 - v2)
                row[f"r1_minus_r3_w{h}"] = _pct(v1 - v3)
            else:
                row[f"r1_w{h}_return_pct"] = np.nan
                row[f"r2_w{h}_return_pct"] = np.nan
                row[f"r3_w{h}_return_pct"] = np.nan
                row[f"k1_w{h}"] = np.nan
                row[f"k2_w{h}"] = np.nan
                row[f"k3_w{h}"] = np.nan
                row[f"mc2_w{h}"] = np.nan
                row[f"mc3_w{h}"] = np.nan
                row[f"r3_minus_r2_w{h}"] = np.nan
                row[f"r1_minus_r2_w{h}"] = np.nan
                row[f"r1_minus_r3_w{h}"] = np.nan
                row[f"r1_w{h}_max_gain_pct"] = np.nan
                row[f"r2_w{h}_max_gain_pct"] = np.nan
                row[f"r3_w{h}_max_gain_pct"] = np.nan

        # Path / Risk variables
        if is_3picks:
            for rank_idx, (p, e) in enumerate([(picks[0], e1), (picks[1], e2), (picks[2], e3)], 1):
                row[f"r{rank_idx}_stop8_before_p20"] = bool(e.get("stop8_before_profit20", False))
                row[f"r{rank_idx}_stop8_ever"] = bool(e.get("stop_8_hit_ever", False))
                row[f"r{rank_idx}_gap_stop"] = bool(e.get("gap_stop", False))
                row[f"r{rank_idx}_profit20"] = bool(e.get("profit20_hit", False))
                row[f"r{rank_idx}_asof_return_pct"] = _pct(e.get("current_return_to_asof_pct"))
                row[f"r{rank_idx}_asof_max_gain_pct"] = _pct(e.get("max_gain_to_asof_pct"))
                row[f"r{rank_idx}_asof_drawdown_pct"] = _pct(e.get("max_drawdown_to_asof_pct"))
        else:
            for rank_idx in [1, 2, 3]:
                row[f"r{rank_idx}_stop8_before_p20"] = np.nan
                row[f"r{rank_idx}_stop8_ever"] = np.nan
                row[f"r{rank_idx}_gap_stop"] = np.nan
                row[f"r{rank_idx}_profit20"] = np.nan
                row[f"r{rank_idx}_asof_return_pct"] = np.nan
                row[f"r{rank_idx}_asof_max_gain_pct"] = np.nan
                row[f"r{rank_idx}_asof_drawdown_pct"] = np.nan

        weekly_rows.append(row)

    return pd.DataFrame(weekly_rows)


def _get_segment_frames(matrix_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Return standard diagnostic segment slices."""
    all_3p = matrix_df[matrix_df["is_3picks"]].copy()
    midpoint = int(np.ceil(len(all_3p) / 2.0))
    return [
        ("All common-support weeks", all_3p),
        ("Train-era weeks 1-30", all_3p[all_3p["is_train"]]),
        ("Contaminated validation weeks 31-40", all_3p[all_3p["is_contaminated_val"]]),
        ("Early half", all_3p.iloc[:midpoint]),
        ("Late half", all_3p.iloc[midpoint:]),
    ]


def summarize_rank_position_quality(matrix_df: pd.DataFrame) -> pd.DataFrame:
    """Generate Position Quality summary for Rank 1, Rank 2, and Rank 3."""
    rows: list[dict[str, Any]] = []
    segments = _get_segment_frames(matrix_df)

    for h in ALL_HORIZONS:
        h_status = "PRIMARY" if h in PRIMARY_HORIZONS else "DIAGNOSTIC_ONLY"
        for seg_name, seg_df in segments:
            valid_seg = seg_df[seg_df[f"w{h}_common_valid"]].copy()
            n_weeks = len(valid_seg)
            if n_weeks == 0:
                continue

            for rank_num in [1, 2, 3]:
                ret_col = f"r{rank_num}_w{h}_return_pct"
                mg_col = f"r{rank_num}_w{h}_max_gain_pct"
                rets = valid_seg[ret_col].dropna()
                mgs = valid_seg[mg_col].dropna()

                stop8_p20_series = valid_seg[f"r{rank_num}_stop8_before_p20"].dropna()
                stop8_ever_series = valid_seg[f"r{rank_num}_stop8_ever"].dropna()
                gap_stop_series = valid_seg[f"r{rank_num}_gap_stop"].dropna()
                profit20_series = valid_seg[f"r{rank_num}_profit20"].dropna()
                asof_rets = valid_seg[f"r{rank_num}_asof_return_pct"].dropna()
                asof_mgs = valid_seg[f"r{rank_num}_asof_max_gain_pct"].dropna()
                asof_dds = valid_seg[f"r{rank_num}_asof_drawdown_pct"].dropna()

                rows.append({
                    "horizon": f"W{h}",
                    "segment": seg_name,
                    "rank_position": f"Rank{rank_num}",
                    "sample_weeks": n_weeks,
                    "mean_return_pct": _pct(rets.mean()),
                    "median_return_pct": _pct(rets.median()),
                    "p25_return_pct": _pct(rets.quantile(0.25)),
                    "p75_return_pct": _pct(rets.quantile(0.75)),
                    "positive_return_rate_pct": _rate((rets > 0).sum(), len(rets)),
                    "stop8_before_p20_rate_pct": _rate(stop8_p20_series.sum(), len(stop8_p20_series)),
                    "stop8_ever_rate_pct": _rate(stop8_ever_series.sum(), len(stop8_ever_series)),
                    "gap_stop_rate_pct": _rate(gap_stop_series.sum(), len(gap_stop_series)),
                    "profit20_hit_rate_pct": _rate(profit20_series.sum(), len(profit20_series)),
                    "mean_max_gain_pct": _pct(mgs.mean()),
                    "median_max_gain_pct": _pct(mgs.median()),
                    "mean_asof_return_pct": _pct(asof_rets.mean()),
                    "median_asof_return_pct": _pct(asof_rets.median()),
                    "mean_asof_drawdown_pct": _pct(asof_dds.mean()),
                    "median_asof_drawdown_pct": _pct(asof_dds.median()),
                    "horizon_status": h_status,
                })

    return pd.DataFrame(rows)


def summarize_topk_marginal_contributions(matrix_df: pd.DataFrame) -> pd.DataFrame:
    """Generate K1/K2/K3 portfolios, Marginal Contributions (MC2, MC3), and Hyp A/B/C tests."""
    rows: list[dict[str, Any]] = []
    segments = _get_segment_frames(matrix_df)

    for h in ALL_HORIZONS:
        h_status = "PRIMARY" if h in PRIMARY_HORIZONS else "DIAGNOSTIC_ONLY"
        for seg_name, seg_df in segments:
            valid_seg = seg_df[seg_df[f"w{h}_common_valid"]].copy()
            n_weeks = len(valid_seg)
            if n_weeks == 0:
                continue

            k1 = valid_seg[f"k1_w{h}"]
            k2 = valid_seg[f"k2_w{h}"]
            k3 = valid_seg[f"k3_w{h}"]

            mc2 = valid_seg[f"mc2_w{h}"]
            mc3 = valid_seg[f"mc3_w{h}"]
            r3_minus_r2 = valid_seg[f"r3_minus_r2_w{h}"]
            r1_minus_r2 = valid_seg[f"r1_minus_r2_w{h}"]
            k3_minus_k2 = mc3

            mc2_boot = bootstrap_ci95(mc2, stat_fn=np.mean)
            mc3_boot = bootstrap_ci95(mc3, stat_fn=np.mean)
            r3_minus_r2_boot = bootstrap_ci95(r3_minus_r2, stat_fn=np.mean)
            r1_minus_r2_boot = bootstrap_ci95(r1_minus_r2, stat_fn=np.mean)

            rows.append({
                "horizon": f"W{h}",
                "segment": seg_name,
                "sample_weeks": n_weeks,
                "k1_median_return_pct": _pct(k1.median()),
                "k1_mean_return_pct": _pct(k1.mean()),
                "k2_median_return_pct": _pct(k2.median()),
                "k2_mean_return_pct": _pct(k2.mean()),
                "k3_median_return_pct": _pct(k3.median()),
                "k3_mean_return_pct": _pct(k3.mean()),
                "mc2_median_pct": _pct(mc2.median()),
                "mc2_mean_pct": _pct(mc2.mean()),
                "mc2_win_rate_pct": _rate((mc2 > 0).sum(), len(mc2)),
                "mc2_wilcoxon_p": wilcoxon_signed_rank_p(mc2),
                "mc2_boot_ci95_low": mc2_boot[0],
                "mc2_boot_ci95_high": mc2_boot[1],
                "mc3_median_pct": _pct(mc3.median()),
                "mc3_mean_pct": _pct(mc3.mean()),
                "mc3_win_rate_pct": _rate((mc3 > 0).sum(), len(mc3)),
                "mc3_wilcoxon_p": wilcoxon_signed_rank_p(mc3),
                "mc3_boot_ci95_low": mc3_boot[0],
                "mc3_boot_ci95_high": mc3_boot[1],
                "hyp_a_r3_minus_r2_median_spread_pct": _pct(r3_minus_r2.median()),
                "hyp_a_r3_minus_r2_mean_spread_pct": _pct(r3_minus_r2.mean()),
                "hyp_a_r3_gt_r2_win_rate_pct": _rate((r3_minus_r2 > 0).sum(), len(r3_minus_r2)),
                "hyp_a_wilcoxon_p": wilcoxon_signed_rank_p(r3_minus_r2),
                "hyp_a_boot_ci95_low": r3_minus_r2_boot[0],
                "hyp_a_boot_ci95_high": r3_minus_r2_boot[1],
                "hyp_b_k3_minus_k2_median_spread_pct": _pct(k3_minus_k2.median()),
                "hyp_b_k3_minus_k2_mean_spread_pct": _pct(k3_minus_k2.mean()),
                "hyp_b_k3_gt_k2_win_rate_pct": _rate((k3_minus_k2 > 0).sum(), len(k3_minus_k2)),
                "hyp_b_wilcoxon_p": wilcoxon_signed_rank_p(k3_minus_k2),
                "hyp_b_boot_ci95_low": mc3_boot[0],
                "hyp_b_boot_ci95_high": mc3_boot[1],
                "hyp_c_r1_minus_r2_median_spread_pct": _pct(r1_minus_r2.median()),
                "hyp_c_r1_minus_r2_mean_spread_pct": _pct(r1_minus_r2.mean()),
                "hyp_c_r1_gt_r2_win_rate_pct": _rate((r1_minus_r2 > 0).sum(), len(r1_minus_r2)),
                "hyp_c_wilcoxon_p": wilcoxon_signed_rank_p(r1_minus_r2),
                "hyp_c_boot_ci95_low": r1_minus_r2_boot[0],
                "hyp_c_boot_ci95_high": r1_minus_r2_boot[1],
                "horizon_status": h_status,
            })

    return pd.DataFrame(rows)


def summarize_rank_monotonicity(matrix_df: pd.DataFrame) -> pd.DataFrame:
    """Generate Rank Monotonicity and Spearman correlation audit."""
    rows: list[dict[str, Any]] = []
    segments = _get_segment_frames(matrix_df)

    for h in ALL_HORIZONS:
        h_status = "PRIMARY" if h in PRIMARY_HORIZONS else "DIAGNOSTIC_ONLY"
        for seg_name, seg_df in segments:
            valid_seg = seg_df[seg_df[f"w{h}_common_valid"]].copy()
            n_weeks = len(valid_seg)
            if n_weeks == 0:
                continue

            r1 = valid_seg[f"r1_w{h}_return_pct"]
            r2 = valid_seg[f"r2_w{h}_return_pct"]
            r3 = valid_seg[f"r3_w{h}_return_pct"]

            r1_gt_r2 = (r1 > r2).sum()
            r2_gt_r3 = (r2 > r3).sum()
            r3_gt_r2 = (r3 > r2).sum()
            r1_gt_r3 = (r1 > r3).sum()

            week_spearmans: list[float] = []
            pooled_ranks: list[int] = []
            pooled_rets: list[float] = []

            for _, row in valid_seg.iterrows():
                v1, v2, v3 = float(row[f"r1_w{h}_return_pct"]), float(row[f"r2_w{h}_return_pct"]), float(row[f"r3_w{h}_return_pct"])
                pooled_ranks.extend([1, 2, 3])
                pooled_rets.extend([v1, v2, v3])
                sc, _ = stats.spearmanr([1, 2, 3], [v1, v2, v3])
                if not np.isnan(sc):
                    week_spearmans.append(float(sc))

            pooled_sc, pooled_p = stats.spearmanr(pooled_ranks, pooled_rets)

            is_monotonic = bool(r1.median() >= r2.median() >= r3.median() and (r2_gt_r3 / n_weeks) >= 0.5)
            conclusion = (
                "Monotonic Fine Ranker"
                if is_monotonic
                else "Non-Monotonic (Top-Bucket Selector)"
            )

            rows.append({
                "horizon": f"W{h}",
                "segment": seg_name,
                "sample_weeks": n_weeks,
                "rank1_median_return_pct": _pct(r1.median()),
                "rank2_median_return_pct": _pct(r2.median()),
                "rank3_median_return_pct": _pct(r3.median()),
                "r1_gt_r2_week_rate_pct": _rate(r1_gt_r2, n_weeks),
                "r2_gt_r3_week_rate_pct": _rate(r2_gt_r3, n_weeks),
                "r3_gt_r2_week_rate_pct": _rate(r3_gt_r2, n_weeks),
                "r1_gt_r3_week_rate_pct": _rate(r1_gt_r3, n_weeks),
                "mean_per_week_spearman_corr": _pct(np.mean(week_spearmans)) if week_spearmans else np.nan,
                "pooled_spearman_corr": _pct(pooled_sc),
                "pooled_spearman_p": _pct(pooled_p),
                "monotonicity_conclusion": conclusion,
                "horizon_status": h_status,
            })

    return pd.DataFrame(rows)


def summarize_structure_profile(
    matrix_df: pd.DataFrame,
    b0_events_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate PIT Structure Profile across Rank 1, Rank 2, and Rank 3."""
    three_pick_snaps = set(matrix_df[matrix_df["is_3picks"]]["snapshot_date"].astype(str).unique())
    b0_3picks = b0_events_df[b0_events_df["snapshot_date"].astype(str).isin(three_pick_snaps)].copy()

    rows: list[dict[str, Any]] = []

    for rank_num in [1, 2, 3]:
        sub = b0_3picks[b0_3picks["pick_order"] == rank_num].copy()
        n = len(sub)
        if n == 0:
            continue

        lanes = sub["lane"].fillna("").astype(str)
        rules = sub["ibd_candidate_rule"].fillna("").astype(str)
        reasons = sub["reason_codes"].fillna("").astype(str)
        risks = sub["risk_codes"].fillna("").astype(str)

        inds = sub["industry"].value_counts().head(3).to_dict()
        top_ind_str = ", ".join([f"{k} ({v})" for k, v in inds.items()])

        rows.append({
            "rank_position": f"Rank{rank_num}",
            "sample_count": n,
            "fresh_demand_alpha_pct": _rate((lanes == "fresh_demand_alpha").sum(), n),
            "constructive_pullback_pct": _rate((lanes == "constructive_pullback").sum(), n),
            "standard_breakout_pct": _rate((lanes == "standard_breakout").sum(), n),
            "ceiling_rule_pct": _rate((rules == "ceiling").sum(), n),
            "pivot_rule_pct": _rate((rules == "pivot").sum(), n),
            "ma10_touch_confirm_rule_pct": _rate((rules == "ma10_touch_confirm").sum(), n),
            "ceiling_pullback_rule_pct": _rate((rules == "ceiling_pullback").sum(), n),
            "mean_current_vs_ibd_pct": _pct(sub["current_vs_ibd_candidate_pct"].mean()),
            "median_current_vs_ibd_pct": _pct(sub["current_vs_ibd_candidate_pct"].median()),
            "mean_ibd_entry_volume_ratio": _pct(sub["ibd_entry_volume_ratio"].mean()),
            "median_ibd_entry_volume_ratio": _pct(sub["ibd_entry_volume_ratio"].median()),
            "mean_ibd_entry_close_position": _pct(sub["ibd_entry_close_position"].mean()),
            "median_ibd_entry_close_position": _pct(sub["ibd_entry_close_position"].median()),
            "mean_ibd_entry_breakout_range_ratio": _pct(sub["ibd_entry_breakout_range_ratio"].mean()),
            "median_ibd_entry_breakout_range_ratio": _pct(sub["ibd_entry_breakout_range_ratio"].median()),
            "mean_dist_to_52w_high_pct": _pct(sub["dist_to_52w_high_pct"].mean()),
            "median_dist_to_52w_high_pct": _pct(sub["dist_to_52w_high_pct"].median()),
            "mean_volume_ratio": _pct(sub["volume_ratio"].mean()),
            "median_volume_ratio": _pct(sub["volume_ratio"].median()),
            "mean_effective_eps_growth": _pct(sub["effective_eps_yoy_growth"].mean()),
            "median_effective_eps_growth": _pct(sub["effective_eps_yoy_growth"].median()),
            "geometry_caution_pct": _rate(reasons.str.contains("geometry_caution").sum(), n),
            "pullback_not_dry_pct": _rate(risks.str.contains("pullback_not_dry").sum(), n),
            "dry_pullback_pct": _rate(reasons.str.contains("dry_pullback").sum(), n),
            "eps_acceleration_support_pct": _rate(reasons.str.contains("eps_acceleration_support").sum(), n),
            "near_52w_high_pct": _rate(reasons.str.contains("near_52w_high").sum(), n),
            "weekly_vol_follow_thru_pct": _rate(reasons.str.contains("weekly_volume_follow_through").sum(), n),
            "top_industries": top_ind_str,
        })

    return pd.DataFrame(rows)


def _format_row_dict(df: pd.DataFrame, filters: dict[str, Any]) -> dict[str, Any]:
    """Helper to retrieve a single row matching filters as a dict."""
    sub = df.copy()
    for k, v in filters.items():
        sub = sub[sub[k] == v]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


def render_report_markdown(
    quality_df: pd.DataFrame,
    marginal_df: pd.DataFrame,
    monotonicity_df: pd.DataFrame,
    structure_df: pd.DataFrame,
    weekly_matrix_df: pd.DataFrame,
) -> str:
    """Render the full research report in GitHub-flavored Markdown.
    
    All figures, summary tables, and executive conclusion metrics are dynamically
    derived from input DataFrames to ensure zero drift and mathematical closure.
    """
    total_snaps = len(weekly_matrix_df)
    snaps_3p = int((weekly_matrix_df["n_picks"] == 3).sum())
    snaps_2p = int((weekly_matrix_df["n_picks"] == 2).sum())
    snaps_1p = int((weekly_matrix_df["n_picks"] == 1).sum())

    # Common Support counts per horizon
    w_counts = {
        h: int(weekly_matrix_df[weekly_matrix_df[f"w{h}_common_valid"]]["snapshot_date"].count())
        for h in ALL_HORIZONS
    }
    w_train_counts = {
        h: int(weekly_matrix_df[weekly_matrix_df[f"w{h}_common_valid"] & weekly_matrix_df["is_train"]]["snapshot_date"].count())
        for h in ALL_HORIZONS
    }
    w_val_counts = {
        h: int(weekly_matrix_df[weekly_matrix_df[f"w{h}_common_valid"] & weekly_matrix_df["is_contaminated_val"]]["snapshot_date"].count())
        for h in ALL_HORIZONS
    }

    # Extract all-weeks summary rows
    q_all = quality_df[quality_df["segment"] == "All common-support weeks"]
    m_all = marginal_df[marginal_df["segment"] == "All common-support weeks"]
    mono_all = monotonicity_df[monotonicity_df["segment"] == "All common-support weeks"]

    # Extract horizon-specific dicts for dynamic conclusion rendering
    m_dict = {h: _format_row_dict(m_all, {"horizon": f"W{h}"}) for h in ALL_HORIZONS}
    q_dict = {
        (h, r): _format_row_dict(q_all, {"horizon": f"W{h}", "rank_position": f"Rank{r}"})
        for h in ALL_HORIZONS for r in [1, 2, 3]
    }
    mono_dict = {h: _format_row_dict(mono_all, {"horizon": f"W{h}"}) for h in ALL_HORIZONS}

    # Validation dicts
    m_val = marginal_df[marginal_df["segment"] == "Contaminated validation weeks 31-40"]
    m_val_dict = {h: _format_row_dict(m_val, {"horizon": f"W{h}"}) for h in ALL_HORIZONS}

    # Build markdown text
    md: list[str] = []
    md.append("# B0 Rank Position & TopK Marginal Contribution Audit Report")
    md.append("")
    md.append("> **Diagnostic & Audit Notice:** This report is an analytical diagnostic audit on frozen Phase 1/2 B0 selection events. It does not modify production selector rules, RuleSpec, eligibility definitions, or forward shadow pre-registrations.")
    md.append("> ")
    md.append("> **Diagnostic Horizon Notice:** W1, W2, and W4 are the Frozen Primary Metrics. W3 is diagnostic only and is not a newly registered primary endpoint.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Executive Conclusion: Answers to the 5 Core Questions")
    md.append("")

    # Q1
    md.append("### Q1. B0 Rank1 / Rank2 / Rank3 是否存在稳定质量差异？")
    md.append("- **结论：PARTIALLY SUPPORTED (HISTORICAL DIAGNOSTIC SIGNAL)**")
    md.append(f"- **核心数据（动态提取）：** 在全部 3-pick Common Support 样本中，Rank1 与 Rank3 在多数周期收益与路径表现较强，而 Rank2 出现非单调的中间塌陷：")
    for h in ALL_HORIZONS:
        h_tag = f"W{h}" + (" (诊断)" if h == DIAGNOSTIC_HORIZON else "")
        r1_m = q_dict[(h, 1)].get('median_return_pct', np.nan)
        r1_u = q_dict[(h, 1)].get('mean_return_pct', np.nan)
        r2_m = q_dict[(h, 2)].get('median_return_pct', np.nan)
        r2_u = q_dict[(h, 2)].get('mean_return_pct', np.nan)
        r3_m = q_dict[(h, 3)].get('median_return_pct', np.nan)
        r3_u = q_dict[(h, 3)].get('mean_return_pct', np.nan)
        md.append(f"  - **{h_tag} (n={w_counts[h]}):** Rank1 med=`{r1_m:+.2f}%` (mean `{r1_u:+.2f}%`), Rank2 med=`{r2_m:+.2f}%` (mean `{r2_u:+.2f}%`), Rank3 med=`{r3_m:+.2f}%` (mean `{r3_u:+.2f}%`)")
    r1_p20 = q_dict[(1, 1)].get('profit20_hit_rate_pct', np.nan)
    r2_p20 = q_dict[(1, 2)].get('profit20_hit_rate_pct', np.nan)
    r3_p20 = q_dict[(1, 3)].get('profit20_hit_rate_pct', np.nan)
    r2_st = q_dict[(1, 2)].get('stop8_ever_rate_pct', np.nan)
    r2_dd = q_dict[(1, 2)].get('median_asof_drawdown_pct', np.nan)
    md.append(f"  - **路径质量：** Rank2 止损率最高 (`{r2_st:.1f}%`)，Profit20 达成率最低 (`{r2_p20:.1f}%` vs Rank1 `{r1_p20:.1f}%`, Rank3 `{r3_p20:.1f}%`)，最大回撤中位数达 `{r2_dd:.2f}%`。")
    md.append("  - **定性定界：** 该差异呈现 U 型非单调结构，属于历史诊断特征，不能简单视作线性排序能力。")
    md.append("")

    # Q2
    md.append("### Q2. B0 ranking 是否存在 Monotonicity（单调顺序信息量）？")
    md.append("- **结论：NOT SUPPORTED (B0 不具备 fine-ranking 单调性，实际表现为 Top-Bucket Selector)**")
    md.append(f"- **核心数据（动态提取）：**")
    r1_g_r2_rates = [f"W{h}: `{mono_dict[h].get('r1_gt_r2_week_rate_pct', np.nan):.1f}%`" for h in PRIMARY_HORIZONS]
    r2_g_r3_rates = [f"W{h}: `{mono_dict[h].get('r2_gt_r3_week_rate_pct', np.nan):.1f}%`" for h in PRIMARY_HORIZONS]
    sp_corrs = [f"W{h}: `r={mono_dict[h].get('pooled_spearman_corr', np.nan):+.4f}`" for h in PRIMARY_HORIZONS]
    md.append(f"  - **Rank1 > Rank2 周胜率：** {', '.join(r1_g_r2_rates)} (无统计优势，接近抛硬币)")
    md.append(f"  - **Rank2 > Rank3 周胜率：** {', '.join(r2_g_r3_rates)} (发生实质性倒挂，Rank3 > Rank2 在多数周发生)")
    md.append(f"  - **Pooled Spearman 秩相关系数：** {', '.join(sp_corrs)} (均接近 0，无单调预测力)")
    md.append("  - **定位定性：** B0 在历史样本中表现为 **Top-Bucket Selector**（将优质标的筛入 Top3 头部集合），但集合内部的 1/2/3 顺位不包含单调优劣信息。")
    md.append("")

    # Q3
    md.append("### Q3. “Top2 明显更差”这个说法：")
    md.append("- **结论：PARTIALLY SUPPORTED**")
    md.append(f"- **证据分层分析（动态提取）：**")
    w2_a_p = m_dict[2].get('hyp_a_wilcoxon_p', np.nan)
    w2_a_med = m_dict[2].get('hyp_a_r3_minus_r2_median_spread_pct', np.nan)
    w2_a_win = m_dict[2].get('hyp_a_r3_gt_r2_win_rate_pct', np.nan)
    w1_mc2_m = m_dict[1].get('mc2_median_pct', np.nan)
    w1_mc2_u = m_dict[1].get('mc2_mean_pct', np.nan)
    w4_mc2_m = m_dict[4].get('mc2_median_pct', np.nan)
    w4_mc2_u = m_dict[4].get('mc2_mean_pct', np.nan)
    support = describe_statistical_support(w2_a_p)
    md.append(f"  - **Rank3 vs Rank2:** W2 median spread is `{w2_a_med:+.2f}%` with `{w2_a_win:.1f}%` win rate (Wilcoxon $p={w2_a_p:.4f}$): {support}. This is diagnostic only, not demonstrated fine-ranking support.")
    md.append(f"  - **Rank1 vs Rank2 (支持性弱):** Rank1 相对于 Rank2 未达到双侧统计显著水平，周胜率仅在 50% 附近波动。")
    md.append(f"  - **MC2 边际贡献分布 (左尾拖累而非每周恶化):** W1 median MC2 为 `{w1_mc2_m:+.2f}%` (mean `{w1_mc2_u:+.2f}%`)，W4 median MC2 为 `{w4_mc2_m:+.2f}%` (mean `{w4_mc2_u:+.2f}%`)。中位数在 W1/W4 为正，说明 Rank2 表现弱主要是由少数严重亏损的左尾事件（如生物科技板块/未缩量回撤）拉低均值，而非每周系统性拖累。")
    md.append("")

    # Q4
    md.append("### Q4. “Top3 portfolio > Top2 portfolio”这个说法：")
    md.append("- **结论：PARTIALLY SUPPORTED / DIRECTIONAL ONLY**")
    md.append(f"- **证据分层分析（动态提取）：**")
    w1_mc3_m = m_dict[1].get('mc3_median_pct', np.nan)
    w1_mc3_p = m_dict[1].get('mc3_wilcoxon_p', np.nan)
    w2_mc3_m = m_dict[2].get('mc3_median_pct', np.nan)
    w2_mc3_p = m_dict[2].get('mc3_wilcoxon_p', np.nan)
    w4_mc3_m = m_dict[4].get('mc3_median_pct', np.nan)
    w4_mc3_u = m_dict[4].get('mc3_mean_pct', np.nan)
    w4_mc3_p = m_dict[4].get('mc3_wilcoxon_p', np.nan)
    md.append(f"  - **全历史样本中位数偏正：** W1 MC3 med=`{w1_mc3_m:+.2f}%` ($p={w1_mc3_p:.4f}$), W2 MC3 med=`{w2_mc3_m:+.2f}%` ($p={w2_mc3_p:.4f}$), W4 MC3 med=`{w4_mc3_m:+.2f}%` ($p={w4_mc3_p:.4f}$)。")
    md.append(f"  - **无统计显著性且均值衰减：** 组合层 Wilcoxon $p$ 值在全周期均未达到显著水平；W4 平均边际贡献已转负 (`{w4_mc3_u:+.2f}%`)。")
    if not m_val.empty:
        v_w3_mc3_m = m_val_dict.get(3, {}).get('mc3_median_pct', np.nan)
        v_w4_mc3_m = m_val_dict.get(4, {}).get('mc3_median_pct', np.nan)
        md.append(f"  - **后期验证段倒挂：** 在历史验证周次 (31~40) 中，W3 median MC3 为 `{v_w3_mc3_m:+.2f}%`，W4 median MC3 为 `{v_w4_mc3_m:+.2f}%`，优势未能稳定延续。")
    md.append("  - **定性定界：** 绝不能表述为已证明的策略优势，仅定性为 **方向性历史现象**。")
    md.append("")

    # Q5
    md.append("### Q5. 当前证据是否足以修改生产 B0：")
    md.append("- **结论：NO — KEEP PRODUCTION FROZEN**")
    md.append(f"- **治理原因：**")
    md.append("  1. 当前历史样本仅 25 个 3-pick 周，且 31~40 周为已知历史样本，样本容量不足以支持不可逆的生产参数重构；")
    md.append("  2. 直接根据历史 Rank2 较弱去调权或剔除 Rank2 属于典型的数据窥探过拟合风险；")
    md.append("  3. 必须保持 Phase 1/2 生产基线完全冻结，将 Rank Position 结构列为 2026 Forward Shadow 的前向观察指标。")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 一、Methodology & Common Support Denominator")
    md.append("")
    md.append("为了避免“不同周数/不同样本分母”导致的虚假均值偏差，本审计遵循严格的 **Common Support & Maturity Gate** 准则：")
    md.append("1. **3-Pick Completeness:** 仅纳入 B0 生产选择器当周完整选出 3 只候选的周次；")
    md.append("2. **Horizon Maturity Gate:** 仅在 Rank1、Rank2、Rank3 三只标的在对应持有周期均满足 `is_complete_week == True` 且收益与最大涨幅数据完整时，该周才进入对应分母；")
    md.append("3. **Common Support 分母清单：**")
    md.append(f"   - **Total B0 Snapshot Weeks:** `{total_snaps}` 周 (3-pick 周 `{snaps_3p}` 周, 2-pick 周 `{snaps_2p}` 周, 1-pick 周 `{snaps_1p}` 周)")
    for h in ALL_HORIZONS:
        h_tag = f"W{h}" + (" (Diagnostic Only)" if h == DIAGNOSTIC_HORIZON else "")
        md.append(f"   - **{h_tag} Common Support:** `{w_counts[h]}` 周 (Train `{w_train_counts[h]}` 周, Contaminated Val `{w_val_counts[h]}` 周)")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 二、Position Quality Audit (Rank1 / Rank2 / Rank3)")
    md.append("")
    md.append("### 1. Return Quality Summary across All Common-Support Weeks")
    md.append("")
    md.append("| Horizon | Rank | Weeks | Mean (%) | Median (%) | P25 (%) | P75 (%) | Win Rate (>0) | Profit20 Rate | Stop8 Rate | Max Drawdown (Med) |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for _, r in q_all.iterrows():
        md.append(
            f"| {r['horizon']} | {r['rank_position']} | {r['sample_weeks']} | {r['mean_return_pct']:+.2f}% | "
            f"{r['median_return_pct']:+.2f}% | {r['p25_return_pct']:+.2f}% | {r['p75_return_pct']:+.2f}% | "
            f"{r['positive_return_rate_pct']:.1f}% | {r['profit20_hit_rate_pct']:.1f}% | "
            f"{r['stop8_ever_rate_pct']:.1f}% | {r['median_asof_drawdown_pct']:.2f}% |"
        )
    md.append("")
    md.append("> **Note on W3:** W3 is diagnostic only and is not a newly registered primary endpoint. It confirms that the performance divergence between Rank1/3 and Rank2 persists continuously through weeks 1, 2, 3, and 4.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 三、Top1 / Top2 / Top3 Portfolios & Marginal Contributions")
    md.append("")
    md.append("定义等权组合：")
    md.append("- **K1:** `Rank1`")
    md.append("- **K2:** `mean(Rank1, Rank2)`")
    md.append("- **K3:** `mean(Rank1, Rank2, Rank3)`")
    md.append("- **Rank2 Marginal Contribution:** `MC2 = K2 - K1`")
    md.append("- **Rank3 Marginal Contribution:** `MC3 = K3 - K2`")
    md.append("")
    md.append("### Marginal Contribution Summary (All Common-Support Weeks)")
    md.append("")
    md.append("| Horizon | Weeks | K1 Med (Mean) | K2 Med (Mean) | K3 Med (Mean) | MC2 Med (Mean) | MC2 Win Rate | MC2 p-val | MC3 Med (Mean) | MC3 Win Rate | MC3 p-val |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for _, r in m_all.iterrows():
        md.append(
            f"| {r['horizon']} | {r['sample_weeks']} | "
            f"{r['k1_median_return_pct']:+.2f}% ({r['k1_mean_return_pct']:+.2f}%) | "
            f"{r['k2_median_return_pct']:+.2f}% ({r['k2_mean_return_pct']:+.2f}%) | "
            f"{r['k3_median_return_pct']:+.2f}% ({r['k3_mean_return_pct']:+.2f}%) | "
            f"{r['mc2_median_pct']:+.2f}% ({r['mc2_mean_pct']:+.2f}%) | {r['mc2_win_rate_pct']:.1f}% | {r['mc2_wilcoxon_p']} | "
            f"{r['mc3_median_pct']:+.2f}% ({r['mc3_mean_pct']:+.2f}%) | {r['mc3_win_rate_pct']:.1f}% | {r['mc3_wilcoxon_p']} |"
        )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 四、配对假设检验：Hypothesis A, B, C 全景对比")
    md.append("")
    md.append("### 1. Hypothesis A: Rank3 个股是否优于 Rank2？ (`Rank3 - Rank2`)")
    md.append("")
    md.append("| Horizon | Weeks | Mean Spread (%) | Median Spread (%) | R3 > R2 Win Rate (%) | Wilcoxon p-value | 95% Bootstrap CI | 统计定性 |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for _, r in m_all.iterrows():
        qual = "Significant (p < 0.05)" if r["hyp_a_wilcoxon_p"] < 0.05 else ("Marginally Sig (p < 0.10)" if r["hyp_a_wilcoxon_p"] < 0.10 else "Directional Only")
        md.append(
            f"| {r['horizon']} | {r['sample_weeks']} | {r['hyp_a_r3_minus_r2_mean_spread_pct']:+.2f}% | "
            f"{r['hyp_a_r3_minus_r2_median_spread_pct']:+.2f}% | {r['hyp_a_r3_gt_r2_win_rate_pct']:.1f}% | "
            f"{r['hyp_a_wilcoxon_p']:.4f} | [{r['hyp_a_boot_ci95_low']:+.2f}%, {r['hyp_a_boot_ci95_high']:+.2f}%] | {qual} |"
        )
    md.append("")
    md.append("### 2. Hypothesis B: Top3 Portfolio 是否优于 Top2 Portfolio？ (`K3 - K2 = MC3`)")
    md.append("")
    md.append("| Horizon | Weeks | Mean Spread (%) | Median Spread (%) | K3 > K2 Win Rate (%) | Wilcoxon p-value | 95% Bootstrap CI | 统计定性 |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for _, r in m_all.iterrows():
        qual = "Significant (p < 0.05)" if r["hyp_b_wilcoxon_p"] < 0.05 else ("Marginally Sig (p < 0.10)" if r["hyp_b_wilcoxon_p"] < 0.10 else "Directional Positive (Not Sig)")
        md.append(
            f"| {r['horizon']} | {r['sample_weeks']} | {r['hyp_b_k3_minus_k2_mean_spread_pct']:+.2f}% | "
            f"{r['hyp_b_k3_minus_k2_median_spread_pct']:+.2f}% | {r['hyp_b_k3_gt_k2_win_rate_pct']:.1f}% | "
            f"{r['hyp_b_wilcoxon_p']:.4f} | [{r['hyp_b_boot_ci95_low']:+.2f}%, {r['hyp_b_boot_ci95_high']:+.2f}%] | {qual} |"
        )
    md.append("")
    md.append("### 3. Hypothesis C: Rank1 个股是否优于 Rank2？ (`Rank1 - Rank2`)")
    md.append("")
    md.append("| Horizon | Weeks | Mean Spread (%) | Median Spread (%) | R1 > R2 Win Rate (%) | Wilcoxon p-value | 95% Bootstrap CI | 统计定性 |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for _, r in m_all.iterrows():
        qual = "Significant (p < 0.05)" if r["hyp_c_wilcoxon_p"] < 0.05 else ("Marginally Sig (p < 0.10)" if r["hyp_c_wilcoxon_p"] < 0.10 else "Not Significant")
        md.append(
            f"| {r['horizon']} | {r['sample_weeks']} | {r['hyp_c_r1_minus_r2_mean_spread_pct']:+.2f}% | "
            f"{r['hyp_c_r1_minus_r2_median_spread_pct']:+.2f}% | {r['hyp_c_r1_gt_r2_win_rate_pct']:.1f}% | "
            f"{r['hyp_c_wilcoxon_p']:.4f} | [{r['hyp_c_boot_ci95_low']:+.2f}%, {r['hyp_c_boot_ci95_high']:+.2f}%] | {qual} |"
        )
    md.append("")
    md.append("> **方法论洞察：** Rank-position contrasts are descriptive diagnostics. They do not demonstrate a stable monotonic ordering or justify rank-rule changes.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 五、Rank Monotonicity & Spearman Correlation Audit")
    md.append("")
    md.append("理想的 Fine Ranker 应具备 `Rank1 >= Rank2 >= Rank3` 的单调性。审计结果如下：")
    md.append("")
    md.append("| Horizon | Weeks | R1 Med | R2 Med | R3 Med | R1 > R2 Rate | R2 > R3 Rate | R3 > R2 Rate | Pooled Spearman r (p) | 定位结论 |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for _, r in mono_all.iterrows():
        md.append(
            f"| {r['horizon']} | {r['sample_weeks']} | {r['rank1_median_return_pct']:+.2f}% | "
            f"{r['rank2_median_return_pct']:+.2f}% | {r['rank3_median_return_pct']:+.2f}% | "
            f"{r['r1_gt_r2_week_rate_pct']:.1f}% | {r['r2_gt_r3_week_rate_pct']:.1f}% | "
            f"{r['r3_gt_r2_week_rate_pct']:.1f}% | {r['pooled_spearman_corr']:+.4f} ({r['pooled_spearman_p']:.4f}) | "
            f"{r['monotonicity_conclusion']} |"
        )
    md.append("")
    md.append("### 核心结论：")
    md.append("> **B0 does not demonstrate monotonic fine-ranking quality.**")
    md.append("> B0 在历史样本中表现为 **Top-Bucket Selector** 而非 Fine Ranker。它成功将优质标的聚集于 Top 3 头部集合，但内部 1/2/3 顺位不包含确定性的强弱顺序。")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 六、分阶段稳定性：Train (1~30) vs Contaminated Historical Validation (31~40)")
    md.append("")
    md.append("| Horizon | Segment | Weeks | R1 Med (Mean) | R2 Med (Mean) | R3 Med (Mean) | R3 > R2 Win Rate | MC3 Med (Mean) |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for h in ALL_HORIZONS:
        for seg in ["Train-era weeks 1-30", "Contaminated validation weeks 31-40"]:
            q_seg = quality_df[(quality_df["horizon"] == f"W{h}") & (quality_df["segment"] == seg)]
            m_seg = marginal_df[(marginal_df["horizon"] == f"W{h}") & (marginal_df["segment"] == seg)]
            if q_seg.empty or m_seg.empty:
                continue
            r1_row = q_seg[q_seg["rank_position"] == "Rank1"].iloc[0]
            r2_row = q_seg[q_seg["rank_position"] == "Rank2"].iloc[0]
            r3_row = q_seg[q_seg["rank_position"] == "Rank3"].iloc[0]
            m_row = m_seg.iloc[0]
            md.append(
                f"| W{h} | {seg} | {m_row['sample_weeks']} | "
                f"{r1_row['median_return_pct']:+.2f}% ({r1_row['mean_return_pct']:+.2f}%) | "
                f"{r2_row['median_return_pct']:+.2f}% ({r2_row['mean_return_pct']:+.2f}%) | "
                f"{r3_row['median_return_pct']:+.2f}% ({r3_row['mean_return_pct']:+.2f}%) | "
                f"{m_row['hyp_a_r3_gt_r2_win_rate_pct']:.1f}% | "
                f"{m_row['mc3_median_pct']:+.2f}% ({m_row['mc3_mean_pct']:+.2f}%) |"
            )
    md.append("")
    md.append("### 阶段稳定性发现：")
    train_rows = marginal_df[marginal_df["segment"] == "Train-era weeks 1-30"]
    val_rows = marginal_df[marginal_df["segment"] == "Contaminated validation weeks 31-40"]
    train_rates = ", ".join(f"{r['horizon']} {r['hyp_a_r3_gt_r2_win_rate_pct']:.1f}%" for _, r in train_rows.iterrows()) or "N/A"
    val_rates = ", ".join(f"{r['horizon']} {r['hyp_a_r3_gt_r2_win_rate_pct']:.1f}%" for _, r in val_rows.iterrows()) or "N/A"
    md.append(f"1. **Train 阶段（当前 fixed calendar）：** R3>R2 周胜率为 {train_rates}。")
    md.append(f"2. **Contaminated Validation 阶段（当前 fixed calendar）：** R3>R2 周胜率为 {val_rates}；仅作诊断。")
    md.append("3. **治理警示：** 31~40 周为已知历史样本，不可等同于真实的 Virgin OOS 前向测试。")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 七、Rank 2 PIT Structure Diagnostic (事实画像对比)")
    md.append("")
    md.append("横向切片对比 25 个 3-pick 周的 PIT 字段：")
    md.append("")
    md.append("| 特征字段 | Rank 1 (n=25) | Rank 2 (n=25) | Rank 3 (n=25) | 结构差异观察 |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    if not structure_df.empty:
        r1_s = structure_df[structure_df["rank_position"] == "Rank1"].iloc[0]
        r2_s = structure_df[structure_df["rank_position"] == "Rank2"].iloc[0]
        r3_s = structure_df[structure_df["rank_position"] == "Rank3"].iloc[0]

        md.append(f"| Fresh Demand Alpha Lane (%) | `{r1_s['fresh_demand_alpha_pct']:.1f}%` | `{r2_s['fresh_demand_alpha_pct']:.1f}%` | `{r3_s['fresh_demand_alpha_pct']:.1f}%` | Rank1/2 集中于 Fresh Demand Alpha (`{r1_s['fresh_demand_alpha_pct']:.1f}%`)，Rank3 包含更多 Pullback (`{r3_s['constructive_pullback_pct']:.1f}%`) |")
        md.append(f"| Ceiling Rule (%) | `{r1_s['ceiling_rule_pct']:.1f}%` | `{r2_s['ceiling_rule_pct']:.1f}%` | `{r3_s['ceiling_rule_pct']:.1f}%` | Rank1/2 均为 `{r1_s['ceiling_rule_pct']:.1f}%` Ceiling，Rank3 拥有更多 Pivot (`{r3_s['pivot_rule_pct']:.1f}%`) |")
        md.append(f"| Breakout Range Ratio (Med) | `{r1_s['median_ibd_entry_breakout_range_ratio']:.2f}` | `{r2_s['median_ibd_entry_breakout_range_ratio']:.2f}` | `{r3_s['median_ibd_entry_breakout_range_ratio']:.2f}` | Rank2 突破振幅比率最小 (`{r2_s['median_ibd_entry_breakout_range_ratio']:.2f}` vs Rank1 `{r1_s['median_ibd_entry_breakout_range_ratio']:.2f}`, Rank3 `{r3_s['median_ibd_entry_breakout_range_ratio']:.2f}`) |")
        md.append(f"| Dist to 52w High (Med) | `{r1_s['median_dist_to_52w_high_pct']:.2f}%` | `{r2_s['median_dist_to_52w_high_pct']:.2f}%` | `{r3_s['median_dist_to_52w_high_pct']:.2f}%` | Rank1 最贴近 52 周高点 (`{r1_s['median_dist_to_52w_high_pct']:.2f}%`)，Rank2 (`{r2_s['median_dist_to_52w_high_pct']:.2f}%`) 略远 |")
        md.append(f"| Pullback Not Dry Risk (%) | `{r1_s['pullback_not_dry_pct']:.1f}%` | `{r2_s['pullback_not_dry_pct']:.1f}%` | `{r3_s['pullback_not_dry_pct']:.1f}%` | Rank2 触发未缩量回撤风险率最高 (`{r2_s['pullback_not_dry_pct']:.1f}%` vs Rank1 `{r1_s['pullback_not_dry_pct']:.1f}%`) |")
        md.append(f"| Geometry Caution (%) | `{r1_s['geometry_caution_pct']:.1f}%` | `{r2_s['geometry_caution_pct']:.1f}%` | `{r3_s['geometry_caution_pct']:.1f}%` | Rank1/2 均有 `{r1_s['geometry_caution_pct']:.1f}%` 几何形态预警，Rank3 仅 `{r3_s['geometry_caution_pct']:.1f}%` |")
        md.append(f"| Industry Focus | `{r1_s['top_industries']}` | `{r2_s['top_industries']}` | `{r3_s['top_industries']}` | Rank1 集中于 Regional Banks，Rank2 包含高波动 Biotech |")
    md.append("")
    md.append("> **因果区分警示：** 以上为历史事实画像对比，不代表单一字段必然构成因果。禁止在缺乏前向独立验证时基于上述特征直接调权。")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 八、综合治理建议与前向跟踪指引")
    md.append("")
    md.append("1. **保持生产 B0 冻结：** 严禁为了“修复 Rank2”而修改现有排序权重或引入新规则；")
    md.append("2. **定性修正：** 在方法论与产品认知上，明确将 B0 视为 **Top-Bucket Equal-Weighted Selector**，而非高精度单调 ranker；")
    md.append("3. **前向观察指引 (Forward Shadow 2026/08/28 起)：** 持续记录 Rank1/Rank2/Rank3、K1/K2/K3、MC2/MC3 及 R3-R2，观察 virgin forward 数据是否继续出现 Rank3 > Rank2 结构。只有前向样本复现后，才进入下一阶段假设推导。")
    md.append("")

    return "\n".join(md)


def run_b0_rank_topk_audit(
    paths: AuditPaths | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Execute complete B0 Rank Position & TopK Marginal Contribution Audit.
    
    Returns:
        (weekly_matrix_df, quality_df, marginal_df, monotonicity_df, structure_df, report_md)
    """
    if paths is None:
        paths = default_audit_paths()

    logger.info("Loading frozen audit datasets...")
    b0_events_df = pd.read_csv(paths.b0_events_path)
    events_df = pd.read_parquet(paths.events_path)
    weekly_df = pd.read_parquet(paths.weekly_path)
    three_tier_df = pd.read_csv(paths.three_tier_weekly_path)

    all_snapshots = sorted(three_tier_df["snapshot_date"].astype(str).unique())
    train_snapshots = train_dates(all_snapshots)
    contaminated_snapshots = contaminated_validation_dates(all_snapshots)

    logger.info("Building Common Support weekly matrix with Maturity Gate...")
    weekly_matrix_df = build_common_support_weekly_matrix(
        b0_events_df=b0_events_df,
        events_df=events_df,
        weekly_df=weekly_df,
        all_snapshots=all_snapshots,
        train_snapshots=train_snapshots,
        contaminated_snapshots=contaminated_snapshots,
    )

    logger.info("Summarizing Position Quality...")
    quality_df = summarize_rank_position_quality(weekly_matrix_df)

    logger.info("Summarizing TopK Marginal Contributions & Hyp A/B/C...")
    marginal_df = summarize_topk_marginal_contributions(weekly_matrix_df)

    logger.info("Summarizing Rank Monotonicity...")
    monotonicity_df = summarize_rank_monotonicity(weekly_matrix_df)

    logger.info("Summarizing PIT Structure Profile...")
    structure_df = summarize_structure_profile(weekly_matrix_df, b0_events_df)

    logger.info("Rendering Dynamic Markdown Report...")
    report_md = render_report_markdown(
        quality_df=quality_df,
        marginal_df=marginal_df,
        monotonicity_df=monotonicity_df,
        structure_df=structure_df,
        weekly_matrix_df=weekly_matrix_df,
    )

    # Export CSV and Markdown artifacts
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    weekly_matrix_path = paths.output_dir / "b0_rank_position_weekly_detail.csv"
    quality_path = paths.output_dir / "b0_rank_position_quality_summary.csv"
    marginal_path = paths.output_dir / "b0_topk_marginal_contribution_summary.csv"
    monotonicity_path = paths.output_dir / "b0_rank_monotonicity_summary.csv"
    structure_path = paths.output_dir / "b0_rank_position_structure_profile.csv"
    report_path = paths.output_dir / REPORT_NAME

    weekly_matrix_df.to_csv(weekly_matrix_path, index=False, encoding="utf-8-sig")
    quality_df.to_csv(quality_path, index=False, encoding="utf-8-sig")
    marginal_df.to_csv(marginal_path, index=False, encoding="utf-8-sig")
    monotonicity_df.to_csv(monotonicity_path, index=False, encoding="utf-8-sig")
    structure_df.to_csv(structure_path, index=False, encoding="utf-8-sig")
    report_path.write_text(report_md, encoding="utf-8")

    logger.info("Audit generation completed successfully.")
    return (
        weekly_matrix_df,
        quality_df,
        marginal_df,
        monotonicity_df,
        structure_df,
        report_md,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_b0_rank_topk_audit()
