from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.b0_top3_quality_audit.generate_b0_quality_vs_matched_random_report import (
    summarize_horizon_comparison,
    summarize_percentile_quality,
    summarize_stability,
)


def _sample_random_distribution() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "snapshot_date": "2026-01-02",
                "actual_picks_count": 0,
                "status": "ZERO_PICKS_MATCHED",
                "b0_is_w1_valid": False,
                "b0_is_w2_valid": False,
                "b0_is_w4_valid": False,
                "b0_actual_w1_mean_return_pct": np.nan,
                "b0_actual_w2_mean_return_pct": np.nan,
                "b0_actual_w4_mean_return_pct": np.nan,
                "w1_mean_return_pct_p50": np.nan,
                "w2_mean_return_pct_p50": np.nan,
                "w4_mean_return_pct_p50": np.nan,
                "b0_w1_return_percentile": np.nan,
                "b0_w2_return_percentile": np.nan,
                "b0_w4_return_percentile": np.nan,
            },
            {
                "snapshot_date": "2026-01-09",
                "actual_picks_count": 2,
                "status": np.nan,
                "b0_is_w1_valid": True,
                "b0_is_w2_valid": True,
                "b0_is_w4_valid": True,
                "b0_actual_w1_mean_return_pct": 2.0,
                "b0_actual_w2_mean_return_pct": 1.0,
                "b0_actual_w4_mean_return_pct": 4.0,
                "w1_mean_return_pct_p50": 1.0,
                "w2_mean_return_pct_p50": 2.0,
                "w4_mean_return_pct_p50": 3.0,
                "b0_w1_return_percentile": 60.0,
                "b0_w2_return_percentile": 40.0,
                "b0_w4_return_percentile": 80.0,
            },
            {
                "snapshot_date": "2026-01-16",
                "actual_picks_count": 3,
                "status": np.nan,
                "b0_is_w1_valid": True,
                "b0_is_w2_valid": False,
                "b0_is_w4_valid": True,
                "b0_actual_w1_mean_return_pct": -1.0,
                "b0_actual_w2_mean_return_pct": np.nan,
                "b0_actual_w4_mean_return_pct": 1.0,
                "w1_mean_return_pct_p50": 0.0,
                "w2_mean_return_pct_p50": 0.5,
                "w4_mean_return_pct_p50": 2.0,
                "b0_w1_return_percentile": 30.0,
                "b0_w2_return_percentile": np.nan,
                "b0_w4_return_percentile": 45.0,
            },
        ]
    )


def test_horizon_comparison_excludes_zero_pick_and_immature_weeks():
    summary = summarize_horizon_comparison(_sample_random_distribution())

    w1 = summary[summary["horizon"] == "W1"].iloc[0]
    assert w1["valid_weeks"] == 2
    assert w1["zero_pick_weeks_excluded"] == 1
    assert w1["b0_median_return_pct"] == 0.5
    assert w1["matched_random_p50_median_return_pct"] == 0.5
    assert w1["paired_spread_median_pct"] == 0.0
    assert w1["beat_random_p50_rate_pct"] == 50.0

    w2 = summary[summary["horizon"] == "W2"].iloc[0]
    assert w2["valid_weeks"] == 1
    assert w2["paired_spread_median_pct"] == -1.0


def test_percentile_summary_uses_valid_b0_percentiles_only():
    summary = summarize_percentile_quality(_sample_random_distribution())

    w4 = summary[summary["horizon"] == "W4"].iloc[0]
    assert w4["valid_percentile_weeks"] == 2
    assert w4["median_percentile"] == 62.5
    assert w4["mean_percentile"] == 62.5
    assert w4["weeks_gt_p50_pct"] == 50.0
    assert w4["weeks_gt_p75_pct"] == 50.0
    assert w4["weeks_gt_p90_pct"] == 0.0


def test_stability_segments_keep_mature_denominators():
    summary = summarize_stability(_sample_random_distribution())

    w1_all = summary[(summary["horizon"] == "W1") & (summary["segment"] == "All valid weeks")].iloc[0]
    assert w1_all["valid_weeks"] == 2
    assert w1_all["beat_random_p50_rate_pct"] == 50.0

    w4_late = summary[(summary["horizon"] == "W4") & (summary["segment"] == "Late half")].iloc[0]
    assert w4_late["valid_weeks"] == 1
    assert w4_late["beat_random_p50_rate_pct"] == 0.0
