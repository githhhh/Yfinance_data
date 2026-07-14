import pandas as pd

from dashboard.data_utils import build_chart_data, build_kpis, normalize_pool_df


def chart_sample_df() -> pd.DataFrame:
    return normalize_pool_df(
        pd.DataFrame(
            [
                {
                    "code": "GAP",
                    "signal_source": "pivot",
                    "ibd_candidate_rule": "pivot",
                    "ibd_entry_valid": True,
                    "ibd_entry_volume_ratio": 2.0,
                    "ibd_entry_close_position": 0.60,
                    "ibd_entry_breakout_range_ratio": 1.20,
                    "ibd_entry_price": 10.0,
                    "pct_above_ceiling": 4.0,
                    "volume_ratio": 1.4,
                    "touched_ema10_count": 2,
                    "pullback_pct_off_peak": -3.0,
                    "pullback_v_is_dry": True,
                    "sector": "Technology Services",
                    "industry": "Software",
                },
                {
                    "code": "SOLID",
                    "signal_source": "pivot",
                    "ibd_candidate_rule": "pivot",
                    "ibd_entry_valid": True,
                    "ibd_entry_volume_ratio": 2.0,
                    "ibd_entry_close_position": 0.80,
                    "ibd_entry_breakout_range_ratio": 0.60,
                    "ibd_entry_price": 10.0,
                    "pct_above_ceiling": 4.0,
                    "volume_ratio": 1.4,
                    "touched_ema10_count": 2,
                    "pullback_pct_off_peak": -3.0,
                    "pullback_v_is_dry": True,
                    "sector": "Technology Services",
                    "industry": "Software",
                },
                {
                    "code": "TRAP",
                    "signal_source": "pivot",
                    "ibd_candidate_rule": "pivot",
                    "ibd_entry_valid": True,
                    "ibd_entry_volume_ratio": 3.0,
                    "ibd_entry_close_position": 0.40,
                    "ibd_entry_breakout_range_ratio": 0.60,
                    "ibd_entry_price": 12.0,
                    "pct_above_ceiling": 8.0,
                    "volume_ratio": 1.8,
                    "touched_ema10_count": 5,
                    "pullback_pct_off_peak": 4.0,
                    "pullback_v_is_dry": False,
                    "sector": "Technology Services",
                    "industry": "Software",
                },
                {
                    "code": "MOD",
                    "signal_source": "ceiling_breakout",
                    "ibd_candidate_rule": "ceiling_pullback",
                    "ibd_entry_valid": True,
                    "ibd_entry_volume_ratio": 4.0,
                    "ibd_entry_close_position": 0.70,
                    "ibd_entry_breakout_range_ratio": 0.20,
                    "ibd_entry_price": 14.0,
                    "pct_above_ceiling": 2.0,
                    "volume_ratio": 2.2,
                    "touched_ema10_count": 1,
                    "pullback_pct_off_peak": -1.0,
                    "pullback_v_is_dry": None,
                    "sector": "Finance",
                    "industry": "Regional Banks",
                },
                {
                    "code": "MARG",
                    "signal_source": "ceiling_breakout",
                    "ibd_candidate_rule": "ceiling",
                    "ibd_entry_valid": True,
                    "ibd_entry_volume_ratio": 1.7,
                    "ibd_entry_close_position": 0.55,
                    "ibd_entry_breakout_range_ratio": 0.08,
                    "ibd_entry_price": 15.0,
                    "pct_above_ceiling": 1.5,
                    "volume_ratio": 1.1,
                    "touched_ema10_count": 3,
                    "pullback_pct_off_peak": 7.0,
                    "pullback_v_is_dry": None,
                    "sector": "Finance",
                    "industry": "Regional Banks",
                },
                {
                    "code": "PEND",
                    "signal_source": "ceiling_breakout",
                    "ibd_candidate_rule": "ceiling",
                    "ibd_entry_valid": None,
                    "ibd_entry_volume_ratio": None,
                    "ibd_entry_close_position": 0.50,
                    "ibd_entry_breakout_range_ratio": 0.80,
                    "ibd_entry_price": 16.0,
                    "pct_above_ceiling": 1.0,
                    "volume_ratio": 0.9,
                    "touched_ema10_count": 3,
                    "pullback_pct_off_peak": 7.0,
                    "pullback_v_is_dry": None,
                    "sector": "Finance",
                    "industry": "Regional Banks",
                },
            ]
        )
    )


def test_route_quality_groups_by_candidate_rule():
    charts = build_chart_data(chart_sample_df())
    route = charts["route_quality"].sort_values("ibd_candidate_rule")

    assert route["ibd_candidate_rule"].tolist() == ["ceiling", "ceiling_pullback", "pivot"]
    assert route["total_count"].tolist() == [2, 1, 3]
    assert route["valid_count"].tolist() == [1, 1, 3]
    assert route["valid_rate_pct"].tolist() == [50.0, 100.0, 100.0]
    assert "median_ibd_entry_close_position" in route.columns
    assert "median_volume_ratio" in route.columns
    assert "median_ibd_entry_breakout_range_ratio" in route.columns


def test_route_quality_excludes_empty_candidate_rules():
    df = normalize_pool_df(
        pd.DataFrame(
            [
                {"code": "AAA", "ibd_candidate_rule": "pivot", "ibd_entry_valid": True},
                {"code": "BBB", "ibd_candidate_rule": "", "ibd_entry_valid": False},
                {"code": "CCC", "ibd_candidate_rule": "(empty)", "ibd_entry_valid": False},
                {"code": "DDD", "ibd_candidate_rule": None, "ibd_entry_valid": False},
            ]
        )
    )

    route = build_chart_data(df)["route_quality"]

    assert route["ibd_candidate_rule"].tolist() == ["pivot"]
    assert len(route) == 1


def test_route_quality_tolerates_missing_optional_metric_columns():
    df = normalize_pool_df(
        pd.DataFrame(
            [
                {
                    "code": "AAA",
                    "ibd_candidate_rule": "pivot",
                    "ibd_entry_valid": True,
                }
            ]
        )
    )

    route = build_chart_data(df)["route_quality"]

    assert route["total_count"].tolist() == [1]
    assert pd.isna(route.loc[0, "median_ibd_entry_volume_ratio"])
    assert pd.isna(route.loc[0, "median_ibd_entry_close_position"])


def test_trend_volume_map_only_keeps_valid_ibd_entries():
    charts = build_chart_data(chart_sample_df())
    action_map = charts["trend_volume_map"]

    assert action_map["code"].tolist() == ["GAP", "SOLID", "TRAP", "MOD", "MARG"]
    assert action_map.loc[action_map["code"].eq("GAP"), "entry_status"].item() == "IBD valid"
    assert {"sector", "industry", "dry_status", "touched_ema10_count", "touched_ema10_jittered", "volume_ratio"}.issubset(action_map.columns)


def test_volume_close_matrix_structure():
    charts = build_chart_data(chart_sample_df())
    matrix = charts["volume_close_matrix"]

    assert {"code", "ibd_entry_close_position", "volume_ratio"}.issubset(matrix.columns)
    assert not matrix.empty


def test_kpis_are_based_on_filtered_dataframe_only():
    kpis = build_kpis(chart_sample_df().iloc[:3])

    assert kpis["filtered_rows"] == 3
    assert kpis["median_current_vs_ibd_candidate_pct"] is None
    assert kpis["median_ibd_entry_volume_ratio"] == 2.0
    assert kpis["median_volume_ratio"] == 1.4
