import pandas as pd

from dashboard.data_utils import build_chart_data, build_kpis, normalize_pool_df


def chart_sample_df() -> pd.DataFrame:
    return normalize_pool_df(
        pd.DataFrame(
            [
                {
                    "code": "AAA",
                    "signal_source": "pivot",
                    "ibd_candidate_rule": "pivot",
                    "ibd_entry_valid": True,
                    "ibd_entry_volume_ratio": 2.0,
                    "ibd_entry_close_position": 0.80,
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
                    "code": "BBB",
                    "signal_source": "pivot",
                    "ibd_candidate_rule": "pivot",
                    "ibd_entry_valid": False,
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
                    "code": "CCC",
                    "signal_source": "ceiling_breakout",
                    "ibd_candidate_rule": "ceiling_pullback",
                    "ibd_entry_valid": True,
                    "ibd_entry_volume_ratio": 4.0,
                    "ibd_entry_close_position": 0.90,
                    "ibd_entry_breakout_range_ratio": 1.50,
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
                    "code": "DDD",
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
    assert route["total_count"].tolist() == [1, 1, 2]
    assert route["valid_count"].tolist() == [0, 1, 1]
    assert route["valid_rate_pct"].tolist() == [0.0, 100.0, 50.0]
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

    assert action_map["code"].tolist() == ["AAA", "CCC"]
    assert action_map.loc[action_map["code"].eq("AAA"), "entry_status"].item() == "IBD valid"
    assert {"sector", "industry", "dry_status", "touched_ema10_count", "touched_ema10_jittered", "volume_ratio"}.issubset(action_map.columns)


def test_volume_close_matrix_structure():
    charts = build_chart_data(chart_sample_df())
    matrix = charts["volume_close_matrix"]

    assert {"code", "ibd_entry_close_position", "volume_ratio"}.issubset(matrix.columns)
    assert not matrix.empty


def test_breakout_quadrant_profile_structure():
    charts = build_chart_data(chart_sample_df())
    profile = charts["breakout_quadrant"]

    assert {"quadrant", "count", "share_pct", "tickers"}.issubset(profile.columns)
    assert len(profile) == 4


def test_sector_concentration_counts_current_rows_and_share():
    charts = build_chart_data(chart_sample_df())
    concentration = charts["sector_concentration"].sort_values("sector")

    assert concentration["sector"].tolist() == ["Finance", "Technology Services"]
    assert concentration["row_count"].tolist() == [2, 2]
    assert concentration["share_pct"].tolist() == [50.0, 50.0]


def test_kpis_are_based_on_filtered_dataframe_only():
    kpis = build_kpis(chart_sample_df().iloc[:3])

    assert kpis["filtered_rows"] == 3
    assert kpis["ibd_valid_rate_pct"] == round(2 / 3 * 100, 2)
    assert kpis["median_ibd_entry_volume_ratio"] == 3.0
    assert kpis["median_ibd_entry_close_position"] == 0.80
    assert kpis["median_ibd_entry_breakout_range_ratio_valid"] == 1.35
