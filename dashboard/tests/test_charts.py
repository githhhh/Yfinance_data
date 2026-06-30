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
                    "ibd_entry_close_vs_trigger_pct": 0.02,
                    "ibd_entry_price": 10.0,
                    "pct_above_ceiling": 4.0,
                    "volume_ratio": 1.4,
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
                    "ibd_entry_close_vs_trigger_pct": 0.04,
                    "ibd_entry_price": 12.0,
                    "pct_above_ceiling": 8.0,
                    "volume_ratio": 1.8,
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
                    "ibd_entry_close_vs_trigger_pct": 0.06,
                    "ibd_entry_price": 14.0,
                    "pct_above_ceiling": 2.0,
                    "volume_ratio": 2.2,
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
                    "ibd_entry_close_vs_trigger_pct": 0.08,
                    "ibd_entry_price": 16.0,
                    "pct_above_ceiling": 1.0,
                    "volume_ratio": 0.9,
                    "pullback_pct_off_peak": 7.0,
                    "pullback_v_is_dry": None,
                    "sector": "Finance",
                    "industry": "Regional Banks",
                },
            ]
        )
    )


def test_valid_rate_chart_counts_current_rows_by_signal_source():
    charts = build_chart_data(chart_sample_df())
    valid_rate = charts["ibd_valid_rate_by_signal_source"].sort_values("signal_source")

    assert valid_rate["signal_source"].tolist() == ["ceiling_breakout", "pivot"]
    assert valid_rate["total_count"].tolist() == [2, 2]
    assert valid_rate["valid_count"].tolist() == [1, 1]
    assert valid_rate["invalid_count"].tolist() == [1, 1]
    assert valid_rate["valid_rate_pct"].tolist() == [50.0, 50.0]


def test_signal_quality_matrix_groups_by_source_and_candidate_rule():
    charts = build_chart_data(chart_sample_df())
    matrix = charts["signal_quality_matrix"].sort_values(["signal_source", "ibd_candidate_rule"])

    assert matrix["signal_source"].tolist() == ["ceiling_breakout", "ceiling_breakout", "pivot"]
    assert matrix["ibd_candidate_rule"].tolist() == ["ceiling", "ceiling_pullback", "pivot"]
    assert matrix["total_count"].tolist() == [1, 1, 2]
    assert matrix["valid_count"].tolist() == [0, 1, 1]
    assert matrix["valid_rate_pct"].tolist() == [0.0, 100.0, 50.0]
    assert "median_pct_above_ceiling" in matrix.columns
    assert "median_volume_ratio" in matrix.columns


def test_signal_quality_matrix_tolerates_missing_optional_metric_columns():
    df = normalize_pool_df(
        pd.DataFrame(
            [
                {
                    "code": "AAA",
                    "signal_source": "pivot",
                    "ibd_candidate_rule": "pivot",
                    "ibd_entry_valid": True,
                }
            ]
        )
    )

    matrix = build_chart_data(df)["signal_quality_matrix"]

    assert matrix["total_count"].tolist() == [1]
    assert pd.isna(matrix.loc[0, "median_ibd_entry_volume_ratio"])
    assert pd.isna(matrix.loc[0, "median_pct_above_ceiling"])


def test_structure_action_map_keeps_review_rows_with_structure_axes():
    charts = build_chart_data(chart_sample_df())
    action_map = charts["structure_action_map"]

    assert action_map["code"].tolist() == ["AAA", "BBB", "CCC", "DDD"]
    assert action_map.loc[action_map["code"].eq("AAA"), "entry_status"].item() == "IBD valid"
    assert action_map.loc[action_map["code"].eq("BBB"), "entry_status"].item() == "IBD invalid"
    assert action_map.loc[action_map["code"].eq("DDD"), "entry_status"].item() == "Pending"
    assert {"sector", "industry", "dry_status", "volume_ratio", "pct_above_ceiling"}.issubset(action_map.columns)


def test_sector_concentration_counts_current_rows_and_share():
    charts = build_chart_data(chart_sample_df())
    concentration = charts["sector_concentration"].sort_values("sector")

    assert concentration["sector"].tolist() == ["Finance", "Technology Services"]
    assert concentration["row_count"].tolist() == [2, 2]
    assert concentration["share_pct"].tolist() == [50.0, 50.0]


def test_scatter_chart_uses_only_rows_with_required_xy_fields():
    charts = build_chart_data(chart_sample_df())
    scatter = charts["volume_close_strength"]

    assert scatter["code"].tolist() == ["AAA", "BBB", "CCC"]
    assert {
        "code",
        "signal_source",
        "ibd_candidate_rule",
        "ibd_entry_price",
        "pct_above_ceiling",
    }.issubset(scatter.columns)


def test_kpis_are_based_on_filtered_dataframe_only():
    kpis = build_kpis(chart_sample_df().iloc[:3])

    assert kpis["filtered_rows"] == 3
    assert kpis["ibd_valid_rate_pct"] == round(2 / 3 * 100, 2)
    assert kpis["median_ibd_entry_volume_ratio"] == 3.0
    assert kpis["median_ibd_entry_close_vs_trigger_pct"] == 0.04
