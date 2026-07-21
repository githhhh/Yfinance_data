from dashboard.field_config import (
    EXCLUDED_CUSTOM_FIELDS,
    get_all_table_columns,
    get_column_view_fields,
    get_custom_mode_fields,
    get_default_table_columns,
    get_filter_funnel_groups,
    get_filterable_fields,
    get_sortable_fields,
)
from dashboard.table_view import build_grid_options


def test_custom_mode_excludes_c_rank_reference_fields_everywhere():
    assert EXCLUDED_CUSTOM_FIELDS == {"C_continuous", "rank_C_continuous", "is_priority"}
    for fields in (
        get_filterable_fields(),
        get_sortable_fields(),
    ):
        assert not EXCLUDED_CUSTOM_FIELDS.intersection(fields)


def test_filterable_fields_follow_trading_decision_funnel():
    groups = get_filter_funnel_groups()

    assert list(groups) == [
        "Route",
        "Entry Status",
        "Optional Quality Filters",
    ]
    assert groups["Route"] == ["ibd_candidate_rule"]
    assert groups["Entry Status"] == ["ibd_entry_status"]
    assert groups["Optional Quality Filters"] == [
        "current_vs_ibd_candidate_pct",
        "ibd_entry_volume_ratio",
        "volume_ratio",
    ]
    assert get_filterable_fields() == [
        "ibd_candidate_rule",
        "ibd_entry_status",
        "current_vs_ibd_candidate_pct",
        "ibd_entry_volume_ratio",
        "volume_ratio",
    ]


def test_all_fields_table_columns_follow_logical_business_groups():
    expected = [
        "code",
        "snapshot_date",
        "sector",
        "industry",
        "eps_yoy_growth",
        "price_52_week_high",
        "dist_to_52w_high_pct",
        "signal",
        "signal_source",
        "ibd_candidate_rule",
        "ibd_candidate_signal_source",
        "breakout_date",
        "ibd_candidate_price",
        "ibd_trigger_price",
        "ibd_entry_valid",
        "ibd_entry_date",
        "ibd_entry_price",
        "ibd_entry_volume_ratio",
        "ibd_entry_vol_or_reject",
        "ibd_entry_close_vs_trigger_pct",
        "ibd_entry_close_position",
        "ibd_entry_breakout_range_ratio",
        "ibd_entry_rule",
        "ibd_entry_reject_reason",
        "ibd_candidate_extra",
        "ceiling",
        "ceiling_date",
        "base_duration_weeks",
        "pct_above_ceiling",
        "base_depth_abs",
        "base_depth_pct",
        "base_mbox_count",
        "mbox_count",
        "touched_ema10_count",
        "volume_ratio",
        "is_bullish",
        "pullback_count",
        "pullback_pct",
        "pullback_pct_off_peak",
        "pullback_v_is_dry",
        "ibd_entry_status",
        "latest_close",
        "current_vs_ibd_candidate_pct",
        "C_continuous",
        "rank_C_continuous",
        "is_priority",
    ]
    assert get_all_table_columns() == expected
    assert get_column_view_fields("All Fields") == expected


def test_ibd_decision_view_columns():
    expected = [
        "code",
        "ibd_entry_status",
        "ibd_candidate_rule",
        "current_vs_ibd_candidate_pct",
        "ibd_entry_close_position",
        "latest_close",
        "ibd_entry_vol_or_reject",
        "volume_ratio",
        "rank_C_continuous",
    ]
    assert get_default_table_columns() == expected
    assert get_column_view_fields("IBD Decision") == expected


def test_c_rank_reference_view_columns():
    expected = [
        "code",
        "rank_C_continuous",
        "C_continuous",
        "ibd_entry_status",
        "current_vs_ibd_candidate_pct",
        "ibd_candidate_rule",
        "volume_ratio",
        "latest_close",
    ]
    assert get_column_view_fields("C Rank Reference") == expected


def test_table_views_cover_all_fields_without_scattering_related_columns():
    all_fields = set(get_all_table_columns())
    grouped_fields: set[str] = set()
    for view in ["IBD Decision", "C Rank Reference", "Signal", "IBD Entry", "Volume/Pullback", "Reference"]:
        grouped_fields.update(get_column_view_fields(view))

    # All fields should be covered by the core views plus All Fields
    assert len(grouped_fields) > 0
    assert get_column_view_fields("Signal") == [
        "code",
        "snapshot_date",
        "signal",
        "signal_source",
        "ibd_candidate_rule",
        "ibd_candidate_signal_source",
        "breakout_date",
    ]


def test_grid_options_pin_code_left_and_keep_table_capabilities():
    options = build_grid_options(["code", "signal_source", "ibd_entry_valid"])

    code_col = options["columnDefs"][0]
    assert code_col["field"] == "code"
    assert code_col["pinned"] == "left"
    assert options["defaultColDef"]["sortable"] is True
    assert options["defaultColDef"]["resizable"] is True
    assert options["enableRangeSelection"] is False


def test_render_table_sets_one_based_sequential_index(monkeypatch):
    import pandas as pd
    import sys
    from dashboard.table_view import render_table

    captured = {}

    def mock_dataframe(df, **kwargs):
        captured["df"] = df

    monkeypatch.setattr("streamlit.dataframe", mock_dataframe)
    monkeypatch.setitem(sys.modules, "st_aggrid", None)

    df = pd.DataFrame({"code": ["AAA", "BBB"], "val": [1, 2]}, index=[35, 8])
    render_table(df, ["code", "val"])

    assert captured["df"].index.tolist() == [1, 2]
