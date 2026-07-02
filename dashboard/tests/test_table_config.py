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
        "Entry Confirmation & Strength",
        "Weekly Volume & Price",
        "Structure",
        "Grouping",
    ]
    assert groups["Route"] == ["ibd_candidate_rule"]
    assert groups["Entry Confirmation & Strength"] == [
        "ibd_entry_valid",
        "ibd_entry_volume_ratio",
        "ibd_entry_close_vs_trigger_pct",
    ]
    assert groups["Weekly Volume & Price"] == ["volume_ratio", "is_bullish"]
    assert groups["Structure"] == ["touched_ema10_count", "pullback_pct"]
    assert groups["Grouping"] == ["sector", "industry"]
    assert get_filterable_fields() == [
        "ibd_candidate_rule",
        "ibd_entry_valid",
        "ibd_entry_volume_ratio",
        "ibd_entry_close_vs_trigger_pct",
        "volume_ratio",
        "is_bullish",
        "touched_ema10_count",
        "pullback_pct",
        "sector",
        "industry",
    ]


def test_all_fields_table_columns_follow_logical_business_groups():
    expected = [
        "code",
        "snapshot_date",
        "sector",
        "industry",
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
        "ibd_entry_close_vs_trigger_pct",
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
        "hold_return",
        "C_continuous",
        "rank_C_continuous",
        "is_priority",
    ]
    assert get_all_table_columns() == expected
    assert get_default_table_columns() == expected
    assert get_column_view_fields("All Fields") == expected


def test_table_views_cover_all_fields_without_scattering_related_columns():
    all_fields = set(get_all_table_columns())
    grouped_fields: set[str] = set()
    for view in ["Signal", "IBD Entry", "Structure", "Volume/Pullback", "Grouping", "Reference"]:
        grouped_fields.update(get_column_view_fields(view))

    assert all_fields.issubset(grouped_fields)
    assert get_column_view_fields("Structure") == [
        "code",
        "ceiling",
        "ceiling_date",
        "base_duration_weeks",
        "pct_above_ceiling",
        "base_depth_abs",
        "base_depth_pct",
        "base_mbox_count",
        "mbox_count",
        "touched_ema10_count",
        "pullback_count",
        "pullback_pct",
        "pullback_pct_off_peak",
    ]


def test_grid_options_pin_code_left_and_keep_table_capabilities():
    options = build_grid_options(["code", "signal_source", "ibd_entry_valid"])

    code_col = options["columnDefs"][0]
    assert code_col["field"] == "code"
    assert code_col["pinned"] == "left"
    assert options["defaultColDef"]["sortable"] is True
    assert options["defaultColDef"]["resizable"] is True
    assert options["enableRangeSelection"] is True
