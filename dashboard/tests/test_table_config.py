from dashboard.field_config import (
    EXCLUDED_CUSTOM_FIELDS,
    get_column_view_fields,
    get_custom_mode_fields,
    get_default_table_columns,
    get_filterable_fields,
    get_sortable_fields,
)
from dashboard.table_view import build_grid_options


def test_custom_mode_excludes_c_rank_reference_fields_everywhere():
    assert EXCLUDED_CUSTOM_FIELDS == {"C_continuous", "rank_C_continuous", "is_priority"}
    for fields in (
        get_custom_mode_fields(),
        get_filterable_fields(),
        get_sortable_fields(),
        get_default_table_columns(),
        get_column_view_fields("Full Custom"),
    ):
        assert not EXCLUDED_CUSTOM_FIELDS.intersection(fields)


def test_default_table_columns_follow_business_chain():
    assert get_default_table_columns() == [
        "code",
        "sector",
        "industry",
        "signal_source",
        "ibd_candidate_rule",
        "ibd_candidate_price",
        "ibd_entry_valid",
        "ibd_entry_date",
        "ibd_entry_price",
        "ibd_entry_volume_ratio",
        "ibd_entry_close_vs_trigger_pct",
        "pct_above_ceiling",
        "touched_ema10_count",
        "volume_ratio",
        "pullback_v_is_dry",
        "pullback_count",
        "pullback_pct_off_peak",
        "hold_return",
        "breakout_date",
        "ceiling",
    ]


def test_grid_options_pin_code_left_and_keep_table_capabilities():
    options = build_grid_options(["code", "signal_source", "ibd_entry_valid"])

    code_col = options["columnDefs"][0]
    assert code_col["field"] == "code"
    assert code_col["pinned"] == "left"
    assert options["defaultColDef"]["sortable"] is True
    assert options["defaultColDef"]["resizable"] is True
    assert options["enableRangeSelection"] is True
