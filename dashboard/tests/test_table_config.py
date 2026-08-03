from dashboard.field_config import (
    EXCLUDED_CUSTOM_FIELDS,
    FIELD_CONFIG,
    NUMBER_FIELDS,
    QUALITY_META,
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
        "pullback_duration_weeks",
        "pullback_pct",
        "pullback_pct_off_peak",
        "pullback_v_is_dry",
        "ibd_entry_status",
        "latest_close",
        "current_vs_ibd_candidate_pct",
        "ibd_breakout_quality",
        "C_continuous",
        "rank_C_continuous",
        "is_priority",
    ]
    assert get_all_table_columns() == expected
    assert get_column_view_fields("All Fields") == expected


def test_pullback_duration_weeks_is_registered_as_upstream_structure_field():
    assert "pullback_duration_weeks" in NUMBER_FIELDS
    assert FIELD_CONFIG["pullback_duration_weeks"] == {
        "label": "Pullback Duration Weeks",
        "type": "number",
        "group": "Risk / Structure",
        "filterable": True,
        "sortable": True,
        "default_table": True,
        "custom_mode": True,
        "c_rank_mode": False,
        "advanced_filter": True,
        "format": None,
        "help": "上游正式产出的回撤/巩固持续时间，用于 Continuation 信号的时长检查。",
    }


def test_ibd_decision_view_columns():
    expected = [
        "code",
        "ibd_entry_status",
        "ibd_candidate_rule",
        "current_vs_ibd_candidate_pct",
        "ibd_breakout_quality",
        "latest_close",
        "ibd_entry_vol_or_reject",
        "volume_ratio",
        "rank_C_continuous",
    ]
    assert get_default_table_columns() == expected
    assert get_column_view_fields("IBD Decision") == expected


def test_breakout_quality_derivation_logic():
    from dashboard.data_utils import _compute_breakout_quality
    import pandas as pd

    cases = [
        (0.80, 0.81, "Powerful Breakout"),
        (0.80, 0.80, "Powerful Breakout"),
        (0.80, 0.7999, "Strong Breakout"),
        (0.7999, 0.80, "Strong Breakout"),
        (0.30, 1.20, "Weak Close"),
        (0.80, 0.50, "Strong Breakout"),
        (0.80, 0.4999, "Constructive Breakout"),
        (0.65, 0.66, "Strong Breakout"),
        (0.65, 0.65, "Strong Breakout"),
        (0.65, 0.6499, "Constructive Breakout"),
        (0.65, 0.50, "Constructive Breakout"),
        (0.65, 0.20, "Marginal Breakout"),
        (0.6499, 1.20, "Weak Close"),
    ]
    for pos, range_ratio, expected in cases:
        assert (
            _compute_breakout_quality(
                pd.Series(
                    {
                        "ibd_entry_close_position": pos,
                        "ibd_entry_breakout_range_ratio": range_ratio,
                    }
                )
            )
            == expected
        )

    assert pd.isna(
        _compute_breakout_quality(
            pd.Series({"ibd_entry_close_position": None, "ibd_entry_breakout_range_ratio": 0.40})
        )
    )
    assert pd.isna(
        _compute_breakout_quality(
            pd.Series(
                {
                    "ibd_entry_close_position": 0.92,
                    "ibd_entry_breakout_range_ratio": -0.01,
                }
            )
        )
    )


def test_breakout_quality_sorting_order():
    from dashboard.data_utils import SortSpec, apply_sort
    import pandas as pd

    df = pd.DataFrame([
        {"code": "A", "ibd_breakout_quality": "Weak Close"},
        {"code": "B", "ibd_breakout_quality": "Powerful Breakout"},
        {"code": "C", "ibd_breakout_quality": "Marginal Breakout"},
        {"code": "D", "ibd_breakout_quality": "Strong Breakout"},
        {"code": "E", "ibd_breakout_quality": "Constructive Breakout"},
    ])

    sorted_asc = apply_sort(df, [SortSpec("ibd_breakout_quality", "asc")])
    assert sorted_asc["code"].tolist() == ["B", "D", "E", "C", "A"]

    sorted_desc = apply_sort(df, [SortSpec("ibd_breakout_quality", "desc")])
    assert sorted_desc["code"].tolist() == ["A", "C", "E", "D", "B"]


def test_breakout_quality_sorting_keeps_alias_labels_compatible():
    from dashboard.data_utils import SortSpec, apply_sort
    import pandas as pd

    df = pd.DataFrame([
        {"code": "A", "ibd_breakout_quality": "Constructive Close"},
        {"code": "B", "ibd_breakout_quality": "Constructive Close (High Close / Thin Thrust)"},
        {"code": "C", "ibd_breakout_quality": "Strong Close"},
        {"code": "D", "ibd_breakout_quality": "High Close, Small Breakout"},
    ])

    sorted_df = apply_sort(df, [SortSpec("ibd_breakout_quality", "asc")])

    assert sorted_df["code"].tolist() == ["C", "B", "D", "A"]


def test_breakout_quality_recomputes_existing_legacy_labels_when_inputs_exist():
    from dashboard.data_utils import normalize_pool_df
    import pandas as pd

    df = pd.DataFrame(
        {
            "code": ["AAA", "BBB"],
            "ibd_breakout_quality": ["Strong Close", "Constructive Close"],
            "ibd_entry_close_position": [0.30, 0.65],
            "ibd_entry_breakout_range_ratio": [1.20, 0.20],
        }
    )

    normalized = normalize_pool_df(df)

    assert normalized["ibd_breakout_quality"].tolist() == ["Weak Close", "Marginal Breakout"]


def test_breakout_quality_sorting_keeps_unknown_values_last():
    from dashboard.data_utils import SortSpec, apply_sort
    import pandas as pd

    df = pd.DataFrame([
        {"code": "A", "ibd_breakout_quality": "Weak Close"},
        {"code": "B", "ibd_breakout_quality": pd.NA},
        {"code": "C", "ibd_breakout_quality": "Powerful Breakout"},
        {"code": "D", "ibd_breakout_quality": "Unmapped"},
    ])

    sorted_asc = apply_sort(df, [SortSpec("ibd_breakout_quality", "asc")])
    assert sorted_asc["code"].tolist() == ["C", "A", "B", "D"]

    sorted_desc = apply_sort(df, [SortSpec("ibd_breakout_quality", "desc")])
    assert sorted_desc["code"].tolist() == ["A", "C", "B", "D"]


def test_breakout_quality_visual_meta_separates_constructive_marginal_and_weak():
    assert QUALITY_META["Powerful Breakout"]["borderWidth"] == "5px"
    assert QUALITY_META["Strong Breakout"]["borderWidth"] == "4px"
    assert QUALITY_META["Constructive Breakout"] == {
        "label": "Constructive Breakout",
        "color": "#4ade80",
        "borderColor": "rgba(34, 197, 94, 0.85)",
        "borderWidth": "3px",
        "backgroundImage": "linear-gradient(90deg, rgba(34, 197, 94, 0.10), rgba(34, 197, 94, 0.03))",
        "fontWeight": "600",
        "rule": "Mixed but valid price action",
    }
    assert QUALITY_META["Marginal Breakout"] == {
        "label": "Marginal Breakout",
        "color": "#86c99d",
        "borderColor": "rgba(134, 239, 172, 0.42)",
        "borderWidth": "2px",
        "backgroundImage": "linear-gradient(90deg, rgba(34, 197, 94, 0.025), rgba(34, 197, 94, 0.005))",
        "fontWeight": "500",
        "rule": "Valid, but close and clearance are both thin",
    }
    assert QUALITY_META["Weak Close"] == {
        "label": "Weak Close",
        "color": "#9eaaa2",
        "borderColor": "rgba(134, 239, 172, 0.20)",
        "borderWidth": "1px",
        "backgroundImage": "none",
        "fontWeight": "400",
        "rule": "Low close",
    }


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
    assert options["rowSelection"] == {
        "mode": "singleRow",
        "enableClickSelection": True,
        "checkboxes": False,
        "headerCheckbox": False,
    }
    assert "enableRangeSelection" not in options
    assert "suppressRowClickSelection" not in options


def test_breakout_quality_column_uses_custom_dom_components_when_js_is_available():
    from dashboard.table_view import HAS_JS_CODE

    options = build_grid_options(["ibd_breakout_quality"])
    quality_col = options["columnDefs"][0]
    assert quality_col["headerName"] == "Breakout Price Quality"
    assert quality_col["width"] == 260
    assert quality_col["minWidth"] == 220
    assert quality_col["maxWidth"] == 300

    if not HAS_JS_CODE:
        assert "components" not in options
        return

    assert quality_col["headerComponent"] == "breakoutQualityHeader"
    assert quality_col["cellRenderer"] == "breakoutQualityCellRenderer"
    assert "headerTooltip" not in quality_col
    assert "breakoutQualityHeader" in options["components"]
    assert "breakoutQualityCellRenderer" in options["components"]
    header_source = options["components"]["breakoutQualityHeader"].js_code
    assert "Breakout Price Quality" in header_source
    assert "Price only" in header_source
    assert "Volume is separate" in header_source
    assert "Constructive" in header_source
    assert "Marginal" in header_source
    assert "Tight" not in header_source
    assert "Defense" not in header_source
    assert "Entry Context" not in header_source
    cell_source = options["components"]["breakoutQualityCellRenderer"].js_code
    assert "breakout-cell-tooltip" in cell_source
    assert "'Close Position ' + posStr + ' · Range Ratio ' + rrStr" in cell_source
    assert "Why:" not in cell_source
    assert "Rule:" not in cell_source
    assert "High close" in cell_source
    assert "clear trigger clearance" in cell_source
    assert "Defense Standard" not in cell_source
    assert "Trigger Position" not in cell_source
    assert "backgroundImage" in quality_col["cellStyle"].js_code
    assert "borderLeft" in quality_col["cellStyle"].js_code


def test_breakout_quality_row_data_keeps_tooltip_support_fields_hidden():
    import pandas as pd
    from dashboard.table_view import _row_data_columns

    df = pd.DataFrame(
        {
            "code": ["AAA"],
            "ibd_breakout_quality": ["Strong Breakout"],
            "ibd_entry_close_vs_trigger_pct": [0.04],
            "ibd_entry_close_position": [0.82],
            "ibd_entry_breakout_range_ratio": [0.75],
        }
    )

    assert _row_data_columns(df, ["code", "ibd_breakout_quality"]) == [
        "code",
        "ibd_breakout_quality",
        "ibd_entry_close_vs_trigger_pct",
        "ibd_entry_close_position",
        "ibd_entry_breakout_range_ratio",
    ]


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


def test_render_table_preserves_hidden_breakout_quality_precision(monkeypatch):
    import pandas as pd
    import sys
    import types
    from dashboard.table_view import render_table

    captured = {}

    def mock_aggrid(df, **kwargs):
        captured["df"] = df
        return {}

    monkeypatch.setitem(sys.modules, "st_aggrid", types.SimpleNamespace(AgGrid=mock_aggrid))

    df = pd.DataFrame(
        {
            "code": ["AAA"],
            "ibd_breakout_quality": ["Strong Breakout"],
            "ibd_entry_close_vs_trigger_pct": [0.00523],
            "ibd_entry_close_position": [0.869565],
            "ibd_entry_breakout_range_ratio": [0.652174],
        }
    )

    render_table(df, ["code", "ibd_breakout_quality"])

    assert captured["df"].loc[1, "ibd_entry_close_vs_trigger_pct"] == 0.00523
    assert captured["df"].loc[1, "ibd_entry_close_position"] == 0.869565
    assert captured["df"].loc[1, "ibd_entry_breakout_range_ratio"] == 0.652174


def test_render_table_uses_a_stable_component_key_for_selection_events(monkeypatch):
    import pandas as pd
    import sys
    import types
    from dashboard.table_view import render_table

    captured = {}

    def mock_aggrid(df, **kwargs):
        captured["kwargs"] = kwargs
        return {}

    monkeypatch.setitem(sys.modules, "st_aggrid", types.SimpleNamespace(AgGrid=mock_aggrid))

    render_table(pd.DataFrame({"code": ["AAA"]}), ["code"])

    assert captured["kwargs"]["key"] == "review_results_grid"


def test_render_table_normalizes_breakout_quality_aliases_before_display(monkeypatch):
    import pandas as pd
    import sys
    import types
    from dashboard.table_view import render_table

    captured = {}

    def mock_aggrid(df, **kwargs):
        captured["df"] = df
        return {}

    monkeypatch.setitem(sys.modules, "st_aggrid", types.SimpleNamespace(AgGrid=mock_aggrid))

    df = pd.DataFrame(
        {
            "code": ["AAA"],
            "ibd_breakout_quality": ["High Close, Small Breakout"],
        }
    )

    render_table(df, ["code", "ibd_breakout_quality"])

    assert captured["df"].loc[1, "ibd_breakout_quality"] == "Constructive Breakout"
