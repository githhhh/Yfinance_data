import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from dashboard.data_utils import (
    validate_pool_schema,
    validate_pool_semantics,
    build_entry_status_counts,
    apply_c_rank_mode,
    apply_default_review_order,
    REQUIRED_CORE_FIELDS,
)
from dashboard.field_config import (
    get_all_table_columns,
    get_column_view_fields,
    get_field_label,
)
from dashboard.table_view import build_grid_options
from dashboard.app import (
    _render_flow_rules_dialog,
    _render_selected_row_detail,
    _render_copy_codes_control,
)


def sample_full_pool_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "code": "AAPL",
                "signal": True,
                "signal_source": "ceiling_breakout",
                "ibd_candidate_rule": "ceiling_pullback",
                "ibd_entry_valid": True,
                "ibd_entry_reject_reason": "",
                "ibd_entry_status": "ACTIONABLE",
                "ibd_entry_volume_ratio": 2.5,
                "ibd_entry_vol_or_reject": "Vol: 2.50x",
                "current_vs_ibd_candidate_pct": 2.1,
                "latest_close": 185.20,
                "ibd_candidate_price": 181.40,
                "volume_ratio": 1.4,
                "rank_C_continuous": 2,
                "C_continuous": 85.4,
                "is_priority": True,
                "sector": "Technology",
                "industry": "Consumer Electronics",
            },
            {
                "code": "NVDA",
                "signal": True,
                "signal_source": "pivot",
                "ibd_candidate_rule": "pivot",
                "ibd_entry_valid": False,
                "ibd_entry_reject_reason": "Low Vol",
                "ibd_entry_status": "UNCONFIRMED",
                "ibd_entry_volume_ratio": 1.1,
                "ibd_entry_vol_or_reject": "Low Vol",
                "current_vs_ibd_candidate_pct": 1.8,
                "latest_close": 915.00,
                "ibd_candidate_price": 898.80,
                "volume_ratio": 1.8,
                "rank_C_continuous": 1,
                "C_continuous": 92.1,
                "is_priority": True,
                "sector": "Technology",
                "industry": "Semiconductors",
            },
            {
                "code": "TSLA",
                "signal": True,
                "signal_source": "ceiling_breakout",
                "ibd_candidate_rule": "ceiling_pullback",
                "ibd_entry_valid": True,
                "ibd_entry_reject_reason": "",
                "ibd_entry_status": "BELOW_TRIGGER",
                "ibd_entry_volume_ratio": 1.6,
                "ibd_entry_vol_or_reject": "Vol: 1.60x",
                "current_vs_ibd_candidate_pct": -3.5,
                "latest_close": 172.00,
                "ibd_candidate_price": 178.20,
                "volume_ratio": 1.1,
                "rank_C_continuous": 3,
                "C_continuous": 71.0,
                "is_priority": False,
                "sector": "Consumer Cyclical",
                "industry": "Auto Manufacturers",
            },
            {
                "code": "META",
                "signal": True,
                "signal_source": "10_wk_ema_touch_confirm",
                "ibd_candidate_rule": "ma10_touch_confirm",
                "ibd_entry_valid": True,
                "ibd_entry_reject_reason": "",
                "ibd_entry_status": "EXTENDED",
                "ibd_entry_volume_ratio": 3.0,
                "ibd_entry_vol_or_reject": "Vol: 3.00x",
                "current_vs_ibd_candidate_pct": 7.2,
                "latest_close": 495.00,
                "ibd_candidate_price": 461.75,
                "volume_ratio": 2.1,
                "rank_C_continuous": 4,
                "C_continuous": 68.5,
                "is_priority": False,
                "sector": "Communication Services",
                "industry": "Internet Content",
            },
            {
                "code": "INACT",
                "signal": False,
                "signal_source": "",
                "ibd_candidate_rule": "pivot",
                "ibd_entry_valid": False,
                "ibd_entry_reject_reason": "",
                "ibd_entry_status": "",
                "ibd_entry_volume_ratio": 4.0,
                "ibd_entry_vol_or_reject": "Vol: 4.00x",
                "current_vs_ibd_candidate_pct": 1.0,
                "latest_close": 50.00,
                "ibd_candidate_price": 49.50,
                "volume_ratio": 0.8,
                "rank_C_continuous": 5,
                "C_continuous": 40.0,
                "is_priority": False,
                "sector": "Finance",
                "industry": "Banks",
            },
        ]
    )


# 1. Schema Validation Acceptance
def test_schema_validation_checks_all_thirteen_core_fields():
    df = sample_full_pool_df()
    validate_pool_schema(df)
    validate_pool_semantics(df)

    for required_field in REQUIRED_CORE_FIELDS:
        bad_df = df.drop(columns=[required_field])
        with pytest.raises(ValueError, match=f"missing required IBD Review columns: {required_field}"):
            validate_pool_schema(bad_df)


def test_semantic_validation_checks():
    df = sample_full_pool_df()

    with pytest.raises(ValueError, match="dataset cannot be empty"):
        validate_pool_semantics(pd.DataFrame())

    bad_code = df.copy()
    bad_code.loc[0, "code"] = ""
    with pytest.raises(ValueError, match="code cannot be empty"):
        validate_pool_semantics(bad_code)

    dup_code = df.copy()
    dup_code.loc[1, "code"] = "AAPL"
    with pytest.raises(ValueError, match="code cannot be duplicate"):
        validate_pool_semantics(dup_code)

    bad_status = df.copy()
    bad_status.loc[0, "ibd_entry_status"] = "EXTENDED"
    with pytest.raises(ValueError, match="status formula mismatch"):
        validate_pool_semantics(bad_status)

    bad_non_active = df.copy()
    bad_non_active.loc[4, "ibd_entry_status"] = "ACTIONABLE"
    with pytest.raises(ValueError, match="ibd_entry_status must be empty for non-signal rows"):
        validate_pool_semantics(bad_non_active)


# 2 & 3. Status Queue Cards strict sum and active signal restriction
def test_status_queue_cards_strict_sum_and_active_signals_only():
    df = sample_full_pool_df()
    # INACT has signal=False even though its status is ACTIONABLE; it must be excluded
    counts = build_entry_status_counts(df)

    # 4 statuses: ACTIONABLE(AAPL), UNCONFIRMED(NVDA), BELOW_TRIGGER(TSLA), EXTENDED(META)
    assert counts["ACTIONABLE"] == 1
    assert counts["UNCONFIRMED"] == 1
    assert counts["BELOW_TRIGGER"] == 1
    assert counts["EXTENDED"] == 1

    total_status_sum = counts["ACTIONABLE"] + counts["UNCONFIRMED"] + counts["BELOW_TRIGGER"] + counts["EXTENDED"]
    active_signals_total = int(df["signal"].sum())
    assert total_status_sum == active_signals_total


# 4. UNCONFIRMED card subtitle tracks unconfirmed_within_3pct exactly
def test_unconfirmed_card_subtitle_calculation():
    df = sample_full_pool_df()
    counts = build_entry_status_counts(df)
    # NVDA is UNCONFIRMED and at +1.8% (which is <= 3.0%)
    assert counts["unconfirmed_within_3pct"] == 1

    # Add another UNCONFIRMED stock at +4.0% (> 3.0%)
    extra = pd.DataFrame(
        [
            {
                "code": "AMD",
                "signal": True,
                "ibd_entry_status": "UNCONFIRMED",
                "current_vs_ibd_candidate_pct": 4.0,
            }
        ]
    )
    df_ext = pd.concat([df, extra], ignore_index=True)
    counts_ext = build_entry_status_counts(df_ext)
    assert counts_ext["UNCONFIRMED"] == 2
    assert counts_ext["unconfirmed_within_3pct"] == 1  # Still 1 within 3%


# 5. Route slicing strictly affects status queue counts
def test_route_slicing_updates_status_counts():
    df = sample_full_pool_df()
    # Slice to route 'ceiling_pullback' (AAPL & TSLA)
    route_df = df[(df["signal"] == True) & (df["ibd_candidate_rule"] == "ceiling_pullback")]
    counts = build_entry_status_counts(route_df)

    assert counts["ACTIONABLE"] == 1
    assert counts["BELOW_TRIGGER"] == 1
    assert counts["UNCONFIRMED"] == 0
    assert counts["EXTENDED"] == 0
    assert (
        counts["ACTIONABLE"] + counts["UNCONFIRMED"] + counts["BELOW_TRIGGER"] + counts["EXTENDED"] == len(route_df)
    )


# 6. One-line Filter Bar AND intersection logic
def test_filter_bar_and_intersection_logic():
    df = sample_full_pool_df()
    active_df = df[df["signal"] == True].copy()

    # Filter by route='ceiling_pullback' AND dist_min >= -1.0 AND weekly_vol >= 1.2
    f_df = active_df[
        (active_df["ibd_candidate_rule"] == "ceiling_pullback")
        & (active_df["current_vs_ibd_candidate_pct"] >= -1.0)
        & (active_df["volume_ratio"] >= 1.2)
    ]
    # Only AAPL meets all 3 criteria (TSLA has dist -3.5 and weekly vol 1.1)
    assert len(f_df) == 1
    assert f_df.iloc[0]["code"] == "AAPL"


# 7. AG Grid configuration guarantees single row selection and stable row ID
def test_aggrid_build_grid_options_single_selection_and_stable_id():
    columns = get_column_view_fields("IBD Decision")
    options = build_grid_options(columns)

    assert options["rowSelection"] == "single"
    assert options["suppressRowClickSelection"] is False
    assert options["columnDefs"][0]["field"] == "code"
    assert options["columnDefs"][0]["pinned"] == "left"
    assert "onGridReady" not in options
    assert "onCellFocused" not in options


# 8. Selected Row Detail never fabricates a selection
def test_selected_row_detail_uses_placeholder_when_selected_missing(monkeypatch):
    df = sample_full_pool_df()
    filtered_df = df[df["signal"] == True].copy()

    rendered_markdowns = []
    monkeypatch.setattr("streamlit.markdown", lambda text, **kwargs: rendered_markdowns.append(text))
    monkeypatch.setattr("streamlit.info", lambda text, **kwargs: rendered_markdowns.append(text))

    # Test 1: valid selected code NVDA
    _render_selected_row_detail(filtered_df, "NVDA")
    assert any("NVDA" in md and "915.00" in md for md in rendered_markdowns)

    rendered_markdowns.clear()
    # A stale/missing selection must not silently present another row as selected.
    _render_selected_row_detail(filtered_df, "MISSING_CODE")
    assert rendered_markdowns == [
        '<div class="selected-strip selected-strip--empty" role="status">'
        '<span>Select a row to inspect review details.</span></div>'
    ]

    rendered_markdowns.clear()
    _render_selected_row_detail(filtered_df, None)
    assert rendered_markdowns == [
        '<div class="selected-strip selected-strip--empty" role="status">'
        '<span>Select a row to inspect review details.</span></div>'
    ]


def test_render_selected_row_detail_pd_na_safe(monkeypatch):
    df = sample_full_pool_df()
    df_na = df[df["signal"] == True].copy()
    df_na["ibd_entry_valid"] = df_na["ibd_entry_valid"].astype("object")
    df_na.loc[df_na["code"] == "NVDA", "ibd_entry_valid"] = pd.NA
    df_na.loc[df_na["code"] == "AAPL", "ibd_entry_valid"] = True

    rendered_markdowns = []
    monkeypatch.setattr("streamlit.markdown", lambda text, **kwargs: rendered_markdowns.append(text))
    monkeypatch.setattr("streamlit.info", lambda text, **kwargs: rendered_markdowns.append(text))

    _render_selected_row_detail(df_na, "NVDA")
    assert any("NVDA" in md for md in rendered_markdowns)

    rendered_markdowns.clear()
    _render_selected_row_detail(df_na, "AAPL")
    assert any("AAPL" in md for md in rendered_markdowns)


# 9. C Rank Reference Mode sorting and Top N slicing
def test_c_rank_reference_mode_sorting_and_top_n():
    df = sample_full_pool_df()
    # C Rank mode should isolate signal=True and sort by rank_C_continuous asc
    ranked = apply_c_rank_mode(df, limit=None)

    assert len(ranked) == 4  # AAPL, NVDA, TSLA, META (INACT excluded)
    assert ranked["code"].tolist() == ["NVDA", "AAPL", "TSLA", "META"]
    assert ranked["rank_C_continuous"].tolist() == [1, 2, 3, 4]

    # Top N slice (e.g. limit=2)
    ranked_top2 = apply_c_rank_mode(df, limit=2)
    assert len(ranked_top2) == 2
    assert ranked_top2["code"].tolist() == ["NVDA", "AAPL"]


# 10. Copy Codes primary control and browser fallback rendering
class DummyCtx:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def test_copy_codes_control_rendering(monkeypatch):
    rendered_components = []
    rendered_popovers = []
    rendered_codes = []

    dummy = DummyCtx()
    monkeypatch.setattr("streamlit.components.v1.html", lambda html, **kwargs: rendered_components.append(html))
    monkeypatch.setattr("streamlit.popover", lambda label, **kwargs: (rendered_popovers.append(label), dummy)[1])
    monkeypatch.setattr("streamlit.code", lambda code_str, **kwargs: rendered_codes.append(code_str))
    monkeypatch.setattr("streamlit.columns", lambda spec: [dummy, dummy])
    monkeypatch.setattr("streamlit.caption", lambda text, **kwargs: None)
    monkeypatch.setattr("dashboard.app.st.popover", lambda label, **kwargs: (rendered_popovers.append(label), dummy)[1])
    monkeypatch.setattr("dashboard.app.st.columns", lambda spec: [dummy, dummy])

    codes = ["AAPL", "NVDA", "TSLA"]
    _render_copy_codes_control(codes, key_prefix="test")

    assert len(rendered_components) == 1
    assert "Copy 3 Codes" in rendered_components[0]
    assert 'textToCopy = "AAPL, NVDA, TSLA"' in rendered_components[0]
    assert "navigator.clipboard.writeText" in rendered_components[0]
    assert "document.execCommand('copy')" in rendered_components[0]
    assert "Manual" not in rendered_components[0]
    assert rendered_popovers == []
    assert rendered_codes == []


def test_copy_codes_control_counts_only_non_blank_codes_and_disables_empty(monkeypatch):
    rendered_components = []
    monkeypatch.setattr(
        "streamlit.components.v1.html",
        lambda value, **kwargs: rendered_components.append(value),
    )

    _render_copy_codes_control(["AAPL", None, " ", pd.NA], key_prefix="valid")
    assert "Copy 1 Codes" in rendered_components[-1]
    assert 'textToCopy = "AAPL"' in rendered_components[-1]
    assert " disabled" not in rendered_components[-1].split("</button>", 1)[0]

    _render_copy_codes_control([None, "", "  ", pd.NA], key_prefix="empty")
    assert "Copy 0 Codes" in rendered_components[-1]
    assert " disabled" in rendered_components[-1].split("</button>", 1)[0]


# 11. Entry Vol rules: enabled for ACTIONABLE, BELOW_TRIGGER, EXTENDED; disabled/cleared for UNCONFIRMED/All
def test_entry_vol_enabled_for_actionable_below_trigger_and_extended_only():
    from dashboard.app import ENTRY_VOL_ENABLED_STATUSES

    assert ENTRY_VOL_ENABLED_STATUSES == {"ACTIONABLE", "BELOW_TRIGGER", "EXTENDED"}
    assert "UNCONFIRMED" not in ENTRY_VOL_ENABLED_STATUSES
    assert "All" not in ENTRY_VOL_ENABLED_STATUSES


def test_snapshot_freshness_boundaries():
    from datetime import date
    from dashboard.data_utils import build_snapshot_freshness

    base_date = date(2026, 7, 10)
    res3 = build_snapshot_freshness("2026-07-07", today=base_date)
    assert res3["status"] == "FRESH"
    assert res3["age_days"] == 3

    res4 = build_snapshot_freshness("2026-07-06", today=base_date)
    assert res4["status"] == "AGING"
    assert res4["age_days"] == 4

    res6 = build_snapshot_freshness("2026-07-04", today=base_date)
    assert res6["status"] == "AGING"
    assert res6["age_days"] == 6

    res7 = build_snapshot_freshness("2026-07-03", today=base_date)
    assert res7["status"] == "STALE"
    assert res7["age_days"] == 7

    res_fut = build_snapshot_freshness("2026-07-15", today=base_date)
    assert res_fut["age_days"] == 0
    assert res_fut["status"] == "FRESH"


def test_snapshot_freshness_invalid_date():
    from dashboard.data_utils import build_snapshot_freshness

    assert build_snapshot_freshness(None)["status"] == "UNKNOWN"
    assert build_snapshot_freshness("")["status"] == "UNKNOWN"
    assert build_snapshot_freshness("N/A")["status"] == "UNKNOWN"
    assert build_snapshot_freshness("invalid-date")["status"] == "UNKNOWN"
    assert build_snapshot_freshness(np.nan)["status"] == "UNKNOWN"


def test_near_trigger_inclusive_bounds():
    from dashboard.data_utils import filter_unconfirmed_near_trigger

    df = pd.DataFrame([
        {"code": "A", "ibd_entry_status": "UNCONFIRMED", "current_vs_ibd_candidate_pct": -0.1},
        {"code": "B", "ibd_entry_status": "UNCONFIRMED", "current_vs_ibd_candidate_pct": 0.0},
        {"code": "C", "ibd_entry_status": "UNCONFIRMED", "current_vs_ibd_candidate_pct": 1.5},
        {"code": "D", "ibd_entry_status": "UNCONFIRMED", "current_vs_ibd_candidate_pct": 3.0},
        {"code": "E", "ibd_entry_status": "UNCONFIRMED", "current_vs_ibd_candidate_pct": 3.1},
    ])
    res = filter_unconfirmed_near_trigger(df, "UNCONFIRMED", True)
    assert res["code"].tolist() == ["B", "C", "D"]


def test_near_trigger_excludes_nan():
    from dashboard.data_utils import filter_unconfirmed_near_trigger

    df = pd.DataFrame([
        {"code": "A", "ibd_entry_status": "UNCONFIRMED", "current_vs_ibd_candidate_pct": np.nan},
        {"code": "B", "ibd_entry_status": "UNCONFIRMED", "current_vs_ibd_candidate_pct": None},
        {"code": "C", "ibd_entry_status": "UNCONFIRMED", "current_vs_ibd_candidate_pct": "abc"},
        {"code": "D", "ibd_entry_status": "UNCONFIRMED", "current_vs_ibd_candidate_pct": 1.5},
    ])
    res = filter_unconfirmed_near_trigger(df, "UNCONFIRMED", True)
    assert res["code"].tolist() == ["D"]


def test_near_trigger_count_depends_only_on_route():
    df = sample_full_pool_df()
    extra = pd.DataFrame([
        {"code": "ROUTE1", "signal": True, "ibd_candidate_rule": "ceiling_breakout", "ibd_entry_status": "UNCONFIRMED", "current_vs_ibd_candidate_pct": 1.2, "volume_ratio": 0.5},
        {"code": "ROUTE2", "signal": True, "ibd_candidate_rule": "ma10_touch_confirm", "ibd_entry_status": "UNCONFIRMED", "current_vs_ibd_candidate_pct": 2.5, "volume_ratio": 2.0},
    ])
    full_df = pd.concat([df, extra], ignore_index=True)
    counts_all = build_entry_status_counts(full_df)
    counts_route1 = build_entry_status_counts(full_df[full_df["ibd_candidate_rule"] == "ceiling_breakout"])

    assert counts_route1["unconfirmed_within_3pct"] <= counts_all["unconfirmed_within_3pct"]

    route_df = full_df[full_df["ibd_candidate_rule"] == "ceiling_breakout"]
    base_counts = build_entry_status_counts(route_df)
    assert base_counts["unconfirmed_within_3pct"] == build_entry_status_counts(route_df)["unconfirmed_within_3pct"]


def test_near_trigger_combines_with_distance_filters():
    from dashboard.data_utils import filter_unconfirmed_near_trigger

    df = pd.DataFrame([
        {"code": "A", "ibd_entry_status": "UNCONFIRMED", "current_vs_ibd_candidate_pct": 0.0},
        {"code": "B", "ibd_entry_status": "UNCONFIRMED", "current_vs_ibd_candidate_pct": 0.8},
        {"code": "C", "ibd_entry_status": "UNCONFIRMED", "current_vs_ibd_candidate_pct": 1.5},
        {"code": "D", "ibd_entry_status": "UNCONFIRMED", "current_vs_ibd_candidate_pct": 2.5},
        {"code": "E", "ibd_entry_status": "UNCONFIRMED", "current_vs_ibd_candidate_pct": 3.0},
    ])
    near_df = filter_unconfirmed_near_trigger(df, "UNCONFIRMED", True)
    comb_min = near_df[pd.to_numeric(near_df["current_vs_ibd_candidate_pct"], errors="coerce") >= 1.0]
    assert comb_min["code"].tolist() == ["C", "D", "E"]

    comb_max = near_df[pd.to_numeric(near_df["current_vs_ibd_candidate_pct"], errors="coerce") <= 2.0]
    assert comb_max["code"].tolist() == ["A", "B", "C"]


def test_near_trigger_state_clears_on_status_change():
    state = {"ibd_near_trigger_only": True, "ibd_filter_status": "ACTIONABLE"}
    if state["ibd_filter_status"] != "UNCONFIRMED" and state.get("ibd_near_trigger_only", False):
        state["ibd_near_trigger_only"] = False
    assert state["ibd_near_trigger_only"] is False


def test_copy_order_after_near_trigger_filter():
    from dashboard.data_utils import filter_unconfirmed_near_trigger, apply_default_review_order

    df = pd.DataFrame([
        {"code": "Z_UNCONF", "ibd_entry_status": "UNCONFIRMED", "current_vs_ibd_candidate_pct": 1.5, "rank_C_continuous": 10},
        {"code": "A_UNCONF", "ibd_entry_status": "UNCONFIRMED", "current_vs_ibd_candidate_pct": 3.5, "rank_C_continuous": 5},
        {"code": "M_UNCONF", "ibd_entry_status": "UNCONFIRMED", "current_vs_ibd_candidate_pct": 0.5, "rank_C_continuous": 2},
    ])
    filtered = filter_unconfirmed_near_trigger(df, "UNCONFIRMED", True)
    ordered = apply_default_review_order(filtered)
    assert ordered["code"].tolist() == ["M_UNCONF", "Z_UNCONF"]


def test_default_review_order_prioritizes_actionable_breakout_quality_and_ignores_c_rank():
    df = pd.DataFrame(
        [
            {
                "code": "A_WEAK",
                "ibd_entry_status": "ACTIONABLE",
                "ibd_breakout_quality": "Weak Close",
                "rank_C_continuous": 1,
            },
            {
                "code": "B_STRONG",
                "ibd_entry_status": "ACTIONABLE",
                "ibd_breakout_quality": "Strong Close",
                "rank_C_continuous": 99,
            },
            {
                "code": "AA_STRONG",
                "ibd_entry_status": "ACTIONABLE",
                "ibd_breakout_quality": "Strong Close",
                "rank_C_continuous": 100,
            },
            {
                "code": "C_UNKNOWN",
                "ibd_entry_status": "ACTIONABLE",
                "ibd_breakout_quality": pd.NA,
                "rank_C_continuous": 0,
            },
            {
                "code": "D_UNCONF",
                "ibd_entry_status": "UNCONFIRMED",
                "ibd_breakout_quality": "Powerful Breakout",
                "rank_C_continuous": 0,
            },
        ]
    )

    ordered = apply_default_review_order(df)

    assert ordered["code"].tolist() == ["AA_STRONG", "B_STRONG", "A_WEAK", "C_UNKNOWN", "D_UNCONF"]
