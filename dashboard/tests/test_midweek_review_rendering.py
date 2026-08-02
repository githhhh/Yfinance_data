from __future__ import annotations

from pathlib import Path

import pandas as pd

from dashboard.app import _render_selected_row_detail
from dashboard.field_config import FLOW_CARD_META, STATUS_META, get_midweek_table_columns
from dashboard.table_view import _code_renderer_jscode, _column_def, build_grid_options


APP_SOURCE = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
STYLE_SOURCE = (Path(__file__).resolve().parents[1] / "review_styles.py").read_text(encoding="utf-8")


def test_ten_flow_cards_share_structured_definition_count_and_click_tooltips():
    expected_flow = {
        "BECAME_ACTIONABLE",
        "LEFT_ACTIONABLE",
        "OTHER_CHANGES",
        "NEW",
        "CARRY",
        "RECONFIRMED",
    }
    assert set(FLOW_CARD_META) == expected_flow
    assert set(STATUS_META) == {"ACTIONABLE", "UNCONFIRMED", "BELOW_TRIGGER", "EXTENDED"}

    for metadata in [*FLOW_CARD_META.values(), *STATUS_META.values()]:
        assert metadata["definition"]
        assert metadata["count_basis"]
        assert metadata["click_effect"]
        assert metadata["tooltip"] == "\n".join(
            [metadata["definition"], metadata["count_basis"], metadata["click_effect"]]
        )


def test_status_metadata_uses_css_tones_instead_of_emoji_dots():
    assert {key: value["tone"] for key, value in STATUS_META.items()} == {
        "ACTIONABLE": "green",
        "UNCONFIRMED": "yellow",
        "BELOW_TRIGGER": "red",
        "EXTENDED": "blue",
    }
    for metadata in STATUS_META.values():
        assert "dot" not in metadata


def test_status_tooltips_remain_truthful_in_midweek_and_weekend_modes():
    for metadata in STATUS_META.values():
        assert "Effective Status" not in metadata["tooltip"]
        assert "Change, Origin" not in metadata["tooltip"]
        assert "applicable filters" in metadata["tooltip"]


def test_midweek_table_adds_one_change_column_after_code_only():
    columns = get_midweek_table_columns()

    assert columns[:2] == ["code", "review_change_label"]
    assert "review_signal_origin" not in columns
    assert _column_def("review_change_label")["headerName"] == "Change"
    assert build_grid_options(columns)["columnDefs"][1]["field"] == "review_change_label"


def test_code_renderer_reserves_origin_badge_slot_without_native_title():
    renderer = _code_renderer_jscode()
    source = renderer.js_code if renderer is not None else Path(
        __file__
    ).resolve().parents[1].joinpath("table_view.py").read_text(encoding="utf-8")

    assert "review_signal_origin" in source
    assert "origin-slot" in source
    assert ".title =" not in source


def test_selected_row_midweek_markup_keeps_five_cells_and_shows_transition(monkeypatch):
    row = {
        "code": "TEST",
        "ibd_candidate_price": 100.0,
        "ibd_candidate_rule": "pivot",
        "current_vs_ibd_candidate_pct": 2.0,
        "latest_close": 102.0,
        "ibd_entry_status": "ACTIONABLE",
        "ibd_entry_vol_or_reject": "2.00x",
        "rank_C_continuous": 1,
        "C_continuous": 2.0,
        "ibd_entry_valid": True,
        "review_signal_origin": "NEW",
        "review_change_label": "NEW → ACTIONABLE",
        "review_baseline_entry_status": "UNCONFIRMED",
        "review_effective_entry_status": "ACTIONABLE",
    }
    rendered: list[str] = []
    monkeypatch.setattr("streamlit.markdown", lambda value, **kwargs: rendered.append(value))
    monkeypatch.setattr("streamlit.info", lambda value, **kwargs: rendered.append(value))

    _render_selected_row_detail(pd.DataFrame([row]), "TEST")
    markup = rendered[-1]

    assert 'data-origin="NEW"' in markup
    assert "NEW → ACTIONABLE" in markup
    assert "UNCONFIRMED →" in markup
    assert 'class="selected-strip"' in markup
    assert markup.count('class="selected-summary-cell') == 5


def test_selected_row_empty_state_keeps_the_same_fixed_surface(monkeypatch):
    rendered: list[str] = []
    monkeypatch.setattr("streamlit.markdown", lambda value, **kwargs: rendered.append(value))
    monkeypatch.setattr(
        "streamlit.info",
        lambda value, **kwargs: rendered.append(f"unexpected-info:{value}"),
    )

    _render_selected_row_detail(pd.DataFrame(), None)

    assert rendered == [
        '<div class="selected-strip selected-strip--empty" role="status">'
        '<span>No matching records found with current filter criteria.</span></div>'
    ]


def test_app_declares_stable_mode_scope_context_filter_and_layout_slots():
    for key in [
        "review_mode_controls",
        "review_scope_controls",
        "review_context_slot",
        "filters_header",
    ]:
        assert f'key="{key}"' in APP_SOURCE
    assert 'key="btn_scope_changes"' in APP_SOURCE
    assert "disabled=not has_comparison" in APP_SOURCE
    assert 'state["filters_expanded"]' in APP_SOURCE
    assert "Weekend Baseline" in APP_SOURCE
    assert "Review Priority" in APP_SOURCE
    assert "C Rank" in APP_SOURCE


def test_cards_use_streamlit_help_and_separate_touch_info_popover():
    assert "def _render_filter_card(" in APP_SOURCE
    assert "help=metadata[\"tooltip\"]" in APP_SOURCE
    assert 'st.popover("ⓘ"' in APP_SOURCE
    assert 'key=f"flow_tooltip_content_{card_id.lower()}"' in APP_SOURCE
    assert "aria-describedby" not in APP_SOURCE  # Generated by Streamlit's help trigger.
    assert "prefers-reduced-motion" in STYLE_SOURCE
