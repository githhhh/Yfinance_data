from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from dashboard.app import (
    _render_ibd_selected_row_detail,
    _render_selected_row_detail,
    _review_grid_key,
)
from dashboard.field_config import FIELD_CONFIG, FLOW_CARD_META, STATUS_META, get_midweek_table_columns
from dashboard.table_view import _code_renderer_jscode, _column_def, build_grid_options


DASHBOARD_DIR = Path(__file__).resolve().parents[1]
APP_SOURCE = (DASHBOARD_DIR / "app.py").read_text(encoding="utf-8")
STYLE_SOURCE = (DASHBOARD_DIR / "review_styles.py").read_text(encoding="utf-8")


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


def test_flow_cards_use_english_visible_copy_and_exact_chinese_tooltip_body():
    assert {key: value["label"] for key, value in FLOW_CARD_META.items()} == {
        "BECAME_ACTIONABLE": "Entered Buy Zone",
        "LEFT_ACTIONABLE": "Left Buy Zone",
        "OTHER_CHANGES": "Other Changes",
        "NEW": "New Signal",
        "CARRY": "Carried Over",
        "RECONFIRMED": "Reconfirmed",
    }
    assert {key: value["subtitle"] for key, value in STATUS_META.items()} == {
        "ACTIONABLE": "In 0%–5% Buy Zone",
        "UNCONFIRMED": "Waiting for Confirmation",
        "BELOW_TRIGGER": "Below Buy Point",
        "EXTENDED": "Over 5% — Don't Chase",
    }
    for metadata in [*FLOW_CARD_META.values(), *STATUS_META.values()]:
        assert not re.search(r"[\u4e00-\u9fff]", metadata["label"])
        assert not re.search(r"[\u4e00-\u9fff]", metadata.get("subtitle", ""))

    expected_definitions = {
        "BECAME_ACTIONABLE": "含义：本次进入买点上方 0%–5% 区间。",
        "LEFT_ACTIONABLE": "含义：上次在买区，本次已离开买区。",
        "OTHER_CHANGES": "含义：状态发生变化，但不属于进入或离开买区。",
        "NEW": "含义：本次首次出现。",
        "CARRY": "含义：周末已有，本次继续保留。",
        "RECONFIRMED": "含义：原有信号本次再次确认。",
        "ACTIONABLE": "含义：已完成入场确认，当前价位于买点上方 0%–5%。",
        "UNCONFIRMED": "含义：尚未满足日线入场确认条件。",
        "BELOW_TRIGGER": "含义：当前价低于有效买点。",
        "EXTENDED": "含义：当前价已超过买点 5%，不宜追高。",
    }
    for key, metadata in {**FLOW_CARD_META, **STATUS_META}.items():
        assert metadata["definition"] == expected_definitions[key]
        assert metadata["count_basis"] == "数量：当前范围内符合条件的标的数。"
        assert metadata["click_effect"] == "点击：只看这类标的，并保留其他已选条件。"


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
        assert "保留其他已选条件" in metadata["tooltip"]


def test_midweek_table_adds_one_change_column_after_code_only():
    columns = get_midweek_table_columns()

    assert columns[:2] == ["code", "review_change_label"]
    assert "review_signal_origin" not in columns
    assert _column_def("review_change_label")["headerName"] == "Change"
    assert build_grid_options(columns, show_origin_badge=True)["columnDefs"][1]["field"] == "review_change_label"


def test_code_renderer_only_reserves_origin_badge_slot_when_enabled():
    origin_renderer = _code_renderer_jscode(show_origin_badge=True)
    plain_renderer = _code_renderer_jscode(show_origin_badge=False)
    origin_source = origin_renderer.js_code if origin_renderer is not None else Path(
        __file__
    ).resolve().parents[1].joinpath("table_view.py").read_text(encoding="utf-8")
    plain_source = plain_renderer.js_code if plain_renderer is not None else ""

    assert "review_signal_origin" in origin_source
    assert "origin-slot" in origin_source
    assert "review_signal_origin" not in plain_source
    assert "origin-slot" not in plain_source
    assert ".title =" not in origin_source


def test_code_renderer_uses_native_copy_button_with_accessible_code_label():
    renderer = _code_renderer_jscode(show_origin_badge=True)
    source = renderer.js_code if renderer is not None else Path(
        __file__
    ).resolve().parents[1].joinpath("table_view.py").read_text(encoding="utf-8")

    assert "document.createElement('button')" in source
    assert "copy.setAttribute('aria-label', '复制 ' + String(codeText))" in source


def test_code_renderer_only_copy_button_click_stops_row_selection_bubbling():
    renderer = _code_renderer_jscode(show_origin_badge=True)
    source = renderer.js_code if renderer is not None else Path(
        __file__
    ).resolve().parents[1].joinpath("table_view.py").read_text(encoding="utf-8")

    assert "this.eGui.addEventListener('click'" not in source
    assert "copy.addEventListener('click'" in source
    assert source.count("stopPropagation()") == 1
    copy_handler = source[source.index("copy.addEventListener('click'") :]
    assert copy_handler.index("stopPropagation()") < copy_handler.index("navigator.clipboard")


def test_app_uses_distinct_grid_identities_and_only_enables_origin_for_valid_midweek():
    assert 'return f"review_results_grid_{view}_{result_digest}"' in APP_SOURCE
    assert 'grid_key="c_rank_reference_grid"' in APP_SOURCE
    assert "grid_key=grid_key" in APP_SOURCE
    assert "show_origin_badge=has_comparison" in APP_SOURCE
    assert "show_origin_badge=False" in APP_SOURCE
    rows = pd.DataFrame(
        {
            "code": ["AAA", "BBB"],
            "ibd_entry_status": ["ACTIONABLE", "UNCONFIRMED"],
        }
    )
    same_rows = rows.copy()
    filtered_rows = rows.iloc[[0]].copy()
    reversed_rows = rows.iloc[::-1].copy()

    weekend_key = _review_grid_key("WEEKEND", rows)
    assert weekend_key.startswith("review_results_grid_weekend_")
    assert _review_grid_key("WEEKEND", same_rows) == weekend_key
    assert _review_grid_key("MIDWEEK", rows).startswith("review_results_grid_midweek_")
    assert _review_grid_key("WEEKEND", filtered_rows) != weekend_key
    assert _review_grid_key("WEEKEND", reversed_rows) != weekend_key
    assert '_review_grid_key(state["mode"], filtered_df)' in APP_SOURCE


def test_code_header_help_matches_row_selection_and_copy_button_behavior():
    help_text = FIELD_CONFIG["code"]["help"]

    assert "Origin 标签或空白处选择该行" in help_text
    assert "仅点击右侧复制按钮复制代码" in help_text
    assert "点击 Code 复制" not in help_text


def test_selected_row_midweek_markup_is_a_compact_five_cell_summary_with_code_only_details(monkeypatch):
    row = {
        "code": "TEST",
        "ibd_candidate_price": 100.0,
        "ibd_candidate_rule": "pivot",
        "current_vs_ibd_candidate_pct": 2.0,
        "latest_close": 102.0,
        "ibd_entry_status": "ACTIONABLE",
        "ibd_entry_vol_or_reject": "2.00x",
        "rank_C_continuous": 1.0,
        "C_continuous": 2.0,
        "ibd_entry_valid": True,
        "review_signal_origin": "NEW",
        "review_change_label": "NEW → ACTIONABLE",
        "review_baseline_entry_status": "UNCONFIRMED",
        "review_effective_entry_status": "ACTIONABLE",
        "pullback_pct": -6.0,
        "pullback_pct_off_peak": -2.0,
    }
    rendered: list[str] = []
    monkeypatch.setattr("streamlit.markdown", lambda value, **kwargs: rendered.append(value))
    monkeypatch.setattr("streamlit.info", lambda value, **kwargs: rendered.append(value))

    _render_ibd_selected_row_detail(pd.DataFrame([row]), "TEST")
    markup = rendered[-1]

    assert "UNCONFIRMED →" in markup
    assert 'class="ibd-selected-strip"' in markup
    assert markup.count('class="selected-summary-cell') == 5
    assert '<details class="code-detail" data-selected-code="TEST">' in markup
    assert '<summary class="code-hover-trigger" aria-label="TEST 股票详情">TEST</summary>' in markup
    assert "1 of 1" not in markup
    assert "RECONF." not in markup
    assert "NEW → ACTIONABLE" not in markup
    assert "▾" not in markup
    assert 'class="selected-secondary"' in markup
    assert "(2.00×)" in markup
    assert "#1 " in markup
    assert "#1.0" not in markup
    assert 'role="region" aria-label="TEST 股票详情"' in markup
    assert "日线入场" in markup
    assert "回撤" in markup
    assert "基本面 / 形态" in markup


def test_selected_row_empty_state_keeps_the_same_fixed_surface(monkeypatch):
    rendered: list[str] = []
    monkeypatch.setattr("streamlit.markdown", lambda value, **kwargs: rendered.append(value))
    monkeypatch.setattr(
        "streamlit.info",
        lambda value, **kwargs: rendered.append(f"unexpected-info:{value}"),
    )

    _render_ibd_selected_row_detail(pd.DataFrame(), None)

    assert rendered == [
        '<div class="ibd-selected-strip ibd-selected-strip--empty" role="status">'
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
    assert 'class="weekend-scope-static"' in APP_SOURCE
    assert 'state["filters_expanded"]' in APP_SOURCE
    assert "Weekend Baseline" in APP_SOURCE
    assert "default_sort_mode" in APP_SOURCE
    assert "fixed_sort" in APP_SOURCE


def test_cards_use_one_shared_tooltip_system_with_a_separate_info_button():
    assert "def _render_filter_card(" in APP_SOURCE
    assert "help=metadata[\"tooltip\"]" not in APP_SOURCE
    assert 'class="flow-info-trigger"' in APP_SOURCE
    assert 'data-flow-tooltip="{tooltip_text}"' in APP_SOURCE
    assert 'data-flow-tooltip-title="{tooltip_title}"' in APP_SOURCE
    assert 'aria-label="{tooltip_label}"' in APP_SOURCE
    assert 'key=f"btn_flow_info_{card_id.lower()}"' not in APP_SOURCE
    assert ">ⓘ</button>" not in APP_SOURCE
    assert ">i</button>" in APP_SOURCE
    assert "flow_tooltip_content_" not in APP_SOURCE
    assert "FLOW_TOOLTIP_BRIDGE_HTML" in APP_SOURCE
    assert "prefers-reduced-motion" in STYLE_SOURCE


def test_flow_tooltip_controller_covers_hover_focus_touch_and_dismissal_contract():
    tooltip_path = DASHBOARD_DIR / "review_tooltip.py"
    assert tooltip_path.exists()
    source = tooltip_path.read_text(encoding="utf-8")

    for event_name in ["pointerover", "pointerout", "focusin", "focusout", "click", "keydown", "scroll"]:
        assert event_name in source
    for contract in [
        "aria-describedby",
        "event.preventDefault()",
        "event.stopPropagation()",
        'event.key === "Escape"',
        "titleElement.textContent = title",
        "body.textContent = content",
        "window.parent.document",
    ]:
        assert contract in source
