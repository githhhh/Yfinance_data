from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from dashboard.app import _render_selected_row_detail
from dashboard.field_config import FLOW_CARD_META, STATUS_META, get_midweek_table_columns
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
        "CARRY": "Carry Over",
        "RECONFIRMED": "Confirmed Again",
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
        "BECAME_ACTIONABLE": "含义：上周不在买区，本次进入买点上方 0%–5% 的买区。",
        "LEFT_ACTIONABLE": "含义：上周在买区，本次已经离开买区。",
        "OTHER_CHANGES": "含义：状态和上周不同，但不是进入或离开买区。",
        "NEW": "含义：完整周没有信号，周中首次出现信号。",
        "CARRY": "含义：周中没有新信号，但完整周信号继续观察，状态按当前价格更新。",
        "RECONFIRMED": "含义：完整周和周中都有信号，以周中数据为准。",
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
