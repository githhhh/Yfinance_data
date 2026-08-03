import pytest
import pandas as pd
import numpy as np
import dashboard.app as dashboard_app

from dashboard.field_config import STATUS_META, FIELD_CONFIG
from dashboard.table_view import _column_def
from dashboard.app import _format_card_val, _render_selected_row_detail


def test_review_position_follows_current_dataframe_order():
    df = pd.DataFrame({"code": ["ACU", "NVDA", "TSLA"]})

    assert dashboard_app.build_review_position(df, "NVDA") == {
        "code": "NVDA",
        "position": 2,
        "total": 3,
        "label": "NVDA · 2 of 3",
    }
    assert dashboard_app.build_review_position(df.iloc[[2, 0]], "ACU")["position"] == 2
    assert dashboard_app.build_review_position(df, "MISSING") == {
        "code": "",
        "position": None,
        "total": 3,
        "label": "",
    }


def test_visits_are_isolated_by_view_and_deduplicated():
    store = dashboard_app._record_review_visit({}, "MIDWEEK", "ACU")
    store = dashboard_app._record_review_visit(store, "MIDWEEK", "ACU")
    store = dashboard_app._record_review_visit(store, "WEEKEND", "NVDA")

    assert store == {"MIDWEEK": {"ACU"}, "WEEKEND": {"NVDA"}}


def _selected_detail_markup(monkeypatch, **overrides):
    row = {
        "code": "TEST",
        "ibd_candidate_price": 100.0,
        "ibd_candidate_rule": "pivot",
        "current_vs_ibd_candidate_pct": 1.5,
        "latest_close": 101.5,
        "ibd_entry_status": "ACTIONABLE",
        "ibd_entry_vol_or_reject": "Vol: 2.00x",
        "rank_C_continuous": 3,
        "C_continuous": 1.2,
        "eps_yoy_growth": 25.0,
        "dist_to_52w_high_pct": -4.0,
        "price_52_week_high": 110.0,
        "base_depth_pct": -18.0,
        "base_duration_weeks": 8,
        "pullback_pct": -6.5,
        "pullback_pct_off_peak": -2.0,
        "ibd_entry_valid": True,
        "ibd_trigger_price": 100.0,
        "ibd_entry_date": "2026-07-10",
        "ibd_entry_volume_ratio": 2.0,
        "ibd_entry_reject_reason": "",
    }
    row.update(overrides)
    rendered = []
    monkeypatch.setattr("streamlit.markdown", lambda text, **kwargs: rendered.append(text))
    monkeypatch.setattr("streamlit.info", lambda text, **kwargs: rendered.append(text))
    _render_selected_row_detail(pd.DataFrame([row]), "TEST")
    return rendered[-1]


def test_status_meta_consistency():
    expected_statuses = ["ACTIONABLE", "UNCONFIRMED", "BELOW_TRIGGER", "EXTENDED"]
    for s in expected_statuses:
        assert s in STATUS_META
        assert "label" in STATUS_META[s]
        assert "tone" in STATUS_META[s]
        assert "color" in STATUS_META[s]
        assert "tooltip" in STATUS_META[s]
        assert len(STATUS_META[s]["tooltip"]) > 0

    assert STATUS_META["ACTIONABLE"]["tone"] == "green"
    assert STATUS_META["UNCONFIRMED"]["tone"] == "yellow"
    assert STATUS_META["BELOW_TRIGGER"]["tone"] == "red"
    assert STATUS_META["EXTENDED"]["tone"] == "blue"


def test_header_tooltip_config():
    expected_tooltips = {
        "code": "点击股票代码、Origin 标签或空白处选择该行；仅点击右侧复制按钮复制代码。",
        "ibd_entry_status": "当前 IBD Review 状态。",
        "ibd_candidate_rule": "IBD Candidate 触发价的结构来源。",
        "current_vs_ibd_candidate_pct": "最新收盘价相对 Candidate Price 的距离。",
        "latest_close": "当前数据快照的最新收盘价，不是实时价格。",
        "ibd_entry_vol_or_reject": "日线突破确认：成功显示日线量比，未确认显示原因。",
        "volume_ratio": "当前周成交量相对 10 周均量的倍数。",
        "rank_C_continuous": "综合质量对照排名（只对 Active Signals 计算和展示分布），数值越小越靠前。",
    }
    for col, tip in expected_tooltips.items():
        col_def = _column_def(col)
        assert col_def.get("headerTooltip") == tip, f"Column {col} missing or wrong headerTooltip"
        assert FIELD_CONFIG[col].get("help") == tip


def test_selected_row_card_val_formatting():
    # Test None / NaN / empty
    assert _format_card_val(None, "%") == "n/a"
    assert _format_card_val(np.nan, "x") == "n/a"
    assert _format_card_val("nan", "") == "n/a"
    assert _format_card_val("", "%") == "n/a"

    # Test percentages
    assert _format_card_val(15.234, "%") == "+15.23%"
    assert _format_card_val(-6.812, "%") == "-6.81%"
    assert _format_card_val(0.0, "%") == "0.00%"

    # Test volume ratio
    assert _format_card_val(1.5, "x") == "1.50x"
    assert _format_card_val("2.10", "x") == "2.10x"

    # Test base duration weeks
    assert _format_card_val(8, "w") == "8w"
    assert _format_card_val(8.0, "w") == "8w"
    assert _format_card_val(8.5, "w") == "8.5w"

    # Test raw decimals
    assert _format_card_val(125.4, "") == "125.40"


def test_selected_row_popup_is_semantic_viewport_aware_and_ordered(monkeypatch):
    markup = _selected_detail_markup(monkeypatch)

    assert "<details" in markup
    assert "<summary" in markup
    assert "code-popup-toggle" not in markup
    assert "position-area: block-start span-inline-end" in markup
    assert "position-try-fallbacks: flip-block" in markup
    assert "position-try-order" not in markup
    assert "max-height: min(360px, calc(50dvh - 12px))" in markup
    assert "overflow-y: auto" in markup
    assert '<div class="code-hover-surface">' in markup
    assert "padding: 8px 0" in markup
    assert "margin: 8px" not in markup
    assert ".code-hover-trigger:focus-visible" in markup
    assert "display: list-item" in markup
    assert ".code-hover-popup:hover" in markup
    assert ".st-key-selected_row:has(.code-detail:hover)" in markup
    assert "\n        .code-" not in markup
    assert "onkeydown=" not in markup
    assert ".code-detail:focus-within" not in markup
    assert markup.index("Daily Entry") < markup.index("Pullback") < markup.index("CANSLIM / Base")
    assert "Daily Entry Vol" in markup
    assert "Ceiling/Base Depth" in markup
    assert "TEST · 1 of 1" in markup


def test_selected_row_markup_has_no_blank_lines_that_end_raw_html(monkeypatch):
    markup = _selected_detail_markup(monkeypatch)

    assert all(line.strip() for line in markup.splitlines())


def test_selected_row_popup_hides_empty_pullback_section(monkeypatch):
    markup = _selected_detail_markup(
        monkeypatch,
        pullback_pct=None,
        pullback_pct_off_peak=None,
    )

    assert 'data-popup-section="pullback"' not in markup
    assert "Pullback Depth" not in markup
    assert "Off Pullback Peak" not in markup


def test_selected_row_popup_highlights_invalid_reject_reason(monkeypatch):
    markup = _selected_detail_markup(
        monkeypatch,
        ibd_entry_valid=False,
        ibd_entry_reject_reason="Low Volume",
    )

    assert "Low Volume" in markup
    assert "code-popup-reject" in markup
