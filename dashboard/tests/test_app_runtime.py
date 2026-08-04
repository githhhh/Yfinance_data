import sys
import warnings
from html import unescape

import pandas as pd
import pytest


pytestmark = pytest.mark.filterwarnings("ignore:Type google.protobuf.pyext._message.*:DeprecationWarning")


def test_default_dashboard_screen_renders_with_real_csv():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf.*")
    from streamlit.testing.v1 import AppTest

    old_argv = sys.argv[:]
    sys.argv = ["dashboard/app.py", "--csv", "us/breakout_follow_pool.csv"]
    try:
        app = AppTest.from_file("dashboard/app.py", default_timeout=10).run(timeout=30)
    finally:
        sys.argv = old_argv

    assert len(app.exception) == 0
    assert [title.value for title in app.title] == []
    assert any("results · Sorted by Review Priority" in item.value for item in app.markdown)
    assert any("Midweek · baseline" in item.value for item in app.markdown)
    assert not any(widget.label == "Route (Rule)" for widget in app.selectbox)
    assert not any(widget.key and str(widget.key).startswith("review_sort_") for widget in app.selectbox)


def _midweek_app():
    from streamlit.testing.v1 import AppTest

    old_argv = sys.argv[:]
    sys.argv = [
        "dashboard/app.py",
        "--csv",
        "us/breakout_follow_pool.csv",
        "--midweek-csv",
        "us/breakout_follow_pool_midweek.csv",
        "--window-date",
        "2026-07-30",
    ]
    try:
        app = AppTest.from_file("dashboard/app.py", default_timeout=10).run(timeout=30)
        app._review_argv = list(sys.argv)
        return app
    finally:
        sys.argv = old_argv


def _complete_window_with_valid_midweek_app():
    from streamlit.testing.v1 import AppTest

    old_argv = sys.argv[:]
    sys.argv = [
        "dashboard/app.py",
        "--csv",
        "us/breakout_follow_pool.csv",
        "--midweek-csv",
        "us/breakout_follow_pool_midweek.csv",
        "--window-date",
        "2026-08-03",
    ]
    try:
        app = AppTest.from_file("dashboard/app.py", default_timeout=10).run(timeout=30)
        app._review_argv = list(sys.argv)
        return app
    finally:
        sys.argv = old_argv


def _button_starting_with(app, label: str):
    normalized = lambda value: value.replace("**", "").lstrip("✓  ↘↗↔+→")
    return next(button for button in app.button if normalized(button.label).startswith(label))


def _prepare_app_test_interaction(app):
    """Normalize Streamlit 1.45's scalar segmented-control test value."""
    for button_group in app.get("button_group"):
        if isinstance(button_group.value, str):
            button_group.set_value([button_group.value])


def _click(app, key: str):
    _prepare_app_test_interaction(app)
    app.button(key=key).click()
    return _run_with_review_argv(app)


def _run_widget_change(app):
    _prepare_app_test_interaction(app)
    return _run_with_review_argv(app)


def _run_with_review_argv(app):
    old_argv = sys.argv[:]
    sys.argv = list(app._review_argv)
    try:
        return app.run(timeout=30)
    finally:
        sys.argv = old_argv


def _header_markup(app):
    return next(
        item.value
        for item in app.markdown
        if '<div class="dashboard-title"' in item.value
        and '<div class="dashboard-snapshot"' in item.value
    )


def test_midweek_runtime_defaults_to_changes_review_priority_and_collapsed_filters():
    app = _midweek_app()

    assert len(app.exception) == 0
    state = app.session_state["review_ui_state"]
    assert state["mode"] == "MIDWEEK"
    assert state["scope"] == "CHANGES"
    assert state["sort_mode"] == "Review Priority"
    assert state["filters_expanded"] is False
    assert app.button(key="btn_filters_toggle").label == "More Filters · None"
    assert not any(widget.label == "Route (Rule)" for widget in app.selectbox)
    assert any("results · Sorted by Review Priority" in item.value for item in app.markdown)


def test_midweek_runtime_exposes_all_ten_structured_tooltips():
    app = _midweek_app()
    card_names = [
        "Entered Buy Zone",
        "Left Buy Zone",
        "Other Changes",
        "New Signal",
        "Carried Over",
        "Reconfirmed",
        "ACTIONABLE",
        "UNCONFIRMED",
        "BELOW TRIGGER",
        "EXTENDED",
    ]

    trigger_markup = [
        unescape(item.value)
        for item in app.markdown
        if 'class="flow-info-trigger"' in item.value
    ]
    assert len(trigger_markup) == 10
    for card_name in card_names:
        button = _button_starting_with(app, card_name)
        assert not button.help
        markup = next(value for value in trigger_markup if f'data-flow-tooltip-title="{card_name}"' in value)
        assert "含义：" in markup
        assert "数量：" in markup
        assert "点击：" in markup
        assert f'aria-label="{card_name} info"' in markup
        assert ">i</button>" in markup
    assert not any(button.key and button.key.startswith("btn_flow_info_") for button in app.button)


def test_runtime_scope_buttons_expose_expected_enabled_states():
    app = _midweek_app()
    assert _button_starting_with(app, "Changes").disabled is False
    assert _button_starting_with(app, "All Signals").disabled is False
    assert _button_starting_with(app, "Midweek Review").disabled is False


def test_runtime_mode_scope_and_quick_filter_flow():
    app = _midweek_app()

    assert app.session_state["review_ui_state"]["change_filter"] == "ALL"
    assert app.session_state["review_ui_state"]["origin_filter"] == "ALL"
    assert not any(button.key == "btn_clear_quick" for button in app.button)

    _click(app, "btn_mode_weekend")
    state = app.session_state["review_ui_state"]
    assert len(app.exception) == 0
    assert state["mode"] == "WEEKEND"
    assert state["scope"] == "ALL_SIGNALS"
    assert state["sort_mode"] == "C Rank"
    assert not any(button.key == "btn_scope_changes" for button in app.button)
    assert any("All Signals · 106" in item.value for item in app.markdown)
    assert any("Weekend Baseline" in item.value for item in app.markdown)

    _click(app, "btn_mode_midweek")
    state = app.session_state["review_ui_state"]
    assert state["mode"] == "MIDWEEK"
    assert state["scope"] == "CHANGES"
    assert state["sort_mode"] == "Review Priority"

    _click(app, "btn_change_filter_BECAME_ACTIONABLE")
    assert app.session_state["review_ui_state"]["change_filter"] == "BECAME_ACTIONABLE"
    assert app.button(key="btn_clear_quick").label == "Clear 1"

    _click(app, "btn_origin_filter_NEW")
    assert app.button(key="btn_clear_quick").label == "Clear 2"

    _click(app, "btn_clear_quick")
    state = app.session_state["review_ui_state"]
    assert state["change_filter"] == "ALL"
    assert state["origin_filter"] == "ALL"
    assert state["mode"] == "MIDWEEK"
    assert state["scope"] == "CHANGES"
    assert not any(button.key == "btn_clear_quick" for button in app.button)


def test_complete_window_manual_midweek_uses_the_valid_baseline_comparison():
    app = _complete_window_with_valid_midweek_app()

    initial = app.session_state["review_ui_state"]
    assert initial["mode"] == "WEEKEND"
    assert initial["scope"] == "ALL_SIGNALS"
    assert initial["sort_mode"] == "C Rank"

    _click(app, "btn_mode_midweek")

    state = app.session_state["review_ui_state"]
    assert len(app.exception) == 0
    assert state["mode"] == "MIDWEEK"
    assert state["scope"] == "CHANGES"
    assert state["sort_mode"] == "Review Priority"
    assert app.button(key="btn_scope_changes").disabled is False
    assert any("Midweek · baseline 2026-07-24" in item.value for item in app.markdown)
    assert any("111 results · Sorted by Review Priority" in item.value for item in app.markdown)


def test_runtime_status_filter_expansion_and_reset_flow():
    app = _midweek_app()

    _click(app, "btn_status_ACTIONABLE")
    assert app.session_state["review_ui_state"]["status_filter"] == "ACTIONABLE"
    assert _button_starting_with(app, "ACTIONABLE").label.startswith("ACTIONABLE")

    _click(app, "btn_filters_toggle")
    assert app.session_state["review_ui_state"]["filters_expanded"] is True
    entry_volume = app.text_input(key="review_entry_vol_0")
    assert entry_volume.disabled is False

    entry_volume.input("1.5")
    _run_widget_change(app)
    assert app.session_state["review_ui_state"]["entry_volume_min"] == "1.5"
    assert "1 active" in app.button(key="btn_filters_toggle").label

    _click(app, "btn_filters_reset")
    state = app.session_state["review_ui_state"]
    assert state["status_filter"] == "ALL"
    assert state["entry_volume_min"] == ""
    assert state["widget_generation"] == 1
    assert app.button(key="btn_filters_toggle").label == "More Filters · None"


def test_runtime_uses_fixed_view_sort_without_toolbar_selector():
    app = _midweek_app()

    assert not any(widget.key and str(widget.key).startswith("review_sort_") for widget in app.selectbox)
    _click(app, "btn_scope_all_signals")

    assert len(app.exception) == 0
    assert app.session_state["review_ui_state"]["sort_mode"] == "C Rank"
    assert any("results · Sorted by C Rank" in item.value for item in app.markdown)


def test_runtime_global_mode_switches_to_c_rank_reference_and_back_to_ibd_review():
    app = _midweek_app()

    app.get("button_group")[0].set_value(["C Rank Reference"])
    app.run(timeout=30)

    assert len(app.exception) == 0
    assert app.session_state["global_mode_selector"] == "C Rank Reference"
    assert any("C Rank Reference View" in item.value for item in app.markdown)
    assert any(widget.label == "Top N Slice" for widget in app.selectbox)

    app.get("button_group")[0].set_value(["IBD Review"])
    _run_with_review_argv(app)

    assert len(app.exception) == 0
    assert app.session_state["global_mode_selector"] == "IBD Review"
    assert any("Review Queue" in item.value for item in app.markdown)
    assert not any(widget.label == "Top N Slice" for widget in app.selectbox)


def test_runtime_header_tracks_midweek_weekend_and_c_rank_data_sources():
    app = _midweek_app()

    midweek_header = _header_markup(app)
    assert "Snapshot <b>2026-07-30</b>" in midweek_header
    assert "<b>751</b> Total Pool" in midweek_header
    assert "<b>190</b> Active Signals" in midweek_header

    _click(app, "btn_mode_weekend")
    weekend_header = _header_markup(app)
    assert "Snapshot <b>2026-07-24</b>" in weekend_header
    assert "<b>745</b> Total Pool" in weekend_header
    assert "<b>106</b> Active Signals" in weekend_header

    _click(app, "btn_mode_midweek")
    app.get("button_group")[0].set_value(["C Rank Reference"])
    _run_with_review_argv(app)

    c_rank_header = _header_markup(app)
    assert "Snapshot <b>2026-07-24</b>" in c_rank_header
    assert "<b>745</b> Total Pool" in c_rank_header
    assert "<b>106</b> Active Signals" in c_rank_header
    assert "snapshot-mode-segment--midweek" not in c_rank_header


def test_c_rank_top_n_keeps_result_summary_complete():
    app = _midweek_app()
    app.get("button_group")[0].set_value(["C Rank Reference"])
    _run_with_review_argv(app)

    assert any(
        "Showing: 106 of 106 Active Signals · Reference Only" in item.value
        for item in app.markdown
    )

    app.selectbox(key="c_rank_top_n_select").select("Top 10")
    _run_widget_change(app)
    assert any(
        "Showing: 10 of 106 Active Signals · Reference Only" in item.value
        for item in app.markdown
    )

    app.selectbox(key="c_rank_top_n_select").select("Top 25")
    _run_widget_change(app)
    assert any(
        "Showing: 25 of 106 Active Signals · Reference Only" in item.value
        for item in app.markdown
    )


def test_runtime_surfaces_missing_baseline_warning(tmp_path):
    from streamlit.testing.v1 import AppTest

    complete = pd.read_csv("us/breakout_follow_pool.csv")
    midweek = pd.read_csv("us/breakout_follow_pool_midweek.csv")
    complete["snapshot_date"] = "2026-07-17"
    midweek["snapshot_date"] = "2026-07-29"
    complete_path = tmp_path / "breakout_follow_pool.csv"
    midweek_path = tmp_path / "breakout_follow_pool_midweek.csv"
    complete.to_csv(complete_path, index=False)
    midweek.to_csv(midweek_path, index=False)

    old_argv = sys.argv[:]
    sys.argv = [
        "dashboard/app.py",
        "--csv",
        str(complete_path),
        "--midweek-csv",
        str(midweek_path),
        "--window-date",
        "2026-07-30",
    ]
    try:
        app = AppTest.from_file("dashboard/app.py", default_timeout=10).run(timeout=30)
        stale_state = dict(app.session_state["review_ui_state"])
        stale_state.update(
            {
                "scope": "ALL_SIGNALS",
                "change_filter": "BECAME_ACTIONABLE",
                "origin_filter": "CARRY",
                "sort_mode": "Review Priority",
            }
        )
        app.session_state["review_ui_state"] = stale_state
        _prepare_app_test_interaction(app)
        app = app.run(timeout=30)
    finally:
        sys.argv = old_argv

    assert len(app.exception) == 0
    assert any("no valid complete-week baseline" in warning.value for warning in app.warning)
    state = app.session_state["review_ui_state"]
    assert (state["mode"], state["scope"], state["sort_mode"]) == (
        "MIDWEEK",
        "ALL_SIGNALS",
        "C Rank",
    )
    assert state["change_filter"] == "ALL"
    assert state["origin_filter"] == "ALL"
    assert not any("0 results ·" in item.value for item in app.markdown)
    assert not any(button.key == "btn_scope_changes" for button in app.button)
    assert any("All Signals ·" in item.value for item in app.markdown)
    assert not any(button.key == "btn_change_filter_BECAME_ACTIONABLE" for button in app.button)
    assert any("Change and Origin comparison is unavailable" in item.value for item in app.markdown)
    assert any("Midweek · baseline unavailable" in item.value for item in app.markdown)
