import sys
import warnings
from html import unescape

import pandas as pd
import pytest


pytestmark = pytest.mark.filterwarnings("ignore:Type google.protobuf.pyext._message.*:DeprecationWarning")


def _pool_row(
    code,
    *,
    snapshot_date,
    signal,
    status=None,
    valid=None,
    candidate=None,
    close=100.0,
    rank=1,
):
    return {
        "code": code,
        "snapshot_date": snapshot_date,
        "signal": signal,
        "signal_source": "pivot" if signal else None,
        "latest_close": close,
        "ibd_candidate_price": candidate,
        "ibd_candidate_rule": "pivot" if signal else None,
        "ibd_entry_valid": valid,
        "ibd_entry_status": status,
        "current_vs_ibd_candidate_pct": (
            (close / candidate - 1.0) * 100.0 if candidate else None
        ),
        "ibd_entry_volume_ratio": 2.0 if valid else None,
        "ibd_entry_reject_reason": None if valid else "Volume not confirmed",
        "volume_ratio": 1.5,
        "rank_C_continuous": rank,
        "C_continuous": float(rank),
    }


@pytest.fixture
def review_paths(tmp_path):
    complete = pd.DataFrame(
        [
            _pool_row("CARRY", snapshot_date="2026-07-24", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=103, rank=1),
            _pool_row("LEFT", snapshot_date="2026-07-24", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=103, rank=2),
            _pool_row("RECONF", snapshot_date="2026-07-24", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=103, rank=3),
            _pool_row("EXITED", snapshot_date="2026-07-24", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=103, rank=4),
            _pool_row("PLAIN", snapshot_date="2026-07-24", signal=False, rank=5),
        ]
    )
    midweek = pd.DataFrame(
        [
            _pool_row("NEW", snapshot_date="2026-07-30", signal=True, status="ACTIONABLE", valid=True, candidate=50, close=51, rank=1),
            _pool_row("CARRY", snapshot_date="2026-07-30", signal=False, close=104, rank=2),
            _pool_row("LEFT", snapshot_date="2026-07-30", signal=True, status="EXTENDED", valid=True, candidate=100, close=108, rank=3),
            _pool_row("RECONF", snapshot_date="2026-07-30", signal=True, status="UNCONFIRMED", valid=False, candidate=100, close=101, rank=4),
            _pool_row("PLAIN", snapshot_date="2026-07-30", signal=False, close=99, rank=5),
        ]
    )
    complete_path = tmp_path / "breakout_follow_pool.csv"
    midweek_path = tmp_path / "breakout_follow_pool_midweek.csv"
    complete.to_csv(complete_path, index=False)
    midweek.to_csv(midweek_path, index=False)
    return {"complete": complete_path, "midweek": midweek_path}


def _review_app(paths, *, window_date):
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf.*")
    from streamlit.testing.v1 import AppTest

    old_argv = sys.argv[:]
    sys.argv = [
        "dashboard/app.py",
        "--csv",
        str(paths["complete"]),
        "--midweek-csv",
        str(paths["midweek"]),
        "--window-date",
        window_date,
    ]
    try:
        app = AppTest.from_file("dashboard/app.py", default_timeout=10).run(timeout=30)
        app._review_argv = list(sys.argv)
        return app
    finally:
        sys.argv = old_argv


def test_default_dashboard_screen_renders_with_fixed_snapshots(review_paths):
    app = _review_app(review_paths, window_date="2026-07-30")

    assert len(app.exception) == 0
    assert [title.value for title in app.title] == []
    assert any("results · Sorted by Review Priority" in item.value for item in app.markdown)
    assert any("Midweek · baseline" in item.value for item in app.markdown)
    assert not any(widget.label == "Route (Rule)" for widget in app.selectbox)
    assert not any(widget.key and str(widget.key).startswith("review_sort_") for widget in app.selectbox)


def _midweek_app(review_paths):
    return _review_app(review_paths, window_date="2026-07-30")


def _complete_window_with_valid_midweek_app(review_paths):
    return _review_app(review_paths, window_date="2026-08-03")


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


def test_midweek_runtime_defaults_to_changes_review_priority_and_collapsed_filters(review_paths):
    app = _midweek_app(review_paths)

    assert len(app.exception) == 0
    state = app.session_state["review_ui_state"]
    assert state["mode"] == "MIDWEEK"
    assert state["scope"] == "CHANGES"
    assert state["sort_mode"] == "Review Priority"
    assert state["filters_expanded"] is False
    assert app.button(key="btn_filters_toggle").label == "More Filters · None"
    assert not any(widget.label == "Route (Rule)" for widget in app.selectbox)
    assert any("results · Sorted by Review Priority" in item.value for item in app.markdown)


def test_midweek_runtime_exposes_all_ten_structured_tooltips(review_paths):
    app = _midweek_app(review_paths)
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


def test_runtime_scope_buttons_expose_expected_enabled_states(review_paths):
    app = _midweek_app(review_paths)
    assert _button_starting_with(app, "Changes").disabled is False
    assert _button_starting_with(app, "All Signals").disabled is False
    assert _button_starting_with(app, "Midweek Review").disabled is False


def test_runtime_mode_scope_and_quick_filter_flow(review_paths):
    app = _midweek_app(review_paths)

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
    assert any("All Signals · 4" in item.value for item in app.markdown)
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


def test_complete_window_manual_midweek_uses_the_valid_baseline_comparison(review_paths):
    app = _complete_window_with_valid_midweek_app(review_paths)

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
    assert any("3 results · Sorted by Review Priority" in item.value for item in app.markdown)


def test_runtime_status_filter_expansion_and_reset_flow(review_paths):
    app = _midweek_app(review_paths)

    _click(app, "btn_status_ACTIONABLE")
    assert app.session_state["review_ui_state"]["status_filter"] == "ACTIONABLE"
    assert _button_starting_with(app, "ACTIONABLE").label.startswith("ACTIONABLE")

    _click(app, "btn_filters_toggle")
    assert app.session_state["review_ui_state"]["filters_expanded"] is True
    entry_volume = app.slider(key="review_entry_vol_0")

    entry_volume.set_value(1.5)
    _run_widget_change(app)
    assert app.session_state["review_ui_state"]["entry_volume_min"] == 1.5
    assert "1 active" in app.button(key="btn_filters_toggle").label
    assert app.button(key="btn_filters_reset").label == "Reset"

    app.slider(key="review_weekly_vol_0").set_value(1.0)
    _run_widget_change(app)
    assert app.button(key="btn_filters_toggle").label == "More Filters · 2 active"
    assert app.button(key="btn_filters_reset").label == "Reset"

    _click(app, "btn_filters_reset")
    state = app.session_state["review_ui_state"]
    assert state["status_filter"] == "ACTIONABLE"
    assert state["entry_volume_min"] is None
    assert state["weekly_volume_min"] is None
    assert state["widget_generation"] == 1
    assert app.button(key="btn_filters_toggle").label == "More Filters · None"
    assert app.session_state["review_ui_state"]["filters_expanded"] is True


def test_runtime_advanced_filters_apply_immediately_and_compose_with_and(review_paths):
    app = _midweek_app(review_paths)
    _click(app, "btn_filters_toggle")

    app.radio(key="review_route_0").set_value("pivot")
    _run_widget_change(app)
    app.slider(key="review_distance_0").set_value((0.0, 5.0))
    _run_widget_change(app)
    app.slider(key="review_entry_vol_0").set_value(1.5)
    _run_widget_change(app)
    app.slider(key="review_weekly_vol_0").set_value(1.0)
    _run_widget_change(app)

    state = app.session_state["review_ui_state"]
    assert state["route_filter"] == "pivot"
    assert state["distance_range"] == (0.0, 5.0)
    assert state["entry_volume_min"] == 1.5
    assert state["weekly_volume_min"] == 1.0
    assert app.button(key="btn_filters_toggle").label == "More Filters · 4 active"
    assert any("results · Sorted by Review Priority" in item.value for item in app.markdown)
    assert app.button(key="btn_filters_reset").disabled is False
    assert not any(
        button.key and str(button.key).startswith("btn_filter_chip_")
        for button in app.button
    )


def test_runtime_uses_fixed_view_sort_without_toolbar_selector(review_paths):
    app = _midweek_app(review_paths)

    assert not any(widget.key and str(widget.key).startswith("review_sort_") for widget in app.selectbox)
    _click(app, "btn_scope_all_signals")

    assert len(app.exception) == 0
    assert app.session_state["review_ui_state"]["sort_mode"] == "C Rank"
    assert any("results · Sorted by C Rank" in item.value for item in app.markdown)


def test_runtime_global_mode_switches_to_c_rank_reference_and_back_to_ibd_review(review_paths):
    app = _midweek_app(review_paths)

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


def test_runtime_header_tracks_midweek_weekend_and_c_rank_data_sources(review_paths):
    app = _midweek_app(review_paths)

    midweek_header = _header_markup(app)
    assert "Snapshot <b>2026-07-30</b>" in midweek_header
    assert "<b>5</b> Total Pool" in midweek_header
    assert "<b>4</b> Active Signals" in midweek_header

    _click(app, "btn_mode_weekend")
    weekend_header = _header_markup(app)
    assert "Snapshot <b>2026-07-24</b>" in weekend_header
    assert "<b>5</b> Total Pool" in weekend_header
    assert "<b>4</b> Active Signals" in weekend_header

    _click(app, "btn_mode_midweek")
    app.get("button_group")[0].set_value(["C Rank Reference"])
    _run_with_review_argv(app)

    c_rank_header = _header_markup(app)
    assert "Snapshot <b>2026-07-24</b>" in c_rank_header
    assert "<b>5</b> Total Pool" in c_rank_header
    assert "<b>4</b> Active Signals" in c_rank_header
    assert "snapshot-mode-segment--midweek" not in c_rank_header


def test_c_rank_top_n_keeps_result_summary_complete(review_paths):
    app = _midweek_app(review_paths)
    app.get("button_group")[0].set_value(["C Rank Reference"])
    _run_with_review_argv(app)

    assert any(
        "Showing: 4 of 4 Active Signals · Reference Only" in item.value
        for item in app.markdown
    )

    app.selectbox(key="c_rank_top_n_select").select("Top 10")
    _run_widget_change(app)
    assert any(
        "Showing: 4 of 4 Active Signals · Reference Only" in item.value
        for item in app.markdown
    )

    app.selectbox(key="c_rank_top_n_select").select("Top 25")
    _run_widget_change(app)
    assert any(
        "Showing: 4 of 4 Active Signals · Reference Only" in item.value
        for item in app.markdown
    )


def test_runtime_surfaces_missing_baseline_warning(tmp_path, review_paths):
    from streamlit.testing.v1 import AppTest

    complete = pd.read_csv(review_paths["complete"])
    midweek = pd.read_csv(review_paths["midweek"])
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


@pytest.mark.parametrize("midweek_state", ["missing", "outdated"])
def test_unavailable_midweek_falls_back_as_a_normal_weekend_state(tmp_path, midweek_state, review_paths):
    from streamlit.testing.v1 import AppTest

    complete = pd.read_csv(review_paths["complete"])
    midweek = pd.read_csv(review_paths["midweek"])
    complete["snapshot_date"] = "2026-07-31"
    midweek["snapshot_date"] = "2026-07-30"
    complete_path = tmp_path / "breakout_follow_pool.csv"
    midweek_path = tmp_path / "breakout_follow_pool_midweek.csv"
    complete.to_csv(complete_path, index=False)
    if midweek_state == "outdated":
        midweek.to_csv(midweek_path, index=False)

    old_argv = sys.argv[:]
    sys.argv = [
        "dashboard/app.py",
        "--csv",
        str(complete_path),
        "--midweek-csv",
        str(midweek_path),
        "--window-date",
        "2026-08-04",
    ]
    try:
        app = AppTest.from_file("dashboard/app.py", default_timeout=10).run(timeout=30)
    finally:
        sys.argv = old_argv

    assert len(app.exception) == 0
    state = app.session_state["review_ui_state"]
    assert (state["mode"], state["scope"], state["sort_mode"]) == (
        "WEEKEND",
        "ALL_SIGNALS",
        "C Rank",
    )
    midweek_button = app.button(key="btn_mode_midweek")
    assert midweek_button.disabled is True
    assert midweek_button.help == (
        "当前没有可与周末基线比较的周中数据，已自动显示最新的周末完整候选池。"
    )
    assert not any("Midweek snapshot" in warning.value for warning in app.warning)

    assert not any(
        "weekend-context-bar--unavailable" in item.value
        or "Midweek unavailable" in item.value
        for item in app.markdown
    )
    assert any("All Signals ·" in item.value for item in app.markdown)

    header = _header_markup(app)
    assert "Snapshot <b>2026-07-31</b>" in header
    assert "Midweek" not in header
