import sys
import warnings

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
    assert any("results · Sorted by C Rank" in item.value for item in app.markdown)
    assert any("Weekend Baseline" in item.value for item in app.markdown)
    assert not any(widget.label == "Route (Rule)" for widget in app.selectbox)
    assert len(app.selectbox) > 0


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
        return AppTest.from_file("dashboard/app.py", default_timeout=10).run(timeout=30)
    finally:
        sys.argv = old_argv


def _button_starting_with(app, label: str):
    normalized = lambda value: value.lstrip("✓  🟢🟡🔴🔵")
    return next(button for button in app.button if normalized(button.label).startswith(label))


def test_midweek_runtime_defaults_to_changes_review_priority_and_collapsed_filters():
    app = _midweek_app()

    assert len(app.exception) == 0
    state = app.session_state["review_ui_state"]
    assert state["mode"] == "MIDWEEK"
    assert state["scope"] == "CHANGES"
    assert state["sort_mode"] == "Review Priority"
    assert state["filters_expanded"] is False
    assert not any(widget.label == "Route (Rule)" for widget in app.selectbox)
    assert any("results · Sorted by Review Priority" in item.value for item in app.markdown)


def test_midweek_runtime_exposes_all_ten_structured_tooltips():
    app = _midweek_app()
    card_names = [
        "Became Actionable",
        "Left Actionable",
        "Other Changes",
        "New",
        "Carry",
        "Reconfirmed",
        "ACTIONABLE",
        "UNCONFIRMED",
        "BELOW TRIGGER",
        "EXTENDED",
    ]

    for card_name in card_names:
        button = _button_starting_with(app, card_name)
        assert "Definition:" in button.help
        assert "Count:" in button.help
        assert "Click:" in button.help


def test_runtime_scope_buttons_expose_expected_enabled_states():
    app = _midweek_app()
    assert _button_starting_with(app, "Changes").disabled is False
    assert _button_starting_with(app, "All Signals").disabled is False
    assert _button_starting_with(app, "Midweek Review").disabled is False


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
    finally:
        sys.argv = old_argv

    assert len(app.exception) == 0
    assert any("no valid complete-week baseline" in warning.value for warning in app.warning)
