from __future__ import annotations

import re
from pathlib import Path


DASHBOARD_DIR = Path(__file__).resolve().parents[1]
APP_SOURCE = DASHBOARD_DIR.joinpath("app.py").read_text(encoding="utf-8")
STYLE_PATH = DASHBOARD_DIR / "review_styles.py"


def _compact_css() -> str:
    assert STYLE_PATH.exists(), "dashboard/review_styles.py must own the Review UI CSS"
    source = STYLE_PATH.read_text(encoding="utf-8")
    return re.sub(r"\s+", "", source)


def _function_source(name: str, next_name: str) -> str:
    start = APP_SOURCE.index(f"def {name}(")
    end = APP_SOURCE.index(f"def {next_name}(", start)
    return APP_SOURCE[start:end]


def test_review_ui_css_is_centralized_and_injected_once():
    assert STYLE_PATH.exists(), "dashboard/review_styles.py must exist"
    assert "from dashboard.review_styles import REVIEW_UI_CSS" in APP_SOURCE
    assert 'st.markdown(f"<style>{REVIEW_UI_CSS}</style>"' in APP_SOURCE
    assert APP_SOURCE.count("REVIEW_UI_CSS") == 2
    assert "padding: 8px 16px 16px" not in APP_SOURCE
    assert "@media (max-width: 900px)" not in APP_SOURCE


def test_reference_tokens_and_desktop_geometry_are_explicit():
    css = _compact_css()
    for declaration in [
        "--bg:#0c1016",
        "--panel:#151b23",
        "--panel-soft:#111720",
        "--input:#202b3a",
        "--line:#35404d",
        "--text:#f4f5f7",
        "--muted:#9ca8b7",
        "--green:#35df65",
        "--cyan:#1fcdb4",
        "--blue:#2791ff",
        "--yellow:#ffd21f",
        "--red:#f04444",
        "padding:29px28px34px",
        "min-height:78px",
        "grid-template-columns:minmax(0,1fr)276px268px",
        "height:48px",
        "grid-template-columns:repeat(4,minmax(0,1fr))",
        "height:70px",
        "height:45px",
        "min-height:56px",
        "height:60px",
        "grid-template-columns:194pxrepeat(4,minmax(0,1fr))",
    ]:
        assert declaration in css


def test_reference_breakpoints_focus_and_reduced_motion_are_explicit():
    css = _compact_css()
    for rule in [
        "@media(width<=1120px)",
        "@media(width<=760px)",
        "@media(width<=480px)",
        "@media(prefers-reduced-motion:reduce)",
        ":focus-visible",
    ]:
        assert rule in css


def test_review_context_is_one_horizontal_flow_with_stable_slots():
    source = _function_source("_render_review_context", "_render_status_queue")

    assert "rows = [" not in source
    assert 'key="quick_context_row"' in source
    assert 'key="quick_label_change"' in source
    assert 'key="quick_divider"' in source
    assert 'key="quick_label_origin"' in source
    assert source.count("_render_quick_group(") == 2
    assert 'key="btn_clear_quick"' in source
    assert "Weekend Baseline" in source


def test_queue_controls_have_stable_heading_and_segmented_group_slots():
    source = _function_source("_render_mode_scope_controls", "df_active_count_for_state")

    for key in [
        "review_queue_heading",
        "review_mode_controls",
        "review_scope_controls",
    ]:
        assert f'key="{key}"' in source
    assert "disabled=not is_midweek" in source
