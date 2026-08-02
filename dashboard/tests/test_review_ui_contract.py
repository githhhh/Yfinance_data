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
        "grid-template-columns:276px268px",
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
