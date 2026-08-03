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
        "padding:24px24px30px",
        "min-height:80px",
        "grid-template-columns:minmax(0,1fr)276px268px",
        "height:48px",
        "grid-template-columns:repeat(4,minmax(0,1fr))",
        "height:70px",
        "height:45px",
        "min-height:56px",
        "height:60px",
        "grid-template-columns:194pxrepeat(4,minmax(0,1fr))",
        "font-size:29px",
        "font-weight:800",
        "margin-top:20px",
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
    assert "No valid complete-week baseline" in source


def test_queue_controls_have_stable_heading_and_segmented_group_slots():
    source = _function_source("_render_mode_scope_controls", "df_active_count_for_state")

    for key in [
        "review_queue_heading",
        "review_mode_controls",
        "review_scope_controls",
    ]:
        assert f'key="{key}"' in source
    assert "disabled=not has_comparison" in source


def test_status_labels_use_fixed_text_slots_without_emoji():
    source = _function_source("_render_status_queue", "_active_filter_count")

    for emoji in ["🟢", "🟡", "🔴", "🔵", "⚪"]:
        assert emoji not in source
    assert 'prefix = "✓ " if is_active else "  "' in source


def test_quick_dots_and_status_orbs_are_styled():
    css = _compact_css()

    for declaration in [
        "width:7px",
        "height:7px",
        "width:19px",
        "height:19px",
        "box-shadow:inset02px3px",
    ]:
        assert declaration in css


def test_copy_control_has_no_permanent_manual_and_keeps_two_copy_paths():
    source = _function_source("_render_copy_codes_control", "_download_current_rows")

    assert 'st.popover("Manual"' not in source
    assert "navigator.clipboard.writeText" in source
    assert "document.execCommand('copy')" in source
    assert "Copy failed" in source
    assert "disabled_attr" in source


def test_results_and_selected_row_use_stable_named_surfaces():
    assert 'key="results_actions"' in APP_SOURCE
    assert 'class="selected-strip"' in APP_SOURCE
    assert 'class="selected-strip selected-strip--empty"' in APP_SOURCE


def test_header_markup_and_control_surfaces_follow_reference_hierarchy():
    css = _compact_css()
    source = _function_source("_render_header_bar", "_render_flow_rules_dialog")

    for marker in [
        'class="dashboard-title"',
        'class="data-badge data-badge--ready"',
        'class="data-badge data-badge--error"',
        'class="dashboard-snapshot"',
        'freshness["snapshot_date_str"]',
        'snapshot-freshness--{freshness["status"].lower()}',
    ]:
        assert marker in APP_SOURCE
    for declaration in [
        "background:var(--panel)",
        "background:#151c24",
        "border:1pxsolid#36414e",
        "white-space:pre-line",
        "justify-content:center",
    ]:
        assert declaration in css
    assert "<hr" not in source


def test_mobile_header_results_and_actions_have_explicit_stack_contracts():
    css = _compact_css()

    assert '.st-key-dashboard_header>div[data-testid="stHorizontalBlock"]' in css
    assert '.st-key-results_toolbar>div[data-testid="stHorizontalBlock"]' in css
    assert '.st-key-results_actions>div[data-testid="stHorizontalBlock"]' in css
    assert '.st-key-review_queue_heading>div[data-testid="stHorizontalBlock"]' in css
    assert (
        '.st-key-review_queue_heading>div[data-testid="stHorizontalBlock"]'
        '>div[data-testid="stColumn"]{width:100%!important;min-width:0!important;flex:none!important;}'
    ) in css
    assert (
        '.st-key-quick_context_row>div[data-testid="stHorizontalBlock"]'
        '{min-width:max-content;height:46px;align-items:center;gap:6px!important;}'
    ) in css
    assert (
        '.st-key-review_context_slotdiv[class*="st-key-flow_card_"]'
        '{position:relative;min-width:154px;}'
    ) in css
    for selector in [
        '.st-key-dashboard_header>div[data-testid="stHorizontalBlock"]>div[data-testid="stColumn"]',
        '.st-key-results_toolbar>div[data-testid="stHorizontalBlock"]>div[data-testid="stColumn"]',
        '.st-key-results_actions>div[data-testid="stHorizontalBlock"]>div[data-testid="stColumn"]',
    ]:
        assert selector + '{width:100%!important;min-width:0!important;flex:none!important;}' in css
    assert "grid-template-columns:minmax(180px,1.5fr)minmax(120px,1fr)" in css
    assert "font-size:25px" in css


def test_status_card_grid_targets_streamlits_direct_horizontal_block():
    css = _compact_css()
    direct = '.st-key-status_cards>div[data-testid="stHorizontalBlock"]'
    nested = (
        '.st-key-status_cards>div[data-testid="stVerticalBlock"]'
        '>div[data-testid="stElementContainer"]>div[data-testid="stHorizontalBlock"]'
    )

    assert direct in css
    assert nested not in css
    assert (
        direct + '>div[data-testid="stColumn"]{width:100%!important;min-width:0!important;flex:none!important;}'
    ) in css


def test_keyed_vertical_blocks_remove_streamlits_default_density_gap():
    css = _compact_css()

    for selector in [
        ".st-key-dashboard_shell{gap:0!important;}",
        ".st-key-review_queue{gap:0!important;}",
        ".st-key-filters{gap:0!important;}",
        ".st-key-results_toolbar{gap:0!important;}",
        ".st-key-selected_row{gap:0!important;}",
        ".st-key-results_grid{gap:0!important;}",
    ]:
        assert selector in css
    title_rule = css.split(".dashboard-title{", 1)[1].split("}", 1)[0]
    assert "padding:0!important" in title_rule
    assert (
        'div[data-testid="stMainBlockContainer"]:has(.st-key-dashboard_shell)'
        '>div[data-testid="stVerticalBlockBorderWrapper"]'
        '>div[data-testid="stVerticalBlock"]{gap:0!important;}'
    ) in css


def test_header_actions_are_right_aligned_to_the_content_edge():
    css = _compact_css()

    assert "grid-template-columns:45px235px" in css
    assert "justify-content:end" in css
    assert (
        '.st-key-dashboard_headerdiv[data-testid="stColumn"]:has(.st-key-btn_info_rules)'
        '>div[data-testid="stVerticalBlockBorderWrapper"]'
        '>div[data-testid="stVerticalBlock"]'
        '>div[data-testid="stHorizontalBlock"]{display:grid!important'
    ) in css
    assert "grid-template-columns:99px136px" in css
    assert "max-width:99px" in css
    assert "max-width:136px" in css
    assert "padding:010px!important" in css


def test_selected_ibd_review_and_context_bar_have_explicit_active_surfaces():
    css = _compact_css()

    selected_rule = css.split(
        '.st-key-dashboard_headerdiv[class*="st-key-global_mode_selector"]button[aria-pressed="true"],',
        1,
    )[1].split("}", 1)[0]
    for declaration in [
        "border-color:#00d897!important",
        "background:rgb(0216151/12%)!important",
        "color:#29f2b0!important",
        "box-shadow:inset0001px#00d897!important",
    ]:
        assert declaration in selected_rule

    assert (
        ".st-key-review_context_slot:has(.st-key-quick_context_row)"
        "{border:1pxsolid#303a46;border-radius:7px;padding:010px;"
        "background:#11171e;box-sizing:border-box;}"
    ) in css


def test_filters_and_results_actions_use_fixed_compact_slots():
    css = _compact_css()
    filter_source = _function_source("_render_filter_bar", "_unique_values")
    view_source = _function_source("_render_ibd_review_view", "_render_mode_scope_controls")

    assert 'f"Filters · {summary}"' in filter_source
    assert "                                      " not in filter_source
    assert ".st-key-filters_headerbutton::after{content:\"⌄\"" in css
    assert "right:14px" in css
    assert "justify-content:flex-start!important" in css

    assert "st.columns([1,0.34]" in re.sub(r"\s+", "", view_source)
    assert "st.columns([1.35,1]" in re.sub(r"\s+", "", view_source)
    assert "min-width:160px" in css
    assert "max-width:180px" in css
    assert "min-width:110px" in css
    assert "max-width:146px" in css


def test_flow_card_main_and_info_buttons_have_separate_fixed_layout_contracts():
    css = _compact_css()

    assert (
        '.st-key-status_cardsdiv[class*="st-key-flow_card_"]'
        '>div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger))'
        'button[kind]{width:100%!important'
    ) in css
    assert (
        'div[class*="st-key-flow_card_"]>div[data-testid="stElementContainer"]:has(.flow-info-trigger)'
        '{position:absolute;top:7px;right:7px;z-index:4;width:16px!important;height:16px!important;}'
    ) in css
    assert (
        '.flow-info-trigger{appearance:none;display:flex;align-items:center;justify-content:center;'
        'width:16px!important;min-width:16px!important;max-width:16px!important;'
        'height:16px!important;min-height:16px!important;max-height:16px!important'
    ) in css
    assert '.flow-tooltip-surface{position:fixed;z-index:1000000;max-width:min(320px,calc(100vw-24px));' in css
    assert "white-space:pre-line!important" in css
    assert 'stPopoverButton' not in css
    for card in [
        "became_actionable",
        "left_actionable",
        "other_changes",
        "new",
        "carry",
        "reconfirmed",
        "actionable",
        "unconfirmed",
        "below_trigger",
        "extended",
    ]:
        assert (
            f'.st-key-flow_card_{card}>div[data-testid="stElementContainer"]'
            ':not(:has(.flow-info-trigger))button[kind]p::before'
        ) in css
