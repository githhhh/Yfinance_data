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
        "padding:8px24px30px",
        "min-height:64px",
        "grid-template-columns:minmax(0,1fr)276px268px",
        "min-height:48px",
        "grid-template-columns:repeat(4,minmax(0,1fr))",
        "height:70px",
        "height:45px",
        "min-height:48px",
        "height:auto",
        "min-height:72px",
        "grid-template-columns:194pxrepeat(4,minmax(0,1fr))",
        "font-size:29px",
        "font-weight:800",
        "margin-top:12px",
        "padding-top:8px",
    ]:
        assert declaration in css


def test_reference_breakpoints_focus_and_reduced_motion_are_explicit():
    css = _compact_css()
    for rule in [
        "@media(width<=1120px)",
        "@media(width<1280px)",
        "@media(width<=760px)",
        "@media(width<=480px)",
        "@media(prefers-reduced-motion:reduce)",
        ":focus-visible",
    ]:
        assert rule in css


def test_review_context_has_semantic_responsive_groups_and_stable_clear_slot():
    source = _function_source("_render_review_context", "_render_status_queue")

    assert "rows = [" not in source
    assert 'key="quick_context_row"' in source
    assert 'key="quick_change_group"' in source
    assert 'key="quick_origin_group"' in source
    assert 'st.caption("WHAT CHANGED")' in source
    assert 'st.caption("SIGNAL SOURCE")' in source
    assert source.count("_render_quick_group(") == 2
    assert 'key="btn_clear_quick"' in source
    assert 'key="quick_clear_slot"' in source
    assert "if quick_filter_count:" in source
    assert "Weekend Baseline" in source
    assert "No valid complete-week baseline" in source


def test_midweek_unavailable_is_only_explained_by_the_disabled_period_control():
    source = _function_source("_render_review_context", "_render_status_queue")
    css = _compact_css()

    assert 'class="weekend-context-bar weekend-context-bar--unavailable"' not in source
    assert 'class="midweek-unavailable-icon"' not in source
    assert "Midweek unavailable" not in source
    assert ".weekend-context-bar--unavailable" not in css
    assert ".midweek-unavailable-icon" not in css
    assert ".st-key-review_mode_controlsbutton:disabled" in css


def test_queue_controls_have_stable_heading_and_segmented_group_slots():
    source = _function_source("_render_mode_scope_controls", "df_active_count_for_state")

    for key in [
        "review_queue_heading",
        "review_mode_controls",
        "review_scope_controls",
    ]:
        assert f'key="{key}"' in source
    assert 'st.caption("PERIOD")' in source
    assert 'st.caption("SCOPE")' in source
    assert 'class="weekend-scope-static"' in source
    assert 'disabled=not has_comparison' not in source
    assert "MIDWEEK_UNAVAILABLE_HELP" in source


def test_status_labels_do_not_duplicate_selected_feedback_with_checkmarks():
    source = _function_source("_render_status_queue", "_active_filter_count")

    for emoji in ["🟢", "🟡", "🔴", "🔵", "⚪"]:
        assert emoji not in source
    assert 'prefix = "✓ " if is_active else "  "' not in source
    assert 'btn_label = f"{display_name} · {count}' in source


def test_quick_symbols_counts_and_status_orbs_are_styled():
    css = _compact_css()

    for declaration in [
        "grid-template-columns:16pxminmax(0,1fr)30px",
        "button[kind]pstrong",
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
        'class="data-badge data-badge--{badge_tone}"',
        '"Data Fresh"',
        '"Data Aging"',
        '"Data Stale"',
        '"Data Loaded"',
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


def test_breakout_pool_title_uses_non_heading_markup_to_avoid_streamlit_anchor():
    source = _function_source("_render_header_bar", "_render_flow_rules_dialog")

    assert '<div class="dashboard-title" role="heading" aria-level="1">' in source
    assert '<h3 class="dashboard-title">' not in source


def test_mobile_header_results_and_actions_have_explicit_stack_contracts():
    css = _compact_css()

    assert '.st-key-dashboard_header>div[data-testid="stHorizontalBlock"]' in css
    assert '.st-key-results_toolbar>div[data-testid="stHorizontalBlock"]' in css
    assert ".st-key-results_actions{width:100%;display:flex;justify-content:flex-start;}" in css
    assert '.st-key-review_queue_heading>div[data-testid="stHorizontalBlock"]' in css
    assert (
        '.st-key-review_queue_heading>div[data-testid="stHorizontalBlock"]'
        '>div[data-testid="stColumn"]{width:100%!important;min-width:0!important;flex:none!important;}'
    ) in css
    assert "grid-template-columns:minmax(0,1fr)minmax(0,1fr)74px" in css
    assert "grid-template-columns:repeat(3,minmax(0,1fr))" in css
    assert '.st-key-review_context_slotdiv[class*="st-key-flow_card_"]{position:relative;min-width:0;}' in css
    for selector in [
        '.st-key-dashboard_header>div[data-testid="stHorizontalBlock"]>div[data-testid="stColumn"]',
        '.st-key-results_toolbar>div[data-testid="stHorizontalBlock"]>div[data-testid="stColumn"]',
    ]:
        assert selector + '{width:100%!important;min-width:0!important;flex:none!important;}' in css
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
        "{border:1pxsolid#303a46;border-radius:7px;padding:3px10px;"
        "background:#11171e;box-sizing:border-box;}"
    ) in css


def test_filters_and_results_actions_use_fixed_compact_slots():
    css = _compact_css()
    filter_source = _function_source("_render_filter_bar", "_render_ibd_selected_row_detail")
    view_source = _function_source("_render_ibd_review_view", "_render_mode_scope_controls")

    assert 'f"More Filters · {summary}"' in filter_source
    assert "                                      " not in filter_source
    assert 'class="filters-state-marker"' in filter_source
    assert 'data-expanded="{str(state["filters_expanded"]).lower()}"' in filter_source
    assert 'div[class*="st-key-btn_filters_toggle"]button::after{content:"⌄"' in css
    assert (
        '.st-key-filters_header:has(.filters-state-marker[data-expanded="true"])'
        'div[class*="st-key-btn_filters_toggle"]button::after{content:"⌃"'
    ) in css
    assert (
        '.st-key-filters_header:has(.filters-state-marker[data-expanded="true"])'
        'div[class*="st-key-btn_filters_reset"]button::after'
    ) not in css
    assert "right:8px" in css
    assert "justify-content:flex-start!important" in css

    assert "st.columns([0.24,0.14,1]" in re.sub(r"\s+", "", view_source)
    assert "review_sort_" not in view_source
    assert "grid-template-columns:max-content154pxminmax(0,1fr)" in css
    assert "justify-content:flex-start" in css
    assert "height:36px!important" in css
    assert (
        '.st-key-results_toolbardiv[data-testid="stMarkdownContainer"]:has(.results-summary)'
        '{margin-bottom:0!important;}'
    ) in css
    assert "min-width:170px" in css
    assert "max-width:190px" in css


def test_filters_header_reset_and_controls_follow_final_flow_contract():
    css = _compact_css()
    filter_source = _function_source("_render_filter_bar", "_render_ibd_selected_row_detail")

    assert "ifactive_count>0:" in re.sub(r"\s+", "", filter_source)
    assert 'key="btn_filters_reset"' in filter_source
    assert 'key="active_filter_chips"' not in filter_source
    assert "btn_filter_chip_" not in filter_source
    assert "SETUP_FILTER_OPTIONS" in filter_source
    assert "st.popover(" in filter_source
    assert "st.radio(" in filter_source
    assert "st.text_input(" not in filter_source
    assert 'class="filter-slider-heading"' in filter_source
    assert 'filter-slider-heading--range' in filter_source
    assert 'class="filter-volume-value' not in filter_source
    assert 'key="filter_entry_volume"' in filter_source
    assert 'key="filter_weekly_volume"' in filter_source
    assert ".st-key-active_filter_chips" not in css
    assert "padding:12px12px16px" in css
    assert '[data-testid="stSliderTickBar"]{display:none!important;}' in css
    assert ".filter-volume-value" not in css


def test_ibd_code_details_are_focusable_unclipped_and_escape_closable():
    css = _compact_css()
    detail_source = _function_source("_render_ibd_selected_row_detail", "_render_selected_row_detail")
    tooltip_source = (DASHBOARD_DIR / "review_tooltip.py").read_text(encoding="utf-8")

    assert '<details class="code-detail"' in detail_source
    assert '<summary class="code-hover-trigger"' in detail_source
    assert "position:fixed" in css
    assert "z-index:999999" in css
    assert ".st-key-ibd_selected_row.code-hover-trigger" in css
    assert "border-bottom:1pxdotted#56a8ff" in css
    assert ".st-key-ibd_selected_row.code-detail:hover>.code-hover-popup" in css
    assert '.closest(".st-key-ibd_selected_row .code-detail")' in tooltip_source
    assert 'event.key === "Escape"' in tooltip_source
    assert "details.open = false" in tooltip_source
    assert 'details.dataset.escapeDismissed = "true"' in tooltip_source
    assert 'trigger.focus({preventScroll: true})' in tooltip_source
    assert "trigger.blur()" not in tooltip_source
    assert '.code-detail:not([data-escape-dismissed="true"])' in css


def test_review_views_wire_session_visits_and_current_result_positions():
    ibd_source = _function_source("_render_ibd_review_view", "_render_mode_scope_controls")
    c_rank_source = _function_source("_render_c_rank_reference_view", "_render_copy_codes_control")
    detail_source = _function_source("_render_selected_row_detail", "_render_c_rank_reference_view")

    assert 'visited_codes=_visited_codes(state["mode"])' in ibd_source
    assert '_store_review_visit(state["mode"], selected_code)' in ibd_source
    assert 'visited_codes=_visited_codes("C_RANK")' in c_rank_source
    assert '_store_review_visit("C_RANK", selected_code)' in c_rank_source
    assert "build_review_position(filtered_df, selected_code)" in detail_source
    assert "Select a row · Use ↑↓ to review" in detail_source


def test_filters_disclosure_state_is_synchronized_to_native_button_aria():
    tooltip_source = (DASHBOARD_DIR / "review_tooltip.py").read_text(encoding="utf-8")

    assert 'querySelector(".filters-state-marker")' in tooltip_source
    assert 'button.setAttribute("aria-expanded", marker.dataset.expanded)' in tooltip_source
    assert "MutationObserver" in tooltip_source


def test_tooltip_hover_is_delayed_but_focus_and_click_remain_immediate():
    tooltip_source = (DASHBOARD_DIR / "review_tooltip.py").read_text(encoding="utf-8")

    assert "const hoverDelayMs = 275" in tooltip_source
    assert "hoverTimer = parentWindow.setTimeout" in tooltip_source
    assert "parentWindow.clearTimeout(hoverTimer)" in tooltip_source
    assert "const onFocusIn" in tooltip_source
    assert "showTooltip(card" in tooltip_source
    assert "const onClick" in tooltip_source
    assert "if (activeCard) hideTooltip();\n        else cancelHoverTimer();" in tooltip_source
    assert "activeCard || hoverTimer !== null" in tooltip_source


def test_flow_tooltip_is_trigger_only_and_positioned_outside_the_card():
    tooltip_source = (DASHBOARD_DIR / "review_tooltip.py").read_text(encoding="utf-8")

    assert 'target.closest(".flow-info-trigger")' in tooltip_source
    assert "const trigger = triggerFor(event.target)" in tooltip_source
    assert "const cardBox = card.getBoundingClientRect()" in tooltip_source
    assert "let top = cardBox.bottom + 8" in tooltip_source
    assert "cardBox.top - tooltipBox.height - 8" in tooltip_source
    assert "anchorBox.bottom + 8" not in tooltip_source


def test_results_actions_reserve_copy_only_and_never_scroll_horizontally():
    css = _compact_css()

    assert "overflow-x:auto" not in css.split(".st-key-results_actions", 1)[1].split("}", 1)[0]
    assert '.st-key-results_toolbardiv[data-baseweb="select"]>div' not in css


def test_selected_detail_auto_height_never_clips_the_table_boundary():
    css = _compact_css()
    selected_rule = css.split(".st-key-selected_row.selected-strip{", 1)[1].split("}", 1)[0]

    assert "height:auto" in selected_rule
    assert "min-height:72px" in selected_rule
    assert "overflow:visible" in selected_rule
    assert "position:absolute" not in selected_rule
    assert (
        '.st-key-selected_rowdiv[data-testid="stMarkdownContainer"]:has(.selected-strip)'
        '{margin-bottom:0!important;}'
    ) in css


def test_public_buy_point_terminology_is_used_in_primary_ui():
    assert "Candidate Price" not in APP_SOURCE
    assert "Current vs Candidate" not in APP_SOURCE
    assert ">Buy Point<" in APP_SOURCE
    assert ">Vs Buy Point<" in APP_SOURCE


def test_hidden_filters_state_marker_does_not_add_vertical_layout_space():
    css = _compact_css()

    assert (
        '.st-key-filters_header>div[data-testid="stVerticalBlock"]'
        '>div[data-testid="stElementContainer"]:has(.filters-state-marker)'
        '{display:none;}'
    ) in css


def test_status_orb_and_two_line_text_use_independent_grid_regions():
    css = _compact_css()

    assert "display:grid!important" in css
    assert "grid-template-columns:19pxminmax(0,1fr)" in css
    assert "column-gap:10px" in css
    assert "grid-column:1" in css


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
    ]:
        assert (
            f'.st-key-flow_card_{card}>div[data-testid="stElementContainer"]'
            ':not(:has(.flow-info-trigger))button[kind]pstrong:first-child'
        ) in css
    for card in ["actionable", "unconfirmed", "below_trigger", "extended"]:
        assert (
            f'.st-key-flow_card_{card}>div[data-testid="stElementContainer"]'
            ':not(:has(.flow-info-trigger))button[kind]p::before'
        ) in css
