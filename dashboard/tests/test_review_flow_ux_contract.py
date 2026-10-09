from __future__ import annotations

from pathlib import Path
import subprocess


DASHBOARD = Path(__file__).resolve().parents[1]
APP = (DASHBOARD / "app.js").read_text(encoding="utf-8")
INDEX = (DASHBOARD / "index.html").read_text(encoding="utf-8")
TABLE = (DASHBOARD / "table_enhancements.js").read_text(encoding="utf-8")
INTERACTION = (DASHBOARD / "interaction_runtime.js").read_text(encoding="utf-8")


def test_review_flow_is_owned_by_existing_layers_without_override_runtime() -> None:
    assert "review_flow_runtime.js" not in INDEX
    assert "review_flow_runtime" not in APP
    assert "review_flow_runtime" not in TABLE
    assert "review_flow_runtime" not in INTERACTION


def test_watch_stage_is_visually_separate_from_entry_status() -> None:
    assert 'const WATCH_STAGE = "NEAR_BREAKOUT"' in APP
    assert 'const ENTRY_STATUS_ORDER = ["ACTIONABLE", "UNCONFIRMED", "BELOW_TRIGGER", "EXTENDED"]' in APP
    assert 'const REVIEW_STATE_ORDER = [WATCH_STAGE, ...ENTRY_STATUS_ORDER]' in APP
    assert "review-stage-status" in APP
    assert "Watch Stage" in APP
    assert "pre-signal · current candidates" in APP
    assert "Entry Status" in APP
    assert "active signals" in APP
    assert '["ibd_entry_status", "Stage / Status"]' in APP


def test_near_breakout_uses_watch_reference_semantics_without_fake_change() -> None:
    assert 'if (isNearBreakout(row)) return "";' in APP
    assert 'return text(row.review_change_label, "");' in APP
    assert 'return text(row.review_buy_point_change_label, "");' in APP
    assert 'const referenceKey = near ? "Watch Trigger" : "Buy Point";' in APP
    assert 'class="selected-reference-primary">${referenceKey} ${fmt(reviewReferencePrice(row))}' in APP
    styles = (DASHBOARD / "styles.css").read_text(encoding="utf-8")
    assert '.selected-reference-primary { color: #f2f5f9; font-size: 12px; font-weight: 800; }' in styles
    assert 'if (field === "current_vs_ibd_candidate_pct") return esc(fmt(reviewDistance(row), "pct"));' in APP
    assert "signal transition labels do not apply yet" in APP
    assert '["current_vs_ibd_candidate_pct", "Vs Reference"]' in APP


def test_setup_displays_single_overridden_signal_only_when_present() -> None:
    assert "function setupHtml(row)" in APP
    assert 'text(row.overridden_signal_source, "")' in APP
    assert "if (!overridden) return esc(primary);" in APP
    assert '<br><small>↳ ${esc(signalSourceLabel(overridden))}</small>' in APP
    assert 'if (field === "ibd_candidate_rule") return setupHtml(row);' in APP


def test_selected_overview_keeps_quality_facts_without_detail_panel() -> None:
    assert 'class="selected-strip"' in APP
    assert "has-pullback" not in APP
    assert 'class="selected-cell selected-pullback"' in APP
    assert "pullbackDry !== null" in APP
    assert 'Selected Overview' in APP
    for field in (
        "row.industry",
        "row.eps_yoy_growth",
        "row.base_depth_pct",
        "row.base_duration_weeks",
        "row.ceiling",
        "row.ceiling_date",
        "row.breakout_date",
        "row.dist_to_52w_high_pct",
        "row.pullback_pct",
        "row.pullback_duration_weeks",
        "row.pullback_peak_date",
        "row.pullback_peak_price",
        "row.pullback_v_is_dry",
        "row.rs_1m_percentile",
        "row.rs_3m_percentile",
        "row.rs_6m_percentile",
    ):
        assert field in APP
    assert 'Ceiling${ceiling === null ? "" : ` ${fmt(ceiling)}`}' in APP
    assert 'BO ${esc(breakoutDate)}' in APP
    assert "pullbackPeakDate = dateText(row.pullback_peak_date)" in APP
    assert "pullbackPeakPrice = num(row.pullback_peak_price)" in APP
    assert 'Start ${esc(pullbackPeakDate || "—")}' in APP
    assert 'pullback_start_date' not in APP
    assert "detailOpen" not in APP
    assert 'data-action="detail"' not in APP
    assert "function detailHtml(row)" not in APP


def test_selected_overview_never_renders_nat_as_a_date() -> None:
    assert '["nat", "nan", "none", "<na>", "null"].includes(out.toLowerCase())' in APP


def test_entry_volume_filters_signals_without_hiding_watch_candidates() -> None:
    assert "if (isNearBreakout(row)) return true;" in APP
    assert 'bounds(rows.filter(isSignalActive), "ibd_entry_volume_ratio", 0, 1)' in APP
    assert "Signal stage only · Watch candidates stay visible as N/A" in APP
    assert 'bounds(rows.filter(isSignalActive), "ibd_entry_volume_ratio")' in TABLE


def test_scope_switch_changes_population_without_resetting_review_intent() -> None:
    scope_block = APP.split('} else if (action === "scope") {', 1)[1].split('} else if (action === "quick") {', 1)[0]
    assert "state.scope = element.dataset.value;" in scope_block
    assert 'state.change = "ALL"' not in scope_block
    assert 'state.newBuyPoint = "ALL"' not in scope_block
    assert 'state.status = "ALL"' not in scope_block
    assert 'state.scope === "CHANGES" && state.status === WATCH_STAGE' in APP
    assert ') ? "ALL" : state.status;' in APP


def test_manual_sort_is_context_scoped_and_has_explicit_default_order() -> None:
    assert "const sortStates = new Map();" in TABLE
    assert 'return `${pressedValue("period") || "WEEKEND"}:${pressedValue("scope") || "ALL_SIGNALS"}`;' in TABLE
    assert "Default order" in TABLE
    assert "restoreDefaultOrder(shell)" in TABLE
    assert "slot.replaceChildren()" not in TABLE
    assert "if (button) {" in TABLE
    assert "button.disabled = mobileReviewLocked(shell);" in TABLE
    assert "sortState" not in INTERACTION
    assert "applyRememberedSort" not in INTERACTION


def test_dynamic_bounds_preserve_active_threshold_without_synthetic_changes() -> None:
    assert "low = Math.min(low, previous);" in TABLE
    assert "high = Math.max(high, previous);" in TABLE
    assert 'dispatchEvent(new Event("change"' not in TABLE
    assert "Math.min(high, Math.max(low, previous))" not in TABLE


def test_review_flow_has_responsive_stage_status_layout() -> None:
    assert ".review-stage-status" in INDEX
    assert ".review-watch-grid.status-grid" in INDEX
    assert ".review-entry-grid.status-grid" in INDEX
    assert "@media (width <= 760px)" in INDEX


def test_desktop_period_is_pinned_right_and_result_tools_match_mobile() -> None:
    styles = (DASHBOARD / "styles.css").read_text(encoding="utf-8")
    desktop = styles.split("@media (width > 760px)", 1)[1].split("@media (width <= 760px)", 1)[0]
    assert ".queue-heading > .scope-block { grid-column: 2; grid-row: 1; }" in desktop
    assert ".queue-heading > .period-block { grid-column: 3; grid-row: 1; justify-self: end; }" in desktop
    assert ".filters-wrap:not(.filters-expanded) { display: none; }" in desktop
    assert "grid-template-columns: minmax(0, 1fr) auto 34px 34px;" in desktop
    assert ".copy-button," in desktop
    assert ".mobile-filter-button {" in desktop
    assert "display: inline-flex;" in desktop
    assert ".review-default-sort {" in desktop
    assert 'content: "Reset";' in desktop
    assert '.mobile-filter-button[data-count]:not([data-count=""])::after' in desktop


def test_mobile_review_demotes_static_scope_and_low_frequency_tools() -> None:
    styles = (DASHBOARD / "styles.css").read_text(encoding="utf-8")
    assert "scope-static-block" in APP
    assert "scope-switch" in APP
    assert "mobile-filter-button" in APP
    assert 'class="results-order-slot"' in APP
    toolbar = APP.split('class="results-toolbar"', 1)[1].split("</div>", 5)
    toolbar_text = "".join(toolbar)
    assert toolbar_text.index('class="results-order-slot"') < toolbar_text.index('class="copy-button"')
    assert toolbar_text.index('class="copy-button"') < toolbar_text.index('class="mobile-filter-button"')
    assert ".scope-static-block { position: absolute;" in styles
    assert ".filters-wrap:not(.filters-expanded) { display: none; }" in styles
    assert ".mobile-filter-button" in styles
    assert ".copy-button::before" in styles
    assert ".mobile-filter-button::before" in styles
    assert 'data-count="${activeFilters || ""}"' in APP
    assert ".results-section > .selected-strip { order: 1; }" in styles
    assert ".results-section > .results-toolbar { order: 2; }" in styles


def test_mobile_selected_overview_prioritizes_dense_metrics_and_structure_context() -> None:
    styles = (DASHBOARD / "styles.css").read_text(encoding="utf-8")
    for token in (
        "selected-eps",
        "selected-high",
        "selected-base",
        "selected-pullback",
        "selected-rs",
        "baseMobileLines",
        "pullbackMobileLines",
        "selected-structure-mobile",
    ):
        assert token in APP
    assert '"eps eps high high rs rs"' in styles
    assert '"base base base pullback pullback pullback"' in styles
    assert ".selected-eps { grid-area: eps; }" in styles
    assert ".selected-high { grid-area: high; }" in styles
    assert ".selected-rs { grid-area: rs; }" in styles
    assert ".selected-base { grid-area: base; }" in styles
    assert ".selected-pullback { grid-area: pullback; }" in styles
    assert ".selected-strip.empty { min-height: 52px; grid-template-columns: 1fr; grid-template-areas: none; }" in styles
    assert ".selected-structure-compact { display: none; }" in styles


def test_default_order_uses_explicit_toolbar_slot() -> None:
    styles = (DASHBOARD / "styles.css").read_text(encoding="utf-8")
    assert 'toolbar?.querySelector(".results-order-slot")' in TABLE
    assert "toolbar?.lastElementChild" not in TABLE
    assert 'button.setAttribute("aria-label", "Default order")' in TABLE
    assert "grid-template-columns: minmax(0, 1fr) 50px 34px 34px;" in styles
    assert ".results-order-slot { grid-column: auto; width: 50px; min-width: 50px;" in styles
    assert ".results-order-slot:empty { display: none; }" not in styles
    assert '.review-default-sort::before {' in styles
    assert 'content: "Reset";' in styles
    assert '.review-default-sort { grid-column: 1 / -1; }' not in INDEX
    assert ".review-default-sort {" not in INDEX


def test_breakout_quality_sort_uses_strength_descending_semantics_and_info_icon() -> None:
    assert 'return index < 0 ? 0 : QUALITY_ORDER.length - index;' in TABLE
    handler = TABLE.split("function onHeaderSort", 1)[1].split("function qualityTooltipHtml", 1)[0]
    assert 'if (field === "ibd_breakout_quality")' in handler
    assert 'previous.direction === "desc"' in handler
    assert '{ field, direction: "desc" }' in handler
    assert '{ field, direction: "asc" }' in handler
    assert 'info.className = "table-info-button";' in TABLE
    assert 'controls.appendChild(info);' in TABLE
    assert 'button.appendChild(info);' not in TABLE
    assert 'qualityTooltipAnchor === info' in TABLE
    assert 'setTextIfChanged(icon, isActive ? (sortState.direction === "asc" ? "▲" : "▼") : "");' in TABLE


def test_rs_and_quality_info_controls_are_separate_from_sort_hit_zones() -> None:
    styles = (DASHBOARD / "styles.css").read_text(encoding="utf-8")
    rs = (DASHBOARD / "rs_runtime.js").read_text(encoding="utf-8")
    index = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    assert 'controls.className = "table-header-control";' in TABLE
    assert 'className = "table-sort-button";' in TABLE
    assert 'className = "table-info-button rs-info-button";' in rs
    assert 'controls.appendChild(button);' in rs
    assert '.table-header-control.with-info { gap: 9px; }' in styles
    assert 'width: 32px;' in styles
    assert 'width: 13px;' in styles
    assert 'padding-right: 46px' not in index
    assert '.rs-info-button::after' not in index


def test_rs_sort_cycles_desc_asc_then_restores_default_order() -> None:
    handler = TABLE.split("function onHeaderSort", 1)[1].split("function qualityTooltipHtml", 1)[0]
    assert 'if (field === "rs_percentile")' in handler
    assert 'previous.direction === "desc"' in handler
    assert 'previous.direction === "asc"' in handler
    assert '{ field, direction: "desc" }' in handler
    assert '{ field, direction: "asc" }' in handler
    assert "restoreDefaultOrder(shell);" in handler


def test_dashboard_javascript_is_syntactically_valid() -> None:
    for script in ("app.js", "table_enhancements.js", "interaction_runtime.js"):
        subprocess.run(
            ["node", "--check", str(DASHBOARD / script)],
            check=True,
            capture_output=True,
            text=True,
        )
