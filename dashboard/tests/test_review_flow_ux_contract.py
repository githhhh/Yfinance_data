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
    assert 'return isNearBreakout(row) ? "" : text(row.review_change_label, "");' in APP
    assert 'const referenceKey = near ? "Watch Trigger" : "Buy Point";' in APP
    assert 'class="selected-reference-primary">${referenceKey} ${fmt(reviewReferencePrice(row))}' in APP
    styles = (DASHBOARD / "styles.css").read_text(encoding="utf-8")
    assert '.selected-reference-primary { color: #f2f5f9; font-size: 12px; font-weight: 800; }' in styles
    assert 'if (field === "current_vs_ibd_candidate_pct") return esc(fmt(reviewDistance(row), "pct"));' in APP
    assert "signal transition labels do not apply yet" in APP
    assert '["current_vs_ibd_candidate_pct", "Vs Reference"]' in APP


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
    assert "detailOpen" not in APP
    assert 'data-action="detail"' not in APP
    assert "function detailHtml(row)" not in APP
    assert 'Ceiling ${ceilingPrice === null ? "—" : fmt(ceilingPrice)}' in APP
    assert 'Start ${esc(baseStartDate || "—")} → Breakout ${esc(breakoutDate || "—")}' in APP
    assert 'Start ${esc(pullbackPeakDate || "—")}' in APP
    assert 'Peak ${pullbackPeakPrice === null ? "—" : fmt(pullbackPeakPrice)}' in APP


def test_entry_volume_filters_signals_without_hiding_watch_candidates() -> None:
    assert "if (isNearBreakout(row)) return true;" in APP
    assert 'bounds(rows.filter(isSignalActive), "ibd_entry_volume_ratio", 0, 1)' in APP
    assert "Signal stage only · Watch candidates stay visible as N/A" in APP
    assert 'bounds(rows.filter(isSignalActive), "ibd_entry_volume_ratio")' in TABLE


def test_scope_switch_changes_population_without_resetting_review_intent() -> None:
    scope_block = APP.split('} else if (action === "scope") {', 1)[1].split('} else if (action === "quick") {', 1)[0]
    assert "state.scope = element.dataset.value;" in scope_block
    assert 'state.change = "ALL"' not in scope_block
    assert 'state.origin = "ALL"' not in scope_block
    assert 'state.status = "ALL"' not in scope_block


def test_manual_sort_is_context_scoped_and_has_explicit_default_order() -> None:
    assert "const sortStates = new Map();" in TABLE
    assert 'return `${pressedValue("period") || "WEEKEND"}:${pressedValue("scope") || "ALL_SIGNALS"}`;' in TABLE
    assert "Default order" in TABLE
    assert "restoreDefaultOrder(shell)" in TABLE
    assert "slot.replaceChildren()" not in TABLE
    assert "if (button) return;" in TABLE
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


def test_dashboard_javascript_is_syntactically_valid() -> None:
    for script in ("app.js", "table_enhancements.js", "interaction_runtime.js"):
        subprocess.run(
            ["node", "--check", str(DASHBOARD / script)],
            check=True,
            capture_output=True,
            text=True,
        )
