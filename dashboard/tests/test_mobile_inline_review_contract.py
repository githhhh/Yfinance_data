from pathlib import Path


DASHBOARD = Path(__file__).resolve().parents[1]
APP = (DASHBOARD / "app.js").read_text(encoding="utf-8")
TABLE = (DASHBOARD / "table_enhancements.js").read_text(encoding="utf-8")
RS = (DASHBOARD / "rs_runtime.js").read_text(encoding="utf-8")
STYLES = (DASHBOARD / "styles.css").read_text(encoding="utf-8")


def test_mobile_results_contract_is_four_columns_without_changing_queue_flow() -> None:
    for label in ("Review Queue", "What Changed", "Signal Source", "Watch Stage", "Entry Status"):
        assert label in APP

    assert 'columns.map(([field, label]) => `<th data-field="${esc(field)}">' in APP
    assert 'columns.map(([field]) => `<td data-field="${esc(field)}"' in APP
    assert '.review-table [data-field="code"] { grid-column: 1; }' in STYLES
    assert '.review-table [data-field="rs_percentile"] { grid-column: 2; }' in STYLES
    assert '.review-table [data-field="ibd_entry_status"] { grid-column: 3; }' in STYLES
    assert '.review-table [data-field="current_vs_ibd_candidate_pct"] { grid-column: 4;' in STYLES
    assert '.review-table tbody tr[data-code] td[data-field="code"],' in STYLES
    assert 'grid-row: 1;' in STYLES
    assert 'content: "STATUS";' in STYLES
    assert 'content: "VS REF";' in STYLES
    assert "overflow-x: hidden;" in STYLES
    assert ".mobile-detail-row { display: none; }" in STYLES
    assert ".results-section > .selected-strip { display: none; }" in STYLES


def test_mobile_row_click_toggles_single_inline_review() -> None:
    assert "function renderMobileDetail(currentRows)" in APP
    assert 'shell?.querySelector(".mobile-detail-row")?.remove();' in APP
    assert 'state.selected[state.period] = String(state.selected[state.period]) === String(code) ? null : code;' in APP
    assert 'detail.className = "mobile-detail-row";' in APP
    assert 'mainRow.insertAdjacentElement("afterend", detail);' in APP
    assert 'shell.dataset.reviewExpanded = "true";' in APP


def test_inline_review_preserves_decision_fields_and_setup_hierarchy() -> None:
    for token in (
        "Watch Trigger",
        "Buy Point",
        "Latest",
        "Vs Ref",
        "Setup",
        "Geometry",
        "Entry",
        "EPS YoY",
        "To 52W High",
        "Base",
        "Pullback",
        "ibd_entry_close_position",
        "ibd_entry_breakout_range_ratio",
        "ibd_entry_volume_ratio",
        "volume_ratio",
        "ceiling_date",
        "breakout_date",
        "pullback_peak_date",
        "pullback_peak_price",
    ):
        assert token in APP
    assert "setupHtml(row)" in APP
    assert "overridden_signal_source" in APP
    assert "data-mobile-rs-reference" in APP
    assert "data-mobile-rs-reference" in RS
    assert "1M " in RS and "3M " in RS and "6M " in RS


def test_mobile_sort_is_frozen_while_inline_review_is_expanded() -> None:
    assert "function mobileReviewLocked(shell)" in TABLE
    assert 'shell?.dataset.reviewExpanded === "true"' in TABLE
    assert "if (mobileReviewLocked(shell)) return;" in TABLE
    assert "button.disabled = locked;" in TABLE
    assert "button.disabled = mobileReviewLocked(shell);" in TABLE
    assert "Collapse the expanded row to sort" in TABLE
    assert "Collapse the expanded row to restore default order" in TABLE
    assert 'app.addEventListener("mobile-review-lock-change"' in TABLE
    assert 'app.dispatchEvent(new CustomEvent("mobile-review-lock-change"))' in APP
    assert ".review-table th.sort-locked > button" in STYLES
    assert ".review-default-sort:disabled" in STYLES
    assert 'window.matchMedia?.("(max-width: 760px)")?.matches' in TABLE
    assert "if (mobileReviewLocked(shell)) return;\n    const body" in TABLE
    assert "function sortTable(shell, field, direction) {\n    const headers" in TABLE
