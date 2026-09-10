from pathlib import Path
import subprocess


DASHBOARD = Path(__file__).resolve().parents[1]
APP = (DASHBOARD / "app.js").read_text(encoding="utf-8")
INDEX = (DASHBOARD / "index.html").read_text(encoding="utf-8")
TABLE = (DASHBOARD / "table_enhancements.js").read_text(encoding="utf-8")
INTERACTION = (DASHBOARD / "interaction_runtime.js").read_text(encoding="utf-8")
BUILD = (DASHBOARD / "build_static.py").read_text(encoding="utf-8")


def test_interaction_runtime_has_valid_javascript():
    subprocess.run(
        ["node", "--check", str(DASHBOARD / "interaction_runtime.js")],
        check=True,
        capture_output=True,
        text=True,
    )


def test_interaction_runtime_is_published_by_static_build():
    assert '"interaction_runtime.js"' in BUILD
    assert 'src="./interaction_runtime.js"' in INDEX


def test_row_selection_updates_in_place_and_keeps_detail_state():
    needle = 'app.querySelectorAll("tbody tr[data-code]").forEach((row) => {'
    row_handler = APP.rsplit(needle, 1)[1].split("bindSelectedEvents(currentRows);", 1)[0]
    assert "renderSelection(currentRows, { focusTable: true })" in row_handler
    assert "state.detailOpen = false" not in row_handler
    assert "render();" not in row_handler


def test_selection_restores_focus_and_horizontal_scroll():
    selection = APP.split("function renderSelection", 1)[1].split("function tableHtml", 1)[0]
    assert "const scrollLeft = shell.scrollLeft" in selection
    assert "shell.scrollLeft = scrollLeft" in selection
    assert 'shell?.focus({ preventScroll: true })' in selection


def test_period_context_is_cached_per_period():
    assert "periodContexts" in APP
    assert "savePeriodContext();" in APP
    assert "restorePeriodContext(period);" in APP
    reset = APP.split("function resetPeriodState", 1)[1].split("function freshness", 1)[0]
    assert "resetAdvanced();" not in reset


def test_manual_sort_is_reapplied_synchronously_after_app_renders():
    assert "rememberManualSort" in INTERACTION
    assert "applyRememberedSort" in INTERACTION
    assert 'appObserver.observe(app, { childList: true, subtree: true })' in INTERACTION
    assert 'event.target.closest?.("thead th > button")' in INTERACTION
    assert 'event.target.closest?.("[data-rs-info], [data-quality-info]")' in INTERACTION


def test_manual_sort_keyboard_review_preserves_horizontal_scroll():
    assert 'document.addEventListener("keydown"' in INTERACTION
    assert 'event.stopImmediatePropagation();' in INTERACTION
    assert "const scrollLeft = shell.scrollLeft" in INTERACTION
    assert "currentShell.scrollLeft = scrollLeft" in INTERACTION
    assert 'currentShell.focus({ preventScroll: true })' in INTERACTION


def test_range_enhancement_keeps_authoritative_context_bounds_semantics():
    assert "if (low === high) input.disabled = true" in TABLE
    assert "const singleton" not in TABLE
    assert "const originalLow = Number(input.min)" not in TABLE
    assert "const originalHigh = Number(input.max)" not in TABLE
    assert 'input.removeAttribute("data-dynamic-bounds")' in INTERACTION
    assert 'input.dataset.rangeBootstrap = "true"' in INTERACTION
    assert "prepareRangeInputs();" in INTERACTION


def test_rs_info_has_large_touch_target_and_non_clickthrough_backdrop():
    assert '.review-table .rs-info-button::after' in INDEX
    assert 'inset: -13px' in INDEX
    assert "rs-popover-backdrop" in INTERACTION
    assert 'event.stopPropagation();' in INTERACTION
    assert 'document.querySelector("[data-rs-info]")?.click()' in INTERACTION
    assert "Keep the backdrop alive through the pointer sequence" in INTERACTION
