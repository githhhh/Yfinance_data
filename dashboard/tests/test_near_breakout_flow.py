from __future__ import annotations

from pathlib import Path
import subprocess

import pandas as pd

from dashboard.build_static import PUBLIC_DASHBOARD_ROW_FIELDS, _records, _status_meta
from dashboard.services.bf_transition import ENTRY_STATUSES


DASHBOARD = Path(__file__).resolve().parents[1]
APP = (DASHBOARD / "app.js").read_text(encoding="utf-8")
BUILD_STATIC = (DASHBOARD / "build_static.py").read_text(encoding="utf-8")
INDEX = (DASHBOARD / "index.html").read_text(encoding="utf-8")
TABLE_RUNTIME = (DASHBOARD / "table_enhancements.js").read_text(encoding="utf-8")
INTERACTION_RUNTIME = (DASHBOARD / "interaction_runtime.js").read_text(encoding="utf-8")

WATCH_FIELDS = {
    "bf_watch_active",
    "bf_watch_type",
    "bf_watch_trigger_price",
    "bf_watch_distance_pct",
    "bf_watch_s_resistance",
    "bf_watch_s_distance_pct",
    "bf_watch_s_side",
    "bf_watch_m_resistance",
    "bf_watch_m_distance_pct",
    "bf_watch_m_side",
}


def test_setup_watch_fields_are_explicitly_public_without_broadening_payload() -> None:
    frame = pd.DataFrame(
        [
            {
                "code": "NEAR",
                "signal": False,
                "bf_watch_active": True,
                "bf_watch_type": "ma10_pullback",
                "bf_watch_trigger_price": 105.0,
                "bf_watch_distance_pct": -1.1,
                "bf_watch_s_resistance": None,
                "bf_watch_s_distance_pct": None,
                "bf_watch_s_side": "RIGHT",
                "bf_watch_m_resistance": 100.0,
                "bf_watch_m_distance_pct": -1.2,
                "bf_watch_m_side": "RIGHT",
                "future_private_field": "must-not-publish",
            }
        ]
    )

    row = _records(frame)[0]

    assert WATCH_FIELDS.issubset(PUBLIC_DASHBOARD_ROW_FIELDS)
    assert row["bf_watch_active"] is True
    assert row["bf_watch_type"] == "ma10_pullback"
    assert row["bf_watch_trigger_price"] == 105.0
    assert row["bf_watch_distance_pct"] == -1.1
    assert row["bf_watch_m_resistance"] == 100.0
    assert row["bf_watch_m_distance_pct"] == -1.2
    assert row["bf_watch_m_side"] == "RIGHT"
    assert "future_private_field" not in row


def test_near_breakout_is_a_ui_stage_not_an_ibd_entry_status() -> None:
    meta = _status_meta()

    assert "NEAR_BREAKOUT" in meta
    assert meta["NEAR_BREAKOUT"]["label"] == "NEAR BREAKOUT"
    assert meta["NEAR_BREAKOUT"]["subtitle"] == "Approaching Trigger"
    assert "NEAR_BREAKOUT" not in ENTRY_STATUSES
    assert 'const WATCH_STAGE = "NEAR_BREAKOUT"' in APP
    assert 'const ENTRY_STATUS_ORDER = ["ACTIONABLE", "UNCONFIRMED", "BELOW_TRIGGER", "EXTENDED"]' in APP
    assert 'return isNearBreakout(row) ? WATCH_STAGE : row.ibd_entry_status;' in APP
    assert 'const status = displayStatus(row);' in APP
    assert '["ibd_entry_status", "Stage / Status"]' in APP


def test_near_breakout_prefers_v2_generic_watch_contract_with_v1_pivot_fallback() -> None:
    assert "function isNearBreakout(row)" in APP
    assert "bool(row.bf_watch_active)" in APP
    assert "!isSignalActive(row)" in APP
    assert 'text(row.bf_watch_type, "pivot")' in APP
    assert "function nearBreakoutTarget(row)" in APP
    assert "bf_watch_trigger_price" in APP
    assert "bf_watch_distance_pct" in APP
    assert "bf_watch_s_resistance" in APP
    assert "bf_watch_m_resistance" in APP
    assert "function nearBreakoutDistance(row)" in APP


def test_v2_watch_types_are_available_to_setup_filter() -> None:
    assert '"ma10_pullback"' in BUILD_STATIC
    assert '"ceiling_pullback"' in BUILD_STATIC
    assert '"three_weeks_tight"' in BUILD_STATIC
    assert '"pivot"' in BUILD_STATIC
    assert 'ma10_pullback: "MA10 Pullback"' in APP


def test_midweek_review_now_scope_keeps_current_near_breakout_rows() -> None:
    assert 'text(row.review_change_group, "UNCHANGED") !== "UNCHANGED" || isNearBreakout(row)' in APP
    assert "Review Now" in APP
    assert "All Review" in APP


def test_near_breakout_overview_keeps_watch_reference_and_source() -> None:
    assert "function nearBreakoutDetailHtml(row)" not in APP
    assert 'const referenceKey = near ? "Watch Trigger" : "Buy Point";' in APP
    assert 'const referenceContext = near ? watchTargetSource(row) : entryDate || buyPointDate;' in APP
    assert 'const contextLabel = near ? "Source" : entryDate ? "Entry" : "Buy Point Date";' in APP
    assert 'return "Recovery High"' in APP
    assert 'return "TWK High"' in APP
    assert 'return "Pending High"' in APP
    assert 'class="selected-reference-primary">${referenceKey} ${fmt(reviewReferencePrice(row))}' in APP
    assert 'if (field === "current_vs_ibd_candidate_pct") return esc(fmt(reviewDistance(row), "pct"));' in APP


def test_status_sorting_is_owned_by_table_runtime_only() -> None:
    expected = 'const STATUS_ORDER = ["NEAR BREAKOUT", "ACTIONABLE", "UNCONFIRMED", "BELOW TRIGGER", "EXTENDED"]'

    assert expected in TABLE_RUNTIME
    assert "sortState" not in INTERACTION_RUNTIME
    assert "applyRememberedSort" not in INTERACTION_RUNTIME


def test_dynamic_filter_runtime_uses_v2_watch_semantics_with_v1_fallback() -> None:
    assert "function isNearBreakout(row)" in TABLE_RUNTIME
    assert "function reviewDistance(row)" in TABLE_RUNTIME
    assert "bf_watch_type" in TABLE_RUNTIME
    assert "bf_watch_distance_pct" in TABLE_RUNTIME
    assert "bf_watch_s_distance_pct" in TABLE_RUNTIME
    assert "bf_watch_m_distance_pct" in TABLE_RUNTIME
    assert 'displayStatus(row) === status' in TABLE_RUNTIME
    assert 'reviewSetup(row) === route' in TABLE_RUNTIME
    assert 'bounds(rows, "review_distance_pct")' in TABLE_RUNTIME
    assert 'bounds(rows.filter(isSignalActive), "ibd_entry_volume_ratio")' in TABLE_RUNTIME
    assert 'String(row.review_change_group || "UNCHANGED") !== "UNCHANGED" || isNearBreakout(row)' in TABLE_RUNTIME


def test_status_cards_use_explicit_watch_and_entry_groups_with_responsive_breakpoints() -> None:
    assert "review-stage-status" in APP
    assert "review-watch-grid" in APP
    assert "review-entry-grid" in APP
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
