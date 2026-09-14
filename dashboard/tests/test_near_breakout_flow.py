from __future__ import annotations

from pathlib import Path
import subprocess

import pandas as pd

from dashboard.build_static import PUBLIC_DASHBOARD_ROW_FIELDS, _records, _status_meta
from dashboard.services.bf_transition import ENTRY_STATUSES


DASHBOARD = Path(__file__).resolve().parents[1]
APP = (DASHBOARD / "app.js").read_text(encoding="utf-8")
INDEX = (DASHBOARD / "index.html").read_text(encoding="utf-8")
TABLE_RUNTIME = (DASHBOARD / "table_enhancements.js").read_text(encoding="utf-8")
INTERACTION_RUNTIME = (DASHBOARD / "interaction_runtime.js").read_text(encoding="utf-8")

WATCH_FIELDS = {
    "bf_watch_active",
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
    assert row["bf_watch_m_resistance"] == 100.0
    assert row["bf_watch_m_distance_pct"] == -1.2
    assert row["bf_watch_m_side"] == "RIGHT"
    assert "future_private_field" not in row


def test_near_breakout_is_a_ui_stage_not_an_ibd_entry_status() -> None:
    meta = _status_meta()

    assert "NEAR_BREAKOUT" in meta
    assert meta["NEAR_BREAKOUT"]["label"] == "NEAR BREAKOUT"
    assert "NEAR_BREAKOUT" not in ENTRY_STATUSES
    assert 'const STATUS_ORDER = ["NEAR_BREAKOUT", "ACTIONABLE"' in APP
    assert 'return isNearBreakout(row) ? "NEAR_BREAKOUT" : row.ibd_entry_status;' in APP
    assert 'const stageKey = near ? "Review Stage" : "Entry Status";' in APP


def test_near_breakout_uses_upstream_active_flag_and_target_fallback() -> None:
    assert "function isNearBreakout(row)" in APP
    assert "bool(row.bf_watch_active)" in APP
    assert "!isSignalActive(row)" in APP
    assert "function nearBreakoutTarget(row)" in APP
    assert "bf_watch_s_resistance" in APP
    assert "bf_watch_m_resistance" in APP
    assert "function nearBreakoutDistance(row)" in APP


def test_midweek_review_now_scope_keeps_current_near_breakout_rows() -> None:
    assert 'text(row.review_change_group, "UNCHANGED") !== "UNCHANGED" || isNearBreakout(row)' in APP
    assert "Review Now" in APP
    assert "All Review" in APP


def test_near_breakout_selected_detail_explains_pre_signal_structure() -> None:
    assert "function nearBreakoutDetailHtml(row)" in APP
    assert 'detailItem("Stage", "Near Breakout")' in APP
    assert 'detailItem("Target Source", targetSource)' in APP
    assert 'detailItem("S Side", text(row.bf_watch_s_side))' in APP
    assert 'detailItem("M Side", text(row.bf_watch_m_side))' in APP
    assert 'const volReason = near ? "Pre-signal"' in APP


def test_status_sorting_runtimes_follow_the_five_stage_review_flow() -> None:
    expected = 'const STATUS_ORDER = ["NEAR BREAKOUT", "ACTIONABLE", "UNCONFIRMED", "BELOW TRIGGER", "EXTENDED"]'

    assert expected in TABLE_RUNTIME
    assert expected in INTERACTION_RUNTIME


def test_dynamic_filter_runtime_uses_near_breakout_review_semantics() -> None:
    assert "function isNearBreakout(row)" in TABLE_RUNTIME
    assert "function reviewDistance(row)" in TABLE_RUNTIME
    assert 'displayStatus(row) === status' in TABLE_RUNTIME
    assert 'reviewSetup(row) === route' in TABLE_RUNTIME
    assert 'bounds(rows, "review_distance_pct")' in TABLE_RUNTIME
    assert 'String(row.review_change_group || "UNCHANGED") !== "UNCHANGED" || isNearBreakout(row)' in TABLE_RUNTIME


def test_status_cards_use_roomy_desktop_and_tablet_breakpoints() -> None:
    assert "@media (width > 1120px)" in INDEX
    assert "grid-template-columns: repeat(5, minmax(0, 1fr));" in INDEX
    assert "@media (width > 760px) and (width <= 1120px)" in INDEX
    assert "grid-template-columns: repeat(3, minmax(0, 1fr));" in INDEX


def test_dashboard_javascript_is_syntactically_valid() -> None:
    for script in ("app.js", "table_enhancements.js", "interaction_runtime.js"):
        subprocess.run(
            ["node", "--check", str(DASHBOARD / script)],
            check=True,
            capture_output=True,
            text=True,
        )
