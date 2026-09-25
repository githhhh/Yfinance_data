from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from dashboard.build_static import (
    PUBLIC_DASHBOARD_ROW_FIELDS,
    _records,
    build_dashboard_payload,
    build_site,
)
from dashboard.data_utils import load_pool_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
COMPLETE = PROJECT_ROOT / "us" / "breakout_follow_pool.csv"
MIDWEEK = PROJECT_ROOT / "us" / "breakout_follow_pool_midweek.csv"


def test_static_payload_uses_authoritative_normalized_complete_pool() -> None:
    payload = build_dashboard_payload(
        complete_path=COMPLETE,
        midweek_path=MIDWEEK,
        window_date=date(2026, 9, 5),
    )
    normalized = load_pool_csv(COMPLETE)

    assert payload["schema_version"] == 2
    assert len(payload["views"]["weekend"]["rows"]) == len(normalized)
    assert payload["meta"]["complete_snapshot_date"] is not None
    assert payload["default_period"] in {"WEEKEND", "MIDWEEK"}
    assert set(payload["views"]) == {"weekend", "midweek"}
    assert "rs_reference" not in payload["meta"]

    row = payload["views"]["weekend"]["rows"][0]
    assert set(row).issubset(PUBLIC_DASHBOARD_ROW_FIELDS)
    for field in (
        "code",
        "signal",
        "ibd_entry_status",
        "ibd_breakout_quality",
        "review_watch_active",
        "review_effective_entry_status",
        "review_priority",
        "buy_point_date",
        "ceiling",
        "ceiling_date",
        "breakout_date",
    ):
        assert field in row

    payload_text = json.dumps(payload)
    for field in (
        "rank_C_continuous",
        "C_continuous",
        "rs_percentile",
        "rs_1m_percentile",
        "rs_3m_percentile",
        "rs_6m_percentile",
        "ibd_candidate_extra",
    ):
        assert field not in payload_text


def test_static_records_fail_closed_on_new_pool_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "code": "SAFE",
                "signal": True,
                "ibd_entry_status": "ACTIONABLE",
                "future_private_field": "must-not-publish",
            }
        ]
    )

    row = _records(frame)[0]

    assert row["code"] == "SAFE"
    assert "future_private_field" not in row
    assert set(row).issubset(PUBLIC_DASHBOARD_ROW_FIELDS)


def test_static_records_publish_structure_timeline_fields_only_when_authoritative() -> None:
    frame = pd.DataFrame(
        [
            {
                "code": "STRUCT",
                "signal": True,
                "ceiling": 52.0,
                "ceiling_date": "2026-02-02",
                "breakout_date": "2026-04-27",
                "pullback_peak_date": "2026-06-22",
                "pullback_peak_price": 61.5,
                "pullback_duration_weeks": 4,
            }
        ]
    )

    row = _records(frame)[0]

    assert row["ceiling_date"] == "2026-02-02"
    assert row["breakout_date"] == "2026-04-27"
    assert row["pullback_peak_date"] == "2026-06-22"
    assert row["pullback_peak_price"] == 61.5


def test_buy_point_provenance_is_setup_aware_and_private_extra_stays_private() -> None:
    frame = pd.DataFrame(
        [
            {
                "code": "CEIL",
                "signal": True,
                "ibd_candidate_rule": "ceiling",
                "ibd_candidate_price": 63.52,
                "ceiling": 63.52,
                "ceiling_date": "2025-10-06",
                "ibd_candidate_extra": "{}",
            },
            {
                "code": "STALE_CEIL",
                "signal": True,
                "ibd_candidate_rule": "ceiling",
                "ibd_candidate_price": 63.52,
                "ceiling": 70.00,
                "ceiling_date": "2026-09-07",
                "ibd_candidate_extra": "{}",
            },
            {
                "code": "PIV",
                "signal": True,
                "ibd_candidate_rule": "pivot",
                "ibd_candidate_price": 93.37,
                "ceiling": 57.68,
                "ceiling_date": "2024-05-20",
                "ibd_candidate_extra": json.dumps(
                    {
                        "pivot_candidates": [
                            {"price": 90.0, "resistance_date": "2026-08-31"},
                            {"price": 93.37, "resistance_date": "2026-09-07"},
                        ]
                    }
                ),
            },
            {
                "code": "PIV_SELECTED",
                "signal": True,
                "ibd_candidate_rule": "pivot",
                "ibd_candidate_price": 43.98,
                "ibd_candidate_extra": json.dumps(
                    {
                        "selected_pivot": {
                            "price": 43.98,
                            "resistance_date": "2026-09-08",
                        },
                        "pivot_candidates": [
                            {"price": 43.98, "resistance_date": "2026-09-08"},
                            {"price": 44.95, "resistance_date": "2026-09-01"},
                        ],
                    }
                ),
            },
            {
                "code": "MA10",
                "signal": True,
                "ibd_candidate_rule": "ma10_touch_confirm",
                "ibd_candidate_price": 38.9,
                "ibd_candidate_extra": json.dumps(
                    {
                        "pending_high": 38.9,
                        "touch_date": "2026-08-17",
                        "confirm_date": "2026-09-07",
                    }
                ),
            },
            {
                "code": "PB",
                "signal": True,
                "ibd_candidate_rule": "ceiling_pullback",
                "ibd_candidate_price": 48.21,
                "ceiling": 45.99,
                "ceiling_date": "2025-12-08",
                "ibd_candidate_extra": json.dumps(
                    {
                        "pending_high": 48.21,
                        "touch_high": 48.21,
                        "touch_date": "2026-08-17",
                        "confirm_date": "2026-09-07",
                    }
                ),
            },
            {
                "code": "3WT",
                "signal": True,
                "snapshot_date": "2026-09-11",
                "ibd_candidate_rule": "three_weeks_tight",
                "ibd_candidate_price": 68.38,
                "ibd_candidate_extra": json.dumps(
                    {
                        "twk_high": 68.38,
                        "status": "breakout",
                        "tight_weeks": 3,
                    }
                ),
            },
        ]
    )

    rows = {row["code"]: row for row in _records(frame)}

    assert rows["CEIL"]["buy_point_date"] == "2025-10-06"
    assert rows["STALE_CEIL"]["buy_point_date"] is None
    assert rows["PIV"]["buy_point_date"] == "2026-09-07"
    assert rows["PIV"]["ceiling"] == 57.68
    assert rows["PIV"]["ceiling_date"] == "2024-05-20"
    assert rows["PIV_SELECTED"]["buy_point_date"] == "2026-09-08"
    assert rows["MA10"]["buy_point_date"] is None
    assert rows["PB"]["buy_point_date"] == "2026-08-17"
    assert rows["3WT"]["buy_point_date"] is None
    assert all("ibd_candidate_extra" not in row for row in rows.values())


def test_static_site_build_is_self_contained(tmp_path: Path) -> None:
    output = build_site(
        tmp_path / "site",
        complete_path=COMPLETE,
        midweek_path=MIDWEEK,
        window_date=date(2026, 9, 5),
    )

    for path in (
        output / "index.html",
        output / "app.js",
        output / "table_enhancements.js",
        output / "rs_runtime.js",
        output / "styles.css",
        output / "manifest.webmanifest",
        output / ".nojekyll",
        output / "data" / "dashboard.json",
    ):
        assert path.exists(), path

    dashboard_json = (output / "data" / "dashboard.json").read_text(encoding="utf-8")
    payload = json.loads(dashboard_json)
    assert payload["views"]["weekend"]["rows"]
    assert "rs_reference" not in payload["meta"]
    for field in (
        "rank_C_continuous",
        "C_continuous",
        "rs_percentile",
        "rs_1m_percentile",
        "rs_3m_percentile",
        "rs_6m_percentile",
        "ibd_candidate_extra",
    ):
        assert field not in dashboard_json
    for view in payload["views"].values():
        for row in view["rows"]:
            assert set(row).issubset(PUBLIC_DASHBOARD_ROW_FIELDS)

    index = (output / "index.html").read_text(encoding="utf-8")
    assert "streamlit" not in index.lower()
    assert "table_enhancements.js" in index
    assert "rs_runtime.js" in index
    assert "Dashboard mode" not in index

    app = (output / "app.js").read_text(encoding="utf-8")
    assert "Selected Overview" in app
    assert "RS Reference" in app
    assert "Buy Point Date" in app
    assert 'row.base_depth_pct' in app
    assert 'row.base_duration_weeks' in app
    assert 'row.ceiling_date' in app
    assert 'row.breakout_date' in app
    assert 'row.pullback_peak_date' in app
    assert 'row.pullback_peak_price' in app
    assert 'row.eps_yoy_growth' in app
    assert 'row.dist_to_52w_high_pct' in app
    assert 'data-action="detail"' not in app
    assert "C Rank" not in app
    assert "rank_C_continuous" not in app
    assert "C_RANK" not in app

    runtime = (output / "rs_runtime.js").read_text(encoding="utf-8")
    assert "Fred6725 / rs-log" in runtime
    assert "https://github.com/Fred6725/rs-log" in runtime
    assert "api.github.com/repos/Fred6725/rs-log/commits" in runtime
    assert "raw.githubusercontent.com/Fred6725/rs-log" in runtime
    assert "Reference only; never used by Pool, Gate, Top3 or default ordering." in runtime
    assert "Older than Pool" in runtime
    assert "Newer than Pool" in runtime
    assert "Loading reference" in runtime
    assert "Public reference · not official IBD RS" in runtime
    assert 'data-rs-info' in runtime
    assert "↻ Refresh" in runtime
    assert "↻ Retry" in runtime
    assert 'removeAttribute("data-rs-enhanced")' in runtime
    assert "Not used in Review Priority" not in runtime
    assert "scheduled" not in runtime.lower()

    enhancements = (output / "table_enhancements.js").read_text(encoding="utf-8")
    assert "Breakout Price Quality" in enhancements
    assert "Powerful" in enhancements
    assert "C Rank" not in enhancements
    assert "data-c-rank-table" not in enhancements

    styles = (output / "styles.css").read_text(encoding="utf-8")
    assert ".reference-header" not in styles
    assert ".reference-rule" not in styles
    assert ".topn-select" not in styles


def test_streamlit_runtime_has_been_removed() -> None:
    for relative in (
        "app.py",
        "run_app.py",
        "table_view.py",
        "review_styles.py",
        "review_tooltip.py",
        ".streamlit/config.toml",
    ):
        assert not (DASHBOARD_DIR / relative).exists()

    requirements = (DASHBOARD_DIR / "requirements.txt").read_text(encoding="utf-8").lower()
    for dependency in ("streamlit", "plotly", "aggrid"):
        assert dependency not in requirements
