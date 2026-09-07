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
    ):
        assert field in row
    for field in (
        "rank_C_continuous",
        "C_continuous",
        "rs_percentile",
        "rs_1m_percentile",
        "rs_3m_percentile",
        "rs_6m_percentile",
    ):
        assert field not in row


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

    payload = json.loads((output / "data" / "dashboard.json").read_text(encoding="utf-8"))
    assert payload["views"]["weekend"]["rows"]
    assert "rs_reference" not in payload["meta"]
    for view in payload["views"].values():
        for row in view["rows"]:
            assert set(row).issubset(PUBLIC_DASHBOARD_ROW_FIELDS)
            assert "rank_C_continuous" not in row
            assert "C_continuous" not in row
            assert "rs_percentile" not in row
            assert "rs_1m_percentile" not in row
            assert "rs_3m_percentile" not in row
            assert "rs_6m_percentile" not in row

    index = (output / "index.html").read_text(encoding="utf-8")
    assert "streamlit" not in index.lower()
    assert "table_enhancements.js" in index
    assert "rs_runtime.js" in index
    assert "Dashboard mode" not in index

    app = (output / "app.js").read_text(encoding="utf-8")
    assert "RS Reference" in app
    assert "C Rank" not in app
    assert "rank_C_continuous" not in app
    assert "C_RANK" not in app

    runtime = (output / "rs_runtime.js").read_text(encoding="utf-8")
    assert "Fred6725/rs-log" in runtime
    assert "api.github.com/repos/Fred6725/rs-log/commits" in runtime
    assert "raw.githubusercontent.com/Fred6725/rs-log" in runtime
    assert "Reference only; never used by Pool, Gate, Top3 or default ordering." in runtime
    assert "scheduled" not in runtime.lower()

    enhancements = (output / "table_enhancements.js").read_text(encoding="utf-8")
    assert "Breakout Price Quality" in enhancements
    assert "Powerful" in enhancements
    assert 'data-rs-enhanced="true"' in enhancements
    assert 'title^="RS "' in enhancements
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
