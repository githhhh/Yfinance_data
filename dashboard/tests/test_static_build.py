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
from dashboard.rs_reference import RSReferenceSnapshot


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
    assert "c_rank" not in payload["views"]

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
        "rs_percentile",
        "rs_1m_percentile",
        "rs_3m_percentile",
        "rs_6m_percentile",
    ):
        assert field in row
    assert "rank_C_continuous" not in row
    assert "C_continuous" not in row


def test_matching_rs_snapshot_is_joined_without_affecting_pool_rows() -> None:
    normalized = load_pool_csv(COMPLETE)
    market_date = pd.to_datetime(normalized["snapshot_date"].iloc[0]).date()
    code = str(normalized["code"].iloc[0])
    reference = RSReferenceSnapshot(
        market_date=market_date,
        ratings={
            code.upper(): {
                "rs_percentile": 97,
                "rs_1m_percentile": 94,
                "rs_3m_percentile": 91,
                "rs_6m_percentile": 86,
            }
        },
        commit_sha="abc123",
    )

    payload = build_dashboard_payload(
        complete_path=COMPLETE,
        midweek_path=MIDWEEK,
        window_date=date(2026, 9, 5),
        rs_reference=reference,
    )

    row = next(item for item in payload["views"]["weekend"]["rows"] if item["code"] == code)
    assert row["rs_percentile"] == 97
    assert row["rs_1m_percentile"] == 94
    assert payload["meta"]["rs_reference"]["matches_complete"] is True


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
        output / "rs_enhancements.js",
        output / "styles.css",
        output / "manifest.webmanifest",
        output / ".nojekyll",
        output / "data" / "dashboard.json",
    ):
        assert path.exists(), path

    payload = json.loads((output / "data" / "dashboard.json").read_text(encoding="utf-8"))
    assert payload["views"]["weekend"]["rows"]
    for view in payload["views"].values():
        for row in view["rows"]:
            assert set(row).issubset(PUBLIC_DASHBOARD_ROW_FIELDS)
            assert "rank_C_continuous" not in row
            assert "C_continuous" not in row

    index = (output / "index.html").read_text(encoding="utf-8").lower()
    assert "streamlit" not in index
    assert "table_enhancements.js" in index
    assert "rs_enhancements.js" in index

    enhancements = (output / "table_enhancements.js").read_text(encoding="utf-8")
    assert "Breakout Price Quality" in enhancements
    assert "Powerful" in enhancements
    assert "data-sort-field" in enhancements

    rs_enhancements = (output / "rs_enhancements.js").read_text(encoding="utf-8")
    assert "Fred6725/rs-log" in rs_enhancements
    assert "Exact trading-date match required" in rs_enhancements


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
