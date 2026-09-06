from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from dashboard.build_static import build_dashboard_payload, build_site
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

    assert payload["schema_version"] == 1
    assert len(payload["views"]["weekend"]["rows"]) == len(normalized)
    assert payload["meta"]["complete_snapshot_date"] is not None
    assert payload["default_period"] in {"WEEKEND", "MIDWEEK"}

    row = payload["views"]["weekend"]["rows"][0]
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
        output / "styles.css",
        output / "manifest.webmanifest",
        output / ".nojekyll",
        output / "data" / "dashboard.json",
    ):
        assert path.exists(), path

    payload = json.loads((output / "data" / "dashboard.json").read_text(encoding="utf-8"))
    assert payload["views"]["weekend"]["rows"]
    index = (output / "index.html").read_text(encoding="utf-8").lower()
    assert "streamlit" not in index
    assert "table_enhancements.js" in index

    enhancements = (output / "table_enhancements.js").read_text(encoding="utf-8")
    assert "Breakout Price Quality" in enhancements
    assert "Powerful" in enhancements
    assert "data-sort-field" in enhancements


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
