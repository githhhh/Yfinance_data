from __future__ import annotations

from pathlib import Path

import pandas as pd

import DataStore
from market_universe import DOWNLOAD_UNIVERSE_SOURCE_FILES, build_download_universe


def _write_codes(root: Path, relative_path: str, codes: list[str]) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"code": codes}).to_csv(path, index=False)


def test_download_universe_uses_only_explicit_strategy_input_sources(tmp_path):
    _write_codes(tmp_path, "us/52wk_new_high_results.csv", ["HIGH", "DOT.NAME"])
    _write_codes(tmp_path, "us/breakout_follow_pool.csv", ["POOL", "HIGH"])
    _write_codes(tmp_path, "us/breakout_follow_pool_midweek.csv", ["MID"])
    _write_codes(tmp_path, "us/eps_growth_screener_results.csv", ["EPS"])
    _write_codes(tmp_path, "us/weekly_vol_screener_results.csv", ["VOL"])
    _write_codes(tmp_path, "us/signal_eps_pit.csv", ["PIT_ONLY"])
    _write_codes(tmp_path, "us/ibd_double_bottom_snapshot.csv", ["DB_ONLY"])
    _write_codes(tmp_path, "us/ibd_double_bottom_snapshot_midweek.csv", ["DB_MID_ONLY"])
    _write_codes(tmp_path, "us/unrelated_export.csv", ["UNRELATED"])

    expected = ["DOT-NAME", "EPS", "HIGH", "MID", "POOL", "VOL"]

    assert build_download_universe(data_root=tmp_path) == expected
    assert DataStore.read_stock_list(str(tmp_path / "us")) == expected
    assert "us/signal_eps_pit.csv" not in DOWNLOAD_UNIVERSE_SOURCE_FILES
    assert "us/ibd_double_bottom_snapshot.csv" not in DOWNLOAD_UNIVERSE_SOURCE_FILES
    assert "us/ibd_double_bottom_snapshot_midweek.csv" not in DOWNLOAD_UNIVERSE_SOURCE_FILES


def test_weekend_signal_final_pick_only_in_next_review_week(tmp_path):
    from datetime import date
    from market_universe import pick_active_weekend_signal_codes

    _write_codes(tmp_path, "us/52wk_new_high_results.csv", [])
    pd.DataFrame([
        {"code": "ALPHA", "snapshot_date": "2026-10-09", "signal": True},
        {"code": "BETA", "snapshot_date": "2026-10-09", "signal": True},
        {"code": "NO", "snapshot_date": "2026-10-09", "signal": False},
    ]).to_csv(tmp_path / "us/breakout_follow_pool.csv", index=False)

    pick = pick_active_weekend_signal_codes(data_root=tmp_path, review_date=date(2026, 10, 13))
    assert pick == ["ALPHA", "BETA"]
    assert pick_active_weekend_signal_codes(data_root=tmp_path, review_date=date(2026, 10, 20)) == []
    assert pick_active_weekend_signal_codes(data_root=tmp_path, review_date=date(2026, 10, 10)) == []
    fallback = ["OTHER", "ALPHA"]
    assert set([*fallback, *pick]) == {"OTHER", "ALPHA", "BETA"}


def test_weekend_signal_pick_matches_snapshot_boolean_and_rejects_bad_date(tmp_path):
    from datetime import date
    from market_universe import pick_active_weekend_signal_codes

    file = tmp_path / "us/breakout_follow_pool.csv"
    file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"code": "YES", "signal": "1.0", "snapshot_date": "2026-10-09"},
        {"code": "CONF", "signal": "yes", "snapshot_date": "2026-10-09"},
        {"code": "NO", "signal": "no", "snapshot_date": "2026-10-09"},
    ]).to_csv(file, index=False)
    assert pick_active_weekend_signal_codes(
        data_root=tmp_path, review_date=date(2026, 10, 13)
    ) == ["CONF", "YES"]
    pd.DataFrame([
        {"code": "YES", "signal": True, "snapshot_date": "2026-10-09"},
        {"code": "BAD", "signal": True, "snapshot_date": "bad-date"},
    ]).to_csv(file, index=False)
    assert pick_active_weekend_signal_codes(
        data_root=tmp_path, review_date=date(2026, 10, 13)
    ) == []
