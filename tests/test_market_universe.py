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
    _write_codes(tmp_path, "us/unrelated_export.csv", ["UNRELATED"])

    expected = ["DOT-NAME", "EPS", "HIGH", "MID", "POOL", "VOL"]

    assert build_download_universe(data_root=tmp_path) == expected
    assert DataStore.read_stock_list(str(tmp_path / "us")) == expected
    assert "us/signal_eps_pit.csv" not in DOWNLOAD_UNIVERSE_SOURCE_FILES
