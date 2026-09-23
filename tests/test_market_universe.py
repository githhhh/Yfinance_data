from __future__ import annotations

from pathlib import Path

import pandas as pd

import DataStore
from market_universe import (
    DOWNLOAD_UNIVERSE_SOURCE_FILES,
    IBD_DOUBLE_BOTTOM_TRACKING_SOURCE_FILES,
    build_download_universe,
)


def _write_codes(root: Path, relative_path: str, codes: list[str]) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"code": codes}).to_csv(path, index=False)


def _write_double_bottom_snapshot(
    root: Path,
    relative_path: str,
    rows: list[dict],
) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_download_universe_uses_only_explicit_strategy_input_sources(tmp_path):
    _write_codes(tmp_path, "us/52wk_new_high_results.csv", ["HIGH", "DOT.NAME"])
    _write_codes(tmp_path, "us/breakout_follow_pool.csv", ["POOL", "HIGH"])
    _write_codes(tmp_path, "us/breakout_follow_pool_midweek.csv", ["MID"])
    _write_codes(tmp_path, "us/eps_growth_screener_results.csv", ["EPS"])
    _write_codes(tmp_path, "us/weekly_vol_screener_results.csv", ["VOL"])
    _write_codes(tmp_path, "us/signal_eps_pit.csv", ["PIT_ONLY"])
    _write_codes(tmp_path, "us/unrelated_export.csv", ["UNRELATED"])

    _write_double_bottom_snapshot(
        tmp_path,
        "us/ibd_double_bottom_snapshot.csv",
        [
            {
                "code": "A_ACTIVE",
                "detection_path": "double_bottom",
                "signal_type": "Broken Support",
                "selection_eligible": False,
            },
            {
                "code": "B_ACTIVE",
                "detection_path": "ibd_double_bottom",
                "signal_type": "Extended",
                "selection_eligible": False,
                "ibd_double_bottom_signal_type": "Extended",
                "ibd_double_bottom_retired_reason": "",
            },
            {
                "code": "B_RETIRED",
                "detection_path": "ibd_double_bottom",
                "signal_type": "Retired",
                "selection_eligible": False,
                "ibd_double_bottom_signal_type": "Retired",
                "ibd_double_bottom_retired_reason": "matured_breakout",
            },
            {
                "code": "DUAL_B_RETIRED",
                "detection_path": "double_bottom+ibd_double_bottom",
                "signal_type": "Pullback Support",
                "selection_eligible": True,
                "ibd_double_bottom_signal_type": "Retired",
                "ibd_double_bottom_retired_reason": "failed_breakout",
            },
        ],
    )
    _write_double_bottom_snapshot(
        tmp_path,
        "us/ibd_double_bottom_snapshot_midweek.csv",
        [
            {
                "code": "B_WATCH",
                "detection_path": "ibd_double_bottom",
                "signal_type": "Early Watch",
                "selection_eligible": True,
                "ibd_double_bottom_signal_type": "Early Watch",
                "ibd_double_bottom_retired_reason": "",
            },
            {
                "code": "B_RETIRED_MID",
                "detection_path": "ibd_double_bottom",
                "signal_type": "Retired",
                "selection_eligible": False,
                "ibd_double_bottom_signal_type": "Retired",
                "ibd_double_bottom_retired_reason": "failed_breakout",
            },
        ],
    )

    expected = [
        "A_ACTIVE",
        "B_ACTIVE",
        "B_WATCH",
        "DOT-NAME",
        "DUAL_B_RETIRED",
        "EPS",
        "HIGH",
        "MID",
        "POOL",
        "VOL",
    ]

    assert build_download_universe(data_root=tmp_path) == expected
    assert DataStore.read_stock_list(str(tmp_path / "us")) == expected

    # Publication/cache artifacts remain excluded unless explicitly modeled.
    assert "us/signal_eps_pit.csv" not in DOWNLOAD_UNIVERSE_SOURCE_FILES
    assert "us/unrelated_export.csv" not in DOWNLOAD_UNIVERSE_SOURCE_FILES

    # Double Bottom snapshots are explicit lifecycle-tracking inputs, but use a
    # filtered projection instead of the generic "all code rows" source path.
    assert "us/ibd_double_bottom_snapshot.csv" not in DOWNLOAD_UNIVERSE_SOURCE_FILES
    assert "us/ibd_double_bottom_snapshot_midweek.csv" not in DOWNLOAD_UNIVERSE_SOURCE_FILES
    assert IBD_DOUBLE_BOTTOM_TRACKING_SOURCE_FILES == (
        "us/ibd_double_bottom_snapshot.csv",
        "us/ibd_double_bottom_snapshot_midweek.csv",
    )
