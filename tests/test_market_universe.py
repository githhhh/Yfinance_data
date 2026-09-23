from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import DataStore
from market_universe import (
    DOWNLOAD_UNIVERSE_SOURCE_FILES,
    IBD_DOUBLE_BOTTOM_TRACKING_SOURCE_FILES,
    IBD_DOUBLE_BOTTOM_TRACKING_STATE_FILE,
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
    *,
    snapshot_date: str,
) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if "snapshot_date" not in frame.columns:
        frame["snapshot_date"] = snapshot_date
    else:
        frame["snapshot_date"] = snapshot_date
    if frame.empty:
        frame = frame.reindex(
            columns=[
                "code",
                "snapshot_date",
                "detection_path",
                "signal_type",
                "selection_eligible",
            ]
        )
    frame.to_csv(path, index=False)


def _write_double_bottom_state(
    root: Path,
    *,
    kind: str,
    snapshot_date: str,
) -> None:
    path = root / IBD_DOUBLE_BOTTOM_TRACKING_STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"kind": kind, "snapshot_date": snapshot_date}),
        encoding="utf-8",
    )


def _write_regular_sources(tmp_path: Path) -> None:
    _write_codes(tmp_path, "us/52wk_new_high_results.csv", ["HIGH", "DOT.NAME"])
    _write_codes(tmp_path, "us/breakout_follow_pool.csv", ["POOL", "HIGH"])
    _write_codes(tmp_path, "us/breakout_follow_pool_midweek.csv", ["MID"])
    _write_codes(tmp_path, "us/eps_growth_screener_results.csv", ["EPS"])
    _write_codes(tmp_path, "us/weekly_vol_screener_results.csv", ["VOL"])
    _write_codes(tmp_path, "us/signal_eps_pit.csv", ["PIT_ONLY"])
    _write_codes(tmp_path, "us/unrelated_export.csv", ["UNRELATED"])


def test_download_universe_uses_only_current_double_bottom_snapshot(tmp_path):
    _write_regular_sources(tmp_path)

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
        snapshot_date="2026-09-18",
    )
    _write_double_bottom_snapshot(
        tmp_path,
        "us/ibd_double_bottom_snapshot_midweek.csv",
        [
            {
                "code": "STALE_MIDWEEK_ACTIVE",
                "detection_path": "ibd_double_bottom",
                "signal_type": "Extended",
                "selection_eligible": False,
            },
        ],
        snapshot_date="2026-09-17",
    )
    _write_double_bottom_state(
        tmp_path,
        kind="complete",
        snapshot_date="2026-09-18",
    )

    expected = [
        "A_ACTIVE",
        "B_ACTIVE",
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

    assert "us/signal_eps_pit.csv" not in DOWNLOAD_UNIVERSE_SOURCE_FILES
    assert "us/unrelated_export.csv" not in DOWNLOAD_UNIVERSE_SOURCE_FILES
    assert "us/ibd_double_bottom_snapshot.csv" not in DOWNLOAD_UNIVERSE_SOURCE_FILES
    assert "us/ibd_double_bottom_snapshot_midweek.csv" not in DOWNLOAD_UNIVERSE_SOURCE_FILES
    assert IBD_DOUBLE_BOTTOM_TRACKING_SOURCE_FILES == (
        "us/ibd_double_bottom_snapshot.csv",
        "us/ibd_double_bottom_snapshot_midweek.csv",
    )


def test_newer_complete_retirement_is_not_revived_by_stale_midweek(tmp_path):
    _write_double_bottom_snapshot(
        tmp_path,
        "us/ibd_double_bottom_snapshot.csv",
        [
            {
                "code": "AAA",
                "detection_path": "ibd_double_bottom",
                "signal_type": "Retired",
                "selection_eligible": False,
                "ibd_double_bottom_signal_type": "Retired",
                "ibd_double_bottom_retired_reason": "matured_breakout",
            }
        ],
        snapshot_date="2026-09-18",
    )
    _write_double_bottom_snapshot(
        tmp_path,
        "us/ibd_double_bottom_snapshot_midweek.csv",
        [
            {
                "code": "AAA",
                "detection_path": "ibd_double_bottom",
                "signal_type": "Extended",
                "selection_eligible": False,
            }
        ],
        snapshot_date="2026-09-17",
    )
    _write_double_bottom_state(tmp_path, kind="complete", snapshot_date="2026-09-18")

    assert build_download_universe(data_root=tmp_path) == []


def test_empty_current_snapshot_clears_stale_tracking(tmp_path):
    _write_double_bottom_snapshot(
        tmp_path,
        "us/ibd_double_bottom_snapshot.csv",
        [],
        snapshot_date="2026-09-18",
    )
    _write_double_bottom_snapshot(
        tmp_path,
        "us/ibd_double_bottom_snapshot_midweek.csv",
        [
            {
                "code": "AAA",
                "detection_path": "ibd_double_bottom",
                "signal_type": "Extended",
                "selection_eligible": False,
            }
        ],
        snapshot_date="2026-09-17",
    )
    _write_double_bottom_state(tmp_path, kind="complete", snapshot_date="2026-09-18")

    assert build_download_universe(data_root=tmp_path) == []


def test_current_snapshot_schema_failure_is_fail_closed(tmp_path):
    path = tmp_path / "us/ibd_double_bottom_snapshot.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "code": "AAA",
                "snapshot_date": "2026-09-18",
                "signal_type": "Extended",
            }
        ]
    ).to_csv(path, index=False)
    _write_double_bottom_state(tmp_path, kind="complete", snapshot_date="2026-09-18")

    with pytest.raises(ValueError, match="missing columns"):
        build_download_universe(data_root=tmp_path)


def test_snapshot_without_state_is_fail_closed(tmp_path):
    _write_double_bottom_snapshot(
        tmp_path,
        "us/ibd_double_bottom_snapshot.csv",
        [
            {
                "code": "AAA",
                "detection_path": "double_bottom",
                "signal_type": "Early Watch",
                "selection_eligible": True,
            }
        ],
        snapshot_date="2026-09-18",
    )

    with pytest.raises(RuntimeError, match="state missing"):
        build_download_universe(data_root=tmp_path)


def test_no_snapshot_and_no_state_is_backward_compatible(tmp_path):
    _write_regular_sources(tmp_path)

    assert build_download_universe(data_root=tmp_path) == [
        "DOT-NAME",
        "EPS",
        "HIGH",
        "MID",
        "POOL",
        "VOL",
    ]
