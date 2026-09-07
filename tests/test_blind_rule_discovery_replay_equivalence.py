from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from backtest.latest_quant_trade_replay import SnapshotWeek
from backtest.blind_rule_discovery import replay_equivalence as eq


def _pool(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_pool_equivalence_ignores_row_order_and_tiny_float_noise():
    baseline = _pool(
        [
            {"code": "AAA", "signal": True, "eps_yoy_growth": 25.0, "ibd_trigger_price": 10.0},
            {"code": "BBB", "signal": False, "eps_yoy_growth": None, "ibd_trigger_price": 20.0},
        ]
    )
    candidate = _pool(
        [
            {"code": "BBB", "signal": False, "eps_yoy_growth": None, "ibd_trigger_price": 20.0},
            {"code": "AAA", "signal": True, "eps_yoy_growth": 25.0 + 1e-13, "ibd_trigger_price": 10.0},
        ]
    )
    baseline = baseline.sort_values("code").reset_index(drop=True)
    candidate = candidate.sort_values("code").reset_index(drop=True)
    assert eq.compare_pool_frames(baseline, candidate) == []


@pytest.mark.parametrize(
    ("column", "candidate_value"),
    [
        ("signal", False),
        ("eps_yoy_growth", 26.0),
        ("ibd_trigger_price", 10.01),
        ("pullback_v_is_dry", False),
    ],
)
def test_pool_equivalence_detects_strategy_field_changes(column: str, candidate_value):
    baseline = _pool(
        [
            {
                "code": "AAA",
                "signal": True,
                "eps_yoy_growth": 25.0,
                "ibd_trigger_price": 10.0,
                "pullback_v_is_dry": True,
            }
        ]
    )
    candidate = baseline.copy()
    candidate.loc[0, column] = candidate_value
    mismatches = eq.compare_pool_frames(baseline, candidate)
    assert mismatches
    assert any(item.get("column") == column for item in mismatches)


def test_pool_equivalence_detects_schema_and_ticker_set_changes():
    baseline = _pool([{"code": "AAA", "signal": True, "x": 1.0}])
    missing_column = _pool([{"code": "AAA", "signal": True}])
    assert eq.compare_pool_frames(baseline, missing_column)[0]["kind"] == "columns"

    different_code = _pool([{"code": "BBB", "signal": True, "x": 1.0}])
    assert eq.compare_pool_frames(baseline, different_code)[0]["kind"] == "code_set"


def test_pool_reader_rejects_duplicate_ticker_keys(tmp_path: Path):
    path = tmp_path / "pool.csv"
    pd.DataFrame(
        [{"code": "AAA", "signal": True}, {"code": "AAA", "signal": False}]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate code"):
        eq._read_pool(path)


def test_equivalence_week_selection_spans_time_and_stresses_high_signal(tmp_path: Path):
    weeks = [
        SnapshotWeek(f"2024-{month:02d}-05", f"2024-{month:02d}-05")
        for month in range(1, 7)
    ]
    signal_counts = [0, 1, 20, 2, 15, 0]
    for week, count in zip(weeks, signal_counts):
        week_dir = tmp_path / week.snapshot_date
        week_dir.mkdir(parents=True)
        pd.DataFrame(
            [{"code": f"S{i}", "signal": True} for i in range(count)]
            or [{"code": "NONE", "signal": False}]
        ).to_csv(week_dir / "breakout_follow_pool.csv", index=False)

    selected = eq.select_equivalence_weeks(
        weeks, baseline_root=tmp_path, sample_weeks=4
    )
    dates = [week.snapshot_date for week in selected]
    assert weeks[0].snapshot_date in dates
    assert weeks[-1].snapshot_date in dates
    assert weeks[2].snapshot_date in dates  # highest signal count
    assert len(dates) == 4


def test_local_eps_assets_fail_closed_and_accept_valid_local_files(tmp_path: Path):
    eps_seed = tmp_path / "eps_seed.csv"
    eps_seed.write_text("snapshot_date,code\n2024-01-05,AAA\n")
    sec = tmp_path / "output" / "eps_pit_cache" / "sec"
    sec.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="companyfacts"):
        eq._assert_local_eps_assets(tmp_path, eps_seed)

    companyfacts = sec / "companyfacts.zip"
    with zipfile.ZipFile(companyfacts, "w") as archive:
        archive.writestr("CIK0000000001.json", json.dumps({"facts": {}}))
    with pytest.raises(FileNotFoundError, match="ticker map"):
        eq._assert_local_eps_assets(tmp_path, eps_seed)

    (sec / "company_tickers.json").write_text(
        json.dumps({"0": {"ticker": "AAA", "cik_str": 1}})
    )
    assets = eq._assert_local_eps_assets(tmp_path, eps_seed)
    assert assets["companyfacts_size"] > 0
    assert assets["eps_seed_sha256"]
