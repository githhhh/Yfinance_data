from __future__ import annotations

import pandas as pd

from backtest.blind_rule_discovery.experiment import build_blind_dataset


def _prices(rows):
    frame = pd.DataFrame(rows, columns=["date", "Open", "High", "Low", "Close"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _spy_path(periods: int = 700):
    dates = pd.bdate_range("2023-01-02", periods=periods)
    return _prices([(d, 100.0, 101.0, 99.0, 100.0) for d in dates])


def test_entry_day_ambiguous_remains_dataset_usable_with_horizon_outcomes():
    dates = pd.bdate_range("2024-01-02", periods=70)
    rows = [(dates[0], 98.0, 101.0, 90.0, 100.0)]
    rows += [(d, 100.0, 105.0, 95.0, 100.0) for d in dates[1:]]
    candidates = pd.DataFrame([
        {
            "code": "AAA",
            "snapshot_date": pd.Timestamp("2024-01-01"),
            "ibd_trigger_price": 100.0,
            "ibd_candidate_price": 100.0,
            "ibd_entry_volume_ratio": 1.5,
        }
    ])
    agent, _, reviewer = build_blind_dataset(candidates, {"AAA": _prices(rows)}, _spy_path())
    assert len(agent) == 1
    assert agent.iloc[0]["Y_primary"] == "ambiguous"
    assert pd.notna(agent.iloc[0]["Y_12w_return"])
    assert reviewer.iloc[0]["reason"] == "entry_day_stop_order_unknown"
