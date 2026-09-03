from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.b0_error_atlas.analysis import feature_redundancy
from backtest.b0_error_atlas.config import PROTOCOL_VERSION
from backtest.b0_error_atlas.data import allowed_raw_features
from backtest.b0_error_atlas.features import (
    add_cross_sectional_context,
    add_market_features,
    add_pre_snapshot_price_features,
)
from backtest.b0_error_atlas.labels import add_path_labels, task_frames
from backtest.b0_error_atlas.modeling import chronological_quarter_splits


def _price_rows(code: str, dates, closes, lows=None, highs=None, opens=None):
    closes = list(map(float, closes))
    lows = closes if lows is None else list(map(float, lows))
    highs = closes if highs is None else list(map(float, highs))
    opens = closes if opens is None else list(map(float, opens))
    return pd.DataFrame({
        "code": code,
        "date": pd.to_datetime(dates),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": 100.0,
        "source": "test",
    })


def test_protocol_v1():
    assert PROTOCOL_VERSION == "b0_error_atlas_v1"


def test_path_label_clean_big_winner():
    panel = pd.DataFrame([{
        "snapshot_date": "2026-01-02",
        "code": "AAA",
        "current_b0_eligible": False,
        "current_b0_selected": False,
        "next_open_price_valid": True,
        "next_open_entry_date": "2026-01-05",
        "next_open_end_date": "2026-02-02",
        "next_open_w4_return_pct": 25.0,
        "next_open_w4_stop8": False,
    }])
    prices = _price_rows(
        "AAA",
        ["2026-01-05", "2026-01-12", "2026-02-02"],
        [100, 115, 125],
        lows=[99, 110, 120],
        highs=[102, 119, 126],
        opens=[100, 114, 124],
    )
    out = add_path_labels(panel, prices).iloc[0]
    assert bool(out["clean_big_winner"]) is True
    assert bool(out["rebound_big_winner"]) is False
    assert out["path_order"] == "PROFIT_ONLY"
    assert bool(out["strict_path_failure"]) is False


def test_path_label_rebound_big_winner_is_not_clean():
    panel = pd.DataFrame([{
        "snapshot_date": "2026-01-02",
        "code": "AAA",
        "current_b0_eligible": False,
        "current_b0_selected": False,
        "next_open_price_valid": True,
        "next_open_entry_date": "2026-01-05",
        "next_open_end_date": "2026-02-02",
        "next_open_w4_return_pct": 30.0,
        "next_open_w4_stop8": True,
    }])
    prices = _price_rows(
        "AAA",
        ["2026-01-05", "2026-01-06", "2026-01-20", "2026-02-02"],
        [100, 94, 125, 130],
        lows=[99, 91, 120, 125],
        highs=[102, 97, 126, 131],
        opens=[100, 95, 124, 129],
    )
    out = add_path_labels(panel, prices).iloc[0]
    assert bool(out["clean_big_winner"]) is False
    assert bool(out["rebound_big_winner"]) is True
    assert out["path_order"] == "STOP_FIRST"
    assert bool(out["strict_path_failure"]) is True


def test_same_day_stop_profit_is_ambiguous_and_excluded_from_strict_failure():
    panel = pd.DataFrame([{
        "snapshot_date": "2026-01-02",
        "code": "AAA",
        "current_b0_eligible": False,
        "current_b0_selected": False,
        "next_open_price_valid": True,
        "next_open_entry_date": "2026-01-05",
        "next_open_end_date": "2026-02-02",
        "next_open_w4_return_pct": 5.0,
        "next_open_w4_stop8": True,
    }])
    prices = _price_rows(
        "AAA",
        ["2026-01-05", "2026-02-02"],
        [100, 105],
        lows=[90, 100],
        highs=[125, 110],
        opens=[100, 104],
    )
    out = add_path_labels(panel, prices).iloc[0]
    assert out["path_order"] == "SAME_DAY_AMBIGUOUS"
    assert pd.isna(out["strict_path_failure"])


def test_task_frames_separate_gate_and_selector_misses():
    base = {
        "path_valid": True,
        "strict_path_failure": False,
        "clean_big_winner": True,
        "path_stop8_hit": False,
        "next_open_w4_return_pct": 25.0,
    }
    panel = pd.DataFrame([
        {
            **base,
            "snapshot_date": "2026-01-02", "code": "G",
            "current_b0_eligible": False, "current_b0_selected": False,
        },
        {
            **base,
            "snapshot_date": "2026-01-02", "code": "S",
            "current_b0_eligible": True, "current_b0_selected": False,
        },
        {
            **base,
            "snapshot_date": "2026-01-02", "code": "P",
            "current_b0_eligible": True, "current_b0_selected": True,
        },
        {
            "snapshot_date": "2026-01-02", "code": "B",
            "path_valid": True, "strict_path_failure": True,
            "clean_big_winner": False, "path_stop8_hit": True,
            "next_open_w4_return_pct": -10.0,
            "current_b0_eligible": True, "current_b0_selected": True,
        },
    ])
    tasks = task_frames(panel)
    assert "G" in tasks["gate_recovery_clean20_vs_fail"]["code"].tolist()
    assert "S" in tasks["selector_recovery_clean20_vs_fail"]["code"].tolist()
    assert set(tasks["all_unselected_recovery_clean20_vs_fail"]["code"]) == {"G", "S"}
    veto = tasks["selected_veto_fail_vs_clean8"]
    assert set(veto["code"]) == {"P", "B"}
    assert dict(zip(veto["code"], veto["target"])) == {"P": 0, "B": 1}


def test_pre_snapshot_features_ignore_future_bar():
    dates = pd.date_range("2025-12-01", periods=30, freq="D")
    prices = _price_rows("AAA", dates, np.linspace(100, 110, len(dates)))
    panel = pd.DataFrame([{"snapshot_date": "2025-12-20", "code": "AAA"}])
    before = add_pre_snapshot_price_features(panel, prices)

    future = _price_rows(
        "AAA",
        ["2026-01-15"],
        [1000],
        lows=[1],
        highs=[2000],
        opens=[500],
    )
    after = add_pre_snapshot_price_features(panel, pd.concat([prices, future], ignore_index=True))

    for col in [
        "pit_max_drawdown_20",
        "pit_max_down_day_20",
        "pit_close_position_mean_20",
    ]:
        a = before.iloc[0][col]
        b = after.iloc[0][col]
        if pd.isna(a):
            assert pd.isna(b)
        else:
            assert a == b


def test_market_features_ignore_future_spy_bar():
    dates = pd.date_range("2025-12-01", periods=70, freq="D")
    prices = _price_rows("SPY", dates, np.linspace(100, 120, len(dates)))
    panel = pd.DataFrame([{"snapshot_date": "2026-01-20", "code": "AAA"}])
    before = add_market_features(panel, prices)

    future = _price_rows("SPY", ["2026-03-01"], [1000])
    after = add_market_features(panel, pd.concat([prices, future], ignore_index=True))
    for col in ["pit_spy_mom20", "pit_spy_mom60", "pit_spy_rv20", "pit_spy_drawdown60"]:
        assert before.iloc[0][col] == after.iloc[0][col]


def test_cross_sectional_context_is_snapshot_local():
    frame = pd.DataFrame([
        {
            "snapshot_date": "2026-01-02", "code": "A", "sector": "Tech",
            "industry": "I1", "mom_20": 0.1, "rv_20": 0.2,
            "ibd_entry_volume_ratio": 1.0, "dist_to_52w_high_pct": -2,
            "is_actionable": 1,
        },
        {
            "snapshot_date": "2026-01-02", "code": "B", "sector": "Tech",
            "industry": "I2", "mom_20": 0.2, "rv_20": 0.3,
            "ibd_entry_volume_ratio": 2.0, "dist_to_52w_high_pct": -1,
            "is_actionable": 0,
        },
        {
            "snapshot_date": "2026-01-09", "code": "C", "sector": "Tech",
            "industry": "I1", "mom_20": 100.0, "rv_20": 100.0,
            "ibd_entry_volume_ratio": 100.0, "dist_to_52w_high_pct": 0,
            "is_actionable": 1,
        },
    ])
    out = add_cross_sectional_context(frame)
    a = out[out["code"] == "A"].iloc[0]
    b = out[out["code"] == "B"].iloc[0]
    assert a["xs_mom20_pct"] == 0.5
    assert b["xs_mom20_pct"] == 1.0
    assert a["sector_candidate_count"] == 2
    assert a["sector_actionable_share"] == 0.5


def test_chronological_splits_never_train_on_future_quarter():
    frame = pd.DataFrame([
        {"snapshot_date": "2025-10-01", "target": 0},
        {"snapshot_date": "2025-10-08", "target": 1},
        {"snapshot_date": "2026-01-01", "target": 0},
        {"snapshot_date": "2026-01-08", "target": 1},
        {"snapshot_date": "2026-01-15", "target": 0},
        {"snapshot_date": "2026-01-22", "target": 1},
        {"snapshot_date": "2026-04-01", "target": 0},
        {"snapshot_date": "2026-04-08", "target": 1},
        {"snapshot_date": "2026-04-15", "target": 0},
        {"snapshot_date": "2026-04-22", "target": 1},
        {"snapshot_date": "2026-04-29", "target": 0},
        {"snapshot_date": "2026-05-06", "target": 1},
        {"snapshot_date": "2026-07-01", "target": 0},
        {"snapshot_date": "2026-07-08", "target": 1},
        {"snapshot_date": "2026-07-15", "target": 0},
        {"snapshot_date": "2026-07-22", "target": 1},
    ])
    for test_q, train_idx, test_idx in chronological_quarter_splits(frame):
        train_dates = pd.to_datetime(frame.iloc[train_idx]["snapshot_date"])
        test_dates = pd.to_datetime(frame.iloc[test_idx]["snapshot_date"])
        assert train_dates.max() < test_dates.min()
        assert str(pd.Period(test_dates.iloc[0], freq="Q")) == test_q


def test_feature_redundancy_detects_duplicate_numeric_dimensions():
    frame = pd.DataFrame({
        "a": np.arange(30, dtype=float),
        "b": np.arange(30, dtype=float) * 2,
        "c": np.sin(np.arange(30, dtype=float)),
    })
    summary, pairs = feature_redundancy(frame, ["a", "b", "c"])
    assert summary["pairs_abs_spearman_ge_0_85"] >= 1
    ab = pairs[
        ((pairs["feature_a"] == "a") & (pairs["feature_b"] == "b"))
        | ((pairs["feature_a"] == "b") & (pairs["feature_b"] == "a"))
    ]
    assert not ab.empty
    assert ab.iloc[0]["abs_spearman"] > 0.99



def test_raw_only_feature_allowlist_excludes_b0_and_outcomes():
    frame = pd.DataFrame({
        "current_vs_ibd_candidate_pct": [1.0],
        "mom_20": [0.1],
        "industry": ["I1"],
        "b0_eligible": [True],
        "w4_return_pct": [10.0],
        "current_b0_raw_rank": [1],
    })
    numeric, categorical = allowed_raw_features(frame)
    assert "current_vs_ibd_candidate_pct" in numeric
    assert "mom_20" in numeric
    assert "industry" in categorical
    forbidden = {
        "b0_eligible",
        "w4_return_pct",
        "current_b0_raw_rank",
    }
    assert forbidden.isdisjoint(set(numeric) | set(categorical))
