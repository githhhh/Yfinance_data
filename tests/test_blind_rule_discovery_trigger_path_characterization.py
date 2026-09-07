from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.blind_rule_discovery.outcomes import OutcomeConfig, RESEARCH_PRICE_MODE
from backtest.blind_rule_discovery.trigger_path_characterization import (
    evaluate_trigger_path,
    matched_selected_vs_unselected,
    rolling_stock_selection,
    search_stock_interactions,
    summarize_path,
    within_snapshot_feature_contrasts,
)


def price_frame(*, path: str) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=70)
    open_px = np.full(len(dates), 100.0)
    high = np.full(len(dates), 102.0)
    low = np.full(len(dates), 98.0)
    close = np.linspace(100.0, 110.0, len(dates))

    # Signal is first row; executable entry is the next session at trigger 100.
    open_px[1] = 99.0
    high[1] = 101.0
    low[1] = 98.0
    close[1] = 100.5

    if path == "winner":
        high[5] = 121.0  # +20% first, within 3w.
        low[10] = 91.0   # stop only later.
        close[5] = 112.0
    elif path == "stop":
        low[4] = 91.0
        high[10] = 121.0
        close[4] = 94.0
    elif path == "ambiguous":
        high[4] = 121.0
        low[4] = 91.0
    elif path == "unresolved":
        pass
    else:
        raise ValueError(path)

    frame = pd.DataFrame({"Open": open_px, "High": high, "Low": low, "Close": close, "date": dates})
    frame.attrs["price_adjustment_mode"] = RESEARCH_PRICE_MODE
    return frame


def spy_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=400)
    close = np.linspace(90.0, 120.0, len(dates))
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.001,
            "Low": close * 0.999,
            "Close": close,
            "date": dates,
        }
    )
    frame.attrs["price_adjustment_mode"] = RESEARCH_PRICE_MODE
    return frame


def test_three_week_first_passage_orders_target_and_stop():
    spy = spy_frame()
    winner = evaluate_trigger_path(
        price_frame(path="winner"),
        pd.Timestamp("2024-01-02"),
        trigger_price=100.0,
        spy_prices=spy,
        config=OutcomeConfig(),
    )
    stop = evaluate_trigger_path(
        price_frame(path="stop"),
        pd.Timestamp("2024-01-02"),
        trigger_price=100.0,
        spy_prices=spy,
        config=OutcomeConfig(),
    )
    assert winner["path_3w"] == "fast_winner_3w"
    assert winner["fast_winner_3w"] == 1
    assert winner["stop_first_3w"] == 0
    assert stop["path_3w"] == "stop_first_3w"
    assert stop["stop_first_3w"] == 1
    assert stop["fast_winner_3w"] == 0


def test_same_bar_target_stop_is_ambiguous_and_not_resolved():
    result = evaluate_trigger_path(
        price_frame(path="ambiguous"),
        pd.Timestamp("2024-01-02"),
        trigger_price=100.0,
        spy_prices=spy_frame(),
    )
    assert result["path_3w"] == "ambiguous_3w"
    assert result["ambiguous_3w"] == 1
    assert result["fast_winner_3w"] == 0
    assert result["stop_first_3w"] == 0


def test_w1_w4_and_mae_mfe_are_measured_from_executable_entry():
    result = evaluate_trigger_path(
        price_frame(path="winner"),
        pd.Timestamp("2024-01-02"),
        trigger_price=100.0,
        spy_prices=spy_frame(),
    )
    assert result["entry_method"] == "intraday_trigger"
    assert result["entry_price"] == pytest.approx(100.0)
    assert result["entry_delay_sessions"] == 1
    assert result["return_w1"] == pytest.approx(price_frame(path="winner").iloc[5]["Close"] / 100.0 - 1.0)
    assert result["mae_3w"] <= -0.08
    assert result["mfe_3w"] >= 0.20
    assert "return_w2" in result and "return_w3" in result and "return_w4" in result


def test_unresolved_remains_in_probability_denominator_and_ambiguous_is_excluded():
    frame = pd.DataFrame(
        {
            "ambiguous_3w": [0, 0, 0, 1],
            "fast_winner_3w": [1, 0, 0, 0],
            "stop_first_3w": [0, 1, 0, 0],
            "unresolved_3w": [0, 0, 1, 0],
            "stop_first_then_winner_12w": [0, 0, 0, 0],
            "return_w1": [0.1, -0.1, 0.0, 0.0],
            "return_w2": [0.1, -0.1, 0.0, 0.0],
            "return_w3": [0.2, -0.1, 0.0, 0.0],
            "return_w4": [0.2, -0.1, 0.0, 0.0],
            "excess_w1": [0.1, -0.1, 0.0, 0.0],
            "excess_w2": [0.1, -0.1, 0.0, 0.0],
            "excess_w3": [0.2, -0.1, 0.0, 0.0],
            "excess_w4": [0.2, -0.1, 0.0, 0.0],
            "mae_3w": [-0.02, -0.10, -0.03, -0.04],
            "mfe_3w": [0.25, 0.05, 0.08, 0.10],
            "mae_4w": [-0.03, -0.11, -0.04, -0.05],
            "mfe_4w": [0.30, 0.08, 0.10, 0.12],
        }
    )
    metrics = summarize_path(frame)
    assert metrics["selected_n"] == 4
    assert metrics["evaluable_n"] == 3
    assert metrics["fast_winner_rate"] == pytest.approx(1 / 3)
    assert metrics["stop_first_rate"] == pytest.approx(1 / 3)
    assert metrics["unresolved_rate"] == pytest.approx(1 / 3)


def minimal_path_rows() -> pd.DataFrame:
    rows = []
    snapshots = [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")]
    for snapshot in snapshots:
        rows.extend(
            [
                {
                    "snapshot_date": snapshot,
                    "entry_quarter": "2024Q1",
                    "r3_favorable_regime": 1,
                    "A": 0.9,
                    "ambiguous_3w": 0,
                    "fast_winner_3w": 1,
                    "stop_first_3w": 0,
                    "unresolved_3w": 0,
                    "stop_first_then_winner_12w": 0,
                    "return_w1": 0.08,
                    "return_w2": 0.12,
                    "return_w3": 0.22,
                    "return_w4": 0.25,
                    "excess_w1": 0.06,
                    "excess_w2": 0.10,
                    "excess_w3": 0.18,
                    "excess_w4": 0.20,
                    "mae_3w": -0.03,
                    "mfe_3w": 0.25,
                    "mae_4w": -0.04,
                    "mfe_4w": 0.30,
                },
                {
                    "snapshot_date": snapshot,
                    "entry_quarter": "2024Q1",
                    "r3_favorable_regime": 1,
                    "A": 0.1,
                    "ambiguous_3w": 0,
                    "fast_winner_3w": 0,
                    "stop_first_3w": 1,
                    "unresolved_3w": 0,
                    "stop_first_then_winner_12w": 0,
                    "return_w1": -0.05,
                    "return_w2": -0.08,
                    "return_w3": -0.10,
                    "return_w4": -0.08,
                    "excess_w1": -0.06,
                    "excess_w2": -0.09,
                    "excess_w3": -0.11,
                    "excess_w4": -0.09,
                    "mae_3w": -0.10,
                    "mfe_3w": 0.04,
                    "mae_4w": -0.12,
                    "mfe_4w": 0.06,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_within_snapshot_contrast_controls_market_by_pairing_same_snapshot():
    frame = minimal_path_rows()
    contrast = within_snapshot_feature_contrasts(frame, ["A"])
    row = contrast.loc[(contrast["scope"] == "all") & (contrast["feature"] == "A")].iloc[0]
    assert row["matched_snapshot_count"] == 2
    assert row["winner_gt_stop_pair_probability"] == pytest.approx(1.0)
    assert row["snapshot_median_difference_p50"] == pytest.approx(0.8)


def repeated_quarter_frame(quarters: int = 8, rows_per_quarter: int = 40) -> pd.DataFrame:
    rows = []
    for q_index in range(quarters):
        quarter = f"{2023 + q_index // 4}Q{q_index % 4 + 1}"
        snapshot = pd.Period(quarter, freq="Q").start_time + pd.Timedelta(days=7)
        for i in range(rows_per_quarter):
            high = i >= rows_per_quarter // 2
            fast = int(high and i % 3 != 0)
            stop = int((not high) and i % 3 != 0)
            unresolved = int(not fast and not stop)
            rows.append(
                {
                    "snapshot_date": snapshot + pd.Timedelta(days=i % 3),
                    "entry_quarter": quarter,
                    "r3_favorable_regime": 1,
                    "A": 1.0 if high else 0.0,
                    "B": float(i) / rows_per_quarter,
                    "ambiguous_3w": 0,
                    "fast_winner_3w": fast,
                    "stop_first_3w": stop,
                    "unresolved_3w": unresolved,
                    "stop_first_then_winner_12w": 0,
                    "return_w1": 0.05 if high else -0.03,
                    "return_w2": 0.08 if high else -0.05,
                    "return_w3": 0.12 if high else -0.07,
                    "return_w4": 0.14 if high else -0.06,
                    "excess_w1": 0.04 if high else -0.04,
                    "excess_w2": 0.06 if high else -0.06,
                    "excess_w3": 0.10 if high else -0.08,
                    "excess_w4": 0.11 if high else -0.07,
                    "mae_3w": -0.03 if high else -0.10,
                    "mfe_3w": 0.22 if high else 0.05,
                    "mae_4w": -0.04 if high else -0.11,
                    "mfe_4w": 0.25 if high else 0.07,
                }
            )
    return pd.DataFrame(rows)


def test_stock_interaction_search_forbids_market_features():
    frame = repeated_quarter_frame()
    frame["M_bad"] = 1.0
    with pytest.raises(ValueError, match="M_\\* market features are forbidden"):
        search_stock_interactions(
            frame,
            ["A", "M_bad"],
            min_selected=20,
            min_evaluable=20,
            min_quarter_n=5,
            min_evaluated_quarters=4,
        )


def test_stock_search_finds_high_A_as_winner_and_lower_stop_risk():
    frame = repeated_quarter_frame()
    scored, audit = search_stock_interactions(
        frame,
        ["A", "B"],
        min_selected=20,
        min_evaluable=20,
        min_quarter_n=5,
        min_evaluated_quarters=4,
    )
    assert audit["generated_stock_condition_count"] >= 4
    assert not scored.empty
    high_a = scored.loc[scored["rule_json"].str.contains('"feature":"A"')]
    assert (high_a["fast_winner_lift"] > 0).any()
    assert (high_a["stop_first_reduction"] > 0).any()


def test_matched_selected_vs_unselected_reports_same_snapshot_stock_edge():
    frame = repeated_quarter_frame()
    rule = '{"all":[{"feature":"A","op":"==","threshold":1.0}]}'
    matched = matched_selected_vs_unselected(frame, rule)
    assert matched["matched_snapshot_count"] > 0
    assert matched["matched_fast_winner_lift_p50"] > 0
    assert matched["matched_stop_first_reduction_p50"] > 0
    assert matched["matched_path_edge_p50"] > 0


def test_rolling_stock_selection_uses_only_prior_quarters():
    frame = repeated_quarter_frame(quarters=8, rows_per_quarter=40)
    rolling = rolling_stock_selection(
        frame,
        ["A", "B"],
        min_train_quarters=4,
        min_quarter_n=5,
    )
    assert not rolling.empty
    assert (~rolling["test_quarter_in_train"].astype(bool)).all()
    for _, row in rolling.iterrows():
        assert row["test_quarter"] > row["train_end_quarter"]
