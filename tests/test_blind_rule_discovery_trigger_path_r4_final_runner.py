from __future__ import annotations

import pandas as pd

from backtest.blind_rule_discovery.trigger_path_characterization_r4_final_runner import (
    rolling_stock_selection,
    summarize_rolling,
)


def frame(quarters: int = 8, rows_per_quarter: int = 40, favorable_from: int = 6) -> pd.DataFrame:
    rows = []
    for q_index in range(quarters):
        quarter = f"{2023 + q_index // 4}Q{q_index % 4 + 1}"
        snapshot = pd.Period(quarter, freq="Q").start_time + pd.Timedelta(days=8)
        favorable = int(q_index >= favorable_from)
        for i in range(rows_per_quarter):
            high = i >= rows_per_quarter // 2
            fast = int(high and i % 3 != 0)
            stop = int((not high) and i % 3 != 0)
            unresolved = int(not fast and not stop)
            rows.append(
                {
                    "snapshot_date": snapshot + pd.Timedelta(days=i % 3),
                    "entry_quarter": quarter,
                    "r3_favorable_regime": favorable,
                    "A": 1.0 if high else 0.0,
                    "B": i / rows_per_quarter,
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


def test_final_runner_preserves_every_fold_and_never_trains_on_test_quarter():
    data = frame()
    rolling = rolling_stock_selection(data, ["A", "B"], min_train_quarters=4, min_quarter_n=5)
    for scope in ("all", "r3_favorable"):
        scoped = rolling.loc[rolling["scope"] == scope]
        assert len(scoped) == 4
    favorable = rolling.loc[rolling["scope"] == "r3_favorable"]
    assert (favorable["train_insufficient"] == 1).any()
    assert (~rolling["test_quarter_in_train"].astype(bool)).all()
    assert all(row.test_quarter > row.train_end_quarter for row in rolling.itertuples())


def test_final_runner_summary_keeps_insufficient_and_zero_selection_in_denominator():
    data = frame()
    rolling = rolling_stock_selection(data, ["A", "B"], min_train_quarters=4, min_quarter_n=5)
    summary = summarize_rolling(rolling)["r3_favorable"]
    assert summary["folds"] == 4
    assert summary["insufficient_training_folds"] >= 1
    assert summary["zero_selection_folds"] >= summary["insufficient_training_folds"]
    assert summary["insufficient_training_fraction"] > 0
    assert "pooled_baseline_fast_winner_rate_selected_folds" in summary
    assert "pooled_baseline_stop_first_rate_selected_folds" in summary
    assert "pooled_matched_path_edge_selected_folds" in summary
