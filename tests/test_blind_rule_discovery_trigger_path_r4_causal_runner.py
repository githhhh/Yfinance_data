from __future__ import annotations

import pandas as pd

from backtest.blind_rule_discovery.trigger_path_characterization_r4_causal_runner import (
    purge_training_rows_for_w3,
    rolling_stock_selection,
    summarize_rolling,
)


def frame(quarters: int = 8, rows_per_quarter: int = 40) -> pd.DataFrame:
    rows = []
    for q_index in range(quarters):
        quarter = f"{2023 + q_index // 4}Q{q_index % 4 + 1}"
        period = pd.Period(quarter, freq="Q")
        snapshot = period.start_time + pd.Timedelta(days=10)
        for i in range(rows_per_quarter):
            high = i >= rows_per_quarter // 2
            fast = int(high and i % 3 != 0)
            stop = int((not high) and i % 3 != 0)
            unresolved = int(not fast and not stop)
            # Most labels finish well within their entry quarter. The last two
            # synthetic rows deliberately cross into the following quarter and
            # therefore must be purged from the next fold's training set.
            exit_w3 = snapshot + pd.Timedelta(days=20)
            if i >= rows_per_quarter - 2:
                exit_w3 = period.end_time.normalize() + pd.Timedelta(days=5)
            rows.append(
                {
                    "snapshot_date": snapshot + pd.Timedelta(days=i % 3),
                    "entry_quarter": quarter,
                    "exit_date_w3": exit_w3,
                    "r3_favorable_regime": 1,
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


def test_purge_removes_w3_labels_that_overlap_test_quarter():
    data = frame(quarters=5)
    raw_train = data.loc[data["entry_quarter"].isin(["2023Q1", "2023Q2", "2023Q3", "2023Q4"])].reset_index(drop=True)
    purged, audit = purge_training_rows_for_w3(raw_train, test_quarter="2024Q1")
    test_start = pd.Timestamp("2024-01-01")
    assert audit["train_rows_purged_for_w3_overlap"] > 0
    assert audit["w3_label_overlap_after_purge"] is False
    assert pd.to_datetime(purged["exit_date_w3"]).max() < test_start


def test_causal_rolling_records_purge_audit_and_never_overlaps_test():
    data = frame()
    rolling = rolling_stock_selection(data, ["A", "B"], min_train_quarters=4, min_quarter_n=5)
    assert not rolling.empty
    assert (~rolling["test_quarter_in_train"].astype(bool)).all()
    assert (~rolling["w3_label_overlap_after_purge"].astype(bool)).all()
    assert (rolling["train_rows_purged_for_w3_overlap"] > 0).any()
    for item in rolling.itertuples():
        if pd.notna(item.train_max_exit_date_w3):
            assert pd.Timestamp(item.train_max_exit_date_w3) < pd.Timestamp(item.test_quarter_start)


def test_causal_summary_reports_purged_rows_and_zero_overlap():
    data = frame()
    rolling = rolling_stock_selection(data, ["A", "B"], min_train_quarters=4, min_quarter_n=5)
    summary = summarize_rolling(rolling)
    for scope in ("all", "r3_favorable"):
        assert summary[scope]["w3_overlap_purge_rows_total"] > 0
        assert summary[scope]["post_purge_overlap_fold_count"] == 0
        assert summary[scope]["folds"] == 4
