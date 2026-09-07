from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.blind_rule_discovery.trigger_path_characterization_r4 import (
    feature_bin_characterization,
    interaction_views,
    search_stock_interactions,
    summarize_path,
)
from backtest.blind_rule_discovery.trigger_path_characterization_r4_runner import (
    rolling_stock_selection,
    summarize_rolling,
)


def row(*, fast: int, stop: int, unresolved: int, recovered: int, value: float, quarter: str, snapshot: pd.Timestamp, favorable: int = 1) -> dict:
    return {
        "snapshot_date": snapshot,
        "entry_quarter": quarter,
        "r3_favorable_regime": favorable,
        "A": value,
        "B": value * 0.5,
        "ambiguous_3w": 0,
        "fast_winner_3w": fast,
        "stop_first_3w": stop,
        "unresolved_3w": unresolved,
        "stop_first_then_winner_12w": recovered,
        "return_w1": 0.08 if fast else (-0.05 if stop else 0.01),
        "return_w2": 0.12 if fast else (-0.08 if stop else 0.02),
        "return_w3": 0.22 if fast else (-0.10 if stop else 0.03),
        "return_w4": 0.25 if fast else (-0.08 if stop else 0.04),
        "excess_w1": 0.06 if fast else (-0.06 if stop else 0.00),
        "excess_w2": 0.10 if fast else (-0.09 if stop else 0.01),
        "excess_w3": 0.18 if fast else (-0.11 if stop else 0.02),
        "excess_w4": 0.20 if fast else (-0.09 if stop else 0.03),
        "mae_3w": -0.03 if fast else (-0.10 if stop else -0.04),
        "mfe_3w": 0.25 if fast else (0.05 if stop else 0.08),
        "mae_4w": -0.04 if fast else (-0.12 if stop else -0.05),
        "mfe_4w": 0.30 if fast else (0.06 if stop else 0.10),
    }


def repeated_frame(quarters: int = 8, rows_per_quarter: int = 40, *, favorable_from: int = 0) -> pd.DataFrame:
    rows = []
    for q_index in range(quarters):
        quarter = f"{2023 + q_index // 4}Q{q_index % 4 + 1}"
        start = pd.Period(quarter, freq="Q").start_time + pd.Timedelta(days=7)
        favorable = int(q_index >= favorable_from)
        for i in range(rows_per_quarter):
            high = i >= rows_per_quarter // 2
            fast = int(high and i % 3 != 0)
            stop = int((not high) and i % 3 != 0)
            unresolved = int(not fast and not stop)
            rows.append(
                row(
                    fast=fast,
                    stop=stop,
                    unresolved=unresolved,
                    recovered=int(stop and i % 5 == 0),
                    value=1.0 if high else 0.0,
                    quarter=quarter,
                    snapshot=start + pd.Timedelta(days=i % 3),
                    favorable=favorable,
                )
            )
    return pd.DataFrame(rows)


def test_summarize_path_keeps_unresolved_and_reports_full_w1_w4_quantiles():
    frame = pd.DataFrame(
        [
            row(fast=1, stop=0, unresolved=0, recovered=0, value=1.0, quarter="2024Q1", snapshot=pd.Timestamp("2024-01-05")),
            row(fast=0, stop=1, unresolved=0, recovered=1, value=0.0, quarter="2024Q1", snapshot=pd.Timestamp("2024-01-05")),
            row(fast=0, stop=1, unresolved=0, recovered=0, value=0.0, quarter="2024Q1", snapshot=pd.Timestamp("2024-01-12")),
            row(fast=0, stop=0, unresolved=1, recovered=0, value=0.5, quarter="2024Q1", snapshot=pd.Timestamp("2024-01-12")),
        ]
    )
    metrics = summarize_path(frame)
    assert metrics["evaluable_n"] == 4
    assert metrics["fast_winner_rate"] == pytest.approx(0.25)
    assert metrics["stop_first_rate"] == pytest.approx(0.50)
    assert metrics["unresolved_rate"] == pytest.approx(0.25)
    assert metrics["stop_first_then_winner_12w_rate"] == pytest.approx(0.50)
    assert metrics["persistent_stop_first_n"] == 1
    assert metrics["persistent_share_of_stop_first"] == pytest.approx(0.50)
    for week in ("w1", "w2", "w3", "w4"):
        for q in ("p25", "p50", "p75"):
            assert f"return_{week}_{q}" in metrics
            assert f"excess_{week}_{q}" in metrics


def test_feature_bins_report_quarter_stability_for_winner_and_loser_extremes():
    frame = repeated_frame()
    bins, extremes = feature_bin_characterization(frame, ["A"], min_quarter_n=5)
    assert not bins.empty
    assert (bins["evaluated_quarters"] >= 2).any()
    assert "positive_path_edge_quarter_fraction" in bins.columns
    assert "higher_stop_risk_quarter_fraction" in bins.columns
    all_extreme = extremes.loc[(extremes["scope"] == "all") & (extremes["feature"] == "A")].iloc[0]
    assert all_extreme["winner_fast_lift"] > 0
    assert all_extreme["loser_stop_first_lift"] > 0


def test_stock_search_is_stock_only_and_separate_winner_loser_views_exist():
    frame = repeated_frame()
    scored, audit = search_stock_interactions(
        frame,
        ["A", "B"],
        min_selected=20,
        min_evaluable=20,
        min_quarter_n=5,
        min_evaluated_quarters=4,
    )
    assert not scored.empty
    assert audit["distinct_feature_pair_count_tested"] > 0
    winner, loser = interaction_views(scored)
    assert not winner.empty and not loser.empty
    assert winner.iloc[0]["fast_winner_lift"] >= winner.iloc[-1]["fast_winner_lift"]
    assert loser.iloc[0]["stop_first_lift"] >= loser.iloc[-1]["stop_first_lift"]

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


def test_rolling_preserves_folds_when_favorable_training_is_insufficient():
    # Favorable regime exists only in the last two quarters; early favorable training
    # therefore has insufficient support and must still appear as explicit folds.
    frame = repeated_frame(quarters=8, rows_per_quarter=40, favorable_from=6)
    rolling = rolling_stock_selection(frame, ["A", "B"], min_train_quarters=4, min_quarter_n=5)
    all_scope = rolling.loc[rolling["scope"] == "all"]
    favorable = rolling.loc[rolling["scope"] == "r3_favorable"]
    assert len(all_scope) == 4
    assert len(favorable) == 4
    assert (favorable["train_insufficient"] == 1).any()
    assert (~rolling["test_quarter_in_train"].astype(bool)).all()
    for _, item in rolling.iterrows():
        assert item["test_quarter"] > item["train_end_quarter"]


def test_rolling_summary_reports_selected_fold_baseline_separately():
    rolling = pd.DataFrame(
        [
            {
                "scope": "all",
                "test_selected_n": 10,
                "test_evaluable_n": 10,
                "test_fast_winner_n": 4,
                "test_stop_first_n": 2,
                "test_baseline_evaluable_n": 20,
                "test_baseline_fast_winner_n": 6,
                "test_baseline_stop_first_n": 6,
                "test_path_edge": 0.20,
                "test_matched_path_edge_p50": 0.10,
                "train_insufficient": 0,
            },
            {
                "scope": "all",
                "test_selected_n": 0,
                "test_evaluable_n": 0,
                "test_fast_winner_n": 0,
                "test_stop_first_n": 0,
                "test_baseline_evaluable_n": 100,
                "test_baseline_fast_winner_n": 10,
                "test_baseline_stop_first_n": 40,
                "test_path_edge": np.nan,
                "test_matched_path_edge_p50": np.nan,
                "train_insufficient": 1,
            },
        ]
    )
    summary = summarize_rolling(rolling)["all"]
    assert summary["folds"] == 2
    assert summary["insufficient_training_folds"] == 1
    assert summary["zero_selection_folds"] == 1
    assert summary["pooled_baseline_fast_winner_rate_all_folds"] != summary["pooled_baseline_fast_winner_rate_selected_folds"]
    assert summary["pooled_baseline_fast_winner_rate_selected_folds"] == pytest.approx(0.30)
    assert summary["pooled_baseline_stop_first_rate_selected_folds"] == pytest.approx(0.30)
