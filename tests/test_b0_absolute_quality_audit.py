from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.b0_absolute_quality_audit.audit import (
    _simple_baseline_codes,
    _winner_gate_row,
    raw_fixed_capacity_k,
)
from backtest.b0_absolute_quality_audit.config import (
    RAW_PRICE_COVERAGE_MIN_FOR_PRIMARY,
)
from backtest.b0_absolute_quality_audit.data import (
    build_snapshot_forward_returns,
    current_b0_eligible,
)
from backtest.b0_absolute_quality_audit.metrics import (
    four_offset_nonoverlap,
    safe_spearman,
)
from backtest.b0_absolute_quality_audit.portfolio import (
    distribution_summary,
    exact_portfolio_distribution,
    greedy_oracle_codes,
    percentile_rank,
)


def _production_like_row(**overrides) -> pd.Series:
    row = {
        "snapshot_date": "2026-01-02",
        "code": "TEST",
        "signal": True,
        "industry": "Industry A",
        "ibd_entry_status": "ACTIONABLE",
        "ibd_candidate_rule": "ceiling_breakout",
        "current_vs_ibd_candidate_pct": 1.0,
        "ibd_entry_volume_ratio": 1.7,
        "volume_ratio": 1.1,
        "eps_yoy_growth": 30.0,
        "dist_to_52w_high_pct": -2.0,
        "ibd_entry_close_position": 0.85,
        "ibd_entry_breakout_range_ratio": 0.60,
        "pullback_v_is_dry": None,
        # Deliberately stale/incorrect helper fields: audit must ignore them.
        "b0_eligible": False,
        "is_b0": 0,
    }
    row.update(overrides)
    return pd.Series(row)


def test_raw_primary_requires_full_price_coverage():
    assert RAW_PRICE_COVERAGE_MIN_FOR_PRIMARY == 1.0


def test_current_eligibility_recomputed_not_old_panel_helper():
    row = _production_like_row(b0_eligible=False)
    assert current_b0_eligible(row, 0) is True


def test_current_production_does_not_hard_reject_freshness_missing():
    row = _production_like_row(current_vs_ibd_candidate_pct=None)
    assert current_b0_eligible(row, 0) is True


def test_current_production_rejects_below_candidate():
    row = _production_like_row(current_vs_ibd_candidate_pct=-0.1)
    assert current_b0_eligible(row, 0) is False


def _combo_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"code": "A", "industry": "I1", "ret": 3.0},
        {"code": "B", "industry": "I1", "ret": 6.0},
        {"code": "C", "industry": "I2", "ret": 9.0},
    ])


def test_exact_distribution_unconstrained_is_hand_checkable():
    dist = exact_portfolio_distribution(
        _combo_frame(),
        k=2,
        return_col="ret",
        distinct_industry=False,
    )
    # Capital adjusted by 3 slots: (3+6)/3=3, (3+9)/3=4, (6+9)/3=5.
    assert np.allclose(np.sort(dist), np.array([3.0, 4.0, 5.0]))


def test_exact_distribution_distinct_industry_removes_same_industry_pair():
    dist = exact_portfolio_distribution(
        _combo_frame(),
        k=2,
        return_col="ret",
        distinct_industry=True,
    )
    assert np.allclose(np.sort(dist), np.array([4.0, 5.0]))


def test_oracle_distinct_industry_is_exact_greedy_for_additive_topk():
    codes = greedy_oracle_codes(
        _combo_frame(),
        k=2,
        return_col="ret",
        distinct_industry=True,
    )
    assert codes == ["C", "B"]


def test_percentile_and_oracle_capture_math():
    dist = np.array([1.0, 2.0, 3.0, 4.0])
    assert percentile_rank(3.0, dist) == 75.0
    summary = distribution_summary(3.0, dist, 4.0)
    assert summary["random_mean"] == 2.5
    assert summary["edge_vs_random_mean"] == 0.5
    assert summary["oracle_capture_ratio"] == round(0.5 / 1.5, 6)


def test_simple_raw_baseline_ignores_b0_state_and_uses_pit_feature():
    frame = pd.DataFrame([
        {
            "code": "A", "industry": "I1",
            "ibd_entry_volume_ratio": 1.1,
            "current_b0_eligible": True, "current_b0_selected": True,
        },
        {
            "code": "B", "industry": "I2",
            "ibd_entry_volume_ratio": 3.0,
            "current_b0_eligible": False, "current_b0_selected": False,
        },
    ])
    picks = _simple_baseline_codes(
        frame,
        k=1,
        feature="ibd_entry_volume_ratio",
        direction="desc",
    )
    assert picks == ["B"]


def test_winner_gate_counts_rejected_future_winner():
    frame = pd.DataFrame([
        {
            "code": "A", "snapshot_price_valid": True,
            "snapshot_w4_return_pct": 30.0,
            "current_b0_eligible": False,
        },
        {
            "code": "B", "snapshot_price_valid": True,
            "snapshot_w4_return_pct": 10.0,
            "current_b0_eligible": True,
        },
        {
            "code": "C", "snapshot_price_valid": True,
            "snapshot_w4_return_pct": -5.0,
            "current_b0_eligible": False,
        },
        {
            "code": "D", "snapshot_price_valid": True,
            "snapshot_w4_return_pct": -10.0,
            "current_b0_eligible": False,
        },
        {
            "code": "E", "snapshot_price_valid": True,
            "snapshot_w4_return_pct": -20.0,
            "current_b0_eligible": False,
        },
    ])
    row = _winner_gate_row(frame, ["B"])
    # top 20% of five names = one winner = A, rejected by eligibility.
    assert row["winner_count"] == 1
    assert row["winner_retention_rate"] == 0.0
    assert row["rejected_winner_count"] == 1
    assert row["primary_valid"] is True


def test_spearman_sign_is_positive_when_lower_rank_predicts_higher_return():
    ranks = pd.Series([1, 2, 3, 4])
    returns = pd.Series([10.0, 5.0, 1.0, -2.0])
    assert safe_spearman(ranks, returns) == 1.0


def test_four_offset_nonoverlap_uses_every_fourth_row():
    frame = pd.DataFrame({
        "snapshot_date": [f"2026-01-{d:02d}" for d in range(1, 9)],
        "x": list(range(8)),
    })
    out = four_offset_nonoverlap(frame, value_col="x")
    assert out.loc[out.offset == 0, "weeks"].iloc[0] == 2
    assert out.loc[out.offset == 0, "value_mean"].iloc[0] == 2.0


def test_snapshot_forward_return_uses_close_and_excludes_same_day_low_from_stop():
    panel = pd.DataFrame([
        {"snapshot_date": "2026-01-02", "code": "AAA"},
    ])
    dates = pd.date_range("2026-01-02", "2026-01-30", freq="D")
    prices = pd.DataFrame({
        "date": dates,
        "code": "AAA",
        "close": np.linspace(100.0, 110.0, len(dates)),
        "low": np.linspace(99.0, 109.0, len(dates)),
    })
    # Snapshot day's intraday low is below -8%, but it happened before snapshot close.
    prices.loc[prices["date"] == pd.Timestamp("2026-01-02"), "low"] = 80.0

    out = build_snapshot_forward_returns(panel, prices, extra_codes=())
    row = out.iloc[0]
    assert row["snapshot_price_valid"]
    assert row["snapshot_w4_return_pct"] == 10.0
    assert bool(row["snapshot_w4_stop8"]) is False



def test_raw_fixed_capacity_is_independent_of_b0_pick_count_and_uses_up_to_three():
    frame = pd.DataFrame([
        {"code": "A", "industry": "I1"},
        {"code": "B", "industry": "I2"},
        {"code": "C", "industry": "I3"},
        {"code": "D", "industry": "I3"},
    ])
    assert raw_fixed_capacity_k(frame) == 3


def test_raw_fixed_capacity_does_not_reject_unknown_industry_metadata():
    frame = pd.DataFrame([
        {"code": "A", "industry": ""},
        {"code": "B", "industry": ""},
        {"code": "C", "industry": "I1"},
    ])
    assert raw_fixed_capacity_k(frame) == 3


def test_snapshot_forward_return_is_not_mature_before_cache_reaches_target():
    panel = pd.DataFrame([
        {"snapshot_date": "2026-01-02", "code": "AAA"},
    ])
    # Cache ends three calendar days before the +28d target. Previous logic
    # would have accepted this as a weekend-like stale end bar; it must fail.
    dates = pd.date_range("2026-01-02", "2026-01-27", freq="D")
    prices = pd.DataFrame({
        "date": dates,
        "code": "AAA",
        "close": np.linspace(100.0, 105.0, len(dates)),
        "low": np.linspace(99.0, 104.0, len(dates)),
    })

    out = build_snapshot_forward_returns(panel, prices, extra_codes=())
    assert bool(out.iloc[0]["snapshot_price_valid"]) is False
    assert pd.isna(out.iloc[0]["snapshot_w4_return_pct"])
