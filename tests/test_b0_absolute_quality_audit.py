from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.b0_absolute_quality_audit.audit import (
    _simple_baseline_codes,
    _winner_gate_row,
    raw_fixed_capacity_k,
)
from backtest.b0_absolute_quality_audit.config import (
    PROTOCOL_VERSION,
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
from backtest.b0_absolute_quality_audit.market_data import (
    build_next_open_forward_returns,
    download_yahoo_supplement,
    find_codes_needing_yahoo,
    spy_momentum_asof,
)
from backtest.b0_absolute_quality_audit.capacity import (
    fill_capacity_codes,
    underfill_cause,
)
from backtest.b0_absolute_quality_audit.v11 import (
    _rejection_summaries,
    active_choice_eligible_rows,
)
from backtest.b0_absolute_quality_audit.diagnostics import (
    capacity_pick_quality,
    momentum_gate_diagnostics,
    support_calendar_summary,
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



def test_next_open_outcome_uses_first_session_after_snapshot_and_entry_day_low():
    panel = pd.DataFrame([
        {"snapshot_date": "2026-01-02", "code": "AAA"},
    ])
    prices = pd.DataFrame([
        {
            "code": "AAA", "date": "2026-01-02",
            "open": 99.0, "high": 101.0, "low": 80.0, "close": 100.0,
            "volume": 1, "source": "test",
        },
        {
            "code": "AAA", "date": "2026-01-05",
            "open": 101.0, "high": 103.0, "low": 92.0, "close": 102.0,
            "volume": 1, "source": "test",
        },
        {
            "code": "AAA", "date": "2026-01-30",
            "open": 109.0, "high": 111.0, "low": 108.0, "close": 110.0,
            "volume": 1, "source": "test",
        },
        {
            "code": "AAA", "date": "2026-02-02",
            "open": 111.0, "high": 112.0, "low": 110.0, "close": 111.0,
            "volume": 1, "source": "test",
        },
    ])
    out = build_next_open_forward_returns(panel, prices, extra_codes=())
    row = out.iloc[0]
    assert row["next_open_entry_date"] == "2026-01-05"
    assert row["next_open_end_date"] == "2026-02-02"
    assert row["next_open_w4_return_pct"] == round((111.0 / 101.0 - 1.0) * 100.0, 6)
    # Snapshot-day low=80 is ignored; entry-session low=92 is below 101*0.92.
    assert bool(row["next_open_w4_stop8"]) is True


def test_next_open_audit_freezes_aug_7_as_immature_at_frozen_asof():
    panel = pd.DataFrame([
        {"snapshot_date": "2026-08-07", "code": "AAA"},
    ])
    prices = pd.DataFrame([
        {
            "code": "AAA", "date": "2026-08-10",
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            "volume": 1, "source": "test",
        },
        {
            "code": "AAA", "date": "2026-09-03",
            "open": 105.0, "high": 106.0, "low": 104.0, "close": 105.0,
            "volume": 1, "source": "test",
        },
    ])
    out = build_next_open_forward_returns(panel, prices, extra_codes=())
    row = out.iloc[0]
    assert bool(row["next_open_price_valid"]) is False
    assert row["next_open_invalid_reason"] == "HORIZON_NOT_MATURE_AS_OF"


def test_yahoo_need_detection_always_includes_spy_qqq_and_missing_code():
    panel = pd.DataFrame([
        {"snapshot_date": "2026-01-02", "code": "MISS"},
    ])
    base = pd.DataFrame(
        columns=["code", "date", "open", "high", "low", "close", "volume", "source"]
    )
    needed = find_codes_needing_yahoo(panel, base)
    assert "MISS" in needed
    assert "SPY" in needed
    assert "QQQ" in needed


def _capacity_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "code": "A", "industry": "I1",
            "current_b0_selected": True, "current_b0_pick_order": 1,
            "current_b0_eligible": True, "current_b0_raw_rank": 1,
            "current_b0_reject_reasons": "",
        },
        {
            "code": "B", "industry": "I1",
            "current_b0_selected": False, "current_b0_pick_order": None,
            "current_b0_eligible": True, "current_b0_raw_rank": 2,
            "current_b0_reject_reasons": "",
        },
        {
            "code": "C", "industry": "I2",
            "current_b0_selected": False, "current_b0_pick_order": None,
            "current_b0_eligible": False, "current_b0_raw_rank": 3,
            "current_b0_reject_reasons": "eps_unknown",
        },
        {
            "code": "D", "industry": "I3",
            "current_b0_selected": False, "current_b0_pick_order": None,
            "current_b0_eligible": False, "current_b0_raw_rank": 4,
            "current_b0_reject_reasons": "non_actionable",
        },
        {
            "code": "E", "industry": "I4",
            "current_b0_selected": False, "current_b0_pick_order": None,
            "current_b0_eligible": False, "current_b0_raw_rank": 5,
            "current_b0_reject_reasons": "non_actionable|eps_unknown",
        },
    ])


def test_fill3_policies_never_replace_original_b0_pick():
    frame = _capacity_frame()
    for policy in [
        "B0_FILL3_RELAX_INDUSTRY",
        "B0_FILL3_EPS_ONLY",
        "B0_FILL3_SINGLE_REJECT",
        "B0_FILL3_ANY_REJECT",
    ]:
        picks = fill_capacity_codes(frame, policy)
        assert picks[0] == "A"
        assert "A" in picks


def test_fill3_relax_industry_changes_only_industry_constraint():
    picks = fill_capacity_codes(_capacity_frame(), "B0_FILL3_RELAX_INDUSTRY")
    assert picks == ["A", "B"]


def test_fill3_eps_only_accepts_only_sole_eps_reject():
    picks = fill_capacity_codes(_capacity_frame(), "B0_FILL3_EPS_ONLY")
    assert picks == ["A", "C"]
    assert "E" not in picks


def test_fill3_single_reject_can_fill_three_but_excludes_multi_failure():
    picks = fill_capacity_codes(_capacity_frame(), "B0_FILL3_SINGLE_REJECT")
    assert picks == ["A", "C", "D"]
    assert "E" not in picks


def test_underfill_cause_distinguishes_eligibility_shortage_from_industry():
    assert underfill_cause(_capacity_frame()) == "ELIGIBILITY_SHORTAGE"

    frame = pd.DataFrame([
        {
            "code": "A", "industry": "I1",
            "current_b0_selected": True, "current_b0_pick_order": 1,
            "current_b0_eligible": True, "current_b0_raw_rank": 1,
            "current_b0_reject_reasons": "",
        },
        {
            "code": "B", "industry": "I1",
            "current_b0_selected": False, "current_b0_pick_order": None,
            "current_b0_eligible": True, "current_b0_raw_rank": 2,
            "current_b0_reject_reasons": "",
        },
        {
            "code": "C", "industry": "I1",
            "current_b0_selected": False, "current_b0_pick_order": None,
            "current_b0_eligible": True, "current_b0_raw_rank": 3,
            "current_b0_reject_reasons": "",
        },
    ])
    assert underfill_cause(frame) == "INDUSTRY_CONSTRAINT"


def test_rejection_summary_separates_exclusive_from_overlap():
    events = pd.DataFrame([
        {
            "snapshot_date": "2026-01-02", "code": "A",
            "reasons": "eps_unknown", "reason_count": 1,
            "exclusive_reason": "eps_unknown",
            "next_open_w4_return_pct": 10.0, "next_open_w4_stop8": False,
            "is_top20_winner": True, "is_big_winner": False,
        },
        {
            "snapshot_date": "2026-01-02", "code": "B",
            "reasons": "eps_unknown|non_actionable", "reason_count": 2,
            "exclusive_reason": "",
            "next_open_w4_return_pct": 20.0, "next_open_w4_stop8": False,
            "is_top20_winner": True, "is_big_winner": True,
        },
    ])
    exclusive, overlap, combos = _rejection_summaries(events)
    eps_exclusive = exclusive[exclusive["exclusive_reason"] == "eps_unknown"].iloc[0]
    eps_overlap = overlap[overlap["reason"] == "eps_unknown"].iloc[0]
    assert eps_exclusive["candidate_events"] == 1
    assert eps_overlap["label_events"] == 2
    assert eps_overlap["multi_reason_rate"] == 0.5
    assert set(combos["reasons"]) == {"eps_unknown", "eps_unknown|non_actionable"}



def test_yahoo_benchmark_download_fails_closed(tmp_path):
    class EmptyProvider:
        def download_batch_stocks(self, symbols, period="1y", interval="1d"):
            return {}, list(symbols)

    panel = pd.DataFrame([
        {"snapshot_date": "2026-01-02", "code": "AAA"},
    ])
    base = pd.DataFrame(
        columns=["code", "date", "open", "high", "low", "close", "volume", "source"]
    )

    import pytest
    with pytest.raises(RuntimeError, match="Yahoo benchmark download failed"):
        download_yahoo_supplement(
            panel,
            base,
            provider=EmptyProvider(),
            supplement_path=tmp_path / "supp.parquet",
            audit_path=tmp_path / "audit.csv",
        )



def test_active_choice_headline_excludes_single_feasible_portfolio_week():
    frame = pd.DataFrame([
        {
            "snapshot_date": "2026-01-02",
            "primary_valid": True,
            "active_choice": False,
            "feasible_portfolio_count": 1,
            "b0_percentile": 100.0,
        },
        {
            "snapshot_date": "2026-01-09",
            "primary_valid": True,
            "active_choice": True,
            "feasible_portfolio_count": 10,
            "b0_percentile": 60.0,
        },
    ])
    active = active_choice_eligible_rows(frame)
    assert active["snapshot_date"].tolist() == ["2026-01-09"]
    assert active["b0_percentile"].tolist() == [60.0]



def test_protocol_v12():
    assert PROTOCOL_VERSION == "b0_absolute_quality_v1_2"


def test_capacity_pick_quality_separates_pick_risk_from_position_count():
    panel = pd.DataFrame([
        {
            "snapshot_date": "2026-01-02", "code": "A",
            "next_open_price_valid": True,
            "next_open_w4_return_pct": 10.0,
            "next_open_w4_stop8": False,
            "current_b0_reject_reasons": "",
        },
        {
            "snapshot_date": "2026-01-02", "code": "B",
            "next_open_price_valid": True,
            "next_open_w4_return_pct": 5.0,
            "next_open_w4_stop8": False,
            "current_b0_reject_reasons": "non_actionable",
        },
        {
            "snapshot_date": "2026-01-02", "code": "C",
            "next_open_price_valid": True,
            "next_open_w4_return_pct": -10.0,
            "next_open_w4_stop8": True,
            "current_b0_reject_reasons": "non_actionable",
        },
    ])
    weekly = pd.DataFrame([
        {
            "snapshot_date": "2026-01-02",
            "policy_id": "B0_FILL3_SINGLE_REJECT",
            "original_pick_count": 1,
            "mature": True,
            "original_codes": '["A"]',
            "added_codes": '["B", "C"]',
        },
    ])

    summary, reasons = capacity_pick_quality(panel, weekly)
    original = summary[summary["cohort"] == "original_b0"].iloc[0]
    added = summary[summary["cohort"] == "added_fill"].iloc[0]

    assert original["picks"] == 1
    assert original["stop8_rate"] == 0.0
    assert original["terminal_le_minus8_rate"] == 0.0
    assert added["picks"] == 2
    assert added["stop8_rate"] == 0.5
    assert added["terminal_le_minus8_rate"] == 0.5
    assert reasons.iloc[0]["reject_reason"] == "non_actionable"


def test_support_calendar_exposes_quarter_concentration():
    snapshots = [
        "2025-12-19", "2025-12-26",
        "2026-01-02", "2026-01-09",
        "2026-04-03",
    ]
    raw = pd.DataFrame([
        {"snapshot_date": d, "primary_valid": d != "2026-04-03"}
        for d in snapshots
    ])
    simple = pd.DataFrame([
        {
            "snapshot_date": d,
            "baseline": "momentum_20",
            "primary_valid": d in {"2025-12-19", "2026-01-02"},
        }
        for d in snapshots
    ])
    out = support_calendar_summary(snapshots, raw, simple)

    q1 = out[
        (out["comparison"] == "simple_momentum_20")
        & (out["quarter"] == "2026Q1")
    ].iloc[0]
    assert q1["support_weeks"] == 1
    assert q1["total_snapshots"] == 2
    assert q1["support_rate"] == 0.5


def test_momentum_gate_diagnostic_locates_incremental_picks_outside_gate():
    panel = pd.DataFrame([
        {
            "snapshot_date": "2026-01-02", "code": "A",
            "current_b0_eligible": True, "current_b0_selected": True,
            "current_b0_reject_reasons": "",
            "next_open_price_valid": True,
            "next_open_w4_return_pct": 10.0,
            "next_open_w4_stop8": False,
        },
        {
            "snapshot_date": "2026-01-02", "code": "B",
            "current_b0_eligible": False, "current_b0_selected": False,
            "current_b0_reject_reasons": "non_actionable",
            "next_open_price_valid": True,
            "next_open_w4_return_pct": 5.0,
            "next_open_w4_stop8": False,
        },
        {
            "snapshot_date": "2026-01-02", "code": "C",
            "current_b0_eligible": False, "current_b0_selected": False,
            "current_b0_reject_reasons": "non_actionable",
            "next_open_price_valid": True,
            "next_open_w4_return_pct": -9.0,
            "next_open_w4_stop8": True,
        },
    ])
    simple = pd.DataFrame([
        {
            "snapshot_date": "2026-01-02",
            "baseline": "momentum_20",
            "primary_valid": True,
            "codes": '["A", "B", "C"]',
        },
    ])

    summary, reasons = momentum_gate_diagnostics(panel, simple)
    eligible = summary[summary["cohort"] == "eligible"].iloc[0]
    outside = summary[summary["cohort"] == "gate_outside"].iloc[0]

    assert eligible["picks"] == 1
    assert eligible["share_of_momentum_picks"] == 1 / 3
    assert eligible["selected_by_b0_rate"] == 1.0
    assert outside["picks"] == 2
    assert outside["share_of_momentum_picks"] == 2 / 3
    assert reasons.iloc[0]["reject_reason"] == "non_actionable"


def test_spy_momentum_asof_uses_only_snapshot_or_earlier_bars():
    dates = pd.date_range("2026-01-01", periods=22, freq="D")
    closes = np.arange(100.0, 122.0)
    prices = pd.DataFrame({
        "code": "SPY",
        "date": dates,
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": 1,
        "source": "test",
    })
    # Add a huge future move that must not affect the snapshot value.
    prices.loc[len(prices)] = {
        "code": "SPY",
        "date": pd.Timestamp("2026-02-15"),
        "open": 1000.0, "high": 1000.0, "low": 1000.0, "close": 1000.0,
        "volume": 1, "source": "test",
    }

    snapshot = str(dates[20].date())
    out = spy_momentum_asof(prices, [snapshot], sessions=20)
    assert out.iloc[0]["spy_momentum"] == (120.0 / 100.0 - 1.0)
