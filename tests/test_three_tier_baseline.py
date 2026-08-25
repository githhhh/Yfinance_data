"""Unit tests for Phase 2 Stage 1: Three-Tier Baseline (L0/L1/L2) and Alpha Decoupling."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backtest.b0_top3_quality_audit.three_tier_baseline import (
    sample_l0_top3,
    sample_l1_top3_with_industry_dedup,
    compute_portfolio_metrics,
    run_three_tier_baseline,
)


@pytest.fixture
def sample_events_and_weekly():
    root_dir = Path(__file__).resolve().parents[1] / "backtest" / "b0_top3_quality_audit"
    events_path = root_dir / "data" / "candidate_event_outcomes.parquet"
    weekly_path = root_dir / "data" / "candidate_weekly_outcomes.parquet"
    
    assert events_path.exists(), f"Missing {events_path}"
    assert weekly_path.exists(), f"Missing {weekly_path}"
    
    events_df = pd.read_parquet(events_path)
    weekly_df = pd.read_parquet(weekly_path)
    return events_df, weekly_df


def test_01_l0_sampling_properties():
    rng = np.random.default_rng(42)
    codes = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL"]
    
    # Draw 3 without replacement
    sampled = sample_l0_top3(codes, pick_limit=3, rng=rng)
    assert len(sampled) == 3
    assert len(set(sampled)) == 3
    assert all(c in codes for c in sampled)
    
    # Draw from small pool (2 candidates)
    sampled_small = sample_l0_top3(["AAPL", "NVDA"], pick_limit=3, rng=rng)
    assert len(sampled_small) == 2
    assert set(sampled_small) == {"AAPL", "NVDA"}
    
    # Draw from empty pool
    assert sample_l0_top3([], pick_limit=3, rng=rng) == []


def test_02_l1_sampling_industry_dedup():
    rng = np.random.default_rng(42)
    df = pd.DataFrame([
        {"code": "AAPL", "industry": "Consumer Electronics"},
        {"code": "DELL", "industry": "Consumer Electronics"},
        {"code": "HPQ", "industry": "Consumer Electronics"},
        {"code": "NVDA", "industry": "Semiconductors"},
        {"code": "AMD", "industry": "Semiconductors"},
        {"code": "MSFT", "industry": "Software - Infrastructure"},
    ])
    
    # Sample 100 times, ensure every sample has unique industries
    for _ in range(100):
        sampled = sample_l1_top3_with_industry_dedup(df, pick_limit=3, rng=rng)
        assert len(sampled) == 3
        
        # Check industries
        sample_df = df[df["code"].isin(sampled)]
        industries = sample_df["industry"].tolist()
        assert len(set(industries)) == 3, f"Duplicate industry found: {industries}"


def test_03_portfolio_metrics_equal_weighted():
    event_lookup = {
        "AAA": {
            "executed_return_to_asof_pct": 10.0,
            "stop8_before_profit20": False,
            "stop_8_hit_ever": False,
        },
        "BBB": {
            "executed_return_to_asof_pct": -8.0,
            "stop8_before_profit20": True,
            "stop_8_hit_ever": True,
        },
    }
    weekly_lookup = {
        ("AAA", 1): {"week_close_return_from_entry_pct": 5.0},
        ("BBB", 1): {"week_close_return_from_entry_pct": -4.0},
    }
    
    m = compute_portfolio_metrics(["AAA", "BBB"], event_lookup, weekly_lookup, "2026-01-09")
    assert m["is_portfolio_valid"] is True
    assert pytest.approx(m["executed_return"]) == 1.0  # (10 + -8)/2
    assert pytest.approx(m["w1_return"]) == 0.5        # (5 + -4)/2
    assert pytest.approx(m["stop8_before_profit20"]) == 0.5
    assert pytest.approx(m["stop_8_hit_ever"]) == 0.5
    assert m["picks_count"] == 2
    assert m["valid_picks_count"] == 2

    # Censored portfolio protocol: if any pick has missing outcome, portfolio is invalid
    m_censored = compute_portfolio_metrics(["AAA", "MISSING_STOCK"], event_lookup, weekly_lookup, "2026-01-09")
    assert m_censored["is_portfolio_valid"] is False
    assert np.isnan(m_censored["executed_return"])
    assert m_censored["picks_count"] == 2
    assert m_censored["valid_picks_count"] == 1


def test_04_three_tier_baseline_regression(sample_events_and_weekly):
    events_df, weekly_df = sample_events_and_weekly
    weekly_comp_df, summary_df, stats_meta = run_three_tier_baseline(
        events_df, weekly_df, n_draws=100, seed=42, pick_limit=3
    )
    
    # 1. Total evaluation weeks invariant
    assert stats_meta["total_weeks"] == 40
    assert stats_meta["total_b0_recommendations"] == 97
    assert stats_meta["full_top3_weeks"] == 25
    assert stats_meta["active_weeks"] == 40
    
    # 2. Check summary DataFrame format (W1, W2, W4, As-Of)
    assert len(summary_df) == 4
    assert "Week 1 Executed Return" in summary_df["metric"].values
    assert "Week 2 Executed Return" in summary_df["metric"].values
    assert "Week 4 Executed Return" in summary_df["metric"].values
    assert "Executed Return (to As-Of Secondary)" in summary_df["metric"].values
    
    # 3. Check Alpha math and maturity accounting
    for _, row in summary_df.iterrows():
        assert "mature_eval_weeks" in row
        assert row["mature_eval_weeks"] <= row["total_calendar_weeks"]
        assert row["win_rate_l2_vs_l0_pct"] >= 0.0
        assert row["win_rate_l2_vs_l1_pct"] >= 0.0
        assert row["win_rate_l1_vs_l0_pct"] >= 0.0
