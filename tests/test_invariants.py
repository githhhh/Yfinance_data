"""Four Core Invariant Tests for Quantitative Audit and Methodological Integrity."""

from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backtest.b0_top3_quality_audit.eligibility import is_production_eligible_pit
from backtest.b0_top3_quality_audit.evaluate_rule_signatures import evaluate_all_rules
from backtest.b0_top3_quality_audit.pareto_champions import (
    compute_file_sha256,
    export_frozen_rules_manifest,
    select_champions,
)
from backtest.b0_top3_quality_audit.random_control import run_random_top3_for_snapshot
from backtest.b0_top3_quality_audit.skill_rule_engine import build_skill_rule_space
from backtest.b0_top3_quality_audit.three_tier_baseline import compute_portfolio_metrics


@pytest.fixture
def test_datasets():
    root_dir = Path(__file__).resolve().parents[1] / "backtest" / "b0_top3_quality_audit"
    events_df = pd.read_parquet(root_dir / "data" / "candidate_event_outcomes.parquet")
    weekly_df = pd.read_parquet(root_dir / "data" / "candidate_weekly_outcomes.parquet")
    baseline_df = pd.read_csv(root_dir / "output" / "three_tier_weekly_comparison.csv")
    return events_df, weekly_df, baseline_df


def test_01_holdout_isolation_invariant(test_datasets):
    """INVARIANT 1: Corrupting all Holdout weeks (31~40) must produce 100% identical Train Champions."""
    events_df, weekly_df, baseline_df = test_datasets
    all_weeks = sorted(baseline_df["snapshot_date"].unique())
    train_weeks = all_weeks[:30]
    holdout_weeks = all_weeks[30:]

    rules = build_skill_rule_space()[:10]  # Fast subset of 10 rules

    # 1. Clean Run
    clean_eval = evaluate_all_rules(
        events_df, weekly_df, baseline_df, rules, train_weeks, holdout_weeks, pick_limit=3
    )
    clean_champs = select_champions(clean_eval)

    # 2. Corrupted Holdout Run: Mutate all holdout price returns
    corrupted_events = events_df.copy()
    holdout_mask = corrupted_events["snapshot_date"].isin(holdout_weeks)
    corrupted_events.loc[holdout_mask, "executed_return_to_asof_pct"] = -999.0
    corrupted_events.loc[holdout_mask, "week1_close_return_pct"] = -888.0

    corrupted_eval = evaluate_all_rules(
        corrupted_events, weekly_df, baseline_df, rules, train_weeks, holdout_weeks, pick_limit=3
    )
    corrupted_champs = select_champions(corrupted_eval)

    # 3. Assert Champions and Train metrics are 100% identical
    for role in clean_champs:
        assert clean_champs[role]["rule_id"] == corrupted_champs[role]["rule_id"], f"Role {role} deviated!"
        assert clean_champs[role]["train_w1_ret_med"] == pytest.approx(corrupted_champs[role]["train_w1_ret_med"])


def test_02_pit_alpha_independence_invariant(test_datasets):
    """INVARIANT 2: Modifying/deleting future price outcomes does not change PIT eligibility."""
    events_df, _, _ = test_datasets

    # For any event row, modifying future fields does NOT change is_production_eligible_pit result
    row = events_df.iloc[0].copy()
    original_elig = is_production_eligible_pit(row)

    # Modify future columns
    corrupted_row = row.copy()
    corrupted_row["latest_close"] = 99999.0
    corrupted_row["executed_return_to_asof_pct"] = 500.0
    corrupted_row["stop_8_hit_ever"] = True
    corrupted_row["week1_close_return_pct"] = -50.0

    corrupted_elig = is_production_eligible_pit(corrupted_row)
    assert original_elig == corrupted_elig, "PIT eligibility predicate was influenced by future price fields!"


def test_03_censored_draw_invariant():
    """INVARIANT 3: If any sampled symbol is missing price data, the draw is invalid and excluded from percentiles."""
    event_lookup = {
        "AAA": {"executed_return_to_asof_pct": 10.0, "stop8_before_profit20": False, "stop_8_hit_ever": False},
        "BBB": {"executed_return_to_asof_pct": -8.0, "stop8_before_profit20": True, "stop_8_hit_ever": True},
        # CCC has missing price
        "CCC": {"executed_return_to_asof_pct": None, "stop8_before_profit20": None, "stop_8_hit_ever": None},
    }
    weekly_lookup = {
        ("AAA", 1): {"week_close_return_from_entry_pct": 5.0, "stop_8_hit_by_week_end": False},
        ("BBB", 1): {"week_close_return_from_entry_pct": -8.0, "stop_8_hit_by_week_end": True},
        ("CCC", 1): {"week_close_return_from_entry_pct": None, "stop_8_hit_by_week_end": False},
    }

    # Test portfolio metric Censoring
    m_valid = compute_portfolio_metrics(["AAA", "BBB"], event_lookup, weekly_lookup, "2026-01-09")
    assert m_valid["is_portfolio_valid"] is True
    assert not np.isnan(m_valid["executed_return"])

    m_invalid = compute_portfolio_metrics(["AAA", "CCC"], event_lookup, weekly_lookup, "2026-01-09")
    assert m_invalid["is_portfolio_valid"] is False
    assert np.isnan(m_invalid["executed_return"])
    assert m_invalid["invalid_reason"] == "MISSING_PRICE_OUTCOME"


def test_04_production_selector_immutability_invariant():
    """INVARIANT 4: Production selector file SHA256 must match golden frozen hash."""
    repo_root = Path(__file__).resolve().parents[1]
    selector_path = repo_root / "dashboard" / "skill_industry_eps_known.py"
    assert selector_path.exists(), "Missing production selector file"

    current_sha256 = compute_file_sha256(selector_path)
    golden_sha256 = "115387c9861f7202c0f6b3c89fe2d2ff594544de93264901ee6d2f72e930c477"

    assert current_sha256 == golden_sha256, (
        f"CRITICAL: Production selector dashboard/skill_industry_eps_known.py was modified! "
        f"Current SHA256: {current_sha256} != Golden: {golden_sha256}"
    )
