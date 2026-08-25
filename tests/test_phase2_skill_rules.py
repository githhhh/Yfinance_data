"""Automated tests for Phase 2 Steps 2~4: Skill Rule Engine & Pareto Champions."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from backtest.b0_top3_quality_audit.skill_rule_engine import (
    build_skill_rule_space,
    deduplicate_rule_signatures,
    evaluate_rule_on_pool,
)
from backtest.b0_top3_quality_audit.pareto_champions import select_champions


@pytest.fixture
def sample_data():
    root_dir = Path(__file__).resolve().parents[1] / "backtest" / "b0_top3_quality_audit"
    events_path = root_dir / "data" / "candidate_event_outcomes.parquet"
    eval_path = root_dir / "output" / "skill_rule_variants_evaluation.csv"
    champ_path = root_dir / "output" / "pareto_champions_matrix.csv"
    
    assert events_path.exists(), f"Missing {events_path}"
    assert eval_path.exists(), f"Missing {eval_path}"
    assert champ_path.exists(), f"Missing {champ_path}"
    
    events_df = pd.read_parquet(events_path)
    eval_df = pd.read_csv(eval_path)
    champ_df = pd.read_csv(champ_path)
    return events_df, eval_df, champ_df


def test_01_skill_rule_space_construction():
    rules = build_skill_rule_space()
    assert len(rules) > 20, "Should generate a rich discrete parameterization space"
    
    # Check rule IDs are unique
    rule_ids = [r.rule_id for r in rules]
    assert len(rule_ids) == len(set(rule_ids)), "Rule IDs must be unique"
    
    # Check complexity is positive
    assert all(r.complexity >= 1 for r in rules)


def test_02_signature_deduplication_budget_limit(sample_data):
    events_df, _, _ = sample_data
    all_weeks = sorted(events_df["snapshot_date"].unique())
    train_weeks = all_weeks[:30]
    
    rules = build_skill_rule_space()
    deduped_rules, signature_map = deduplicate_rule_signatures(
        rules, events_df, train_weeks, pick_limit=3, budget_limit=200
    )
    
    assert len(deduped_rules) <= 200, "Must strictly respect the <= 200 hard signature budget"
    assert len(deduped_rules) == len(signature_map), "Must map 1-to-1 with unique train signatures"


def test_03_champion_selection_invariants(sample_data):
    _, eval_df, champ_df = sample_data
    champions = select_champions(eval_df)
    
    # Check all 4 champion roles exist
    expected_roles = {
        "HISTORICAL_RETURN_WINNER",
        "LOWEST_STOP_CANDIDATE",
        "SIMPLER_EQUIVALENT",
        "PARETO_BALANCED_RULE",
    }
    assert set(champions.keys()) == expected_roles
    
    # Check Simpler Equivalent complexity < B0
    b0_row = eval_df[eval_df["rule_id"] == "B0_BASELINE"].iloc[0]
    simpler = champions["SIMPLER_EQUIVALENT"]
    assert simpler["complexity"] < b0_row["complexity"]
    
    # Non-inferiority test (train_exec_ret_med >= B0 - 0.5%)
    assert simpler["train_exec_ret_med"] >= b0_row["train_exec_ret_med"] - 0.5


def test_04_pareto_balanced_production_candidate(sample_data):
    _, _, champ_df = sample_data
    pareto_row = champ_df[champ_df["champion_role"] == "PARETO_BALANCED_RULE"].iloc[0]
    
    # Pareto Balanced Rule should have positive return on Holdout
    assert pareto_row["holdout_exec_ret_med"] > 0.0, "Holdout must maintain positive return without collapse"
    assert pareto_row["train_act_rank_win_vs_l1_pct"] >= 50.0, "Must achieve >= 50% win rate vs L1 on active weeks"
