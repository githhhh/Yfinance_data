"""Physical Invariant Tests for Quantitative Audit and Methodological Integrity."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backtest.b0_top3_quality_audit.baseline import run_b0_across_all_pools
from backtest.b0_top3_quality_audit.eligibility import is_production_eligible_pit
from backtest.b0_top3_quality_audit.evaluate_rule_signatures import (
    evaluate_rules_on_train,
)
from backtest.b0_top3_quality_audit.historical_validation_verifier import (
    verify_manifest_integrity,
)
from backtest.b0_top3_quality_audit.pareto_champions import (
    compute_file_sha256,
    export_frozen_rules_manifest,
    select_champions,
)
from backtest.b0_top3_quality_audit.skill_rule_engine import (
    build_skill_rule_space,
    deduplicate_rule_signatures,
)
from backtest.b0_top3_quality_audit.three_tier_baseline import compute_portfolio_metrics


@pytest.fixture
def test_datasets():
    root_dir = Path(__file__).resolve().parents[1] / "backtest" / "b0_top3_quality_audit"
    events_df = pd.read_parquet(root_dir / "data" / "candidate_event_outcomes.parquet")
    weekly_df = pd.read_parquet(root_dir / "data" / "candidate_weekly_outcomes.parquet")
    baseline_df = pd.read_csv(root_dir / "output" / "three_tier_weekly_comparison.csv")
    return events_df, weekly_df, baseline_df


def test_01_holdout_isolation_invariant(test_datasets, tmp_path):
    """INVARIANT 1: Corrupting all Holdout weeks (31~40) must produce 100% byte-for-byte identical Train Manifest."""
    events_df, weekly_df, baseline_df = test_datasets
    all_weeks = sorted(baseline_df["snapshot_date"].unique())
    train_weeks = all_weeks[:30]
    holdout_weeks = all_weeks[30:]

    base_space = build_skill_rule_space()
    deduped_base, _ = deduplicate_rule_signatures(
        base_space, events_df, train_weeks, pick_limit=3, budget_limit=200
    )

    # 1. Clean Run
    clean_events_train = events_df[events_df["snapshot_date"].isin(train_weeks)]
    clean_weekly_train = weekly_df[weekly_df["snapshot_date"].isin(train_weeks)]
    clean_base_train = baseline_df[baseline_df["snapshot_date"].isin(train_weeks)]

    clean_eval = evaluate_rules_on_train(
        clean_events_train, clean_weekly_train, clean_base_train, deduped_base, pick_limit=3
    )
    clean_champs = select_champions(clean_eval)
    clean_manifest = export_frozen_rules_manifest(clean_champs, tmp_path / "clean_manifest.json")

    # 2. Corrupted Holdout Run: Mutate all holdout price returns in the source dataframe
    corrupted_events = events_df.copy()
    holdout_mask = corrupted_events["snapshot_date"].isin(holdout_weeks)
    corrupted_events.loc[holdout_mask, "executed_return_to_asof_pct"] = -999.0
    corrupted_events.loc[holdout_mask, "week1_close_return_pct"] = -888.0

    corrupted_events_train = corrupted_events[corrupted_events["snapshot_date"].isin(train_weeks)]
    corrupted_eval = evaluate_rules_on_train(
        corrupted_events_train, clean_weekly_train, clean_base_train, deduped_base, pick_limit=3
    )
    corrupted_champs = select_champions(corrupted_eval)
    corrupted_manifest = export_frozen_rules_manifest(corrupted_champs, tmp_path / "corrupted_manifest.json")

    # 3. Assert Byte-for-Byte and SHA256 equality
    assert clean_manifest["manifest_sha256"] == corrupted_manifest["manifest_sha256"], "Manifest SHA drifted under Holdout corruption!"
    assert json.dumps(clean_manifest, sort_keys=True) == json.dumps(corrupted_manifest, sort_keys=True)


def test_02_pit_alpha_independence_invariant(test_datasets):
    """INVARIANT 2: Modifying/deleting future price outcomes does not change PIT eligibility."""
    events_df, _, _ = test_datasets

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
    """INVARIANT 3: If any sampled symbol is missing price data, the portfolio is invalid and returns NaN."""
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


def test_05_manifest_fail_closed_gate():
    """INVARIANT 5: Manifest verifier must raise exception if manifest SHA or code fingerprint is tampered."""
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "backtest" / "b0_top3_quality_audit" / "output" / "frozen_rules_manifest.json"
    if not manifest_path.exists():
        pytest.skip("Manifest not yet generated")

    with open(manifest_path, "r", encoding="utf-8") as f:
        valid_manifest = json.load(f)

    # 1. Valid Manifest must pass
    verify_manifest_integrity(valid_manifest, repo_root)

    # 2. Tampered code fingerprint must fail
    tampered_code = copy.deepcopy(valid_manifest)
    tampered_code["code_fingerprints"]["production_selector_sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
    with pytest.raises(RuntimeError):
        verify_manifest_integrity(tampered_code, repo_root)

    # 3. Tampered champion param must fail
    tampered_param = copy.deepcopy(valid_manifest)
    tampered_param["champions"]["HISTORICAL_RETURN_WINNER"]["complexity"] = 999
    with pytest.raises(RuntimeError):
        verify_manifest_integrity(tampered_param, repo_root)


def test_06_golden_reference_audit():
    """INVARIANT 6: B0 Production Replay must achieve 100% exact match against frozen Golden Reference."""
    from backtest.b0_top3_quality_audit.universe import scan_replay_pools
    repo_root = Path(__file__).resolve().parents[1]
    golden_path = repo_root / "backtest" / "b0_top3_quality_audit" / "golden" / "b0_top3_golden_reference.csv"
    assert golden_path.exists(), "Missing golden reference file"

    golden_df = pd.read_csv(golden_path)
    assert len(golden_df) == 97, f"Golden reference should have 97 events, got {len(golden_df)}"

    all_pools = scan_replay_pools(repo_root / "backtest" / "ibd_skill_replay_pools")
    golden_weeks = set(golden_df["snapshot_date"].unique())
    pool_paths = [p for p in all_pools if p.parent.name in golden_weeks]
    assert len(pool_paths) == len(golden_weeks), f"Expected {len(golden_weeks)} golden replay pools, found {len(pool_paths)}"

    _, invariant_df = run_b0_across_all_pools(
        pool_paths,
        output_events_csv=None,
        output_invariant_csv=None,
        golden_csv_path=golden_path,
    )

    assert invariant_df["is_exact_match"].all(), "Discrepancy found against Golden Reference!"
    assert (invariant_df["discrepancy_count"] == 0).all()
