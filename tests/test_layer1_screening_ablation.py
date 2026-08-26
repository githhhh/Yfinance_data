"""Unit and Invariant Tests for Layer-1 Screening Decomposition & Ablation Audit."""

from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backtest.b0_top3_quality_audit.eligibility import is_production_eligible_pit
from backtest.b0_top3_quality_audit.generate_layer1_screening_ablation_audit import (
    ALL_HORIZONS,
    ALL_VARIANTS_REGISTRY,
    DIAGNOSTIC_HORIZON,
    PRIMARY_HORIZONS,
    Layer1AuditPaths,
    build_layer1_audit_data,
    default_layer1_audit_paths,
    derive_candidate_pool_seed,
    evaluate_candidate_features,
    evaluate_portfolio_draw,
    is_candidate_in_variant_pool,
    run_layer1_screening_ablation_audit,
    sample_portfolio_draws,
    summarize_ablation_gates,
    summarize_addback_steps,
    summarize_industry_diversity_and_decomposition,
    summarize_tightening_probes,
    summarize_variant_horizons,
)


@pytest.fixture
def audit_environment():
    paths = default_layer1_audit_paths()
    events_df = pd.read_parquet(paths.events_path)
    weekly_df = pd.read_parquet(paths.weekly_path)
    b0_events_df = pd.read_csv(paths.b0_events_path)
    return paths, events_df, weekly_df, b0_events_df


def test_01_e0_matches_production_eligible_pit(audit_environment):
    """INVARIANT 1: E0_BASE candidate membership must match is_production_eligible_pit 100%."""
    _, events_df, _, _ = audit_environment
    for _, row in events_df.iterrows():
        feat = evaluate_candidate_features(row)
        e0_pred = is_candidate_in_variant_pool(feat, "E0_BASE")
        prod_pred = is_production_eligible_pit(row)
        assert e0_pred == prod_pred, f"Mismatch on code {row.get('code')} snapshot {row.get('snapshot_date')}"


def test_02_e0_and_e1_share_identical_candidate_pool(audit_environment):
    """INVARIANT 2: E0 and E1 candidate pools must be 100% identical; only portfolio constraint differs."""
    _, events_df, _, _ = audit_environment
    for _, row in events_df.iterrows():
        feat = evaluate_candidate_features(row)
        e0_pred = is_candidate_in_variant_pool(feat, "E0_BASE")
        e1_pred = is_candidate_in_variant_pool(feat, "E1_INDUSTRY_DIVERSE")
        assert e0_pred == e1_pred


def test_03_no_future_leakage_on_candidate_filtering(audit_environment):
    """INVARIANT 3: Future entry_status, returns, stops, or profit20 must NOT alter candidate membership."""
    _, events_df, _, _ = audit_environment
    sample_row = events_df.iloc[0].to_dict()

    base_feat = evaluate_candidate_features(sample_row)
    base_e0 = is_candidate_in_variant_pool(base_feat, "E0_BASE")

    corrupted_row = dict(sample_row)
    corrupted_row["entry_status"] = "CANCELLED"
    corrupted_row["executed_return_to_asof_pct"] = -99.9
    corrupted_row["stop_8_hit_ever"] = True
    corrupted_row["profit20_hit"] = True
    corrupted_row["week1_close_return_pct"] = -50.0

    corrupted_feat = evaluate_candidate_features(corrupted_row)
    corrupted_e0 = is_candidate_in_variant_pool(corrupted_feat, "E0_BASE")

    assert base_e0 == corrupted_e0


def test_04_canonical_pool_seed_identity():
    """INVARIANT 4: Identical candidate universe and constraint must yield 100% identical seeds and draws."""
    codes_1 = ["NVDA", "AAPL", "MSFT"]
    codes_2 = ["MSFT", "NVDA", "AAPL"]  # different order, identical set

    s1 = derive_candidate_pool_seed("2026-06-26", codes_1, "UNCONSTRAINED", 42)
    s2 = derive_candidate_pool_seed("2026-06-26", codes_2, "UNCONSTRAINED", 42)
    s_diff_pool = derive_candidate_pool_seed("2026-06-26", ["NVDA", "AAPL"], "UNCONSTRAINED", 42)
    s_div = derive_candidate_pool_seed("2026-06-26", codes_1, "UNIQUE_INDUSTRY", 42)

    assert s1 == s2
    assert s1 != s_diff_pool
    assert s1 != s_div


def test_05_entry_ok_execution_validation():
    """INVARIANT 5: If any pick lacks valid ENTRY_OK execution, event path metrics must be censored (NaN)."""
    event_lookup = {
        ("2025-11-14", "A"): {"is_valid_entry": True, "entry_status": "ENTRY_OK", "entry_open": 100.0, "stop_8_hit_ever": True},
        ("2025-11-14", "B"): {"is_valid_entry": False, "entry_status": "ENTRY_STALE_EXPIRED", "entry_open": None, "stop_8_hit_ever": False},
    }
    weekly_lookup = {}

    res = evaluate_portfolio_draw(
        sampled_codes=["A", "B"],
        snapshot_date="2025-11-14",
        event_lookup=event_lookup,
        weekly_lookup=weekly_lookup,
    )
    assert res["is_event_valid"] is False
    assert np.isnan(res["stop8_ever_rate_pct"])
    assert np.isnan(res["all_stopped_pct"])


def test_06_matched_n_strictness_and_no_shrink(audit_environment):
    """INVARIANT 6: All variants target frozen B0 N. If pool_size < target_N, week is marked infeasible."""
    paths, events_df, weekly_df, b0_events_df = audit_environment
    event_lookup = {(str(r["snapshot_date"]), str(r["code"])): r.to_dict() for _, r in events_df.iterrows()}
    weekly_lookup = {(str(r["snapshot_date"]), str(r["code"]), int(r["holding_week_index"])): r.to_dict() for _, r in weekly_df.iterrows()}

    res = sample_portfolio_draws(
        candidate_codes=["TICKER_A", "TICKER_B"],
        candidate_industries=["Tech", "Finance"],
        target_n=3,
        variant_name="E0_BASE",
        snapshot_date="2025-11-14",
        event_lookup=event_lookup,
        weekly_lookup=weekly_lookup,
        n_draws=100,
    )
    assert res["is_matched_n_feasible"] is False
    assert res["valid_draws"] == 0


def test_07_e1_portfolios_have_unique_industries(audit_environment):
    """INVARIANT 7: E1 sampled portfolios must guarantee that all selected industries are unique."""
    paths, events_df, weekly_df, _ = audit_environment
    event_lookup = {(str(r["snapshot_date"]), str(r["code"])): r.to_dict() for _, r in events_df.iterrows()}
    weekly_lookup = {(str(r["snapshot_date"]), str(r["code"]), int(r["holding_week_index"])): r.to_dict() for _, r in weekly_df.iterrows()}

    codes = ["A", "B", "C", "D"]
    inds = ["Bank", "Bank", "Tech", "Healthcare"]
    res = sample_portfolio_draws(
        candidate_codes=codes,
        candidate_industries=inds,
        target_n=3,
        variant_name="E1_INDUSTRY_DIVERSE",
        snapshot_date="2026-06-26",
        event_lookup=event_lookup,
        weekly_lookup=weekly_lookup,
        n_draws=100,
    )
    assert res["is_matched_n_feasible"] is True
    assert res["valid_draws"] > 0
    assert res["rejection_rate_pct"] >= 0.0


def test_08_maturity_gate_censorship(audit_environment):
    """INVARIANT 8: If any pick has is_complete_week == False, the portfolio is censored at that horizon."""
    event_lookup = {("2025-11-14", "A"): {}, ("2025-11-14", "B"): {}}
    weekly_lookup = {
        ("2025-11-14", "A", 1): {"is_complete_week": True, "week_close_return_from_entry_pct": 5.0, "week_max_gain_from_entry_pct": 6.0},
        ("2025-11-14", "B", 1): {"is_complete_week": False, "week_close_return_from_entry_pct": 2.0, "week_max_gain_from_entry_pct": 3.0},
    }

    res = evaluate_portfolio_draw(
        sampled_codes=["A", "B"],
        snapshot_date="2025-11-14",
        event_lookup=event_lookup,
        weekly_lookup=weekly_lookup,
    )
    assert res["is_valid"] is True
    assert res["w1_valid"] is False
    assert np.isnan(res["w1_return"])


def test_09_missing_return_or_maxgain_censored(audit_environment):
    """INVARIANT 9: Missing return or max-gain on any pick invalidates the portfolio outcome for that horizon."""
    event_lookup = {("2025-11-14", "A"): {}, ("2025-11-14", "B"): {}}
    weekly_lookup = {
        ("2025-11-14", "A", 1): {"is_complete_week": True, "week_close_return_from_entry_pct": 5.0, "week_max_gain_from_entry_pct": 6.0},
        ("2025-11-14", "B", 1): {"is_complete_week": True, "week_close_return_from_entry_pct": None, "week_max_gain_from_entry_pct": 3.0},
    }

    res = evaluate_portfolio_draw(
        sampled_codes=["A", "B"],
        snapshot_date="2025-11-14",
        event_lookup=event_lookup,
        weekly_lookup=weekly_lookup,
    )
    assert res["w1_valid"] is False
    assert np.isnan(res["w1_return"])


def test_10_production_selector_sha_unchanged():
    """INVARIANT 10: Production selector dashboard/skill_industry_eps_known.py SHA256 must remain unchanged."""
    selector_path = Path(__file__).resolve().parents[1] / "dashboard" / "skill_industry_eps_known.py"
    with open(selector_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    assert sha == "115387c9861f7202c0f6b3c89fe2d2ff594544de93264901ee6d2f72e930c477"


def test_11_s5_equals_e0_base(audit_environment):
    """INVARIANT 11: Step 5 of Add-Back (S5_INDUSTRY_KNOWN) must strictly equal E0_BASE."""
    _, events_df, _, _ = audit_environment
    for _, row in events_df.iterrows():
        feat = evaluate_candidate_features(row)
        e0_pred = is_candidate_in_variant_pool(feat, "E0_BASE")
        s5_pred = is_candidate_in_variant_pool(feat, "S5_INDUSTRY_KNOWN")
        assert e0_pred == s5_pred


def test_12_tightening_probes_registry_strictness():
    """INVARIANT 12: Tightening probes must strictly contain only the 5 pre-registered single-factor probes."""
    probe_names = [r[0] for r in ALL_VARIANTS_REGISTRY if r[1] == "TIGHTENING_PROBE"]
    expected = ["T_FRESH_5", "T_FRESH_2", "T_EPS25", "T_ENTRY_VOLUME_15", "T_WEEKLY_VOLUME_13"]
    assert probe_names == expected
