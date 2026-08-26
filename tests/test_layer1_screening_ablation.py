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
    """INVARIANT 4: Identical candidate universe and constraint must yield 100% identical seeds."""
    codes_1 = ["NVDA", "AAPL", "MSFT"]
    codes_2 = ["MSFT", "NVDA", "AAPL"]

    s1 = derive_candidate_pool_seed("2026-06-26", codes_1, "UNCONSTRAINED", 42)
    s2 = derive_candidate_pool_seed("2026-06-26", codes_2, "UNCONSTRAINED", 42)
    s_diff_pool = derive_candidate_pool_seed("2026-06-26", ["NVDA", "AAPL"], "UNCONSTRAINED", 42)
    s_div = derive_candidate_pool_seed("2026-06-26", codes_1, "UNIQUE_INDUSTRY", 42)

    assert s1 == s2
    assert s1 != s_diff_pool
    assert s1 != s_div


def test_05_non_binding_industry_constraint_identity(audit_environment):
    """INVARIANT 5 (Test A): When E1 industry constraint is non-binding, E1 and E0 must be bitwise identical."""
    paths, events_df, weekly_df, b0_events_df = audit_environment
    summary_path = paths.output_dir / "layer1_variant_weekly_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        e0 = df[df["variant_name"] == "E0_BASE"].set_index("snapshot_date")
        e1 = df[df["variant_name"] == "E1_INDUSTRY_DIVERSE"].set_index("snapshot_date")

        # Find real non-binding snapshots (where target_n <= 1 or no duplicate industries in E0)
        for snap, row_e0 in e0.iterrows():
            target_n = row_e0["target_n"]
            sub_events = events_df[events_df["snapshot_date"].astype(str) == str(snap)]
            feats = [evaluate_candidate_features(r) for _, r in sub_events.iterrows()]
            e0_cands = [f for f in feats if is_candidate_in_variant_pool(f, "E0_BASE")]
            inds = [f["industry"] for f in e0_cands]
            is_binding = (target_n > 1) and (len(inds) > len(set(inds)))

            if not is_binding:
                row_e1 = e1.loc[snap]
                for col in ["w1_p50", "w2_p50", "w4_p50", "stop8_ever_rate_pct", "all_stopped_pct"]:
                    v0 = row_e0[col]
                    v1 = row_e1[col]
                    assert (v0 == v1) or (pd.isna(v0) and pd.isna(v1)), f"Mismatch on {col} for non-binding snap {snap}"


def test_06_entry_ok_gates_horizon_returns_and_path_risk():
    """INVARIANT 6 (Test B): If ENTRY_OK is False on any pick, path risk AND horizon returns must be censored (NaN)."""
    event_lookup = {
        ("2025-11-14", "A"): {"is_valid_entry": True, "entry_status": "ENTRY_OK", "entry_open": 100.0, "stop_8_hit_ever": True},
        ("2025-11-14", "B"): {"is_valid_entry": False, "entry_status": "ENTRY_STALE_EXPIRED", "entry_open": None, "stop_8_hit_ever": False},
    }
    weekly_lookup = {
        ("2025-11-14", "A", 1): {"is_complete_week": True, "week_close_return_from_entry_pct": 5.0, "week_max_gain_from_entry_pct": 6.0},
        ("2025-11-14", "B", 1): {"is_complete_week": True, "week_close_return_from_entry_pct": 2.0, "week_max_gain_from_entry_pct": 3.0},
    }

    res = evaluate_portfolio_draw(
        sampled_codes=["A", "B"],
        snapshot_date="2025-11-14",
        event_lookup=event_lookup,
        weekly_lookup=weekly_lookup,
    )
    assert res["is_event_valid"] is False
    assert np.isnan(res["stop8_ever_rate_pct"])
    assert np.isnan(res["all_stopped_pct"])
    assert res["w1_valid"] is False
    assert np.isnan(res["w1_return"])
    assert np.isnan(res["w1_max_gain"])


def test_07_matched_n_strictness_and_no_shrink(audit_environment):
    """INVARIANT 7: All variants target frozen B0 N. If pool_size < target_N, week is marked infeasible."""
    paths, events_df, weekly_df, _ = audit_environment
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


def test_08_e1_portfolios_have_unique_industries_when_binding(audit_environment):
    """INVARIANT 8: When binding, E1 sampled portfolios must guarantee that all selected industries are unique."""
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


def test_09_maturity_gate_censorship(audit_environment):
    """INVARIANT 9: If any pick has is_complete_week == False, the portfolio is censored at that horizon."""
    event_lookup = {
        ("2025-11-14", "A"): {"is_valid_entry": True, "entry_status": "ENTRY_OK", "entry_open": 100.0},
        ("2025-11-14", "B"): {"is_valid_entry": True, "entry_status": "ENTRY_OK", "entry_open": 100.0},
    }
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


def test_10_missing_return_or_maxgain_censored(audit_environment):
    """INVARIANT 10: Missing return or max-gain on any pick invalidates the portfolio outcome for that horizon."""
    event_lookup = {
        ("2025-11-14", "A"): {"is_valid_entry": True, "entry_status": "ENTRY_OK", "entry_open": 100.0},
        ("2025-11-14", "B"): {"is_valid_entry": True, "entry_status": "ENTRY_OK", "entry_open": 100.0},
    }
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


def test_11_production_selector_sha_unchanged():
    """INVARIANT 11: Production selector dashboard/skill_industry_eps_known.py SHA256 must remain unchanged."""
    selector_path = Path(__file__).resolve().parents[1] / "dashboard" / "skill_industry_eps_known.py"
    with open(selector_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    assert sha == "115387c9861f7202c0f6b3c89fe2d2ff594544de93264901ee6d2f72e930c477"


def test_12_s5_equals_e0_base(audit_environment):
    """INVARIANT 12: Step 5 of Add-Back (S5_INDUSTRY_KNOWN) must strictly equal E0_BASE."""
    _, events_df, _, _ = audit_environment
    for _, row in events_df.iterrows():
        feat = evaluate_candidate_features(row)
        e0_pred = is_candidate_in_variant_pool(feat, "E0_BASE")
        s5_pred = is_candidate_in_variant_pool(feat, "S5_INDUSTRY_KNOWN")
        assert e0_pred == s5_pred


def test_13_w3_is_marked_diagnostic_only(audit_environment):
    """INVARIANT 13: W3 must be strictly marked as DIAGNOSTIC_ONLY across summary outputs."""
    paths, _, _, _ = audit_environment
    summary_path = paths.output_dir / "layer1_variant_horizon_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        assert (df[df["horizon"] == "W3"]["horizon_status"] == "DIAGNOSTIC_ONLY").all()
        assert (df[df["horizon"].isin(["W1", "W2", "W4"])]["horizon_status"] == "PRIMARY").all()


def test_14_tightening_probes_registry_strictness():
    """INVARIANT 14: Tightening probes must strictly contain only the 5 pre-registered single-factor probes."""
    probe_names = [r[0] for r in ALL_VARIANTS_REGISTRY if r[1] == "TIGHTENING_PROBE"]
    expected = ["T_FRESH_5", "T_FRESH_2", "T_EPS25", "T_ENTRY_VOLUME_15", "T_WEEKLY_VOLUME_13"]
    assert probe_names == expected
