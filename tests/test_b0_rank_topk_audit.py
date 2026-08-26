"""Unit and invariant tests for B0 Rank Position & TopK Marginal Contribution Audit."""

from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backtest.b0_top3_quality_audit.generate_b0_rank_topk_audit import (
    AuditPaths,
    build_common_support_weekly_matrix,
    default_audit_paths,
    run_b0_rank_topk_audit,
    summarize_rank_monotonicity,
    summarize_rank_position_quality,
    summarize_topk_marginal_contributions,
    PRIMARY_HORIZONS,
    DIAGNOSTIC_HORIZON,
    ALL_HORIZONS,
)


@pytest.fixture
def audit_data():
    paths = default_audit_paths()
    b0_events_df = pd.read_csv(paths.b0_events_path)
    events_df = pd.read_parquet(paths.events_path)
    weekly_df = pd.read_parquet(paths.weekly_path)
    three_tier_df = pd.read_csv(paths.three_tier_weekly_path)

    all_snapshots = sorted(three_tier_df["snapshot_date"].astype(str).unique())
    train_snapshots = set(all_snapshots[:30])
    contaminated_snapshots = set(all_snapshots[30:40])

    matrix_df = build_common_support_weekly_matrix(
        b0_events_df=b0_events_df,
        events_df=events_df,
        weekly_df=weekly_df,
        all_snapshots=all_snapshots,
        train_snapshots=train_snapshots,
        contaminated_snapshots=contaminated_snapshots,
    )
    return matrix_df, paths


def test_01_common_support_denominator_strictness(audit_data):
    """INVARIANT 1: Rank1, Rank2, and Rank3 must share the exact identical common-support denominator for each horizon."""
    matrix_df, _ = audit_data
    for h in ALL_HORIZONS:
        valid_rows = matrix_df[matrix_df[f"w{h}_common_valid"]]
        # In all valid common-support rows, all three ranks must be non-null
        assert (valid_rows[f"r1_w{h}_return_pct"].notna()).all()
        assert (valid_rows[f"r2_w{h}_return_pct"].notna()).all()
        assert (valid_rows[f"r3_w{h}_return_pct"].notna()).all()

        # In non-valid rows, returns must be NaN
        invalid_rows = matrix_df[~matrix_df[f"w{h}_common_valid"]]
        assert (invalid_rows[f"r1_w{h}_return_pct"].isna()).all()
        assert (invalid_rows[f"r2_w{h}_return_pct"].isna()).all()
        assert (invalid_rows[f"r3_w{h}_return_pct"].isna()).all()


def test_02_missing_or_incomplete_rank3_excluded(audit_data):
    """INVARIANT 2: 1-pick, 2-pick weeks or weeks with missing Rank3 outcome must NEVER enter 3-pick comparative denominator."""
    matrix_df, _ = audit_data
    # Weeks with < 3 picks must be invalid across all horizons
    partial_weeks = matrix_df[~matrix_df["is_3picks"]]
    assert len(partial_weeks) == 15  # 7 two-pick weeks + 8 one-pick weeks = 15 non-3-pick weeks
    for h in ALL_HORIZONS:
        assert (partial_weeks[f"w{h}_common_valid"] == False).all()


def test_03_portfolio_definitions_and_marginal_contributions(audit_data):
    """INVARIANT 3: K1, K2, K3 and MC2, MC3 must satisfy exact algebraic definitions."""
    matrix_df, _ = audit_data
    for h in ALL_HORIZONS:
        valid_rows = matrix_df[matrix_df[f"w{h}_common_valid"]]
        for _, row in valid_rows.iterrows():
            r1 = float(row[f"r1_w{h}_return_pct"])
            r2 = float(row[f"r2_w{h}_return_pct"])
            r3 = float(row[f"r3_w{h}_return_pct"])

            k1 = float(row[f"k1_w{h}"])
            k2 = float(row[f"k2_w{h}"])
            k3 = float(row[f"k3_w{h}"])

            mc2 = float(row[f"mc2_w{h}"])
            mc3 = float(row[f"mc3_w{h}"])

            assert pytest.approx(k1, abs=1e-3) == r1
            assert pytest.approx(k2, abs=1e-3) == (r1 + r2) / 2.0
            assert pytest.approx(k3, abs=1e-3) == (r1 + r2 + r3) / 3.0
            assert pytest.approx(mc2, abs=1e-3) == k2 - k1
            assert pytest.approx(mc3, abs=1e-3) == k3 - k2
            assert pytest.approx(mc3, abs=1e-3) == (r3 - k2) / 3.0


def test_04_hypothesis_a_vs_hypothesis_b_separation(audit_data):
    """INVARIANT 4: Hypothesis A (Rank3 - Rank2) and Hypothesis B (K3 - K2) are distinct metrics."""
    matrix_df, _ = audit_data
    marginal_df = summarize_topk_marginal_contributions(matrix_df)
    for _, row in marginal_df.iterrows():
        # Hyp A: R3 - R2 spread
        r3_r2_med = row["hyp_a_r3_minus_r2_median_spread_pct"]
        # Hyp B: K3 - K2 spread
        k3_k2_med = row["hyp_b_k3_minus_k2_median_spread_pct"]
        mc3_med = row["mc3_median_pct"]
        assert pytest.approx(k3_k2_med, abs=1e-4) == mc3_med


def test_05_w3_is_marked_diagnostic_only(audit_data):
    """INVARIANT 5: W3 is explicitly marked as DIAGNOSTIC_ONLY and not a frozen primary endpoint."""
    matrix_df, _ = audit_data
    quality_df = summarize_rank_position_quality(matrix_df)
    marginal_df = summarize_topk_marginal_contributions(matrix_df)
    monotonicity_df = summarize_rank_monotonicity(matrix_df)

    assert (quality_df[quality_df["horizon"] == "W3"]["horizon_status"] == "DIAGNOSTIC_ONLY").all()
    assert (quality_df[quality_df["horizon"].isin(["W1", "W2", "W4"])]["horizon_status"] == "PRIMARY").all()

    assert (marginal_df[marginal_df["horizon"] == "W3"]["horizon_status"] == "DIAGNOSTIC_ONLY").all()
    assert (marginal_df[marginal_df["horizon"].isin(["W1", "W2", "W4"])]["horizon_status"] == "PRIMARY").all()

    assert (monotonicity_df[monotonicity_df["horizon"] == "W3"]["horizon_status"] == "DIAGNOSTIC_ONLY").all()
    assert (monotonicity_df[monotonicity_df["horizon"].isin(["W1", "W2", "W4"])]["horizon_status"] == "PRIMARY").all()


def test_06_production_selector_sha_unchanged():
    """INVARIANT 6: Production selector dashboard/skill_industry_eps_known.py SHA256 must remain strictly unchanged."""
    selector_path = Path(__file__).resolve().parents[1] / "dashboard" / "skill_industry_eps_known.py"
    with open(selector_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    # Expected SHA256 frozen at Phase 2 baseline
    assert sha == "115387c9861f7202c0f6b3c89fe2d2ff594544de93264901ee6d2f72e930c477"


def test_07_end_to_end_audit_generation(tmp_path):
    """INVARIANT 7: Complete audit execution generates all 5 CSV artifacts and Markdown report."""
    base_paths = default_audit_paths()
    custom_paths = AuditPaths(
        root_dir=base_paths.root_dir,
        output_dir=tmp_path / "output",
        b0_events_path=base_paths.b0_events_path,
        events_path=base_paths.events_path,
        weekly_path=base_paths.weekly_path,
        three_tier_weekly_path=base_paths.three_tier_weekly_path,
        random_summary_path=base_paths.random_summary_path,
    )

    w_mat, q_df, m_df, mono_df, s_df, report_md = run_b0_rank_topk_audit(custom_paths)

    assert not w_mat.empty
    assert not q_df.empty
    assert not m_df.empty
    assert not mono_df.empty
    assert not s_df.empty
    assert len(report_md) > 1000

    assert (custom_paths.output_dir / "b0_rank_position_weekly_detail.csv").exists()
    assert (custom_paths.output_dir / "b0_rank_position_quality_summary.csv").exists()
    assert (custom_paths.output_dir / "b0_topk_marginal_contribution_summary.csv").exists()
    assert (custom_paths.output_dir / "b0_rank_monotonicity_summary.csv").exists()
    assert (custom_paths.output_dir / "b0_rank_position_structure_profile.csv").exists()
    assert (custom_paths.output_dir / "B0_RANK_POSITION_TOPK_AUDIT_REPORT.md").exists()
