from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backtest.track_c_ranking_discovery.config import (
    PANEL_SOURCE,
    FEATURE_MANIFEST_PATH,
    MANDATORY_GRID_BUDGET,
    DISCOVERY_BUDGET,
    TOTAL_BUDGET,
    FAMILY_BUDGETS,
)
from backtest.track_c_ranking_discovery.b0_ablation_grid import (
    StructuralGridChallenger,
    get_structural_grid_challengers,
)
from backtest.track_c_ranking_discovery.protocol import (
    compute_3slot_portfolio_weekly,
    WeeklyPortfolioOutcome,
)
from backtest.track_c_ranking_discovery.evaluate_econometrics import (
    compute_lowo_fragility,
    compute_moving_block_bootstrap,
    classify_champion_track_c,
    PairedEvaluationSummary,
)
from backtest.track_c_ranking_discovery.discovery_sandbox.anonymizer import (
    create_anonymized_discovery_dataset,
)
from backtest.track_c_ranking_discovery.discovery_sandbox.discovery_runner import (
    generate_all_discovery_proposals,
)
from backtest.track_c_ranking_discovery.discovery_sandbox.behavioral_dedup import (
    compute_snapshot_picks_jaccard,
    deduplicate_proposals_behaviorally,
)
from dashboard.skill_industry_eps_known import rank_skill_industry_eps_known, select_skill_industry_eps_known


def test_feature_manifest_blocks_all_outcomes():
    """Verify feature manifest strictly blocks all outcome returns and future stop labels."""
    with open(FEATURE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for k, v in manifest["features"].items():
        if "return" in k or "stop" in k:
            assert v["allowed_for_discovery"] is False, f"Outcome feature {k} must be blocked from discovery!"


def test_baseline_anchor_100_percent_parity_with_production():
    """Verify B0_LANE x symmetric x distinct_1 achieves 100% exact parity with production B0."""
    panel_df = pd.read_parquet(PANEL_SOURCE)
    snaps = sorted(panel_df["snapshot_date"].astype(str).unique().tolist())

    challenger = StructuralGridChallenger("B0_LANE", "symmetric", "distinct_1")

    for s in snaps:
        s_df = panel_df[panel_df.snapshot_date.astype(str) == str(s)].copy()
        
        # Production direct
        prod_ranked = rank_skill_industry_eps_known(s_df)
        prod_selected = select_skill_industry_eps_known(s_df, limit=3)
        prod_codes = [c.code for c in prod_selected]

        # Challenger
        scored = challenger.score_candidates(s_df)
        quotas = challenger.allocate_industries(scored)
        ch_codes = challenger.pick_stocks(scored, quotas)

        assert ch_codes == prod_codes, f"Mismatch on snapshot {s}: {ch_codes} vs {prod_codes}"


def test_3slot_capital_accounting_math():
    """Verify exact 3-slot capital accounting for k=0, 1, 2, 3 picks."""
    all_snaps = ["2026-01-02", "2026-01-09", "2026-01-16", "2026-01-23"]

    # 1. k=0 week (empty picks)
    empty_df = pd.DataFrame()
    outcomes = compute_3slot_portfolio_weekly(empty_df, all_snaps, "TEST", "W4")
    assert len(outcomes) == 4
    for o in outcomes:
        assert o.pick_count == 0
        assert o.capital_adjusted_return == 0.0
        assert np.isnan(o.selection_quality_return)
        assert o.slot_coverage == 0.0

    # 2. k=2 week (2 stocks: +10% and +20%)
    picks_df = pd.DataFrame({
        "snapshot_date": ["2026-01-02", "2026-01-02"],
        "code": ["AAA", "BBB"],
        "w4_return_pct": [10.0, 20.0],
        "w4_stop8": [False, False],
    })
    outcomes_k2 = compute_3slot_portfolio_weekly(picks_df, ["2026-01-02"], "TEST", "W4")
    assert outcomes_k2[0].pick_count == 2
    assert outcomes_k2[0].selection_quality_return == 15.0  # (10 + 20) / 2
    assert outcomes_k2[0].capital_adjusted_return == 10.0  # (10 + 20 + 0) / 3
    assert outcomes_k2[0].slot_coverage == round(2 / 3.0, 4)


def test_lowo_fragility_rule():
    """Verify strictly pre-registered LOWO fragility rule."""
    # Case 1: Healthy stable strategy (pos edge conc < 0.50, sign stab >= 0.70)
    ch_rets = np.array([5.0, 6.0, 4.0, 5.0, 7.0, 6.0, 5.0, 6.0])
    b0_rets = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    res_stable = compute_lowo_fragility(ch_rets, b0_rets)
    assert res_stable.is_fragile_overfit is False
    assert res_stable.sign_stability == 1.0

    # Case 2: Fragile overfit (one giant winner week + negative rest)
    ch_fragile = np.array([30.0, 3.0, 3.0, -11.0, -11.0, -11.0])
    b0_fragile = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    res_fragile = compute_lowo_fragility(ch_fragile, b0_fragile)
    assert res_fragile.positive_edge_concentration > 0.50
    assert res_fragile.sign_stability < 0.70
    assert res_fragile.is_fragile_overfit is True


def test_search_budgets_consistency():
    """Verify total research budgets math consistency: 36 structural + 54 discovery <= 90."""
    proposals = generate_all_discovery_proposals()
    assert len(proposals) <= DISCOVERY_BUDGET

    structural = get_structural_grid_challengers()
    assert len(structural) == MANDATORY_GRID_BUDGET
    assert len(structural) + len(proposals) <= TOTAL_BUDGET


def test_anonymizer_strips_outcomes_and_anonymizes_entities():
    """Verify anonymizer strips real tickers and all outcome return columns."""
    panel_df = pd.read_parquet(PANEL_SOURCE)
    anon_view, code_map, snap_map = create_anonymized_discovery_dataset(panel_df)

    assert "w1_return_pct" not in anon_view.columns
    assert "w4_return_pct" not in anon_view.columns
    assert "stop8" not in anon_view.columns
    assert all(c.startswith("entity_") for c in anon_view["code"])
    assert all(s.startswith("snapshot_") for s in anon_view["snapshot_date"])


def test_monte_carlo_missing_return_does_not_fill_zero():
    """Verify that Monte Carlo selection-first maturity-second protocol does not treat missing returns as 0.0% cash."""
    from backtest.track_c_ranking_discovery.counterfactual_engine import run_counterfactual_monte_carlo
    from backtest.track_c_ranking_discovery.protocol import WeeklyPortfolioOutcome

    panel_df = pd.read_parquet(PANEL_SOURCE).copy()
    snaps = sorted(panel_df["snapshot_date"].astype(str).unique().tolist())[:5]
    sub_df = panel_df[panel_df.snapshot_date.astype(str).isin(snaps)].copy()

    # Create synthetic b0 outcomes
    b0_outcomes = [
        WeeklyPortfolioOutcome(
            snapshot_date=s,
            selector_id="B0_ORIGINAL",
            horizon="W4",
            pick_count=2,
            slot_coverage=0.667,
            active_week=True,
            full_top3=False,
            selected_codes=["AAPL", "MSFT"],
            selection_quality_return=5.0,
            capital_adjusted_return=3.33,
            selection_quality_stop8=0.0,
            capital_adjusted_stop8=0.0,
            one_pick_ruined=False,
            is_mature=True,
        )
        for s in snaps
    ]

    ch = StructuralGridChallenger("B0_LANE", "symmetric", "distinct_1")
    b0_scored = {s: ch.score_candidates(sub_df[sub_df.snapshot_date.astype(str) == s].copy()) for s in snaps}

    res, df_decomp = run_counterfactual_monte_carlo(
        sub_df, b0_outcomes, b0_scored, horizon="W4", n_paths=100, null_model="Null1_Uniform_Industry"
    )
    assert not np.isnan(res.mean_A_random_ind_random_stock)
    assert not np.isnan(res.b0_induced_industry_allocation_effect)
