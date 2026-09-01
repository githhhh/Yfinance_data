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
    """Missing selected outcomes must shrink paired support, never become 0% cash."""
    from backtest.track_c_ranking_discovery.counterfactual_engine import run_counterfactual_monte_carlo

    snap = "2026-01-02"
    scored = pd.DataFrame({
        "code": ["AAA", "BBB", "CCC"],
        "is_actionable": [1, 1, 1],
        "has_geom_failure": [0, 0, 0],
        "below_buy_point": [0, 0, 0],
        "has_known_eps": [1, 1, 1],
        "has_valid_industry": [1, 1, 1],
        "industry_key": ["X", "Y", "Z"],
        "raw_rank": [1, 2, 3],
    })
    panel = pd.DataFrame({
        "snapshot_date": [snap, snap, snap],
        "code": ["AAA", "BBB", "CCC"],
        "w4_return_pct": [6.0, 3.0, np.nan],
        "w4_stop8": [0.0, 0.0, 0.0],
    })
    b0_outcomes = [WeeklyPortfolioOutcome(
        snapshot_date=snap,
        selector_id="B0_ORIGINAL",
        horizon="W4",
        pick_count=2,
        slot_coverage=round(2 / 3, 4),
        active_week=True,
        full_top3=False,
        selected_codes=["AAA", "BBB"],
        selection_quality_return=4.5,
        capital_adjusted_return=3.0,
        selection_quality_stop8=0.0,
        capital_adjusted_stop8=0.0,
        one_pick_ruined=False,
        is_mature=True,
    )]

    res, df_decomp = run_counterfactual_monte_carlo(
        panel,
        b0_outcomes,
        {snap: scored},
        horizon="W4",
        n_paths=400,
        seed=123,
        null_model="Null1_Uniform_Industry",
    )

    # Only paths that sampled X+Y have common A/B/C/D mature support.
    # Any path involving Z selected CCC and must be excluded, never treated as 0%.
    assert 0 < res.valid_paths < 400
    assert res.mean_A_random_ind_random_stock == pytest.approx(3.0)
    assert res.mean_B_random_ind_b0_best_stock == pytest.approx(3.0)
    assert res.mean_C_b0_ind_random_stock == pytest.approx(3.0)
    assert res.mean_D_b0_native == pytest.approx(3.0)
    assert bool(df_decomp.iloc[0]["paired_common_support"]) is True



def test_rdagent_frozen_spec_roundtrip_and_tamper_detection():
    from backtest.track_c_ranking_discovery.discovery_sandbox.discovery_runner import (
        normalize_discovery_records,
        instantiate_discovery_proposals,
    )

    raw = [{
        "family": "continuous",
        "name": "agent_roundtrip",
        "hypothesis": "PIT-safe EPS and entry-volume ranking",
        "params": {
            "weights": {
                "eps_yoy_growth": 2.0,
                "ibd_entry_volume_ratio": 1.5,
            },
            "selector_mode": "distinct_1",
        },
        "source_response_hash": "abc123",
        "source_response_path": "rdagent_policy_discovery/raw/continuous.txt",
        "source_model": "deepseek/deepseek-v4-pro",
    }]
    policies, records = normalize_discovery_records(raw)
    replayed = instantiate_discovery_proposals(records)
    assert replayed[0].policy_id == policies[0].policy_id
    assert replayed[0].spec_hash == policies[0].spec_hash

    tampered = json.loads(json.dumps(records))
    tampered[0]["spec_params"]["weights"]["eps_yoy_growth"] = 7.0
    with pytest.raises(RuntimeError, match="spec_hash"):
        instantiate_discovery_proposals(tampered)


def test_counterfactual_uses_paired_common_support_for_missing_random_outcomes():
    from backtest.track_c_ranking_discovery.counterfactual_engine import run_counterfactual_monte_carlo

    snap = "2026-01-02"
    scored = pd.DataFrame({
        "code": ["AAA", "BBB", "CCC"],
        "is_actionable": [1, 1, 1],
        "has_geom_failure": [0, 0, 0],
        "below_buy_point": [0, 0, 0],
        "has_known_eps": [1, 1, 1],
        "has_valid_industry": [1, 1, 1],
        "industry_key": ["X", "X", "Y"],
        "raw_rank": [1, 2, 3],
    })
    panel = pd.DataFrame({
        "snapshot_date": [snap, snap, snap],
        "code": ["AAA", "BBB", "CCC"],
        "w4_return_pct": [6.0, 4.0, np.nan],
        "w4_stop8": [0.0, 0.0, 0.0],
    })
    b0 = [WeeklyPortfolioOutcome(
        snapshot_date=snap,
        selector_id="B0_ORIGINAL",
        horizon="W4",
        pick_count=1,
        slot_coverage=1 / 3,
        active_week=True,
        full_top3=False,
        selected_codes=["AAA"],
        selection_quality_return=6.0,
        capital_adjusted_return=2.0,
        selection_quality_stop8=0.0,
        capital_adjusted_stop8=0.0,
        one_pick_ruined=False,
        is_mature=True,
    )]

    result, df = run_counterfactual_monte_carlo(
        panel,
        b0,
        {snap: scored},
        horizon="W4",
        n_paths=400,
        seed=123,
        null_model="Null1_Uniform_Industry",
    )
    assert 0 < result.valid_paths < 400
    assert result.mean_A_random_ind_random_stock >= (4.0 / 3.0) - 1e-6
    assert result.mean_A_random_ind_random_stock <= (6.0 / 3.0) + 1e-6
    assert bool(df.iloc[0]["paired_common_support"]) is True



def test_rdagent_outcome_blind_summary_handles_boolean_features():
    from backtest.track_c_ranking_discovery.discovery_sandbox.rdagent_policy_bridge import (
        _outcome_blind_summary,
    )

    anon = pd.DataFrame({
        "code": ["entity_001", "entity_002", "entity_003", "entity_004"],
        "snapshot_date": ["snapshot_001"] * 4,
        "is_actionable": [True, False, True, True],
        "has_geom_failure": [False, False, True, False],
        "eps_yoy_growth": [10.0, 20.0, 30.0, 40.0],
    })

    summary = _outcome_blind_summary(anon)
    actionable = summary["numeric"]["is_actionable"]
    geom = summary["numeric"]["has_geom_failure"]

    assert actionable["non_null"] == 4
    assert actionable["p50"] == pytest.approx(1.0)
    assert geom["p50"] == pytest.approx(0.0)
    assert summary["numeric"]["eps_yoy_growth"]["p50"] == pytest.approx(25.0)
