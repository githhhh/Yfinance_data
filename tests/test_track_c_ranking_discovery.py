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
    policies, records, rejected = normalize_discovery_records(raw)
    assert rejected == []
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



def test_rdagent_model_config_honors_custom_deepseek_endpoint(monkeypatch):
    from backtest.track_c_ranking_discovery.discovery_sandbox.rdagent_policy_bridge import (
        _load_rdagent_model_config,
    )

    monkeypatch.delenv("TRACK_C_RDAGENT_MODEL", raising=False)
    monkeypatch.delenv("RD_AGENT_MODEL", raising=False)
    monkeypatch.setenv("CHAT_MODEL", "deepseek/deepseek-v4-pro")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_API_BASE", "https://example.invalid/v1")
    monkeypatch.setenv("REASONING_EFFORT", "high")
    monkeypatch.setenv("RETRY_WAIT_SECONDS", "15")
    monkeypatch.setenv("MAX_RETRY", "15")

    cfg = _load_rdagent_model_config()

    assert cfg["model"] == "deepseek/deepseek-v4-pro"
    assert cfg["api_key"] == "sk-test"
    assert cfg["api_base"] == "https://example.invalid/v1"
    assert cfg["reasoning_effort"] == "high"
    assert cfg["retry_wait_seconds"] == 15.0
    assert cfg["max_retry"] == 15



def test_rdagent_schema_rejects_bad_item_without_killing_valid_family():
    from backtest.track_c_ranking_discovery.discovery_sandbox.discovery_runner import (
        normalize_discovery_records,
    )

    base_meta = {
        "source_response_hash": "family-response-hash",
        "source_response_path": "rdagent_policy_discovery/raw/industry_breadth.txt",
        "source_model": "deepseek/deepseek-v4-pro",
    }
    raw = [
        {
            **base_meta,
            "family": "industry_breadth",
            "name": "valid_breadth",
            "hypothesis": "valid family member",
            "params": {
                "breadth_metric": "actionable_count",
                "allow_dynamic_2_plus_1": True,
                "min_breadth_for_2": 2,
            },
        },
        {
            **base_meta,
            "family": "industry_breadth",
            "name": "invalid_breadth_4",
            "hypothesis": "model violated the pre-registered schema",
            "params": {
                "breadth_metric": "actionable_count",
                "allow_dynamic_2_plus_1": False,
                "min_breadth_for_2": 4,
            },
        },
    ]

    policies, records, rejected = normalize_discovery_records(raw)

    assert len(policies) == 1
    assert len(records) == 1
    assert len(rejected) == 1
    assert rejected[0]["name"] == "invalid_breadth_4"
    assert "min_breadth_for_2 must be 2 or 3" in rejected[0]["reason"]


def test_rdagent_schema_fails_closed_when_family_has_no_valid_proposal():
    from backtest.track_c_ranking_discovery.discovery_sandbox.discovery_runner import (
        normalize_discovery_records,
    )

    raw = [{
        "family": "industry_breadth",
        "name": "only_invalid",
        "hypothesis": "invalid family",
        "params": {
            "breadth_metric": "actionable_count",
            "allow_dynamic_2_plus_1": False,
            "min_breadth_for_2": 4,
        },
        "source_response_hash": "hash",
        "source_response_path": "rdagent_policy_discovery/raw/industry_breadth.txt",
        "source_model": "deepseek/deepseek-v4-pro",
    }]

    with pytest.raises(RuntimeError, match="zero executable proposals"):
        normalize_discovery_records(raw)



def test_discovery_controlled_eligibility_exactly_matches_panel_b0_eligible():
    from backtest.track_c_ranking_discovery.discovery_sandbox.discovery_runner import (
        controlled_eligibility_mask,
    )

    panel = pd.read_parquet(PANEL_SOURCE)
    mask = controlled_eligibility_mask(panel)
    expected = panel["b0_eligible"].fillna(False).astype(bool)

    assert np.array_equal(mask.to_numpy(dtype=bool), expected.to_numpy(dtype=bool))


def test_all_discovery_families_use_exact_common_b0_eligible_universe():
    from backtest.track_c_ranking_discovery.discovery_sandbox.discovery_runner import (
        IndustryBreadthChallenger,
        ContinuousScoreChallenger,
        MultiFeatureLinearChallenger,
        PortfolioUtilityChallenger,
        NovelHeuristicChallenger,
    )

    # DROP_EPS intentionally has much stronger factor values but is outside the
    # production B0 common universe. DROP_BLANK also has an invalid industry.
    df = pd.DataFrame({
        "code": ["KEEP", "DROP_EPS", "DROP_BLANK"],
        "industry": ["Industry A", "Industry B", ""],
        "b0_eligible": [True, False, False],
        "is_actionable": [1, 1, 1],
        "clear_geometry_failure": [0, 0, 0],
        "current_vs_ibd_candidate_pct": [1.0, 1.0, 1.0],
        "dist_to_52w_high_pct": [-2.0, 0.0, 0.0],
        "eps_yoy_growth": [25.0, 500.0, 500.0],
        "ibd_entry_volume_ratio": [1.5, 5.0, 5.0],
        "volume_ratio": [1.2, 5.0, 5.0],
        "mom_20": [0.05, 0.80, 0.90],
        "mom_60": [0.10, 1.20, 1.30],
        "pullback_v_is_dry": [True, True, True],
        "base_depth_pct": [20.0, 10.0, 10.0],
    })

    challengers = [
        IndustryBreadthChallenger(
            "eligibility_gate",
            breadth_metric="actionable_count",
            allow_dynamic_2_plus_1=False,
        ),
        ContinuousScoreChallenger(
            "eligibility_gate",
            {"mom_20": 2.0, "eps_yoy_growth": 1.0},
            selector_mode="pure_top3",
        ),
        MultiFeatureLinearChallenger(
            "eligibility_gate",
            ["mom_20", "eps_yoy_growth"],
            regularization=1.0,
            selector_mode="pure_top3",
        ),
        PortfolioUtilityChallenger(
            "eligibility_gate",
            concentration_lambda=1.0,
            stock_quality_metric="balanced",
        ),
        NovelHeuristicChallenger(
            "eligibility_gate",
            dry_weight=2.0,
            base_depth_penalty=1.0,
            volume_spike_bonus=2.0,
            selector_mode="pure_top3",
        ),
    ]

    for challenger in challengers:
        scored = challenger.score_candidates(df)
        quotas = challenger.allocate_industries(scored)
        picks = challenger.pick_stocks(scored, quotas)
        assert picks == ["KEEP"], f"{challenger.family} escaped the B0 common universe: {picks}"


def test_discovery_common_universe_fails_closed_without_b0_eligible():
    from backtest.track_c_ranking_discovery.discovery_sandbox.discovery_runner import (
        controlled_eligibility_mask,
    )

    with pytest.raises(RuntimeError, match="b0_eligible is missing"):
        controlled_eligibility_mask(pd.DataFrame({
            "code": ["AAA"],
            "industry": ["Industry A"],
        }))
