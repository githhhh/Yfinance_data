from __future__ import annotations

import pandas as pd

from dashboard.skill_industry_eps_known import SkillCandidate, select_skill_industry_eps_known
from backtest.track_c_ranking_discovery.b0_ablation_grid import reasoned_item_variant
from backtest.track_c_ranking_discovery.config import PANEL_SOURCE

from backtest.track_e_soft_lane_audit.config import TARGET_SOFT_LANES
from backtest.track_e_soft_lane_audit.experiment import (
    _rank_crossover_codes,
    build_event_summary,
    build_segments,
    load_panel,
)
from backtest.track_e_soft_lane_audit.policy import (
    PairwiseSoftLaneChallenger,
    baseline_policy,
    reorder_target_lanes,
)


def _item(code: str, lane: str, sort_key: tuple) -> SkillCandidate:
    return SkillCandidate(
        code=code,
        raw_rank=0,
        entry_status="ACTIONABLE",
        lane=lane,
        industry=f"Industry {code}",
        sort_key=sort_key,
        reason_codes=[],
        risk_codes=[],
        feature_values={"effective_eps_yoy_growth": 30.0},
    )


def test_pairwise_soft_lane_reorders_only_fresh_standard_slots():
    fresh = _item(
        "FRESH",
        "fresh_demand_alpha",
        (0, 0, 0, -3, 1, 0, 0, 0, -1.5, "FRESH", 0),
    )
    constructive = _item(
        "PULL",
        "constructive_pullback",
        (0, 1, 0, -3, 0, 0, 0, 0, -1.6, "PULL", 1),
    )
    standard = _item(
        "STD",
        "standard_breakout",
        (0, 2, 0, -4, 0, 0, 0, 0, -1.7, "STD", 2),
    )
    incomplete = _item(
        "INC",
        "incomplete_evidence",
        (0, 3, 0, -8, 0, 0, 0, 0, -2.0, "INC", 3),
    )

    reordered = reorder_target_lanes([fresh, constructive, standard, incomplete])

    assert [x.code for x in reordered] == ["STD", "PULL", "FRESH", "INC"]
    # Non-target absolute positions are frozen.
    assert reordered[1].code == "PULL"
    assert reordered[3].code == "INC"


def test_constructive_is_not_a_soft_target():
    assert set(TARGET_SOFT_LANES) == {"fresh_demand_alpha", "standard_breakout"}
    assert "constructive_pullback" not in TARGET_SOFT_LANES


def test_dry_false_is_neutral_but_true_keeps_positive_evidence():
    base = {
        "code": "TEST",
        "signal": True,
        "industry": "Industry A",
        "ibd_entry_status": "ACTIONABLE",
        "ibd_candidate_rule": "pivot",
        "current_vs_ibd_candidate_pct": 1.0,
        "ibd_entry_volume_ratio": 1.6,
        "volume_ratio": 1.2,
        "eps_yoy_growth": 30.0,
        "dist_to_52w_high_pct": -2.0,
        "ibd_entry_close_position": 0.85,
        "ibd_entry_breakout_range_ratio": 0.60,
    }

    false_item = reasoned_item_variant(
        pd.Series({**base, "pullback_v_is_dry": False}),
        0,
        dry_policy="reward_only",
        lane_policy="B0_LANE",
    )
    true_item = reasoned_item_variant(
        pd.Series({**base, "pullback_v_is_dry": True}),
        0,
        dry_policy="reward_only",
        lane_policy="B0_LANE",
    )

    assert "pullback_not_dry" not in false_item.risk_codes
    assert "dry_pullback" not in false_item.reason_codes
    assert "dry_pullback" in true_item.reason_codes


def test_rank_crossover_detects_standard_moving_above_fresh():
    scored = pd.DataFrame({
        "code": ["STD", "PULL", "FRESH"],
        "lane": ["standard_breakout", "constructive_pullback", "fresh_demand_alpha"],
        "skeleton_rank": [3, 2, 1],
        "raw_rank": [1, 2, 3],
    })
    standards, fresh = _rank_crossover_codes(scored)
    assert standards == ["STD"]
    assert fresh == ["FRESH"]


def test_event_summary_distinguishes_order_change_from_membership_change():
    events = pd.DataFrame([
        {
            "order_changed": True,
            "membership_changed": False,
            "target_selection_swap": False,
            "target_rank_crossover": True,
            "selection_pair_delta_w4": None,
            "rank_pair_delta_w4": 2.0,
            "portfolio_spread_w4": 0.0,
            "segment": "confirmation",
            "w4_mature": True,
        },
        {
            "order_changed": True,
            "membership_changed": True,
            "target_selection_swap": True,
            "target_rank_crossover": True,
            "selection_pair_delta_w4": 3.0,
            "rank_pair_delta_w4": 3.0,
            "portfolio_spread_w4": 1.0,
            "segment": "confirmation",
            "w4_mature": True,
        },
    ])
    summary = build_event_summary(events)
    assert summary["order_changed_weeks"] == 2
    assert summary["membership_changed_weeks"] == 1
    assert summary["target_rank_crossover_weeks"] == 2
    assert summary["target_selection_swap_weeks"] == 1


def test_track_e_segment_accounting_matches_track_d_40_snapshot_design():
    panel = load_panel()
    segments, split = build_segments(panel)

    assert len(split["all_used_snapshots"]) == 40
    assert len(segments["discovery_train_18"]) == 18
    assert len(segments["purge_4"]) == 4
    assert len(segments["screening_6"]) == 6
    assert len(segments["confirmation_12"]) == 12
    assert len(segments["locked_forward_18"]) == 18
    assert len(segments["retrospective_all_40"]) == 40


def test_pairwise_challenger_preserves_non_target_skeleton_positions():
    panel = pd.read_parquet(PANEL_SOURCE)
    challenger = PairwiseSoftLaneChallenger()
    snaps = sorted(panel["snapshot_date"].astype(str).unique().tolist())[:8]

    for snap in snaps:
        s_df = panel[panel["snapshot_date"].astype(str) == snap].copy()
        scored = challenger.score_candidates(s_df)
        non_target = scored[~scored["lane"].isin(TARGET_SOFT_LANES)]
        assert (
            non_target["raw_rank"].astype(int).to_numpy()
            == non_target["skeleton_rank"].astype(int).to_numpy()
        ).all()


def test_b0_anchor_still_matches_production_and_b01_stays_in_common_universe():
    panel = pd.read_parquet(PANEL_SOURCE)
    snaps = sorted(panel["snapshot_date"].astype(str).unique().tolist())[:8]
    baseline = baseline_policy()
    challenger = PairwiseSoftLaneChallenger()

    for snap in snaps:
        s_df = panel[panel["snapshot_date"].astype(str) == snap].copy()

        prod_codes = [x.code for x in select_skill_industry_eps_known(s_df, limit=3)]
        b0_scored = baseline.score_candidates(s_df)
        b0_codes = baseline.pick_stocks(b0_scored, baseline.allocate_industries(b0_scored))
        assert b0_codes == prod_codes

        b1_scored = challenger.score_candidates(s_df)
        b1_codes = challenger.pick_stocks(b1_scored, challenger.allocate_industries(b1_scored))
        eligible = set(
            s_df[s_df["b0_eligible"].fillna(False).astype(bool)]["code"].astype(str)
        )
        assert set(b1_codes).issubset(eligible)
