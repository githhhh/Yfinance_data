from __future__ import annotations

import pandas as pd

from dashboard.skill_industry_eps_known import select_skill_industry_eps_known
from backtest.track_c_ranking_discovery.config import PANEL_SOURCE

from backtest.track_f_lane_composition_audit.config import POLICY_SPECS
from backtest.track_f_lane_composition_audit.experiment import (
    build_segments,
    build_taxonomy_rows,
    historical_support_decision,
    load_panel,
    parity_anchor_audit,
)
from backtest.track_f_lane_composition_audit.policy import (
    CompositionSpec,
    LaneCompositionPolicy,
    all_policies,
    production_baseline,
)
from backtest.track_f_lane_composition_audit.taxonomy import classify_lane_facts


def _base_row(**overrides) -> pd.Series:
    data = {
        "snapshot_date": "2026-01-02",
        "code": "TEST",
        "signal": True,
        "industry": "Industry A",
        "b0_eligible": True,
        "ibd_entry_status": "ACTIONABLE",
        "ibd_candidate_rule": "ceiling_breakout",
        "current_vs_ibd_candidate_pct": 1.0,
        "ibd_entry_volume_ratio": 1.6,
        "volume_ratio": 1.0,
        "eps_yoy_growth": 10.0,
        "dist_to_52w_high_pct": -2.0,
        "ibd_entry_close_position": 0.85,
        "ibd_entry_breakout_range_ratio": 0.60,
        "pullback_v_is_dry": None,
    }
    data.update(overrides)
    return pd.Series(data)


def test_orthogonal_taxonomy_fresh_is_confirmed_non_pullback():
    facts = classify_lane_facts(
        _base_row(eps_yoy_growth=35.0),
        0,
    )
    assert facts.current_lane == "fresh_demand_alpha"
    assert facts.setup_route == "non_pullback"
    assert facts.quality_state == "confirmed"
    assert facts.composition_group == "confirmed_non_pullback"


def test_standard_breakout_can_be_pullback_route():
    facts = classify_lane_facts(
        _base_row(
            ibd_candidate_rule="pivot",
            eps_yoy_growth=10.0,
            volume_ratio=1.0,
            pullback_v_is_dry=True,
        ),
        0,
    )
    assert facts.current_lane == "standard_breakout"
    assert facts.setup_route == "pullback"
    assert facts.quality_state == "standard"
    assert facts.composition_group == "standard"


def test_constructive_pullback_has_actionable_confirmed_branch():
    facts = classify_lane_facts(
        _base_row(
            ibd_candidate_rule="pivot",
            eps_yoy_growth=35.0,
            pullback_v_is_dry=True,
        ),
        0,
    )
    assert facts.current_lane == "constructive_pullback"
    assert facts.setup_route == "pullback"
    assert facts.quality_state == "confirmed"
    assert facts.actionable_pullback_context_branch is True
    assert facts.non_actionable_pullback_context_branch is False


def test_constructive_pullback_has_non_actionable_context_branch():
    facts = classify_lane_facts(
        _base_row(
            ibd_entry_status="UNCONFIRMED",
            b0_eligible=False,
            ibd_candidate_rule="pivot",
            ibd_entry_volume_ratio=1.0,
            eps_yoy_growth=10.0,
            pullback_v_is_dry=True,
        ),
        0,
    )
    assert facts.current_lane == "constructive_pullback"
    assert facts.entry_status == "UNCONFIRMED"
    assert facts.non_actionable_pullback_context_branch is True
    assert facts.actionable_pullback_context_branch is False


def _scored_row(
    code: str,
    rank: int,
    group: str,
    quality: str,
    industry: str,
) -> dict:
    return {
        "code": code,
        "industry": industry,
        "industry_key": industry.lower(),
        "b0_eligible": True,
        "raw_rank": rank,
        "quality_state": quality,
        "composition_group": group,
    }


def test_confirmed_only_takes_up_to_three_from_both_confirmed_routes():
    scored = pd.DataFrame([
        _scored_row("F1", 1, "confirmed_non_pullback", "confirmed", "A"),
        _scored_row("F2", 2, "confirmed_non_pullback", "confirmed", "B"),
        _scored_row("C1", 3, "confirmed_pullback", "confirmed", "C"),
        _scored_row("S1", 4, "standard", "standard", "D"),
    ])
    policy = LaneCompositionPolicy(
        CompositionSpec("TEST", "confirmed_only", "distinct_1", "primary")
    )
    quotas = policy.allocate_industries(scored)
    assert policy.pick_stocks(scored, quotas) == ["F1", "F2", "C1"]


def test_fcs_max1_does_not_force_duplicate_group():
    scored = pd.DataFrame([
        _scored_row("F1", 1, "confirmed_non_pullback", "confirmed", "A"),
        _scored_row("F2", 2, "confirmed_non_pullback", "confirmed", "B"),
        _scored_row("C1", 3, "confirmed_pullback", "confirmed", "C"),
        _scored_row("S1", 4, "standard", "standard", "D"),
    ])
    policy = LaneCompositionPolicy(
        CompositionSpec("TEST", "fcs_max1", "distinct_1", "primary")
    )
    quotas = policy.allocate_industries(scored)
    assert policy.pick_stocks(scored, quotas) == ["F1", "C1", "S1"]


def test_fcs_max1_does_not_force_fill_when_group_absent():
    scored = pd.DataFrame([
        _scored_row("F1", 1, "confirmed_non_pullback", "confirmed", "A"),
        _scored_row("F2", 2, "confirmed_non_pullback", "confirmed", "B"),
    ])
    policy = LaneCompositionPolicy(
        CompositionSpec("TEST", "fcs_max1", "distinct_1", "primary")
    )
    quotas = policy.allocate_industries(scored)
    assert policy.pick_stocks(scored, quotas) == ["F1"]


def test_secondary_no_industry_variant_removes_only_industry_quota():
    scored = pd.DataFrame([
        _scored_row("F1", 1, "confirmed_non_pullback", "confirmed", "A"),
        _scored_row("F2", 2, "confirmed_non_pullback", "confirmed", "A"),
        _scored_row("C1", 3, "confirmed_pullback", "confirmed", "B"),
    ])
    distinct = LaneCompositionPolicy(
        CompositionSpec("D", "confirmed_only", "distinct_1", "primary")
    )
    no_ind = LaneCompositionPolicy(
        CompositionSpec("N", "confirmed_only", "pure_top3", "secondary")
    )

    d_q = distinct.allocate_industries(scored)
    n_q = no_ind.allocate_industries(scored)
    assert distinct.pick_stocks(scored, d_q) == ["F1", "C1"]
    assert no_ind.pick_stocks(scored, n_q) == ["F1", "F2", "C1"]


def test_track_f_has_exactly_three_primary_and_three_secondary_specs():
    assert len(POLICY_SPECS) == 6
    roles = [spec[3] for spec in POLICY_SPECS]
    assert roles.count("primary") == 3
    assert roles.count("secondary") == 3


def test_track_f_reuses_track_d_40_snapshot_regime_slices():
    panel = load_panel()
    segments, split = build_segments(panel)
    assert len(split["all_used_snapshots"]) == 40
    assert len(segments["discovery_train_18"]) == 18
    assert len(segments["purge_4"]) == 4
    assert len(segments["screening_6"]) == 6
    assert len(segments["confirmation_12"]) == 12
    assert len(segments["locked_forward_18"]) == 18
    assert len(segments["retrospective_track_d_40"]) == 40


def test_b0_eligible_current_lane_mapping_is_orthogonally_consistent():
    panel = load_panel()
    rows = build_taxonomy_rows(panel)
    eligible = rows[rows["b0_eligible"] == True]

    assert not eligible.empty
    fresh = eligible[eligible["current_lane"] == "fresh_demand_alpha"]
    constructive = eligible[eligible["current_lane"] == "constructive_pullback"]
    standard = eligible[eligible["current_lane"] == "standard_breakout"]

    assert (fresh["composition_group"] == "confirmed_non_pullback").all()
    assert (constructive["composition_group"] == "confirmed_pullback").all()
    assert (standard["composition_group"] == "standard").all()


def test_normalized_parity_exactly_anchors_track_c_pullback_parity():
    panel = load_panel()
    audit = parity_anchor_audit(panel)
    assert audit["mismatch_count"] == 0


def test_production_baseline_anchor_matches_production_top3():
    panel = pd.read_parquet(PANEL_SOURCE)
    baseline = production_baseline()

    for snapshot in sorted(panel["snapshot_date"].astype(str).unique().tolist()):
        s_df = panel[panel["snapshot_date"].astype(str) == snapshot].copy()
        prod = [x.code for x in select_skill_industry_eps_known(s_df, limit=3)]
        scored = baseline.score_candidates(s_df)
        picks = baseline.pick_stocks(scored, baseline.allocate_industries(scored))
        assert picks == prod



def test_historical_support_gate_is_primary_only_and_fail_closed():
    rows = []
    for policy_id, role in [
        ("TRACK_F__GOOD", "primary"),
        ("TRACK_F__BAD", "primary"),
        ("TRACK_F__SECONDARY", "secondary"),
    ]:
        for segment, support in [
            ("retrospective_track_d_40", 40),
            ("locked_forward_18", 18),
        ]:
            good = policy_id != "TRACK_F__BAD"
            rows.append({
                "policy_id": policy_id,
                "role": role,
                "segment": segment,
                "support_weeks": support,
                "mean_spread": 0.2 if good else -0.1,
                "median_spread": 0.1 if good else -0.2,
                "cvar_delta": 0.0,
                "stop_delta_pct": 0.0,
                "one_pick_ruins_delta_pct": 0.0,
                "ci_low": -0.5,
            })

    decision = historical_support_decision(pd.DataFrame(rows))

    assert decision["shadow_candidates"] == ["TRACK_F__GOOD"]
    verdicts = {
        row["policy_id"]: row["verdict"]
        for row in decision["policy_verdicts"]
    }
    assert verdicts["TRACK_F__GOOD"] == "HISTORICAL_SHADOW_CANDIDATE"
    assert verdicts["TRACK_F__BAD"] == "NO_HISTORICAL_SUPPORT_TO_REPLACE_B0"
    assert "TRACK_F__SECONDARY" not in verdicts



def test_track_f_rankers_do_not_expose_future_outcome_columns():
    panel = load_panel()
    snapshot = sorted(panel["snapshot_date"].astype(str).unique().tolist())[0]
    s_df = panel[panel["snapshot_date"].astype(str) == snapshot].copy()

    for policy in all_policies():
        scored = policy.score_candidates(s_df)
        forbidden = [
            col for col in scored.columns
            if col.lower().startswith(("w1_", "w2_", "w4_"))
            or "return" in col.lower()
            or "stop8" in col.lower()
        ]
        assert forbidden == [], f"{policy.policy_id} leaked outcome columns: {forbidden}"
