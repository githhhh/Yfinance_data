from __future__ import annotations

import pandas as pd

from dashboard.skill_industry_eps_known import select_skill_industry_eps_known
from backtest.track_c_ranking_discovery.config import PANEL_SOURCE

from backtest.track_e_soft_lane_audit.experiment import (
    build_event_summary,
    build_segments,
    collect_selection_events,
    load_panel,
)
from backtest.track_e_soft_lane_audit.policy import (
    PairwiseFreshStandardChallenger,
    baseline_policy,
    production_reference_policy,
    strictly_stronger_standard,
)


def _row(
    code: str,
    lane: str,
    rank: int,
    industry: str,
    *,
    risk: int,
    cur: float,
    entry_vol: float,
) -> dict:
    return {
        "code": code,
        "industry": industry,
        "industry_key": industry.lower(),
        "lane": lane,
        "raw_rank": rank,
        "sort_key": str((rank,)),
        "evidence_count": 0,
        "risk_count": risk,
        "current_vs_ibd_candidate_pct": cur,
        "ibd_entry_volume_ratio": entry_vol,
        "reason_codes": "",
        "risk_codes": "",
        "is_actionable": 1,
        "has_geom_failure": 0,
        "below_buy_point": 0,
        "has_known_eps": 1,
        "has_valid_industry": 1,
    }


def test_pareto_dominance_requires_no_worse_and_one_strict_improvement():
    fresh = _row(
        "FRESH", "fresh_demand_alpha", 1, "A",
        risk=0, cur=2.0, entry_vol=1.6,
    )
    stronger = _row(
        "STD", "standard_breakout", 4, "D",
        risk=0, cur=1.0, entry_vol=2.0,
    )
    volume_tradeoff = _row(
        "STD2", "standard_breakout", 5, "E",
        risk=0, cur=1.0, entry_vol=1.5,
    )
    equal = _row(
        "STD3", "standard_breakout", 6, "F",
        risk=0, cur=2.0, entry_vol=1.6,
    )

    assert strictly_stronger_standard(stronger, fresh) is True
    assert strictly_stronger_standard(volume_tradeoff, fresh) is False
    assert strictly_stronger_standard(equal, fresh) is False


def test_pairwise_challenger_replaces_only_selected_fresh_slot():
    scored = pd.DataFrame([
        _row("FRESH", "fresh_demand_alpha", 1, "A", risk=0, cur=2.0, entry_vol=1.6),
        _row("PULL", "constructive_pullback", 2, "B", risk=0, cur=1.0, entry_vol=2.0),
        _row("FRESH2", "fresh_demand_alpha", 3, "C", risk=0, cur=0.5, entry_vol=2.5),
        _row("STD", "standard_breakout", 4, "D", risk=0, cur=1.0, entry_vol=2.0),
    ])
    challenger = PairwiseFreshStandardChallenger()
    quotas = {x: 1 for x in scored["industry_key"]}

    picks = challenger.pick_stocks(scored, quotas)

    assert picks == ["STD", "PULL", "FRESH2"]


def test_pairwise_challenger_does_not_replace_when_standard_has_tradeoff():
    scored = pd.DataFrame([
        _row("FRESH", "fresh_demand_alpha", 1, "A", risk=0, cur=2.0, entry_vol=1.8),
        _row("PULL", "constructive_pullback", 2, "B", risk=0, cur=1.0, entry_vol=2.0),
        _row("FRESH2", "fresh_demand_alpha", 3, "C", risk=0, cur=0.5, entry_vol=2.5),
        _row("STD", "standard_breakout", 4, "D", risk=0, cur=1.0, entry_vol=1.7),
    ])
    challenger = PairwiseFreshStandardChallenger()
    quotas = {x: 1 for x in scored["industry_key"]}

    picks = challenger.pick_stocks(scored, quotas)

    assert picks == ["FRESH", "PULL", "FRESH2"]


def test_pairwise_challenger_preserves_distinct1_on_replacement():
    scored = pd.DataFrame([
        _row("FRESH", "fresh_demand_alpha", 1, "A", risk=0, cur=2.0, entry_vol=1.6),
        _row("PULL", "constructive_pullback", 2, "B", risk=0, cur=1.0, entry_vol=2.0),
        _row("FRESH2", "fresh_demand_alpha", 3, "C", risk=0, cur=0.5, entry_vol=2.5),
        _row("STD", "standard_breakout", 4, "B", risk=0, cur=1.0, entry_vol=2.0),
    ])
    challenger = PairwiseFreshStandardChallenger()
    quotas = {"a": 1, "b": 1, "c": 1}

    picks = challenger.pick_stocks(scored, quotas)

    assert picks == ["FRESH", "PULL", "FRESH2"]


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


def test_production_reference_anchor_matches_production_selector():
    panel = pd.read_parquet(PANEL_SOURCE)
    production = production_reference_policy()
    snaps = sorted(panel["snapshot_date"].astype(str).unique().tolist())[:8]

    for snap in snaps:
        s_df = panel[panel["snapshot_date"].astype(str) == snap].copy()
        prod_codes = [x.code for x in select_skill_industry_eps_known(s_df, limit=3)]
        scored = production.score_candidates(s_df)
        codes = production.pick_stocks(scored, production.allocate_industries(scored))
        assert codes == prod_codes


def test_dry_neutral_control_and_challenger_stay_in_common_b0_universe():
    panel = pd.read_parquet(PANEL_SOURCE)
    control = baseline_policy()
    challenger = PairwiseFreshStandardChallenger()
    snaps = sorted(panel["snapshot_date"].astype(str).unique().tolist())[:8]

    for snap in snaps:
        s_df = panel[panel["snapshot_date"].astype(str) == snap].copy()
        eligible = set(
            s_df[s_df["b0_eligible"].fillna(False).astype(bool)]["code"].astype(str)
        )

        for policy in (control, challenger):
            scored = policy.score_candidates(s_df)
            picks = policy.pick_stocks(scored, policy.allocate_industries(scored))
            assert set(picks).issubset(eligible)


def test_real_panel_event_collection_enforces_only_fresh_to_standard_changes():
    panel = load_panel()
    _, split = build_segments(panel)
    events = collect_selection_events(panel, split)

    changed = events[events["membership_changed_vs_control"] == True]
    for raw in changed["swap_pairs_json"].tolist():
        pairs = __import__("json").loads(raw)
        assert pairs
        for pair in pairs:
            assert pair["fresh_code"]
            assert pair["standard_code"]


def test_event_summary_counts_opportunities_and_actual_swaps_separately():
    events = pd.DataFrame([
        {
            "pairwise_opportunity_count": 2,
            "target_selection_swap": False,
            "swap_count": 0,
            "mean_swap_pair_delta_w4": None,
            "swap_pairs_json": "[]",
            "membership_changed_vs_control": False,
            "portfolio_spread_vs_control_w4": 0.0,
            "control_vs_production_order_changed": False,
            "control_vs_production_membership_changed": False,
            "segment": "confirmation",
            "w4_mature_vs_control": True,
        },
        {
            "pairwise_opportunity_count": 1,
            "target_selection_swap": True,
            "swap_count": 1,
            "mean_swap_pair_delta_w4": 3.0,
            "swap_pairs_json": (
                '[{"fresh_code":"F","standard_code":"S","pair_delta_w4":3.0}]'
            ),
            "membership_changed_vs_control": True,
            "portfolio_spread_vs_control_w4": 1.0,
            "control_vs_production_order_changed": False,
            "control_vs_production_membership_changed": False,
            "segment": "confirmation",
            "w4_mature_vs_control": True,
        },
    ])

    summary = build_event_summary(events)

    assert summary["opportunity_weeks"] == 2
    assert summary["opportunity_pairs"] == 3
    assert summary["actual_swap_weeks"] == 1
    assert summary["actual_swap_pairs"] == 1
    assert summary["swap_pair_mean_w4_delta"] == 3.0
