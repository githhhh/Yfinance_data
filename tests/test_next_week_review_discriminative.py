import pandas as pd

from backtest.next_week_review_selection.discriminative import (
    B0_NAME,
    MAX_ATTENTION_MULTIPLIER,
    RefinementRule,
    anchor_rule,
    candidate_library,
    choose_train_refinement,
    oos_policy_status,
    select_refined,
)
from backtest.next_week_review_selection.utils import (
    ZERO_TOL,
    is_nonnegative,
    is_nonpositive,
    is_positive,
)


def test_anchor_is_fixed_near5_ub_e2_geometry_allow():
    rule = anchor_rule()
    assert rule.near_below_pct == 5.0
    assert rule.supplemental_statuses == ("UNCONFIRMED", "BELOW_TRIGGER")
    assert rule.min_evidence_families == 2
    assert rule.exclude_clear_geometry_failure is False


def test_candidate_library_is_small_and_max_two_conditions():
    rules = candidate_library()
    assert len(rules) == 25
    assert len({rule.name for rule in rules}) == 25
    assert max(len(rule.conditions) for rule in rules) <= 2


def test_refinement_filters_only_supplemental_but_keeps_actionable():
    pool = pd.DataFrame([
        _row("ACT", "ACTIONABLE", 2.0, base_depth=-80, rs=-20),
        _row("KEEP", "UNCONFIRMED", 1.0, base_depth=-25, rs=-2),
        _row("DROP", "UNCONFIRMED", 1.0, base_depth=-60, rs=-2),
    ])
    rule = RefinementRule(
        name="TEST_BASE33",
        conditions=("BASE_DEPTH_LE_33",),
    )
    selected = select_refined(pool, rule)
    assert selected["code"].tolist() == ["ACT", "KEEP"]


def test_missing_refinement_field_does_not_qualify():
    pool = pd.DataFrame([
        _row("A", "UNCONFIRMED", 1.0, base_depth=None, rs=-2),
    ])
    rule = RefinementRule(
        name="TEST_BASE33",
        conditions=("BASE_DEPTH_LE_33",),
    )
    assert select_refined(pool, rule).empty


def test_b0_no_expansion_is_valid_fallback():
    pool = pd.DataFrame([
        _row("ACT", "ACTIONABLE", 2.0),
        _row("UNC", "UNCONFIRMED", 1.0),
    ])
    selected = select_refined(pool, None)
    assert selected["code"].tolist() == ["ACT"]
    assert selected["variant"].tolist() == [B0_NAME]


def test_numeric_zero_tolerance_treats_machine_epsilon_as_zero():
    eps = 2.220446049250313e-16
    assert ZERO_TOL > eps
    assert is_nonnegative(-eps)
    assert is_nonnegative(eps)
    assert is_nonpositive(-eps)
    assert is_nonpositive(eps)
    assert not is_positive(eps)


def test_policy_status_accepts_machine_epsilon_loser_delta_as_nonworse():
    eps = 2.220446049250313e-16
    summary = pd.DataFrame([
        {
            "evaluation_role": "STATIC_DISCOVERY_RULE",
            "folds": 5,
            "expanded_fold_rate": 0.8,
            "opportunity_positive_rate": 0.8,
            "winner_lift_nonnegative_rate": 0.8,
            "loser_lift_nonworse_rate": 0.8,
            "mean_opportunity_delta": 0.10,
            "mean_winner_lift_delta": 0.03,
            "mean_loser_lift_delta": eps,
            "mean_attention_multiplier_vs_b0": 1.3,
            "mean_incremental_opportunities_per_added_review": 0.8,
            "mean_winner_lift_delta_2w": 0.01,
            "mean_loser_lift_delta_2w": eps,
            "mean_winner_lift_delta_3w": 0.02,
            "mean_loser_lift_delta_3w": -0.02,
            "mean_winner_lift_delta_4w": 0.03,
            "mean_loser_lift_delta_4w": -0.01,
        }
    ])
    assert oos_policy_status(
        summary, static_rule_exists=True
    ) == "RETROSPECTIVE_DISCRIMINATIVE_CANDIDATE"


def test_policy_status_rejects_hidden_2w_degradation():
    summary = pd.DataFrame([
        {
            "evaluation_role": "STATIC_DISCOVERY_RULE",
            "folds": 5,
            "expanded_fold_rate": 1.0,
            "opportunity_positive_rate": 0.8,
            "winner_lift_nonnegative_rate": 0.8,
            "loser_lift_nonworse_rate": 0.8,
            "mean_opportunity_delta": 0.10,
            "mean_winner_lift_delta": 0.03,
            "mean_loser_lift_delta": -0.02,
            "mean_attention_multiplier_vs_b0": 1.3,
            "mean_incremental_opportunities_per_added_review": 0.8,
            "mean_winner_lift_delta_2w": -0.01,
            "mean_loser_lift_delta_2w": -0.01,
            "mean_winner_lift_delta_3w": 0.02,
            "mean_loser_lift_delta_3w": -0.02,
            "mean_winner_lift_delta_4w": 0.03,
            "mean_loser_lift_delta_4w": -0.01,
        }
    ])
    assert oos_policy_status(
        summary, static_rule_exists=True
    ) == "NO_STABLE_DISCRIMINATIVE_RULE"


def test_policy_status_passes_attention_capped_horizon_consistent_rule():
    summary = pd.DataFrame([
        {
            "evaluation_role": "STATIC_DISCOVERY_RULE",
            "folds": 5,
            "expanded_fold_rate": 0.8,
            "opportunity_positive_rate": 0.8,
            "winner_lift_nonnegative_rate": 0.8,
            "loser_lift_nonworse_rate": 0.8,
            "mean_opportunity_delta": 0.10,
            "mean_winner_lift_delta": 0.03,
            "mean_loser_lift_delta": -0.02,
            "mean_attention_multiplier_vs_b0": MAX_ATTENTION_MULTIPLIER,
            "mean_incremental_opportunities_per_added_review": 0.8,
            "mean_winner_lift_delta_2w": 0.01,
            "mean_loser_lift_delta_2w": -0.01,
            "mean_winner_lift_delta_3w": 0.02,
            "mean_loser_lift_delta_3w": -0.02,
            "mean_winner_lift_delta_4w": 0.03,
            "mean_loser_lift_delta_4w": -0.01,
        }
    ])
    assert oos_policy_status(
        summary, static_rule_exists=True
    ) == "RETROSPECTIVE_DISCRIMINATIVE_CANDIDATE"


def test_choose_train_refinement_can_abstain_to_b0():
    panel = pd.DataFrame([
        _row("ACT", "ACTIONABLE", 1.0),
    ])
    rule, grid = choose_train_refinement(panel)
    assert rule is None
    assert not grid.empty


def _row(
    code,
    status,
    vs_buy,
    *,
    base_depth=-25.0,
    base_duration=20.0,
    pullback_pct=-10.0,
    pullback_duration=5.0,
    rs=-2.0,
    volume=1.5,
    eps=30.0,
    dry=True,
):
    return {
        "snapshot_date": "2026-01-02",
        "code": code,
        "signal": True,
        "signal_source": "test",
        "ibd_candidate_rule": "pivot",
        "ibd_candidate_price": 100.0,
        "latest_close": 101.0,
        "ibd_entry_status": status,
        "current_vs_ibd_candidate_pct": vs_buy,
        "ibd_entry_volume_ratio": volume,
        "ibd_entry_close_position": 0.9,
        "ibd_entry_breakout_range_ratio": 0.7,
        "volume_ratio": volume,
        "base_depth_pct": base_depth,
        "base_duration_weeks": base_duration,
        "pullback_pct": pullback_pct,
        "pullback_duration_weeks": pullback_duration,
        "dist_to_52w_high_pct": rs,
        "pullback_v_is_dry": dry,
        "pit_eps_state": "VERIFIED",
        "pit_eps_yoy_growth": eps,
        "forward_1w_censored": True,
        "forward_2w_censored": True,
        "forward_3w_censored": True,
        "forward_4w_censored": True,
        "review_opportunity_1w": False,
        "opp_forward_1w_censored": True,
        "opp_forward_2w_censored": True,
        "opp_forward_3w_censored": True,
        "opp_forward_4w_censored": True,
        "big_winner_any_1w": False,
        "big_loser_any_1w": False,
        "big_winner_any_2w": False,
        "big_loser_any_2w": False,
        "big_winner_any_3w": False,
        "big_loser_any_3w": False,
        "big_winner_any_4w": False,
        "big_loser_any_4w": False,
        "opp_big_winner_any_1w": False,
        "opp_big_loser_any_1w": False,
        "opp_big_winner_any_2w": False,
        "opp_big_loser_any_2w": False,
        "opp_big_winner_any_3w": False,
        "opp_big_loser_any_3w": False,
        "opp_big_winner_any_4w": False,
        "opp_big_loser_any_4w": False,
        "opp_severe_loser_1w": False,
        "opp_severe_loser_2w": False,
        "opp_severe_loser_3w": False,
        "opp_severe_loser_4w": False,
    }
