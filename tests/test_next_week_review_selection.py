import pandas as pd

from backtest.next_week_review_selection.asof import panel_asof_cutoff
from backtest.next_week_review_selection.coverage import build_price_path_audits
from backtest.next_week_review_selection.labels import add_forward_labels
from backtest.next_week_review_selection.optimizer import selection_signature
from backtest.next_week_review_selection.oracle import add_weekly_oracle_flags
from backtest.next_week_review_selection.search_space import (
    generate_core_rules,
    generate_evidence_ablations,
)
from backtest.next_week_review_selection.selectors import (
    ReviewRule,
    evidence_family_count,
    primary_rule,
    select_review_variant,
)
from backtest.next_week_review_selection.walk_forward import (
    adaptive_policy_status,
    partition_walk_forward_weeks,
)


def test_actionable_rows_are_never_refiltered_by_research_rule():
    pool = pd.DataFrame([
        _row(
            "ACT", "ACTIONABLE", 3.0,
            entry_volume=None, weekly_volume=0.8, eps=None,
            dist=-20, close_position=0.50,
        )
    ])
    selected = select_review_variant(pool, primary_rule())
    assert selected["code"].tolist() == ["ACT"]


def test_primary_r1_does_not_geometry_hard_reject_supplemental():
    pool = pd.DataFrame([
        _row(
            "UNC", "UNCONFIRMED", -1.0,
            weekly_volume=1.5, close_position=0.60,
        )
    ])
    assert select_review_variant(pool, primary_rule())["code"].tolist() == ["UNC"]


def test_geometry_is_available_as_independent_ablation():
    pool = pd.DataFrame([
        _row(
            "UNC", "UNCONFIRMED", -1.0,
            weekly_volume=1.5, close_position=0.60,
        )
    ])
    rule = ReviewRule(name="GX", exclude_clear_geometry_failure=True)
    assert select_review_variant(pool, rule).empty


def test_volume_facts_count_as_one_independent_evidence_family():
    row = pd.Series(_row(
        "A", "UNCONFIRMED", -1.0,
        entry_volume=2.0, weekly_volume=2.0,
        eps=None, dist=-10,
    ))
    assert evidence_family_count(row) == 1


def test_false_or_missing_supply_evidence_is_neutral():
    false_dry = pd.Series(_row(
        "A", "UNCONFIRMED", -1.0, rule="pivot", dry=False,
        entry_volume=None, weekly_volume=1.4, eps=None, dist=-10,
    ))
    missing_dry = pd.Series(_row(
        "B", "UNCONFIRMED", -1.0, rule="pivot", dry=None,
        entry_volume=None, weekly_volume=1.4, eps=None, dist=-10,
    ))
    assert evidence_family_count(false_dry) == 1
    assert evidence_family_count(missing_dry) == 1


def test_extended_is_not_in_core_selector():
    pool = pd.DataFrame([
        _row("EXT", "EXTENDED", 6.0, weekly_volume=2.0),
        _row("UNC", "UNCONFIRMED", -1.0, weekly_volume=2.0),
    ])
    assert select_review_variant(pool, primary_rule())["code"].tolist() == ["UNC"]


def test_c_rank_never_changes_selection():
    pool = pd.DataFrame([
        _row("A", "UNCONFIRMED", -1.0, c_rank=999),
        _row("B", "UNCONFIRMED", -6.0, c_rank=1),
    ])
    before = select_review_variant(pool, primary_rule())["code"].tolist()
    pool["rank_C_continuous"] = [1, 999]
    after = select_review_variant(pool, primary_rule())["code"].tolist()
    assert before == after == ["A"]


def test_dual_clock_and_horizon_end_dates_for_delayed_opportunity():
    events = pd.DataFrame([
        _row(
            "A", "UNCONFIRMED", -2.0,
            pivot=100.0, latest_close=98.0,
        )
    ])
    dates = pd.bdate_range("2026-08-31", periods=25)
    close = [98.5, 99.0, 99.5, 100.5] + [101.0 + i for i in range(21)]
    bars = pd.DataFrame(
        {
            "Open": [value - 0.2 for value in close],
            "High": [value + 0.5 for value in close],
            "Low": [value - 0.5 for value in close],
            "Close": close,
        },
        index=dates,
    )
    labeled = add_forward_labels(events, {"A": bars}).iloc[0]
    assert labeled["review_opportunity_1w"] == True
    assert labeled["opportunity_delay_sessions"] == 4
    assert labeled["opp_forward_4w_censored"] == False


def test_price_path_audit_marks_missing_symbol():
    events = pd.DataFrame([_row("MISS", "UNCONFIRMED", -1.0)])
    labeled = add_forward_labels(events, {})
    assert labeled.loc[0, "price_path_state"] == "MISSING_SYMBOL"
    audits = build_price_path_audits(labeled)
    assert audits["price_path_coverage_summary.csv"].iloc[0]["complete_1w_count"] == 0


def test_asof_masks_future_4w_but_keeps_resolved_1w():
    events = pd.DataFrame([_row("A", "ACTIONABLE", 1.0)])
    dates = pd.bdate_range("2026-08-31", periods=25)
    bars = pd.DataFrame(
        {
            "Open": [101.0] * 25,
            "High": [103.0] * 25,
            "Low": [100.0] * 25,
            "Close": [102.0] * 25,
        },
        index=dates,
    )
    panel = add_weekly_oracle_flags(add_forward_labels(events, {"A": bars}))
    asof = panel_asof_cutoff(panel, "2026-09-11").iloc[0]
    assert asof["forward_1w_censored"] == False
    assert asof["forward_4w_censored"] == True


def test_two_stage_search_space_is_small():
    core = generate_core_rules()
    assert len(core) == 24
    assert len(generate_evidence_ablations(core[0])) == 5


def test_selection_signature_collapses_behaviorally_identical_rules():
    pool = pd.DataFrame([
        _row("A", "UNCONFIRMED", -1.0),
        _row("B", "ACTIONABLE", 2.0),
    ])
    r3 = ReviewRule(name="N3", near_below_pct=3.0)
    r5 = ReviewRule(name="N5", near_below_pct=5.0)
    assert selection_signature(pool, r3) == selection_signature(pool, r5)


def test_walk_forward_partition_uses_five_full_folds_and_two_week_tail():
    weeks = [f"2026-W{i:02d}" for i in range(42)]
    formal, tail = partition_walk_forward_weeks(
        weeks, min_train_weeks=20, test_weeks=4
    )
    assert len(formal) == 5
    assert all(len(block["test_weeks"]) == 4 for block in formal)
    assert len(tail) == 2


def test_adaptive_policy_gate_can_pass_with_different_fold_rules():
    summary = pd.DataFrame([
        {
            "folds": 5,
            "opportunity_positive_rate": 0.8,
            "tradable_winner_lift_nonnegative_rate": 0.6,
            "tradable_loser_lift_nonworse_rate": 0.6,
            "mean_opportunity_delta": 0.2,
            "mean_tradable_winner_lift_delta": 0.03,
            "mean_tradable_loser_lift_delta": -0.02,
            "mean_incremental_opportunity_efficiency": 0.4,
            "mean_tradable_winner_lift_delta_2w": 0.01,
            "mean_tradable_loser_lift_delta_2w": -0.01,
            "mean_tradable_winner_lift_delta_3w": 0.02,
            "mean_tradable_loser_lift_delta_3w": -0.02,
            "mean_tradable_winner_lift_delta_4w": 0.03,
            "mean_tradable_loser_lift_delta_4w": -0.01,
        }
    ])
    assert adaptive_policy_status(summary) == "RETROSPECTIVE_ADAPTIVE_CANDIDATE"



def test_adaptive_policy_gate_rejects_single_horizon_degradation():
    summary = pd.DataFrame([
        {
            "folds": 5,
            "opportunity_positive_rate": 0.8,
            "tradable_winner_lift_nonnegative_rate": 0.8,
            "tradable_loser_lift_nonworse_rate": 0.8,
            "mean_opportunity_delta": 0.2,
            "mean_tradable_winner_lift_delta": 0.03,
            "mean_tradable_loser_lift_delta": -0.02,
            "mean_incremental_opportunity_efficiency": 0.4,
            "mean_tradable_winner_lift_delta_2w": -0.01,
            "mean_tradable_loser_lift_delta_2w": -0.01,
            "mean_tradable_winner_lift_delta_3w": 0.02,
            "mean_tradable_loser_lift_delta_3w": -0.02,
            "mean_tradable_winner_lift_delta_4w": 0.03,
            "mean_tradable_loser_lift_delta_4w": -0.01,
        }
    ])
    assert adaptive_policy_status(summary) == "NO_STABLE_ADAPTIVE_POLICY"

def _row(
    code,
    status,
    vs_buy,
    *,
    rule="ceiling",
    pivot=100.0,
    latest_close=101.0,
    entry_volume=1.6,
    weekly_volume=1.4,
    eps=30.0,
    dist=-1.0,
    dry=None,
    close_position=0.9,
    range_ratio=0.7,
    c_rank=10,
):
    return {
        "snapshot_date": "2026-08-28",
        "code": code,
        "signal": True,
        "signal_source": "test",
        "ibd_candidate_rule": rule,
        "ibd_candidate_price": pivot,
        "ibd_trigger_price": pivot,
        "latest_close": latest_close,
        "ibd_entry_status": status,
        "current_vs_ibd_candidate_pct": vs_buy,
        "ibd_entry_volume_ratio": entry_volume,
        "ibd_entry_close_position": close_position,
        "ibd_entry_breakout_range_ratio": range_ratio,
        "volume_ratio": weekly_volume,
        "dist_to_52w_high_pct": dist,
        "pullback_v_is_dry": dry,
        "pit_eps_state": "VERIFIED" if eps is not None else "UNKNOWN",
        "pit_eps_yoy_growth": eps,
        "rank_C_continuous": c_rank,
        "C_continuous": float(c_rank),
    }
