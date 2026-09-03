import pandas as pd

from backtest.next_week_review_selection.asof import panel_asof_cutoff
from backtest.next_week_review_selection.coverage import build_price_path_audits
from backtest.next_week_review_selection.labels import add_forward_labels
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
    assert selected["selection_source"].tolist() == ["ACTIONABLE_BASELINE"]


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
    selected = select_review_variant(pool, primary_rule())
    assert selected["code"].tolist() == ["UNC"]


def test_below_trigger_near_buy_point_can_be_supplemented():
    pool = pd.DataFrame([
        _row(
            "BEL", "BELOW_TRIGGER", -4.0,
            entry_volume=None, weekly_volume=1.5, eps=None, dist=-10,
        )
    ])
    assert select_review_variant(pool, primary_rule())["code"].tolist() == ["BEL"]


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
    assert labeled["opportunity_anchor_date"] == "2026-09-03"
    assert labeled["forward_1w_end_date"] == "2026-09-04"
    assert labeled["opp_forward_4w_censored"] == False
    assert pd.notna(labeled["opp_forward_4w_end_date"])


def test_price_path_audit_marks_missing_symbol():
    events = pd.DataFrame([_row("MISS", "UNCONFIRMED", -1.0)])
    labeled = add_forward_labels(events, {})
    assert labeled.loc[0, "price_path_state"] == "MISSING_SYMBOL"
    audits = build_price_path_audits(labeled)
    summary = audits["price_path_coverage_summary.csv"].iloc[0]
    assert summary["complete_1w_count"] == 0
    assert len(audits["price_path_missing_1w_cases.csv"]) == 1


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
    assert pd.isna(asof["forward_4w_return_pct"])


def test_oracle_marks_snapshot_and_opportunity_winners():
    rows = []
    for i in range(6):
        row = _row(f"T{i}", "ACTIONABLE", 1.0)
        row.update({
            "review_opportunity_1w": True,
            "forward_1w_censored": False,
            "forward_1w_return_pct": float(i),
            "mfe_1w_pct": float(i + 1),
            "mae_1w_pct": float(-i),
            "opp_forward_1w_censored": False,
            "opp_forward_1w_return_pct": float(i),
            "opp_mfe_1w_pct": float(i + 1),
            "opp_mae_1w_pct": float(-i),
        })
        for horizon in ("2w", "3w", "4w"):
            row[f"forward_{horizon}_censored"] = True
            row[f"opp_forward_{horizon}_censored"] = True
        rows.append(row)
    panel = add_weekly_oracle_flags(pd.DataFrame(rows))
    assert panel["big_winner_any_1w"].sum() >= 5
    assert panel["opp_big_winner_any_1w"].sum() >= 5


def test_two_stage_search_space_is_small():
    core = generate_core_rules()
    assert len(core) == 24
    assert len({rule.name for rule in core}) == 24
    ablations = generate_evidence_ablations(core[0])
    assert len(ablations) == 5
    assert len({rule.name for rule in ablations}) == 5


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
