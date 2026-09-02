import pandas as pd

from backtest.next_week_review_selection.labels import add_forward_labels
from backtest.next_week_review_selection.oracle import add_weekly_oracle_flags
from backtest.next_week_review_selection.search_space import generate_candidate_rules
from backtest.next_week_review_selection.selectors import (
    ReviewRule,
    primary_rule,
    select_review_variant,
    support_count,
)


def test_actionable_rows_are_never_refiltered_by_research_rule():
    pool = pd.DataFrame([
        _row("ACT", "ACTIONABLE", 3.0, entry_volume=None, weekly_volume=0.8,
             eps=None, dist=-20, close_position=0.50)
    ])
    selected = select_review_variant(pool, primary_rule())
    assert selected["code"].tolist() == ["ACT"]
    assert selected["selection_source"].tolist() == ["ACTIONABLE_BASELINE"]


def test_near_unconfirmed_with_one_positive_evidence_is_supplemented():
    pool = pd.DataFrame([
        _row("ACT", "ACTIONABLE", 2.0),
        _row("UNC", "UNCONFIRMED", -2.0, entry_volume=None,
             weekly_volume=1.4, eps=None, dist=-10),
    ])
    selected = select_review_variant(pool, primary_rule())
    assert set(selected["code"]) == {"ACT", "UNC"}
    assert selected.loc[selected["code"].eq("UNC"), "selection_source"].item() == "SUPPLEMENTAL"


def test_false_or_missing_positive_evidence_is_neutral_not_negative():
    false_dry = pd.Series(_row(
        "A", "UNCONFIRMED", -1.0, rule="pivot", dry=False,
        entry_volume=None, weekly_volume=1.4, eps=None, dist=-10
    ))
    missing_dry = pd.Series(_row(
        "B", "UNCONFIRMED", -1.0, rule="pivot", dry=None,
        entry_volume=None, weekly_volume=1.4, eps=None, dist=-10
    ))
    assert support_count(false_dry) == 1
    assert support_count(missing_dry) == 1


def test_extended_is_not_in_core_selector():
    pool = pd.DataFrame([
        _row("EXT", "EXTENDED", 6.0, weekly_volume=2.0),
        _row("UNC", "UNCONFIRMED", -1.0, weekly_volume=2.0),
    ])
    selected = select_review_variant(pool, primary_rule())
    assert selected["code"].tolist() == ["UNC"]


def test_below_trigger_near_buy_point_can_be_supplemented():
    pool = pd.DataFrame([
        _row("BEL", "BELOW_TRIGGER", -4.0, entry_volume=None,
             weekly_volume=1.5, eps=None, dist=-10)
    ])
    assert select_review_variant(pool, primary_rule())["code"].tolist() == ["BEL"]


def test_clear_geometry_failure_only_blocks_supplemental_when_enabled():
    pool = pd.DataFrame([
        _row("FAIL", "UNCONFIRMED", -1.0, weekly_volume=1.5, close_position=0.60)
    ])
    assert select_review_variant(pool, primary_rule()).empty
    allow = ReviewRule(name="ALLOW", exclude_clear_geometry_failure=False)
    assert select_review_variant(pool, allow)["code"].tolist() == ["FAIL"]


def test_c_rank_never_changes_selection():
    pool = pd.DataFrame([
        _row("A", "UNCONFIRMED", -1.0, c_rank=999),
        _row("B", "UNCONFIRMED", -6.0, c_rank=1),
    ])
    before = select_review_variant(pool, primary_rule())["code"].tolist()
    pool["rank_C_continuous"] = [1, 999]
    after = select_review_variant(pool, primary_rule())["code"].tolist()
    assert before == after == ["A"]


def test_forward_labels_cover_1w_to_4w():
    events = pd.DataFrame([_row("A", "UNCONFIRMED", -2.0, pivot=100.0)])
    dates = pd.bdate_range("2026-08-31", periods=20)
    bars = pd.DataFrame({
        "Open": [99.0 + i * 0.2 for i in range(20)],
        "High": [100.0 + i * 0.2 for i in range(20)],
        "Low": [98.0 + i * 0.2 for i in range(20)],
        "Close": [99.0, 100.5] + [101.0 + i * 0.2 for i in range(18)],
    }, index=dates)
    labeled = add_forward_labels(events, {"A": bars}).iloc[0]
    assert labeled["review_opportunity_1w"] == True
    assert labeled["forward_1w_censored"] == False
    assert labeled["forward_2w_censored"] == False
    assert labeled["forward_3w_censored"] == False
    assert labeled["forward_4w_censored"] == False


def test_oracle_marks_weekly_big_winner_and_loser():
    rows = []
    for i in range(6):
        row = _row(f"T{i}", "UNCONFIRMED", -1.0)
        row.update({
            "forward_1w_censored": False,
            "forward_1w_return_pct": float(i),
            "mfe_1w_pct": float(i + 1),
            "mae_1w_pct": float(-i),
            "forward_2w_censored": True,
            "forward_3w_censored": True,
            "forward_4w_censored": True,
        })
        rows.append(row)
    panel = add_weekly_oracle_flags(pd.DataFrame(rows))
    assert panel["winner_return_top5_1w"].sum() == 5
    assert panel["loser_return_bottom5_1w"].sum() == 5
    assert panel.loc[panel["code"].eq("T5"), "winner_return_top5_1w"].item() == True


def test_search_space_is_small_and_deterministic():
    rules = generate_candidate_rules()
    assert len(rules) == 144
    assert len({rule.name for rule in rules}) == 144


def _row(
    code, status, vs_buy, *, rule="ceiling", pivot=100.0,
    entry_volume=1.6, weekly_volume=1.4, eps=30.0, dist=-1.0,
    dry=None, close_position=0.9, range_ratio=0.7, c_rank=10,
):
    return {
        "snapshot_date": "2026-08-28",
        "code": code,
        "signal": True,
        "ibd_candidate_rule": rule,
        "ibd_candidate_price": pivot,
        "ibd_trigger_price": pivot,
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
