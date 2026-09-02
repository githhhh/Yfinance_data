import pandas as pd

from backtest.next_week_review_selection.labels import add_next_week_labels
from backtest.next_week_review_selection.selectors import (
    ReviewRule,
    review_rules,
    select_attention_matched,
    select_b0_actionable,
    select_review_variant,
    support_count,
)


def test_b0_reproduces_actionable_only_review_eligibility():
    pool = pd.DataFrame(
        [
            _row("ACT", "ACTIONABLE", 2.0),
            _row("UNC", "UNCONFIRMED", -1.0),
            _row("EXT", "EXTENDED", 7.0),
        ]
    )

    selected = select_b0_actionable(pool)

    assert selected["code"].tolist() == ["ACT"]


def test_r1_path_allows_near_non_actionable_states_but_rejects_out_of_path_and_clear_failure():
    pool = pd.DataFrame(
        [
            _row("ACT", "ACTIONABLE", 2.0),
            _row("UNC", "UNCONFIRMED", -2.0),
            _row("BEL", "BELOW_TRIGGER", -4.0),
            _row("EXT", "EXTENDED", 8.0),
            _row("FARLOW", "UNCONFIRMED", -6.0),
            _row("FAREXT", "EXTENDED", 12.0),
            _row("FAIL", "UNCONFIRMED", -1.0, close_position=0.60),
        ]
    )

    selected = select_review_variant(pool, review_rules()["R1_PATH"])

    assert set(selected["code"]) == {"ACT", "UNC", "BEL", "EXT"}


def test_missing_geometry_is_unknown_not_failure():
    pool = pd.DataFrame(
        [
            _row(
                "UNC",
                "UNCONFIRMED",
                -1.0,
                close_position=None,
                range_ratio=None,
                weekly_volume=1.4,
            )
        ]
    )

    selected = select_review_variant(pool, review_rules()["R2_BALANCED"])

    assert selected["code"].tolist() == ["UNC"]


def test_pullback_not_dry_is_neutral_not_negative_support():
    dry_false = pd.Series(
        _row(
            "FALSE",
            "UNCONFIRMED",
            -1.0,
            rule="pivot",
            dry=False,
            entry_volume=None,
            weekly_volume=1.4,
            eps=None,
            dist=-10.0,
        )
    )
    dry_missing = pd.Series(
        _row(
            "MISS",
            "UNCONFIRMED",
            -1.0,
            rule="pivot",
            dry=None,
            entry_volume=None,
            weekly_volume=1.4,
            eps=None,
            dist=-10.0,
        )
    )

    assert support_count(dry_false) == 1
    assert support_count(dry_missing) == 1


def test_r2_requires_one_positive_support_and_r3_requires_two():
    pool = pd.DataFrame(
        [
            _row(
                "ONE",
                "UNCONFIRMED",
                -1.0,
                entry_volume=None,
                weekly_volume=1.4,
                eps=None,
                dist=-10.0,
            ),
            _row(
                "TWO",
                "UNCONFIRMED",
                -2.0,
                entry_volume=None,
                weekly_volume=1.4,
                eps=30.0,
                dist=-10.0,
            ),
            _row(
                "ZERO",
                "UNCONFIRMED",
                -0.5,
                entry_volume=None,
                weekly_volume=1.0,
                eps=None,
                dist=-10.0,
            ),
        ]
    )

    r2 = select_review_variant(pool, review_rules()["R2_BALANCED"])
    r3 = select_review_variant(pool, review_rules()["R3_STRICT"])

    assert set(r2["code"]) == {"ONE", "TWO"}
    assert r3["code"].tolist() == ["TWO"]


def test_c_rank_cannot_change_capped_review_selection():
    pool_a = pd.DataFrame(
        [
            _row("AAA", "UNCONFIRMED", -1.0, c_rank=999),
            _row("BBB", "UNCONFIRMED", -1.0, c_rank=1),
        ]
    )
    pool_b = pool_a.copy()
    pool_b.loc[pool_b["code"].eq("AAA"), "rank_C_continuous"] = 1
    pool_b.loc[pool_b["code"].eq("BBB"), "rank_C_continuous"] = 999

    picked_a = select_review_variant(pool_a, review_rules()["R2_BALANCED"], cap=1)
    picked_b = select_review_variant(pool_b, review_rules()["R2_BALANCED"], cap=1)

    assert picked_a["code"].tolist() == ["AAA"]
    assert picked_b["code"].tolist() == ["AAA"]


def test_attention_matched_has_same_count_as_b0_but_can_choose_non_actionable():
    pool = pd.DataFrame(
        [
            _row("ACT1", "ACTIONABLE", 4.5, weekly_volume=1.0, eps=None, dist=-10.0),
            _row("ACT2", "ACTIONABLE", 4.0, weekly_volume=1.4),
            _row("UNC1", "UNCONFIRMED", -0.5, weekly_volume=1.8),
            _row("UNC2", "UNCONFIRMED", -1.0, weekly_volume=1.7),
        ]
    )

    baseline = select_b0_actionable(pool)
    matched = select_attention_matched(pool, review_rules()["R2_BALANCED"])

    assert len(matched) == len(baseline) == 2
    assert set(matched["code"]) == {"UNC1", "UNC2"}


def test_next_week_label_marks_unconfirmed_entering_frozen_buy_zone():
    events = pd.DataFrame(
        [
            _row("UNC", "UNCONFIRMED", -2.0, pivot=100.0),
        ]
    )
    prices = {
        "UNC": _bars(
            closes=[99.0, 100.5, 102.0, 106.0, 104.0],
            opens=[99.0, 99.5, 101.0, 103.0, 105.0],
            highs=[100.0, 102.0, 103.0, 107.0, 106.0],
            lows=[98.0, 99.0, 100.0, 102.0, 103.0],
        )
    }

    labeled = add_next_week_labels(events, prices).iloc[0]

    assert labeled["label_available"] == True
    assert labeled["review_opportunity_5d"] == True
    assert labeled["opportunity_type"] == "UNCONFIRMED_TO_ZONE"
    assert labeled["first_zone_date"] == "2026-09-01"


def test_extended_can_become_retest_review_opportunity_without_being_reclassified_actionable():
    events = pd.DataFrame(
        [
            _row("EXT", "EXTENDED", 8.0, pivot=100.0),
        ]
    )
    prices = {
        "EXT": _bars(
            closes=[108.0, 106.0, 104.0, 103.0, 102.0],
            opens=[109.0, 108.0, 106.0, 104.0, 103.0],
            highs=[110.0, 109.0, 107.0, 105.0, 104.0],
            lows=[107.0, 105.0, 103.0, 102.0, 101.0],
        )
    }

    labeled = add_next_week_labels(events, prices).iloc[0]

    assert labeled["review_opportunity_5d"] == True
    assert labeled["opportunity_type"] == "EXTENDED_RETEST_TO_ZONE"
    assert events.iloc[0]["ibd_entry_status"] == "EXTENDED"


def test_incomplete_forward_window_is_censored_and_not_counted_as_opportunity():
    events = pd.DataFrame([_row("ACT", "ACTIONABLE", 1.0, pivot=100.0)])
    full = _bars(
        closes=[101.0, 102.0, 103.0, 104.0, 105.0],
        opens=[101.0, 102.0, 103.0, 104.0, 105.0],
        highs=[102.0, 103.0, 104.0, 105.0, 106.0],
        lows=[100.0, 101.0, 102.0, 103.0, 104.0],
    )
    prices = {"ACT": full.head(3)}

    labeled = add_next_week_labels(events, prices).iloc[0]

    assert labeled["forward_5d_censored"] == True
    assert labeled["label_available"] == False
    assert labeled["review_opportunity_5d"] == False


def _row(
    code: str,
    status: str,
    current_vs_buy: float,
    *,
    rule: str = "ceiling",
    pivot: float = 100.0,
    entry_volume: float | None = 1.6,
    weekly_volume: float = 1.4,
    eps: float | None = 30.0,
    dist: float = -1.0,
    dry: bool | None = None,
    close_position: float | None = 0.90,
    range_ratio: float | None = 0.70,
    c_rank: int = 10,
) -> dict[str, object]:
    return {
        "snapshot_date": "2026-08-28",
        "code": code,
        "signal": True,
        "ibd_candidate_rule": rule,
        "ibd_candidate_price": pivot,
        "ibd_trigger_price": pivot,
        "ibd_entry_status": status,
        "current_vs_ibd_candidate_pct": current_vs_buy,
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


def _bars(*, closes, opens, highs, lows) -> pd.DataFrame:
    dates = pd.bdate_range("2026-08-31", periods=len(closes))
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
        },
        index=dates,
    )
