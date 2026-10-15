"""Regression protection for A/B observation; trading projection is untouched."""
from datetime import date

import pandas as pd

from dashboard.services.bf_observation import (
    build_observation_attention,
    build_observation_view,
    _price_path,
)


def row(code, *, snapshot="2026-10-09", signal=True, trigger=100, close=103,
        status="ACTIONABLE", valid=True, rule="pivot", anchor="2026-10-05"):
    return {
        "code": code, "snapshot_date": snapshot, "signal": signal,
        "ibd_candidate_rule": rule, "ibd_candidate_price": trigger,
        "ibd_candidate_extra": (
            '{"selected_pivot":{"price":' + str(trigger) +
            ',"resistance_date":"' + anchor + '","box_type":"M_BOX"}}'
        ) if rule == "pivot" else "",
        "ibd_entry_valid": valid, "ibd_entry_status": status,
        "latest_close": close, "ibd_entry_reject_reason": None,
    }


def test_weekend_signal_out_of_pool_remains_tracked_from_daily_prices():
    complete = pd.DataFrame([row("OUT", close=108)])
    current = pd.DataFrame([row("OTHER", snapshot="2026-10-13", trigger=50, close=52)])
    daily = {"OUT": pd.DataFrame(
        {"Close": [108, 99, 103]},
        index=pd.to_datetime(["2026-10-09", "2026-10-12", "2026-10-13"]),
    )}
    result = build_observation_view(
        complete, current, pd.DataFrame(), daily=daily, as_of=date(2026, 10, 13),
    ).set_index("code")
    out = result.loc["OUT"]
    assert out["review_a_active"]
    assert out["review_membership"] == "OUT"
    assert out["review_a_price"] == 100
    assert out["review_a_event"] == "BELOW → ZONE"
    assert out["review_a_week_event"] == "EXT → ZONE"
    assert out["review_a_zone"] == "ZONE"
    assert out["review_a_as_of"] == "2026-10-13"
    assert out["ibd_entry_status"] == "ACTIONABLE"


def test_weekend_a_and_distinct_midweek_b_do_not_replace_each_other():
    complete = pd.DataFrame([row("BOTH", trigger=100, close=103)])
    current = pd.DataFrame([row("BOTH", snapshot="2026-10-13", trigger=120,
                                 close=115, status="UNCONFIRMED", valid=False,
                                 anchor="2026-10-13")])
    # Trade eligibility is a different projection (B UNCONFIRMED).
    trade = current.copy()
    trade["review_futu_actionable"] = False
    trade["review_watch_active"] = True
    result = build_observation_view(
        complete, current, trade,
        daily={"BOTH": pd.DataFrame({"Close": [102, 105]},
                                    index=pd.to_datetime(["2026-10-12", "2026-10-13"]))},
        as_of=date(2026, 10, 13),
    ).iloc[0]
    assert result["ibd_candidate_price"] == 100
    assert result["review_a_price"] == 100
    assert result["ibd_entry_status"] == "ACTIONABLE"
    assert result["review_b_price"] == 120
    assert result["review_b_status"] == "UNCONFIRMED"
    assert result["review_b_event"] == "NEW BP"
    assert result["review_futu_actionable"] == False


def test_no_market_data_keeps_weekend_ticker_without_fabricated_changes():
    complete = pd.DataFrame([row("MISSING")])
    result = build_observation_view(
        complete, pd.DataFrame(), pd.DataFrame(),
        daily={}, as_of=date(2026, 10, 13),
    ).iloc[0]
    assert result["review_a_active"]
    assert not result["review_data_available"]
    assert result["review_change_label"] == "NO DATA"
    assert result["latest_close"] is None
    assert not result["review_has_value_change"]


def test_fallback_current_snapshot_gives_zone_but_not_fake_daily_history():
    complete = pd.DataFrame([row("S", close=107)])
    current = pd.DataFrame([row("S", snapshot="2026-10-13", close=97, signal=False)])
    result = build_observation_view(
        complete, current, pd.DataFrame(),
        daily={}, as_of=date(2026, 10, 13),
    ).iloc[0]
    assert result["review_a_zone"] == "BELOW"
    assert result["review_a_as_of"] == "2026-10-13"
    assert result["review_a_event"] == "EXT → BELOW"


def test_b_snapshot_does_not_keep_yesterday_buy_point():
    complete = pd.DataFrame([row("A", trigger=100)])
    previous = pd.DataFrame([row("A", snapshot="2026-10-12", trigger=120,
                                 anchor="2026-10-12")])
    current = pd.DataFrame([row("A", snapshot="2026-10-13", trigger=100,
                                 anchor="2026-10-05")])
    result = build_observation_view(
        complete, current, pd.DataFrame(), previous=previous,
        daily={}, as_of=date(2026, 10, 13),
    ).iloc[0]
    assert not result["review_b_active"]
    assert not result["review_new_buy_point"]


def test_pool_reentry_requires_previous_snapshot_and_shows_zone():
    complete = pd.DataFrame([row("REENTRY")])
    previous = pd.DataFrame([row("OTHER", snapshot="2026-10-12", trigger=30)])
    current = pd.DataFrame([row("REENTRY", snapshot="2026-10-13", signal=False,
                                 close=102)])
    daily = {"REENTRY": pd.DataFrame({"Close": [102]},
                index=pd.to_datetime(["2026-10-13"]))}
    result = build_observation_view(
        complete, current, pd.DataFrame(), previous=previous,
        daily=daily, as_of=date(2026, 10, 13),
    )
    reentry = result.set_index("code").loc["REENTRY"]
    assert reentry["review_membership_event"] == "OUT → IN"
    assert reentry["review_a_zone"] == "ZONE"
    events = build_observation_attention(
        current, complete, pd.DataFrame(), previous=previous, daily=daily,
        as_of=date(2026, 10, 13),
    )
    assert any(event["event"].startswith("OUT → IN") and
               "ZONE" in event["event"] for event in events)


def test_unconfirmed_weekend_zone_never_auto_confirms():
    complete = pd.DataFrame([row("UNC", valid=False, status="UNCONFIRMED")])
    current = pd.DataFrame([row("UNC", snapshot="2026-10-13", signal=False,
                                 close=102)])
    result = build_observation_view(
        complete, current, pd.DataFrame(), daily={},
        as_of=date(2026, 10, 13),
    ).iloc[0]
    assert result["review_a_zone"] == "ZONE"
    assert result["ibd_entry_status"] == "UNCONFIRMED"


def test_daily_rebased_against_weekend_snapshot_fails_closed():
    weekend = row("SPLIT", trigger=100, close=105)
    daily = pd.DataFrame(
        {"Close": [52.5, 53.0]},
        index=pd.to_datetime(["2026-10-09", "2026-10-13"]),
    )
    result = _price_path(weekend, None, daily, as_of=date(2026, 10, 13))
    assert result["close"] is None
    assert result["latest_event"] == ""


def test_previous_day_unchanged_does_not_repeat_old_price_event():
    complete = pd.DataFrame([row("STALE", trigger=100, close=108)])
    previous = pd.DataFrame([row("STALE", snapshot="2026-10-12", signal=False, close=99)])
    current = pd.DataFrame([row("STALE", snapshot="2026-10-13", signal=False, close=99)])
    daily = {"STALE": pd.DataFrame({"Close": [108, 99]},
             index=pd.to_datetime(["2026-10-09", "2026-10-12"]))}
    events = build_observation_attention(
        current, complete, pd.DataFrame(), previous=previous,
        daily=daily, as_of=date(2026, 10, 13),
    )
    assert events == []
