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
    assert out["review_a_path"] == "EXT → BELOW → ZONE"
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
    assert result["review_a_event"] == ""
    assert result["review_a_week_event"] == "EXT → BELOW"
    assert result["review_a_path"] == "BELOW"


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
    # Pool re-entry remains a dashboard fact; it is not a Telegram entry
    # opportunity unless the fixed weekend price zone really changes.
    assert events == []


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


def test_stable_b_with_unconfirmed_entry_still_tracks_price_zone_movement():
    complete = pd.DataFrame([row("ABC", trigger=100)])
    previous = pd.DataFrame([row("ABC", snapshot="2026-10-12", trigger=120,
                                 close=121, valid=False, status="UNCONFIRMED",
                                 anchor="2026-10-12")])
    current = pd.DataFrame([row("ABC", snapshot="2026-10-13", trigger=120,
                                close=127, valid=False, status="UNCONFIRMED",
                                anchor="2026-10-12")])
    result = build_observation_view(
        complete, current, pd.DataFrame(), previous=previous, daily={},
        as_of=date(2026, 10, 13),
    ).iloc[0]
    assert result["review_b_event"] == "ZONE → EXT"
    assert result["review_b_status"] == "UNCONFIRMED"
    assert result["review_b_zone"] == "EXT"


def test_daily_loader_reads_production_split_pickle_and_actual_trade_date(tmp_path):
    import pickle
    from dashboard.services.bf_observation import load_daily_snapshot

    pkl_dir = tmp_path / "results_pkl"
    pkl_dir.mkdir()
    us_dates = pd.to_datetime(["2026-10-09", "2026-10-13"]).tz_localize("US/Eastern")
    content = {"LIVE": pd.DataFrame({"Close": [105.0, 102.4]}, index=us_dates).to_dict("split")}
    with (pkl_dir / "stock_data_141026_1d.pkl").open("wb") as handle:
        pickle.dump(content, handle)
    snapshot = load_daily_snapshot(tmp_path, window_date=date(2026, 10, 14))
    assert set(snapshot) == {"LIVE"}
    assert snapshot["LIVE"]["Close"].iloc[-1] == 102.4
    assert str(snapshot["LIVE"].index[-1].date()) == "2026-10-13"
    assert load_daily_snapshot(tmp_path, window_date=date(2026, 10, 13)) == {}


def test_previous_midweek_uses_committed_same_week_snapshot_not_current(tmp_path):
    import subprocess
    from dashboard.services.bf_observation import load_previous_midweek

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=tmp_path, check=True)
    pool = tmp_path / "us" / "breakout_follow_pool_midweek.csv"
    pool.parent.mkdir()
    for market_date in ["2026-10-12", "2026-10-13", "2026-10-14"]:
        pd.DataFrame([row("HISTORY", snapshot=market_date)]).to_csv(pool, index=False)
        subprocess.run(["git", "add", "us/breakout_follow_pool_midweek.csv"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", market_date], cwd=tmp_path, check=True)
    previous = load_previous_midweek(tmp_path, date(2026, 10, 14))
    assert previous is not None
    assert previous.iloc[0]["snapshot_date"] == "2026-10-13"
    assert load_previous_midweek(tmp_path, date(2026, 10, 12)) is None


def test_no_yesterday_bar_never_invents_latest_daily_transition():
    complete = pd.DataFrame([row("GAP", trigger=100, close=108)])
    # 10/12 was an open US trading day, but it is absent from the PKL.
    daily = pd.DataFrame({"Close": [108, 97]},
                         index=pd.to_datetime(["2026-10-09", "2026-10-13"]))
    path = _price_path(complete.iloc[0], None, daily, as_of=date(2026, 10, 13))
    assert path["zone"] == "BELOW_BUY_POINT"
    assert path["week_event"] == "EXT → BELOW"
    assert path["latest_event"] == ""


def test_first_midweek_observation_is_weekend_net_not_latest_change():
    complete = pd.DataFrame([row("MON", trigger=100, close=108)])
    daily = pd.DataFrame({"Close": [108, 97]},
                         index=pd.to_datetime(["2026-10-09", "2026-10-12"]))
    path = _price_path(complete.iloc[0], None, daily, as_of=date(2026, 10, 12))
    assert path["latest_event"] == ""
    assert path["week_event"] == "EXT → BELOW"


def test_nonconsecutive_same_week_daily_bars_still_produce_latest_change():
    weekend = row("GAP", close=108)
    daily = pd.DataFrame(
        {"Close": [108, 97, 103]},
        index=pd.to_datetime(["2026-10-09", "2026-10-12", "2026-10-15"]),
    )
    result = _price_path(weekend, None, daily, as_of=date(2026, 10, 15))
    assert result["latest_event"] == "BELOW → ZONE"
    assert result["week_event"] == "EXT → ZONE"
    assert result["path"] == "EXT → BELOW → ZONE"
    assert result["as_of"] == "2026-10-15"


def test_previous_snapshot_is_comparison_boundary_even_with_intermediate_daily_k():
    complete = pd.DataFrame([row("A", close=108)])
    previous = pd.DataFrame([
        row("A", snapshot="2026-10-12", close=97, signal=False),
    ])
    current = pd.DataFrame([
        row("A", snapshot="2026-10-15", close=104, signal=False),
    ])
    # Tue/Wed bars are real, but this run compares the last published snapshot
    # on Monday with the current Thursday state.
    daily = {"A": pd.DataFrame(
        {"Close": [108, 97, 108, 103, 104]},
        index=pd.to_datetime([
            "2026-10-09", "2026-10-12", "2026-10-13",
            "2026-10-14", "2026-10-15",
        ]),
    )}
    result = build_observation_view(
        complete, current, pd.DataFrame(),
        previous=previous, daily=daily, as_of=date(2026, 10, 15),
    ).iloc[0]
    assert result["review_a_event"] == "BELOW → ZONE"
    assert result["review_a_week_event"] == "EXT → ZONE"
    events = build_observation_attention(
        current, complete, pd.DataFrame(),
        previous=previous, daily=daily, as_of=date(2026, 10, 15),
    )
    assert [event["event"] for event in events
            if event["source"] == "WEEKEND"] == ["BELOW → ZONE"]


def test_previous_and_current_same_week_pool_close_work_without_daily_pkl():
    complete = pd.DataFrame([row("POOL", close=108)])
    previous = pd.DataFrame([
        row("POOL", snapshot="2026-10-12", close=97, signal=False),
    ])
    current = pd.DataFrame([
        row("POOL", snapshot="2026-10-15", close=103, signal=False),
    ])
    view = build_observation_view(
        complete, current, pd.DataFrame(),
        previous=previous, daily={}, as_of=date(2026, 10, 15),
    ).iloc[0]
    assert view["review_a_event"] == "BELOW → ZONE"
    assert view["review_a_path"] == "BELOW → ZONE"
    assert view["review_a_zone"] == "ZONE"


def test_previous_midweek_snapshot_from_different_week_never_generates_transition():
    complete = pd.DataFrame([row("CROSS", close=108)])
    previous = pd.DataFrame([
        row("CROSS", snapshot="2026-10-08", close=97, signal=False),
    ])
    current = pd.DataFrame([
        row("CROSS", snapshot="2026-10-12", close=103, signal=False),
    ])
    view = build_observation_view(
        complete, current, pd.DataFrame(),
        previous=previous, daily={}, as_of=date(2026, 10, 12),
    ).iloc[0]
    assert view["review_a_event"] == ""
    assert view["review_a_week_event"] == "EXT → ZONE"
    assert view["review_membership_event"] == ""


def test_missing_intermediate_bars_never_invent_multiple_transitions():
    complete = pd.DataFrame([row("SPARSE", close=108)])
    previous = pd.DataFrame([
        row("SPARSE", snapshot="2026-10-12", close=97, signal=False),
    ])
    current = pd.DataFrame([
        row("SPARSE", snapshot="2026-10-15", close=103, signal=False),
    ])
    view = build_observation_view(
        complete, current, pd.DataFrame(),
        previous=previous, daily={}, as_of=date(2026, 10, 15),
    ).iloc[0]
    assert view["review_a_path"] == "BELOW → ZONE"
    assert view["review_a_event"] == "BELOW → ZONE"
    assert "EXT" not in view["review_a_path"]


def test_telegram_weekend_only_price_returns_to_zone_with_original_confirmation():
    complete = pd.DataFrame([
        row("RECLAIM", close=97, status="UNCONFIRMED", valid=False),
        row("CHASING", close=103, valid=True),
        row("FALL", close=103, valid=True),
    ])
    previous = pd.DataFrame([
        row("RECLAIM", snapshot="2026-10-12", close=98, signal=False),
        row("CHASING", snapshot="2026-10-12", close=104, signal=False),
        row("FALL", snapshot="2026-10-12", close=102, signal=False),
    ])
    current = pd.DataFrame([
        row("RECLAIM", snapshot="2026-10-15", close=102, signal=False),
        row("CHASING", snapshot="2026-10-15", close=107, signal=False),
        row("FALL", snapshot="2026-10-15", close=98, signal=False),
    ])
    events = build_observation_attention(
        current, complete, pd.DataFrame(), previous=previous, daily={},
        as_of=date(2026, 10, 15),
    )
    assert len(events) == 1
    assert events[0]["code"] == "RECLAIM"
    assert events[0]["event"] == "BELOW → ZONE"
    assert events[0]["zone"] == "ZONE"
    assert events[0]["status"] == "UNCONFIRMED"
    assert events[0]["membership"] == "IN"


def test_telegram_midweek_only_new_independent_in_zone_buy_points():
    complete = pd.DataFrame([row("SEED", signal=False)])
    previous = pd.DataFrame([
        row("UPDATE", snapshot="2026-10-12", trigger=110, anchor="2026-10-12"),
        row("CHANGED", snapshot="2026-10-12", trigger=110, anchor="2026-10-12"),
    ])
    current = pd.DataFrame([
        row("NEW", snapshot="2026-10-15", trigger=100, close=102,
            status="UNCONFIRMED", valid=False),
        row("NEWEXT", snapshot="2026-10-15", trigger=100, close=108),
        row("UPDATE", snapshot="2026-10-15", trigger=112, close=113,
            anchor="2026-10-12"),
        row("CHANGED", snapshot="2026-10-15", trigger=105, close=106,
            anchor="2026-10-15"),
    ])
    events = build_observation_attention(
        current, complete, current, previous=previous, daily={},
        as_of=date(2026, 10, 15),
    )
    assert {event["code"] for event in events} == {"NEW", "CHANGED"}
    for event in events:
        assert event["event"] == "NEW BP"
        assert event["zone"] == "ZONE"
        assert event["status"] in {"ACTIONABLE", "UNCONFIRMED"}


def test_telegram_does_not_repeat_stale_weekend_price_and_ignores_pool_only_transitions():
    complete = pd.DataFrame([row("S", close=97)])
    previous = pd.DataFrame([row("S", snapshot="2026-10-12", close=98, signal=False)])
    current = pd.DataFrame([row("S", snapshot="2026-10-15", close=102, signal=False)])
    daily = {"S": pd.DataFrame(
        {"Close": [97, 102]},
        index=pd.to_datetime(["2026-10-09", "2026-10-12"]),
    )}
    # No later market candle: the actual latest observed bar is from 10/12.
    events = build_observation_attention(
        current, complete, pd.DataFrame(), previous=previous,
        daily=daily, as_of=date(2026, 10, 15),
    )
    # The dated current Pool close (10/15) legitimately supplies this observation.
    assert len(events) == 1
    assert events[0]["as_of"] == "2026-10-15"


def test_midweek_telegram_act_requires_both_actionable_status_and_valid_resolver():
    complete = pd.DataFrame([row("SEED", signal=False)])
    current = pd.DataFrame([
        row("GOOD", snapshot="2026-10-15", trigger=100, close=102,
            status="ACTIONABLE", valid=True),
        row("BAD", snapshot="2026-10-15", trigger=100, close=102,
            status="ACTIONABLE", valid=False),
        row("WAIT", snapshot="2026-10-15", trigger=100, close=102,
            status="UNCONFIRMED", valid=False),
    ])
    events = build_observation_attention(
        current, complete, current, previous=None, daily={},
        as_of=date(2026, 10, 15),
    )
    statuses = {event["code"]: event["status"] for event in events}
    assert statuses == {
        "GOOD": "ACTIONABLE",
        "BAD": "UNCONFIRMED",
        "WAIT": "UNCONFIRMED",
    }


def test_unverified_pivot_does_not_claim_new_buy_point():
    from dashboard.services.bf_review_presentation import decorate_observation_view
    candidate = row("SN", snapshot="2026-10-20", trigger=194.54, close=198.72)
    candidate["ibd_candidate_extra"] = (
        '{"pivot_candidates":['
        '{"box_type":"M_BOX","resistance_date":"2026-10-19","price":194.54},'
        '{"box_type":"S_BOX","resistance_date":"2026-10-19","price":194.54}]}'
    )
    current = pd.DataFrame([candidate])
    raw = build_observation_view(
        pd.DataFrame(), current, current, daily={}, as_of=date(2026, 10, 20),
    )
    actual = decorate_observation_view(
        raw, complete=pd.DataFrame(), current=current, previous=None,
    ).iloc[0]
    assert actual["review_b_active"]
    assert actual["review_b_identity_state"] == "AMBIGUOUS"
    assert actual["review_b_display_event"] == "UNVERIFIED"
    assert not actual["review_b_verified"]
    assert not actual["review_new_buy_point"]


def test_dual_buy_point_statuses_remain_independent():
    from dashboard.services.bf_review_presentation import decorate_observation_view
    weekend = row("TALO", snapshot="2026-10-16", trigger=17.005, close=17.77,
                  anchor="2026-10-12", status="ACTIONABLE")
    midweek = row("TALO", snapshot="2026-10-20", trigger=17.92, close=18.05,
                  anchor="2026-10-19", status="ACTIONABLE")
    complete, current = pd.DataFrame([weekend]), pd.DataFrame([midweek])
    raw = build_observation_view(
        complete, current, pd.DataFrame(), daily={}, as_of=date(2026, 10, 20),
    )
    actual = decorate_observation_view(
        raw, complete=complete, current=current, previous=None,
    ).iloc[0]
    assert actual["ibd_entry_status"] == "EXTENDED"
    assert actual["review_b_status"] == "ACTIONABLE"
    assert actual["review_b_verified"]
    assert actual["review_a_price"] == 17.005
    assert actual["review_b_price"] == 17.92


def test_daily_candles_after_as_of_are_excluded():
    weekend = row("CLIP", snapshot="2026-10-16", trigger=100, close=101)
    midweek = row("CLIP", snapshot="2026-10-20", signal=False, close=103)
    daily = {"CLIP": pd.DataFrame(
        {"Close": [101, 103, 120]},
        index=pd.to_datetime(["2026-10-16", "2026-10-20", "2026-10-21"]),
    )}
    actual = build_observation_view(
        pd.DataFrame([weekend]), pd.DataFrame([midweek]), pd.DataFrame(),
        daily=daily, as_of=date(2026, 10, 20),
    ).iloc[0]
    assert actual["latest_close"] == 103
    assert actual["review_a_as_of"] == "2026-10-20"
    assert actual["ibd_entry_status"] == "ACTIONABLE"


def test_midweek_b_trigger_revision_is_reported_even_if_status_unchanged():
    from dashboard.services.bf_review_presentation import decorate_observation_view
    weekend = pd.DataFrame([row("OTHER", snapshot="2026-10-16")])
    first = row("REVISED", snapshot="2026-10-19", trigger=100, close=102,
                anchor="2026-10-19")
    changed = row("REVISED", snapshot="2026-10-20", trigger=101, close=103,
                  anchor="2026-10-19")
    base = pd.DataFrame([changed])
    raw = build_observation_view(weekend, base, base, as_of=date(2026, 10, 20))
    result = decorate_observation_view(
        raw, complete=weekend, current=base, previous=pd.DataFrame([first])
    ).set_index("code").loc["REVISED"]
    assert result["review_b_verified"]
    assert result["review_b_display_event"] == "BP UPDATED"
    assert result["review_b_has_value_change"]
    assert result["review_new_buy_point_zone"] == "BUY_ZONE"


def test_no_prior_midweek_snapshot_must_not_repeat_new_event():
    from dashboard.services.bf_review_presentation import decorate_observation_view
    weekend = pd.DataFrame([row("OTHER", snapshot="2026-10-16")])
    current = pd.DataFrame([row("B", snapshot="2026-10-21", trigger=100,
                                close=103, anchor="2026-10-19")])
    raw = build_observation_view(weekend, current, current, as_of=date(2026, 10, 21))
    result = decorate_observation_view(
        raw, complete=weekend, current=current, previous=None
    ).set_index("code").loc["B"]
    assert result["review_b_verified"]
    assert result["review_b_display_event"] == "SNAPSHOT"
    assert result["review_b_display_basis"] == "UNAVAILABLE"
    assert result["review_new_buy_point_zone"] == "BUY_ZONE"
    assert not result["review_b_has_value_change"]


def test_v2_recovered_b_populates_zone_and_filter_source():
    from dashboard.services.bf_review_presentation import decorate_observation_view
    # V1 sees same pivot date and calls it A; V2 distinguishes the box type.
    weekend_row = row("BOX", snapshot="2026-10-16", trigger=100,
                      close=101, anchor="2026-10-15")
    current_row = row("BOX", snapshot="2026-10-19", trigger=100,
                      close=102, anchor="2026-10-15")
    current_row["ibd_candidate_extra"] = (
        '{"selected_pivot":{"price":100,"resistance_date":"2026-10-15",'
        '"box_type":"S_BOX"}}'
    )
    wk, mid = pd.DataFrame([weekend_row]), pd.DataFrame([current_row])
    raw = build_observation_view(wk, mid, mid, as_of=date(2026, 10, 19))
    result = decorate_observation_view(
        raw, complete=wk, current=mid, previous=None
    ).iloc[0]
    assert result["review_b_active"]
    assert result["review_b_verified"]
    assert result["review_new_buy_point_zone"] == "BUY_ZONE"
    assert result["review_b_zone"] == "ZONE"
    assert result["review_b_display_event"] == "NEW BP"


def test_weekend_unknown_identity_must_not_create_verified_new_b():
    from dashboard.services.bf_review_presentation import decorate_observation_view
    weekend_row = row("UNK", snapshot="2026-10-16", trigger=100, close=101,
                      anchor="2026-10-15")
    weekend_row["ibd_candidate_extra"] = "{}"  # weekend identity unknown
    now = row("UNK", snapshot="2026-10-19", trigger=110, close=112,
              anchor="2026-10-19")
    w, m = pd.DataFrame([weekend_row]), pd.DataFrame([now])
    raw = build_observation_view(w, m, m, as_of=date(2026, 10, 19))
    result = decorate_observation_view(
        raw, complete=w, current=m, previous=None
    ).iloc[0]
    assert not result["review_new_buy_point"]
    assert not result["review_b_has_value_change"]
    assert not result["review_b_verified"]


def test_previous_was_weekend_anchor_today_first_distinct_b_is_new_not_changed():
    from dashboard.services.bf_review_presentation import decorate_observation_view
    wk = row("SWITCH", snapshot="2026-10-16", trigger=100, close=102,
             anchor="2026-10-15")
    yesterday = row("SWITCH", snapshot="2026-10-19", trigger=100, close=103,
                    anchor="2026-10-15")
    today = row("SWITCH", snapshot="2026-10-20", trigger=105, close=106,
                anchor="2026-10-20")
    complete, current, previous = (
        pd.DataFrame([wk]), pd.DataFrame([today]), pd.DataFrame([yesterday])
    )
    raw = build_observation_view(
        complete, current, current, previous=previous,
        as_of=date(2026, 10, 20),
    )
    actual = decorate_observation_view(
        raw, complete=complete, current=current, previous=previous
    ).iloc[0]
    assert actual["review_b_verified"]
    assert actual["review_b_display_event"] == "NEW BP"
    assert actual["review_b_display_basis"] == "PREVIOUS_SNAPSHOT"
    assert actual["review_b_has_value_change"]


def test_out_of_pool_weekend_price_event_uses_prior_session_not_missing_snapshot():
    from dashboard.services.bf_review_presentation import decorate_observation_view
    complete = pd.DataFrame([row("GABC", snapshot="2026-10-16",
                                 trigger=100, close=102)])
    previous = pd.DataFrame([row("OTHER", snapshot="2026-10-19")])
    current = pd.DataFrame([row("OTHER", snapshot="2026-10-20")])
    daily = {"GABC": pd.DataFrame(
        {"Close": [99, 102]},
        index=pd.to_datetime(["2026-10-19", "2026-10-20"]),
    )}
    raw = build_observation_view(
        complete, current, current, previous=previous, daily=daily,
        as_of=date(2026, 10, 20),
    )
    actual = decorate_observation_view(
        raw, complete=complete, current=current, previous=previous
    ).set_index("code").loc["GABC"]
    assert actual["review_membership"] == "OUT"
    assert actual["review_a_event"] == "BELOW → ZONE"
    assert actual["review_a_focus_basis"] == "PRIOR_SESSION"
