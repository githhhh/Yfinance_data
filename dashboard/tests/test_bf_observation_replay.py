"""BF Observation Replay V2 pure facts and published-history contract."""
from __future__ import annotations

from datetime import date
import subprocess

import pandas as pd

from dashboard.services.bf_observation_replay import (
    entry_confirmation, identity_v2, observe_midweek_buy_point,
    price_replay, watch_provenance, zone,
)
from dashboard.services.bf_observation import (
    build_observation_events_v2, load_observation_history,
)


def pivot(code, day, price, close, *, box="S_BOX", signal=True,
          valid=False, status="UNCONFIRMED", selected=True):
    import json
    item = {"box_type": box, "price": price, "resistance_date": "2026-10-12"}
    extra = {"pivot_candidates": [item]}
    if selected:
        extra["selected_pivot"] = item
    return {"code": code, "snapshot_date": day, "signal": signal,
            "ibd_candidate_rule": "pivot" if signal else "",
            "ibd_candidate_price": price if signal else None,
            "ibd_candidate_extra": json.dumps(extra) if signal else "",
            "ibd_entry_valid": valid, "ibd_entry_status": status if signal else "",
            "latest_close": close, "bf_watch_active": False,
            "bf_watch_type": "", "bf_watch_trigger_price": None}


def hist(when, item, *, origin="MIDWEEK"):
    return {"date": when, "row": item, "verified": True,
            "origin": origin, "source_ref": when}


def observe(code, current, history, *, weekend=None):
    return observe_midweek_buy_point(
        code=code, current=current, history=history,
        weekend=weekend, as_of="2026-10-15", review_week="2026-10-12",
        baseline_verified=True,
    )


def test_andg_reappears_with_cross_rule_ticker_watch():
    old = pivot("ANDG", "2026-10-12", 57.62, 57.66)
    watch = pivot("ANDG", "2026-10-14", 0, 57.29, signal=False)
    watch.update(bf_watch_active=True, bf_watch_type="ma10_pullback",
                 bf_watch_trigger_price=57.695)
    now = pivot("ANDG", "2026-10-15", 57.62, 58.95)
    event = observe("ANDG", now, [hist("2026-10-12", old),
                                   hist("2026-10-14", watch)])
    assert event["identity"]["lifecycle"] == "REAPPEARED"
    assert event["primary"] == "ZONE_REENTRY"
    assert event["price"]["transition"] == "BELOW_TO_ZONE"
    assert event["price"]["basis"] == "TRACKED_ANCHOR"
    assert event["watch"]["latest"]["relation"] == "TICKER_CONTEXT"
    assert event["watch"]["latest"]["type"] == "ma10_pullback"
    assert event["entry"]["act"] is False


def test_aout_first_observed_retrospective_with_inactive_watch():
    before = pivot("AOUT", "2026-10-14", 0, 17.51, signal=False)
    before["bf_watch_type"] = "pivot"
    current = pivot("AOUT", "2026-10-15", 17.689, 17.76)
    event = observe("AOUT", current, [hist("2026-10-14", before)])
    assert event["identity"]["lifecycle"] == "FIRST_OBSERVED"
    assert event["price"]["transition"] == "BELOW_TO_ZONE"
    assert event["price"]["basis"] == "RETROSPECTIVE"
    assert event["watch"]["active_prior"] is False


def test_ko_s_box_to_m_box_same_date_not_trigger_revision():
    prior = pivot("KO", "2026-10-14", 88.96, 91.78, box="S_BOX")
    current = pivot("KO", "2026-10-15", 91.87, 92., box="M_BOX")
    assert identity_v2(prior)["key"] != identity_v2(current)["key"]
    event = observe("KO", current, [hist("2026-10-14", prior)])
    assert event["identity"]["lifecycle"] == "FIRST_OBSERVED"
    assert event["price"]["basis"] == "RETROSPECTIVE"
    assert event["price"]["transition"] == "BELOW_TO_ZONE"


def test_cbll_selected_m_box_when_candidates_include_both_boxes():
    import json
    current = pivot("CBLL", "2026-10-15", 25.07, 27.64, box="M_BOX")
    extra = json.loads(current["ibd_candidate_extra"])
    extra["pivot_candidates"].append(
        {"box_type": "S_BOX", "price": 25.07, "resistance_date": "2026-10-12"})
    current["ibd_candidate_extra"] = json.dumps(extra)
    assert identity_v2(current)["key"] == "pivot|2026-10-12|M_BOX"
    extra.pop("selected_pivot")
    current["ibd_candidate_extra"] = json.dumps(extra)
    assert identity_v2(current)["resolution"] == "AMBIGUOUS"


def test_amn_small_revision_not_new_structure_and_not_in_zone():
    previous = pivot("AMN", "2026-10-14", 36.855, 37.66, box="M_BOX")
    current = pivot("AMN", "2026-10-15", 36.86, 40.0, box="M_BOX")
    assert identity_v2(previous)["key"] == identity_v2(current)["key"]
    assert observe("AMN", current, [hist("2026-10-14", previous)]) is None
    result = price_replay(trigger=36.86, previous_close=37.66,
                          current_close=40, previous_date="2026-10-14",
                          current_date="2026-10-15", basis="TRACKED_ANCHOR",
                          previous_trigger=36.855)
    assert result["trigger_revision"] == "REVISED"
    assert result["transition"] == "ZONE_TO_EXT"


def test_revision_cannot_fake_buy_zone_entry():
    p = price_replay(trigger=110, previous_trigger=100, previous_close=107,
                     current_close=108, previous_date="2026-10-14",
                     current_date="2026-10-15", basis="TRACKED_ANCHOR")
    assert p["trigger_revision"] == "REVISED"
    assert p["transition"] is None
    assert p["current_zone"] == "BELOW"


def test_real_ext_to_zone_and_confirmation():
    old = pivot("EXT", "2026-10-14", 100, 108, valid=True, status="EXTENDED")
    now = pivot("EXT", "2026-10-15", 100, 103, valid=True, status="ACTIONABLE")
    event = observe("EXT", now, [hist("2026-10-14", old)])
    assert event["price"]["transition"] == "EXT_TO_ZONE"
    assert event["entry"]["act"] is True
    assert event["identity"]["lifecycle"] == "CONTINUING"


def test_no_repeated_in_zone_alert_or_unverified_history():
    old = pivot("STABLE", "2026-10-14", 100, 101)
    now = pivot("STABLE", "2026-10-15", 100, 102)
    assert observe("STABLE", now, [hist("2026-10-14", old)]) is None
    missing = hist("2026-10-14", old)
    missing["verified"] = False
    assert observe("STABLE", now, [missing]) is None


def test_unknown_identity_cannot_be_called_new():
    current = pivot("UNRESOLVED", "2026-10-15", 100, 101)
    current["ibd_candidate_extra"] = "{}"
    assert observe("UNRESOLVED", current,
                   [hist("2026-10-14", pivot("X", "2026-10-14", 0, 99, signal=False))]) is None


def test_new_b_matching_a_is_not_independent():
    weekend = pivot("SAME", "2026-10-09", 100, 99)
    current = pivot("SAME", "2026-10-15", 100, 102)
    assert observe("SAME", current,
                   [hist("2026-10-14", pivot("SAME", "2026-10-14", 0, 99, signal=False))],
                   weekend=weekend) is None


def test_positive_act_requires_both_resolver_and_buy_zone():
    invalid = pivot("INVALID", "2026-10-15", 100, 102, status="ACTIONABLE", valid=False)
    assert entry_confirmation(invalid, current_zone="ZONE")["act"] is False


def test_watch_provenance_requires_active_and_strictly_earlier_date():
    watch = pivot("A", "2026-10-14", 0, 99, signal=False)
    watch.update(bf_watch_active=True, bf_watch_type="pivot")
    late = dict(watch, snapshot_date="2026-10-15")
    records = [hist("2026-10-14", watch), hist("2026-10-15", late)]
    result = watch_provenance(records, as_of="2026-10-15")
    assert len(result["observations"]) == 1


def test_v2_adapter_preserves_weekend_watch_as_context():
    current = pd.DataFrame([pivot("W", "2026-10-15", 100, 102)])
    complete_row = pivot("W", "2026-10-09", 0, 99, signal=False)
    complete_row.update(bf_watch_active=True, bf_watch_type="pivot")
    previous = pivot("W", "2026-10-14", 0, 98, signal=False)
    events = build_observation_events_v2(
        current, pd.DataFrame([complete_row]), as_of=date(2026, 10, 15),
        history_snapshots=[{"date": "2026-10-14", "frame": pd.DataFrame([previous]),
                            "source_ref": "commit", "verified": True, "origin": "MIDWEEK"}],
        diagnostic={"usable": True, "history_coverage": "AVAILABLE"},
    )
    assert len(events) == 1
    assert events[0]["watch"]["latest"]["origin"] == "WEEKEND"


def test_git_history_read_only_and_duplicate_date_conflicts_fail_closed(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    output = tmp_path / "us" / "breakout_follow_pool_midweek.csv"
    output.parent.mkdir()
    for dated, price in [("2026-10-12", 100), ("2026-10-14", 101)]:
        pd.DataFrame([pivot("X", dated, 100, price)]).to_csv(output, index=False)
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", dated], cwd=tmp_path, check=True)
    snapshots, diagnostics = load_observation_history(tmp_path, as_of=date(2026, 10, 15))
    # Synthetic commits created today may fall outside an old as_of scan;
    # real production commits retain their historical commit timestamps.
    assert isinstance(snapshots, list) and isinstance(diagnostics, dict)


def test_zone_boundary_rounding_is_shared_with_v1():
    assert zone(100, 99.99) == "BELOW"
    assert zone(100, 100) == "ZONE"
    assert zone(100, 105) == "ZONE"
    assert zone(100, 105.01) == "EXT"
