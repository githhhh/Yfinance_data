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
    assert identity_v2(current)["key"] == "price|25.07"
    extra.pop("selected_pivot")
    current["ibd_candidate_extra"] = json.dumps(extra)
    assert identity_v2(current)["resolution"] == "EXACT"


def test_same_price_different_signals_is_one_buy_point():
    import json
    old = pivot("ONE", "2026-10-14", 100, 101, box="S_BOX")
    current = pivot("ONE", "2026-10-15", 100, 102, box="M_BOX")
    current["ibd_candidate_extra"] = json.dumps({
        "pivot_candidates": [
            {"box_type":"M_BOX","price":100,"resistance_date":"2026-10-12"},
            {"box_type":"S_BOX","price":100,"resistance_date":"2026-10-12"},
        ]
    })
    assert identity_v2(old)["key"] == identity_v2(current)["key"]
    assert observe("ONE", current, [hist("2026-10-14", old)]) is None


def test_amn_small_revision_not_new_structure_and_not_in_zone():
    previous = pivot("AMN", "2026-10-14", 36.855, 37.66, box="M_BOX")
    current = pivot("AMN", "2026-10-15", 36.86, 40.0, box="M_BOX")
    assert identity_v2(previous)["key"] != identity_v2(current)["key"]
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
    event = observe("UNRESOLVED", current,
                    [hist("2026-10-14", pivot("X", "2026-10-14", 0, 99, signal=False))])
    assert event is not None
    assert event["primary"] == "FIRST_OBSERVED_BP"


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
    previous = pd.DataFrame([pivot("X", "2026-10-14", 100, 101)])
    snapshots, diagnostics = load_observation_history(
        tmp_path, as_of=date(2026, 10, 15), previous=previous)
    assert diagnostics["usable"] is True
    assert [x["date"] for x in snapshots] == ["2026-10-12", "2026-10-14"]

    # The standalone preflight has no captured previous Pool; it validates
    # archive availability, while LIVE replay verifies the actual capture.
    probe, inspection = load_observation_history(
        tmp_path, as_of=date(2026, 10, 15), require_previous=False)
    assert inspection["usable"] is True
    assert inspection["previous_boundary_verified"] is False
    assert len(probe) == 2

    # Same market date, incompatible published revisions: never choose one
    # arbitrarily and call it an authoritative prior observation.
    pd.DataFrame([pivot("X", "2026-10-14", 100, 103)]).to_csv(output, index=False)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "conflicting 2026-10-14"], cwd=tmp_path, check=True)
    snapshots, diagnostics = load_observation_history(
        tmp_path, as_of=date(2026, 10, 15), previous=previous)
    assert snapshots == []
    assert diagnostics["usable"] is False
    assert "conflicting" in diagnostics["reason"]


def test_zone_boundary_rounding_is_shared_with_v1():
    assert zone(100, 99.99) == "BELOW"
    assert zone(100, 100) == "ZONE"
    assert zone(100, 105) == "ZONE"
    assert zone(100, 105.01) == "EXT"


def test_git_preflight_rejects_shallow_and_previous_unverified(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    output = tmp_path / "us" / "breakout_follow_pool_midweek.csv"
    output.parent.mkdir()
    pd.DataFrame([pivot("X", "2026-10-14", 100, 101)]).to_csv(output, index=False)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "published"], cwd=tmp_path, check=True)
    bad_previous = pd.DataFrame([pivot("X", "2026-10-14", 100, 999)])
    _, state = load_observation_history(tmp_path, as_of=date(2026, 10, 15),
                                        previous=bad_previous)
    assert state["usable"] is False and "does not match" in state["reason"]
    good_previous = pd.DataFrame([pivot("X", "2026-10-14", 100, 101)])
    _, state = load_observation_history(tmp_path, as_of=date(2026, 10, 15),
                                        previous=good_previous)
    assert state["usable"] is True


def test_retrospective_daily_price_does_not_create_a_past_signal():
    current = pd.DataFrame([pivot("FIRST", "2026-10-15", 100, 102)])
    complete = pd.DataFrame([pivot("OTHER", "2026-10-09", 100, 103)])
    previous = pd.DataFrame([pivot("OTHER", "2026-10-14", 100, 102)])
    frame = pd.DataFrame({"Close": [97, 102]},
                         index=pd.to_datetime(["2026-10-14", "2026-10-15"]))
    events = build_observation_events_v2(
        current, complete, as_of=date(2026, 10, 15),
        history_snapshots=[{"date": "2026-10-14", "frame": previous,
                            "source_ref": "commit", "verified": True, "origin": "MIDWEEK"}],
        diagnostic={"usable": True, "history_coverage": "AVAILABLE"},
        daily={"FIRST": frame},
    )
    assert len(events) == 1
    assert events[0]["identity"]["lifecycle"] == "FIRST_OBSERVED"
    assert events[0]["price"]["basis"] == "RETROSPECTIVE"
    assert events[0]["price"]["transition"] == "BELOW_TO_ZONE"
    assert events[0]["price"]["daily_path"][-1]["zone"] == "ZONE"


def test_last_complete_week_baseline_permits_first_midweek_session():
    current = pd.DataFrame([pivot("FIRST", "2026-10-12", 100, 101)])
    baseline = pd.DataFrame([pivot("FIRST", "2026-10-09", 0, 99, signal=False)])
    events = build_observation_events_v2(
        current, baseline, as_of=date(2026, 10, 12),
        history_snapshots=[],
        diagnostic={"usable": True, "history_coverage": "AVAILABLE"},
    )
    assert len(events) == 1
    assert events[0]["price"]["transition"] == "BELOW_TO_ZONE"


def test_weekend_a_legacy_functions_ast_identical_to_main(tmp_path):
    """Protect all existing A and replay functions; allow the B price event rewrite."""
    import ast
    from pathlib import Path
    import pytest

    repo = Path(__file__).resolve().parents[2]
    relative = "dashboard/services/bf_observation.py"
    original = subprocess.run(["git", "show", f"origin/main:{relative}"],
                              cwd=repo, capture_output=True, text=True)
    if original.returncode:
        pytest.skip("origin/main unavailable in non-full local checkout")
    latest = (repo / relative).read_text()
    def functions(text):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(text).body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    original_api = functions(original.stdout)
    current_api = functions(latest)
    # _same_identity (old structural helper) is deleted; _b_event is
    # intentionally replaced with price-only B logic. Every other legacy
    # function, especially the fixed Weekend A path, is AST-identical.
    intentionally_changed_b_only = {"_same_identity", "_b_event"}
    assert all(
        current_api.get(name) == impl
        for name, impl in original_api.items()
        if name not in intentionally_changed_b_only
    )
    assert {"_price_path", "build_observation_view",
            "build_observation_attention"} <= original_api.keys()
    assert "_same_identity" not in current_api


def test_publication_cutoff_excludes_future_history(tmp_path):
    from datetime import datetime, timezone
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    p = tmp_path / "us" / "breakout_follow_pool_midweek.csv"
    p.parent.mkdir()
    pd.DataFrame([pivot("X", "2026-10-12", 100, 102)]).to_csv(p, index=False)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "published later"], cwd=tmp_path, check=True)
    old_cutoff = datetime(2026, 10, 12, 0, 0, tzinfo=timezone.utc)
    history, diagnostics = load_observation_history(
        tmp_path, as_of=date(2026, 10, 15), run_cutoff=old_cutoff)
    assert history == []
    assert diagnostics["snapshot_dates"] == []


def test_a_legacy_v1_event_remains_eligible_after_v2_import():
    from dashboard.services.bf_observation import build_observation_attention
    weekend = pivot("FIXED", "2026-10-09", 100, 108, valid=False)
    previous = pivot("FIXED", "2026-10-14", 0, 98, signal=False)
    current = pivot("FIXED", "2026-10-15", 0, 102, signal=False)
    events = build_observation_attention(
        pd.DataFrame([current]), pd.DataFrame([weekend]),
        pd.DataFrame(), previous=pd.DataFrame([previous]),
        daily={}, as_of=date(2026, 10, 15),
    )
    assert [(e["source"], e["event"], e["status"]) for e in events] == [
        ("WEEKEND", "BELOW → ZONE", "UNCONFIRMED")
    ]


def test_real_october_published_history_and_1d_replay(tmp_path):
    """Historical evidence integration: no invented A/B/WATCH states.

    All Pool inputs are pinned 2026-10-16 Git objects, not the mutable
    current CSV. Git history is cut off before later publications.
    """
    from datetime import datetime, timezone
    from io import StringIO
    from pathlib import Path
    from dashboard.services.bf_observation import load_daily_snapshot

    root = Path(__file__).resolve().parents[2]
    sha_current = "25839698482295ee2a076df5e0cdf590f12e03db"
    sha_previous = "451ffb7f0fdc5e8c5be29803d54e613e67724a2e"

    def pool(sha, filename):
        r = subprocess.run(["git", "show", f"{sha}:us/{filename}"],
                           cwd=root, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return pd.read_csv(StringIO(r.stdout), dtype={"code": str})

    current = pool(sha_current, "breakout_follow_pool_midweek.csv")
    complete = pool(sha_current, "breakout_follow_pool.csv")
    previous = pool(sha_previous, "breakout_follow_pool_midweek.csv")
    cutoff = datetime(2026, 10, 16, 9, 0, tzinfo=timezone.utc)
    snapshots, preflight = load_observation_history(
        root, as_of=date(2026, 10, 15), previous=previous, run_cutoff=cutoff)
    assert preflight["usable"] is True, preflight
    assert {"2026-10-12", "2026-10-14"} <= set(preflight["snapshot_dates"])

    # Use the *historically pinned* 1D PKL even when main rotates to newer
    # artifacts. This regression must not silently skip as data files roll.
    raw_pkl = subprocess.run(
        ["git", "show", f"{sha_current}:results_pkl/stock_data_161026_1d.pkl"],
        cwd=root, capture_output=True, check=False,
    )
    assert raw_pkl.returncode == 0, raw_pkl.stderr.decode(errors="replace")
    results = tmp_path / "results_pkl"
    results.mkdir()
    (results / "stock_data_161026_1d.pkl").write_bytes(raw_pkl.stdout)
    daily = load_daily_snapshot(tmp_path, window_date=date(2026, 10, 16))
    assert daily, "pinned historical 1D PKL is required for real replay"
    events = build_observation_events_v2(
        current, complete, as_of=date(2026, 10, 15),
        history_snapshots=snapshots, diagnostic=preflight, daily=daily)
    by_code = {item["code"]: item for item in events}
    andg = by_code["ANDG"]
    assert andg["identity"]["lifecycle"] == "REAPPEARED"
    assert andg["price"]["transition"] == "BELOW_TO_ZONE"
    assert andg["price"]["basis"] == "TRACKED_ANCHOR"
    assert andg["watch"]["latest"]["type"] == "ma10_pullback"
    assert andg["watch"]["latest"]["origin"] == "MIDWEEK"
    assert andg["watch"]["latest"]["relation"] == "TICKER_CONTEXT"
    aout = by_code["AOUT"]
    assert aout["identity"]["lifecycle"] == "FIRST_OBSERVED"
    assert aout["price"]["basis"] == "RETROSPECTIVE"
    assert aout["watch"]["active_prior"] is False
    assert aout["price"]["transition"] == "BELOW_TO_ZONE"
    ko = by_code["KO"]
    assert ko["identity"]["lifecycle"] == "FIRST_OBSERVED"
    assert ko["anchor_key_v2"] == "price|91.87"


def test_git_history_duplicate_ticker_cannot_be_authoritative(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    path = tmp_path / "us" / "breakout_follow_pool_midweek.csv"
    path.parent.mkdir()
    entry = pivot("DUP", "2026-10-14", 100, 102)
    pd.DataFrame([entry, entry]).to_csv(path, index=False)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "invalid published pool"], cwd=tmp_path, check=True)
    found, diagnostic = load_observation_history(tmp_path, as_of=date(2026, 10, 15))
    assert found == [] and diagnostic["usable"] is False
    assert "duplicates" in diagnostic["reason"]


def test_missing_history_after_first_market_day_fails_closed(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "--allow-empty", "-qm", "initial"], cwd=tmp_path, check=True)
    stale = pd.DataFrame([pivot("X", "2026-10-09", 0, 100, signal=False)])
    current_week = pd.DataFrame([pivot("X", "2026-10-12", 100, 102)])
    _, state = load_observation_history(
        tmp_path, as_of=date(2026, 10, 14), previous=stale)
    assert state["usable"] is False and "missing prior" in state["reason"]
    _, state = load_observation_history(
        tmp_path, as_of=date(2026, 10, 12), previous=stale)
    assert state["usable"] is True
    assert state["snapshot_dates"] == []
    # An uncommitted same-week prior must NOT be treated as a first session.
    _, state = load_observation_history(
        tmp_path, as_of=date(2026, 10, 14), previous=current_week)
    assert state["usable"] is False


def test_monday_market_holiday_first_session_is_tuesday(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "--allow-empty", "-qm", "initial"], cwd=tmp_path, check=True)
    # Memorial Day 2026: Monday May 25 closed; Tuesday May 26 is first session.
    old = pd.DataFrame([pivot("X", "2026-05-22", 0, 99, signal=False)])
    _, state = load_observation_history(
        tmp_path, as_of=date(2026, 5, 26), previous=old)
    assert state["usable"] is True


def test_thursday_complete_on_good_friday_week_is_valid_for_monday_review():
    # April 3, 2026 was Good Friday; April 2 is the complete snapshot.
    prior = pivot("HOL", "2026-04-02", 0, 99, signal=False)
    current = pivot("HOL", "2026-04-06", 100, 102)
    events = build_observation_events_v2(
        pd.DataFrame([current]), pd.DataFrame([prior]),
        as_of=date(2026, 4, 6),
        history_snapshots=[], diagnostic={
            "usable": True, "history_coverage": "AVAILABLE",
        },
    )
    assert len(events) == 1
    assert events[0]["price"]["transition"] == "BELOW_TO_ZONE"


def test_bad_complete_snapshot_cannot_seed_b_replay():
    current = pivot("X", "2026-10-15", 100, 102)
    prior = pivot("X", "2026-10-02", 0, 99, signal=False)
    events = build_observation_events_v2(
        pd.DataFrame([current]), pd.DataFrame([prior]),
        as_of=date(2026, 10, 15),
        history_snapshots=[], diagnostic={
            "usable": True, "history_coverage": "AVAILABLE",
        },
    )
    assert events == []



def test_watch_combines_only_with_same_previous_price_observation():
    old = pivot("W", "2026-10-12", 100, 98, valid=False)
    watched = pivot("W", "2026-10-13", 0, 99, signal=False)
    watched.update(bf_watch_active=True, bf_watch_type="pivot")
    no_watch = pivot("W", "2026-10-14", 0, 99, signal=False)
    current = pivot("W", "2026-10-15", 100, 102)
    event = observe("W", current, [
        hist("2026-10-12", old), hist("2026-10-13", watched),
        hist("2026-10-14", no_watch),
    ])
    assert event["watch"]["active_prior"] is True
    assert event["watch"]["active_at_price_boundary"] is False
    watched["snapshot_date"] = "2026-10-14"
    event = observe("W", current, [
        hist("2026-10-12", old), hist("2026-10-14", watched),
    ])
    assert event["watch"]["active_at_price_boundary"] is True


def test_v2_rejects_inconsistent_resolver_state_even_with_zone_price():
    before = pivot("INVALID", "2026-10-14", 0, 99, signal=False)
    current = pivot("INVALID", "2026-10-15", 100, 102)
    for status, valid in (
        ("EXTENDED", False),
        ("BELOW_TRIGGER", False),
        ("ACTIONABLE", False),
        ("UNCONFIRMED", True),
    ):
        current["ibd_entry_status"] = status
        current["ibd_entry_valid"] = valid
        assert observe("INVALID", current, [hist("2026-10-14", before)]) is None


def test_b_price_zone_is_shared_with_authoritative_a_classifier():
    from dashboard.services.bf_transition import _price_zone
    aliases = {"BELOW_BUY_POINT": "BELOW", "BUY_ZONE": "ZONE", "EXTENDED": "EXT"}
    for close in (99.999, 100, 101, 104.994, 105, 105.004, 105.011):
        assert zone(100, close) == aliases[_price_zone(100, close)]


def test_midweek_facts_are_independent_of_telegram_policy():
    from dashboard.services.bf_observation_replay import (
        resolve_midweek_facts, midweek_opportunity_policy,
    )
    prior = pivot("X", "2026-10-14", 100, 103, valid=True, status="ACTIONABLE")
    extended = pivot("X", "2026-10-15", 100, 108, valid=True, status="EXTENDED")
    facts = resolve_midweek_facts(
        code="X", current=extended, history=[hist("2026-10-14", prior)],
        weekend=None, as_of="2026-10-15", review_week="2026-10-12",
        baseline_verified=True,
    )
    assert facts["identity"]["lifecycle"] == "CONTINUING"
    assert facts["price"]["current_zone"] == "EXT"
    assert facts["entry"]["status"] == "EXTENDED"
    assert midweek_opportunity_policy(facts) is None


def test_even_minor_pkl_pool_zone_disagreement_must_fail_closed():
    current = pivot("MISMATCH", "2026-10-15", 100, 100.2)
    complete = pivot("MISMATCH", "2026-10-09", 0, 98, signal=False)
    prior = pivot("MISMATCH", "2026-10-14", 0, 99, signal=False)
    daily = pd.DataFrame(
        {"Close": [99.0, 99.9]},
        index=pd.to_datetime(["2026-10-14", "2026-10-15"]),
    )
    # The current PKL vs Pool difference is < 2% but crosses the actual
    # buy point boundary (BELOW vs ZONE).
    assert abs(99.9 / 100.2 - 1) < 0.02
    out = build_observation_events_v2(
        pd.DataFrame([current]), pd.DataFrame([complete]),
        as_of=date(2026, 10, 15), daily={"MISMATCH": daily},
        history_snapshots=[{"date": "2026-10-14", "frame": pd.DataFrame([prior]),
                            "source_ref": "published", "verified": True,
                            "origin": "MIDWEEK"}],
        diagnostic={"usable": True, "history_coverage": "AVAILABLE"},
    )
    assert out == []


def test_missing_previous_pool_close_uses_verified_daily_without_inventing_signal():
    previous = pivot("MISSING", "2026-10-14", 0, 99, signal=False)
    previous["latest_close"] = None
    current = pivot("MISSING", "2026-10-15", 100, 102, status="UNCONFIRMED")
    baseline = pivot("MISSING", "2026-10-09", 0, 98, signal=False)
    daily = pd.DataFrame(
        {"Close": [99.0, 102.0]},
        index=pd.to_datetime(["2026-10-14", "2026-10-15"]),
    )
    events = build_observation_events_v2(
        pd.DataFrame([current]), pd.DataFrame([baseline]),
        as_of=date(2026, 10, 15),
        history_snapshots=[{
            "date": "2026-10-14", "frame": pd.DataFrame([previous]),
            "source_ref": "published", "verified": True, "origin": "MIDWEEK",
        }],
        diagnostic={"usable": True, "history_coverage": "AVAILABLE"},
        daily={"MISSING": daily},
    )
    assert len(events) == 1
    assert events[0]["price"]["previous_zone"] == "BELOW"
    assert events[0]["price"]["transition"] == "BELOW_TO_ZONE"
    assert events[0]["price"]["basis"] == "RETROSPECTIVE"


def test_invalid_historical_signal_never_becomes_false_first_observed(tmp_path):
    """An unparseable historical signal must fail closed, not erase identity."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"],
                   cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    target = tmp_path / "us" / "breakout_follow_pool_midweek.csv"
    target.parent.mkdir()
    historical = pivot("CORRUPT", "2026-10-14", 100, 102)
    historical["signal"] = "UNKNOWN"
    published = pd.DataFrame([historical])
    published.to_csv(target, index=False)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "historical bad boolean"],
                   cwd=tmp_path, check=True)

    snapshots, diagnostics = load_observation_history(
        tmp_path, as_of=date(2026, 10, 15), previous=published)
    assert snapshots == []
    assert diagnostics["usable"] is False
    assert "invalid booleans" in diagnostics["reason"]


def test_duplicate_daily_dates_with_conflicting_closes_fail_closed():
    """Conflicting 1D bars must not be flattened into an arbitrary direction."""
    current = pivot("DUPDAY", "2026-10-15", 100, 102)
    baseline = pivot("DUPDAY", "2026-10-09", 0, 98, signal=False)
    prior = pivot("DUPDAY", "2026-10-14", 0, None, signal=False)
    conflicting = pd.DataFrame(
        {"Close": [99.0, 101.0, 102.0]},
        index=pd.to_datetime(["2026-10-14", "2026-10-14", "2026-10-15"]),
    )
    result = build_observation_events_v2(
        pd.DataFrame([current]), pd.DataFrame([baseline]),
        as_of=date(2026, 10, 15),
        history_snapshots=[{"date": "2026-10-14", "frame": pd.DataFrame([prior]),
                            "source_ref": "published", "verified": True,
                            "origin": "MIDWEEK"}],
        diagnostic={"usable": True, "history_coverage": "AVAILABLE"},
        daily={"DUPDAY": conflicting},
    )
    assert result == []
