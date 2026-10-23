"""Data-driven invariant audit of the checked-in, latest Midweek Pool.

These assertions intentionally avoid ticker-specific counts and accommodate the
ordinary daily CSV updates while protecting source-specific Review invariants.
"""
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from bf_snapshot import monday_of_week

from dashboard.services.bf_observation import build_observation_view, load_previous_midweek
from dashboard.services.bf_observation_replay import zone
from dashboard.services.bf_transition import _same_buy_point_price
from dashboard.services.bf_review_presentation import decorate_observation_view
from dashboard.services.bf_transition import to_bool

ROOT = Path(__file__).resolve().parents[2]


def test_current_authoritative_midweek_records_preserve_buy_point_boundaries():
    weekend = pd.read_csv(ROOT / "us/breakout_follow_pool.csv", dtype={"code": str})
    current = pd.read_csv(ROOT / "us/breakout_follow_pool_midweek.csv", dtype={"code": str})
    snapshot = pd.to_datetime(current["snapshot_date"], errors="coerce").dropna().dt.date.unique()
    if len(snapshot) != 1:
        pytest.skip("No singular current Midweek market date")
    as_of = snapshot[0]
    weekend_dates = pd.to_datetime(weekend["snapshot_date"], errors="coerce").dropna().dt.date.unique()
    if (len(weekend_dates) != 1 or as_of <= weekend_dates[0]
            or (monday_of_week(as_of) - monday_of_week(weekend_dates[0])).days != 7):
        pytest.skip("Current Pool does not follow the previous complete trading week")

    previous = load_previous_midweek(ROOT, as_of)
    raw = build_observation_view(
        weekend, current, current, previous=previous, daily={}, as_of=as_of,
    )
    result = decorate_observation_view(
        raw, complete=weekend, current=current, previous=previous,
    )
    assert not result["code"].duplicated().any()
    indexed = result.set_index("code")
    for _, source in current.iterrows():
        if to_bool(source["signal"]) is not True:
            continue
        observed = indexed.loc[source["code"]]
        if bool(observed["review_b_active"]):
            expected_zone = zone(source["ibd_candidate_price"], source["latest_close"])
            assert observed["review_b_zone"] == expected_zone, source["code"]
            wk = weekend.loc[weekend["code"] == source["code"]]
            if not wk.empty and to_bool(wk.iloc[0]["signal"]) is True:
                assert not _same_buy_point_price(
                    source["ibd_candidate_price"], wk.iloc[0]["ibd_candidate_price"]
                ), source["code"]
            assert observed["review_new_buy_point"], source["code"]
            assert observed["review_new_buy_point_zone"] == {
                "BELOW": "BELOW_BUY_POINT", "ZONE": "BUY_ZONE", "EXT": "EXTENDED",
            }[expected_zone], source["code"]

    # A historical Weekend-vs-now difference remains visible cumulatively,
    # but if there is no latest price-zone or membership transition it must
    # not repeatedly enter Review Now on a later trading session.
    from dashboard.services.bf_review_presentation import _first_trading_session
    if not _first_trading_session(as_of):
        for _, observation in result[result["review_a_active"].eq(True)].iterrows():
            if (bool(observation.get("review_a_week_event"))
                    and not bool(observation.get("review_a_event"))
                    and not bool(observation.get("review_membership_event"))):
                assert not bool(observation["review_a_has_value_change"]), observation["code"]

    # Exactly one immutable Weekend anchor per fixed weekend signal.
    tracked = weekend[weekend["signal"].map(to_bool).eq(True)]
    for _, source in tracked.iterrows():
        row = indexed.loc[source["code"]]
        assert bool(row["review_a_active"]), source["code"]
        assert float(row["review_a_price"]) == pytest.approx(float(source["ibd_candidate_price"]))
        observed_date = row.get("review_a_as_of")
        if isinstance(observed_date, str) and observed_date:
            assert date.fromisoformat(observed_date) <= as_of, source["code"]
