from __future__ import annotations

import json

import pandas as pd

from dashboard.services import analyze_bf_transitions
from dashboard.services.bf_transition import PUSH_BASELINE_PREVIOUS_MIDWEEK


def _row(code: str, day: str, *, signal: bool, status=None, valid=None, close=100.0):
    candidate = 100.0 if signal else None
    return {
        "code": code,
        "snapshot_date": day,
        "signal": signal,
        "signal_source": "pivot" if signal else None,
        "latest_close": close,
        "ibd_candidate_price": candidate,
        "ibd_candidate_rule": "pivot" if signal else None,
        "ibd_entry_valid": valid,
        "ibd_entry_status": status,
        "ibd_entry_volume_ratio": 1.7 if signal else None,
        "volume_ratio": 1.4,
        "eps_yoy_growth": 30.0,
        "pullback_v_is_dry": True,
        "sector": "Technology",
        "industry": "Software",
    }


def test_push_signal_origin_is_relative_to_previous_midweek_not_weekend():
    complete = pd.DataFrame([
        _row("BASE", "2026-07-24", signal=False, close=80.0),
    ])
    previous = pd.DataFrame([
        _row(
            "NEW_THIS_WEEK",
            "2026-07-27",
            signal=True,
            status="UNCONFIRMED",
            valid=False,
            close=99.0,
        ),
    ])
    current = pd.DataFrame([
        _row(
            "NEW_THIS_WEEK",
            "2026-07-28",
            signal=True,
            status="ACTIONABLE",
            valid=True,
            close=102.0,
        ),
    ])

    result = analyze_bf_transitions(current, complete, previous)
    event = result.attention_events[0]

    assert result.push_baseline == PUSH_BASELINE_PREVIOUS_MIDWEEK
    assert event.event_type == "BECAME_ACTIONABLE"
    assert event.signal_origin == "RECONFIRMED"
    assert event.reasons == ("BECAME_ACTIONABLE",)
    assert event.facts["weekend_signal_origin"] == "NEW"

    payload = result.push_payload()
    assert payload["baseline"] == PUSH_BASELINE_PREVIOUS_MIDWEEK
    assert payload["baseline_snapshot_date"] == "2026-07-27"
    assert payload["current_snapshot_date"] == "2026-07-28"
    assert payload["events"][0]["signal_origin"] == "RECONFIRMED"
    json.dumps(payload)
