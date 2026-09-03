from __future__ import annotations

import pandas as pd

from dashboard.services import analyze_bf_transitions
from dashboard.services.bf_midweek_review import build_midweek_review


def _row(
    code: str,
    *,
    snapshot_date: str,
    signal: bool,
    status: str | None = None,
    valid: bool | None = None,
    candidate: float | None = None,
    close: float = 100.0,
    rule: str | None = None,
    entry_volume: float | None = None,
) -> dict[str, object]:
    return {
        "code": code,
        "snapshot_date": snapshot_date,
        "signal": signal,
        "signal_source": "pivot" if signal else None,
        "latest_close": close,
        "ibd_candidate_price": candidate,
        "ibd_candidate_rule": rule,
        "ibd_entry_valid": valid,
        "ibd_entry_status": status,
        "current_vs_ibd_candidate_pct": (
            (close / candidate - 1.0) * 100.0 if candidate not in (None, 0) else None
        ),
        "ibd_entry_volume_ratio": entry_volume,
        "ibd_entry_reject_reason": None if valid else "Volume not confirmed",
        "ibd_entry_close_position": 0.82 if valid else None,
        "ibd_entry_breakout_range_ratio": 0.55 if valid else None,
        "volume_ratio": 1.4,
        "eps_yoy_growth": 31.0,
        "pullback_v_is_dry": True,
        "sector": "Technology",
        "industry": "Software",
        "rank_C_continuous": 1,
        "C_continuous": 1.0,
    }


def test_shared_transition_api_preserves_dashboard_projection_contract():
    complete = pd.DataFrame(
        [
            _row(
                "BECAME",
                snapshot_date="2026-07-24",
                signal=True,
                status="UNCONFIRMED",
                valid=False,
                candidate=100,
                close=99,
                rule="pivot",
            ),
            _row(
                "LEFT",
                snapshot_date="2026-07-24",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=100,
                close=103,
                rule="pivot",
                entry_volume=1.8,
            ),
        ]
    )
    current = pd.DataFrame(
        [
            _row(
                "BECAME",
                snapshot_date="2026-07-29",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=100,
                close=102,
                rule="pivot",
                entry_volume=1.7,
            ),
            _row(
                "LEFT",
                snapshot_date="2026-07-29",
                signal=True,
                status="BELOW_TRIGGER",
                valid=True,
                candidate=100,
                close=99,
                rule="pivot",
                entry_volume=1.6,
            ),
        ]
    )

    api = analyze_bf_transitions(current, complete)
    dashboard = build_midweek_review(current, complete)

    pd.testing.assert_frame_equal(api.rows, dashboard.current_review)
    pd.testing.assert_frame_equal(api.exited_pool, dashboard.exited_pool)
    assert api.review_summary == dashboard.summary
    assert api.actionable_codes == dashboard.actionable_codes == ("BECAME",)

    review = api.rows.set_index("code")
    assert review.loc["BECAME", "review_change_group"] == "BECAME_ACTIONABLE"
    assert review.loc["LEFT", "review_entry_change"] == "ACTIONABLE_TO_BELOW_TRIGGER"


def test_attention_api_returns_only_material_actionable_boundary_events_with_facts():
    complete = pd.DataFrame(
        [
            _row(
                "BECAME",
                snapshot_date="2026-07-24",
                signal=True,
                status="UNCONFIRMED",
                valid=False,
                candidate=100,
                close=99,
                rule="pivot",
            ),
            _row(
                "BELOW",
                snapshot_date="2026-07-24",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=100,
                close=102,
                rule="ceiling",
                entry_volume=1.9,
            ),
            _row(
                "EXTENDED",
                snapshot_date="2026-07-24",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=100,
                close=102,
                rule="pivot",
                entry_volume=1.6,
            ),
            _row(
                "OTHER",
                snapshot_date="2026-07-24",
                signal=True,
                status="BELOW_TRIGGER",
                valid=True,
                candidate=100,
                close=99,
                rule="pivot",
                entry_volume=1.4,
            ),
        ]
    )
    current = pd.DataFrame(
        [
            _row(
                "BECAME",
                snapshot_date="2026-07-29",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=100,
                close=102,
                rule="pivot",
                entry_volume=1.7,
            ),
            _row(
                "BELOW",
                snapshot_date="2026-07-29",
                signal=True,
                status="BELOW_TRIGGER",
                valid=True,
                candidate=100,
                close=99,
                rule="ceiling",
                entry_volume=1.5,
            ),
            _row(
                "EXTENDED",
                snapshot_date="2026-07-29",
                signal=True,
                status="EXTENDED",
                valid=True,
                candidate=100,
                close=108,
                rule="pivot",
                entry_volume=1.6,
            ),
            _row(
                "OTHER",
                snapshot_date="2026-07-29",
                signal=True,
                status="UNCONFIRMED",
                valid=False,
                candidate=100,
                close=101,
                rule="pivot",
            ),
        ]
    )

    result = analyze_bf_transitions(current, complete)
    events = {event.code: event for event in result.attention_events}

    assert set(events) == {"BECAME", "BELOW", "EXTENDED"}
    assert events["BECAME"].event_type == "BECAME_ACTIONABLE"
    assert events["BECAME"].importance == "HIGH"
    assert events["BECAME"].candidate_rule == "pivot"
    assert events["BECAME"].candidate_price == 100.0
    assert events["BECAME"].latest_close == 102.0
    assert events["BECAME"].entry_volume_ratio == 1.7
    assert events["BECAME"].eps_yoy_growth == 31.0
    assert events["BELOW"].event_type == "ACTIONABLE_TO_BELOW_TRIGGER"
    assert events["BELOW"].importance == "HIGH"
    assert events["EXTENDED"].event_type == "ACTIONABLE_TO_EXTENDED"
    assert events["EXTENDED"].importance == "MEDIUM"
    assert result.attention_summary == {
        "TOTAL": 3,
        "HIGH": 2,
        "MEDIUM": 1,
        "NOTIFICATION_ELIGIBLE": 3,
    }


def test_previous_pool_only_marks_new_material_events_for_notification():
    complete = pd.DataFrame(
        [
            _row(
                "OLD_BECAME",
                snapshot_date="2026-07-24",
                signal=True,
                status="UNCONFIRMED",
                valid=False,
                candidate=100,
                close=99,
                rule="pivot",
            ),
            _row(
                "NEW_BECAME",
                snapshot_date="2026-07-24",
                signal=True,
                status="UNCONFIRMED",
                valid=False,
                candidate=100,
                close=99,
                rule="pivot",
            ),
        ]
    )
    previous = pd.DataFrame(
        [
            _row(
                "OLD_BECAME",
                snapshot_date="2026-07-28",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=100,
                close=101,
                rule="pivot",
                entry_volume=1.6,
            ),
            _row(
                "NEW_BECAME",
                snapshot_date="2026-07-28",
                signal=True,
                status="UNCONFIRMED",
                valid=False,
                candidate=100,
                close=99,
                rule="pivot",
            ),
        ]
    )
    current = pd.DataFrame(
        [
            _row(
                "OLD_BECAME",
                snapshot_date="2026-07-29",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=100,
                close=102,
                rule="pivot",
                entry_volume=1.7,
            ),
            _row(
                "NEW_BECAME",
                snapshot_date="2026-07-29",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=100,
                close=102,
                rule="pivot",
                entry_volume=1.8,
            ),
        ]
    )

    result = analyze_bf_transitions(current, complete, previous)

    events = {event.code: event for event in result.attention_events}
    assert events["OLD_BECAME"].is_new_since_previous is False
    assert events["NEW_BECAME"].is_new_since_previous is True
    assert tuple(event.code for event in result.notification_events) == ("NEW_BECAME",)


def test_new_signal_actionable_is_high_attention_with_origin_context():
    complete = pd.DataFrame(
        [
            _row(
                "BASE",
                snapshot_date="2026-07-24",
                signal=False,
                close=80,
            )
        ]
    )
    current = pd.DataFrame(
        [
            _row(
                "NEW",
                snapshot_date="2026-07-29",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=100,
                close=102,
                rule="pivot",
                entry_volume=2.0,
            )
        ]
    )

    event = analyze_bf_transitions(current, complete).attention_events[0]

    assert event.code == "NEW"
    assert event.event_type == "BECAME_ACTIONABLE"
    assert event.signal_origin == "NEW"
    assert event.reasons == ("BECAME_ACTIONABLE", "NEW_SIGNAL")


def test_weekend_actionable_exiting_current_pool_is_attention_only_not_dashboard_row():
    complete = pd.DataFrame(
        [
            _row(
                "EXITED",
                snapshot_date="2026-07-24",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=100,
                close=102,
                rule="pivot",
                entry_volume=1.8,
            ),
            _row(
                "STAY",
                snapshot_date="2026-07-24",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=50,
                close=51,
                rule="pivot",
                entry_volume=1.6,
            ),
        ]
    )
    current = pd.DataFrame(
        [
            _row(
                "STAY",
                snapshot_date="2026-07-29",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=50,
                close=51,
                rule="pivot",
                entry_volume=1.6,
            )
        ]
    )

    result = analyze_bf_transitions(current, complete)

    assert list(result.rows["code"]) == ["STAY"]
    assert list(result.exited_pool["code"]) == ["EXITED"]
    event = next(event for event in result.attention_events if event.code == "EXITED")
    assert event.event_type == "ACTIONABLE_EXITED_POOL"
    assert event.importance == "HIGH"
