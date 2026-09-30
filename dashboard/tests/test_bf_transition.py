from __future__ import annotations

import json
from datetime import date

import pandas as pd

from dashboard.services import analyze_bf_transitions
from dashboard.services.bf_midweek_review import build_midweek_review
from dashboard.services.bf_transition import (
    PUSH_BASELINE_COMPLETE,
    PUSH_BASELINE_PREVIOUS_MIDWEEK,
)


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
    candidate_extra: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "code": code,
        "snapshot_date": snapshot_date,
        "signal": signal,
        "signal_source": "pivot" if signal else None,
        "latest_close": close,
        "ibd_candidate_price": candidate,
        "ibd_candidate_rule": rule,
        "ibd_candidate_extra": json.dumps(candidate_extra) if candidate_extra is not None else None,
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


def _event_map(result):
    return {event.code: event for event in result.attention_events}


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
                snapshot_date="2026-07-27",
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
                snapshot_date="2026-07-27",
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


def test_first_midweek_run_uses_complete_baseline_and_returns_material_events():
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
                snapshot_date="2026-07-27",
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
                snapshot_date="2026-07-27",
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
                snapshot_date="2026-07-27",
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
                snapshot_date="2026-07-27",
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
    events = _event_map(result)

    assert result.push_baseline == PUSH_BASELINE_COMPLETE
    assert result.push_baseline_snapshot_date == date(2026, 7, 24)
    assert result.current_snapshot_date == date(2026, 7, 27)
    assert result.push_ready is True
    assert result.push_warnings == ()
    assert set(events) == {"BECAME", "BELOW", "EXTENDED"}
    assert events["BECAME"].event_type == "BECAME_ACTIONABLE"
    assert events["BECAME"].importance == "HIGH"
    assert events["BECAME"].facts["candidate_rule"] == "pivot"
    assert events["BECAME"].facts["candidate_price"] == 100.0
    assert events["BECAME"].facts["latest_close"] == 102.0
    assert events["BECAME"].facts["entry_volume_ratio"] == 1.7
    assert events["BELOW"].event_type == "ACTIONABLE_TO_BELOW_TRIGGER"
    assert events["EXTENDED"].event_type == "ACTIONABLE_TO_EXTENDED"
    assert result.attention_summary == {"TOTAL": 3, "HIGH": 2, "MEDIUM": 1}
    json.dumps(events["BECAME"].to_dict())


def test_previous_midweek_is_real_push_baseline_for_actionable_to_below_trigger():
    complete = pd.DataFrame(
        [
            _row(
                "AAPL",
                snapshot_date="2026-07-24",
                signal=True,
                status="UNCONFIRMED",
                valid=False,
                candidate=100,
                close=99,
                rule="pivot",
            )
        ]
    )
    previous = pd.DataFrame(
        [
            _row(
                "AAPL",
                snapshot_date="2026-07-27",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=100,
                close=102,
                rule="pivot",
                entry_volume=1.7,
            )
        ]
    )
    current = pd.DataFrame(
        [
            _row(
                "AAPL",
                snapshot_date="2026-07-28",
                signal=True,
                status="BELOW_TRIGGER",
                valid=True,
                candidate=100,
                close=99,
                rule="pivot",
                entry_volume=1.7,
            )
        ]
    )

    result = analyze_bf_transitions(current, complete, previous)
    event = result.attention_events[0]

    # Dashboard stays weekend -> current and therefore sees only OTHER_CHANGES.
    assert result.rows.iloc[0]["review_change_group"] == "OTHER_CHANGES"
    # Push correctly sees yesterday ACTIONABLE -> today BELOW_TRIGGER.
    assert result.push_baseline == PUSH_BASELINE_PREVIOUS_MIDWEEK
    assert result.push_baseline_snapshot_date == date(2026, 7, 27)
    assert event.event_type == "ACTIONABLE_TO_BELOW_TRIGGER"
    assert event.baseline_status == "ACTIONABLE"
    assert event.current_status == "BELOW_TRIGGER"


def test_previous_midweek_captures_actionable_to_extended_even_when_weekend_was_extended():
    complete = pd.DataFrame(
        [
            _row(
                "XYZ",
                snapshot_date="2026-07-24",
                signal=True,
                status="EXTENDED",
                valid=True,
                candidate=100,
                close=108,
                rule="pivot",
            )
        ]
    )
    previous = pd.DataFrame(
        [
            _row(
                "XYZ",
                snapshot_date="2026-07-27",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=100,
                close=103,
                rule="pivot",
            )
        ]
    )
    current = pd.DataFrame(
        [
            _row(
                "XYZ",
                snapshot_date="2026-07-28",
                signal=True,
                status="EXTENDED",
                valid=True,
                candidate=100,
                close=108,
                rule="pivot",
            )
        ]
    )

    result = analyze_bf_transitions(current, complete, previous)
    event = result.attention_events[0]

    assert result.rows.iloc[0]["review_entry_change"] == "UNCHANGED"
    assert event.event_type == "ACTIONABLE_TO_EXTENDED"
    assert event.importance == "MEDIUM"


def test_newer_complete_snapshot_resets_push_baseline_for_new_review_week():
    previous = pd.DataFrame(
        [
            _row(
                "RESET",
                snapshot_date="2026-07-30",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=100,
                close=102,
                rule="pivot",
            )
        ]
    )
    complete = pd.DataFrame(
        [
            _row(
                "RESET",
                snapshot_date="2026-07-31",
                signal=True,
                status="UNCONFIRMED",
                valid=False,
                candidate=100,
                close=99,
                rule="pivot",
            )
        ]
    )
    current = pd.DataFrame(
        [
            _row(
                "RESET",
                snapshot_date="2026-08-03",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=100,
                close=102,
                rule="pivot",
            )
        ]
    )

    result = analyze_bf_transitions(current, complete, previous)

    assert result.push_baseline == PUSH_BASELINE_COMPLETE
    assert result.push_baseline_snapshot_date == date(2026, 7, 31)
    assert result.attention_events[0].event_type == "BECAME_ACTIONABLE"
    assert result.attention_events[0].baseline_status == "UNCONFIRMED"


def test_previous_comparison_uses_effective_carry_state_not_raw_previous_status():
    complete = pd.DataFrame(
        [
            _row(
                "CARRY",
                snapshot_date="2026-07-24",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=100,
                close=102,
                rule="ceiling",
                entry_volume=2.1,
            )
        ]
    )
    # Raw midweek signal is false, but Dashboard Carry resolution makes it
    # effectively ACTIONABLE using the weekend candidate and latest close.
    previous = pd.DataFrame(
        [
            _row(
                "CARRY",
                snapshot_date="2026-07-27",
                signal=False,
                close=102,
            )
        ]
    )
    current = pd.DataFrame(
        [
            _row(
                "CARRY",
                snapshot_date="2026-07-28",
                signal=False,
                close=99,
            )
        ]
    )

    result = analyze_bf_transitions(current, complete, previous)
    event = result.attention_events[0]

    assert event.event_type == "ACTIONABLE_TO_BELOW_TRIGGER"
    assert event.baseline_status == "ACTIONABLE"
    assert event.current_status == "BELOW_TRIGGER"
    assert event.facts["candidate_price"] == 100.0


def test_previous_actionable_that_exits_current_pool_is_high_attention():
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
    previous = pd.DataFrame(
        [
            _row(
                "EXIT",
                snapshot_date="2026-07-27",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=100,
                close=102,
                rule="pivot",
            ),
            _row(
                "KEEP",
                snapshot_date="2026-07-27",
                signal=False,
                close=80,
            ),
        ]
    )
    current = pd.DataFrame(
        [
            _row(
                "KEEP",
                snapshot_date="2026-07-28",
                signal=False,
                close=80,
            )
        ]
    )

    result = analyze_bf_transitions(current, complete, previous)
    event = result.attention_events[0]

    assert event.code == "EXIT"
    assert event.event_type == "ACTIONABLE_EXITED_POOL"
    assert event.importance == "HIGH"
    assert event.baseline_status == "ACTIONABLE"
    assert event.current_status is None


def test_same_or_older_current_snapshot_suppresses_push_without_changing_dashboard_rows():
    complete = pd.DataFrame(
        [
            _row(
                "STALE",
                snapshot_date="2026-07-24",
                signal=True,
                status="UNCONFIRMED",
                valid=False,
                candidate=100,
                close=99,
                rule="pivot",
            )
        ]
    )
    previous = pd.DataFrame(
        [
            _row(
                "STALE",
                snapshot_date="2026-07-28",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=100,
                close=102,
                rule="pivot",
            )
        ]
    )
    current = pd.DataFrame(
        [
            _row(
                "STALE",
                snapshot_date="2026-07-28",
                signal=True,
                status="BELOW_TRIGGER",
                valid=True,
                candidate=100,
                close=99,
                rule="pivot",
            )
        ]
    )

    result = analyze_bf_transitions(current, complete, previous)

    assert result.push_ready is False
    assert result.attention_events == ()
    assert result.attention_summary == {"TOTAL": 0, "HIGH": 0, "MEDIUM": 0}
    assert result.push_warnings
    assert result.rows.iloc[0]["review_change_group"] == "OTHER_CHANGES"


def test_midweek_buy_point_review_tracks_weekend_and_new_anchor_independently():
    weekend_extra = {"pivot_candidates": [{"box_type": "M_BOX", "price": 100.0, "resistance_date": "2026-07-24"}]}
    new_extra = {"pivot_candidates": [{"box_type": "S_BOX", "price": 106.0, "resistance_date": "2026-07-27"}]}
    complete = pd.DataFrame([_row("DUAL", snapshot_date="2026-07-24", signal=True, status="ACTIONABLE", valid=True, candidate=100.0, close=102.0, rule="pivot", entry_volume=1.8, candidate_extra=weekend_extra)])
    current = pd.DataFrame([_row("DUAL", snapshot_date="2026-07-27", signal=True, status="UNCONFIRMED", valid=False, candidate=106.0, close=108.0, rule="pivot", candidate_extra=new_extra)])
    row = analyze_bf_transitions(current, complete).rows.iloc[0]
    assert row["review_weekend_buy_point_previous_zone"] == "BUY_ZONE"
    assert row["review_weekend_buy_point_zone"] == "EXTENDED"
    assert row["review_weekend_buy_point_change"] == "BECAME_EXTENDED"
    assert bool(row["review_new_buy_point"]) is True
    assert row["review_new_buy_point_zone"] == "BUY_ZONE"
    assert row["review_new_buy_point_date"] == "2026-07-27"
    assert bool(row["review_has_value_change"]) is True
    assert row["review_buy_point_change_label"] == "Became Extended · New BP · In Buy Zone"
    assert row["review_effective_entry_status"] == "UNCONFIRMED"


def test_new_buy_point_identity_uses_anchor_not_trigger_price():
    complete = pd.DataFrame([_row("ANCHOR", snapshot_date="2026-07-24", signal=True, status="ACTIONABLE", valid=True, candidate=100.0, close=102.0, rule="pivot", candidate_extra={"pivot_candidates": [{"box_type": "M_BOX", "price": 100.0, "resistance_date": "2026-07-24"}]})])
    current = pd.DataFrame([_row("ANCHOR", snapshot_date="2026-07-27", signal=True, status="ACTIONABLE", valid=True, candidate=100.0, close=103.0, rule="pivot", candidate_extra={"pivot_candidates": [{"box_type": "M_BOX", "price": 100.0, "resistance_date": "2026-07-27"}]})])
    row = analyze_bf_transitions(current, complete).rows.iloc[0]
    assert bool(row["review_new_buy_point"]) is True
    assert row["review_new_buy_point_date"] == "2026-07-27"



def test_same_rule_and_anchor_date_is_not_new_when_box_metadata_changes():
    complete = pd.DataFrame([
        _row(
            "SAME",
            snapshot_date="2026-07-24",
            signal=True,
            status="ACTIONABLE",
            valid=True,
            candidate=100.0,
            close=102.0,
            rule="pivot",
            candidate_extra={
                "pivot_candidates": [
                    {"box_type": "S_BOX", "price": 100.0, "resistance_date": "2026-07-24"},
                    {"box_type": "M_BOX", "price": 100.0, "resistance_date": "2026-07-24"},
                ]
            },
        )
    ])
    current = pd.DataFrame([
        _row(
            "SAME",
            snapshot_date="2026-07-27",
            signal=True,
            status="ACTIONABLE",
            valid=True,
            candidate=100.0,
            close=103.0,
            rule="pivot",
            candidate_extra={
                "selected_pivot": {
                    "box_type": "M_BOX",
                    "price": 100.0,
                    "resistance_date": "2026-07-24",
                },
                "pivot_candidates": [
                    {"box_type": "M_BOX", "price": 100.0, "resistance_date": "2026-07-24"}
                ],
            },
        )
    ])

    row = analyze_bf_transitions(current, complete).rows.iloc[0]
    assert bool(row["review_new_buy_point"]) is False


def test_different_current_rule_is_new_when_legacy_weekend_anchor_date_is_missing():
    complete = pd.DataFrame([
        _row(
            "LEGACY",
            snapshot_date="2026-07-24",
            signal=True,
            status="ACTIONABLE",
            valid=True,
            candidate=100.0,
            close=102.0,
            rule="ma10_touch_confirm",
            candidate_extra={"pending_high": 100.0, "touch_date": "2026-07-20"},
        )
    ])
    current = pd.DataFrame([
        _row(
            "LEGACY",
            snapshot_date="2026-07-27",
            signal=True,
            status="UNCONFIRMED",
            valid=False,
            candidate=106.0,
            close=108.0,
            rule="pivot",
            candidate_extra={
                "pivot_candidates": [
                    {"box_type": "S_BOX", "price": 106.0, "resistance_date": "2026-07-27"}
                ]
            },
        )
    ])

    row = analyze_bf_transitions(current, complete).rows.iloc[0]
    assert bool(row["review_new_buy_point"]) is True
    assert row["review_new_buy_point_zone"] == "BUY_ZONE"
