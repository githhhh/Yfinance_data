from __future__ import annotations

import json

from dashboard.build_static import _buy_point_date


def _row(rule: str, price: float, extra: dict) -> dict:
    return {
        "ibd_candidate_rule": rule,
        "ibd_candidate_price": price,
        "ibd_candidate_extra": json.dumps(extra),
    }


def test_ceiling_pullback_legacy_touch_date_is_exact_when_high_never_changed():
    row = _row(
        "ceiling_pullback",
        48.21,
        {
            "pending_high": 48.21,
            "touch_high": 48.21,
            "touch_date": "2026-08-17",
            "confirm_date": "2026-09-07",
        },
    )

    assert _buy_point_date(row) == "2026-08-17"


def test_ceiling_pullback_does_not_mislabel_confirm_date_as_formation_date():
    row = _row(
        "ceiling_pullback",
        50.0,
        {
            "pending_high": 50.0,
            "touch_high": 48.21,
            "touch_date": "2026-08-17",
            "confirm_date": "2026-09-07",
        },
    )

    assert _buy_point_date(row) is None


def test_ceiling_pullback_prefers_explicit_pending_high_date():
    row = _row(
        "ceiling_pullback",
        50.0,
        {
            "pending_high": 50.0,
            "pending_high_date": "2026-08-31",
            "touch_high": 48.21,
            "touch_date": "2026-08-17",
            "confirm_date": "2026-09-07",
        },
    )

    assert _buy_point_date(row) == "2026-08-31"


def test_ma10_requires_authoritative_pending_high_date():
    legacy = _row(
        "ma10_touch_confirm",
        55.785,
        {
            "pending_high": 55.785,
            "touch_date": "2026-08-24",
            "confirm_date": "2026-09-07",
        },
    )
    authoritative = _row(
        "ma10_touch_confirm",
        55.785,
        {
            "pending_high": 55.785,
            "pending_high_date": "2026-08-31",
            "touch_date": "2026-08-24",
            "confirm_date": "2026-09-07",
        },
    )

    assert _buy_point_date(legacy) is None
    assert _buy_point_date(authoritative) == "2026-08-31"


def test_three_weeks_tight_requires_authoritative_twk_high_date():
    legacy = _row(
        "three_weeks_tight",
        68.38,
        {
            "twk_high": 68.38,
            "confirm_date": "2026-09-07",
        },
    )
    authoritative = _row(
        "three_weeks_tight",
        68.38,
        {
            "twk_high": 68.38,
            "twk_high_date": "2026-08-31",
            "confirm_date": "2026-09-07",
        },
    )

    assert _buy_point_date(legacy) is None
    assert _buy_point_date(authoritative) == "2026-08-31"
