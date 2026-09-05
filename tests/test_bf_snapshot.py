from datetime import date

import pytest

from bf_snapshot import (
    BreakoutFollowPoolKind,
    classify_breakout_follow_pool,
    complete_target_week,
    is_valid_complete_baseline,
)


@pytest.mark.parametrize(
    ("run_date", "expected"),
    [
        ("2026-06-15", BreakoutFollowPoolKind.COMPLETE),
        ("2026-06-16", BreakoutFollowPoolKind.MIDWEEK),
        ("2026-06-17", BreakoutFollowPoolKind.MIDWEEK),
        ("2026-06-18", BreakoutFollowPoolKind.MIDWEEK),
        ("2026-06-19", BreakoutFollowPoolKind.MIDWEEK),
        ("2026-06-20", BreakoutFollowPoolKind.COMPLETE),
        ("2026-06-21", BreakoutFollowPoolKind.COMPLETE),
    ],
)
def test_scheduler_window_classification(run_date, expected):
    assert classify_breakout_follow_pool(run_date) is expected


def test_market_data_classification_handles_friday_holiday_recovery():
    assert classify_breakout_follow_pool(
        "2026-07-31", file_date="2026-08-01"
    ) is BreakoutFollowPoolKind.COMPLETE
    assert classify_breakout_follow_pool(
        "2026-07-30", file_date="2026-08-01"
    ) is BreakoutFollowPoolKind.COMPLETE
    assert classify_breakout_follow_pool(
        "2026-07-30", file_date="2026-07-31"
    ) is BreakoutFollowPoolKind.MIDWEEK
    assert classify_breakout_follow_pool(
        "2026-08-05", file_date="2026-08-06"
    ) is BreakoutFollowPoolKind.MIDWEEK
    assert classify_breakout_follow_pool(
        "2026-08-03", file_date="2026-08-04"
    ) is BreakoutFollowPoolKind.MIDWEEK


def test_complete_target_week_accepts_thursday_for_known_complete_pool():
    assert complete_target_week(date(2026, 7, 23)) == date(2026, 7, 27)
    assert complete_target_week(date(2026, 7, 24)) == date(2026, 7, 27)


@pytest.mark.parametrize("snapshot", [date(2026, 7, 21), date(2026, 7, 22)])
def test_complete_target_week_rejects_tuesday_and_wednesday(snapshot):
    with pytest.raises(ValueError, match="complete snapshot"):
        complete_target_week(snapshot)


def test_thursday_complete_is_valid_baseline_for_next_week_midweek():
    assert is_valid_complete_baseline(date(2026, 7, 23), date(2026, 7, 28))
    assert is_valid_complete_baseline(date(2026, 7, 23), date(2026, 7, 29))
    assert is_valid_complete_baseline(date(2026, 7, 23), date(2026, 7, 30))


def test_stale_complete_is_not_a_valid_baseline():
    assert not is_valid_complete_baseline(date(2026, 7, 16), date(2026, 7, 29))
