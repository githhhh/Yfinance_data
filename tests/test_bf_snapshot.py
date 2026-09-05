from datetime import date

import pytest

from bf_snapshot import (
    BreakoutFollowPoolKind,
    classify_breakout_follow_pool,
    complete_snapshot_week,
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
        "2026-07-02", file_date="2026-07-04"
    ) is BreakoutFollowPoolKind.COMPLETE
    assert classify_breakout_follow_pool(
        "2026-07-02", file_date="2026-07-03"
    ) is BreakoutFollowPoolKind.MIDWEEK
    assert classify_breakout_follow_pool(
        "2026-08-05", file_date="2026-08-06"
    ) is BreakoutFollowPoolKind.MIDWEEK
    assert classify_breakout_follow_pool(
        "2026-08-03", file_date="2026-08-04"
    ) is BreakoutFollowPoolKind.MIDWEEK


def test_market_data_classification_rejects_stale_thursday_on_saturday():
    assert classify_breakout_follow_pool(
        "2026-07-30", file_date="2026-08-01"
    ) is BreakoutFollowPoolKind.MIDWEEK


def test_juneteenth_is_not_applied_before_exchange_observance_started():
    assert classify_breakout_follow_pool(
        "2021-06-17", file_date="2021-06-19"
    ) is BreakoutFollowPoolKind.MIDWEEK
    with pytest.raises(ValueError, match="complete snapshot"):
        complete_snapshot_week(date(2021, 6, 17))


def test_new_years_day_saturday_does_not_close_prior_friday():
    assert classify_breakout_follow_pool(
        "2027-12-30", file_date="2028-01-01"
    ) is BreakoutFollowPoolKind.MIDWEEK
    assert classify_breakout_follow_pool(
        "2027-12-31", file_date="2028-01-01"
    ) is BreakoutFollowPoolKind.COMPLETE


def test_market_data_classification_does_not_treat_delayed_wednesday_as_complete():
    assert classify_breakout_follow_pool(
        "2026-07-29", file_date="2026-08-01"
    ) is BreakoutFollowPoolKind.MIDWEEK


def test_complete_target_week_accepts_thursday_for_known_complete_pool():
    assert complete_snapshot_week(date(2026, 7, 2)) == date(2026, 7, 3)
    assert complete_snapshot_week(date(2026, 7, 24)) == date(2026, 7, 24)
    assert complete_snapshot_week(date(2026, 12, 31)) == date(2027, 1, 1)
    assert complete_target_week(date(2026, 7, 2)) == date(2026, 7, 6)
    assert complete_target_week(date(2026, 7, 24)) == date(2026, 7, 27)
    assert complete_target_week(date(2026, 12, 31)) == date(2027, 1, 4)


@pytest.mark.parametrize(
    "snapshot",
    [
        date(2026, 7, 21),
        date(2026, 7, 22),
        date(2026, 7, 30),
        date(2026, 8, 1),
        date(2026, 8, 2),
        date(2026, 8, 3),
    ],
)
def test_complete_target_week_rejects_non_final_market_snapshots(snapshot):
    with pytest.raises(ValueError, match="complete snapshot"):
        complete_target_week(snapshot)


def test_thursday_complete_is_valid_baseline_for_next_week_midweek():
    assert is_valid_complete_baseline(date(2026, 7, 2), date(2026, 7, 7))
    assert is_valid_complete_baseline(date(2026, 7, 2), date(2026, 7, 8))
    assert is_valid_complete_baseline(date(2026, 7, 2), date(2026, 7, 9))


def test_ordinary_thursday_complete_is_not_a_valid_baseline():
    assert not is_valid_complete_baseline(date(2026, 7, 30), date(2026, 8, 4))


def test_stale_complete_is_not_a_valid_baseline():
    assert not is_valid_complete_baseline(date(2026, 7, 16), date(2026, 7, 29))
