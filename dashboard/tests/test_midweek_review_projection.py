from __future__ import annotations

from datetime import date
import pandas as pd
import pytest

from dashboard.services.bf_midweek_review import (
    PoolMode,
    PoolWindow,
    analyze_breakout_follow_pool,
    build_review_filter_counts,
    build_midweek_review,
    build_midweek_review_for_snapshots,
    complete_target_week,
    default_review_state,
    materialize_review_view,
    resolve_window,
    switch_review_mode,
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
    rank: int = 1,
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
        "ibd_entry_volume_ratio": 2.0 if valid else None,
        "ibd_entry_reject_reason": None if valid else "Volume not confirmed",
        "volume_ratio": 1.4,
        "rank_C_continuous": rank,
        "C_continuous": float(rank),
    }


@pytest.mark.parametrize("weekday", [1, 2, 3, 4])
def test_resolve_window_uses_tuesday_through_friday_for_midweek(weekday):
    assert resolve_window(date(2026, 6, 15 + weekday)) is PoolWindow.MIDWEEK


@pytest.mark.parametrize("value", [date(2026, 6, 15), date(2026, 6, 20), date(2026, 6, 21)])
def test_resolve_window_uses_saturday_through_monday_for_complete(value):
    assert resolve_window(value) is PoolWindow.COMPLETE


@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        (date(2026, 7, 2), date(2026, 7, 6)),
        (date(2026, 7, 24), date(2026, 7, 27)),
        (date(2026, 12, 31), date(2027, 1, 4)),
    ],
)
def test_complete_target_week_maps_supported_snapshot_days(snapshot, expected):
    assert complete_target_week(snapshot) == expected


@pytest.mark.parametrize(
    "snapshot",
    [
        date(2026, 7, 21),
        date(2026, 7, 22),
        date(2026, 7, 30),
        date(2026, 7, 25),
        date(2026, 7, 26),
        date(2026, 7, 27),
    ],
)
def test_complete_target_week_fails_closed_for_non_final_market_snapshots(snapshot):
    with pytest.raises(ValueError, match="complete snapshot"):
        complete_target_week(snapshot)


def test_midweek_projection_is_current_left_and_applies_atomic_signal_ownership():
    complete = pd.DataFrame(
        [
            _row("CARRY", snapshot_date="2026-07-24", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=104, rule="ceiling", rank=1),
            _row("RECONF", snapshot_date="2026-07-24", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=103, rule="ceiling", rank=2),
            _row("EXITED", snapshot_date="2026-07-24", signal=True, status="ACTIONABLE", valid=True, candidate=40, close=42, rule="pivot", rank=3),
            _row("PLAIN", snapshot_date="2026-07-24", signal=False, rank=4),
        ]
    )
    current = pd.DataFrame(
        [
            _row("NEW", snapshot_date="2026-07-29", signal=True, status="ACTIONABLE", valid=True, candidate=50, close=51, rule="pivot", rank=10),
            _row("CARRY", snapshot_date="2026-07-29", signal=False, close=105, rank=11),
            _row("RECONF", snapshot_date="2026-07-29", signal=True, status="UNCONFIRMED", valid=False, candidate=120, close=121, rule="ma10_touch_confirm", rank=12),
            _row("PLAIN", snapshot_date="2026-07-29", signal=False, close=99, rank=13),
        ]
    )

    result = build_midweek_review(current, complete)
    review = result.current_review.set_index("code")

    assert list(result.current_review["code"]) == ["NEW", "CARRY", "RECONF", "PLAIN"]
    assert set(result.current_review["code"]) == set(current["code"])
    assert list(result.exited_pool["code"]) == ["EXITED"]
    assert review.loc["NEW", "review_signal_origin"] == "NEW"
    assert review.loc["CARRY", "review_signal_origin"] == "CARRY"
    assert review.loc["RECONF", "review_signal_origin"] == "RECONFIRMED"
    assert review.loc["PLAIN", "review_signal_origin"] == "NONE"
    assert review.loc["CARRY", "review_candidate_price"] == 100
    assert review.loc["CARRY", "review_effective_entry_status"] == "ACTIONABLE"
    assert review.loc["RECONF", "review_candidate_price"] == 120
    assert review.loc["RECONF", "review_effective_entry_status"] == "UNCONFIRMED"
    assert review.loc["RECONF", "review_entry_valid"] is False
    assert result.actionable_codes == ("NEW", "CARRY")
    assert "EXITED" not in result.actionable_codes

    display = materialize_review_view(result.current_review).set_index("code")
    assert display.loc["CARRY", "ibd_candidate_rule"] == "ceiling"
    assert display.loc["CARRY", "ibd_entry_status"] == "ACTIONABLE"
    assert display.loc["RECONF", "ibd_candidate_rule"] == "ma10_touch_confirm"
    assert display.loc["RECONF", "ibd_entry_status"] == "UNCONFIRMED"


def test_projection_classifies_entry_changes_and_summary_from_the_same_fields():
    complete = pd.DataFrame(
        [
            _row("BECAME", snapshot_date="2026-07-24", signal=True, status="UNCONFIRMED", valid=False, candidate=100, close=99, rule="pivot"),
            _row("LEFT", snapshot_date="2026-07-24", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=103, rule="pivot"),
            _row("OTHER", snapshot_date="2026-07-24", signal=True, status="BELOW_TRIGGER", valid=True, candidate=100, close=99, rule="pivot"),
            _row("SAME", snapshot_date="2026-07-24", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=103, rule="pivot"),
        ]
    )
    current = pd.DataFrame(
        [
            _row("BECAME", snapshot_date="2026-07-29", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=102, rule="pivot"),
            _row("LEFT", snapshot_date="2026-07-29", signal=True, status="EXTENDED", valid=True, candidate=100, close=108, rule="pivot"),
            _row("OTHER", snapshot_date="2026-07-29", signal=True, status="UNCONFIRMED", valid=False, candidate=100, close=101, rule="pivot"),
            _row("SAME", snapshot_date="2026-07-29", signal=False, close=104),
        ]
    )

    result = build_midweek_review(current, complete)
    review = result.current_review.set_index("code")

    assert review.loc["BECAME", "review_change_group"] == "BECAME_ACTIONABLE"
    assert review.loc["LEFT", "review_change_group"] == "LEFT_ACTIONABLE"
    assert review.loc["LEFT", "review_entry_change"] == "ACTIONABLE_TO_EXTENDED"
    assert review.loc["OTHER", "review_change_group"] == "OTHER_CHANGES"
    assert review.loc["SAME", "review_change_group"] == "UNCHANGED"
    assert review.loc["SAME", "review_entry_change"] == "STILL_ACTIONABLE"
    assert result.summary["BECAME_ACTIONABLE"] == 1
    assert result.summary["LEFT_ACTIONABLE"] == 1
    assert result.summary["OTHER_CHANGES"] == 1
    assert result.summary["UNCHANGED"] == 1


def test_carry_status_uses_the_rounded_funnel_distance_at_zero_and_five_percent_boundaries():
    complete = pd.DataFrame(
        [
            _row("JUST_OVER", snapshot_date="2026-07-24", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=102, rule="pivot"),
            _row("JUST_UNDER", snapshot_date="2026-07-24", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=102, rule="pivot"),
        ]
    )
    current = pd.DataFrame(
        [
            _row("JUST_OVER", snapshot_date="2026-07-29", signal=False, close=105.004),
            _row("JUST_UNDER", snapshot_date="2026-07-29", signal=False, close=99.996),
        ]
    )

    review = build_midweek_review(current, complete).current_review.set_index("code")

    assert review.loc["JUST_OVER", "review_current_vs_candidate_pct"] == 5.0
    assert review.loc["JUST_OVER", "review_effective_entry_status"] == "ACTIONABLE"
    assert review.loc["JUST_UNDER", "review_current_vs_candidate_pct"] == 0.0
    assert review.loc["JUST_UNDER", "review_effective_entry_status"] == "ACTIONABLE"


def test_projection_without_baseline_suppresses_comparison_facts():
    current = pd.DataFrame(
        [
            _row("CURRENT", snapshot_date="2026-07-29", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=102, rule="pivot"),
            _row("POOL", snapshot_date="2026-07-29", signal=False, close=80),
        ]
    )

    result = build_midweek_review(current, pd.DataFrame())
    review = result.current_review.set_index("code")

    assert result.baseline_available is False
    assert review.loc["CURRENT", "review_signal_origin"] == "NONE"
    assert review.loc["CURRENT", "review_change_group"] == "UNCHANGED"
    assert review.loc["CURRENT", "review_change_label"] == ""
    assert result.summary["NEW"] == 0
    assert result.summary["BECAME_ACTIONABLE"] == 0
    assert result.actionable_codes == ("CURRENT",)


@pytest.mark.parametrize(
    "invalid_baseline",
    [
        "semantic_mismatch",
        "missing_enrichment",
        "missing_status",
        "null_entry_valid",
    ],
)
def test_snapshot_builder_discards_a_schema_or_semantically_invalid_baseline(invalid_baseline):
    complete = pd.DataFrame(
        [
            _row(
                "OLD_CARRY",
                snapshot_date="2026-07-24",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=100,
                close=101,
                rule="pivot",
            )
        ]
    )
    if invalid_baseline == "semantic_mismatch":
        complete.loc[0, "ibd_entry_status"] = "UNCONFIRMED"
    elif invalid_baseline == "missing_enrichment":
        complete = complete.drop(columns=["ibd_entry_valid"])
    elif invalid_baseline == "missing_status":
        complete = complete.drop(columns=["ibd_entry_status"])
    else:
        complete["ibd_entry_valid"] = complete["ibd_entry_valid"].astype("object")
        complete.loc[0, "ibd_entry_valid"] = None
        complete.loc[0, "ibd_entry_status"] = "UNCONFIRMED"
    current = pd.DataFrame(
        [
            _row(
                "CURRENT",
                snapshot_date="2026-07-29",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=50,
                close=51,
                rule="pivot",
            ),
            _row("OLD_CARRY", snapshot_date="2026-07-29", signal=False, close=101),
        ]
    )

    result = build_midweek_review_for_snapshots(current, complete)
    review = result.current_review.set_index("code")

    assert result.baseline_available is False
    assert result.actionable_codes == ("CURRENT",)
    assert review.loc["OLD_CARRY", "review_signal_origin"] == "NONE"


def test_thursday_complete_snapshot_carries_into_next_week_midweek():
    complete = pd.DataFrame(
        [
            _row(
                "HOLIDAY_CARRY",
                snapshot_date="2026-07-02",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=100,
                close=103,
                rule="ceiling",
            )
        ]
    )
    current = pd.DataFrame(
        [_row("HOLIDAY_CARRY", snapshot_date="2026-07-08", signal=False, close=104)]
    )

    result = build_midweek_review_for_snapshots(current, complete)
    row = result.current_review.set_index("code").loc["HOLIDAY_CARRY"]

    assert result.baseline_available is True
    assert result.actionable_codes == ("HOLIDAY_CARRY",)
    assert row["review_signal_origin"] == "CARRY"
    assert row["review_effective_entry_status"] == "ACTIONABLE"


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("latest_close", "bad", "latest_close"),
        ("latest_close", 0, "latest_close"),
        ("ibd_candidate_price", "bad", "candidate"),
        ("ibd_candidate_price", 0, "candidate"),
    ],
)
def test_projection_rejects_invalid_prices_for_watched_signals(column, value, message):
    complete = pd.DataFrame(
        [_row("BAD", snapshot_date="2026-07-24", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=102, rule="pivot")]
    )
    current = pd.DataFrame(
        [_row("BAD", snapshot_date="2026-07-29", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=102, rule="pivot")]
    )
    current[column] = current[column].astype("object")
    current.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        build_midweek_review(current, complete)


def test_projection_rejects_duplicate_codes():
    current = pd.DataFrame(
        [
            _row("DUP", snapshot_date="2026-07-29", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=102, rule="pivot"),
            _row("DUP", snapshot_date="2026-07-29", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=102, rule="pivot"),
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        build_midweek_review(current, pd.DataFrame())


def test_analyze_uses_valid_midweek_in_window_and_preserves_source_files(tmp_path):
    complete_path = tmp_path / "breakout_follow_pool.csv"
    midweek_path = tmp_path / "breakout_follow_pool_midweek.csv"
    pd.DataFrame(
        [_row("CARRY", snapshot_date="2026-07-24", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=104, rule="ceiling")]
    ).to_csv(complete_path, index=False)
    pd.DataFrame(
        [_row("CARRY", snapshot_date="2026-07-29", signal=False, close=105)]
    ).to_csv(midweek_path, index=False)
    complete_before = complete_path.read_bytes()
    midweek_before = midweek_path.read_bytes()

    result = analyze_breakout_follow_pool(
        complete_path,
        midweek_path,
        window_date=date(2026, 7, 30),
    )

    assert result.mode is PoolMode.MIDWEEK
    assert result.midweek_available is True
    assert result.midweek_baseline_available is True
    assert result.complete_snapshot_date == date(2026, 7, 24)
    assert result.midweek_snapshot_date == date(2026, 7, 29)
    assert result.review_week_start == date(2026, 7, 27)
    assert result.actionable_codes == ("CARRY",)
    assert complete_path.read_bytes() == complete_before
    assert midweek_path.read_bytes() == midweek_before


def test_analyze_uses_thursday_holiday_complete_as_valid_baseline(tmp_path):
    complete_path = tmp_path / "breakout_follow_pool.csv"
    midweek_path = tmp_path / "breakout_follow_pool_midweek.csv"
    pd.DataFrame(
        [_row("CARRY", snapshot_date="2026-07-02", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=103, rule="ceiling")]
    ).to_csv(complete_path, index=False)
    pd.DataFrame(
        [_row("CARRY", snapshot_date="2026-07-08", signal=False, close=104)]
    ).to_csv(midweek_path, index=False)

    result = analyze_breakout_follow_pool(
        complete_path,
        midweek_path,
        window_date=date(2026, 7, 9),
    )

    assert result.mode is PoolMode.MIDWEEK
    assert result.midweek_baseline_available is True
    assert result.complete_snapshot_date == date(2026, 7, 2)
    assert result.review_week_start == date(2026, 7, 6)
    assert result.actionable_codes == ("CARRY",)


def test_analyze_rejects_stale_ordinary_thursday_complete_baseline(tmp_path):
    complete_path = tmp_path / "breakout_follow_pool.csv"
    midweek_path = tmp_path / "breakout_follow_pool_midweek.csv"
    pd.DataFrame(
        [_row("STALE", snapshot_date="2026-07-30", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=103, rule="ceiling")]
    ).to_csv(complete_path, index=False)
    pd.DataFrame(
        [_row("STALE", snapshot_date="2026-08-04", signal=False, close=104)]
    ).to_csv(midweek_path, index=False)

    result = analyze_breakout_follow_pool(
        complete_path,
        midweek_path,
        window_date=date(2026, 8, 5),
    )

    assert result.mode is PoolMode.MIDWEEK_WITHOUT_VALID_BASELINE
    assert result.midweek_baseline_available is False
    assert result.actionable_codes == ()


def test_analyze_ignores_stale_midweek_without_deleting_it(tmp_path):
    complete_path = tmp_path / "breakout_follow_pool.csv"
    midweek_path = tmp_path / "breakout_follow_pool_midweek.csv"
    pd.DataFrame(
        [_row("CURRENT", snapshot_date="2026-07-24", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=102, rule="pivot")]
    ).to_csv(complete_path, index=False)
    pd.DataFrame(
        [_row("STALE", snapshot_date="2026-07-22", signal=True, status="ACTIONABLE", valid=True, candidate=50, close=51, rule="pivot")]
    ).to_csv(midweek_path, index=False)

    result = analyze_breakout_follow_pool(
        complete_path,
        midweek_path,
        window_date=date(2026, 7, 28),
    )

    assert result.mode is PoolMode.COMPLETE
    assert result.midweek_available is False
    assert result.midweek_baseline_available is False
    assert result.actionable_codes == ("CURRENT",)
    assert midweek_path.exists()
    assert any(
        "Midweek snapshot is unavailable for the current complete-week baseline."
        in warning
        for warning in result.warnings
    )
    assert not any("stale" in warning.lower() for warning in result.warnings)


def test_analyze_defaults_to_complete_outside_midweek_but_keeps_valid_review_available(tmp_path):
    complete_path = tmp_path / "breakout_follow_pool.csv"
    midweek_path = tmp_path / "breakout_follow_pool_midweek.csv"
    pd.DataFrame(
        [_row("BASE", snapshot_date="2026-07-24", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=102, rule="pivot")]
    ).to_csv(complete_path, index=False)
    pd.DataFrame(
        [_row("MID", snapshot_date="2026-07-29", signal=True, status="ACTIONABLE", valid=True, candidate=50, close=51, rule="pivot")]
    ).to_csv(midweek_path, index=False)

    result = analyze_breakout_follow_pool(
        complete_path,
        midweek_path,
        window_date=date(2026, 8, 1),
    )

    assert result.mode is PoolMode.COMPLETE
    assert result.midweek_available is True
    assert result.midweek_baseline_available is True
    assert result.actionable_codes == ("BASE",)


def test_complete_window_keeps_valid_midweek_baseline_available_for_manual_review(tmp_path):
    complete_path = tmp_path / "breakout_follow_pool.csv"
    midweek_path = tmp_path / "breakout_follow_pool_midweek.csv"
    pd.DataFrame(
        [
            _row(
                "BASE",
                snapshot_date="2026-07-24",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=100,
                close=102,
                rule="pivot",
            )
        ]
    ).to_csv(complete_path, index=False)
    pd.DataFrame(
        [
            _row(
                "MID",
                snapshot_date="2026-07-30",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=50,
                close=51,
                rule="pivot",
            )
        ]
    ).to_csv(midweek_path, index=False)

    result = analyze_breakout_follow_pool(
        complete_path,
        midweek_path,
        window_date=date(2026, 8, 3),
    )

    assert result.mode is PoolMode.COMPLETE
    assert result.midweek_available is True
    assert result.midweek_baseline_available is True


def test_complete_window_midweek_counts_are_derived_from_snapshot_rows(tmp_path):
    complete_path = tmp_path / "breakout_follow_pool.csv"
    midweek_path = tmp_path / "breakout_follow_pool_midweek.csv"
    pd.DataFrame(
        [
            _row("CARRY", snapshot_date="2026-07-24", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=103, rule="pivot", rank=1),
            _row("LEFT", snapshot_date="2026-07-24", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=103, rule="pivot", rank=2),
        ]
    ).to_csv(complete_path, index=False)
    pd.DataFrame(
        [
            _row("NEW", snapshot_date="2026-07-29", signal=True, status="ACTIONABLE", valid=True, candidate=50, close=51, rule="pivot", rank=1),
            _row("CARRY", snapshot_date="2026-07-29", signal=False, close=104, rank=2),
            _row("LEFT", snapshot_date="2026-07-29", signal=True, status="EXTENDED", valid=True, candidate=100, close=108, rule="pivot", rank=3),
        ]
    ).to_csv(midweek_path, index=False)

    result = analyze_breakout_follow_pool(
        complete_path,
        midweek_path,
        window_date=date(2026, 7, 30),
    )
    state = switch_review_mode(
        default_review_state(result.mode),
        "MIDWEEK",
        midweek_has_baseline=result.midweek_baseline_available,
    )
    counts = build_review_filter_counts(
        materialize_review_view(result.midweek_review),
        state,
    )

    assert result.mode is PoolMode.MIDWEEK
    assert result.complete_snapshot_date == date(2026, 7, 24)
    assert result.midweek_snapshot_date == date(2026, 7, 29)
    assert result.summary["ACTIVE_SIGNALS"] == 3
    assert counts["change"] == {
        "BECAME_ACTIONABLE": 1,
        "LEFT_ACTIONABLE": 1,
        "OTHER_CHANGES": 0,
        "UNCHANGED": 0,
    }
    assert counts["origin"] == {"NEW": 1, "CARRY": 0, "RECONFIRMED": 1}
    assert counts["status"] == {
        "ACTIONABLE": 1,
        "UNCONFIRMED": 0,
        "BELOW_TRIGGER": 0,
        "EXTENDED": 1,
    }
    assert counts["result"] == 2


def test_analyze_without_valid_baseline_never_carries_complete_signal(tmp_path):
    complete_path = tmp_path / "breakout_follow_pool.csv"
    midweek_path = tmp_path / "breakout_follow_pool_midweek.csv"
    pd.DataFrame(
        [_row("OLD", snapshot_date="2026-07-17", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=102, rule="pivot")]
    ).to_csv(complete_path, index=False)
    pd.DataFrame(
        [
            _row("CURRENT_SIGNAL", snapshot_date="2026-07-29", signal=True, status="ACTIONABLE", valid=True, candidate=50, close=51, rule="pivot"),
            _row("NO_SIGNAL", snapshot_date="2026-07-29", signal=False, close=80),
        ]
    ).to_csv(midweek_path, index=False)

    result = analyze_breakout_follow_pool(
        complete_path,
        midweek_path,
        window_date=date(2026, 7, 30),
    )

    assert result.mode is PoolMode.MIDWEEK_WITHOUT_VALID_BASELINE
    assert result.midweek_baseline_available is False
    assert result.actionable_codes == ("CURRENT_SIGNAL",)
    assert set(result.midweek_review.loc[result.midweek_review["review_watch_active"], "code"]) == {"CURRENT_SIGNAL"}
    assert result.summary["NEW"] == 0
    assert result.summary["BECAME_ACTIONABLE"] == 0
    assert result.midweek_review["review_signal_origin"].eq("NONE").all()


def test_analyze_warns_when_midweek_snapshot_is_missing_during_midweek_window(tmp_path):
    complete_path = tmp_path / "breakout_follow_pool.csv"
    pd.DataFrame(
        [_row("BASE", snapshot_date="2026-07-24", signal=True, status="ACTIONABLE", valid=True, candidate=100, close=102, rule="pivot")]
    ).to_csv(complete_path, index=False)

    result = analyze_breakout_follow_pool(
        complete_path,
        tmp_path / "missing_midweek.csv",
        window_date=date(2026, 7, 30),
    )

    assert result.mode is PoolMode.COMPLETE
    assert any("Midweek snapshot is unavailable" in warning for warning in result.warnings)


@pytest.mark.parametrize("window_date", [date(2026, 7, 30), date(2026, 8, 1)])
def test_analyze_fails_closed_to_complete_when_midweek_projection_is_invalid(
    tmp_path,
    window_date,
):
    complete_path = tmp_path / "breakout_follow_pool.csv"
    midweek_path = tmp_path / "breakout_follow_pool_midweek.csv"
    pd.DataFrame(
        [
            _row(
                "BASE",
                snapshot_date="2026-07-24",
                signal=True,
                status="ACTIONABLE",
                valid=True,
                candidate=100,
                close=102,
                rule="pivot",
            )
        ]
    ).to_csv(complete_path, index=False)
    invalid_midweek = pd.DataFrame(
        [
            _row(
                "BASE",
                snapshot_date="2026-07-29",
                signal=False,
                close=0,
            )
        ]
    )
    invalid_midweek.to_csv(midweek_path, index=False)

    result = analyze_breakout_follow_pool(
        complete_path,
        midweek_path,
        window_date=window_date,
    )

    assert result.mode is PoolMode.COMPLETE
    assert result.midweek_available is False
    assert result.actionable_codes == ("BASE",)
    assert any("projection failed closed" in warning for warning in result.warnings)
