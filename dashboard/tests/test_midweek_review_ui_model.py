from __future__ import annotations

import pandas as pd

from dashboard.services.bf_midweek_review import (
    PoolMode,
    apply_review_filters,
    build_review_filter_counts,
    clear_quick_filters,
    default_review_state,
    reconcile_review_state,
    reset_to_all_signals,
    sort_review_rows,
    switch_review_mode,
)


def _review_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "code": "A",
                "review_watch_active": True,
                "review_change_group": "BECAME_ACTIONABLE",
                "review_signal_origin": "NEW",
                "review_effective_entry_status": "ACTIONABLE",
                "review_priority": 0,
                "ibd_candidate_rule": "pivot",
                "current_vs_ibd_candidate_pct": 1.0,
                "ibd_entry_volume_ratio": 2.0,
                "volume_ratio": 1.5,
                "rank_C_continuous": 3,
            },
            {
                "code": "B",
                "review_watch_active": True,
                "review_change_group": "BECAME_ACTIONABLE",
                "review_signal_origin": "CARRY",
                "review_effective_entry_status": "ACTIONABLE",
                "review_priority": 1,
                "ibd_candidate_rule": "ceiling",
                "current_vs_ibd_candidate_pct": 2.0,
                "ibd_entry_volume_ratio": 1.8,
                "volume_ratio": 1.2,
                "rank_C_continuous": 1,
            },
            {
                "code": "C",
                "review_watch_active": True,
                "review_change_group": "LEFT_ACTIONABLE",
                "review_signal_origin": "CARRY",
                "review_effective_entry_status": "EXTENDED",
                "review_priority": 12,
                "ibd_candidate_rule": "ceiling",
                "current_vs_ibd_candidate_pct": 8.0,
                "ibd_entry_volume_ratio": 2.4,
                "volume_ratio": 1.7,
                "rank_C_continuous": 2,
            },
            {
                "code": "D",
                "review_watch_active": True,
                "review_change_group": "OTHER_CHANGES",
                "review_signal_origin": "RECONFIRMED",
                "review_effective_entry_status": "UNCONFIRMED",
                "review_priority": 21,
                "ibd_candidate_rule": "pivot",
                "current_vs_ibd_candidate_pct": 0.5,
                "ibd_entry_volume_ratio": None,
                "volume_ratio": 2.0,
                "rank_C_continuous": 4,
            },
            {
                "code": "E",
                "review_watch_active": True,
                "review_change_group": "UNCHANGED",
                "review_signal_origin": "CARRY",
                "review_effective_entry_status": "ACTIONABLE",
                "review_priority": 30,
                "ibd_candidate_rule": "pivot",
                "current_vs_ibd_candidate_pct": 4.0,
                "ibd_entry_volume_ratio": 2.2,
                "volume_ratio": 1.8,
                "rank_C_continuous": 0,
            },
            {
                "code": "POOL_ONLY",
                "review_watch_active": False,
                "review_change_group": "UNCHANGED",
                "review_signal_origin": "NONE",
                "review_effective_entry_status": None,
                "review_priority": 99,
                "ibd_candidate_rule": None,
                "current_vs_ibd_candidate_pct": None,
                "ibd_entry_volume_ratio": None,
                "volume_ratio": 1.0,
                "rank_C_continuous": 99,
            },
        ]
    )


def test_change_origin_and_status_filters_combine_with_and_semantics():
    state = default_review_state(PoolMode.MIDWEEK)
    state.update(
        {
            "change_filter": "BECAME_ACTIONABLE",
            "origin_filter": "CARRY",
            "status_filter": "ACTIONABLE",
        }
    )

    filtered = apply_review_filters(_review_rows(), state)

    assert filtered["code"].tolist() == ["B"]


def test_no_baseline_rows_ignore_hidden_comparison_filters():
    rows = _review_rows().copy()
    rows["review_baseline_available"] = False
    rows["review_signal_origin"] = "NONE"
    rows["review_change_group"] = "UNCHANGED"
    state = default_review_state(PoolMode.MIDWEEK_WITHOUT_VALID_BASELINE)
    state.update(
        {
            "change_filter": "BECAME_ACTIONABLE",
            "origin_filter": "CARRY",
        }
    )

    filtered = apply_review_filters(rows, state)

    assert filtered["code"].tolist() == ["A", "B", "C", "D", "E"]


def test_no_baseline_state_clears_only_incompatible_comparison_state():
    state = default_review_state(PoolMode.MIDWEEK)
    state.update(
        {
            "scope": "ALL_SIGNALS",
            "change_filter": "BECAME_ACTIONABLE",
            "origin_filter": "CARRY",
            "status_filter": "ACTIONABLE",
            "route_filter": "pivot",
        }
    )

    reconciled = reconcile_review_state(
        state,
        PoolMode.MIDWEEK_WITHOUT_VALID_BASELINE,
    )

    assert reconciled["scope"] == "ALL_SIGNALS"
    assert reconciled["change_filter"] == "ALL"
    assert reconciled["origin_filter"] == "ALL"
    assert reconciled["sort_mode"] == "C Rank"
    assert reconciled["status_filter"] == "ACTIONABLE"
    assert reconciled["route_filter"] == "pivot"


def test_filter_counts_use_the_same_composed_filter_model_as_results():
    state = default_review_state(PoolMode.MIDWEEK)
    state.update({"origin_filter": "CARRY", "status_filter": "ACTIONABLE"})

    counts = build_review_filter_counts(_review_rows(), state)

    assert counts["change"]["BECAME_ACTIONABLE"] == 1
    assert counts["change"]["UNCHANGED"] == 0  # Changes scope excludes unchanged rows.
    assert counts["origin"]["CARRY"] == 1
    assert counts["status"]["ACTIONABLE"] == 1
    assert counts["result"] == len(apply_review_filters(_review_rows(), state)) == 1


def test_status_facet_counts_apply_the_target_status_cleanup_rules():
    rows = _review_rows()
    state = default_review_state(PoolMode.MIDWEEK)
    state.update(
        {
            "scope": "ALL_SIGNALS",
            "status_filter": "ACTIONABLE",
            "entry_volume_min": "1.5",
        }
    )

    counts = build_review_filter_counts(rows, state)

    assert counts["status"]["UNCONFIRMED"] == 1

    state.update(
        {
            "status_filter": "UNCONFIRMED",
            "entry_volume_min": "",
            "near_trigger_only": True,
        }
    )
    counts = build_review_filter_counts(rows, state)
    assert counts["status"]["ACTIONABLE"] == 3


def test_clear_only_resets_change_and_origin_without_touching_status_or_filters():
    state = default_review_state(PoolMode.MIDWEEK)
    state.update(
        {
            "change_filter": "LEFT_ACTIONABLE",
            "origin_filter": "CARRY",
            "status_filter": "EXTENDED",
            "route_filter": "ceiling",
            "distance_min": "1",
        }
    )

    cleared = clear_quick_filters(state)

    assert cleared["change_filter"] == "ALL"
    assert cleared["origin_filter"] == "ALL"
    assert cleared["status_filter"] == "EXTENDED"
    assert cleared["route_filter"] == "ceiling"
    assert cleared["distance_min"] == "1"


def test_all_signals_performs_explicit_scope_and_filter_reset():
    state = default_review_state(PoolMode.MIDWEEK)
    state.update(
        {
            "change_filter": "LEFT_ACTIONABLE",
            "origin_filter": "CARRY",
            "status_filter": "EXTENDED",
            "route_filter": "ceiling",
            "distance_min": "1",
            "distance_max": "9",
            "entry_volume_min": "2",
            "weekly_volume_min": "1.5",
            "near_trigger_only": True,
            "filters_expanded": True,
        }
    )

    reset = reset_to_all_signals(state)

    assert reset["scope"] == "ALL_SIGNALS"
    assert reset["change_filter"] == "ALL"
    assert reset["origin_filter"] == "ALL"
    assert reset["status_filter"] == "ALL"
    assert reset["route_filter"] == "All"
    assert reset["distance_min"] == ""
    assert reset["distance_max"] == ""
    assert reset["entry_volume_min"] == ""
    assert reset["weekly_volume_min"] == ""
    assert reset["near_trigger_only"] is False
    assert reset["filters_expanded"] is False


def test_default_modes_choose_compatible_scope_sort_and_collapsed_filters():
    midweek = default_review_state(PoolMode.MIDWEEK)
    no_baseline = default_review_state(PoolMode.MIDWEEK_WITHOUT_VALID_BASELINE)
    weekend = default_review_state(PoolMode.COMPLETE)

    assert (midweek["mode"], midweek["scope"], midweek["sort_mode"]) == (
        "MIDWEEK",
        "CHANGES",
        "Review Priority",
    )
    assert (weekend["mode"], weekend["scope"], weekend["sort_mode"]) == (
        "WEEKEND",
        "ALL_SIGNALS",
        "C Rank",
    )
    assert (no_baseline["mode"], no_baseline["scope"], no_baseline["sort_mode"]) == (
        "MIDWEEK",
        "ALL_SIGNALS",
        "C Rank",
    )
    assert midweek["filters_expanded"] is False
    assert weekend["filters_expanded"] is False


def test_repeated_mode_click_preserves_state_but_real_switch_resets_incompatible_state():
    state = default_review_state(PoolMode.MIDWEEK)
    state.update(
        {
            "origin_filter": "CARRY",
            "status_filter": "ACTIONABLE",
            "filters_expanded": True,
            "copy_state": "COPIED",
            "sort_mode": "C Rank",
        }
    )

    unchanged = switch_review_mode(state, "MIDWEEK")
    switched = switch_review_mode(state, "WEEKEND")

    assert unchanged == state
    assert switched["mode"] == "WEEKEND"
    assert switched["scope"] == "ALL_SIGNALS"
    assert switched["origin_filter"] == "ALL"
    assert switched["status_filter"] == "ALL"
    assert switched["filters_expanded"] is False
    assert switched["copy_state"] == "IDLE"
    assert switched["sort_mode"] == "C Rank"

    no_baseline = switch_review_mode(
        switched,
        "MIDWEEK",
        midweek_has_baseline=False,
    )
    assert (no_baseline["mode"], no_baseline["scope"], no_baseline["sort_mode"]) == (
        "MIDWEEK",
        "ALL_SIGNALS",
        "C Rank",
    )


def test_sort_control_label_matches_actual_review_and_weekend_order():
    rows = _review_rows().query("review_watch_active == True").copy()

    review_sorted = sort_review_rows(rows, "Review Priority")
    c_rank_sorted = sort_review_rows(rows, "C Rank")

    assert review_sorted["code"].tolist() == ["A", "B", "C", "D", "E"]
    assert c_rank_sorted["code"].tolist() == ["E", "B", "C", "A", "D"]
