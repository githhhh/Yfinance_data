from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
import math
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from dashboard.data_utils import (
    load_pool_csv,
    validate_pool_schema,
    validate_pool_semantics,
)
from dashboard.services.bf_transition import (
    ENTRY_STATUSES,
    SOURCE_FACT_FIELDS,
    analyze_bf_transitions,
    normalize_transition_pool as _normalized_pool,
    to_bool as _to_bool,
)


BUSINESS_TIMEZONE = "Asia/Shanghai"
SETUP_FILTER_OPTIONS = (
    "All",
    "ceiling",
    "ceiling_pullback",
    "ma10_touch_confirm",
    "pivot",
    "three_weeks_tight",
)


class PoolWindow(Enum):
    MIDWEEK = "midweek"
    COMPLETE = "complete"


class PoolMode(Enum):
    COMPLETE = "complete"
    MIDWEEK = "midweek"
    MIDWEEK_WITHOUT_VALID_BASELINE = "midweek_without_valid_baseline"


@dataclass(frozen=True)
class MidweekReviewResult:
    current_review: pd.DataFrame
    exited_pool: pd.DataFrame
    summary: dict[str, int]
    actionable_codes: tuple[str, ...]
    baseline_available: bool


@dataclass(frozen=True)
class PoolAnalysisResult:
    mode: PoolMode
    window: PoolWindow
    complete_snapshot_date: date | None
    midweek_snapshot_date: date | None
    review_week_start: date | None
    complete_pool: pd.DataFrame
    midweek_pool: pd.DataFrame
    midweek_review: pd.DataFrame
    exited_pool: pd.DataFrame
    summary: dict[str, int]
    actionable_codes: tuple[str, ...]
    warnings: tuple[str, ...]
    midweek_available: bool
    midweek_baseline_available: bool = False


def resolve_window(window_date: date) -> PoolWindow:
    if window_date.weekday() in {1, 2, 3, 4}:
        return PoolWindow.MIDWEEK
    return PoolWindow.COMPLETE


def monday_of_week(value: date) -> date:
    return value - timedelta(days=value.weekday())


def complete_target_week(complete_date: date) -> date:
    weekday = complete_date.weekday()
    if weekday == 0:
        return complete_date
    if weekday in {4, 5, 6}:
        return complete_date + timedelta(days=7 - weekday)
    raise ValueError(f"Invalid complete snapshot date: {complete_date.isoformat()}")


def build_midweek_review(
    current_pool: pd.DataFrame,
    complete_pool: pd.DataFrame,
) -> MidweekReviewResult:
    """Compatibility adapter over the shared BF transition API.

    The Dashboard contract intentionally remains unchanged; transition
    semantics now live in bf_transition so external consumers use the same
    facts without re-implementing them.
    """
    transition = analyze_bf_transitions(current_pool, complete_pool)
    return MidweekReviewResult(
        current_review=transition.rows,
        exited_pool=transition.exited_pool,
        summary=transition.review_summary,
        actionable_codes=transition.actionable_codes,
        baseline_available=transition.baseline_available,
    )


def materialize_review_view(review: pd.DataFrame) -> pd.DataFrame:
    result = review.copy()
    if result.empty:
        return result
    result["signal"] = pd.Series(result["review_watch_active"].tolist(), index=result.index, dtype="object")
    for field in SOURCE_FACT_FIELDS:
        review_field = f"review_{field}"
        if review_field in result.columns:
            result[field] = result[review_field]
    result["ibd_candidate_price"] = result["review_candidate_price"]
    result["ibd_entry_valid"] = pd.Series(result["review_entry_valid"].tolist(), index=result.index, dtype="object")
    result["ibd_entry_status"] = result["review_effective_entry_status"]
    result["current_vs_ibd_candidate_pct"] = result["review_current_vs_candidate_pct"]

    def vol_or_reason(row: pd.Series) -> str:
        if row.get("review_entry_valid") is not True:
            reason = row.get("ibd_entry_reject_reason")
            return str(reason).strip() if reason is not None and not pd.isna(reason) and str(reason).strip() else "Volume not confirmed"
        volume = pd.to_numeric(row.get("ibd_entry_volume_ratio"), errors="coerce")
        return "n/a" if pd.isna(volume) else f"{float(volume):.2f}x"

    result["ibd_entry_vol_or_reject"] = result.apply(vol_or_reason, axis=1)
    return result


def default_review_state(mode: PoolMode) -> dict[str, Any]:
    mode_value = getattr(mode, "value", mode)
    is_midweek = mode_value in {
        PoolMode.MIDWEEK.value,
        PoolMode.MIDWEEK_WITHOUT_VALID_BASELINE.value,
    }
    has_comparison = mode_value == PoolMode.MIDWEEK.value
    return {
        "mode": "MIDWEEK" if is_midweek else "WEEKEND",
        "scope": "CHANGES" if has_comparison else "ALL_SIGNALS",
        "change_filter": "ALL",
        "origin_filter": "ALL",
        "status_filter": "ALL",
        "route_filter": "All",
        "distance_range": None,
        "entry_volume_min": None,
        "weekly_volume_min": None,
        "filters_expanded": False,
        "copy_state": "IDLE",
        "sort_mode": "Review Priority" if has_comparison else "C Rank",
        "widget_generation": 0,
    }


def normalize_review_state(state: dict[str, Any]) -> dict[str, Any]:
    """Migrate persisted text-input state to the slider-based filter model."""
    result = dict(state)
    distance = result.get("distance_range")
    if isinstance(distance, (tuple, list)) and len(distance) == 2:
        try:
            lower, upper = float(distance[0]), float(distance[1])
            distance = (lower, upper) if math.isfinite(lower) and math.isfinite(upper) and lower <= upper else None
        except (TypeError, ValueError):
            distance = None
    else:
        distance = None
    result["distance_range"] = distance

    for field in ("entry_volume_min", "weekly_volume_min"):
        raw_value = result.get(field)
        try:
            value = None if raw_value is None or str(raw_value).strip() == "" else float(raw_value)
        except (TypeError, ValueError):
            value = None
        result[field] = value if value is not None and math.isfinite(value) else None

    if result.get("route_filter", "All") not in SETUP_FILTER_OPTIONS:
        result["route_filter"] = "All"
    for legacy_field in ("distance_min", "distance_max", "near_trigger_only"):
        result.pop(legacy_field, None)
    return result


def default_sort_mode(
    mode: str,
    scope: str,
    *,
    has_comparison: bool,
) -> str:
    """Return the fixed public default order for a review view."""
    if (
        str(mode).upper() == "MIDWEEK"
        and str(scope).upper() == "CHANGES"
        and has_comparison
    ):
        return "Review Priority"
    return "C Rank"


def reconcile_review_state(state: dict[str, Any], mode: PoolMode) -> dict[str, Any]:
    """Remove comparison-only state when a midweek baseline is unavailable."""
    result = dict(state)
    mode_value = getattr(mode, "value", mode)
    if (
        mode_value != PoolMode.MIDWEEK_WITHOUT_VALID_BASELINE.value
        or str(result.get("mode", "")).upper() != "MIDWEEK"
    ):
        return result
    result["scope"] = "ALL_SIGNALS"
    result["change_filter"] = "ALL"
    result["origin_filter"] = "ALL"
    if result.get("sort_mode") == "Review Priority":
        result["sort_mode"] = "C Rank"
        result["widget_generation"] = int(result.get("widget_generation", 0)) + 1
    return result


def switch_review_mode(
    state: dict[str, Any],
    target_mode: str,
    *,
    midweek_has_baseline: bool = True,
) -> dict[str, Any]:
    target = target_mode.upper()
    if target == str(state.get("mode", "")).upper():
        return dict(state)
    mode = (
        PoolMode.MIDWEEK if midweek_has_baseline else PoolMode.MIDWEEK_WITHOUT_VALID_BASELINE
    ) if target == "MIDWEEK" else PoolMode.COMPLETE
    result = default_review_state(mode)
    result["widget_generation"] = int(state.get("widget_generation", 0)) + 1
    return result


def clear_quick_filters(state: dict[str, Any]) -> dict[str, Any]:
    result = dict(state)
    result["change_filter"] = "ALL"
    result["origin_filter"] = "ALL"
    return result


def select_status_filter(state: dict[str, Any], status: str) -> dict[str, Any]:
    result = dict(state)
    result["status_filter"] = status
    return result


def toggle_status_filter(state: dict[str, Any], status: str) -> dict[str, Any]:
    target = "ALL" if state.get("status_filter") == status else status
    return select_status_filter(state, target)


def reset_to_all_signals(state: dict[str, Any]) -> dict[str, Any]:
    result = dict(state)
    result.update(
        {
            "scope": "ALL_SIGNALS",
            "change_filter": "ALL",
            "origin_filter": "ALL",
            "status_filter": "ALL",
            "route_filter": "All",
            "distance_range": None,
            "entry_volume_min": None,
            "weekly_volume_min": None,
            "filters_expanded": False,
            "copy_state": "IDLE",
            "widget_generation": int(state.get("widget_generation", 0)) + 1,
        }
    )
    for legacy_field in ("distance_min", "distance_max", "near_trigger_only"):
        result.pop(legacy_field, None)
    return result


def _number_filter(
    frame: pd.DataFrame,
    field: str,
    raw_value: Any,
    *,
    minimum: bool,
) -> pd.DataFrame:
    if raw_value is None or str(raw_value).strip() == "" or field not in frame.columns:
        return frame
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return frame
    numbers = pd.to_numeric(frame[field], errors="coerce")
    return frame.loc[numbers.ge(value) if minimum else numbers.le(value)].copy()


def apply_review_filters(
    review: pd.DataFrame,
    state: dict[str, Any],
    *,
    exclude_dimension: str | None = None,
) -> pd.DataFrame:
    if review.empty:
        return review.copy()
    result = review.copy()
    active_field = "review_watch_active" if "review_watch_active" in result.columns else "signal"
    result = result.loc[result[active_field].map(_to_bool).eq(True)].copy()
    is_midweek = str(state.get("mode", "MIDWEEK")).upper() == "MIDWEEK"
    has_comparison = is_midweek and (
        "review_baseline_available" not in result.columns
        or result["review_baseline_available"].map(_to_bool).eq(True).all()
    )
    if has_comparison and state.get("scope") == "CHANGES" and "review_change_group" in result.columns:
        result = result.loc[result["review_change_group"].ne("UNCHANGED")].copy()

    if has_comparison and exclude_dimension != "change":
        selected_change = state.get("change_filter", "ALL")
        if selected_change != "ALL" and "review_change_group" in result.columns:
            result = result.loc[result["review_change_group"].eq(selected_change)].copy()
    if has_comparison and exclude_dimension != "origin":
        selected_origin = state.get("origin_filter", "ALL")
        if selected_origin != "ALL" and "review_signal_origin" in result.columns:
            result = result.loc[result["review_signal_origin"].eq(selected_origin)].copy()

    status_field = (
        "review_effective_entry_status"
        if "review_effective_entry_status" in result.columns
        else "ibd_entry_status"
    )
    if exclude_dimension != "status":
        selected_status = state.get("status_filter", "ALL")
        if selected_status != "ALL" and status_field in result.columns:
            result = result.loc[result[status_field].eq(selected_status)].copy()

    if exclude_dimension != "advanced":
        route = state.get("route_filter", "All")
        if route != "All" and "ibd_candidate_rule" in result.columns:
            result = result.loc[result["ibd_candidate_rule"].eq(route)].copy()
        distance_range = state.get("distance_range")
        if isinstance(distance_range, (tuple, list)) and len(distance_range) == 2:
            result = _number_filter(
                result,
                "current_vs_ibd_candidate_pct",
                distance_range[0],
                minimum=True,
            )
            result = _number_filter(
                result,
                "current_vs_ibd_candidate_pct",
                distance_range[1],
                minimum=False,
            )
        result = _number_filter(
            result,
            "ibd_entry_volume_ratio",
            state.get("entry_volume_min"),
            minimum=True,
        )
        result = _number_filter(
            result,
            "volume_ratio",
            state.get("weekly_volume_min"),
            minimum=True,
        )
    return result


def build_review_filter_counts(
    review: pd.DataFrame,
    state: dict[str, Any],
) -> dict[str, Any]:
    change_base = apply_review_filters(review, state, exclude_dimension="change")
    origin_base = apply_review_filters(review, state, exclude_dimension="origin")
    status_field = (
        "review_effective_entry_status"
        if "review_effective_entry_status" in review.columns
        else "ibd_entry_status"
    )
    status_counts: dict[str, int] = {}
    for value in ENTRY_STATUSES:
        target_state = select_status_filter(state, value)
        status_base = apply_review_filters(review, target_state, exclude_dimension="status")
        status_counts[value] = int(
            status_base.get(status_field, pd.Series(dtype=object)).eq(value).sum()
        )
    return {
        "change": {
            value: int(change_base.get("review_change_group", pd.Series(dtype=object)).eq(value).sum())
            for value in ("BECAME_ACTIONABLE", "LEFT_ACTIONABLE", "OTHER_CHANGES", "UNCHANGED")
        },
        "origin": {
            value: int(origin_base.get("review_signal_origin", pd.Series(dtype=object)).eq(value).sum())
            for value in ("NEW", "CARRY", "RECONFIRMED")
        },
        "status": status_counts,
        "result": len(apply_review_filters(review, state)),
    }


def sort_review_rows(review: pd.DataFrame, sort_mode: str) -> pd.DataFrame:
    if review.empty:
        return review.copy()
    result = review.copy()
    if sort_mode == "C Rank":
        by = [field for field in ("rank_C_continuous", "code") if field in result.columns]
        return result.sort_values(by=by, ascending=True, na_position="last", kind="mergesort").copy()
    if sort_mode == "Distance":
        by = [field for field in ("current_vs_ibd_candidate_pct", "code") if field in result.columns]
        return result.sort_values(by=by, ascending=True, na_position="last", kind="mergesort").copy()
    status_field = (
        "review_effective_entry_status"
        if "review_effective_entry_status" in result.columns
        else "ibd_entry_status"
    )
    result["_review_status_rank"] = result.get(status_field, pd.Series(index=result.index)).map(
        {status: rank for rank, status in enumerate(ENTRY_STATUSES)}
    )
    by = [field for field in ("review_priority", "_review_status_rank", "code") if field in result.columns]
    return result.sort_values(by=by, ascending=True, na_position="last", kind="mergesort").drop(
        columns=["_review_status_rank"], errors="ignore"
    )


def _snapshot_date(pool: pd.DataFrame, *, label: str) -> date:
    if "snapshot_date" not in pool.columns:
        raise ValueError(f"{label} pool missing snapshot_date")
    parsed = pd.to_datetime(pool["snapshot_date"], errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{label} pool snapshot_date must be parseable")
    dates = {value.date() for value in parsed}
    if len(dates) != 1:
        raise ValueError(f"{label} pool snapshot_date must contain exactly one date")
    return next(iter(dates))


def _has_valid_complete_baseline(complete_date: date, midweek_date: date) -> bool:
    try:
        return (
            monday_of_week(midweek_date) == complete_target_week(complete_date)
            and midweek_date > complete_date
        )
    except ValueError:
        return False


def build_midweek_review_for_snapshots(
    current_pool: pd.DataFrame,
    complete_pool: pd.DataFrame,
) -> MidweekReviewResult:
    """Use Carry only when both snapshots form a valid review-week pair."""
    current = _normalized_pool(current_pool, label="current")
    current_date = _snapshot_date(current, label="midweek")
    baseline = pd.DataFrame()
    if complete_pool is not None and not complete_pool.empty:
        try:
            validate_pool_schema(complete_pool)
            complete = _normalized_pool(complete_pool, label="complete")
            active_complete = complete.loc[complete["signal"].map(_to_bool).eq(True)]
            if active_complete["ibd_entry_valid"].map(_to_bool).isna().any():
                raise ValueError("complete pool IBD enrichment is incomplete")
            validate_pool_semantics(complete)
            complete_date = _snapshot_date(complete, label="complete")
            if _has_valid_complete_baseline(complete_date, current_date):
                baseline = complete
        except (TypeError, ValueError):
            baseline = pd.DataFrame()
    return build_midweek_review(current, baseline)


def _complete_actionable_codes(pool: pd.DataFrame) -> tuple[str, ...]:
    if pool.empty:
        return ()
    mask = pool["signal"].map(_to_bool).eq(True) & pool["ibd_entry_status"].eq("ACTIONABLE")
    return tuple(pool.loc[mask, "code"].astype(str).tolist())


def analyze_breakout_follow_pool(
    complete_pool_path: str | Path,
    midweek_pool_path: str | Path | None,
    *,
    window_date: date | None = None,
    business_timezone: str = BUSINESS_TIMEZONE,
) -> PoolAnalysisResult:
    business_date = window_date or datetime.now(ZoneInfo(business_timezone)).date()
    window = resolve_window(business_date)
    warnings: list[str] = []
    complete = pd.DataFrame()
    complete_date: date | None = None
    complete_week: date | None = None
    try:
        complete = _normalized_pool(load_pool_csv(complete_pool_path), label="complete")
        complete_date = _snapshot_date(complete, label="complete")
        complete_week = complete_target_week(complete_date)
    except Exception as exc:
        warnings.append(f"Complete pool is not a valid baseline: {exc}")

    midweek = pd.DataFrame()
    midweek_date: date | None = None
    midweek_error: Exception | None = None
    midweek_path_exists = midweek_pool_path is not None and Path(midweek_pool_path).exists()
    if midweek_path_exists:
        try:
            midweek = _normalized_pool(load_pool_csv(midweek_pool_path), label="midweek")
            midweek_date = _snapshot_date(midweek, label="midweek")
        except Exception as exc:
            midweek_error = exc
            warnings.append(f"Midweek pool is invalid: {exc}")

    empty_review = pd.DataFrame()
    empty_exited = pd.DataFrame()
    empty_summary: dict[str, int] = {}
    review_result: MidweekReviewResult | None = None
    midweek_available = False
    no_baseline = False
    review_week: date | None = complete_week

    if midweek_date is not None:
        midweek_week = monday_of_week(midweek_date)
        review_week = complete_week or midweek_week
        try:
            if complete_date is not None and _has_valid_complete_baseline(complete_date, midweek_date):
                review_result = build_midweek_review(midweek, complete)
                midweek_available = True
            elif complete_week is None or midweek_week > complete_week:
                review_result = build_midweek_review(midweek, pd.DataFrame())
                midweek_available = True
                no_baseline = True
                warnings.append("Midweek snapshot has no valid complete-week baseline; Carry is disabled.")
            else:
                warnings.append(
                    "Midweek snapshot is unavailable for the current complete-week baseline."
                )
        except Exception as exc:
            warnings.append(f"Midweek projection failed closed: {exc}")

    if window == PoolWindow.MIDWEEK and midweek_error is not None:
        warnings.append("Midweek review failed closed; the complete pool remains selected.")
    if window == PoolWindow.MIDWEEK and not midweek_path_exists:
        warnings.append("Midweek snapshot is unavailable; the complete pool remains selected.")

    if window == PoolWindow.MIDWEEK and review_result is not None:
        mode = PoolMode.MIDWEEK_WITHOUT_VALID_BASELINE if no_baseline else PoolMode.MIDWEEK
        actionable = review_result.actionable_codes
    else:
        mode = PoolMode.COMPLETE
        if complete.empty:
            if review_result is not None and window == PoolWindow.MIDWEEK:
                mode = PoolMode.MIDWEEK_WITHOUT_VALID_BASELINE
                actionable = review_result.actionable_codes
            else:
                raise ValueError("Complete pool is unavailable and no usable midweek review exists")
        else:
            actionable = _complete_actionable_codes(complete)

    return PoolAnalysisResult(
        mode=mode,
        window=window,
        complete_snapshot_date=complete_date,
        midweek_snapshot_date=midweek_date,
        review_week_start=review_week,
        complete_pool=complete,
        midweek_pool=midweek,
        midweek_review=review_result.current_review if review_result else empty_review,
        exited_pool=review_result.exited_pool if review_result else empty_exited,
        summary=review_result.summary if review_result else empty_summary,
        actionable_codes=actionable,
        warnings=tuple(warnings),
        midweek_available=midweek_available,
        midweek_baseline_available=bool(
            review_result is not None and review_result.baseline_available
        ),
    )
