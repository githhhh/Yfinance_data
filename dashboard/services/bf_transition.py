from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from dashboard.data_utils import normalize_pool_df


ENTRY_STATUSES = ("ACTIONABLE", "UNCONFIRMED", "BELOW_TRIGGER", "EXTENDED")

# Resolver-owned facts. Dashboard review projection and the external transition API
# must use the same atomic ownership rule for these fields.
SOURCE_FACT_FIELDS = (
    "signal_source",
    "ibd_candidate_rule",
    "ibd_candidate_price",
    "ibd_candidate_signal_source",
    "ibd_candidate_extra",
    "ibd_entry_valid",
    "ibd_entry_date",
    "ibd_entry_price",
    "ibd_trigger_price",
    "ibd_entry_volume_ratio",
    "ibd_entry_close_vs_trigger_pct",
    "ibd_entry_close_position",
    "ibd_entry_breakout_range_ratio",
    "ibd_entry_rule",
    "ibd_entry_reject_reason",
)

ATTENTION_HIGH = "HIGH"
ATTENTION_MEDIUM = "MEDIUM"


@dataclass(frozen=True)
class BFAttentionEvent:
    """A compact, serialization-safe fact bundle for downstream notification logic.

    This is deliberately not notification copy. The data repository owns the
    transition facts and coarse importance; a caller such as market_analysis may
    decide which events to mention and how to phrase them.
    """

    code: str
    event_type: str
    importance: str
    change_group: str
    signal_origin: str
    baseline_status: str | None
    previous_status: str | None
    current_status: str | None
    is_new_since_previous: bool | None
    reasons: tuple[str, ...]
    snapshot_date: str | None
    signal_source: str | None
    candidate_rule: str | None
    candidate_price: float | None
    latest_close: float | None
    vs_candidate_pct: float | None
    entry_valid: bool | None
    entry_volume_ratio: float | None
    entry_reject_reason: str | None
    entry_close_position: float | None
    breakout_range_ratio: float | None
    breakout_quality: str | None
    weekly_volume_ratio: float | None
    eps_yoy_growth: float | None
    dist_to_52w_high_pct: float | None
    pullback_v_is_dry: bool | None
    sector: str | None
    industry: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True)
class BFTransitionResult:
    """Authoritative BreakoutFollow transition result shared by all consumers.

    rows:
        Current-left review rows. These retain the existing review_* contract so
        the Dashboard can consume them without re-deriving transition semantics.
    attention_events:
        Only material ACTIONABLE-boundary events; not a status-change ledger.
    notification_events:
        attention_events that are new versus previous_pool when provided. When no
        previous_pool is supplied, all attention_events are returned here.
    """

    rows: pd.DataFrame
    exited_pool: pd.DataFrame
    actionable_codes: tuple[str, ...]
    review_summary: dict[str, int]
    attention_events: tuple[BFAttentionEvent, ...]
    notification_events: tuple[BFAttentionEvent, ...]
    attention_summary: dict[str, int]
    baseline_available: bool
    previous_available: bool

    def attention_payload(self, *, notification_only: bool = False) -> tuple[dict[str, Any], ...]:
        events = self.notification_events if notification_only else self.attention_events
        return tuple(event.to_dict() for event in events)


def to_bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1", "1.0"}:
        return True
    if text in {"false", "f", "no", "n", "0", "0.0"}:
        return False
    return None


def normalize_transition_pool(
    pool: pd.DataFrame,
    *,
    label: str,
    allow_empty: bool = False,
) -> pd.DataFrame:
    if pool is None or pool.empty:
        if allow_empty:
            return pd.DataFrame(columns=list(pool.columns) if pool is not None else [])
        raise ValueError(f"{label} pool cannot be empty")
    result = normalize_pool_df(pool)
    required = {"code", "signal"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"{label} pool missing required columns: {sorted(missing)}")
    codes = result["code"].astype("string").str.strip()
    if codes.isna().any() or codes.eq("").any() or codes.str.lower().eq("nan").any():
        raise ValueError(f"{label} pool code cannot be empty")
    if codes.duplicated().any():
        raise ValueError(f"{label} pool code cannot be duplicate")
    result = result.copy()
    result["code"] = codes.astype(str)
    converted = result["signal"].map(to_bool)
    if converted.isna().any():
        raise ValueError(f"{label} pool signal must be a valid boolean")
    result["signal"] = pd.Series(converted.tolist(), index=result.index, dtype="object")
    return result


def _positive_number(value: Any, *, field: str, code: str) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed) or not np.isfinite(parsed) or float(parsed) <= 0:
        raise ValueError(f"{code}: {field} must be a positive finite number")
    return float(parsed)


def _nonempty_status(value: Any, *, code: str) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        raise ValueError(f"{code}: IBD enrichment is incomplete")
    status = str(value).strip()
    if status not in ENTRY_STATUSES:
        raise ValueError(f"{code}: invalid ibd_entry_status {status}")
    return status


def _calculate_status(entry_valid: Any, candidate: float, latest_close: float) -> str:
    if to_bool(entry_valid) is not True:
        return "UNCONFIRMED"
    # Keep Carry status aligned with the two-decimal funnel field published by
    # quant_trade for ordinary signal rows.
    current_vs_candidate_pct = round((latest_close / candidate - 1.0) * 100.0, 2)
    if current_vs_candidate_pct < 0.0:
        return "BELOW_TRIGGER"
    if current_vs_candidate_pct <= 5.0:
        return "ACTIONABLE"
    return "EXTENDED"


def _entry_change(baseline: str | None, effective: str | None) -> tuple[str, str]:
    # This contract is intentionally identical to the pre-extraction Dashboard
    # logic. Do not change its labels without a dedicated UI/API migration.
    if baseline != "ACTIONABLE" and effective == "ACTIONABLE":
        return "BECAME_ACTIONABLE", "BECAME_ACTIONABLE"
    if baseline == "ACTIONABLE" and effective != "ACTIONABLE":
        if effective == "EXTENDED":
            detail = "ACTIONABLE_TO_EXTENDED"
        elif effective == "BELOW_TRIGGER":
            detail = "ACTIONABLE_TO_BELOW_TRIGGER"
        else:
            detail = "LEFT_ACTIONABLE"
        return detail, "LEFT_ACTIONABLE"
    if baseline == "ACTIONABLE" and effective == "ACTIONABLE":
        return "STILL_ACTIONABLE", "UNCHANGED"
    if baseline != effective:
        return "STATUS_CHANGED", "OTHER_CHANGES"
    return "UNCHANGED", "UNCHANGED"


def _change_label(origin: str, baseline: str | None, effective: str | None, entry_change: str) -> str:
    if entry_change == "STILL_ACTIONABLE":
        return "STILL ACTIONABLE"
    if baseline == "ACTIONABLE" and effective:
        return f"ACTIONABLE → {effective.replace('_', ' ')}"
    if origin in {"NEW", "CARRY", "RECONFIRMED"} and effective:
        return f"{origin} → {effective.replace('_', ' ')}"
    return effective.replace("_", " ") if effective else origin


def _build_summary(review: pd.DataFrame, exited: pd.DataFrame) -> dict[str, int]:
    summary: dict[str, int] = {
        "CURRENT_POOL": len(review),
        "EXITED_POOL": len(exited),
        "ACTIVE_SIGNALS": int(review.get("review_watch_active", pd.Series(dtype=bool)).sum()),
    }
    for value in ("BECAME_ACTIONABLE", "LEFT_ACTIONABLE", "OTHER_CHANGES", "UNCHANGED"):
        summary[value] = int(review.get("review_change_group", pd.Series(dtype=object)).eq(value).sum())
    for value in ("NEW", "CARRY", "RECONFIRMED", "NONE"):
        summary[value] = int(review.get("review_signal_origin", pd.Series(dtype=object)).eq(value).sum())
    for value in ENTRY_STATUSES:
        summary[value] = int(review.get("review_effective_entry_status", pd.Series(dtype=object)).eq(value).sum())
    return summary


def _project_rows(
    current: pd.DataFrame,
    complete: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], dict[str, int], bool]:
    baseline_available = not complete.empty
    complete_by_code = {
        str(row["code"]): row
        for _, row in complete.iterrows()
    }
    current_codes = set(current["code"])
    rows: list[dict[str, Any]] = []

    for _, current_row in current.iterrows():
        code = str(current_row["code"])
        complete_row = complete_by_code.get(code)
        signal_current = to_bool(current_row.get("signal")) is True
        signal_complete = complete_row is not None and to_bool(complete_row.get("signal")) is True
        origin = "NONE"
        if baseline_available:
            origin = (
                "RECONFIRMED"
                if signal_current and signal_complete
                else "NEW"
                if signal_current
                else "CARRY"
                if signal_complete
                else "NONE"
            )
        watch_active = signal_current or signal_complete
        projected = current_row.to_dict()
        projected["review_pool_change"] = (
            "ENTERED" if complete_row is None else "STAYED"
        ) if baseline_available else "UNAVAILABLE"
        projected["review_signal_origin"] = origin
        projected["review_watch_active"] = bool(watch_active)
        projected["review_baseline_available"] = baseline_available

        selected_row = current_row if signal_current else complete_row if signal_complete else None
        for field in SOURCE_FACT_FIELDS:
            projected[f"review_{field}"] = selected_row.get(field) if selected_row is not None else None

        baseline_status = (
            _nonempty_status(complete_row.get("ibd_entry_status"), code=code)
            if signal_complete
            else None
        )
        effective_status: str | None = None
        candidate: float | None = None
        current_vs_candidate: float | None = None
        entry_valid: bool | None = None
        if watch_active:
            latest_close = _positive_number(current_row.get("latest_close"), field="latest_close", code=code)
            candidate = _positive_number(selected_row.get("ibd_candidate_price"), field="candidate", code=code)
            raw_valid = selected_row.get("ibd_entry_valid")
            if raw_valid is None or pd.isna(raw_valid) or to_bool(raw_valid) is None:
                raise ValueError(f"{code}: IBD enrichment is incomplete")
            entry_valid = to_bool(raw_valid)
            current_vs_candidate = round((latest_close / candidate - 1.0) * 100.0, 2)
            if signal_current:
                effective_status = _nonempty_status(current_row.get("ibd_entry_status"), code=code)
            else:
                effective_status = _calculate_status(entry_valid, candidate, latest_close)

        if baseline_available:
            entry_change, change_group = _entry_change(baseline_status, effective_status)
            change_label = _change_label(origin, baseline_status, effective_status, entry_change)
        else:
            entry_change, change_group, change_label = "UNAVAILABLE", "UNCHANGED", ""
        projected["review_candidate_price"] = candidate
        projected["review_entry_valid"] = entry_valid
        projected["review_baseline_entry_status"] = baseline_status
        projected["review_effective_entry_status"] = effective_status
        projected["review_current_vs_candidate_pct"] = current_vs_candidate
        projected["review_entry_change"] = entry_change
        projected["review_change_group"] = change_group
        projected["review_change_label"] = change_label
        projected["review_futu_actionable"] = bool(watch_active and effective_status == "ACTIONABLE")
        change_rank = {
            "BECAME_ACTIONABLE": 0,
            "LEFT_ACTIONABLE": 1,
            "OTHER_CHANGES": 2,
            "UNCHANGED": 3,
        }[change_group]
        status_rank = {status: rank for rank, status in enumerate(ENTRY_STATUSES)}.get(effective_status, 9)
        projected["review_priority"] = change_rank * 10 + status_rank
        rows.append(projected)

    review = pd.DataFrame(rows, index=current.index)
    if "review_entry_valid" in review.columns:
        review["review_entry_valid"] = pd.Series(
            review["review_entry_valid"].tolist(), index=review.index, dtype="object"
        )
    exited = complete.loc[~complete["code"].isin(current_codes)].copy() if not complete.empty else complete.copy()
    actionable = tuple(review.loc[review["review_futu_actionable"], "code"].astype(str).tolist())
    summary = _build_summary(review, exited)
    return review, exited, actionable, summary, baseline_available


def _optional_float(value: Any) -> float | None:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed) or not np.isfinite(parsed):
        return None
    return float(parsed)


def _optional_bool(value: Any) -> bool | None:
    return to_bool(value)


def _optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none"} else None


def _snapshot_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return _optional_text(value)
    return parsed.date().isoformat()


def _attention_descriptor(row: pd.Series) -> tuple[str, str, tuple[str, ...]] | None:
    entry_change = _optional_text(row.get("review_entry_change"))
    current_status = _optional_text(row.get("review_effective_entry_status"))
    origin = _optional_text(row.get("review_signal_origin")) or "NONE"

    if entry_change == "BECAME_ACTIONABLE":
        reasons = ["BECAME_ACTIONABLE"]
        if origin == "NEW":
            reasons.append("NEW_SIGNAL")
        return "BECAME_ACTIONABLE", ATTENTION_HIGH, tuple(reasons)
    if entry_change == "ACTIONABLE_TO_BELOW_TRIGGER":
        return (
            "ACTIONABLE_TO_BELOW_TRIGGER",
            ATTENTION_HIGH,
            ("LEFT_ACTIONABLE", "BELOW_TRIGGER"),
        )
    if entry_change == "ACTIONABLE_TO_EXTENDED":
        return (
            "ACTIONABLE_TO_EXTENDED",
            ATTENTION_MEDIUM,
            ("LEFT_ACTIONABLE", "EXTENDED"),
        )
    if entry_change == "LEFT_ACTIONABLE":
        event_type = (
            "ACTIONABLE_TO_UNCONFIRMED"
            if current_status == "UNCONFIRMED"
            else "LEFT_ACTIONABLE"
        )
        return event_type, ATTENTION_HIGH, ("LEFT_ACTIONABLE",)
    return None


def _event_from_row(
    row: pd.Series,
    *,
    previous_status: str | None,
    previous_event_type: str | None,
    previous_available: bool,
) -> BFAttentionEvent | None:
    descriptor = _attention_descriptor(row)
    if descriptor is None:
        return None
    event_type, importance, reasons = descriptor
    is_new = None if not previous_available else event_type != previous_event_type

    return BFAttentionEvent(
        code=str(row.get("code")),
        event_type=event_type,
        importance=importance,
        change_group=_optional_text(row.get("review_change_group")) or "UNCHANGED",
        signal_origin=_optional_text(row.get("review_signal_origin")) or "NONE",
        baseline_status=_optional_text(row.get("review_baseline_entry_status")),
        previous_status=previous_status,
        current_status=_optional_text(row.get("review_effective_entry_status")),
        is_new_since_previous=is_new,
        reasons=reasons,
        snapshot_date=_snapshot_text(row.get("snapshot_date")),
        signal_source=_optional_text(row.get("review_signal_source")),
        candidate_rule=_optional_text(row.get("review_ibd_candidate_rule")),
        candidate_price=_optional_float(row.get("review_candidate_price")),
        latest_close=_optional_float(row.get("latest_close")),
        vs_candidate_pct=_optional_float(row.get("review_current_vs_candidate_pct")),
        entry_valid=_optional_bool(row.get("review_entry_valid")),
        entry_volume_ratio=_optional_float(row.get("review_ibd_entry_volume_ratio")),
        entry_reject_reason=_optional_text(row.get("review_ibd_entry_reject_reason")),
        entry_close_position=_optional_float(row.get("review_ibd_entry_close_position")),
        breakout_range_ratio=_optional_float(row.get("review_ibd_entry_breakout_range_ratio")),
        breakout_quality=_optional_text(row.get("ibd_breakout_quality")),
        weekly_volume_ratio=_optional_float(row.get("volume_ratio")),
        eps_yoy_growth=_optional_float(row.get("eps_yoy_growth")),
        dist_to_52w_high_pct=_optional_float(row.get("dist_to_52w_high_pct")),
        pullback_v_is_dry=_optional_bool(row.get("pullback_v_is_dry")),
        sector=_optional_text(row.get("sector")),
        industry=_optional_text(row.get("industry")),
    )


def _exited_actionable_event(
    row: pd.Series,
    *,
    previous_codes: set[str],
    previous_status_by_code: dict[str, str | None],
    previous_available: bool,
) -> BFAttentionEvent | None:
    if to_bool(row.get("signal")) is not True:
        return None
    if _optional_text(row.get("ibd_entry_status")) != "ACTIONABLE":
        return None
    code = str(row.get("code"))
    is_new = None if not previous_available else code in previous_codes
    return BFAttentionEvent(
        code=code,
        event_type="ACTIONABLE_EXITED_POOL",
        importance=ATTENTION_HIGH,
        change_group="LEFT_ACTIONABLE",
        signal_origin="EXITED",
        baseline_status="ACTIONABLE",
        previous_status=previous_status_by_code.get(code) if previous_available else None,
        current_status=None,
        is_new_since_previous=is_new,
        reasons=("LEFT_ACTIONABLE", "EXITED_POOL"),
        snapshot_date=None,
        signal_source=_optional_text(row.get("signal_source")),
        candidate_rule=_optional_text(row.get("ibd_candidate_rule")),
        candidate_price=_optional_float(row.get("ibd_candidate_price")),
        latest_close=None,
        vs_candidate_pct=None,
        entry_valid=_optional_bool(row.get("ibd_entry_valid")),
        entry_volume_ratio=_optional_float(row.get("ibd_entry_volume_ratio")),
        entry_reject_reason=_optional_text(row.get("ibd_entry_reject_reason")),
        entry_close_position=_optional_float(row.get("ibd_entry_close_position")),
        breakout_range_ratio=_optional_float(row.get("ibd_entry_breakout_range_ratio")),
        breakout_quality=_optional_text(row.get("ibd_breakout_quality")),
        weekly_volume_ratio=_optional_float(row.get("volume_ratio")),
        eps_yoy_growth=_optional_float(row.get("eps_yoy_growth")),
        dist_to_52w_high_pct=_optional_float(row.get("dist_to_52w_high_pct")),
        pullback_v_is_dry=_optional_bool(row.get("pullback_v_is_dry")),
        sector=_optional_text(row.get("sector")),
        industry=_optional_text(row.get("industry")),
    )


def analyze_bf_transitions(
    current_pool: pd.DataFrame,
    complete_pool: pd.DataFrame,
    previous_pool: pd.DataFrame | None = None,
) -> BFTransitionResult:
    """Analyze BF lifecycle facts without any UI, Futu, Telegram, or file I/O.

    complete_pool is the frozen weekend baseline used by the Dashboard.
    previous_pool is optional and only answers whether a material event is newly
    occurring since the prior successful intrawweek snapshot. It does not change
    the authoritative weekend→current Dashboard transition.
    """

    current = normalize_transition_pool(current_pool, label="current")
    complete = normalize_transition_pool(complete_pool, label="complete", allow_empty=True)
    review, exited, actionable, review_summary, baseline_available = _project_rows(current, complete)

    previous_available = previous_pool is not None and not previous_pool.empty
    previous_by_code: dict[str, pd.Series] = {}
    previous_event_type_by_code: dict[str, str | None] = {}
    previous_status_by_code: dict[str, str | None] = {}
    previous_codes: set[str] = set()
    if previous_available:
        previous = normalize_transition_pool(previous_pool, label="previous")
        previous_review, _, _, _, _ = _project_rows(previous, complete)
        previous_codes = set(previous["code"].astype(str))
        for _, row in previous_review.iterrows():
            code = str(row["code"])
            previous_by_code[code] = row
            previous_status_by_code[code] = _optional_text(
                row.get("review_effective_entry_status")
            )
            descriptor = _attention_descriptor(row)
            previous_event_type_by_code[code] = descriptor[0] if descriptor else None

    events: list[BFAttentionEvent] = []
    for _, row in review.iterrows():
        code = str(row["code"])
        previous_row = previous_by_code.get(code)
        event = _event_from_row(
            row,
            previous_status=(
                _optional_text(previous_row.get("review_effective_entry_status"))
                if previous_row is not None
                else None
            ),
            previous_event_type=previous_event_type_by_code.get(code),
            previous_available=previous_available,
        )
        if event is not None:
            events.append(event)

    if baseline_available and not exited.empty:
        for _, row in exited.iterrows():
            event = _exited_actionable_event(
                row,
                previous_codes=previous_codes,
                previous_status_by_code=previous_status_by_code,
                previous_available=previous_available,
            )
            if event is not None:
                events.append(event)

    importance_order = {ATTENTION_HIGH: 0, ATTENTION_MEDIUM: 1}
    events.sort(key=lambda event: (importance_order.get(event.importance, 9), event.code, event.event_type))
    attention_events = tuple(events)
    notification_events = tuple(
        event
        for event in attention_events
        if not previous_available or event.is_new_since_previous is True
    )
    attention_summary = {
        "TOTAL": len(attention_events),
        "HIGH": sum(event.importance == ATTENTION_HIGH for event in attention_events),
        "MEDIUM": sum(event.importance == ATTENTION_MEDIUM for event in attention_events),
        "NOTIFICATION_ELIGIBLE": len(notification_events),
    }

    return BFTransitionResult(
        rows=review,
        exited_pool=exited,
        actionable_codes=actionable,
        review_summary=review_summary,
        attention_events=attention_events,
        notification_events=notification_events,
        attention_summary=attention_summary,
        baseline_available=baseline_available,
        previous_available=previous_available,
    )


__all__ = [
    "ATTENTION_HIGH",
    "ATTENTION_MEDIUM",
    "BFAttentionEvent",
    "BFTransitionResult",
    "ENTRY_STATUSES",
    "SOURCE_FACT_FIELDS",
    "analyze_bf_transitions",
]
