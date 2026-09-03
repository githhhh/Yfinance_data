from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dashboard.data_utils import normalize_pool_df


ENTRY_STATUSES = ("ACTIONABLE", "UNCONFIRMED", "BELOW_TRIGGER", "EXTENDED")

# Resolver-owned fields must remain atomically owned by the current signal row,
# or by the weekend row only when the existing Carry rule applies.
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
    """Material transition facts for downstream analysis/notification.

    facts is deliberately factual rather than notification copy. Consumers
    can combine it with market context and decide whether/how to present it.
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
    facts: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "event_type": self.event_type,
            "importance": self.importance,
            "change_group": self.change_group,
            "signal_origin": self.signal_origin,
            "baseline_status": self.baseline_status,
            "previous_status": self.previous_status,
            "current_status": self.current_status,
            "is_new_since_previous": self.is_new_since_previous,
            "reasons": list(self.reasons),
            "facts": dict(self.facts),
        }


@dataclass(frozen=True)
class BFTransitionResult:
    """Single transition contract shared by Dashboard, analysis and Futu."""

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
    missing = {"code", "signal"}.difference(result.columns)
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
    distance = round((latest_close / candidate - 1.0) * 100.0, 2)
    if distance < 0.0:
        return "BELOW_TRIGGER"
    if distance <= 5.0:
        return "ACTIONABLE"
    return "EXTENDED"


def _entry_change(baseline: str | None, effective: str | None) -> tuple[str, str]:
    # Stable Dashboard contract: extraction must not change these labels.
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


def _project(
    current: pd.DataFrame,
    complete: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], dict[str, int], bool]:
    baseline_available = not complete.empty
    complete_by_code = {str(row["code"]): row for _, row in complete.iterrows()}
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
                "RECONFIRMED" if signal_current and signal_complete
                else "NEW" if signal_current
                else "CARRY" if signal_complete
                else "NONE"
            )
        watch_active = signal_current or signal_complete

        projected = current_row.to_dict()
        projected["review_pool_change"] = (
            ("ENTERED" if complete_row is None else "STAYED")
            if baseline_available else "UNAVAILABLE"
        )
        projected["review_signal_origin"] = origin
        projected["review_watch_active"] = bool(watch_active)
        projected["review_baseline_available"] = baseline_available

        selected_row = current_row if signal_current else complete_row if signal_complete else None
        for field in SOURCE_FACT_FIELDS:
            projected[f"review_{field}"] = selected_row.get(field) if selected_row is not None else None

        baseline_status = (
            _nonempty_status(complete_row.get("ibd_entry_status"), code=code)
            if signal_complete else None
        )
        effective_status = None
        candidate = None
        current_vs_candidate = None
        entry_valid = None
        if watch_active:
            latest_close = _positive_number(current_row.get("latest_close"), field="latest_close", code=code)
            candidate = _positive_number(selected_row.get("ibd_candidate_price"), field="candidate", code=code)
            raw_valid = selected_row.get("ibd_entry_valid")
            if raw_valid is None or pd.isna(raw_valid) or to_bool(raw_valid) is None:
                raise ValueError(f"{code}: IBD enrichment is incomplete")
            entry_valid = to_bool(raw_valid)
            current_vs_candidate = round((latest_close / candidate - 1.0) * 100.0, 2)
            effective_status = (
                _nonempty_status(current_row.get("ibd_entry_status"), code=code)
                if signal_current
                else _calculate_status(entry_valid, candidate, latest_close)
            )

        if baseline_available:
            entry_change, change_group = _entry_change(baseline_status, effective_status)
            change_label = _change_label(origin, baseline_status, effective_status, entry_change)
        else:
            entry_change, change_group, change_label = "UNAVAILABLE", "UNCHANGED", ""

        projected.update(
            {
                "review_candidate_price": candidate,
                "review_entry_valid": entry_valid,
                "review_baseline_entry_status": baseline_status,
                "review_effective_entry_status": effective_status,
                "review_current_vs_candidate_pct": current_vs_candidate,
                "review_entry_change": entry_change,
                "review_change_group": change_group,
                "review_change_label": change_label,
                "review_futu_actionable": bool(watch_active and effective_status == "ACTIONABLE"),
            }
        )
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
    return review, exited, actionable, _review_summary(review, exited), baseline_available


def _review_summary(review: pd.DataFrame, exited: pd.DataFrame) -> dict[str, int]:
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


def _text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none"} else None


def _number(value: Any) -> float | None:
    parsed = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(parsed) or not np.isfinite(parsed) else float(parsed)


def _date_text(value: Any) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce")
    return _text(value) if pd.isna(parsed) else parsed.date().isoformat()


def _attention_descriptor(row: pd.Series) -> tuple[str, str, tuple[str, ...]] | None:
    change = _text(row.get("review_entry_change"))
    current_status = _text(row.get("review_effective_entry_status"))
    origin = _text(row.get("review_signal_origin")) or "NONE"
    if change == "BECAME_ACTIONABLE":
        reasons = ("BECAME_ACTIONABLE", "NEW_SIGNAL") if origin == "NEW" else ("BECAME_ACTIONABLE",)
        return "BECAME_ACTIONABLE", ATTENTION_HIGH, reasons
    if change == "ACTIONABLE_TO_BELOW_TRIGGER":
        return "ACTIONABLE_TO_BELOW_TRIGGER", ATTENTION_HIGH, ("LEFT_ACTIONABLE", "BELOW_TRIGGER")
    if change == "ACTIONABLE_TO_EXTENDED":
        return "ACTIONABLE_TO_EXTENDED", ATTENTION_MEDIUM, ("LEFT_ACTIONABLE", "EXTENDED")
    if change == "LEFT_ACTIONABLE":
        event_type = "ACTIONABLE_TO_UNCONFIRMED" if current_status == "UNCONFIRMED" else "LEFT_ACTIONABLE"
        return event_type, ATTENTION_HIGH, ("LEFT_ACTIONABLE",)
    return None


def _facts(row: pd.Series) -> dict[str, Any]:
    return {
        "snapshot_date": _date_text(row.get("snapshot_date")),
        "signal_source": _text(row.get("review_signal_source")),
        "candidate_rule": _text(row.get("review_ibd_candidate_rule")),
        "candidate_price": _number(row.get("review_candidate_price")),
        "latest_close": _number(row.get("latest_close")),
        "vs_candidate_pct": _number(row.get("review_current_vs_candidate_pct")),
        "entry_valid": to_bool(row.get("review_entry_valid")),
        "entry_volume_ratio": _number(row.get("review_ibd_entry_volume_ratio")),
        "entry_reject_reason": _text(row.get("review_ibd_entry_reject_reason")),
        "entry_close_position": _number(row.get("review_ibd_entry_close_position")),
        "breakout_range_ratio": _number(row.get("review_ibd_entry_breakout_range_ratio")),
        "breakout_quality": _text(row.get("ibd_breakout_quality")),
        "weekly_volume_ratio": _number(row.get("volume_ratio")),
        "eps_yoy_growth": _number(row.get("eps_yoy_growth")),
        "dist_to_52w_high_pct": _number(row.get("dist_to_52w_high_pct")),
        "pullback_v_is_dry": to_bool(row.get("pullback_v_is_dry")),
        "sector": _text(row.get("sector")),
        "industry": _text(row.get("industry")),
    }


def analyze_bf_transitions(
    current_pool: pd.DataFrame,
    complete_pool: pd.DataFrame,
    previous_pool: pd.DataFrame | None = None,
) -> BFTransitionResult:
    """Return authoritative weekend→current facts plus material attention events.

    previous_pool is optional and only provides notification de-duplication.
    It never changes the weekend→current rows consumed by the Dashboard.
    """

    current = normalize_transition_pool(current_pool, label="current")
    complete = normalize_transition_pool(complete_pool, label="complete", allow_empty=True)
    review, exited, actionable, summary, baseline_available = _project(current, complete)

    previous_available = previous_pool is not None and not previous_pool.empty
    previous_by_code: dict[str, pd.Series] = {}
    previous_event_type: dict[str, str | None] = {}
    if previous_available:
        previous = normalize_transition_pool(previous_pool, label="previous")
        previous_review, _, _, _, _ = _project(previous, complete)
        for _, row in previous_review.iterrows():
            code = str(row["code"])
            previous_by_code[code] = row
            descriptor = _attention_descriptor(row)
            previous_event_type[code] = descriptor[0] if descriptor else None

    events: list[BFAttentionEvent] = []
    for _, row in review.iterrows():
        descriptor = _attention_descriptor(row)
        if descriptor is None:
            continue
        event_type, importance, reasons = descriptor
        code = str(row["code"])
        previous_row = previous_by_code.get(code)
        events.append(
            BFAttentionEvent(
                code=code,
                event_type=event_type,
                importance=importance,
                change_group=_text(row.get("review_change_group")) or "UNCHANGED",
                signal_origin=_text(row.get("review_signal_origin")) or "NONE",
                baseline_status=_text(row.get("review_baseline_entry_status")),
                previous_status=(
                    _text(previous_row.get("review_effective_entry_status"))
                    if previous_row is not None else None
                ),
                current_status=_text(row.get("review_effective_entry_status")),
                is_new_since_previous=(
                    None if not previous_available
                    else event_type != previous_event_type.get(code)
                ),
                reasons=reasons,
                facts=_facts(row),
            )
        )

    order = {ATTENTION_HIGH: 0, ATTENTION_MEDIUM: 1}
    events.sort(key=lambda event: (order.get(event.importance, 9), event.code, event.event_type))
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
        review_summary=summary,
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
