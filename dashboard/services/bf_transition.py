from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from typing import Any

import numpy as np
import pandas as pd

from dashboard.data_utils import normalize_pool_df


ENTRY_STATUSES = ("ACTIONABLE", "UNCONFIRMED", "BELOW_TRIGGER", "EXTENDED")
PUSH_BASELINE_COMPLETE = "COMPLETE"
PUSH_BASELINE_PREVIOUS_MIDWEEK = "PREVIOUS_MIDWEEK"

# Resolver-owned fields must remain atomically owned by the current signal row,
# or by the complete-week row only when the existing Carry rule applies.
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
    """Material transition facts for downstream analysis/notification."""

    code: str
    event_type: str
    importance: str
    change_group: str
    signal_origin: str
    baseline_status: str | None
    current_status: str | None
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
            "current_status": self.current_status,
            "reasons": list(self.reasons),
            "facts": dict(self.facts),
        }


@dataclass(frozen=True)
class BFTransitionResult:
    """Single transition contract shared by Dashboard, analysis and Futu.

    ``rows`` / ``exited_pool`` / ``review_summary`` are always the stable
    complete-week -> current Dashboard projection.

    ``attention_events`` are a separate daily comparison. On the first run of
    a review week they use complete-week -> current; once a newer midweek
    snapshot exists they use previous-midweek -> current. Previous and current
    midweek states are first resolved through the same complete-week Carry
    semantics used by the Dashboard, so the comparison is between effective
    states rather than raw CSV status fields.
    """

    rows: pd.DataFrame
    exited_pool: pd.DataFrame
    actionable_codes: tuple[str, ...]
    review_summary: dict[str, int]
    attention_events: tuple[BFAttentionEvent, ...]
    attention_summary: dict[str, int]
    baseline_available: bool
    previous_available: bool
    push_baseline: str
    push_baseline_snapshot_date: date | None
    current_snapshot_date: date | None
    push_ready: bool
    push_warnings: tuple[str, ...]

    def attention_payload(self) -> tuple[dict[str, Any], ...]:
        return tuple(event.to_dict() for event in self.attention_events)

    def push_payload(self) -> dict[str, Any]:
        """Return the JSON-friendly payload intended for the analysis repo."""
        return {
            "baseline": self.push_baseline,
            "baseline_snapshot_date": (
                self.push_baseline_snapshot_date.isoformat()
                if self.push_baseline_snapshot_date is not None
                else None
            ),
            "current_snapshot_date": (
                self.current_snapshot_date.isoformat()
                if self.current_snapshot_date is not None
                else None
            ),
            "ready": self.push_ready,
            "warnings": list(self.push_warnings),
            "summary": dict(self.attention_summary),
            "events": list(self.attention_payload()),
        }


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


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none"} else None


def _candidate_extra(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = _clean_text(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _same_price(left: Any, right: Any) -> bool:
    lhs = pd.to_numeric(left, errors="coerce")
    rhs = pd.to_numeric(right, errors="coerce")
    if pd.isna(lhs) or pd.isna(rhs):
        return False
    return bool(np.isclose(float(lhs), float(rhs), rtol=1e-9, atol=1e-4))


def _iso_date(value: Any) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.date().isoformat()


def _anchor(*, rule: str, trigger_price: float, anchor_date: str | None, anchor_type: str | None) -> dict[str, Any] | None:
    if not anchor_date:
        return None
    kind = _clean_text(anchor_type)
    return {
        "identity": "|".join((rule, kind or "-", anchor_date)),
        "rule": rule,
        "trigger_price": trigger_price,
        "anchor_date": anchor_date,
        "anchor_type": kind,
    }


def _buy_point_anchor(row: pd.Series | None) -> dict[str, Any] | None:
    """Return structural buy-point identity; trigger price is not the identity."""
    if row is None:
        return None
    rule = (_clean_text(row.get("ibd_candidate_rule")) or "").lower()
    candidate = pd.to_numeric(row.get("ibd_candidate_price"), errors="coerce")
    if not rule or pd.isna(candidate) or not np.isfinite(candidate) or float(candidate) <= 0:
        return None
    trigger = float(candidate)
    extra = _candidate_extra(row.get("ibd_candidate_extra"))
    if rule in {"ceiling", "ceiling_breakout"}:
        if not _same_price(trigger, row.get("ceiling")):
            return None
        return _anchor(rule="ceiling", trigger_price=trigger, anchor_date=_iso_date(row.get("ceiling_date")), anchor_type="CEILING")
    if rule == "pivot":
        selected = extra.get("selected_pivot")
        if isinstance(selected, dict) and _same_price(trigger, selected.get("price")):
            resolved = _anchor(rule=rule, trigger_price=trigger, anchor_date=_iso_date(selected.get("resistance_date")), anchor_type=_clean_text(selected.get("box_type")) or "BOX")
            if resolved is not None:
                return resolved
        matches: list[tuple[str, str]] = []
        for item in extra.get("pivot_candidates", []):
            if not isinstance(item, dict) or not _same_price(trigger, item.get("price")):
                continue
            anchor_date = _iso_date(item.get("resistance_date"))
            if anchor_date:
                matches.append((_clean_text(item.get("box_type")) or "BOX", anchor_date))
        if not matches:
            return None
        dates = {item[1] for item in matches}
        if len(dates) != 1:
            return None
        return _anchor(rule=rule, trigger_price=trigger, anchor_date=next(iter(dates)), anchor_type="+".join(sorted({item[0] for item in matches})))
    if rule == "ceiling_pullback":
        pending_high = extra.get("pending_high")
        if pending_high is not None and not _same_price(trigger, pending_high):
            return None
        anchor_date = _iso_date(extra.get("pending_high_date"))
        touch_high = extra.get("touch_high")
        if not anchor_date and touch_high is not None and _same_price(trigger, touch_high) and _same_price(pending_high, touch_high):
            anchor_date = _iso_date(extra.get("touch_date"))
        return _anchor(rule=rule, trigger_price=trigger, anchor_date=anchor_date, anchor_type="PULLBACK_HIGH")
    if rule == "ma10_touch_confirm":
        pending_high = extra.get("pending_high")
        if pending_high is not None and not _same_price(trigger, pending_high):
            return None
        return _anchor(rule=rule, trigger_price=trigger, anchor_date=_iso_date(extra.get("pending_high_date")), anchor_type="PENDING_HIGH")
    if rule == "three_weeks_tight":
        twk_high = extra.get("twk_high")
        if twk_high is not None and not _same_price(trigger, twk_high):
            return None
        return _anchor(rule=rule, trigger_price=trigger, anchor_date=_iso_date(extra.get("twk_high_date")), anchor_type="TIGHT_HIGH")
    return None


def _price_zone(candidate: Any, latest_close: Any) -> str | None:
    trigger = pd.to_numeric(candidate, errors="coerce")
    close = pd.to_numeric(latest_close, errors="coerce")
    if pd.isna(trigger) or pd.isna(close):
        return None
    if not np.isfinite(trigger) or not np.isfinite(close) or float(trigger) <= 0 or float(close) <= 0:
        return None
    distance = round((float(close) / float(trigger) - 1.0) * 100.0, 2)
    if distance < 0:
        return "BELOW_BUY_POINT"
    if distance <= 5.0:
        return "BUY_ZONE"
    return "EXTENDED"


def _tracked_buy_point_change(previous_zone: str | None, current_zone: str | None) -> str:
    if not previous_zone or not current_zone or previous_zone == current_zone:
        return "UNCHANGED"
    return {"BUY_ZONE": "ENTERED_BUY_ZONE", "BELOW_BUY_POINT": "BELOW_BUY_POINT", "EXTENDED": "BECAME_EXTENDED"}[current_zone]


def _buy_point_change_label(weekend_change: str, new_zone: str | None) -> str:
    parts = []
    weekend_label = {"ENTERED_BUY_ZONE": "Entered Buy Zone", "BELOW_BUY_POINT": "Below Buy Point", "BECAME_EXTENDED": "Became Extended"}.get(weekend_change)
    if weekend_label:
        parts.append(weekend_label)
    if new_zone:
        new_label = {"BUY_ZONE": "In Buy Zone", "BELOW_BUY_POINT": "Below Buy Point", "EXTENDED": "Extended"}[new_zone]
        parts.append(f"New BP · {new_label}")
    return " · ".join(parts)


def _buy_point_priority(weekend_change: str, new_zone: str | None) -> int:
    ranks: list[int] = []
    ranks.extend({"ENTERED_BUY_ZONE": [0], "BELOW_BUY_POINT": [1], "BECAME_EXTENDED": [2]}.get(weekend_change, []))
    ranks.extend({"BUY_ZONE": [0], "BELOW_BUY_POINT": [1], "EXTENDED": [2]}.get(new_zone, []))
    return min(ranks) if ranks else 9

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
    """Resolve one midweek pool against the complete-week authority."""

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
        latest_close = None
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

        weekend_anchor = _buy_point_anchor(complete_row) if signal_complete else None
        current_anchor = _buy_point_anchor(current_row) if signal_current else None
        weekend_candidate = None
        weekend_previous_zone = None
        weekend_current_zone = None
        weekend_change = "UNCHANGED"
        if signal_complete:
            weekend_candidate = _positive_number(complete_row.get("ibd_candidate_price"), field="weekend candidate", code=code)
            weekend_close = _positive_number(complete_row.get("latest_close"), field="weekend latest_close", code=code)
            weekend_previous_zone = _price_zone(weekend_candidate, weekend_close)
            weekend_current_zone = _price_zone(weekend_candidate, latest_close)
            weekend_change = _tracked_buy_point_change(weekend_previous_zone, weekend_current_zone)
        new_buy_point = False
        if signal_current and current_anchor is not None:
            if not signal_complete:
                new_buy_point = True
            elif weekend_anchor is not None:
                new_buy_point = current_anchor["identity"] != weekend_anchor["identity"]
        new_buy_point_zone = _price_zone(current_anchor["trigger_price"], latest_close) if new_buy_point and current_anchor is not None else None
        has_value_change = bool(weekend_change != "UNCHANGED" or new_buy_point_zone is not None)
        buy_point_change_label = _buy_point_change_label(weekend_change, new_buy_point_zone)
        buy_point_priority = _buy_point_priority(weekend_change, new_buy_point_zone)

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
                "review_weekend_buy_point_change": weekend_change,
                "review_weekend_buy_point_previous_zone": weekend_previous_zone,
                "review_weekend_buy_point_zone": weekend_current_zone,
                "review_weekend_buy_point_price": weekend_candidate,
                "review_weekend_buy_point_date": weekend_anchor["anchor_date"] if weekend_anchor is not None else None,
                "review_weekend_buy_point_rule": weekend_anchor["rule"] if weekend_anchor is not None else None,
                "review_new_buy_point": bool(new_buy_point),
                "review_new_buy_point_zone": new_buy_point_zone,
                "review_new_buy_point_price": current_anchor["trigger_price"] if new_buy_point and current_anchor is not None else None,
                "review_new_buy_point_date": current_anchor["anchor_date"] if new_buy_point and current_anchor is not None else None,
                "review_new_buy_point_rule": current_anchor["rule"] if new_buy_point and current_anchor is not None else None,
                "review_new_buy_point_anchor_type": current_anchor["anchor_type"] if new_buy_point and current_anchor is not None else None,
                "review_has_value_change": has_value_change,
                "review_buy_point_change_label": buy_point_change_label,
                "review_buy_point_priority": buy_point_priority,
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


def _snapshot_date(pool: pd.DataFrame) -> date | None:
    if pool.empty or "snapshot_date" not in pool.columns:
        return None
    parsed = pd.to_datetime(pool["snapshot_date"], errors="coerce")
    if parsed.isna().any():
        return None
    values = tuple(dict.fromkeys(value.date() for value in parsed.tolist()))
    return values[0] if len(values) == 1 else None


def _select_push_baseline(
    complete: pd.DataFrame,
    previous: pd.DataFrame | None,
) -> tuple[str, date | None, tuple[str, ...]]:
    complete_date = _snapshot_date(complete)
    if previous is None or previous.empty:
        return PUSH_BASELINE_COMPLETE, complete_date, ()

    previous_date = _snapshot_date(previous)
    if complete_date is None or previous_date is None:
        return (
            PUSH_BASELINE_COMPLETE,
            complete_date,
            ("Push baseline snapshot is unavailable or ambiguous; chronology cannot be verified.",),
        )
    if complete_date >= previous_date:
        return PUSH_BASELINE_COMPLETE, complete_date, ()
    return PUSH_BASELINE_PREVIOUS_MIDWEEK, previous_date, ()


def _signal_origin_between(
    baseline_row: pd.Series | None,
    current_row: pd.Series,
) -> str:
    baseline_signal = baseline_row is not None and to_bool(baseline_row.get("signal")) is True
    current_signal = to_bool(current_row.get("signal")) is True
    if baseline_signal and current_signal:
        return "RECONFIRMED"
    if current_signal:
        return "NEW"
    if baseline_signal:
        return "CARRY"
    return "NONE"


def _attention_descriptor(
    baseline_status: str | None,
    current_status: str | None,
    signal_origin: str,
) -> tuple[str, str, str, tuple[str, ...]] | None:
    entry_change, change_group = _entry_change(baseline_status, current_status)
    if entry_change == "BECAME_ACTIONABLE":
        reasons = (
            ("BECAME_ACTIONABLE", "NEW_SIGNAL")
            if signal_origin == "NEW"
            else ("BECAME_ACTIONABLE",)
        )
        return "BECAME_ACTIONABLE", ATTENTION_HIGH, change_group, reasons
    if entry_change == "ACTIONABLE_TO_BELOW_TRIGGER":
        return (
            "ACTIONABLE_TO_BELOW_TRIGGER",
            ATTENTION_HIGH,
            change_group,
            ("LEFT_ACTIONABLE", "BELOW_TRIGGER"),
        )
    if entry_change == "ACTIONABLE_TO_EXTENDED":
        return (
            "ACTIONABLE_TO_EXTENDED",
            ATTENTION_MEDIUM,
            change_group,
            ("LEFT_ACTIONABLE", "EXTENDED"),
        )
    if entry_change == "LEFT_ACTIONABLE":
        event_type = (
            "ACTIONABLE_TO_UNCONFIRMED"
            if current_status == "UNCONFIRMED"
            else "LEFT_ACTIONABLE"
        )
        return event_type, ATTENTION_HIGH, change_group, ("LEFT_ACTIONABLE",)
    return None


def _facts(row: pd.Series) -> dict[str, Any]:
    return {
        "snapshot_date": _date_text(row.get("snapshot_date")),
        "weekend_signal_origin": _text(row.get("review_signal_origin")),
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


def _event_facts(current_row: pd.Series, baseline_row: pd.Series | None) -> dict[str, Any]:
    facts = _facts(current_row)
    if baseline_row is None:
        return facts
    baseline_facts = _facts(baseline_row)
    for key in (
        "signal_source",
        "candidate_rule",
        "candidate_price",
        "entry_valid",
        "entry_volume_ratio",
        "entry_reject_reason",
        "entry_close_position",
        "breakout_range_ratio",
    ):
        if facts.get(key) is None:
            facts[key] = baseline_facts.get(key)
    return facts


def _attention_events_from_effective_states(
    current_review: pd.DataFrame,
    baseline_review: pd.DataFrame,
) -> tuple[BFAttentionEvent, ...]:
    """Compare resolved states while retaining current market facts."""

    baseline_by_code = {
        str(row["code"]): row
        for _, row in baseline_review.iterrows()
    }
    current_codes = set(current_review["code"].astype(str))
    events: list[BFAttentionEvent] = []

    for _, current_row in current_review.iterrows():
        code = str(current_row["code"])
        baseline_row = baseline_by_code.get(code)
        baseline_status = (
            _text(baseline_row.get("review_effective_entry_status"))
            if baseline_row is not None
            else None
        )
        current_status = _text(current_row.get("review_effective_entry_status"))
        signal_origin = _signal_origin_between(baseline_row, current_row)
        descriptor = _attention_descriptor(baseline_status, current_status, signal_origin)
        if descriptor is None:
            continue
        event_type, importance, change_group, reasons = descriptor
        events.append(
            BFAttentionEvent(
                code=code,
                event_type=event_type,
                importance=importance,
                change_group=change_group,
                signal_origin=signal_origin,
                baseline_status=baseline_status,
                current_status=current_status,
                reasons=reasons,
                facts=_event_facts(current_row, baseline_row),
            )
        )

    # A baseline ACTIONABLE that disappears from the current pool has no
    # current-left row, but the exit is still a material lifecycle event.
    for code, baseline_row in baseline_by_code.items():
        if code in current_codes:
            continue
        baseline_status = _text(baseline_row.get("review_effective_entry_status"))
        if baseline_status != "ACTIONABLE":
            continue
        events.append(
            BFAttentionEvent(
                code=code,
                event_type="ACTIONABLE_EXITED_POOL",
                importance=ATTENTION_HIGH,
                change_group="LEFT_ACTIONABLE",
                signal_origin="EXITED",
                baseline_status="ACTIONABLE",
                current_status=None,
                reasons=("LEFT_ACTIONABLE", "EXITED_POOL"),
                facts=_facts(baseline_row),
            )
        )

    order = {ATTENTION_HIGH: 0, ATTENTION_MEDIUM: 1}
    events.sort(key=lambda event: (order.get(event.importance, 9), event.code, event.event_type))
    return tuple(events)


def analyze_bf_transitions(
    current_pool: pd.DataFrame,
    complete_pool: pd.DataFrame,
    previous_pool: pd.DataFrame | None = None,
) -> BFTransitionResult:
    """Analyze Dashboard and daily-attention transitions with separate baselines.

    Dashboard is always complete-week -> current. Daily attention selects its
    baseline by snapshot chronology. If complete is equally/newer than previous
    midweek, it is the first comparison of a new review week. Otherwise the
    previous and current midweek pools are independently resolved against the
    same complete pool, then their effective states are compared.
    """

    current = normalize_transition_pool(current_pool, label="current")
    complete = normalize_transition_pool(complete_pool, label="complete", allow_empty=True)

    # Stable Dashboard contract. Never substitute previous_pool here.
    review, exited, actionable, summary, baseline_available = _project(current, complete)

    previous_available = previous_pool is not None and not previous_pool.empty
    previous = (
        normalize_transition_pool(previous_pool, label="previous")
        if previous_available
        else None
    )
    push_baseline, push_baseline_date, baseline_warnings = _select_push_baseline(
        complete,
        previous,
    )
    current_date = _snapshot_date(current)
    warnings = list(baseline_warnings)

    # No-previous callers keep legacy behavior even without snapshot metadata.
    # Once previous_pool is supplied, chronology is part of the push contract
    # and therefore fails closed if it cannot be proven.
    push_ready = True
    if previous_available and (push_baseline_date is None or current_date is None):
        push_ready = False
        warnings.append(
            "Push snapshot ordering cannot be verified; attention events suppressed."
        )
    elif push_baseline_date is not None and current_date is not None and current_date <= push_baseline_date:
        push_ready = False
        warnings.append(
            "Current snapshot is not newer than the selected push baseline; attention events suppressed."
        )

    attention: tuple[BFAttentionEvent, ...] = ()
    if push_ready:
        if push_baseline == PUSH_BASELINE_PREVIOUS_MIDWEEK:
            assert previous is not None
            previous_review, _, _, _, _ = _project(previous, complete)
            attention = _attention_events_from_effective_states(review, previous_review)
        elif not complete.empty:
            complete_review, _, _, _, _ = _project(complete, complete)
            attention = _attention_events_from_effective_states(review, complete_review)

    attention_summary = {
        "TOTAL": len(attention),
        "HIGH": sum(event.importance == ATTENTION_HIGH for event in attention),
        "MEDIUM": sum(event.importance == ATTENTION_MEDIUM for event in attention),
    }
    return BFTransitionResult(
        rows=review,
        exited_pool=exited,
        actionable_codes=actionable,
        review_summary=summary,
        attention_events=attention,
        attention_summary=attention_summary,
        baseline_available=baseline_available,
        previous_available=previous_available,
        push_baseline=push_baseline,
        push_baseline_snapshot_date=push_baseline_date,
        current_snapshot_date=current_date,
        push_ready=push_ready,
        push_warnings=tuple(warnings),
    )


__all__ = [
    "ATTENTION_HIGH",
    "ATTENTION_MEDIUM",
    "BFAttentionEvent",
    "BFTransitionResult",
    "ENTRY_STATUSES",
    "PUSH_BASELINE_COMPLETE",
    "PUSH_BASELINE_PREVIOUS_MIDWEEK",
    "SOURCE_FACT_FIELDS",
    "analyze_bf_transitions",
]
