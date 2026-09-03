from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from dashboard.skill_industry_eps_known import (
    is_pullback_rule,
    reasoned_item,
    to_bool,
)


@dataclass(frozen=True)
class LaneFacts:
    code: str
    current_lane: str
    entry_status: str
    setup_route: str
    fresh_demand: bool
    follow_through: bool
    geometry_failure: bool
    quality_state: str
    composition_group: str
    actionable_pullback_context_branch: bool
    non_actionable_pullback_context_branch: bool


def _bool_from_panel(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, (int, float)):
        return float(value) == 1.0
    return str(value).strip().lower() in {"true", "1", "1.0", "yes"}


def classify_lane_facts(row: pd.Series, row_idx: int) -> LaneFacts:
    """Decompose the current Lane into orthogonal route/evidence/status facts.

    This is research-only. It deliberately reuses Production reason_codes/risk_codes
    so the audit diagnoses the current semantics rather than inventing new thresholds.
    """
    item = reasoned_item(row, row_idx)
    reason = set(item.reason_codes)
    risk = set(item.risk_codes)
    rule = str(row.get("ibd_candidate_rule", "") or "").strip()

    setup_route = "pullback" if is_pullback_rule(rule) else "non_pullback"
    fresh_demand = (
        "near_buy_point" in reason
        and "volume_confirms_breakout" in reason
    )
    follow_through = (
        "eps_acceleration_support" in reason
        or "weekly_volume_follow_through" in reason
    )
    geometry_failure = "clear_geometry_failure" in risk

    if geometry_failure:
        quality_state = "failure"
    elif fresh_demand and follow_through:
        quality_state = "confirmed"
    elif fresh_demand:
        quality_state = "standard"
    else:
        quality_state = "incomplete"

    if quality_state == "confirmed" and setup_route == "non_pullback":
        composition_group = "confirmed_non_pullback"
    elif quality_state == "confirmed" and setup_route == "pullback":
        composition_group = "confirmed_pullback"
    elif quality_state == "standard":
        composition_group = "standard"
    elif quality_state == "failure":
        composition_group = "failure"
    else:
        composition_group = "incomplete"

    near = "near_buy_point" in reason
    pullback_context = (
        "near_52w_high" in reason
        or "weekly_volume_follow_through" in reason
        or "dry_pullback" in reason
    )
    non_actionable_branch = bool(
        item.entry_status != "ACTIONABLE"
        and setup_route == "pullback"
        and near
        and pullback_context
        and item.lane == "constructive_pullback"
    )
    actionable_branch = bool(
        item.entry_status == "ACTIONABLE"
        and setup_route == "pullback"
        and fresh_demand
        and follow_through
        and item.lane == "constructive_pullback"
    )

    return LaneFacts(
        code=item.code,
        current_lane=item.lane,
        entry_status=item.entry_status,
        setup_route=setup_route,
        fresh_demand=fresh_demand,
        follow_through=follow_through,
        geometry_failure=geometry_failure,
        quality_state=quality_state,
        composition_group=composition_group,
        actionable_pullback_context_branch=actionable_branch,
        non_actionable_pullback_context_branch=non_actionable_branch,
    )


def lane_fact_rows(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row_idx, (_, row) in enumerate(snapshot_df.iterrows()):
        if to_bool(row.get("signal")) is not True:
            continue
        if not str(row.get("ibd_candidate_rule", "") or "").strip():
            continue

        facts = classify_lane_facts(row, row_idx)
        rows.append({
            "snapshot_date": str(row.get("snapshot_date", "") or ""),
            "code": facts.code,
            "current_lane": facts.current_lane,
            "entry_status": facts.entry_status,
            "setup_route": facts.setup_route,
            "fresh_demand": facts.fresh_demand,
            "follow_through": facts.follow_through,
            "geometry_failure": facts.geometry_failure,
            "quality_state": facts.quality_state,
            "composition_group": facts.composition_group,
            "actionable_pullback_context_branch": facts.actionable_pullback_context_branch,
            "non_actionable_pullback_context_branch": facts.non_actionable_pullback_context_branch,
            "b0_eligible": _bool_from_panel(row.get("b0_eligible")),
            "industry": str(row.get("industry", "") or "").strip(),
        })
    return pd.DataFrame(rows)


def expected_actionable_mapping_ok(row: pd.Series) -> bool:
    """Invariant for B0-eligible/actionable rows under current lane_for semantics."""
    if not bool(row.get("b0_eligible", False)):
        return True

    lane = str(row["current_lane"])
    group = str(row["composition_group"])
    if lane == "fresh_demand_alpha":
        return group == "confirmed_non_pullback"
    if lane == "constructive_pullback":
        return group == "confirmed_pullback"
    if lane == "standard_breakout":
        return group == "standard"
    if lane == "incomplete_evidence":
        return group == "incomplete"
    if lane == "tail_risk":
        return group == "failure"
    return False
