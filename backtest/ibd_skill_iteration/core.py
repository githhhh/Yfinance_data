from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from backtest.ibd_skill_replay.core import to_bool, to_float


@dataclass
class ReasonedCandidate:
    snapshot_date: str
    code: str
    raw_rank: int
    final_group: str
    lane: str
    entry_status: str
    industry: str
    sort_key: tuple
    reason_codes: list[str] = field(default_factory=list)
    risk_codes: list[str] = field(default_factory=list)
    feature_values: dict[str, object] = field(default_factory=dict)


def rank_reasoning_candidates(
    pool: pd.DataFrame,
    *,
    universe: str = "review",
    version: str = "v1",
) -> list[ReasonedCandidate]:
    candidates = []
    for row_idx, row in pool.iterrows():
        if not _is_review_universe(row):
            continue
        if universe == "actionable" and _entry_status(row) != "ACTIONABLE":
            continue
        candidates.append(_reasoned_item(row, row_idx, version=version))
    candidates.sort(key=lambda item: item.sort_key)
    for rank, item in enumerate(candidates, 1):
        item.raw_rank = rank
    return candidates


def rank_non_actionable_alpha_radar(
    pool: pd.DataFrame,
    *,
    version: str = "v2",
) -> list[ReasonedCandidate]:
    candidates = [
        item
        for item in rank_reasoning_candidates(pool, universe="review", version=version)
        if item.entry_status != "ACTIONABLE" and _is_non_actionable_alpha_worthy(item, version=version)
    ]
    if version == "v2":
        candidates = _balanced_v2_non_actionable_alpha(candidates)
    elif version == "v3":
        candidates = _balanced_v3_non_actionable_alpha(candidates)
    else:
        candidates.sort(key=_non_actionable_alpha_key_v1)
    for rank, item in enumerate(candidates, 1):
        item.raw_rank = rank
        item.final_group = "ALPHA_RADAR"
    return candidates


def rank_non_actionable_pullback_scout(
    pool: pd.DataFrame,
    *,
    version: str = "v3",
) -> list[ReasonedCandidate]:
    candidates = [
        item
        for item in rank_reasoning_candidates(pool, universe="review", version=version)
        if item.entry_status != "ACTIONABLE" and _non_actionable_alpha_lane_v3(item) == "constructive_pullback_scout"
    ]
    candidates.sort(key=_pullback_scout_key_v3)
    for rank, item in enumerate(candidates, 1):
        item.raw_rank = rank
        item.final_group = "ALPHA_RADAR"
    return candidates


def rank_signal_shadow_top3(
    pool: pd.DataFrame,
    *,
    version: str = "v3",
    limit: int = 3,
    industry_cap: bool = True,
) -> list[ReasonedCandidate]:
    """Qlib-inspired audit layer over all signal rows, without changing official picks."""
    ranked = rank_reasoning_candidates(pool, universe="review", version=version)
    selected: list[ReasonedCandidate] = []
    covered: set[str] = set()
    for item in ranked:
        if _has_clear_failure(item):
            continue
        industry_key = str(item.industry or "").strip().lower()
        if industry_cap and industry_key and industry_key in covered:
            continue
        item.final_group = "SIGNAL_SHADOW"
        selected.append(item)
        if industry_cap and industry_key:
            covered.add(industry_key)
        if len(selected) >= limit:
            break
    for rank, item in enumerate(selected, 1):
        item.raw_rank = rank
    return selected


def rank_shadow_portfolio_top3(
    pool: pd.DataFrame,
    *,
    version: str = "v3",
    limit: int = 3,
) -> list[ReasonedCandidate]:
    """Portfolio audit layer for the current best ACTIONABLE shadow rule."""
    selected = [
        item
        for item in rank_reasoning_candidates(pool, universe="actionable", version=version)
        if _is_shadow_portfolio_candidate(item)
    ]
    selected.sort(key=_shadow_portfolio_key)
    selected = selected[:limit]
    for rank, item in enumerate(selected, 1):
        item.raw_rank = rank
        item.final_group = "SHADOW_PORTFOLIO"
    return selected


def build_reasoning_skill_picks(
    pool: pd.DataFrame,
    *,
    snapshot_date: str,
    version: str = "v1",
    priority_limit: int = 3,
    radar_limit: int = 5,
) -> list[ReasonedCandidate]:
    ranked = rank_reasoning_candidates(pool, universe="review", version=version)
    selected_priority = 0
    selected_radar = 0
    for item in ranked:
        if item.entry_status == "ACTIONABLE" and not _has_clear_failure(item) and selected_priority < priority_limit:
            item.final_group = "PRIORITY"
            selected_priority += 1
        elif selected_radar < radar_limit and _is_radar_worthy(item):
            item.final_group = "ALPHA_RADAR"
            selected_radar += 1
        else:
            item.final_group = "OTHER"
        item.snapshot_date = snapshot_date
    return ranked


def _reasoned_item(row: pd.Series, row_idx: int, *, version: str) -> ReasonedCandidate:
    code = str(row.get("code", "") or "").strip()
    status = _entry_status(row)
    industry = str(row.get("industry", "") or "").strip()
    cur = to_float(row.get("current_vs_ibd_candidate_pct"))
    entry_vol = to_float(row.get("ibd_entry_volume_ratio"))
    weekly_vol = to_float(row.get("volume_ratio"))
    eps = to_float(row.get("eps_yoy_growth"))
    if eps is None:
        snap = str(row.get("snapshot_date", "")).strip()
        if snap and code:
            try:
                from eps_pit.lookup import get_signal_eps
                eps = get_signal_eps(snap, code)
            except Exception:
                pass
    dist = to_float(row.get("dist_to_52w_high_pct"))
    rule = str(row.get("ibd_candidate_rule", "") or "").strip()

    reason_codes: list[str] = []
    risk_codes: list[str] = []

    clear_failure = _clear_geometry_failure(row, version=version)
    if clear_failure:
        risk_codes.append("clear_geometry_failure")
    elif _geometry_caution(row):
        reason_codes.append("geometry_caution_not_failure")

    if status != "ACTIONABLE":
        risk_codes.append("non_actionable_radar_only")
    if cur is None:
        risk_codes.append("freshness_missing")
    elif cur < 0:
        risk_codes.append("below_candidate_buy_point")
    elif cur <= 5:
        reason_codes.append("near_buy_point")
    else:
        risk_codes.append("extended_from_buy_point")

    if entry_vol is None:
        risk_codes.append("entry_volume_missing")
    elif entry_vol >= 1.5:
        reason_codes.append("volume_confirms_breakout")
    else:
        risk_codes.append("entry_volume_below_standard")

    if eps is None:
        reason_codes.append("eps_needs_manual_check")
    elif eps >= 25:
        reason_codes.append("eps_acceleration_support")

    if weekly_vol is not None and weekly_vol >= 1.3:
        reason_codes.append("weekly_volume_follow_through")
    if dist is not None and dist > -5:
        reason_codes.append("near_52w_high")
    if _is_pullback_rule(rule):
        reason_codes.append("pullback_structure")
        if to_bool(row.get("pullback_v_is_dry")) is True:
            reason_codes.append("dry_pullback")
        elif to_bool(row.get("pullback_v_is_dry")) is False:
            risk_codes.append("pullback_not_dry")

    lane = _lane(rule, reason_codes, risk_codes, status=status, version=version)
    evidence_count = sum(
        code_name in reason_codes
        for code_name in [
            "near_buy_point",
            "volume_confirms_breakout",
            "eps_acceleration_support",
            "weekly_volume_follow_through",
            "near_52w_high",
            "dry_pullback",
        ]
    )
    fresh_bucket = _fresh_bucket(cur)
    status_bucket = 0 if status == "ACTIONABLE" else 1
    risk_count = sum(
        code_name in risk_codes
        for code_name in [
            "non_actionable_radar_only",
            "freshness_missing",
            "below_candidate_buy_point",
            "extended_from_buy_point",
            "entry_volume_missing",
            "entry_volume_below_standard",
            "pullback_not_dry",
        ]
    )
    if version in {"v2", "v3"}:
        sort_key = (
            1 if clear_failure else 0,
            _LANE_ORDER[lane],
            status_bucket if lane != "constructive_pullback" else min(status_bucket, 0),
            -(evidence_count - risk_count),
            risk_count,
            fresh_bucket,
            0 if eps is not None and eps >= 25 else 1,
            0 if weekly_vol is not None and weekly_vol >= 1.3 else 1,
            -(entry_vol or 0.0),
            code,
            row_idx,
        )
    else:
        sort_key = (
            1 if clear_failure else 0,
            _LANE_ORDER[lane],
            status_bucket if lane != "constructive_pullback" else min(status_bucket, 0),
            fresh_bucket,
            -evidence_count,
            -(entry_vol or 0.0),
            0 if eps is not None and eps >= 25 else 1,
            0 if weekly_vol is not None and weekly_vol >= 1.3 else 1,
            code,
            row_idx,
        )
    return ReasonedCandidate(
        snapshot_date=str(row.get("snapshot_date", "") or ""),
        code=code,
        raw_rank=0,
        final_group="OTHER",
        lane=lane,
        entry_status=status,
        industry=industry,
        sort_key=sort_key,
        reason_codes=reason_codes,
        risk_codes=risk_codes,
        feature_values=_feature_values(row),
    )


_LANE_ORDER = {
    "fresh_demand_alpha": 0,
    "constructive_pullback": 1,
    "standard_breakout": 2,
    "incomplete_evidence": 3,
    "tail_risk": 4,
}


def _lane(
    rule: str,
    reason_codes: list[str],
    risk_codes: list[str],
    *,
    status: str = "",
    version: str = "v1",
) -> str:
    if "clear_geometry_failure" in risk_codes:
        return "tail_risk"
    has_fresh_demand = "near_buy_point" in reason_codes and "volume_confirms_breakout" in reason_codes
    has_follow_through = (
        "eps_acceleration_support" in reason_codes or "weekly_volume_follow_through" in reason_codes
    )
    if version == "v3" and status != "ACTIONABLE" and _is_pullback_rule(rule) and "near_buy_point" in reason_codes:
        has_pullback_context = (
            "near_52w_high" in reason_codes
            or "weekly_volume_follow_through" in reason_codes
            or "dry_pullback" in reason_codes
        )
        if has_pullback_context:
            return "constructive_pullback"
    if _is_pullback_rule(rule) and has_fresh_demand and has_follow_through:
        return "constructive_pullback"
    if has_fresh_demand and has_follow_through:
        return "fresh_demand_alpha"
    if has_fresh_demand:
        return "standard_breakout"
    return "incomplete_evidence"


def _is_review_universe(row: pd.Series) -> bool:
    return to_bool(row.get("signal")) is True and bool(str(row.get("ibd_candidate_rule", "") or "").strip())


def _entry_status(row: pd.Series) -> str:
    return str(row.get("ibd_entry_status", "") or "").strip().upper()


def _has_clear_failure(item: ReasonedCandidate) -> bool:
    return "clear_geometry_failure" in item.risk_codes or "below_candidate_buy_point" in item.risk_codes


def _is_shadow_portfolio_candidate(item: ReasonedCandidate) -> bool:
    if item.entry_status != "ACTIONABLE" or _has_clear_failure(item):
        return False
    return (
        "eps_acceleration_support" in item.reason_codes
        and "volume_confirms_breakout" in item.reason_codes
        and "weekly_volume_follow_through" in item.reason_codes
        and "near_buy_point" in item.reason_codes
        and "extended_from_buy_point" not in item.risk_codes
        and "entry_volume_below_standard" not in item.risk_codes
        and "entry_volume_missing" not in item.risk_codes
        and "freshness_missing" not in item.risk_codes
        and "geometry_caution_not_failure" not in item.reason_codes
    )


def _shadow_portfolio_key(item: ReasonedCandidate) -> tuple:
    cur = to_float(item.feature_values.get("current_vs_ibd_candidate_pct"))
    entry_vol = to_float(item.feature_values.get("ibd_entry_volume_ratio")) or 0.0
    weekly_vol = to_float(item.feature_values.get("volume_ratio")) or 0.0
    dist = to_float(item.feature_values.get("dist_to_52w_high_pct"))
    distance_from_high = abs(dist) if dist is not None else 999.0
    return (
        _fresh_bucket(cur),
        cur if cur is not None else 999.0,
        0 if "near_52w_high" in item.reason_codes else 1,
        -entry_vol,
        -weekly_vol,
        distance_from_high,
        item.code,
    )


def _is_radar_worthy(item: ReasonedCandidate) -> bool:
    if "clear_geometry_failure" in item.risk_codes:
        return False
    return item.lane in {"fresh_demand_alpha", "constructive_pullback", "standard_breakout"}


def _is_non_actionable_alpha_worthy(item: ReasonedCandidate, *, version: str) -> bool:
    if "clear_geometry_failure" in item.risk_codes or "below_candidate_buy_point" in item.risk_codes:
        return False
    if version == "v3":
        return _non_actionable_alpha_lane_v3(item) != "other"
    if version == "v2":
        return _non_actionable_alpha_lane_v2(item) != "other"
    evidence = _non_actionable_evidence_count(item)
    has_structure = "pullback_structure" in item.reason_codes or "near_52w_high" in item.reason_codes
    has_confirmation = (
        "eps_acceleration_support" in item.reason_codes
        or "weekly_volume_follow_through" in item.reason_codes
        or "volume_confirms_breakout" in item.reason_codes
    )
    return evidence >= 3 and has_structure and has_confirmation


def _balanced_v2_non_actionable_alpha(candidates: list[ReasonedCandidate]) -> list[ReasonedCandidate]:
    extended = sorted(
        [item for item in candidates if _non_actionable_alpha_lane_v2(item) == "extended_demand_continuation"],
        key=_non_actionable_alpha_key_v2,
    )
    unconfirmed = sorted(
        [item for item in candidates if _non_actionable_alpha_lane_v2(item) == "unconfirmed_watch"],
        key=_non_actionable_alpha_key_v2,
    )
    other = sorted(
        [
            item
            for item in candidates
            if _non_actionable_alpha_lane_v2(item)
            not in {"extended_demand_continuation", "unconfirmed_watch"}
        ],
        key=_non_actionable_alpha_key_v2,
    )
    selected = extended[:5] + unconfirmed[:5]
    if len(selected) < 10:
        selected.extend([item for item in extended[5:] + unconfirmed[5:] + other if item not in selected][: 10 - len(selected)])
    return selected


def _balanced_v3_non_actionable_alpha(candidates: list[ReasonedCandidate]) -> list[ReasonedCandidate]:
    extended = sorted(
        [item for item in candidates if _non_actionable_alpha_lane_v3(item) == "extended_demand_continuation"],
        key=_non_actionable_alpha_key_v2,
    )
    pullback = sorted(
        [item for item in candidates if _non_actionable_alpha_lane_v3(item) == "constructive_pullback_scout"],
        key=_pullback_scout_key_v3,
    )
    unconfirmed = sorted(
        [item for item in candidates if _non_actionable_alpha_lane_v3(item) == "unconfirmed_watch"],
        key=_non_actionable_alpha_key_v2,
    )
    selected = extended[:3] + pullback[:5] + unconfirmed[:2]
    if len(selected) < 10:
        selected.extend([item for item in extended[3:] + pullback[5:] + unconfirmed[2:] if item not in selected][: 10 - len(selected)])
    return selected


def _non_actionable_alpha_lane_v2(item: ReasonedCandidate) -> str:
    has_demand_confirmation = (
        "volume_confirms_breakout" in item.reason_codes or "weekly_volume_follow_through" in item.reason_codes
    )
    if item.entry_status == "EXTENDED" and has_demand_confirmation:
        return "extended_demand_continuation"
    if item.entry_status == "UNCONFIRMED" and (
        "near_52w_high" in item.reason_codes or "pullback_structure" in item.reason_codes
    ):
        return "unconfirmed_watch"
    return "other"


def _non_actionable_alpha_lane_v3(item: ReasonedCandidate) -> str:
    has_demand_confirmation = (
        "volume_confirms_breakout" in item.reason_codes or "weekly_volume_follow_through" in item.reason_codes
    )
    if item.entry_status == "EXTENDED" and has_demand_confirmation:
        return "extended_demand_continuation"
    if item.entry_status == "UNCONFIRMED" and item.lane == "constructive_pullback":
        return "constructive_pullback_scout"
    if item.entry_status == "UNCONFIRMED" and (
        "near_52w_high" in item.reason_codes or "pullback_structure" in item.reason_codes
    ):
        return "unconfirmed_watch"
    return "other"


def _non_actionable_alpha_key_v1(item: ReasonedCandidate) -> tuple:
    risk_count = sum(
        code_name in item.risk_codes
        for code_name in [
            "freshness_missing",
            "below_candidate_buy_point",
            "entry_volume_below_standard",
            "pullback_not_dry",
        ]
    )
    return (
        _NON_ACTIONABLE_STATUS_ORDER.get(item.entry_status, 2),
        0 if "extended_from_buy_point" not in item.risk_codes else 1,
        -_non_actionable_evidence_count(item),
        risk_count,
        0 if "dry_pullback" in item.reason_codes else 1,
        0 if "near_52w_high" in item.reason_codes else 1,
        0 if "eps_acceleration_support" in item.reason_codes else 1,
        0 if "weekly_volume_follow_through" in item.reason_codes else 1,
        item.code,
    )


def _non_actionable_alpha_key_v2(item: ReasonedCandidate) -> tuple:
    entry_vol = to_float(item.feature_values.get("ibd_entry_volume_ratio")) or 0.0
    weekly_vol = to_float(item.feature_values.get("volume_ratio")) or 0.0
    dist = to_float(item.feature_values.get("dist_to_52w_high_pct"))
    distance_from_high = abs(dist) if dist is not None else 999.0
    risk_count = sum(
        code_name in item.risk_codes
        for code_name in [
            "freshness_missing",
            "entry_volume_below_standard",
            "pullback_not_dry",
        ]
    )
    return (
        risk_count,
        -_non_actionable_evidence_count(item),
        -entry_vol,
        -weekly_vol,
        distance_from_high,
        item.code,
    )


def _pullback_scout_key_v3(item: ReasonedCandidate) -> tuple:
    weekly_vol = to_float(item.feature_values.get("volume_ratio")) or 0.0
    cur = to_float(item.feature_values.get("current_vs_ibd_candidate_pct"))
    dist = to_float(item.feature_values.get("dist_to_52w_high_pct"))
    distance_from_high = abs(dist) if dist is not None else 999.0
    return (
        _fresh_bucket(cur),
        0 if str(item.feature_values.get("ibd_candidate_rule") or "").strip() == "ceiling_pullback" else 1,
        0 if "near_52w_high" in item.reason_codes else 1,
        -_non_actionable_evidence_count(item),
        -weekly_vol,
        distance_from_high,
        item.code,
    )


_NON_ACTIONABLE_STATUS_ORDER = {
    "UNCONFIRMED": 0,
    "EXTENDED": 1,
}


def _non_actionable_evidence_count(item: ReasonedCandidate) -> int:
    return sum(
        code_name in item.reason_codes
        for code_name in [
            "near_buy_point",
            "volume_confirms_breakout",
            "eps_acceleration_support",
            "weekly_volume_follow_through",
            "near_52w_high",
            "pullback_structure",
            "dry_pullback",
        ]
    )


def _fresh_bucket(cur: float | None) -> int:
    if cur is None:
        return 3
    if 0 <= cur <= 2:
        return 0
    if 2 < cur <= 5:
        return 1
    if cur > 5:
        return 2
    return 4


def _is_pullback_rule(rule: str) -> bool:
    return rule in {"ceiling_pullback", "pivot", "ma10_touch_confirm", "three_weeks_tight"}


def _clear_geometry_failure(row: pd.Series, *, version: str = "v1") -> bool:
    pos = to_float(row.get("ibd_entry_close_position"))
    rr = to_float(row.get("ibd_entry_breakout_range_ratio"))
    if rr is not None and rr <= 0:
        return True
    min_close_position = 0.65 if version in {"v2", "v3"} else 0.50
    if pos is not None and pos < min_close_position:
        return True
    return False


def _geometry_caution(row: pd.Series) -> bool:
    pos = to_float(row.get("ibd_entry_close_position"))
    rr = to_float(row.get("ibd_entry_breakout_range_ratio"))
    if pos is None or rr is None:
        return True
    if 0.50 <= pos < 0.65:
        return True
    trigger_pos = pos - rr
    return 0.65 <= pos < 0.80 and trigger_pos > 0 and rr < 0.50


def _feature_values(row: pd.Series) -> dict[str, object]:
    columns = [
        "ibd_candidate_price",
        "latest_close",
        "ibd_candidate_rule",
        "current_vs_ibd_candidate_pct",
        "ibd_entry_volume_ratio",
        "ibd_entry_close_position",
        "ibd_entry_breakout_range_ratio",
        "ibd_entry_close_vs_trigger_pct",
        "dist_to_52w_high_pct",
        "volume_ratio",
        "eps_yoy_growth",
        "base_depth_pct",
        "base_duration_weeks",
        "pullback_pct",
        "pullback_duration_weeks",
        "pullback_v_is_dry",
        "sector",
        "industry",
    ]
    return {column: row.get(column) for column in columns}
