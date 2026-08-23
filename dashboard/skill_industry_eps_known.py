from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class SkillCandidate:
    code: str
    raw_rank: int
    entry_status: str
    lane: str
    industry: str
    sort_key: tuple
    reason_codes: list[str] = field(default_factory=list)
    risk_codes: list[str] = field(default_factory=list)
    feature_values: dict[str, object] = field(default_factory=dict)


LANE_ORDER = {
    "fresh_demand_alpha": 0,
    "constructive_pullback": 1,
    "standard_breakout": 2,
    "incomplete_evidence": 3,
    "tail_risk": 4,
}


def select_skill_industry_eps_known(pool: pd.DataFrame, *, limit: int = 3) -> list[SkillCandidate]:
    ranked = rank_skill_industry_eps_known(pool)
    selected: list[SkillCandidate] = []
    covered: set[str] = set()
    for item in ranked:
        if item.entry_status != "ACTIONABLE":
            continue
        if "clear_geometry_failure" in item.risk_codes:
            continue
        if "below_candidate_buy_point" in item.risk_codes:
            continue
        if effective_eps(item) is None:
            continue
        industry_key = item.industry.strip().lower()
        if not industry_key:
            continue
        if industry_key in covered:
            continue
        selected.append(item)
        covered.add(industry_key)
        if len(selected) >= limit:
            break
    return selected


def rank_skill_industry_eps_known(pool: pd.DataFrame) -> list[SkillCandidate]:
    candidates = []
    for row_idx, row in pool.iterrows():
        if not is_review_universe(row):
            continue
        candidates.append(reasoned_item(row, row_idx))
    candidates.sort(key=lambda item: item.sort_key)
    for rank, item in enumerate(candidates, 1):
        item.raw_rank = rank
    return candidates


def reasoned_item(row: pd.Series, row_idx: int) -> SkillCandidate:
    code = str(row.get("code", "") or "").strip()
    status = entry_status(row)
    industry = str(row.get("industry", "") or "").strip()
    cur = to_float(row.get("current_vs_ibd_candidate_pct"))
    entry_vol = to_float(row.get("ibd_entry_volume_ratio"))
    weekly_vol = to_float(row.get("volume_ratio"))
    eps = row_eps(row, code)
    dist = to_float(row.get("dist_to_52w_high_pct"))
    rule = str(row.get("ibd_candidate_rule", "") or "").strip()

    reason_codes: list[str] = []
    risk_codes: list[str] = []

    clear_failure = clear_geometry_failure(row)
    if clear_failure:
        risk_codes.append("clear_geometry_failure")
    elif geometry_caution(row):
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
    if is_pullback_rule(rule):
        reason_codes.append("pullback_structure")
        dry = to_bool(row.get("pullback_v_is_dry"))
        if dry is True:
            reason_codes.append("dry_pullback")
        elif dry is False:
            risk_codes.append("pullback_not_dry")

    lane = lane_for(rule, reason_codes, risk_codes, status=status)
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
    status_bucket = 0 if status == "ACTIONABLE" else 1
    sort_key = (
        1 if clear_failure else 0,
        LANE_ORDER[lane],
        status_bucket if lane != "constructive_pullback" else min(status_bucket, 0),
        -(evidence_count - risk_count),
        risk_count,
        fresh_bucket(cur),
        0 if eps is not None and eps >= 25 else 1,
        0 if weekly_vol is not None and weekly_vol >= 1.3 else 1,
        -(entry_vol or 0.0),
        code,
        row_idx,
    )
    return SkillCandidate(
        code=code,
        raw_rank=0,
        entry_status=status,
        lane=lane,
        industry=industry,
        sort_key=sort_key,
        reason_codes=reason_codes,
        risk_codes=risk_codes,
        feature_values=feature_values(row, eps),
    )


def is_review_universe(row: pd.Series) -> bool:
    return to_bool(row.get("signal")) is True and bool(str(row.get("ibd_candidate_rule", "") or "").strip())


def entry_status(row: pd.Series) -> str:
    return str(row.get("ibd_entry_status", "") or "").strip().upper()


def row_eps(row: pd.Series, code: str) -> float | None:
    eps = to_float(row.get("eps_yoy_growth"))
    if eps is not None:
        return eps
    snapshot = str(row.get("snapshot_date", "") or "").strip()
    if not snapshot or not code:
        return None
    try:
        from eps_pit.lookup import get_signal_eps

        return get_signal_eps(snapshot, code)
    except Exception:
        return None


def effective_eps(item: SkillCandidate) -> float | None:
    return to_float(item.feature_values.get("effective_eps_yoy_growth"))


def lane_for(rule: str, reason_codes: list[str], risk_codes: list[str], *, status: str) -> str:
    if "clear_geometry_failure" in risk_codes:
        return "tail_risk"
    has_fresh_demand = "near_buy_point" in reason_codes and "volume_confirms_breakout" in reason_codes
    has_follow_through = "eps_acceleration_support" in reason_codes or "weekly_volume_follow_through" in reason_codes
    if status != "ACTIONABLE" and is_pullback_rule(rule) and "near_buy_point" in reason_codes:
        has_pullback_context = (
            "near_52w_high" in reason_codes
            or "weekly_volume_follow_through" in reason_codes
            or "dry_pullback" in reason_codes
        )
        if has_pullback_context:
            return "constructive_pullback"
    if is_pullback_rule(rule) and has_fresh_demand and has_follow_through:
        return "constructive_pullback"
    if has_fresh_demand and has_follow_through:
        return "fresh_demand_alpha"
    if has_fresh_demand:
        return "standard_breakout"
    return "incomplete_evidence"


def clear_geometry_failure(row: pd.Series) -> bool:
    pos = to_float(row.get("ibd_entry_close_position"))
    rr = to_float(row.get("ibd_entry_breakout_range_ratio"))
    if rr is not None and rr <= 0:
        return True
    if pos is not None and pos < 0.65:
        return True
    return False


def geometry_caution(row: pd.Series) -> bool:
    pos = to_float(row.get("ibd_entry_close_position"))
    rr = to_float(row.get("ibd_entry_breakout_range_ratio"))
    if pos is None or rr is None:
        return True
    if 0.50 <= pos < 0.65:
        return True
    trigger_pos = pos - rr
    return 0.65 <= pos < 0.80 and trigger_pos > 0 and rr < 0.50


def fresh_bucket(cur: float | None) -> int:
    if cur is None:
        return 3
    if 0 <= cur <= 2:
        return 0
    if 2 < cur <= 5:
        return 1
    if cur > 5:
        return 2
    return 4


def is_pullback_rule(rule: str) -> bool:
    return rule in {"ceiling_pullback", "pivot", "ma10_touch_confirm", "three_weeks_tight"}


def feature_values(row: pd.Series, eps: float | None) -> dict[str, object]:
    columns = [
        "snapshot_date",
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
    values = {column: row.get(column) for column in columns}
    values["effective_eps_yoy_growth"] = eps
    return values


def to_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "<na>", "nat"}:
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def to_bool(value: object) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0"}:
        return True
    if text in {"false", "0", "0.0"}:
        return False
    return None


def artifact(pool: pd.DataFrame) -> dict[str, Any]:
    selected = select_skill_industry_eps_known(pool)
    ranked = rank_skill_industry_eps_known(pool)
    return {
        "baseline": "skill_industry_eps_known",
        "priority_top3": [candidate_row(item) for item in selected],
        "review_raw_top10": [candidate_row(item) for item in ranked[:10]],
    }


def candidate_row(item: SkillCandidate) -> dict[str, Any]:
    return {
        "raw_rank": item.raw_rank,
        "code": item.code,
        "entry_status": item.entry_status,
        "lane": item.lane,
        "industry": item.industry,
        "reason_codes": item.reason_codes,
        "risk_codes": item.risk_codes,
        "sort_key": list(item.sort_key),
        "fields": item.feature_values,
    }


def json_default(value: object) -> object:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run deterministic skill_industry_eps_known prescreen.")
    parser.add_argument("--pool", required=True, help="Pool CSV path.")
    parser.add_argument("--codes-only", action="store_true", help="Print comma-separated priority Top3 codes.")
    args = parser.parse_args(argv)

    pool = pd.read_csv(Path(args.pool), encoding="utf-8-sig")
    result = artifact(pool)
    if args.codes_only:
        print(",".join(row["code"] for row in result["priority_top3"]))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
