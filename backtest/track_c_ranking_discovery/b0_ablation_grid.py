from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import Any
import pandas as pd
from dashboard.skill_industry_eps_known import (
    SkillCandidate,
    clear_geometry_failure,
    entry_status,
    geometry_caution,
    is_pullback_rule,
    is_review_universe,
    lane_for,
    rank_skill_industry_eps_known,
    reasoned_item,
    row_eps,
    select_skill_industry_eps_known,
    to_bool,
    to_float,
    LANE_ORDER,
)
from .config import TOP_N, LANE_POLICIES, DRY_POLICIES, SELECTOR_POLICIES


def compute_structural_sort_key(
    item: SkillCandidate,
    lane_policy: str,
) -> tuple:
    """Compute explicit pre-registered sort key permutation for structural grid."""
    # Production original sort key elements:
    # 0: clear_geometry_failure (0 or 1)
    # 1: lane_priority (0..4)
    # 2: status_rank (0=ACTIONABLE, 1=RADAR, 2=EXTENDED, 3=UNKNOWN)
    # 3: -evidence_risk_balance (- (evidence_count - risk_count))
    # 4: risk_count
    # 5: freshness_rank
    # 6: -effective_eps
    # 7: -entry_vol_ratio
    # 8: -vol_ratio
    # 9: -close_position
    # 10: -range_ratio
    # 11: dist_to_52w_high
    # 12: row_idx

    fail = item.sort_key[0]
    orig_lane = item.sort_key[1]
    status = item.sort_key[2]
    ev_risk_bal = item.sort_key[3]
    risk_cnt = item.sort_key[4]
    rest = item.sort_key[5:]

    if lane_policy == "B0_LANE":
        eff_lane = orig_lane
        return (fail, eff_lane, status, ev_risk_bal, risk_cnt, *rest)

    elif lane_policy == "PULLBACK_PARITY":
        # Parity: fresh_demand_alpha (0) and constructive_pullback (1) both get priority 0
        eff_lane = 0 if orig_lane in (0, 1) else (orig_lane - 1)
        return (fail, eff_lane, status, ev_risk_bal, risk_cnt, *rest)

    elif lane_policy == "LANE_NEUTRAL":
        # Neutral: lane is calculated and recorded, but assigns equal priority 0 to all lanes
        eff_lane = 0
        return (fail, eff_lane, status, ev_risk_bal, risk_cnt, *rest)

    elif lane_policy == "SCORE_BEFORE_LANE":
        # Permutation: (fail, status, ev_risk_bal, risk_cnt, lane, *rest)
        return (fail, status, ev_risk_bal, risk_cnt, orig_lane, *rest)

    elif lane_policy == "PULLBACK_FIRST":
        # Stress test: pullback (0) > fresh (1) > standard (2) > incomplete (3) > tail (4)
        eff_lane = 0 if orig_lane == 1 else (1 if orig_lane == 0 else orig_lane)
        return (fail, eff_lane, status, ev_risk_bal, risk_cnt, *rest)

    else:
        raise ValueError(f"Unknown lane policy: {lane_policy}")


def reasoned_item_variant(
    row: pd.Series,
    row_idx: int,
    dry_policy: str,
    lane_policy: str,
) -> SkillCandidate:
    """Evaluate candidate reasoning with minimal diff on dry_policy and lane_policy."""
    # First get production reasoned item
    base_item = reasoned_item(row, row_idx)

    # Apply dry_policy modification if not symmetric
    reason_codes = list(base_item.reason_codes)
    risk_codes = list(base_item.risk_codes)
    rule = str(row.get("ibd_candidate_rule", "") or "").strip()

    if is_pullback_rule(rule):
        dry = to_bool(row.get("pullback_v_is_dry"))
        if dry_policy == "reward_only":
            if "pullback_not_dry" in risk_codes:
                risk_codes.remove("pullback_not_dry")
        elif dry_policy == "ignored":
            if "dry_pullback" in reason_codes:
                reason_codes.remove("dry_pullback")
            if "pullback_not_dry" in risk_codes:
                risk_codes.remove("pullback_not_dry")

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

    lane = lane_for(rule, reason_codes, risk_codes, status=base_item.entry_status)
    from dashboard.skill_industry_eps_known import LANE_ORDER
    orig_lane_priority = LANE_ORDER.get(lane, 4)

    # Reconstruct raw sort key
    fail = base_item.sort_key[0]
    status_r = base_item.sort_key[2]
    rest = base_item.sort_key[5:]

    raw_sort_key = (
        fail,
        orig_lane_priority,
        status_r,
        -(evidence_count - risk_count),
        risk_count,
        *rest,
    )

    temp_item = SkillCandidate(
        code=base_item.code,
        raw_rank=base_item.raw_rank,
        entry_status=base_item.entry_status,
        lane=lane,
        industry=base_item.industry,
        sort_key=raw_sort_key,
        reason_codes=reason_codes,
        risk_codes=risk_codes,
        feature_values=base_item.feature_values,
    )

    final_sort_key = compute_structural_sort_key(temp_item, lane_policy)
    temp_item.sort_key = final_sort_key
    return temp_item


class StructuralGridChallenger:
    """Pre-registered Structural Grid Challenger combining Lane, Dry, and Industry policies."""

    def __init__(self, lane_policy: str, dry_policy: str, selector_policy: str):
        self.lane_policy = lane_policy
        self.dry_policy = dry_policy
        self.selector_policy = selector_policy
        self.family = "structural"
        self.policy_id = f"STRUCTURAL__{lane_policy}__{dry_policy}__{selector_policy}"
        self.spec_hash = f"structural_{lane_policy}_{dry_policy}_{selector_policy}"
        self.fitted_state_hash = "none"

    def score_candidates(self, snapshot_df: pd.DataFrame) -> pd.DataFrame:
        """Layer 1: Score & rank candidates using the structural reasoning rule."""
        if snapshot_df.empty:
            return pd.DataFrame()

        # If baseline original, passthrough directly to production ranker
        if self.lane_policy == "B0_LANE" and self.dry_policy == "symmetric":
            items = rank_skill_industry_eps_known(snapshot_df)
        else:
            candidates = []
            for row_idx, (_, row) in enumerate(snapshot_df.iterrows()):
                if not is_review_universe(row):
                    continue
                item = reasoned_item_variant(row, row_idx, self.dry_policy, self.lane_policy)
                candidates.append(item)
            candidates.sort(key=lambda x: x.sort_key)
            for i, c in enumerate(candidates, 1):
                c.raw_rank = i
            items = candidates

        rows = []
        for it in items:
            eps_val = to_float(it.feature_values.get("effective_eps_yoy_growth"))
            has_eps = int(eps_val is not None)
            ind_clean = str(it.industry or "").strip().lower()
            has_ind = int(bool(ind_clean))
            rows.append({
                "code": it.code,
                "industry": it.industry,
                "industry_key": ind_clean,
                "lane": it.lane,
                "raw_rank": it.raw_rank,
                "sort_key": str(it.sort_key),
                "is_actionable": int(it.entry_status == "ACTIONABLE"),
                "has_geom_failure": int("clear_geometry_failure" in it.risk_codes),
                "below_buy_point": int("below_candidate_buy_point" in it.risk_codes),
                "has_known_eps": has_eps,
                "has_valid_industry": has_ind,
            })
        return pd.DataFrame(rows)

    def allocate_industries(self, scored_df: pd.DataFrame) -> dict[str, int]:
        """Layer 2: Allocate industry quotas based on selector policy."""
        if scored_df.empty:
            return {}

        # Filter to actionable eligible candidates
        eligible = scored_df[
            (scored_df.is_actionable == 1) &
            (scored_df.has_geom_failure == 0) &
            (scored_df.below_buy_point == 0) &
            (scored_df.has_known_eps == 1) &
            (scored_df.has_valid_industry == 1)
        ].copy()

        if eligible.empty:
            return {}

        industries = eligible["industry_key"].unique()
        if self.selector_policy == "distinct_1":
            return {ind: 1 for ind in industries}
        elif self.selector_policy == "max_2_per_ind":
            return {ind: 2 for ind in industries}
        elif self.selector_policy == "pure_top3":
            return {ind: TOP_N for ind in industries}
        else:
            return {ind: 1 for ind in industries}

    def pick_stocks(self, scored_df: pd.DataFrame, industry_quotas: dict[str, int]) -> list[str]:
        """Layer 3: Pick up to TOP_N stocks respecting Layer 2 industry quotas."""
        if scored_df.empty or not industry_quotas:
            return []

        eligible = scored_df[
            (scored_df.is_actionable == 1) &
            (scored_df.has_geom_failure == 0) &
            (scored_df.below_buy_point == 0) &
            (scored_df.has_known_eps == 1) &
            (scored_df.has_valid_industry == 1)
        ].sort_values("raw_rank")

        selected = []
        ind_counts: dict[str, int] = {}

        for _, row in eligible.iterrows():
            if len(selected) >= TOP_N:
                break
            code = str(row["code"])
            ind_key = str(row["industry_key"])
            quota = industry_quotas.get(ind_key, 0)
            cur_cnt = ind_counts.get(ind_key, 0)

            if cur_cnt < quota:
                selected.append(code)
                ind_counts[ind_key] = cur_cnt + 1

        return selected


def get_structural_grid_challengers() -> dict[str, StructuralGridChallenger]:
    """Get all 36 pre-registered mandatory factorial structural challengers."""
    challengers = {}
    for lp in LANE_POLICIES:
        for dp in DRY_POLICIES:
            for sp in SELECTOR_POLICIES:
                c = StructuralGridChallenger(lp, dp, sp)
                challengers[c.policy_id] = c
    return challengers
