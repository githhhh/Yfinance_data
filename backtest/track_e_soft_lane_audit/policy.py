from __future__ import annotations

from typing import Any

import pandas as pd

from dashboard.skill_industry_eps_known import (
    SkillCandidate,
    is_review_universe,
    to_float,
)
from backtest.track_c_ranking_discovery.b0_ablation_grid import (
    StructuralGridChallenger,
    reasoned_item_variant,
)

from .config import CHALLENGER_POLICY_ID, TARGET_SOFT_LANES


def target_pair_sort_key(item: SkillCandidate) -> tuple[Any, ...]:
    """Evidence/risk-first comparator used only inside fresh/standard target slots."""
    base = item.sort_key
    if len(base) < 9:
        raise RuntimeError(f"Unexpected B0 sort key shape for {item.code}: {base}")
    return (
        base[2],          # status
        base[3],          # -(evidence_count-risk_count)
        base[4],          # risk_count
        base[1],          # original Lane becomes only a tie-break
        *base[5:],        # remaining B0 tie-breaks
    )


def reorder_target_lanes(base_items: list[SkillCandidate]) -> list[SkillCandidate]:
    """Keep the reward-only-B0 rank skeleton fixed except for fresh/standard slots.

    Non-target items keep their exact absolute position. Only candidates occupying
    fresh_demand_alpha or standard_breakout slots are permuted among those same
    target slots according to target_pair_sort_key().
    """
    ordered = sorted(base_items, key=lambda x: x.sort_key)
    target_positions = [
        idx for idx, item in enumerate(ordered)
        if item.lane in TARGET_SOFT_LANES
    ]
    if len(target_positions) <= 1:
        return ordered

    target_items = [ordered[idx] for idx in target_positions]
    target_items.sort(key=target_pair_sort_key)

    out = list(ordered)
    for idx, item in zip(target_positions, target_items):
        out[idx] = item
    return out


class PairwiseSoftLaneChallenger(StructuralGridChallenger):
    """B0.1: dry=False neutral + isolated fresh-vs-standard soft ordering."""

    def __init__(self) -> None:
        super().__init__(
            lane_policy="B0_LANE",
            dry_policy="reward_only",
            selector_policy="distinct_1",
        )
        self.policy_id = CHALLENGER_POLICY_ID
        self.family = "track_e_pairwise_soft_lane"
        self.spec_hash = (
            "dry_true_reward__dry_false_neutral__"
            "reward_only_b0_skeleton__fresh_standard_slots_only__"
            "status_evidence_risk_before_lane__distinct1"
        )

    def score_candidates(self, snapshot_df: pd.DataFrame) -> pd.DataFrame:
        if snapshot_df.empty:
            return pd.DataFrame()

        base_items: list[SkillCandidate] = []
        for row_idx, (_, row) in enumerate(snapshot_df.iterrows()):
            if not is_review_universe(row):
                continue
            item = reasoned_item_variant(
                row,
                row_idx,
                dry_policy="reward_only",
                lane_policy="B0_LANE",
            )
            base_items.append(item)

        skeleton = sorted(base_items, key=lambda x: x.sort_key)
        skeleton_rank = {item.code: idx + 1 for idx, item in enumerate(skeleton)}
        candidates = reorder_target_lanes(base_items)

        for rank, item in enumerate(candidates, 1):
            item.raw_rank = rank

        rows: list[dict[str, Any]] = []
        for item in candidates:
            eps_val = to_float(item.feature_values.get("effective_eps_yoy_growth"))
            industry_key = str(item.industry or "").strip().lower()
            evidence_count = sum(
                code in item.reason_codes
                for code in [
                    "near_buy_point",
                    "volume_confirms_breakout",
                    "eps_acceleration_support",
                    "weekly_volume_follow_through",
                    "near_52w_high",
                    "dry_pullback",
                ]
            )
            risk_count = sum(
                code in item.risk_codes
                for code in [
                    "non_actionable_radar_only",
                    "freshness_missing",
                    "below_candidate_buy_point",
                    "extended_from_buy_point",
                    "entry_volume_missing",
                    "entry_volume_below_standard",
                    "pullback_not_dry",
                ]
            )
            rows.append({
                "code": item.code,
                "industry": item.industry,
                "industry_key": industry_key,
                "lane": item.lane,
                "raw_rank": item.raw_rank,
                "skeleton_rank": int(skeleton_rank[item.code]),
                "sort_key": str(item.sort_key),
                "evidence_count": evidence_count,
                "risk_count": risk_count,
                "evidence_balance": evidence_count - risk_count,
                "reason_codes": "|".join(item.reason_codes),
                "risk_codes": "|".join(item.risk_codes),
                "is_actionable": int(item.entry_status == "ACTIONABLE"),
                "has_geom_failure": int("clear_geometry_failure" in item.risk_codes),
                "below_buy_point": int("below_candidate_buy_point" in item.risk_codes),
                "has_known_eps": int(eps_val is not None),
                "has_valid_industry": int(bool(industry_key)),
            })
        return pd.DataFrame(rows)


# Compatibility alias for earlier Track E imports; semantics are v2 pairwise isolation.
SoftActiveLaneChallenger = PairwiseSoftLaneChallenger


def baseline_policy() -> StructuralGridChallenger:
    return StructuralGridChallenger("B0_LANE", "symmetric", "distinct_1")
