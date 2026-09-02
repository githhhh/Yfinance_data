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

from .config import ACTIVE_SOFT_LANES, CHALLENGER_POLICY_ID


def soft_active_lane_sort_key(item: SkillCandidate) -> tuple[Any, ...]:
    """Soften only the three investable lanes while preserving downgrade semantics.

    Production B0 orders lane before evidence/risk. B0.1 instead:
      clear failure
      -> active-vs-incomplete/tail guard
      -> status
      -> evidence/risk
      -> original active-lane prior
      -> remaining B0 tie-breaks

    Therefore a stronger standard_breakout may outrank a weaker
    fresh_demand_alpha, but incomplete_evidence remains below every active lane.
    tail_risk remains protected by both clear-failure and lane guard.
    """
    base = item.sort_key
    if len(base) < 9:
        raise RuntimeError(f"Unexpected B0 sort key shape for {item.code}: {base}")

    lane_guard = 0 if item.lane in ACTIVE_SOFT_LANES else (
        1 if item.lane == "incomplete_evidence" else 2
    )
    return (
        base[0],          # clear_geometry_failure
        lane_guard,       # active lanes always above incomplete/tail
        base[2],          # status
        base[3],          # -(evidence_count-risk_count)
        base[4],          # risk_count
        base[1],          # original lane is now only a prior/tie-break
        *base[5:],        # freshness/EPS/weekly vol/entry vol/code/row_idx
    )


class SoftActiveLaneChallenger(StructuralGridChallenger):
    """Single B0.1 challenger: dry=False neutral + soft active-lane hierarchy."""

    def __init__(self) -> None:
        super().__init__(
            lane_policy="B0_LANE",
            dry_policy="reward_only",
            selector_policy="distinct_1",
        )
        self.policy_id = CHALLENGER_POLICY_ID
        self.family = "track_e_soft_lane"
        self.spec_hash = (
            "dry_true_reward__dry_false_neutral__"
            "active_lane_guard__evidence_risk_before_lane__distinct1"
        )

    def score_candidates(self, snapshot_df: pd.DataFrame) -> pd.DataFrame:
        if snapshot_df.empty:
            return pd.DataFrame()

        candidates: list[SkillCandidate] = []
        for row_idx, (_, row) in enumerate(snapshot_df.iterrows()):
            if not is_review_universe(row):
                continue
            item = reasoned_item_variant(
                row,
                row_idx,
                dry_policy="reward_only",
                lane_policy="B0_LANE",
            )
            item.sort_key = soft_active_lane_sort_key(item)
            candidates.append(item)

        candidates.sort(key=lambda x: x.sort_key)
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


def baseline_policy() -> StructuralGridChallenger:
    return StructuralGridChallenger("B0_LANE", "symmetric", "distinct_1")
