from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from dashboard.skill_industry_eps_known import (
    is_review_universe,
    reasoned_item,
    to_float,
)
from backtest.track_c_ranking_discovery.b0_ablation_grid import StructuralGridChallenger
from backtest.track_c_ranking_discovery.discovery_sandbox.discovery_runner import controlled_eligible

from .config import BASELINE_POLICY_ID, POLICY_SPECS, QUALITY_ORDER, TOP_N
from .taxonomy import classify_lane_facts


@dataclass(frozen=True)
class CompositionSpec:
    policy_id: str
    mode: str
    industry_policy: str
    role: str


def composition_specs() -> list[CompositionSpec]:
    return [
        CompositionSpec(
            policy_id=policy_id,
            mode=mode,
            industry_policy=industry_policy,
            role=role,
        )
        for policy_id, mode, industry_policy, role in POLICY_SPECS
    ]


def production_baseline() -> StructuralGridChallenger:
    return StructuralGridChallenger("B0_LANE", "symmetric", "distinct_1")


def legacy_pullback_parity_anchor() -> StructuralGridChallenger:
    return StructuralGridChallenger("PULLBACK_PARITY", "symmetric", "distinct_1")


def _evidence_risk(item) -> tuple[int, int]:
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
    return evidence_count, risk_count


def normalized_sort_key(item, quality_state: str) -> tuple[Any, ...]:
    """Route-neutral ranking inside the orthogonal quality taxonomy.

    Production B0 sort key is:
      failure -> Lane -> status -> evidence/risk -> freshness -> EPS -> weekly vol -> entry vol.

    Track F removes route from the quality ordering and replaces current Lane priority
    with quality_state only. All remaining B0 tie-breaks are retained exactly.
    """
    base = item.sort_key
    if len(base) < 11:
        raise RuntimeError(f"Unexpected Production B0 sort key shape for {item.code}: {base}")
    return (
        base[0],
        QUALITY_ORDER[quality_state],
        base[2],
        base[3],
        base[4],
        *base[5:],
    )


class LaneCompositionPolicy:
    family = "track_f_lane_composition"
    fitted_state_hash = "none"

    def __init__(self, spec: CompositionSpec):
        self.spec = spec
        self.policy_id = f"TRACK_F__{spec.policy_id}"
        self.spec_hash = (
            f"{self.policy_id}__{spec.mode}__{spec.industry_policy}__"
            "production_thresholds__orthogonal_quality_route"
        )

    def score_candidates(self, snapshot_df: pd.DataFrame) -> pd.DataFrame:
        if snapshot_df.empty:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for row_idx, (_, row) in enumerate(snapshot_df.iterrows()):
            if not is_review_universe(row):
                continue

            item = reasoned_item(row, row_idx)
            facts = classify_lane_facts(row, row_idx)
            evidence_count, risk_count = _evidence_risk(item)

            rows.append({
                "code": item.code,
                "industry": item.industry,
                "industry_key": str(item.industry or "").strip().lower(),
                "b0_eligible": row.get("b0_eligible", False),
                "current_lane": item.lane,
                "entry_status": item.entry_status,
                "setup_route": facts.setup_route,
                "fresh_demand": facts.fresh_demand,
                "follow_through": facts.follow_through,
                "quality_state": facts.quality_state,
                "composition_group": facts.composition_group,
                "normalized_sort_key": normalized_sort_key(item, facts.quality_state),
                "evidence_count": evidence_count,
                "risk_count": risk_count,
                "effective_eps": to_float(item.feature_values.get("effective_eps_yoy_growth")),
            })

        scored = pd.DataFrame(rows)
        if scored.empty:
            return scored

        scored = scored.sort_values("normalized_sort_key", kind="stable").reset_index(drop=True)
        scored["raw_rank"] = range(1, len(scored) + 1)
        return scored

    def allocate_industries(self, scored_df: pd.DataFrame) -> dict[str, int]:
        if scored_df.empty:
            return {}
        eligible = controlled_eligible(scored_df)
        industries = (
            eligible["industry_key"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
        )
        quota = 1 if self.spec.industry_policy == "distinct_1" else TOP_N
        return {ind: quota for ind in industries if ind}

    def _mode_filter(self, eligible: pd.DataFrame) -> pd.DataFrame:
        if self.spec.mode == "parity_fallback":
            return eligible
        if self.spec.mode == "confirmed_only":
            return eligible[eligible["quality_state"] == "confirmed"].copy()
        if self.spec.mode == "fcs_max1":
            return eligible[
                eligible["composition_group"].isin(
                    ["confirmed_non_pullback", "confirmed_pullback", "standard"]
                )
            ].copy()
        raise RuntimeError(f"Unknown Track F composition mode: {self.spec.mode}")

    def pick_stocks(self, scored_df: pd.DataFrame, industry_quotas: dict[str, int]) -> list[str]:
        if scored_df.empty or not industry_quotas:
            return []

        eligible = controlled_eligible(scored_df).sort_values("raw_rank", kind="stable")
        eligible = self._mode_filter(eligible)

        selected: list[str] = []
        industry_counts: dict[str, int] = {}
        group_counts: dict[str, int] = {}

        for _, row in eligible.iterrows():
            if len(selected) >= TOP_N:
                break

            industry = str(row["industry_key"]).strip().lower()
            if industry_counts.get(industry, 0) >= int(industry_quotas.get(industry, 0)):
                continue

            group = str(row["composition_group"])
            if self.spec.mode == "fcs_max1" and group_counts.get(group, 0) >= 1:
                continue

            selected.append(str(row["code"]))
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
            group_counts[group] = group_counts.get(group, 0) + 1

        return selected


def all_policies() -> list[LaneCompositionPolicy]:
    return [LaneCompositionPolicy(spec) for spec in composition_specs()]
