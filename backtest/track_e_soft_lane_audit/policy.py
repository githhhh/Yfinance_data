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

from .config import (
    BASELINE_POLICY_ID,
    CHALLENGER_POLICY_ID,
    TARGET_FRESH_LANE,
    TARGET_STANDARD_LANE,
)


def target_strength_key(row: pd.Series | dict[str, Any]) -> tuple[Any, ...]:
    """Deterministic ordering among pairwise candidates; lower is better.

    This key is only used to choose among already-Pareto-valid candidates.
    It excludes Lane-defining follow-through evidence so the experiment does
    not tautologically favor fresh_demand_alpha by construction.
    """
    get = row.get
    risk_count = int(get("risk_count", 0))
    cur = to_float(get("current_vs_ibd_candidate_pct"))
    entry_vol = to_float(get("ibd_entry_volume_ratio"))
    code = str(get("code", ""))
    return (
        risk_count,
        abs(cur) if cur is not None else float("inf"),
        -(entry_vol if entry_vol is not None else -float("inf")),
        code,
    )


def strictly_stronger_standard(
    standard_row: pd.Series | dict[str, Any],
    fresh_row: pd.Series | dict[str, Any],
) -> bool:
    """Unweighted Pareto dominance on independent strength axes.

    A standard breakout may challenge a selected fresh candidate only if it has:
      - no more B0 risk flags,
      - no worse buy-point distance,
      - no weaker entry volume,
    and is strictly better on at least one of those axes.

    EPS/weekly-volume follow-through is intentionally excluded because it is
    exactly what defines fresh_demand_alpha and would make the test circular.
    """
    s_risk = int(standard_row.get("risk_count", 0))
    f_risk = int(fresh_row.get("risk_count", 0))
    s_cur = to_float(standard_row.get("current_vs_ibd_candidate_pct"))
    f_cur = to_float(fresh_row.get("current_vs_ibd_candidate_pct"))
    s_vol = to_float(standard_row.get("ibd_entry_volume_ratio"))
    f_vol = to_float(fresh_row.get("ibd_entry_volume_ratio"))

    if None in {s_cur, f_cur, s_vol, f_vol}:
        return False

    no_worse = (
        s_risk <= f_risk
        and abs(s_cur) <= abs(f_cur)
        and s_vol >= f_vol
    )
    strictly_better = (
        s_risk < f_risk
        or abs(s_cur) < abs(f_cur)
        or s_vol > f_vol
    )
    return bool(no_worse and strictly_better)


class DryNeutralHardLaneBaseline(StructuralGridChallenger):
    """Adopted dry semantics, otherwise Production B0 hard-Lane behavior."""

    def __init__(self) -> None:
        super().__init__(
            lane_policy="B0_LANE",
            dry_policy="reward_only",
            selector_policy="distinct_1",
        )
        self.policy_id = BASELINE_POLICY_ID
        self.family = "track_e_control"
        self.spec_hash = "dry_true_reward__dry_false_neutral__hard_b0_lane__distinct1"


class PairwiseFreshStandardChallenger(DryNeutralHardLaneBaseline):
    """Only allow an unselected standard_breakout to replace a selected fresh candidate."""

    def __init__(self) -> None:
        super().__init__()
        self.policy_id = CHALLENGER_POLICY_ID
        self.family = "track_e_pairwise_replacement"
        self.spec_hash = (
            "dry_true_reward__dry_false_neutral__hard_b0_lane_control__"
            "pairwise_standard_challenges_selected_fresh__"
            "pareto_risk_buypoint_entryvol__distinct1"
        )

    def score_candidates(self, snapshot_df: pd.DataFrame) -> pd.DataFrame:
        if snapshot_df.empty:
            return pd.DataFrame()

        items: list[SkillCandidate] = []
        source_by_code: dict[str, pd.Series] = {}
        for row_idx, (_, row) in enumerate(snapshot_df.iterrows()):
            if not is_review_universe(row):
                continue
            item = reasoned_item_variant(
                row,
                row_idx,
                dry_policy="reward_only",
                lane_policy="B0_LANE",
            )
            items.append(item)
            source_by_code[item.code] = row

        items.sort(key=lambda x: x.sort_key)
        for rank, item in enumerate(items, 1):
            item.raw_rank = rank

        rows: list[dict[str, Any]] = []
        for item in items:
            src = source_by_code.get(item.code)
            if src is None:
                raise RuntimeError(f"Track E missing source row for {item.code}")

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
                "current_vs_ibd_candidate_pct": to_float(src.get("current_vs_ibd_candidate_pct")),
                "ibd_entry_volume_ratio": to_float(src.get("ibd_entry_volume_ratio")),
                "reason_codes": "|".join(item.reason_codes),
                "risk_codes": "|".join(item.risk_codes),
                "is_actionable": int(item.entry_status == "ACTIONABLE"),
                "has_geom_failure": int("clear_geometry_failure" in item.risk_codes),
                "below_buy_point": int("below_candidate_buy_point" in item.risk_codes),
                "has_known_eps": int(eps_val is not None),
                "has_valid_industry": int(bool(industry_key)),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _eligible(scored_df: pd.DataFrame) -> pd.DataFrame:
        return scored_df[
            (scored_df["is_actionable"] == 1)
            & (scored_df["has_geom_failure"] == 0)
            & (scored_df["below_buy_point"] == 0)
            & (scored_df["has_known_eps"] == 1)
            & (scored_df["has_valid_industry"] == 1)
        ].copy()

    def pairwise_opportunities(
        self,
        scored_df: pd.DataFrame,
        baseline_selected: list[str],
    ) -> list[dict[str, Any]]:
        eligible = self._eligible(scored_df)
        by_code = {
            str(row["code"]): row
            for _, row in eligible.iterrows()
        }
        selected_set = set(baseline_selected)

        selected_fresh = [
            by_code[code]
            for code in baseline_selected
            if code in by_code and by_code[code]["lane"] == TARGET_FRESH_LANE
        ]
        unselected_standard = [
            row for _, row in eligible.iterrows()
            if str(row["code"]) not in selected_set
            and row["lane"] == TARGET_STANDARD_LANE
        ]

        opportunities: list[dict[str, Any]] = []
        for standard in unselected_standard:
            std_code = str(standard["code"])
            for fresh in selected_fresh:
                fresh_code = str(fresh["code"])
                if not strictly_stronger_standard(standard, fresh):
                    continue

                trial = [
                    std_code if code == fresh_code else code
                    for code in baseline_selected
                ]
                industries = [
                    str(by_code[code]["industry_key"]).strip().lower()
                    for code in trial
                    if code in by_code
                ]
                if len(industries) != len(trial) or len(set(industries)) != len(industries):
                    continue

                opportunities.append({
                    "standard_code": std_code,
                    "fresh_code": fresh_code,
                    "standard_strength": target_strength_key(standard),
                    "fresh_strength": target_strength_key(fresh),
                })
        opportunities.sort(
            key=lambda x: (
                x["standard_strength"],
                x["fresh_strength"],
                x["standard_code"],
                x["fresh_code"],
            )
        )
        return opportunities

    def pick_stocks(self, scored_df: pd.DataFrame, industry_quotas: dict[str, int]) -> list[str]:
        baseline_selected = super().pick_stocks(scored_df, industry_quotas)
        if not baseline_selected:
            return []

        eligible = self._eligible(scored_df)
        by_code = {
            str(row["code"]): row
            for _, row in eligible.iterrows()
        }
        selected = list(baseline_selected)

        fresh_slots = [
            idx for idx, code in enumerate(selected)
            if code in by_code and by_code[code]["lane"] == TARGET_FRESH_LANE
        ]
        fresh_slots.sort(
            key=lambda idx: target_strength_key(by_code[selected[idx]]),
            reverse=True,
        )

        standards = [
            row for _, row in eligible.iterrows()
            if str(row["code"]) not in set(baseline_selected)
            and row["lane"] == TARGET_STANDARD_LANE
        ]
        standards.sort(key=target_strength_key)

        for standard in standards:
            std_code = str(standard["code"])
            for slot in list(fresh_slots):
                fresh_code = selected[slot]
                fresh = by_code[fresh_code]
                if not strictly_stronger_standard(standard, fresh):
                    continue

                trial = list(selected)
                trial[slot] = std_code
                industries = [
                    str(by_code[code]["industry_key"]).strip().lower()
                    for code in trial
                    if code in by_code
                ]
                if len(industries) != len(trial) or len(set(industries)) != len(industries):
                    continue

                selected[slot] = std_code
                fresh_slots.remove(slot)
                break

            if not fresh_slots:
                break

        return selected


def baseline_policy() -> DryNeutralHardLaneBaseline:
    return DryNeutralHardLaneBaseline()


def challenger_policy() -> PairwiseFreshStandardChallenger:
    return PairwiseFreshStandardChallenger()


def production_reference_policy() -> StructuralGridChallenger:
    return StructuralGridChallenger("B0_LANE", "symmetric", "distinct_1")
