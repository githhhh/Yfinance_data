"""Rule-based B0 challengers for Track B dry-policy × selector experiment.

B0_ORIGINAL: directly calls production rank/select — zero reimplementation.
reward_only / ignored: thin adapter over production helpers, only modifying
the dry-up branch in reasoned_item().

Selector variants (distinct_1 / pure_top3 / max_2_per_ind) share identical
eligibility universe; only the industry concentration constraint differs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

# ── direct production imports (frozen, never modified here) ──────────────
from dashboard.skill_industry_eps_known import (
    LANE_ORDER,
    SkillCandidate,
    clear_geometry_failure,
    effective_eps,
    entry_status,
    feature_values,
    fresh_bucket,
    geometry_caution,
    is_pullback_rule,
    is_review_universe,
    lane_for,
    rank_skill_industry_eps_known,
    row_eps,
    select_skill_industry_eps_known,
    to_bool,
    to_float,
)

DryPolicy = Literal["symmetric", "reward_only", "ignored"]
SelectorVariant = Literal["distinct_1", "pure_top3", "max_2_per_ind"]

# ── B0_ORIGINAL: 100 % production passthrough ───────────────────────────

def rank_b0_original(pool: pd.DataFrame) -> list[SkillCandidate]:
    """Identical to production rank_skill_industry_eps_known."""
    return rank_skill_industry_eps_known(pool)


def select_b0_original(pool: pd.DataFrame, *, limit: int = 3) -> list[SkillCandidate]:
    """Identical to production select_skill_industry_eps_known."""
    return select_skill_industry_eps_known(pool, limit=limit)


# ── Parameterised ranking with dry_policy ────────────────────────────────

def _reasoned_item_variant(row: pd.Series, row_idx: int,
                           dry_policy: DryPolicy) -> SkillCandidate:
    """Production reasoned_item() logic with a single parameterised branch.

    Only the pullback_v_is_dry handling differs:
      symmetric  – production behaviour (False → risk_code)
      reward_only – True → reason_code, False → neutral, Missing → neutral
      ignored    – dry-up field completely ignored (no codes generated)
    """
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

        if dry_policy == "symmetric":
            # ── production behaviour ──
            if dry is True:
                reason_codes.append("dry_pullback")
            elif dry is False:
                risk_codes.append("pullback_not_dry")

        elif dry_policy == "reward_only":
            # ── only reward True; False and Missing are neutral ──
            if dry is True:
                reason_codes.append("dry_pullback")
            # False / Missing → no code generated

        elif dry_policy == "ignored":
            pass  # dry-up completely ignored

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


def rank_b0_variant(pool: pd.DataFrame, *,
                    dry_policy: DryPolicy) -> list[SkillCandidate]:
    """Rank candidates using production logic + parameterised dry_policy."""
    if dry_policy == "symmetric":
        return rank_b0_original(pool)

    candidates = []
    for row_idx, row in pool.iterrows():
        if not is_review_universe(row):
            continue
        candidates.append(_reasoned_item_variant(row, row_idx, dry_policy))
    candidates.sort(key=lambda item: item.sort_key)
    for rank, item in enumerate(candidates, 1):
        item.raw_rank = rank
    return candidates


# ── Selector variants ────────────────────────────────────────────────────

def _apply_eligibility_filter(ranked: list[SkillCandidate]) -> list[SkillCandidate]:
    """Shared eligibility gate — identical to production select_skill_industry_eps_known
    minus the industry dedup, so all selector variants share the same universe."""
    eligible = []
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
        eligible.append(item)
    return eligible


def select_b0_variant(pool: pd.DataFrame, *,
                      dry_policy: DryPolicy,
                      selector: SelectorVariant,
                      limit: int = 3) -> list[SkillCandidate]:
    """Apply dry_policy ranking + selector variant."""
    if dry_policy == "symmetric" and selector == "distinct_1":
        return select_b0_original(pool, limit=limit)

    ranked = rank_b0_variant(pool, dry_policy=dry_policy)
    eligible = _apply_eligibility_filter(ranked)

    if selector == "distinct_1":
        selected: list[SkillCandidate] = []
        covered: set[str] = set()
        for item in eligible:
            key = item.industry.strip().lower()
            if key in covered:
                continue
            selected.append(item)
            covered.add(key)
            if len(selected) >= limit:
                break
        return selected

    elif selector == "pure_top3":
        return eligible[:limit]

    elif selector == "max_2_per_ind":
        selected = []
        ind_count: dict[str, int] = {}
        for item in eligible:
            key = item.industry.strip().lower()
            if ind_count.get(key, 0) >= 2:
                continue
            selected.append(item)
            ind_count[key] = ind_count.get(key, 0) + 1
            if len(selected) >= limit:
                break
        return selected

    else:
        raise ValueError(f"Unknown selector variant: {selector}")


# ── Convenience: generate all challenger picks for a single snapshot ─────

ALL_DRY_POLICIES: list[DryPolicy] = ["symmetric", "reward_only", "ignored"]
ALL_SELECTORS: list[SelectorVariant] = ["distinct_1", "pure_top3", "max_2_per_ind"]


def challenger_id(dry_policy: DryPolicy, selector: SelectorVariant) -> str:
    prefix_map = {
        "symmetric": "B0_ORIGINAL",
        "reward_only": "B0_DRY_REWARD_ONLY",
        "ignored": "B0_DRY_IGNORED",
    }
    return f"{prefix_map[dry_policy]}__{selector}"


def generate_all_rule_based_picks(
    snapshot_pool: pd.DataFrame,
    *,
    limit: int = 3,
) -> dict[str, list[SkillCandidate]]:
    """Run all dry_policy × selector combinations on a single-snapshot pool.

    Returns {challenger_id: list[SkillCandidate]}.
    B0_ORIGINAL__distinct_1 is guaranteed to use production passthrough.
    """
    results: dict[str, list[SkillCandidate]] = {}
    for dp in ALL_DRY_POLICIES:
        for sel in ALL_SELECTORS:
            cid = challenger_id(dp, sel)
            results[cid] = select_b0_variant(
                snapshot_pool, dry_policy=dp, selector=sel, limit=limit,
            )
    return results
