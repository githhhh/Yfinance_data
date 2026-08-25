"""Skill Rule Mutation Engine & Signature Deduplicator (Phase 2 Step 2 - Production Aligned).

Core Design Principles:
  1. Base Guardrail & Candidate Pool 100% Aligned with Production:
     - Leverages production `rank_skill_industry_eps_known`
     - Requires ACTIONABLE, effective_eps known, no clear_geometry_failure, cur >= 0
     - Industry Deduplication (max 1 per industry)
  2. B0 Baseline 100% Identical to Production `select_skill_industry_eps_known`:
     - Guarantees exact week-by-week identity with Step 1 Level 2
  3. Bounded Mutation Space on Ranking Keys:
     - Freshness-first, Volume-first, ClosePos-first, 52wHigh-first
     - Composite linear weighting
  4. Signature Deduplication (<= 200 Budget):
     - Computed strictly on Train Set (Weeks 1~30)
     - Preserves minimal complexity C
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from dashboard.skill_industry_eps_known import (
    SkillCandidate,
    effective_eps,
    rank_skill_industry_eps_known,
)

logger = logging.getLogger(__name__)


@dataclass
class RuleSpec:
    rule_id: str
    description: str
    complexity: int  # C = n_cond + n_route
    sort_key_fn: Callable[[SkillCandidate, dict[str, Any]], tuple]
    params: dict[str, Any] = field(default_factory=dict)


def get_production_eligible_pool(pool_df: pd.DataFrame) -> list[SkillCandidate]:
    """Extract production eligible candidate pool (100% aligned with select_skill_industry_eps_known)."""
    candidates = rank_skill_industry_eps_known(pool_df)
    eligible: list[SkillCandidate] = []
    for item in candidates:
        if item.entry_status != "ACTIONABLE":
            continue
        if "clear_geometry_failure" in item.risk_codes:
            continue
        if "below_candidate_buy_point" in item.risk_codes:
            continue
        if effective_eps(item) is None:
            continue
        if not item.industry.strip():
            continue
        eligible.append(item)
    return eligible


def build_skill_rule_space() -> list[RuleSpec]:
    """Build bounded discrete parameterizations of Skill ranking variants."""
    rules: list[RuleSpec] = []

    # -------------------------------------------------------------
    # 0. Production B0 Baseline (Exact item.sort_key)
    # -------------------------------------------------------------
    rules.append(RuleSpec(
        rule_id="B0_BASELINE",
        description="Production B0 Baseline (Exact 11-tuple sort_key)",
        complexity=3,
        sort_key_fn=lambda item, p: item.sort_key,
    ))

    # -------------------------------------------------------------
    # 1. Simpler Minimalist Variants (极简纯特征排序)
    # -------------------------------------------------------------
    rules.append(RuleSpec(
        rule_id="SIMPLER_PURE_FRESHNESS",
        description="Pure Freshness Sort (Lowest premium above buy point)",
        complexity=1,
        sort_key_fn=lambda item, p: (
            float(item.feature_values.get("current_vs_ibd_candidate_pct") if item.feature_values.get("current_vs_ibd_candidate_pct") is not None else 999.0),
            item.code,
        ),
    ))

    rules.append(RuleSpec(
        rule_id="SIMPLER_PURE_CLOSE_POS",
        description="Pure Close Position Sort (Highest intraday close position)",
        complexity=1,
        sort_key_fn=lambda item, p: (
            -float(item.feature_values.get("ibd_entry_close_position") or 0.0),
            item.code,
        ),
    ))

    rules.append(RuleSpec(
        rule_id="SIMPLER_PURE_VOLUME",
        description="Pure Volume Sort (Highest entry breakout volume ratio)",
        complexity=1,
        sort_key_fn=lambda item, p: (
            -float(item.feature_values.get("ibd_entry_volume_ratio") or 0.0),
            item.code,
        ),
    ))

    # -------------------------------------------------------------
    # 2. Volume Priority Variants (量能分桶优先)
    # -------------------------------------------------------------
    def sort_vol_first(item: SkillCandidate, params: dict[str, Any]) -> tuple:
        cur = float(item.feature_values.get("current_vs_ibd_candidate_pct") if item.feature_values.get("current_vs_ibd_candidate_pct") is not None else 999.0)
        entry_vol = float(item.feature_values.get("ibd_entry_volume_ratio") or 0.0)
        pos = float(item.feature_values.get("ibd_entry_close_position") or 0.0)
        vol_bucket = 0 if entry_vol >= params.get("high_vol", 2.0) else (1 if entry_vol >= 1.5 else 2)
        fresh_bucket = 0 if cur <= 2.0 else 1
        return (vol_bucket, fresh_bucket, -pos, -entry_vol, item.code)

    for hv in [1.8, 2.0, 2.5]:
        rules.append(RuleSpec(
            rule_id=f"RANK_VOL_FIRST_HV_{hv}",
            description=f"Volume Priority (Vol >= {hv} Tier -> Freshness -> Pos)",
            complexity=4,
            sort_key_fn=sort_vol_first,
            params={"high_vol": hv},
        ))

    # -------------------------------------------------------------
    # 3. Close Position Strength Priority Variants (收盘强硬优先)
    # -------------------------------------------------------------
    def sort_pos_first(item: SkillCandidate, params: dict[str, Any]) -> tuple:
        cur = float(item.feature_values.get("current_vs_ibd_candidate_pct") if item.feature_values.get("current_vs_ibd_candidate_pct") is not None else 999.0)
        entry_vol = float(item.feature_values.get("ibd_entry_volume_ratio") or 0.0)
        pos = float(item.feature_values.get("ibd_entry_close_position") or 0.0)
        pos_bucket = 0 if pos >= params.get("high_pos", 0.85) else (1 if pos >= 0.70 else 2)
        fresh_bucket = 0 if cur <= 2.0 else 1
        return (pos_bucket, fresh_bucket, -entry_vol, -pos, item.code)

    for hp in [0.80, 0.85, 0.90]:
        rules.append(RuleSpec(
            rule_id=f"RANK_POS_FIRST_HP_{hp}",
            description=f"Close Pos Priority (Pos >= {hp} Tier -> Freshness -> Volume)",
            complexity=4,
            sort_key_fn=sort_pos_first,
            params={"high_pos": hp},
        ))

    # -------------------------------------------------------------
    # 4. Near 52W High Momentum Priority
    # -------------------------------------------------------------
    def sort_52w_first(item: SkillCandidate, params: dict[str, Any]) -> tuple:
        cur = float(item.feature_values.get("current_vs_ibd_candidate_pct") if item.feature_values.get("current_vs_ibd_candidate_pct") is not None else 999.0)
        dist_52w = float(item.feature_values.get("dist_to_52w_high_pct") or -99.0)
        entry_vol = float(item.feature_values.get("ibd_entry_volume_ratio") or 0.0)
        pos = float(item.feature_values.get("ibd_entry_close_position") or 0.0)
        near_52w = 0 if dist_52w >= params.get("min_52w", -3.0) else 1
        fresh_bucket = 0 if cur <= 2.0 else 1
        return (near_52w, fresh_bucket, -entry_vol, -pos, item.code)

    for d52 in [-2.0, -3.0, -5.0]:
        rules.append(RuleSpec(
            rule_id=f"RANK_52W_MOMENTUM_D_{abs(d52)}",
            description=f"52W High Priority (dist >= {d52}% -> Freshness -> Vol)",
            complexity=4,
            sort_key_fn=sort_52w_first,
            params={"min_52w": d52},
        ))

    # -------------------------------------------------------------
    # 5. Route Pullback Dry-Up Specialization
    # -------------------------------------------------------------
    def sort_route_dry_first(item: SkillCandidate, params: dict[str, Any]) -> tuple:
        is_pullback = "pullback_structure" in item.reason_codes
        dry = "dry_pullback" in item.reason_codes
        cur = float(item.feature_values.get("current_vs_ibd_candidate_pct") if item.feature_values.get("current_vs_ibd_candidate_pct") is not None else 999.0)
        entry_vol = float(item.feature_values.get("ibd_entry_volume_ratio") or 0.0)
        pos = float(item.feature_values.get("ibd_entry_close_position") or 0.0)
        if is_pullback and dry:
            route_tier = 0
        elif not is_pullback:
            route_tier = 1
        else:
            route_tier = 2
        fresh_bucket = 0 if cur <= 2.0 else 1
        return (route_tier, fresh_bucket, -entry_vol, -pos, item.code)

    rules.append(RuleSpec(
        rule_id="RANK_ROUTE_PULLBACK_DRY_PRIORITY",
        description="Route Specialization (Dry Pullback > Base > Non-Dry Pullback)",
        complexity=4,
        sort_key_fn=sort_route_dry_first,
    ))

    # -------------------------------------------------------------
    # 6. Coarse Discrete Linear Composite Scores
    # -------------------------------------------------------------
    def sort_composite_score(item: SkillCandidate, params: dict[str, Any]) -> tuple:
        w_f = params.get("w_fresh", 2)
        w_v = params.get("w_vol", 2)
        w_p = params.get("w_pos", 1)
        w_d = params.get("w_dry", 1)
        w_52 = params.get("w_52", 1)

        cur = float(item.feature_values.get("current_vs_ibd_candidate_pct") if item.feature_values.get("current_vs_ibd_candidate_pct") is not None else 999.0)
        entry_vol = float(item.feature_values.get("ibd_entry_volume_ratio") or 0.0)
        pos = float(item.feature_values.get("ibd_entry_close_position") or 0.0)
        dist_52w = float(item.feature_values.get("dist_to_52w_high_pct") or -99.0)
        dry = 1.0 if "dry_pullback" in item.reason_codes else 0.0

        s_fresh = max(0.0, 1.0 - cur / 5.0)
        s_vol = min(1.0, max(0.0, (entry_vol - 1.0) / 2.0))
        s_pos = min(1.0, max(0.0, (pos - 0.5) / 0.5))
        s_52 = 1.0 if dist_52w >= -5.0 else 0.0
        s_dry = dry

        total_score = w_f * s_fresh + w_v * s_vol + w_p * s_pos + w_d * s_dry + w_52 * s_52
        return (-total_score, item.code)

    for wf in [1, 2, 3]:
        for wv in [1, 2, 3]:
            for wp in [1, 2]:
                for wd in [0, 1]:
                    for w52 in [0, 1]:
                        cid = f"COMPOSITE_F{wf}_V{wv}_P{wp}_D{wd}_52W{w52}"
                        rules.append(RuleSpec(
                            rule_id=cid,
                            description=f"Linear Composite (Fresh:{wf}, Vol:{wv}, Pos:{wp}, Dry:{wd}, 52W:{w52})",
                            complexity=2 + (1 if wd > 0 else 0) + (1 if w52 > 0 else 0),
                            sort_key_fn=sort_composite_score,
                            params={"w_fresh": wf, "w_vol": wv, "w_pos": wp, "w_dry": wd, "w_52": w52},
                        ))

    return rules


def evaluate_rule_on_pool(
    pool_df: pd.DataFrame,
    rule: RuleSpec,
    pick_limit: int = 3,
) -> list[str]:
    """Execute rule ranking and industry deduplication on a single snapshot pool."""
    if pool_df.empty:
        return []

    eligible = get_production_eligible_pool(pool_df)
    if not eligible:
        return []

    eligible.sort(key=lambda item: rule.sort_key_fn(item, rule.params))

    selected_codes: list[str] = []
    covered_industries: set[str] = set()

    for item in eligible:
        code = item.code.strip()
        ind = item.industry.strip().lower()
        if ind and ind in covered_industries:
            continue
        selected_codes.append(code)
        if ind:
            covered_industries.add(ind)
        if len(selected_codes) >= pick_limit:
            break

    return selected_codes


def compute_rule_train_signature(
    rule: RuleSpec,
    events_df: pd.DataFrame,
    train_weeks: list[str],
    pick_limit: int = 3,
) -> str:
    """Compute deterministic MD5 signature of rule's weekly picks across Train Weeks (Weeks 1~30)."""
    signature_payload: list[str] = []
    for snap_date in sorted(train_weeks):
        snap_df = events_df[events_df["snapshot_date"] == snap_date].copy()
        snap_df["snapshot_date"] = snap_date
        picks = evaluate_rule_on_pool(snap_df, rule, pick_limit=pick_limit)
        signature_payload.append(f"{snap_date}:{','.join(picks)}")
    raw_str = "|".join(signature_payload)
    return hashlib.md5(raw_str.encode("utf-8")).hexdigest()


def deduplicate_rule_signatures(
    rules: list[RuleSpec],
    events_df: pd.DataFrame,
    train_weeks: list[str],
    pick_limit: int = 3,
    budget_limit: int = 200,
) -> tuple[list[RuleSpec], dict[str, list[RuleSpec]]]:
    """Deduplicate rule configurations based on Train set signatures.
    
    If multiple rules produce identical picks across Weeks 1~30, keep the one with minimal complexity C.
    Enforces hard budget limit <= 200 unique signatures.
    """
    signature_map: dict[str, list[RuleSpec]] = {}

    for rule in rules:
        sig = compute_rule_train_signature(rule, events_df, train_weeks, pick_limit=pick_limit)
        signature_map.setdefault(sig, []).append(rule)

    deduped_rules: list[RuleSpec] = []
    for sig, matching_rules in signature_map.items():
        best_rule = min(matching_rules, key=lambda r: (r.complexity, r.rule_id))
        deduped_rules.append(best_rule)

    deduped_rules.sort(key=lambda r: (r.complexity, r.rule_id))
    if len(deduped_rules) > budget_limit:
        logger.warning(f"Truncating {len(deduped_rules)} signatures to hard budget limit {budget_limit}")
        deduped_rules = deduped_rules[:budget_limit]

    return deduped_rules, signature_map
