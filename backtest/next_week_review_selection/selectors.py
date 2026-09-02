from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import to_bool, to_float


PULLBACK_RULES = {"ceiling_pullback", "pivot", "ma10_touch_confirm", "three_weeks_tight"}
EVIDENCE_FAMILIES = ("volume", "eps", "rs_near_high", "supply_contraction")


@dataclass(frozen=True)
class ReviewRule:
    """Rule for supplemental non-ACTIONABLE review candidates.

    ACTIONABLE rows are always retained unchanged. The rule only decides which
    extra UNCONFIRMED / BELOW_TRIGGER rows are added to the Review List.
    """

    name: str
    near_below_pct: float = 5.0
    supplemental_statuses: tuple[str, ...] = ("UNCONFIRMED", "BELOW_TRIGGER")
    min_evidence_families: int = 1
    exclude_clear_geometry_failure: bool = False
    enabled_evidence_families: tuple[str, ...] = EVIDENCE_FAMILIES


def primary_rule() -> ReviewRule:
    """Core R1: Near Buy Point + >=1 independent positive evidence family.

    Geometry is deliberately NOT a hard reject in the core hypothesis.
    """
    return ReviewRule(name="R1_NEAR_BUY_POINT_PLUS_EVIDENCE")


def is_review_universe(row: pd.Series) -> bool:
    return (
        to_bool(row.get("signal")) is True
        and bool(str(row.get("ibd_candidate_rule", "") or "").strip())
    )


def clear_geometry_failure(row: pd.Series) -> bool:
    """Observed geometry failure only; missing geometry remains UNKNOWN."""
    rr = to_float(row.get("ibd_entry_breakout_range_ratio"))
    pos = to_float(row.get("ibd_entry_close_position"))
    if rr is not None and rr <= 0:
        return True
    if pos is not None and pos < 0.65:
        return True
    return False


def raw_positive_evidence(row: pd.Series) -> dict[str, bool]:
    """Raw positive facts. False/missing is neutral, never negative."""
    entry_vol = to_float(row.get("ibd_entry_volume_ratio"))
    weekly_vol = to_float(row.get("volume_ratio"))
    eps_state = str(row.get("pit_eps_state", "") or "").strip().upper()
    pit_eps = to_float(row.get("pit_eps_yoy_growth")) if eps_state == "VERIFIED" else None
    dist = to_float(row.get("dist_to_52w_high_pct"))
    rule = str(row.get("ibd_candidate_rule", "") or "").strip()
    dry = to_bool(row.get("pullback_v_is_dry"))
    return {
        "entry_volume_confirmed": entry_vol is not None and entry_vol >= 1.5,
        "weekly_volume_follow_through": weekly_vol is not None and weekly_vol >= 1.3,
        "eps_support": pit_eps is not None and pit_eps >= 25.0,
        "near_52w_high": dist is not None and dist > -5.0,
        "dry_pullback": rule in PULLBACK_RULES and dry is True,
    }


def evidence_family_flags(row: pd.Series) -> dict[str, bool]:
    """Independent evidence families.

    Entry-volume and weekly-volume are one Volume family, so two related volume
    facts can never count as two independent pieces of evidence.
    """
    raw = raw_positive_evidence(row)
    return {
        "volume": raw["entry_volume_confirmed"] or raw["weekly_volume_follow_through"],
        "eps": raw["eps_support"],
        "rs_near_high": raw["near_52w_high"],
        "supply_contraction": raw["dry_pullback"],
    }


def evidence_family_count(
    row: pd.Series,
    enabled_families: Iterable[str] = EVIDENCE_FAMILIES,
) -> int:
    enabled = set(enabled_families)
    families = evidence_family_flags(row)
    return sum(bool(value) for key, value in families.items() if key in enabled)


def supplemental_path_eligible(row: pd.Series, rule: ReviewRule) -> bool:
    """Near-Buy-Point path; EXTENDED remains outside the core selector."""
    status = str(row.get("ibd_entry_status", "") or "").strip().upper()
    if status not in rule.supplemental_statuses:
        return False
    cur = to_float(row.get("current_vs_ibd_candidate_pct"))
    if cur is None:
        return False
    if status == "UNCONFIRMED":
        return -rule.near_below_pct <= cur <= 5.0
    if status == "BELOW_TRIGGER":
        return -rule.near_below_pct <= cur < 0.0
    return False


def enrich_review_features(pool: pd.DataFrame, rule: ReviewRule | None = None) -> pd.DataFrame:
    rule = rule or primary_rule()
    frame = pool.copy()
    frame["_source_row_order"] = np.arange(len(frame))
    frame = frame.loc[frame.apply(is_review_universe, axis=1)].copy()
    frame["_status"] = (
        frame["ibd_entry_status"].fillna("").astype(str).str.strip().str.upper()
    )
    frame["_vs_buy_point"] = frame["current_vs_ibd_candidate_pct"].map(to_float)
    frame["_geometry_failure"] = frame.apply(clear_geometry_failure, axis=1)

    raw = frame.apply(raw_positive_evidence, axis=1)
    for key in (
        "entry_volume_confirmed",
        "weekly_volume_follow_through",
        "eps_support",
        "near_52w_high",
        "dry_pullback",
    ):
        frame[f"_positive_{key}"] = raw.map(lambda item, k=key: bool(item.get(k, False)))

    families = frame.apply(evidence_family_flags, axis=1)
    for family in EVIDENCE_FAMILIES:
        frame[f"_evidence_{family}"] = families.map(
            lambda item, key=family: bool(item.get(key, False))
        )
    frame["_evidence_family_count"] = frame.apply(
        lambda row: evidence_family_count(row, rule.enabled_evidence_families),
        axis=1,
    )
    return frame


def select_b0_actionable(pool: pd.DataFrame) -> pd.DataFrame:
    """Actual weekend baseline: every ACTIONABLE active signal."""
    frame = enrich_review_features(pool)
    selected = frame.loc[frame["_status"].eq("ACTIONABLE")].copy()
    selected["variant"] = "B0_ACTIONABLE_ONLY"
    selected["selection_source"] = "ACTIONABLE_BASELINE"
    selected["review_reason"] = "weekend_actionable"
    return _stable_output(selected)


def select_supplemental(pool: pd.DataFrame, rule: ReviewRule) -> pd.DataFrame:
    frame = enrich_review_features(pool, rule)
    eligible = frame.apply(lambda row: supplemental_path_eligible(row, rule), axis=1)
    eligible &= frame["_evidence_family_count"].ge(rule.min_evidence_families)
    if rule.exclude_clear_geometry_failure:
        eligible &= ~frame["_geometry_failure"]

    selected = frame.loc[eligible].copy()
    selected["variant"] = rule.name
    selected["selection_source"] = "SUPPLEMENTAL"
    selected["review_reason"] = selected.apply(_supplemental_reason, axis=1)
    return _stable_output(selected)


def select_review_variant(pool: pd.DataFrame, rule: ReviewRule) -> pd.DataFrame:
    """B0 ACTIONABLE + supplemental candidates. No TopN and no ranking."""
    baseline = select_b0_actionable(pool)
    supplemental = select_supplemental(pool, rule)
    if baseline.empty:
        combined = supplemental.copy()
    elif supplemental.empty:
        combined = baseline.copy()
    else:
        combined = pd.concat([baseline, supplemental], ignore_index=True, sort=False)
    if combined.empty:
        return combined
    combined["variant"] = rule.name
    return (
        combined.sort_values("_source_row_order", kind="mergesort")
        .drop_duplicates(["snapshot_date", "code"], keep="first")
        .reset_index(drop=True)
    )


def rule_to_dict(rule: ReviewRule) -> dict[str, object]:
    return asdict(rule)


def rule_complexity(rule: ReviewRule) -> int:
    disabled_families = len(
        set(EVIDENCE_FAMILIES) - set(rule.enabled_evidence_families)
    )
    status_specialization = (
        0
        if set(rule.supplemental_statuses) == {"UNCONFIRMED", "BELOW_TRIGGER"}
        else 1
    )
    threshold_specialization = 0 if rule.near_below_pct == 5.0 else 1
    evidence_strictness = max(rule.min_evidence_families - 1, 0)
    geometry_specialization = 1 if rule.exclude_clear_geometry_failure else 0
    return (
        disabled_families
        + status_specialization
        + threshold_specialization
        + evidence_strictness
        + geometry_specialization
    )


def _supplemental_reason(row: pd.Series) -> str:
    status = str(row.get("_status", ""))
    families = [
        family
        for family in EVIDENCE_FAMILIES
        if bool(row.get(f"_evidence_{family}", False))
    ]
    path = (
        "unconfirmed_near_buy_point"
        if status == "UNCONFIRMED"
        else "below_trigger_near_buy_point"
    )
    return "|".join([path, *families])


def _stable_output(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    return frame.sort_values("_source_row_order", kind="mergesort").reset_index(drop=True)
