from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .labels import classify_geometry
from .utils import to_bool, to_float
from dashboard.skill_industry_eps_known import rank_skill_industry_eps_known, select_skill_industry_eps_known


PULLBACK_RULES = {"pivot", "ceiling_pullback", "ma10_touch_confirm", "three_weeks_tight"}
BASE_RULES = {"ceiling", "ceiling_breakout"}


@dataclass(frozen=True)
class SelectorConfig:
    name: str
    status_mode: str = "hard"
    valid_mode: str = "drop"
    close_mode: str = "soft"
    fresh_mode: str = "continuous"
    volume_mode: str = "soft"
    geometry_mode: str = "failure_only"
    pullback_dry_mode: str = "soft"
    eps_mode: str = "known_hard"
    industry_cover: bool = True
    top_n: int = 3
    changed_rules: tuple[str, ...] = ()


def selector_configs() -> dict[str, SelectorConfig]:
    base = SelectorConfig("B0_PIT_VERIFIED")
    configs = {
        "B0_REPO_EXACT": SelectorConfig("B0_REPO_EXACT", eps_mode="repo_soft"),
        "B0_PIT_VERIFIED": base,
        "B0 status soft": SelectorConfig("B0 status soft", status_mode="soft", changed_rules=("status",)),
        "B0 status supplemental UNCONFIRMED": SelectorConfig(
            "B0 status supplemental UNCONFIRMED", status_mode="supplemental_unconfirmed", changed_rules=("status",)
        ),
        "B0 no entry_valid": SelectorConfig("B0 no entry_valid", valid_mode="drop", changed_rules=("entry_valid",)),
        "B0 close trigger soft": SelectorConfig("B0 close trigger soft", close_mode="soft", changed_rules=("close_trigger",)),
        "B0 fresh continuous": SelectorConfig("B0 fresh continuous", fresh_mode="continuous", changed_rules=("fresh",)),
        "B0 volume soft": SelectorConfig("B0 volume soft", volume_mode="soft", changed_rules=("volume",)),
        "B0 volume route-specific": SelectorConfig("B0 volume route-specific", volume_mode="route_specific", changed_rules=("volume",)),
        "B0 geometry soft": SelectorConfig("B0 geometry soft", geometry_mode="soft", changed_rules=("geometry",)),
        "B0 geometry failure only": SelectorConfig("B0 geometry failure only", geometry_mode="failure_only", changed_rules=("geometry",)),
        "B0 pullback dry hard": SelectorConfig("B0 pullback dry hard", pullback_dry_mode="hard", changed_rules=("pullback_dry",)),
        "B0 pullback dry bonus": SelectorConfig("B0 pullback dry bonus", pullback_dry_mode="bonus", changed_rules=("pullback_dry",)),
        "B0 pullback dry drop": SelectorConfig("B0 pullback dry drop", pullback_dry_mode="drop", changed_rules=("pullback_dry",)),
        "B0 EPS >=25 hard/bonus/drop": SelectorConfig("B0 EPS >=25 hard/bonus/drop", eps_mode="hard_25", changed_rules=("eps",)),
        "B0 EPS unknown manual-review": SelectorConfig(
            "B0 EPS unknown manual-review", eps_mode="unknown_manual_review", changed_rules=("eps",)
        ),
        "B0 no industry cover": SelectorConfig("B0 no industry cover", industry_cover=False, changed_rules=("industry_cover",)),
        "B0 top1": SelectorConfig("B0 top1", top_n=1, changed_rules=("topk",)),
        "R1_ATOMIC_IMPROVEMENTS": SelectorConfig(
            "R1_ATOMIC_IMPROVEMENTS",
            fresh_mode="continuous",
            volume_mode="soft",
            geometry_mode="failure_only",
            pullback_dry_mode="bonus",
            changed_rules=("fresh", "volume", "geometry", "pullback_dry"),
        ),
        "R2_BALANCED_SOFT": SelectorConfig(
            "R2_BALANCED_SOFT",
            status_mode="soft",
            fresh_mode="continuous",
            volume_mode="soft",
            geometry_mode="soft",
            pullback_dry_mode="bonus",
            eps_mode="known_hard",
            changed_rules=("status", "fresh", "volume", "geometry", "pullback_dry"),
        ),
        "R3_MINIMAL_TECHNICAL": SelectorConfig(
            "R3_MINIMAL_TECHNICAL",
            eps_mode="drop",
            fresh_mode="continuous",
            geometry_mode="failure_only",
            changed_rules=("eps", "fresh", "geometry"),
        ),
    }
    return configs


def enrich_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "pit_eps_yoy_growth" not in out.columns:
        out["pit_eps_yoy_growth"] = out.get("eps_yoy_growth", pd.NA)
    if "pit_eps_state" not in out.columns:
        out["pit_eps_state"] = out["pit_eps_yoy_growth"].map(lambda value: "VERIFIED" if to_float(value) is not None else "UNKNOWN")
    out["geometry"] = [
        classify_geometry(to_float(row.get("ibd_entry_close_position")), to_float(row.get("ibd_entry_breakout_range_ratio")))
        for _, row in out.iterrows()
    ]
    out["trigger_pos"] = [
        None
        if to_float(row.get("ibd_entry_close_position")) is None or to_float(row.get("ibd_entry_breakout_range_ratio")) is None
        else to_float(row.get("ibd_entry_close_position")) - to_float(row.get("ibd_entry_breakout_range_ratio"))
        for _, row in out.iterrows()
    ]
    out["pullback_dry_state"] = [
        _pullback_dry_state(str(row.get("ibd_candidate_rule", "") or ""), row.get("pullback_v_is_dry"))
        for _, row in out.iterrows()
    ]
    return out


def select_weekly(pool: pd.DataFrame, config: SelectorConfig) -> pd.DataFrame:
    frame = enrich_features(pool)
    if config.name in {"B0_REPO_EXACT", "B0_PIT_VERIFIED"}:
        return _select_production_b0(frame, config)
    rows: list[dict[str, Any]] = []
    for idx, row in frame.iterrows():
        allowed, score, reasons, risks = score_row(row, config)
        if not allowed:
            continue
        out = row.to_dict()
        out["score"] = round(score, 6)
        out["reason_codes"] = ";".join(reasons)
        out["risk_codes"] = ";".join(risks)
        out["selected_by"] = config.name
        out["production_raw_rank"] = pd.NA
        out["production_sort_key"] = ""
        out["_row_index"] = idx
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=list(frame.columns) + ["score", "reason_codes", "risk_codes", "selected_by", "pick_order"])
    ranked = pd.DataFrame(rows).sort_values(["score", "code", "_row_index"], ascending=[False, True, True])
    selected: list[pd.Series] = []
    covered: set[str] = set()
    for _, row in ranked.iterrows():
        industry = str(row.get("industry", "") or "").strip().lower()
        if config.industry_cover and industry and industry in covered:
            continue
        selected.append(row)
        if config.industry_cover and industry:
            covered.add(industry)
        if len(selected) >= config.top_n:
            break
    out = pd.DataFrame(selected).drop(columns=["_row_index"], errors="ignore")
    if not out.empty:
        out.insert(2, "pick_order", range(1, len(out) + 1))
    return out


def _select_production_b0(frame: pd.DataFrame, config: SelectorConfig) -> pd.DataFrame:
    pool = frame.copy()
    if config.name == "B0_PIT_VERIFIED":
        pool["eps_yoy_growth"] = pool["pit_eps_yoy_growth"]
    selected = select_skill_industry_eps_known(pool, limit=config.top_n)
    by_code = {str(row.get("code", "")).strip(): row for _, row in frame.iterrows()}
    rows: list[dict[str, Any]] = []
    for pick_order, item in enumerate(selected, 1):
        source = by_code.get(item.code)
        out = source.to_dict() if source is not None else {}
        out["code"] = item.code
        out["pick_order"] = pick_order
        out["score"] = None
        out["reason_codes"] = ";".join(item.reason_codes)
        out["risk_codes"] = ";".join(item.risk_codes)
        out["selected_by"] = config.name
        out["production_raw_rank"] = item.raw_rank
        out["production_sort_key"] = repr(item.sort_key)
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=list(frame.columns) + ["pick_order", "score", "reason_codes", "risk_codes", "selected_by"])
    return pd.DataFrame(rows)


def production_selected_codes(pool: pd.DataFrame) -> list[tuple[str, int, str, str]]:
    return [
        (item.code, order, ";".join(item.reason_codes), ";".join(item.risk_codes))
        for order, item in enumerate(select_skill_industry_eps_known(pool), 1)
    ]


def select_all_weeks(panel: pd.DataFrame, config: SelectorConfig) -> pd.DataFrame:
    chunks = []
    for snapshot, group in panel.groupby("snapshot_date", sort=True):
        selected = select_weekly(group, config)
        if not selected.empty:
            selected["snapshot_date"] = snapshot
            chunks.append(selected)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def score_row(row: pd.Series, config: SelectorConfig) -> tuple[bool, float, list[str], list[str]]:
    if to_bool(row.get("signal")) is not True or not str(row.get("ibd_candidate_rule", "") or "").strip():
        return False, 0.0, [], ["not_signal_universe"]
    status = str(row.get("ibd_entry_status", "") or "").strip().upper()
    rule = str(row.get("ibd_candidate_rule", "") or "")
    score = 0.0
    reasons: list[str] = []
    risks: list[str] = []

    if status == "ACTIONABLE":
        score += 6
        reasons.append("status_actionable")
    elif config.status_mode == "hard":
        return False, score, reasons, [f"status_{status or 'UNKNOWN'}"]
    elif config.status_mode == "supplemental_unconfirmed" and status != "UNCONFIRMED":
        return False, score, reasons, [f"status_{status or 'UNKNOWN'}"]
    else:
        score += 1 if status == "UNCONFIRMED" else -1
        risks.append(f"status_{status or 'UNKNOWN'}")

    valid = to_bool(row.get("ibd_entry_valid"))
    if config.valid_mode != "drop":
        if valid is not True:
            return False, score, reasons, ["entry_valid_not_true"]
        score += 1

    close = to_float(row.get("ibd_entry_close_vs_trigger_pct"))
    if close is None:
        if config.close_mode == "hard":
            return False, score, reasons, ["close_trigger_unknown"]
        risks.append("close_trigger_unknown")
    elif close < 0:
        if config.close_mode == "hard":
            return False, score, reasons, ["close_below_trigger"]
        score -= 2
        risks.append("close_below_trigger")
    else:
        score += max(0.0, 2.0 - min(close, 6.0) / 3.0)
        reasons.append("close_above_trigger")

    cur = to_float(row.get("current_vs_ibd_candidate_pct"))
    if config.fresh_mode == "hard":
        if cur is None or cur < 0 or cur > 5:
            return False, score, reasons, ["fresh_hard_fail"]
        score += 3 if cur <= 2 else 1
    else:
        fresh_score, fresh_risk = _fresh_score(cur)
        score += fresh_score
        if fresh_risk:
            risks.append(fresh_risk)

    vol = to_float(row.get("ibd_entry_volume_ratio"))
    vol_min = 1.3 if config.volume_mode == "route_specific" and rule in PULLBACK_RULES else 1.5
    if vol is None:
        if config.volume_mode == "hard":
            return False, score, reasons, ["volume_unknown"]
        risks.append("volume_unknown")
    elif vol < vol_min:
        if config.volume_mode == "hard":
            return False, score, reasons, ["volume_below_standard"]
        score -= 0.5
        risks.append("volume_below_standard")
    else:
        score += min(3.0, vol)
        reasons.append("volume_confirms")

    geometry = str(row.get("geometry", "UNKNOWN") or "UNKNOWN")
    if geometry in {"Defensive Failure", "Squat / Upper Shadow"}:
        if config.geometry_mode in {"hard", "failure_only"}:
            return False, score, reasons, ["geometry_failure"]
        score -= 2
        risks.append("geometry_failure")
    elif geometry == "UNKNOWN":
        if config.geometry_mode == "hard":
            return False, score, reasons, ["geometry_unknown"]
        risks.append("geometry_unknown")
    else:
        score += {"Full-range Breakout": 3, "Strong Finish": 2.5, "Constructive Breakout": 1.5, "Marginal Breakout": 0.5}.get(geometry, 0)
        reasons.append(f"geometry_{geometry}")

    dry_state = str(row.get("pullback_dry_state", "UNKNOWN"))
    if dry_state == "FAIL":
        if config.pullback_dry_mode == "hard":
            return False, score, reasons, ["pullback_not_dry"]
        if config.pullback_dry_mode != "drop":
            score -= 0.5
            risks.append("pullback_not_dry")
    elif dry_state == "PASS" and config.pullback_dry_mode in {"soft", "bonus"}:
        score += 0.75
        reasons.append("dry_pullback")

    eps = to_float(row.get("pit_eps_yoy_growth")) if config.eps_mode != "repo_soft" else to_float(row.get("eps_yoy_growth"))
    eps_state = str(row.get("pit_eps_state", "") or "UNKNOWN")
    if config.eps_mode == "drop":
        return True, score, reasons, risks
    if eps is None:
        if config.eps_mode == "unknown_manual_review":
            risks.append("eps_unknown_manual_review")
        elif config.eps_mode == "known_hard":
            return False, score, reasons, ["eps_unknown"]
        elif config.eps_mode == "hard_25":
            return False, score, reasons, ["eps_unknown_or_below_25"]
        else:
            risks.append(f"eps_{eps_state.lower()}")
    elif eps >= 25:
        score += 1.5
        reasons.append("eps_25")
    elif config.eps_mode == "hard_25":
        return False, score, reasons, ["eps_unknown_or_below_25"]
    else:
        risks.append("eps_below_25")

    weekly = to_float(row.get("volume_ratio"))
    if weekly is not None and weekly >= 1.3:
        score += 0.5
        reasons.append("weekly_volume")
    return True, score, reasons, risks


def _pullback_dry_state(rule: str, value: object) -> str:
    if rule not in PULLBACK_RULES:
        return "NOT_APPLICABLE"
    parsed = to_bool(value)
    if parsed is True:
        return "PASS"
    if parsed is False:
        return "FAIL"
    return "UNKNOWN"


def _fresh_score(cur: float | None) -> tuple[float, str]:
    if cur is None:
        return -0.5, "fresh_unknown"
    if cur < 0:
        return -2.5, "below_candidate_buy_point"
    if cur <= 2:
        return 3.0, ""
    if cur <= 5:
        return 1.5, ""
    if cur <= 10:
        return -0.5, "extended_5_10"
    return -1.5, "extended_gt_10"
