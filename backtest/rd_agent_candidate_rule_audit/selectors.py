from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

import dashboard.skill_industry_eps_known as production

from .labels import classify_geometry
from .utils import to_bool, to_float


PULLBACK_RULES = {"pivot", "ceiling_pullback", "ma10_touch_confirm", "three_weeks_tight"}
TRACE_DIMENSIONS = (
    "status",
    "entry_valid",
    "close_trigger",
    "fresh",
    "volume",
    "geometry",
    "pullback_dry",
    "eps",
    "industry_cover",
    "topk",
)


@dataclass(frozen=True)
class SelectorConfig:
    name: str
    eps_source: str = "pit_verified"
    status_eligibility: str = "actionable"
    entry_valid_gate: bool = False
    close_trigger_gate: bool = False
    fresh_rank: str = "production"
    volume_policy: str = "production"
    geometry_eligibility: str = "exclude_failure"
    pullback_dry_policy: str = "production"
    eps_eligibility: str = "known"
    eps_rank: str = "production"
    industry_cover: bool = True
    top_n: int = 3
    changed_rules: tuple[str, ...] = ()
    source_atomic_variants: tuple[str, ...] = ()


def selector_configs() -> dict[str, SelectorConfig]:
    base = SelectorConfig("B0_PIT_VERIFIED")
    return {
        "B0_REPO_EXACT": replace(base, name="B0_REPO_EXACT", eps_source="repo"),
        "B0_PIT_VERIFIED": base,
        "B0 status soft": replace(
            base,
            name="B0 status soft",
            status_eligibility="all",
            changed_rules=("status",),
        ),
        "B0 status supplemental UNCONFIRMED": replace(
            base,
            name="B0 status supplemental UNCONFIRMED",
            status_eligibility="actionable_unconfirmed",
            changed_rules=("status",),
        ),
        # Production B0 never gates on entry_valid or close-vs-trigger. These
        # registered controls must therefore report no treatment contrast.
        "B0 no entry_valid": replace(base, name="B0 no entry_valid", changed_rules=("entry_valid",)),
        "B0 close trigger soft": replace(base, name="B0 close trigger soft", changed_rules=("close_trigger",)),
        "B0 fresh continuous": replace(
            base,
            name="B0 fresh continuous",
            fresh_rank="continuous",
            changed_rules=("fresh",),
        ),
        "B0 volume soft": replace(base, name="B0 volume soft", changed_rules=("volume",)),
        "B0 volume route-specific": replace(
            base,
            name="B0 volume route-specific",
            volume_policy="route_specific",
            changed_rules=("volume",),
        ),
        "B0 geometry soft": replace(
            base,
            name="B0 geometry soft",
            geometry_eligibility="allow_failure",
            changed_rules=("geometry",),
        ),
        "B0 geometry failure only": replace(base, name="B0 geometry failure only", changed_rules=("geometry",)),
        "B0 pullback dry hard": replace(
            base,
            name="B0 pullback dry hard",
            pullback_dry_policy="hard",
            changed_rules=("pullback_dry",),
        ),
        "B0 pullback dry bonus": replace(base, name="B0 pullback dry bonus", changed_rules=("pullback_dry",)),
        "B0 pullback dry drop": replace(
            base,
            name="B0 pullback dry drop",
            pullback_dry_policy="drop",
            changed_rules=("pullback_dry",),
        ),
        "B0 EPS >=25 hard": replace(
            base,
            name="B0 EPS >=25 hard",
            eps_eligibility="at_least_25",
            changed_rules=("eps",),
        ),
        "B0 EPS >=25 bonus": replace(base, name="B0 EPS >=25 bonus", changed_rules=("eps",)),
        "B0 EPS >=25 drop": replace(
            base,
            name="B0 EPS >=25 drop",
            eps_rank="drop",
            changed_rules=("eps",),
        ),
        "B0 EPS unknown manual-review": replace(
            base,
            name="B0 EPS unknown manual-review",
            eps_eligibility="allow_unknown",
            changed_rules=("eps",),
        ),
        "B0 no industry cover": replace(
            base,
            name="B0 no industry cover",
            industry_cover=False,
            changed_rules=("industry_cover",),
        ),
        "B0 top1": replace(base, name="B0 top1", top_n=1, changed_rules=("topk",)),
    }


def atomic_selector_configs() -> dict[str, SelectorConfig]:
    return {name: cfg for name, cfg in selector_configs().items() if cfg.changed_rules}


def compose_selector_config(name: str, atomic_names: list[str] | tuple[str, ...]) -> SelectorConfig:
    base = selector_configs()["B0_PIT_VERIFIED"]
    values = base.__dict__.copy()
    changed: list[str] = []
    for atomic_name in atomic_names:
        atomic = selector_configs()[atomic_name]
        if len(atomic.changed_rules) != 1:
            raise ValueError(f"{atomic_name} is not a one-rule atomic config")
        target = atomic.changed_rules[0]
        for field in _fields_for_dimension(target):
            values[field] = getattr(atomic, field)
        changed.append(target)
    values["name"] = name
    values["changed_rules"] = tuple(changed)
    values["source_atomic_variants"] = tuple(atomic_names)
    return SelectorConfig(**values)


def enrich_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "pit_eps_yoy_growth" not in out.columns:
        out["pit_eps_yoy_growth"] = pd.NA
    if "pit_eps_state" not in out.columns:
        out["pit_eps_state"] = "UNKNOWN"
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
    if config.name == "B0_REPO_EXACT":
        return _select_production_repo_exact(frame, config)
    selected, _ = _select_parameterized(frame, config)
    return selected


def select_all_weeks(panel: pd.DataFrame, config: SelectorConfig) -> pd.DataFrame:
    chunks = []
    for snapshot, group in panel.groupby("snapshot_date", sort=True):
        selected = select_weekly(group, config)
        if not selected.empty:
            selected["snapshot_date"] = snapshot
            chunks.append(selected)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def candidate_rule_traces(pool: pd.DataFrame, config: SelectorConfig) -> pd.DataFrame:
    frame = enrich_features(pool)
    _, trace = _select_parameterized(frame, config)
    return trace


def audit_atomic_variant(pool: pd.DataFrame, baseline: SelectorConfig, variant: SelectorConfig) -> dict[str, Any]:
    target = variant.changed_rules[0] if len(variant.changed_rules) == 1 else ""
    weekly = []
    for _, group in pool.groupby("snapshot_date", sort=True, dropna=False):
        weekly.append(_audit_atomic_variant_week(group, baseline, variant, target))
    contrast_weeks = sum(row["selection_changed"] for row in weekly)
    return {
        "variant": variant.name,
        "target_rule": target,
        "audited_weeks": len(weekly),
        "actual_target_trace_changes": sum(row["target_changes"] for row in weekly),
        "non_target_trace_violations": sum(row["non_target_violations"] for row in weekly),
        "selection_contrast_weeks": contrast_weeks,
        "baseline_selected_count": sum(row["baseline_selected_count"] for row in weekly),
        "variant_selected_count": sum(row["variant_selected_count"] for row in weekly),
        "treatment_contrast": "OK" if contrast_weeks else "NO_TREATMENT_CONTRAST",
    }


def _audit_atomic_variant_week(
    pool: pd.DataFrame,
    baseline: SelectorConfig,
    variant: SelectorConfig,
    target: str,
) -> dict[str, int]:
    baseline_selected = select_weekly(pool, baseline)
    variant_selected = select_weekly(pool, variant)
    baseline_trace = candidate_rule_traces(pool, baseline).set_index(["snapshot_date", "code"])
    variant_trace = candidate_rule_traces(pool, variant).set_index(["snapshot_date", "code"])
    target_changes = 0
    non_target_violations = 0
    for key in baseline_trace.index.union(variant_trace.index):
        before = json.loads(baseline_trace.loc[key, "dimension_trace_json"])
        after = json.loads(variant_trace.loc[key, "dimension_trace_json"])
        target_changes += int(before.get(target) != after.get(target))
        non_target_violations += sum(
            before.get(dimension) != after.get(dimension)
            for dimension in TRACE_DIMENSIONS
            if dimension != target
        )
    return {
        "target_changes": target_changes,
        "non_target_violations": non_target_violations,
        "selection_changed": int(_selection_keys(baseline_selected) != _selection_keys(variant_selected)),
        "baseline_selected_count": len(baseline_selected),
        "variant_selected_count": len(variant_selected),
    }


def audit_production_b0_replay_pools(pool_root: Path) -> pd.DataFrame:
    rows = []
    repo_config = selector_configs()["B0_REPO_EXACT"]
    for path in sorted(pool_root.glob("*/breakout_follow_pool.csv")):
        pool = pd.read_csv(path)
        if pool.empty:
            continue
        snapshot = path.parent.name
        if "snapshot_date" not in pool.columns:
            pool["snapshot_date"] = snapshot
        pool["snapshot_date"] = pool["snapshot_date"].fillna(snapshot).astype(str)
        production_rows = _item_keys(production.select_skill_industry_eps_known(pool))
        exact_rows = _frame_keys(select_weekly(pool, repo_config))
        parameterized, _ = _select_parameterized(enrich_features(pool), repo_config)
        parameterized_rows = _frame_keys(parameterized)
        rows.append(
            {
                "snapshot_date": snapshot,
                "production_selected_count": len(production_rows),
                "code_order_mismatches": int([row[:2] for row in production_rows] != [row[:2] for row in exact_rows]),
                "reason_code_mismatches": int([row[2] for row in production_rows] != [row[2] for row in exact_rows]),
                "risk_code_mismatches": int([row[3] for row in production_rows] != [row[3] for row in exact_rows]),
                "parameterized_baseline_mismatches": int(production_rows != parameterized_rows),
            }
        )
    return pd.DataFrame(rows)


def production_selected_codes(pool: pd.DataFrame) -> list[tuple[str, int, str, str]]:
    return _item_keys(production.select_skill_industry_eps_known(pool))


def _select_production_repo_exact(frame: pd.DataFrame, config: SelectorConfig) -> pd.DataFrame:
    return _items_to_frame(frame, production.select_skill_industry_eps_known(frame, limit=config.top_n), config.name)


def _select_parameterized(frame: pd.DataFrame, config: SelectorConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates: list[tuple[production.SkillCandidate, dict[str, Any], pd.Series]] = []
    traces: list[dict[str, Any]] = []
    for row_idx, row in frame.iterrows():
        if not production.is_review_universe(row):
            continue
        item, trace = _parameterized_item(row, row_idx, config)
        candidates.append((item, trace, row))
    candidates.sort(key=lambda value: value[0].sort_key)
    for raw_rank, (item, trace, row) in enumerate(candidates, 1):
        item.raw_rank = raw_rank
        traces.append(
            {
                "snapshot_date": str(row.get("snapshot_date", "")),
                "code": item.code,
                "dimension_trace_json": json.dumps(trace["dimensions"], sort_keys=True, separators=(",", ":")),
                "eligible": _eligible(item, row, trace, config),
                "raw_rank": raw_rank,
            }
        )

    selected: list[production.SkillCandidate] = []
    covered: set[str] = set()
    for item, trace, row in candidates:
        if not _eligible(item, row, trace, config):
            continue
        industry_key = item.industry.strip().lower()
        if not industry_key:
            continue
        if config.industry_cover and industry_key in covered:
            continue
        selected.append(item)
        if config.industry_cover:
            covered.add(industry_key)
        if len(selected) >= config.top_n:
            break
    return _items_to_frame(frame, selected, config.name), pd.DataFrame(traces)


def _parameterized_item(row: pd.Series, row_idx: int, config: SelectorConfig) -> tuple[production.SkillCandidate, dict[str, Any]]:
    code = str(row.get("code", "") or "").strip()
    status = production.entry_status(row)
    industry = str(row.get("industry", "") or "").strip()
    cur = production.to_float(row.get("current_vs_ibd_candidate_pct"))
    entry_vol = production.to_float(row.get("ibd_entry_volume_ratio"))
    weekly_vol = production.to_float(row.get("volume_ratio"))
    eps = _effective_eps(row, code, config)
    dist = production.to_float(row.get("dist_to_52w_high_pct"))
    rule = str(row.get("ibd_candidate_rule", "") or "").strip()
    clear_failure = production.clear_geometry_failure(row)

    reasons: list[str] = []
    risks: list[str] = []
    dimensions: dict[str, Any] = {}

    geometry_reasons: list[str] = []
    geometry_risks: list[str] = []
    if clear_failure:
        geometry_risks.append("clear_geometry_failure")
    elif production.geometry_caution(row):
        geometry_reasons.append("geometry_caution_not_failure")
    reasons.extend(geometry_reasons)
    risks.extend(geometry_risks)
    dimensions["geometry"] = {
        "clear_failure": clear_failure,
        "eligibility": config.geometry_eligibility,
        "reasons": geometry_reasons,
        "risks": geometry_risks,
    }

    status_risks = [] if status == "ACTIONABLE" else ["non_actionable_radar_only"]
    risks.extend(status_risks)
    dimensions["status"] = {"value": status, "eligibility": config.status_eligibility, "risks": status_risks}

    fresh_reasons: list[str] = []
    fresh_risks: list[str] = []
    if cur is None:
        fresh_risks.append("freshness_missing")
    elif cur < 0:
        fresh_risks.append("below_candidate_buy_point")
    elif cur <= 5:
        fresh_reasons.append("near_buy_point")
    else:
        fresh_risks.append("extended_from_buy_point")
    reasons.extend(fresh_reasons)
    risks.extend(fresh_risks)
    dimensions["fresh"] = {
        "value": cur,
        "rank_policy": config.fresh_rank,
        "reasons": fresh_reasons,
        "risks": fresh_risks,
    }

    volume_threshold = 1.3 if config.volume_policy == "route_specific" and production.is_pullback_rule(rule) else 1.5
    volume_reasons: list[str] = []
    volume_risks: list[str] = []
    if config.volume_policy != "drop":
        if entry_vol is None:
            volume_risks.append("entry_volume_missing")
        elif entry_vol >= volume_threshold:
            volume_reasons.append("volume_confirms_breakout")
        else:
            volume_risks.append("entry_volume_below_standard")
    reasons.extend(volume_reasons)
    risks.extend(volume_risks)
    dimensions["volume"] = {
        "value": entry_vol,
        "policy": config.volume_policy,
        "threshold": volume_threshold,
        "reasons": volume_reasons,
        "risks": volume_risks,
    }

    eps_reasons: list[str] = []
    if eps is None:
        eps_reasons.append("eps_needs_manual_check")
    elif eps >= 25 and config.eps_rank != "drop":
        eps_reasons.append("eps_acceleration_support")
    reasons.extend(eps_reasons)
    dimensions["eps"] = {
        "value": eps,
        "source": config.eps_source,
        "eligibility": config.eps_eligibility,
        "rank_policy": config.eps_rank,
        "reasons": eps_reasons,
    }

    if weekly_vol is not None and weekly_vol >= 1.3:
        reasons.append("weekly_volume_follow_through")
    if dist is not None and dist > -5:
        reasons.append("near_52w_high")
    dry_reasons: list[str] = []
    dry_risks: list[str] = []
    if production.is_pullback_rule(rule):
        reasons.append("pullback_structure")
        dry = production.to_bool(row.get("pullback_v_is_dry"))
        if config.pullback_dry_policy != "drop":
            if dry is True:
                dry_reasons.append("dry_pullback")
            elif dry is False:
                dry_risks.append("pullback_not_dry")
        reasons.extend(dry_reasons)
        risks.extend(dry_risks)
    else:
        dry = None
    dimensions["pullback_dry"] = {
        "applicable": production.is_pullback_rule(rule),
        "value": dry,
        "policy": config.pullback_dry_policy,
        "reasons": dry_reasons,
        "risks": dry_risks,
    }

    lane = production.lane_for(rule, reasons, risks, status=status)
    evidence_count = sum(
        code_name in reasons
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
        code_name in risks
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
    eps_bucket = 0 if eps is not None and eps >= 25 else 1
    if config.eps_rank == "drop":
        eps_bucket = 0
    entry_volume_sort = -(entry_vol or 0.0) if config.volume_policy != "drop" else 0.0
    sort_key = (
        1 if clear_failure else 0,
        production.LANE_ORDER[lane],
        status_bucket if lane != "constructive_pullback" else min(status_bucket, 0),
        -(evidence_count - risk_count),
        risk_count,
        _fresh_rank_value(cur, config.fresh_rank),
        eps_bucket,
        0 if weekly_vol is not None and weekly_vol >= 1.3 else 1,
        entry_volume_sort,
        code,
        row_idx,
    )
    dimensions["entry_valid"] = {"gate": config.entry_valid_gate, "value": to_bool(row.get("ibd_entry_valid"))}
    dimensions["close_trigger"] = {
        "gate": config.close_trigger_gate,
        "value": to_float(row.get("ibd_entry_close_vs_trigger_pct")),
    }
    dimensions["industry_cover"] = {"enabled": config.industry_cover, "industry": industry.strip().lower()}
    dimensions["topk"] = {"top_n": config.top_n}
    item = production.SkillCandidate(
        code=code,
        raw_rank=0,
        entry_status=status,
        lane=lane,
        industry=industry,
        sort_key=sort_key,
        reason_codes=reasons,
        risk_codes=risks,
        feature_values=production.feature_values(row, eps),
    )
    return item, {"dimensions": dimensions, "effective_eps": eps, "clear_failure": clear_failure}


def _eligible(item: production.SkillCandidate, row: pd.Series, trace: dict[str, Any], config: SelectorConfig) -> bool:
    status = item.entry_status
    if config.status_eligibility == "actionable" and status != "ACTIONABLE":
        return False
    if config.status_eligibility == "actionable_unconfirmed" and status not in {"ACTIONABLE", "UNCONFIRMED"}:
        return False
    if config.geometry_eligibility == "exclude_failure" and trace["clear_failure"]:
        return False
    if "below_candidate_buy_point" in item.risk_codes:
        return False
    eps = trace["effective_eps"]
    if config.eps_eligibility == "known" and eps is None:
        return False
    if config.eps_eligibility == "at_least_25" and (eps is None or eps < 25):
        return False
    if config.entry_valid_gate and to_bool(row.get("ibd_entry_valid")) is not True:
        return False
    close = to_float(row.get("ibd_entry_close_vs_trigger_pct"))
    if config.close_trigger_gate and (close is None or close < 0):
        return False
    if config.pullback_dry_policy == "hard" and production.is_pullback_rule(str(row.get("ibd_candidate_rule", "") or "")):
        if to_bool(row.get("pullback_v_is_dry")) is not True:
            return False
    return True


def _effective_eps(row: pd.Series, code: str, config: SelectorConfig) -> float | None:
    if config.eps_source == "repo":
        return production.row_eps(row, code)
    if str(row.get("pit_eps_state", "") or "").strip().upper() != "VERIFIED":
        return None
    return to_float(row.get("pit_eps_yoy_growth"))


def _fresh_rank_value(cur: float | None, policy: str) -> float:
    if policy == "continuous":
        if cur is None:
            return 30.0
        if cur < 0:
            return 40.0 + abs(cur)
        return min(cur, 20.0)
    return float(production.fresh_bucket(cur))


def _items_to_frame(frame: pd.DataFrame, items: list[production.SkillCandidate], selected_by: str) -> pd.DataFrame:
    by_code = {str(row.get("code", "") or "").strip(): row for _, row in frame.iterrows()}
    rows: list[dict[str, Any]] = []
    for pick_order, item in enumerate(items, 1):
        source = by_code.get(item.code)
        out = source.to_dict() if source is not None else {}
        out.update(
            {
                "code": item.code,
                "pick_order": pick_order,
                "score": None,
                "reason_codes": ";".join(item.reason_codes),
                "risk_codes": ";".join(item.risk_codes),
                "selected_by": selected_by,
                "production_raw_rank": item.raw_rank,
                "production_sort_key": repr(item.sort_key),
            }
        )
        rows.append(out)
    columns = list(frame.columns) + [
        "pick_order",
        "score",
        "reason_codes",
        "risk_codes",
        "selected_by",
        "production_raw_rank",
        "production_sort_key",
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)


def _item_keys(items: list[production.SkillCandidate]) -> list[tuple[str, int, str, str]]:
    return [
        (item.code, order, ";".join(item.reason_codes), ";".join(item.risk_codes))
        for order, item in enumerate(items, 1)
    ]


def _frame_keys(frame: pd.DataFrame) -> list[tuple[str, int, str, str]]:
    if frame.empty:
        return []
    return [
        (str(row["code"]), int(row["pick_order"]), str(row["reason_codes"]), str(row["risk_codes"]))
        for _, row in frame.sort_values("pick_order").iterrows()
    ]


def _selection_keys(frame: pd.DataFrame) -> list[tuple[str, int]]:
    if frame.empty:
        return []
    return [(str(row["code"]), int(row["pick_order"])) for _, row in frame.sort_values("pick_order").iterrows()]


def _fields_for_dimension(dimension: str) -> tuple[str, ...]:
    return {
        "status": ("status_eligibility",),
        "entry_valid": ("entry_valid_gate",),
        "close_trigger": ("close_trigger_gate",),
        "fresh": ("fresh_rank",),
        "volume": ("volume_policy",),
        "geometry": ("geometry_eligibility",),
        "pullback_dry": ("pullback_dry_policy",),
        "eps": ("eps_eligibility", "eps_rank"),
        "industry_cover": ("industry_cover",),
        "topk": ("top_n",),
    }[dimension]


def _pullback_dry_state(rule: str, value: object) -> str:
    if rule not in PULLBACK_RULES:
        return "NOT_APPLICABLE"
    parsed = to_bool(value)
    if parsed is True:
        return "PASS"
    if parsed is False:
        return "FAIL"
    return "UNKNOWN"
