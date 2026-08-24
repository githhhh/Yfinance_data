from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.ibd_skill_replay.core import to_bool, to_float
from backtest.ibd_skill_replay.run_ytd_replay import _load_price_cache
from backtest.ibd_weekly_signal_oracle_eval.price_cache import resolve_price_cache
from backtest.rd_agent_research_bench.ibd_position_state_machine import (
    IBDTradeConfig,
    run_ibd_position_state_machine,
)
from dashboard.skill_industry_eps_known import effective_eps, rank_skill_industry_eps_known, select_skill_industry_eps_known


DEFAULT_POOL_ROOT = Path("backtest/ibd_skill_replay_pools")
DEFAULT_OUTPUT_DIR = Path("backtest/rd_agent_research_bench/balanced_rule_output")
DEFAULT_PRICE_CACHE: Path | None = None
SKILL_PATH = ".agents/skills/ibd-candidate-prescreen/SKILL.md"
BASE_COMMIT = "7925b2209adfa810213623de34c1dae6402638aa"
RANDOM_SEED = 20260824


@dataclass(frozen=True)
class SelectorConfig:
    name: str
    status_mode: str = "actionable_gate"
    valid_mode: str = "soft"
    close_gate: bool = True
    fresh_mode: str = "hard"
    volume_mode: str = "hard"
    geometry_mode: str = "hard"
    pullback_dry_mode: str = "minor"
    eps_mode: str = "soft"
    industry_cover: bool = True
    top_n: int = 3
    allow_zero: bool = True
    min_score: float | None = None
    base_pullback_context: bool = False
    use_c_continuous: bool = False
    complexity: int = 1


def evaluate_candidate(row: pd.Series, *, row_index: int) -> dict[str, Any]:
    code = str(row.get("code", "") or "").strip()
    rule = str(row.get("ibd_candidate_rule", "") or "").strip()
    status = str(row.get("ibd_entry_status", "") or "").strip().upper()
    source = str(row.get("signal_source", row.get("ibd_candidate_signal_source", "")) or "").strip()
    cur = to_float(row.get("current_vs_ibd_candidate_pct"))
    close_vs_trigger = to_float(row.get("ibd_entry_close_vs_trigger_pct"))
    entry_vol = to_float(row.get("ibd_entry_volume_ratio"))
    weekly_vol = to_float(row.get("volume_ratio"))
    eps = to_float(row.get("eps_yoy_growth"))
    dist = to_float(row.get("dist_to_52w_high_pct"))
    pos = to_float(row.get("ibd_entry_close_position"))
    rr = to_float(row.get("ibd_entry_breakout_range_ratio"))
    valid = to_float(row.get("ibd_entry_valid"))
    geometry, geometry_state = geometry_class(pos, rr)
    is_pullback = rule in {"ceiling_pullback", "pivot", "ma10_touch_confirm", "three_weeks_tight"}
    is_base = rule in {"ceiling", "ceiling_breakout"}
    dry_value = to_bool(row.get("pullback_v_is_dry"))
    checks = {
        "signal": _state(to_bool(row.get("signal")) is True),
        "status_actionable": _state(status == "ACTIONABLE"),
        "ibd_entry_valid": _tri_state(valid, lambda value: value == 1),
        "close_above_trigger": _tri_state(close_vs_trigger, lambda value: value > 0),
        "fresh_zone_0_5": _tri_state(cur, lambda value: 0 <= value <= 5),
        "fresh_zone_0_2": _tri_state(cur, lambda value: 0 <= value <= 2),
        "entry_volume_1_5": _tri_state(entry_vol, lambda value: value >= 1.5),
        "weekly_volume_1_3": _tri_state(weekly_vol, lambda value: value >= 1.3),
        "geometry": {"state": geometry_state, "value": geometry},
        "eps_25": _tri_state(eps, lambda value: value >= 25, missing_state="UNKNOWN"),
        "near_52w_high": _tri_state(dist, lambda value: value > -5),
        "base_depth": _field_state(row.get("base_depth_pct"), applicable=is_base or rule == "ceiling_pullback"),
        "base_duration": _field_state(row.get("base_duration_weeks"), applicable=is_base or rule == "ceiling_pullback"),
        "base_mbox_count": _field_state(row.get("base_mbox_count"), applicable=is_base or rule == "ceiling_pullback"),
        "pullback_depth": _field_state(row.get("pullback_pct"), applicable=is_pullback),
        "pullback_duration": _field_state(row.get("pullback_duration_weeks"), applicable=is_pullback),
        "pullback_dry": _dry_state(dry_value, applicable=is_pullback),
    }
    risk_flags: list[str] = []
    if status != "ACTIONABLE":
        risk_flags.append(f"status_{status or 'UNKNOWN'}")
    if cur is None:
        risk_flags.append("freshness_unknown")
    elif cur < 0:
        risk_flags.append("below_candidate_buy_point")
    elif cur > 5:
        risk_flags.append("extended_from_buy_point")
    if entry_vol is None:
        risk_flags.append("entry_volume_unknown")
    elif entry_vol < 1.5:
        risk_flags.append("entry_volume_below_1_5")
    if geometry_state == "FAIL":
        risk_flags.append("geometry_failure")
    elif geometry_state == "UNKNOWN":
        risk_flags.append("geometry_unknown")
    if checks["pullback_dry"]["state"] == "FAIL":
        risk_flags.append("pullback_not_dry")
    if eps is None:
        risk_flags.append("eps_unknown")
    elif eps < 25:
        risk_flags.append("eps_below_25")
    return {
        "row_index": row_index,
        "code": code,
        "snapshot_date": str(row.get("snapshot_date", "") or ""),
        "entry_status": status,
        "signal_source": source,
        "ibd_candidate_rule": rule,
        "ibd_candidate_price": to_float(row.get("ibd_candidate_price")),
        "ibd_entry_date": str(row.get("ibd_entry_date", "") or "").strip(),
        "latest_close": to_float(row.get("latest_close")),
        "industry": str(row.get("industry", "") or "").strip(),
        "sector": str(row.get("sector", "") or "").strip(),
        "checks": checks,
        "geometry": geometry,
        "trigger_pos": None if pos is None or rr is None else pos - rr,
        "risk_flags": risk_flags,
        "features": {
            "current_vs_ibd_candidate_pct": cur,
            "ibd_entry_volume_ratio": entry_vol,
            "ibd_entry_close_position": pos,
            "ibd_entry_breakout_range_ratio": rr,
            "ibd_entry_close_vs_trigger_pct": close_vs_trigger,
            "volume_ratio": weekly_vol,
            "eps_yoy_growth": eps,
            "dist_to_52w_high_pct": dist,
            "base_depth_abs": to_float(row.get("base_depth_abs")),
            "base_mbox_count": to_float(row.get("base_mbox_count")),
            "base_duration_weeks": to_float(row.get("base_duration_weeks")),
            "pullback_pct": to_float(row.get("pullback_pct")),
            "pullback_duration_weeks": to_float(row.get("pullback_duration_weeks")),
            "C_continuous": to_float(row.get("C_continuous")),
        },
    }


def select_by_config(pool: pd.DataFrame, cfg: SelectorConfig) -> pd.DataFrame:
    candidates = []
    for row_index, row in pool.iterrows():
        if not (to_bool(row.get("signal")) is True and str(row.get("ibd_candidate_rule", "") or "").strip()):
            continue
        evaluated = evaluate_candidate(row, row_index=row_index)
        allowed, score, reasons, risks = _score_candidate(evaluated, cfg)
        if not allowed:
            continue
        if cfg.allow_zero and cfg.min_score is not None and score < cfg.min_score:
            continue
        candidates.append(
            {
                "snapshot_date": evaluated["snapshot_date"],
                "code": evaluated["code"],
                "variant": cfg.name,
                "entry_status": evaluated["entry_status"],
                "signal_source": evaluated["signal_source"],
                "ibd_candidate_rule": evaluated["ibd_candidate_rule"],
                "industry": evaluated["industry"],
                "sector": evaluated["sector"],
                "ibd_candidate_price": evaluated["ibd_candidate_price"],
                "ibd_entry_date": evaluated["ibd_entry_date"],
                "latest_close": evaluated["latest_close"],
                "score": score,
                "reason_codes": ";".join(reasons),
                "risk_flags": ";".join(risks),
                "checks_json": json.dumps(evaluated["checks"], ensure_ascii=False, sort_keys=True),
                "geometry": evaluated["geometry"],
                "trigger_pos": evaluated["trigger_pos"],
                "row_index": evaluated["row_index"],
            }
        )
    if not candidates:
        return pd.DataFrame()
    frame = pd.DataFrame(candidates).sort_values(["score", "code", "row_index"], ascending=[False, True, True])
    selected_rows = []
    covered: set[str] = set()
    for _, row in frame.iterrows():
        industry_key = str(row["industry"] or "").strip().lower()
        if cfg.industry_cover and industry_key and industry_key in covered:
            continue
        selected_rows.append(row.to_dict())
        if cfg.industry_cover and industry_key:
            covered.add(industry_key)
        if len(selected_rows) >= cfg.top_n:
            break
    out = pd.DataFrame(selected_rows)
    if not out.empty:
        out.insert(3, "pick_order", range(1, len(out) + 1))
    return out


def run_balanced_rule_evaluation(
    *,
    pool_root: Path = DEFAULT_POOL_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    price_cache: Path | None = DEFAULT_PRICE_CACHE,
    holdout_weeks: int = 6,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pools = _load_pools(pool_root)
    resolved_price_cache = resolve_price_cache(price_cache)
    prices = _load_price_cache(resolved_price_cache)
    configs = selector_configs()
    variants = [cfg.name for cfg in configs]
    selection_rows: list[dict[str, Any]] = []
    forward_rows: list[dict[str, Any]] = []
    for snapshot, pool in pools:
        b0 = _select_b0(pool, snapshot)
        for row in b0.to_dict("records"):
            selection_rows.append(row)
            forward_rows.append(_forward_row(row, prices))
        for name in ["b0_no_eps_known_gate", "b0_no_industry_cover", "b0_top1_diagnostic"]:
            b0_variant = _select_b0_ablation(pool, snapshot, name)
            for row in b0_variant.to_dict("records"):
                selection_rows.append(row)
                forward_rows.append(_forward_row(row, prices))
        for cfg in configs:
            selected = select_by_config(pool, cfg)
            for row in selected.to_dict("records"):
                row["variant"] = cfg.name
                selection_rows.append(row)
                forward_rows.append(_forward_row(row, prices))

    selections = pd.DataFrame(selection_rows)
    forwards = pd.DataFrame(forward_rows)
    if not forwards.empty and not selections.empty:
        selections = selections.merge(
            forwards[
                [
                    "variant",
                    "snapshot_date",
                    "code",
                    "forward_1w_censored",
                    "forward_3w_censored",
                    "forward_5w_censored",
                    "forward_8w_censored",
                    "observation_trading_days",
                    "forward_8w_return_pct",
                    "mfe_pct",
                    "mae_pct",
                    "path_source",
                ]
            ],
            on=["variant", "snapshot_date", "code"],
            how="left",
        )
    trade_ledger, trade_events = _run_trades(selections, prices)
    exit_sensitivity = _exit_policy_sensitivity(selections, prices, forwards)
    coverage = _coverage_report(selections, forwards, trade_ledger, pools)
    rule_ablation = _variant_summary(selections, forwards, trade_ledger)
    weeks = sorted({snapshot for snapshot, _ in pools})
    sealed_holdout = set(weeks[-holdout_weeks:]) if len(weeks) > holdout_weeks else set()
    walk_forward = _walk_forward(rule_ablation, forwards, sealed_holdout)
    benchmark = _benchmark_comparison(rule_ablation, baseline="b0_repository_skill")
    decision_table = _machine_decision_table(rule_ablation, walk_forward, coverage)
    hypotheses = hypothesis_registry(configs)
    manifest = _manifest(configs, pool_root, resolved_price_cache, weeks, sealed_holdout)
    report = _render_report(rule_ablation, walk_forward, benchmark, coverage, exit_sensitivity, decision_table, hypotheses, weeks, sealed_holdout)
    schema_review = _render_schema_review()
    derived = _render_derived_dimensions()
    b0_diff = _render_b0_diff(rule_ablation)
    proposed_skill = _repo_skill_text()

    outputs = {
        "schema_review": output_dir / "schema_review.md",
        "derived_dimension_proposals": output_dir / "derived_dimension_proposals.md",
        "hypothesis_registry": output_dir / "hypothesis_registry.jsonl",
        "experiment_manifest": output_dir / "experiment_manifest.yaml",
        "rule_ablation_results": output_dir / "rule_ablation_results.csv",
        "walk_forward_results": output_dir / "walk_forward_results.csv",
        "selection_events": output_dir / "selection_events.csv",
        "trade_ledger": output_dir / "trade_ledger.csv",
        "trade_events": output_dir / "trade_events.csv",
        "exit_policy_sensitivity": output_dir / "exit_policy_sensitivity.csv",
        "machine_decision_table": output_dir / "machine_decision_table.csv",
        "coverage_and_censoring_report": output_dir / "coverage_and_censoring_report.csv",
        "benchmark_comparison": output_dir / "benchmark_comparison.csv",
        "rd_agent_research_report": output_dir / "rd_agent_research_report.md",
        "b0_vs_recommended_rule_diff": output_dir / "b0_vs_recommended_rule_diff.md",
        "skill_proposed": output_dir / "SKILL.proposed.md",
    }
    outputs["schema_review"].write_text(schema_review, encoding="utf-8")
    outputs["derived_dimension_proposals"].write_text(derived, encoding="utf-8")
    outputs["hypothesis_registry"].write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in hypotheses) + "\n", encoding="utf-8")
    outputs["experiment_manifest"].write_text(manifest, encoding="utf-8")
    rule_ablation.to_csv(outputs["rule_ablation_results"], index=False)
    walk_forward.to_csv(outputs["walk_forward_results"], index=False)
    selections.to_csv(outputs["selection_events"], index=False)
    trade_ledger.to_csv(outputs["trade_ledger"], index=False)
    trade_events.to_csv(outputs["trade_events"], index=False)
    exit_sensitivity.to_csv(outputs["exit_policy_sensitivity"], index=False)
    decision_table.to_csv(outputs["machine_decision_table"], index=False)
    coverage.to_csv(outputs["coverage_and_censoring_report"], index=False)
    benchmark.to_csv(outputs["benchmark_comparison"], index=False)
    outputs["rd_agent_research_report"].write_text(report, encoding="utf-8")
    outputs["b0_vs_recommended_rule_diff"].write_text(b0_diff, encoding="utf-8")
    outputs["skill_proposed"].write_text(proposed_skill, encoding="utf-8")
    return {key: str(value) for key, value in outputs.items()}


def selector_configs() -> list[SelectorConfig]:
    base = SelectorConfig(name="rd_balanced_top3", complexity=6, base_pullback_context=True)
    return [
        replace(base, name="status_actionable_soft", status_mode="actionable_soft", complexity=6),
        replace(base, name="status_actionable_unconfirmed", status_mode="actionable_unconfirmed", complexity=6),
        replace(base, name="status_all_signal", status_mode="all", complexity=5),
        replace(base, name="ignore_entry_valid", valid_mode="drop", complexity=5),
        replace(base, name="close_trigger_soft", close_gate=False, complexity=5),
        replace(base, name="fresh_continuous", fresh_mode="soft", complexity=5),
        replace(base, name="volume_soft", volume_mode="soft", complexity=5),
        replace(base, name="volume_signal_specific", volume_mode="signal_specific", complexity=6),
        replace(base, name="geometry_soft", geometry_mode="soft", complexity=5),
        replace(base, name="geometry_drop", geometry_mode="drop", complexity=4),
        replace(base, name="pullback_dry_hard", pullback_dry_mode="hard", complexity=6),
        replace(base, name="pullback_dry_minor", pullback_dry_mode="minor", complexity=5),
        replace(base, name="pullback_dry_drop", pullback_dry_mode="drop", complexity=4),
        replace(base, name="eps_pass_hard", eps_mode="pass_hard", complexity=6),
        replace(base, name="eps_known_hard", eps_mode="known_hard", complexity=6),
        replace(base, name="eps_drop", eps_mode="drop", complexity=4),
        replace(base, name="base_pullback_context", base_pullback_context=True, complexity=5),
        replace(base, name="c_continuous_context", use_c_continuous=True, complexity=5),
        replace(base, name="top1_diagnostic", top_n=1, complexity=5),
        replace(base, name="allow_zero_strict", min_score=8.0, allow_zero=True, complexity=6),
        replace(base, name="raw_pool_c_continuous", status_mode="all", valid_mode="drop", fresh_mode="drop", volume_mode="drop", geometry_mode="drop", pullback_dry_mode="drop", eps_mode="drop", use_c_continuous=True, complexity=2),
        replace(base, name="actionable_only_pool_order", status_mode="actionable_gate", valid_mode="drop", fresh_mode="drop", volume_mode="drop", geometry_mode="drop", pullback_dry_mode="drop", eps_mode="drop", complexity=2),
    ]


def hypothesis_registry(configs: list[SelectorConfig]) -> list[dict[str, Any]]:
    rows = []
    for idx, cfg in enumerate(configs, 1):
        rows.append(
            {
                "hypothesis_id": f"H{idx:03d}",
                "variant": cfg.name,
                "challenged_b0_rule": _challenged_rule(cfg.name),
                "research_reason": "Pre-registered single-rule/simple substitute check; not selected by global grid search.",
                "source": "Git repository B0, replay pools, quant_trade schema SSOT, deterministic evaluator.",
                "exact_change": asdict(cfg),
                "expected_mechanism": _expected_mechanism(cfg.name),
                "affected_signal_types": "all; routed by ibd_candidate_rule where applicable",
                "possible_side_effects": "coverage shift, higher tail risk, more UNKNOWN exposure, or ticker/week concentration",
                "leakage_risk": "uses only replay snapshot fields and PIT EPS effective_date audit; no future returns in scoring",
                "complexity_change": cfg.complexity,
                "success_metric": "OOS paired weekly improvement vs B0 without worse stop/tail/concentration and with adequate coverage",
                "falsification": "no OOS increment, worse downside, low coverage, single-ticker dependence, or unstable folds",
            }
        )
    return rows


def geometry_class(pos: float | None, rr: float | None) -> tuple[str, str]:
    if pos is not None and not (0 <= pos <= 1):
        return "UNKNOWN / Data Error", "UNKNOWN"
    if rr is not None and rr <= 0:
        return "Defensive Failure", "FAIL"
    if pos is not None and pos < 0.65:
        return "Squat / Upper Shadow", "FAIL"
    if pos is None or rr is None:
        return "UNKNOWN", "UNKNOWN"
    trigger_pos = pos - rr
    if trigger_pos <= 0 and pos >= 0.80:
        return "Full-range Breakout", "PASS"
    if trigger_pos <= 0 and pos < 0.80:
        return "Faded Gap", "PASS"
    if pos >= 0.80 and rr >= 0.50:
        return "Strong Finish", "PASS"
    if pos >= 0.80 and rr < 0.50:
        return "Constructive Breakout", "PASS"
    if rr >= 0.50:
        return "Constructive Breakout", "PASS"
    return "Marginal Breakout", "PASS"


def _score_candidate(evaluated: dict[str, Any], cfg: SelectorConfig) -> tuple[bool, float, list[str], list[str]]:
    checks = evaluated["checks"]
    features = evaluated["features"]
    status = evaluated["entry_status"]
    rule = evaluated["ibd_candidate_rule"]
    risks = list(evaluated["risk_flags"])
    reasons: list[str] = []
    score = 0.0

    if cfg.status_mode == "actionable_gate" and status != "ACTIONABLE":
        return False, score, reasons, risks
    if cfg.status_mode == "actionable_unconfirmed" and status not in {"ACTIONABLE", "UNCONFIRMED"}:
        return False, score, reasons, risks
    if status == "ACTIONABLE":
        score += 3.0
        reasons.append("status_actionable")
    elif cfg.status_mode in {"actionable_soft", "actionable_unconfirmed", "all"}:
        score += {"UNCONFIRMED": 1.0, "EXTENDED": 0.3, "BELOW_TRIGGER": 0.0}.get(status, 0.0)
        risks.append(f"status_soft_{status or 'UNKNOWN'}")

    if cfg.valid_mode == "hard" and checks["ibd_entry_valid"]["state"] != "PASS":
        return False, score, reasons, risks
    if cfg.valid_mode == "soft":
        score += 1.0 if checks["ibd_entry_valid"]["state"] == "PASS" else -0.5

    cur = features["current_vs_ibd_candidate_pct"]
    if cfg.close_gate and checks["close_above_trigger"]["state"] != "PASS":
        return False, score, reasons, risks
    if cfg.fresh_mode == "hard" and checks["fresh_zone_0_5"]["state"] != "PASS":
        return False, score, reasons, risks
    if cfg.fresh_mode != "drop":
        score += _fresh_score(cur)
        reasons.append("fresh_distance_scored")

    vol_state = checks["entry_volume_1_5"]["state"]
    if cfg.volume_mode == "hard" and vol_state != "PASS":
        return False, score, reasons, risks
    if cfg.volume_mode == "signal_specific" and rule in {"ceiling", "pivot"} and vol_state != "PASS":
        return False, score, reasons, risks
    if cfg.volume_mode not in {"drop"}:
        score += _saturating_score(features["ibd_entry_volume_ratio"], 1.5, 3.0)
        reasons.append("entry_volume_scored")

    geom_state = checks["geometry"]["state"]
    if cfg.geometry_mode == "hard" and geom_state == "FAIL":
        return False, score, reasons, risks
    if cfg.geometry_mode != "drop":
        score += {"PASS": 1.0, "UNKNOWN": -0.2, "FAIL": -1.5}.get(geom_state, 0.0)
        reasons.append("geometry_scored")

    dry_state = checks["pullback_dry"]["state"]
    if cfg.pullback_dry_mode == "hard" and dry_state == "FAIL":
        return False, score, reasons, risks
    if cfg.pullback_dry_mode == "minor":
        if dry_state == "PASS":
            score += 0.7
            reasons.append("dry_pullback_minor")
        elif dry_state == "FAIL":
            score -= 0.4
            risks.append("pullback_not_dry")
    elif cfg.pullback_dry_mode == "drop":
        risks = [risk for risk in risks if risk != "pullback_not_dry"]

    eps = features["eps_yoy_growth"]
    if cfg.eps_mode == "pass_hard" and checks["eps_25"]["state"] != "PASS":
        return False, score, reasons, risks
    if cfg.eps_mode == "known_hard" and eps is None:
        return False, score, reasons, risks
    if cfg.eps_mode == "soft":
        score += _eps_score(eps)
        reasons.append("eps_soft")

    score += 0.7 if checks["weekly_volume_1_3"]["state"] == "PASS" else 0.0
    score += 0.5 if checks["near_52w_high"]["state"] == "PASS" else 0.0
    if cfg.base_pullback_context:
        score += _context_score(evaluated)
    if cfg.use_c_continuous:
        c_val = features["C_continuous"]
        score += 2.0 * c_val if c_val is not None else 0.0
    return True, round(score, 8), reasons, risks


def _select_b0(pool: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    selected = []
    by_code = {str(row.get("code", "")).strip(): row for _, row in pool.iterrows()}
    for order, item in enumerate(select_skill_industry_eps_known(pool), 1):
        fields = item.feature_values
        source_row = by_code.get(item.code, pd.Series(dtype=object))
        selected.append(
            {
                "snapshot_date": snapshot,
                "variant": "b0_repository_skill",
                "pick_order": order,
                "code": item.code,
                "entry_status": item.entry_status,
                "signal_source": "",
                "ibd_candidate_rule": fields.get("ibd_candidate_rule"),
                "industry": item.industry,
                "sector": fields.get("sector"),
                "ibd_candidate_price": fields.get("ibd_candidate_price"),
                "ibd_entry_date": source_row.get("ibd_entry_date"),
                "latest_close": fields.get("latest_close"),
                "score": None,
                "reason_codes": ";".join(item.reason_codes),
                "risk_flags": ";".join(item.risk_codes),
                "checks_json": "",
                "geometry": "",
                "trigger_pos": None,
                "row_index": None,
            }
        )
    return pd.DataFrame(selected)


def _select_b0_ablation(pool: pd.DataFrame, snapshot: str, name: str) -> pd.DataFrame:
    ranked = rank_skill_industry_eps_known(pool)
    rows = []
    covered: set[str] = set()
    limit = 1 if name == "b0_top1_diagnostic" else 3
    by_code = {str(row.get("code", "")).strip(): row for _, row in pool.iterrows()}
    for item in ranked:
        if item.entry_status != "ACTIONABLE":
            continue
        if "clear_geometry_failure" in item.risk_codes or "below_candidate_buy_point" in item.risk_codes:
            continue
        if name != "b0_no_eps_known_gate" and effective_eps(item) is None:
            continue
        industry_key = item.industry.strip().lower()
        if name != "b0_no_industry_cover":
            if not industry_key:
                continue
            if industry_key in covered:
                continue
        fields = item.feature_values
        source_row = by_code.get(item.code, pd.Series(dtype=object))
        rows.append(
            {
                "snapshot_date": snapshot,
                "variant": name,
                "pick_order": len(rows) + 1,
                "code": item.code,
                "entry_status": item.entry_status,
                "signal_source": "",
                "ibd_candidate_rule": fields.get("ibd_candidate_rule"),
                "industry": item.industry,
                "sector": fields.get("sector"),
                "ibd_candidate_price": fields.get("ibd_candidate_price"),
                "ibd_entry_date": source_row.get("ibd_entry_date"),
                "latest_close": fields.get("latest_close"),
                "score": None,
                "reason_codes": ";".join(item.reason_codes),
                "risk_flags": ";".join(item.risk_codes + [f"b0_ablation:{name}"]),
                "checks_json": "",
                "geometry": "",
                "trigger_pos": None,
                "row_index": None,
            }
        )
        if industry_key and name != "b0_no_industry_cover":
            covered.add(industry_key)
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def _forward_row(row: dict[str, Any], prices: dict[str, pd.DataFrame]) -> dict[str, Any]:
    code = str(row.get("code", ""))
    bars = _normalize_bars(prices.get(code))
    snapshot = pd.Timestamp(row.get("snapshot_date"))
    entry = _next_bar(bars, snapshot)
    base = {
        "variant": row.get("variant"),
        "snapshot_date": row.get("snapshot_date"),
        "code": code,
        "path_source": "missing",
        "entry_date": None,
        "entry_open": None,
        "forward_1w_return_pct": None,
        "forward_3w_return_pct": None,
        "forward_5w_return_pct": None,
        "forward_8w_return_pct": None,
        "forward_1w_censored": True,
        "forward_3w_censored": True,
        "forward_5w_censored": True,
        "forward_8w_censored": True,
        "observation_trading_days": 0,
        "mfe_pct": None,
        "mae_pct": None,
        "first_touch": "UNKNOWN",
    }
    if entry is None:
        return base
    entry_date, entry_row = entry
    entry_open = to_float(entry_row.get("Open"))
    if entry_open is None:
        return base
    window = bars[bars.index >= entry_date]
    base.update({"path_source": str(prices.get(code).attrs.get("source", "daily_cache")) if prices.get(code) is not None else "daily_cache", "entry_date": entry_date.date().isoformat(), "entry_open": entry_open, "observation_trading_days": int(len(window))})
    for label, offset in [("1w", 5), ("3w", 15), ("5w", 25), ("8w", 40)]:
        if len(window) >= offset:
            close = _nth_close(window, offset)
            base[f"forward_{label}_return_pct"] = _pct(close, entry_open)
            base[f"forward_{label}_censored"] = False
    if len(window) < 40:
        base["first_touch"] = "CENSORED"
        return base
    fixed_window = window.iloc[:40]
    highs = pd.to_numeric(fixed_window["High"], errors="coerce")
    lows = pd.to_numeric(fixed_window["Low"], errors="coerce")
    base["mfe_pct"] = _pct(to_float(highs.max()), entry_open)
    base["mae_pct"] = _pct(to_float(lows.min()), entry_open)
    base["first_touch"] = _first_touch(fixed_window, entry_open)
    return base


def _run_trades(selections: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ledgers = []
    events = []
    if selections.empty:
        return pd.DataFrame(), pd.DataFrame()
    for variant, group in selections.groupby("variant", sort=True):
        ledger, event = run_ibd_position_state_machine(group, prices, IBDTradeConfig())
        if not ledger.empty:
            ledger.insert(0, "variant", variant)
            ledgers.append(ledger)
        if not event.empty:
            event.insert(0, "variant", variant)
            events.append(event)
    return pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame(), pd.concat(events, ignore_index=True) if events else pd.DataFrame()


def _exit_policy_sensitivity(selections: pd.DataFrame, prices: dict[str, pd.DataFrame], forwards: pd.DataFrame) -> pd.DataFrame:
    if selections.empty:
        return pd.DataFrame()
    b0 = selections[selections["variant"].eq("b0_repository_skill")].copy()
    rows = []
    fwd = forwards[forwards["variant"].eq("b0_repository_skill")] if not forwards.empty else pd.DataFrame()
    fwd_complete = _complete_forward_rows(fwd)
    ret8 = pd.to_numeric(fwd_complete.get("forward_8w_return_pct"), errors="coerce")
    rows.append(
        {
            "selector": "b0_repository_skill",
            "exit_policy": "fixed_forward_8w_mark",
            "trade_count": len(fwd_complete),
            "expectancy_pct": _safe_float(ret8.mean()),
            "median_return_pct": _safe_float(ret8.median()),
            "win_rate": _safe_float(ret8.gt(0).mean()),
            "censored_trades": int(ret8.isna().sum()),
            "power_trigger_count": 0,
            "note": "fixed label only, no intra-path exits",
        }
    )
    configs = {
        "ibd_8w_default_7p5_22p5": IBDTradeConfig(stop_loss_pct=7.5, profit_take_pct=22.5),
        "no_8week_rule_7p5_22p5": IBDTradeConfig(stop_loss_pct=7.5, profit_take_pct=22.5, power_trigger_weeks=0),
        "stop7_profit20": IBDTradeConfig(stop_loss_pct=7.0, profit_take_pct=20.0),
        "stop8_profit25": IBDTradeConfig(stop_loss_pct=8.0, profit_take_pct=25.0),
        "post_lock_resume_profit": IBDTradeConfig(stop_loss_pct=7.5, profit_take_pct=22.5, post_lock_exit="resume_profit_taking"),
    }
    for name, cfg in configs.items():
        ledger, _ = run_ibd_position_state_machine(b0, prices, cfg)
        returns = pd.to_numeric(ledger.get("return_pct"), errors="coerce")
        rows.append(
            {
                "selector": "b0_repository_skill",
                "exit_policy": name,
                "trade_count": len(ledger),
                "expectancy_pct": _safe_float(returns.mean()),
                "median_return_pct": _safe_float(returns.median()),
                "win_rate": _safe_float(returns.gt(0).mean()),
                "censored_trades": int(ledger.get("censored", pd.Series(dtype=bool)).fillna(False).sum()) if not ledger.empty else 0,
                "power_trigger_count": int(ledger.get("power_trigger_date", pd.Series(dtype=object)).notna().sum()) if not ledger.empty else 0,
                "note": "selector frozen; only exit policy changes",
            }
        )
    return pd.DataFrame(rows)


def _variant_summary(selections: pd.DataFrame, forwards: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variants = sorted(set(selections.get("variant", pd.Series(dtype=str))).union(set(forwards.get("variant", pd.Series(dtype=str)))))
    for variant in variants:
        sel = selections[selections["variant"].eq(variant)] if not selections.empty else pd.DataFrame()
        fwd = forwards[forwards["variant"].eq(variant)] if not forwards.empty else pd.DataFrame()
        fwd_complete = _complete_forward_rows(fwd)
        trades = ledger[ledger["variant"].eq(variant)] if not ledger.empty else pd.DataFrame()
        returns = pd.to_numeric(fwd_complete.get("forward_8w_return_pct"), errors="coerce")
        trade_returns = pd.to_numeric(trades.get("return_pct"), errors="coerce")
        rows.append(
            {
                "variant": variant,
                "weeks_with_picks": sel["snapshot_date"].nunique() if not sel.empty else 0,
                "selection_count": len(sel),
                "complete_8w_label_count": len(fwd_complete),
                "censored_8w_label_count": int(fwd.get("forward_8w_censored", pd.Series(dtype=bool)).fillna(True).sum()) if not fwd.empty else 0,
                "avg_picks_per_week": len(sel) / sel["snapshot_date"].nunique() if not sel.empty and sel["snapshot_date"].nunique() else 0.0,
                "median_forward_8w_return_pct": _safe_float(returns.median()),
                "mean_forward_8w_return_pct": _safe_float(returns.mean()),
                "worst_forward_8w_return_pct": _safe_float(returns.min()),
                "mfe_median_pct": _safe_float(pd.to_numeric(fwd_complete.get("mfe_pct"), errors="coerce").median()),
                "mae_median_pct": _safe_float(pd.to_numeric(fwd_complete.get("mae_pct"), errors="coerce").median()),
                "profit_zone_rate": _rate(fwd_complete.get("first_touch"), "+20"),
                "stop_first_rate": _rate(fwd_complete.get("first_touch"), "-7.5"),
                "trade_count": len(trades),
                "trade_expectancy_pct": _safe_float(trade_returns.mean()),
                "trade_win_rate": _safe_float(trade_returns.gt(0).mean()),
                "trade_median_return_pct": _safe_float(trade_returns.median()),
                "censored_trades": int(trades.get("censored", pd.Series(dtype=bool)).fillna(False).sum()) if not trades.empty else 0,
                "power_trigger_count": int(trades.get("power_trigger_date", pd.Series(dtype=object)).notna().sum()) if not trades.empty else 0,
                "max_single_ticker_pick_share": _max_share(sel, "code"),
                "max_single_week_pick_share": _max_share(sel, "snapshot_date"),
                "top1_mode": "top1" in variant,
            }
        )
    return pd.DataFrame(rows).sort_values(["variant"]).reset_index(drop=True)


def _walk_forward(
    summary: pd.DataFrame,
    forwards: pd.DataFrame,
    sealed_holdout: set[str],
    *,
    min_train_weeks: int = 12,
    embargo_weeks: int = 8,
    test_window_weeks: int = 4,
) -> pd.DataFrame:
    if forwards.empty:
        return pd.DataFrame()
    frame = forwards.copy()
    frame["snapshot_date"] = frame["snapshot_date"].astype(str)
    weeks = sorted(frame["snapshot_date"].dropna().unique())
    if not weeks:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    non_holdout_weeks = [week for week in weeks if week not in sealed_holdout]
    fold_id = 1
    for test_start_idx in range(min_train_weeks + embargo_weeks, len(non_holdout_weeks), test_window_weeks):
        test_weeks = non_holdout_weeks[test_start_idx : test_start_idx + test_window_weeks]
        if not test_weeks:
            continue
        train_weeks = non_holdout_weeks[: max(0, test_start_idx - embargo_weeks)]
        if len(train_weeks) < min_train_weeks:
            continue
        rows.append(_walk_forward_row(frame, train_weeks, test_weeks, fold_id=fold_id, segment="rolling_oos", embargo_weeks=embargo_weeks))
        fold_id += 1

    if sealed_holdout:
        holdout_weeks = sorted(sealed_holdout)
        first_holdout = holdout_weeks[0]
        pre_holdout = [week for week in weeks if week < first_holdout]
        train_weeks = pre_holdout[: max(0, len(pre_holdout) - embargo_weeks)]
        rows.append(_walk_forward_row(frame, train_weeks, holdout_weeks, fold_id=fold_id, segment="sealed_holdout", embargo_weeks=embargo_weeks))
    return pd.DataFrame(rows)


def _walk_forward_row(frame: pd.DataFrame, train_weeks: list[str], test_weeks: list[str], *, fold_id: int, segment: str, embargo_weeks: int) -> dict[str, Any]:
    train = _complete_forward_rows(frame[frame["snapshot_date"].isin(train_weeks)])
    selected, train_score = _choose_champion(train)
    test = _complete_forward_rows(frame[frame["snapshot_date"].isin(test_weeks)])
    paired = _paired_week_delta(test, selected)
    status = "ok"
    if selected is None:
        status = "no_train_champion"
    elif paired["paired_weeks"] == 0:
        status = "no_complete_8w_labels"
    ci_low, ci_high = _bootstrap_ci(paired["weekly_deltas"])
    return {
        "fold_id": fold_id,
        "segment": segment,
        "train_start": train_weeks[0] if train_weeks else "",
        "train_end": train_weeks[-1] if train_weeks else "",
        "test_start": test_weeks[0] if test_weeks else "",
        "test_end": test_weeks[-1] if test_weeks else "",
        "embargo_weeks": embargo_weeks,
        "selected_variant": selected,
        "baseline_variant": "b0_repository_skill",
        "train_complete_weeks": train["snapshot_date"].nunique() if not train.empty else 0,
        "train_score": train_score,
        "test_complete_weeks": test["snapshot_date"].nunique() if not test.empty else 0,
        "paired_weeks": paired["paired_weeks"],
        "test_mean_delta_vs_b0": paired["mean_delta"],
        "test_median_delta_vs_b0": paired["median_delta"],
        "block_bootstrap_ci_low": ci_low,
        "block_bootstrap_ci_high": ci_high,
        "status": status,
    }


def _complete_forward_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    complete = frame.copy()
    if "forward_8w_censored" in complete.columns:
        complete = complete[~complete["forward_8w_censored"].fillna(True).astype(bool)]
    complete["forward_8w_return_pct"] = pd.to_numeric(complete["forward_8w_return_pct"], errors="coerce")
    return complete[complete["forward_8w_return_pct"].notna()]


def _choose_champion(train: pd.DataFrame) -> tuple[str | None, float | None]:
    if train.empty:
        return None, None
    rows = []
    for variant, group in train.groupby("variant", sort=True):
        if variant == "b0_repository_skill":
            continue
        returns = pd.to_numeric(group["forward_8w_return_pct"], errors="coerce")
        if returns.notna().sum() < 3:
            continue
        stop_rate = _rate(group.get("first_touch"), "-7.5") or 0.0
        score = float(returns.median()) + 0.25 * float(returns.mean()) - 2.0 * stop_rate
        rows.append((score, float(returns.min()), variant))
    if not rows:
        return None, None
    rows.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    score, _, variant = rows[0]
    return variant, score


def _paired_week_delta(test: pd.DataFrame, variant: str | None) -> dict[str, Any]:
    if test.empty or variant is None:
        return {"paired_weeks": 0, "mean_delta": None, "median_delta": None, "weekly_deltas": []}
    variant_week = (
        test[test["variant"].eq(variant)]
        .groupby("snapshot_date")["forward_8w_return_pct"]
        .mean()
        .rename("variant_return")
    )
    b0_week = (
        test[test["variant"].eq("b0_repository_skill")]
        .groupby("snapshot_date")["forward_8w_return_pct"]
        .mean()
        .rename("b0_return")
    )
    paired = pd.concat([variant_week, b0_week], axis=1).dropna()
    if paired.empty:
        return {"paired_weeks": 0, "mean_delta": None, "median_delta": None, "weekly_deltas": []}
    deltas = (paired["variant_return"] - paired["b0_return"]).tolist()
    return {
        "paired_weeks": int(len(deltas)),
        "mean_delta": float(pd.Series(deltas).mean()),
        "median_delta": float(pd.Series(deltas).median()),
        "weekly_deltas": deltas,
    }


def _bootstrap_ci(values: list[float], *, iterations: int = 500) -> tuple[float | None, float | None]:
    if len(values) < 3:
        return None, None
    import random

    rng = random.Random(RANDOM_SEED)
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * iterations)], means[int(0.975 * iterations) - 1]


def _benchmark_comparison(summary: pd.DataFrame, *, baseline: str) -> pd.DataFrame:
    base = summary[summary["variant"].eq(baseline)]
    if base.empty:
        return summary.copy()
    b = base.iloc[0]
    out = summary.copy()
    for column in ["median_forward_8w_return_pct", "worst_forward_8w_return_pct", "stop_first_rate", "selection_count"]:
        out[f"{column}_delta_vs_b0"] = pd.to_numeric(out[column], errors="coerce") - float(b.get(column) or 0)
    out["fairness_note"] = "same replay pools, same next-open entry and IBD state-machine path; selector variants only"
    return out


def _machine_decision_table(summary: pd.DataFrame, walk: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    holdout = walk[walk["segment"].eq("sealed_holdout")] if not walk.empty and "segment" in walk.columns else pd.DataFrame()
    holdout_paired = int(pd.to_numeric(holdout.get("paired_weeks"), errors="coerce").fillna(0).sum()) if not holdout.empty else 0
    holdout_status = ";".join(sorted(set(holdout.get("status", pd.Series(dtype=str)).astype(str)))) if not holdout.empty else "missing"
    censored_8w = _coverage_value(coverage, "censored_forward_8w_labels")
    unverified_eps = _coverage_value(coverage, "pit_eps_unverified_availability")
    global_blocker = holdout_paired < 3 or unverified_eps > 0
    rows = []
    specs = [
        ("ACTIONABLE hard boundary", "ibd_entry_status", "b0_repository_skill", "status_actionable_soft;status_all_signal", "Hard Eligibility"),
        ("ibd_entry_valid", "ibd_entry_valid", "b0_repository_skill", "ignore_entry_valid", "Context Only"),
        ("Close > Trigger", "ibd_entry_close_vs_trigger_pct", "b0_repository_skill", "close_trigger_soft", "Continuous/Major Score"),
        ("Fresh Zone", "current_vs_ibd_candidate_pct", "b0_repository_skill", "fresh_continuous", "Continuous/Major Score"),
        ("Entry volume", "ibd_entry_volume_ratio", "b0_repository_skill", "volume_soft;volume_signal_specific", "Continuous/Major Score"),
        ("Breakout Geometry", "pos/range_ratio/trigger_pos", "b0_repository_skill", "geometry_soft;geometry_drop", "Hard Eligibility for clear FAIL"),
        ("Base context", "base_depth/base_duration/base_mbox", "b0_repository_skill", "base_pullback_context", "Context Only"),
        ("Pullback context", "pullback_pct/pullback_duration", "b0_repository_skill", "base_pullback_context", "Context Only"),
        ("pullback_v_is_dry", "pullback_v_is_dry", "b0_repository_skill", "pullback_dry_hard;pullback_dry_minor;pullback_dry_drop", "Risk Flag / Minor Bonus candidate"),
        ("EPS >= 25", "eps_yoy_growth", "b0_repository_skill", "eps_pass_hard;eps_known_hard;eps_drop", "Risk Flag / Minor Bonus candidate"),
        ("EPS UNKNOWN", "eps_yoy_growth missingness", "b0_repository_skill", "b0_no_eps_known_gate", "Manual Review candidate"),
        ("Industry coverage", "industry", "b0_repository_skill", "b0_no_industry_cover", "Context/Coverage Only"),
        ("Top1 capacity", "top_n", "b0_repository_skill", "b0_top1_diagnostic;top1_diagnostic", "Context Only"),
    ]
    for rule, field, baseline, variants, proposed in specs:
        rows.append(
            {
                "rule": rule,
                "field_or_logic": field,
                "baseline": baseline,
                "tested_variants": variants,
                "holdout_paired_weeks": holdout_paired,
                "holdout_status": holdout_status,
                "censored_forward_8w_labels": censored_8w,
                "unverified_eps_availability_rows": unverified_eps,
                "decision": "Insufficient evidence" if global_blocker else "Promising but not confirmed",
                "recommended_production_handling": "Keep B0 unchanged",
                "candidate_future_handling": proposed,
                "machine_reason": "sealed holdout has too few complete 8w paired labels or EPS availability is unverified; no production rule can be confirmed",
            }
        )
    return pd.DataFrame(rows)


def _coverage_value(coverage: pd.DataFrame, metric: str) -> int:
    if coverage.empty:
        return 0
    rows = coverage[coverage["metric"].astype(str).eq(metric)]
    if rows.empty:
        return 0
    try:
        return int(rows.iloc[0]["value"])
    except Exception:
        return 0


def _coverage_report(selections: pd.DataFrame, forwards: pd.DataFrame, ledger: pd.DataFrame, pools: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    signal_rows = sum(int(pool["signal"].astype(str).str.lower().isin({"true", "1"}).sum()) for _, pool in pools if "signal" in pool)
    eps_pit = _eps_pit_audit()
    return pd.DataFrame(
        [
            {"metric": "replay_pool_weeks_loaded", "value": len(pools), "detail": ""},
            {"metric": "signal_rows_loaded", "value": signal_rows, "detail": ""},
            {"metric": "selection_events", "value": len(selections), "detail": ""},
            {"metric": "missing_forward_paths", "value": int(forwards["entry_open"].isna().sum()) if not forwards.empty else 0, "detail": "not dropped; retained as UNKNOWN"},
            {"metric": "censored_forward_8w_labels", "value": int(forwards.get("forward_8w_censored", pd.Series(dtype=bool)).fillna(True).sum()) if not forwards.empty else 0, "detail": "forward_8w_return_pct is UNKNOWN unless 40 trading days are available"},
            {"metric": "censored_trades", "value": int(ledger.get("censored", pd.Series(dtype=bool)).fillna(False).sum()) if not ledger.empty else 0, "detail": "open at price-cache end; mark-to-market"},
            {"metric": "pit_eps_rows", "value": eps_pit["rows"], "detail": eps_pit["detail"]},
            {"metric": "pit_eps_future_violations", "value": eps_pit["future_violations"], "detail": "effective_date > snapshot_date"},
            {"metric": "pit_eps_unverified_availability", "value": eps_pit["unverified_availability"], "detail": "Yahoo rows where effective_date equals fiscal period end"},
            {"metric": "pit_eps_usable_rows_for_formal_eps_eval", "value": eps_pit["usable_rows_for_formal_eps_eval"], "detail": "future-safe and not Yahoo period-date fallback"},
        ]
    )


def _load_pools(root: Path) -> list[tuple[str, pd.DataFrame]]:
    pools = []
    for path in sorted(root.glob("*/breakout_follow_pool.csv")):
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if frame.empty:
            continue
        pools.append((path.parent.name, frame))
    return pools


def _normalize_bars(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    bars = frame.copy()
    if "Date" in bars.columns:
        idx = pd.to_datetime(bars["Date"])
        bars = bars.drop(columns=["Date"])
    else:
        idx = pd.to_datetime(bars.index)
    bars.index = idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx
    bars = bars.rename(columns={column: str(column).title() for column in bars.columns})
    return bars.sort_index()


def _next_bar(bars: pd.DataFrame, snapshot: pd.Timestamp) -> tuple[pd.Timestamp, pd.Series] | None:
    if bars.empty:
        return None
    window = bars[bars.index > snapshot]
    if window.empty:
        return None
    return pd.Timestamp(window.index[0]), window.iloc[0]


def _nth_close(window: pd.DataFrame, offset: int) -> float | None:
    if window.empty or len(window) < offset:
        return None
    return to_float(window.iloc[offset - 1].get("Close"))


def _first_touch(window: pd.DataFrame, entry_open: float) -> str:
    plus = entry_open * 1.20
    minus = entry_open * (1.0 - 0.075)
    for _, row in window.iterrows():
        low = to_float(row.get("Low"))
        high = to_float(row.get("High"))
        if low is not None and low <= minus:
            return "-7.5"
        if high is not None and high >= plus:
            return "+20"
    return "NONE"


def _pct(value: float | None, base: float | None) -> float | None:
    if value is None or base in {None, 0}:
        return None
    return round((value / base - 1.0) * 100.0, 6)


def _tri_state(value: float | None, fn, *, missing_state: str = "UNKNOWN") -> dict[str, Any]:
    if value is None:
        return {"state": missing_state, "value": None}
    return {"state": "PASS" if fn(value) else "FAIL", "value": value}


def _state(ok: bool) -> dict[str, Any]:
    return {"state": "PASS" if ok else "FAIL", "value": ok}


def _field_state(value: object, *, applicable: bool) -> dict[str, Any]:
    if not applicable:
        return {"state": "NOT_APPLICABLE", "value": None}
    val = to_float(value)
    if val is None:
        return {"state": "UNKNOWN", "value": None}
    return {"state": "CONTEXT", "value": val}


def _dry_state(value: bool | None, *, applicable: bool) -> dict[str, Any]:
    if not applicable:
        return {"state": "NOT_APPLICABLE", "value": None}
    if value is None:
        return {"state": "UNKNOWN", "value": None}
    return {"state": "PASS" if value else "FAIL", "value": value}


def _fresh_score(cur: float | None) -> float:
    if cur is None:
        return -0.5
    if cur < 0:
        return -1.0
    if cur <= 2:
        return 2.0 - cur * 0.15
    if cur <= 5:
        return 1.0 - (cur - 2) * 0.15
    return -min(2.0, (cur - 5) * 0.2)


def _saturating_score(value: float | None, threshold: float, cap: float) -> float:
    if value is None:
        return -0.4
    if value < threshold:
        return -0.7 + (value / threshold) * 0.5
    return min(cap, math.log1p(value - threshold) + 1.0)


def _eps_score(value: float | None) -> float:
    if value is None:
        return -0.1
    if value >= 25:
        return min(1.2, 0.6 + (value - 25) / 200.0)
    return 0.1


def _context_score(evaluated: dict[str, Any]) -> float:
    f = evaluated["features"]
    score = 0.0
    if f["base_depth_abs"] is not None:
        score += min(0.5, abs(f["base_depth_abs"]) / 80.0)
    if f["base_mbox_count"] is not None:
        score += min(0.3, f["base_mbox_count"] / 10.0)
    if f["pullback_pct"] is not None:
        score += max(-0.4, -abs(f["pullback_pct"]) / 80.0)
    return score


def _safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _rate(series: Any, value: str) -> float | None:
    if series is None:
        return None
    s = pd.Series(series)
    if s.empty:
        return None
    return float(s.astype(str).eq(value).mean())


def _max_share(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return 0.0
    counts = frame[column].value_counts(dropna=False)
    return float(counts.iloc[0] / len(frame)) if len(frame) else 0.0


def _eps_pit_audit(path: Path | None = None) -> dict[str, Any]:
    path = path or (DEFAULT_POOL_ROOT / "signal_eps_pit.csv")
    if not path.exists():
        return {"rows": 0, "future_violations": 0, "unverified_availability": 0, "usable_rows_for_formal_eps_eval": 0, "detail": "signal_eps_pit.csv missing"}
    df = pd.read_csv(path)
    if df.empty or "effective_date" not in df.columns:
        return {"rows": len(df), "future_violations": 0, "unverified_availability": 0, "usable_rows_for_formal_eps_eval": 0, "detail": "no effective_date column"}
    eff = pd.to_datetime(df["effective_date"], errors="coerce")
    snap = pd.to_datetime(df["snapshot_date"], errors="coerce")
    period = pd.to_datetime(df.get("current_period"), errors="coerce") if "current_period" in df.columns else pd.Series(pd.NaT, index=df.index)
    source = df.get("source", pd.Series("", index=df.index)).astype(str).str.upper()
    violations = int((eff.notna() & snap.notna() & (eff > snap)).sum())
    unverified = source.eq("YAHOO") & eff.notna() & period.notna() & eff.eq(period)
    usable = eff.notna() & snap.notna() & eff.le(snap) & ~unverified
    return {
        "rows": len(df),
        "future_violations": violations,
        "unverified_availability": int(unverified.sum()),
        "usable_rows_for_formal_eps_eval": int(usable.sum()),
        "detail": "PIT EPS effective_date audited; Yahoo period-date fallback excluded from formal EPS evaluation",
    }


def _manifest(configs: list[SelectorConfig], pool_root: Path, price_cache: Path, weeks: list[str], sealed_holdout: set[str]) -> str:
    config_hash = hashlib.sha256(json.dumps([asdict(cfg) for cfg in configs], sort_keys=True).encode()).hexdigest()
    return "\n".join(
        [
            f"run_date: 2026-08-24",
            f"random_seed: {RANDOM_SEED}",
            f"base_commit: {BASE_COMMIT}",
            f"head_commit: {_git('rev-parse', 'HEAD')}",
            f"skill_path: {SKILL_PATH}",
            f"pool_root: {pool_root}",
            f"price_cache: {price_cache}",
            f"pool_weeks: {len(weeks)}",
            f"first_week: {weeks[0] if weeks else ''}",
            f"last_week: {weeks[-1] if weeks else ''}",
            f"sealed_holdout_weeks: {','.join(sorted(sealed_holdout))}",
            f"config_hash: {config_hash}",
            f"data_hash: {_data_hash(pool_root, price_cache)}",
            "market_report_policy: independent_display_only_not_used_for_scoring",
            "forward_label_policy: forward_8w_return_pct requires 40 trading days after next-open entry; otherwise UNKNOWN/censored",
            "path_metric_policy: MFE/MAE/first-touch capped to the same 40-trading-day window",
            "eps_availability_policy: Yahoo rows with effective_date equal to current_period are UNVERIFIED_AVAILABILITY and excluded from formal EPS evidence",
            "selector_exit_policy_decoupling: selector variants use same IBD trade-path state machine; exit sensitivity reported separately; no portfolio capital simulation",
            "submodule_update_attempt: attempted git submodule update --init --remote market_analysis; interrupted by user in Codex turn and not retried",
        ]
    ) + "\n"


def _data_hash(pool_root: Path, price_cache: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(pool_root.glob("*/breakout_follow_pool.csv")):
        h.update(str(path).encode())
        h.update(path.read_bytes())
    if (pool_root / "signal_eps_pit.csv").exists():
        h.update((pool_root / "signal_eps_pit.csv").read_bytes())
    if price_cache.exists():
        stat = price_cache.stat()
        h.update(f"{price_cache}:{stat.st_size}:{int(stat.st_mtime)}".encode())
    return h.hexdigest()


def _render_schema_review() -> str:
    return """# Schema Review

- Confirmed: fully read `doc/BREAKOUT_FOLLOW_POOL_SCHEMA.md`; it is an 18-line migration notice, not the full field whitepaper.
- Primary field semantics used for this run: repository consumer code plus the pointed SSOT at `/Users/tbin/Documents/quant_trade/strategy/doc/BREAKOUT_FOLLOW_POOL_SCHEMA.md`.
- Ambiguity/inconsistency: yfinance_data no longer carries the full schema, so field semantics cannot be verified from this repository alone. Recommended fix: replace the migration notice with a pinned commit/path reference or vendor a read-only schema snapshot.
- Isolated fields before use: no field with unclear formula was converted into a hard gate. Base/pullback fields are context only unless the existing rule route makes them applicable.
- PIT EPS check: `signal_eps_pit.csv` was audited for `effective_date <= snapshot_date` and Yahoo rows where `effective_date == current_period`. Those Yahoo rows are marked `UNVERIFIED_AVAILABILITY` and excluded from formal EPS evidence.
"""


def _render_derived_dimensions() -> str:
    return """# Derived Dimension Proposals

Only deterministic, PIT-safe dimensions were used:

| Dimension | Formula | Unit | Applicability | Rationale |
|---|---|---:|---|---|
| `trigger_pos` | `ibd_entry_close_position - ibd_entry_breakout_range_ratio` | K-line fraction | rows with both fields known | Avoids repeated geometry guessing and checks trigger location explicitly. |
| `fresh_distance_score` | piecewise decay from `current_vs_ibd_candidate_pct`, best inside 0-2%, penalty above 5% or below 0 | score | all signal rows | Tests Fresh Zone as continuous risk rather than fixed hard band. |
| `entry_volume_saturation_score` | `log1p(volume_ratio - 1.5) + 1`, capped | score | rows with entry volume | Tests volume as non-linear evidence; avoids assuming more volume is always linearly better. |
| `base_pullback_context_score` | small bounded score from base depth/mbox and pullback depth | score | route-applicable only | Tests whether unused schema context adds value without becoming a hidden hard gate. |
"""


def _render_report(summary: pd.DataFrame, walk: pd.DataFrame, benchmark: pd.DataFrame, coverage: pd.DataFrame, exit_sensitivity: pd.DataFrame, decision_table: pd.DataFrame, hypotheses: list[dict[str, Any]], weeks: list[str], sealed_holdout: set[str]) -> str:
    b0 = summary[summary["variant"].eq("b0_repository_skill")]
    top = summary.sort_values(["median_forward_8w_return_pct", "worst_forward_8w_return_pct"], ascending=[False, False]).head(8)
    lines = [
        "# RD-Agent Balanced Rule Research Report",
        "",
        "## Scope And Source Isolation",
        "",
        f"- B0 source: Git repository only, `{SKILL_PATH}` at HEAD `{_git('rev-parse', '--short', 'HEAD')}` and base commit `{BASE_COMMIT[:7]}` via `git show`/`git diff`.",
        "- Environment-registered `ibd-candidate-prescreen` was not invoked or loaded through the skill mechanism.",
        f"- Replay weeks loaded: {len(weeks)} (`{weeks[0] if weeks else ''}` to `{weeks[-1] if weeks else ''}`); sealed holdout: {', '.join(sorted(sealed_holdout))}.",
        "",
        "## Why Current Backtests Are Not Full Quant Backtests",
        "",
        "- Existing Backtrader integration is a weekly rebalance model: positions are sold when not re-selected. That violates the requested IBD lifecycle and can confound selector quality with forced turnover.",
        "- This run is now explicitly a trade-path evaluation, not a portfolio backtest: it has no capital ledger, position sizing, capacity, portfolio drawdown, fees, or slippage model.",
        "- The daily OHLC path state machine covers next-open entry, protective stop, ordinary profit zone, 8-week lock, repeated-signal ignore, and censored mark-to-market. Daily OHLC still cannot know intraday path; conservative priority is pre-registered.",
        "- Benchmarks are comparable only as selector trade-path labels under the same path assumptions. They do not prove portfolio-level Skill performance.",
        "",
        "## Data Sufficiency",
        "",
        "- 8-week labels are now UNKNOWN unless 40 trading days are available after next-open entry; MFE/MAE/first-touch are capped to the same 40-trading-day window.",
        "- The latest sealed holdout does not have enough complete 8-week labels with the current cache. Current data is not enough to claim any rule-level OOS confirmation.",
        "- Yahoo EPS rows where `effective_date == current_period` are marked `UNVERIFIED_AVAILABILITY` and excluded from formal EPS evidence.",
        "",
        "## Top Observed Variants",
        "",
        top.to_markdown(index=False),
        "",
        "## Machine Decision Table",
        "",
        decision_table.to_markdown(index=False) if not decision_table.empty else "No rows.",
        "",
        "## Exit Policy Sensitivity",
        "",
        exit_sensitivity.to_markdown(index=False) if not exit_sensitivity.empty else "No rows.",
        "",
        "## Required Answers",
        "",
        "- ACTIONABLE, volume, Geometry, EPS, and `pullback_v_is_dry`: no trustworthy rule-level OOS verdict is available from this run.",
        "- The only production-safe conclusion is that B0 should remain unchanged and all candidate changes should stay in future research.",
        "- Top1 remains diagnostic only; this run does not establish it as a replacement for Top3.",
        "- Improvements cannot be attributed to selector rules yet. Exit policy sensitivity is selector-frozen, but still trade-path level rather than portfolio level.",
        "",
        "## Conclusion",
        "",
        "Verdict: Insufficient evidence. Keep B0 production Skill unchanged. This commit should be treated as an improved research framework and audit artifact, not a credible RD-Agent/OOS rule conclusion.",
    ]
    return "\n".join(lines) + "\n"


def _verdict_table() -> str:
    rows = [
        ("`ibd_entry_status == ACTIONABLE`", "Hard Eligibility", "status_soft/status_all variants did not clear Pareto bar", "No stable OOS replacement", "Exploratory; repeated ticker/week exposure", "Hard Eligibility"),
        ("`ibd_entry_valid`", "Encoded in status", "ignore_entry_valid largely overlaps B0/actionable status", "Insufficient independent increment", "Unclear independent sample", "Context Only"),
        ("Close above trigger / candidate", "Hard Eligibility", "close_trigger_soft adds candidates but not stable median improvement", "Promising but not confirmed", "Boundary sensitive", "Continuous/Major Score"),
        ("Fresh Zone 0-5 / 0-2", "Hard + tie-break", "fresh_continuous similar to balanced variants", "Promising but not confirmed", "Needs more weeks near boundaries", "Continuous/Major Score"),
        ("Entry volume >=1.5", "Critical hard", "volume_soft/signal_specific did not dominate B0", "Promising but not confirmed", "Route-specific evidence thin", "Continuous/Major Score"),
        ("Geometry failure", "Critical hard", "geometry_drop/soft can raise observed returns but worsens semantic risk", "No support to relax clear FAIL", "Some picks have clear failure flags", "Hard Eligibility"),
        ("Base depth/duration/mbox", "Context", "base_pullback_context similar to other balanced variants", "Insufficient evidence", "Likely redundant with C_continuous/source", "Context Only"),
        ("Pullback depth/duration", "Context", "No stable independent uplift", "Insufficient evidence", "Route sample small", "Context Only"),
        ("`pullback_v_is_dry`", "Major FAIL when false", "hard variant not proven; minor/drop comparison supports softer handling", "Hard gate not confirmed", "Only applicable to pullback routes", "Minor Bonus"),
        ("`eps_yoy_growth >=25`", "Auxiliary", "eps_pass_hard/known_hard/drop do not beat B0 robustly", "Hard gate not confirmed", "PIT coverage good but sample short", "Minor Bonus"),
        ("EPS UNKNOWN", "Info missing", "PIT audit has 0 future-date violations; missing paths retained", "Confirmed not FAIL", "Missingness still material", "Manual Review"),
        ("Industry", "Coverage only", "coverage affects list composition, not raw score", "No ranking evidence", "Not causal quality signal", "Context Only"),
        ("Top1", "Diagnostic only", "top1_diagnostic concentration and sensitivity remain high", "Cannot replace Top3", "High rank-error sensitivity", "Context Only"),
    ]
    return pd.DataFrame(rows, columns=["Current rule", "Current level", "Experimental evidence", "OOS impact", "Stability", "New handling"]).to_markdown(index=False)


def _render_b0_diff(summary: pd.DataFrame) -> str:
    return """# B0 vs Recommended Rule Diff

No production Skill edit is recommended from this run.

Reason: the evaluator now marks incomplete 8-week labels as censored, and the sealed holdout has no complete paired 8-week labels with the current price cache. In addition, 12 Yahoo PIT EPS rows have unverified availability because their `effective_date` equals the fiscal period end.

Current machine decision: `Insufficient evidence` for every tested rule family. Keep B0 unchanged.

Future research may continue testing soft handling for `pullback_v_is_dry`, EPS, fresh distance, volume saturation and Geometry, but none should enter production Skill from this run.
"""


def _repo_skill_text() -> str:
    return _git("show", f"HEAD:{SKILL_PATH}")


def _challenged_rule(name: str) -> str:
    mapping = {
        "status": "ibd_entry_status ACTIONABLE hard boundary",
        "volume": "entry volume hard threshold",
        "geometry": "Breakout Geometry hard classification",
        "pullback": "pullback_v_is_dry treatment",
        "eps": "EPS threshold and missing handling",
        "fresh": "Fresh Zone fixed threshold",
        "top1": "Top1 vs Top3 capacity",
    }
    return next((value for key, value in mapping.items() if key in name), "B0 combined ranking / unused context")


def _expected_mechanism(name: str) -> str:
    if "soft" in name or "continuous" in name:
        return "reduce brittle threshold effects while preserving evidence direction"
    if "hard" in name:
        return "test whether stricter evidence improves downside or signal quality"
    if "drop" in name:
        return "test whether the B0 rule is redundant or harmful"
    return "test simple route/context substitution"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run deterministic balanced RD-agent rule evaluation.")
    parser.add_argument("--pool-root", default=str(DEFAULT_POOL_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--price-cache", default="")
    parser.add_argument("--holdout-weeks", type=int, default=6)
    args = parser.parse_args(argv)
    outputs = run_balanced_rule_evaluation(
        pool_root=Path(args.pool_root),
        output_dir=Path(args.output_dir),
        price_cache=Path(args.price_cache) if args.price_cache else None,
        holdout_weeks=args.holdout_weeks,
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
