"""Historical Validation Verifier (Single-Direction Unblinding on Weeks 31~40).

This module performs one-way evaluation of frozen rules on Weeks 31~40.

IMPORTANT:
Weeks 31~40 have been exposed during prior research exploration and are formally designated
as 'Contaminated Historical Validation'. They do NOT serve as virgin out-of-sample evidence.
True immutable virgin out-of-sample testing begins with the forward shadow ledger on 2026-08-28+.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.b0_top3_quality_audit.pareto_champions import compute_file_sha256
from backtest.b0_top3_quality_audit.skill_rule_engine import (
    RuleSpec,
    build_skill_rule_space,
    evaluate_rule_on_pool,
)
from backtest.b0_top3_quality_audit.three_tier_baseline import compute_portfolio_metrics
from backtest.b0_top3_quality_audit.research_windows import (
    CONTAMINATED_VALIDATION_END,
    CONTAMINATED_VALIDATION_START,
    contaminated_validation_dates,
)

logger = logging.getLogger(__name__)


def verify_manifest_integrity(manifest: dict[str, Any], repo_root: Path) -> None:
    """Fail-closed integrity gate: verify all cryptographic signatures before execution."""
    stored_sha = manifest.get("manifest_sha256")
    if not stored_sha:
        raise RuntimeError("Integrity Error: manifest_sha256 is missing from manifest!")

    manifest_copy = dict(manifest)
    manifest_copy.pop("manifest_sha256", None)
    canonical_json_str = json.dumps(manifest_copy, sort_keys=True, ensure_ascii=False)
    computed_manifest_sha = hashlib.sha256(canonical_json_str.encode("utf-8")).hexdigest()

    if computed_manifest_sha != stored_sha:
        raise RuntimeError(
            f"Integrity Error: Manifest SHA256 mismatch! "
            f"Computed: {computed_manifest_sha} != Stored: {stored_sha}"
        )

    # Verify code fingerprints
    code_fps = manifest.get("code_fingerprints", {})
    code_paths = {
        "production_selector_sha256": repo_root / "dashboard" / "skill_industry_eps_known.py",
        "eligibility_predicate_sha256": repo_root / "backtest" / "b0_top3_quality_audit" / "eligibility.py",
        "skill_rule_engine_sha256": repo_root / "backtest" / "b0_top3_quality_audit" / "skill_rule_engine.py",
        "three_tier_baseline_sha256": repo_root / "backtest" / "b0_top3_quality_audit" / "three_tier_baseline.py",
        "evaluate_rule_signatures_sha256": repo_root / "backtest" / "b0_top3_quality_audit" / "evaluate_rule_signatures.py",
        "research_windows_sha256": repo_root / "backtest" / "b0_top3_quality_audit" / "research_windows.py",
        "replay_eps_sha256": repo_root / "backtest" / "replay_eps.py",
    }
    for key, expected_hash in code_fps.items():
        file_p = code_paths.get(key)
        if file_p is None or not file_p.exists():
            raise RuntimeError(f"Integrity Error: Required code file {file_p} is missing!")
        actual_hash = compute_file_sha256(file_p)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Integrity Error: Code file {file_p} has drifted! "
                f"Actual: {actual_hash} != Expected: {expected_hash}"
            )

    # Verify immutable train data fingerprints (Weeks 1~30 only)
    data_fps = manifest.get("data_fingerprints", {})
    data_paths = {
        "train_candidate_events_parquet_sha256": repo_root / "backtest" / "b0_top3_quality_audit" / "data" / "frozen" / "train_candidate_event_outcomes.parquet",
        "train_candidate_weekly_outcomes_parquet_sha256": repo_root / "backtest" / "b0_top3_quality_audit" / "data" / "frozen" / "train_candidate_weekly_outcomes.parquet",
    }
    for key, expected_hash in data_fps.items():
        file_p = data_paths.get(key)
        if file_p is None or not file_p.exists():
            raise RuntimeError(f"Integrity Error: Required frozen train data file {file_p} is missing!")
        actual_hash = compute_file_sha256(file_p)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Integrity Error: Data file {file_p} has drifted! "
                f"Actual: {actual_hash} != Expected: {expected_hash}"
            )


def run_historical_validation_unblind(
    manifest_path: Path,
    events_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, str]:
    """Load frozen manifest, execute fail-closed integrity gate, and evaluate on Weeks 31~40."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    repo_root = Path(__file__).resolve().parents[2]
    # Execute Fail-Closed Integrity Gate
    verify_manifest_integrity(manifest, repo_root)

    manifest_sha = manifest.get("manifest_sha256", "UNKNOWN")
    champions = manifest.get("champions", {})
    logger.info(f"Integrity check passed! Loaded frozen manifest (SHA256: {manifest_sha}) with {len(champions)} champions.")

    all_weeks = sorted(events_df["snapshot_date"].astype(str).unique())
    validation_weeks = sorted(contaminated_validation_dates(all_weeks))
    if not validation_weeks:
        raise RuntimeError(
            "No contaminated historical validation weeks found in fixed date window "
            f"{CONTAMINATED_VALIDATION_START}..{CONTAMINATED_VALIDATION_END}"
        )
    logger.info(
        "Evaluating %s Historical Validation weeks (%s to %s)...",
        len(validation_weeks),
        validation_weeks[0],
        validation_weeks[-1],
    )

    # Map of all rules to evaluate: B0 + Champions
    base_rules: dict[str, RuleSpec] = {r.rule_id: r for r in build_skill_rule_space()}
    rules_to_eval: list[tuple[str, RuleSpec]] = []
    
    b0_spec = base_rules.get("B0_BASELINE")
    if b0_spec:
        rules_to_eval.append(("PRODUCTION_BASELINE", b0_spec))

    for role, champ in champions.items():
        rid = champ["rule_id"]
        frozen_params = champ.get("params", {})
        # Find matching spec with exact params (strictly fail-closed: NO fallback)
        matching_spec = None
        for r in build_skill_rule_space():
            if r.rule_id == rid and r.params == frozen_params:
                matching_spec = r
                break
        if matching_spec is None:
            raise RuntimeError(
                f"Integrity Error: Frozen RuleSpec mismatch! Rule ID {rid} with params {frozen_params} does not exist in skill rule space."
            )
        rules_to_eval.append((role, matching_spec))

    # Precompute snapshots
    snapshot_data: dict[str, dict[str, Any]] = {}
    for snap_date in validation_weeks:
        snap_events = events_df[events_df["snapshot_date"] == snap_date].copy()
        snap_events["snapshot_date"] = snap_date
        snap_event_dict = {str(r["code"]): r.to_dict() for _, r in snap_events.iterrows()}
        snap_weekly_dict = {
            (str(w["code"]), int(w["holding_week_index"])): w.to_dict()
            for _, w in weekly_df[weekly_df["snapshot_date"] == snap_date].iterrows()
        }
        snapshot_data[snap_date] = {
            "events_df": snap_events,
            "event_dict": snap_event_dict,
            "weekly_dict": snap_weekly_dict,
        }

    b0_weekly_map = baseline_df.set_index("snapshot_date").to_dict(orient="index")
    val_records: list[dict[str, Any]] = []

    for role_label, rule_spec in rules_to_eval:
        weekly_res = []
        for snap_date in validation_weeks:
            s_data = snapshot_data[snap_date]
            selected_codes = evaluate_rule_on_pool(s_data["events_df"], rule_spec, pick_limit=3)
            m = compute_portfolio_metrics(selected_codes, s_data["event_dict"], s_data["weekly_dict"], snap_date)

            b0_info = b0_weekly_map.get(snap_date, {})
            l1_exec_med = b0_info.get("l1_executed_ret_med", np.nan)
            l1_w1_med = b0_info.get("l1_w1_ret_med", np.nan)

            weekly_res.append({
                "snapshot_date": snap_date,
                "executed_return": m["executed_return"],
                "w1_return": m["w1_return"],
                "w2_return": m["w2_return"],
                "w4_return": m["w4_return"],
                "stop8_before_p20": m["stop8_before_profit20"],
                "win_vs_l1_exec": bool(m["executed_return"] > l1_exec_med) if not np.isnan(m["executed_return"]) and not np.isnan(l1_exec_med) else False,
                "win_vs_l1_w1": bool(m["w1_return"] > l1_w1_med) if not np.isnan(m["w1_return"]) and not np.isnan(l1_w1_med) else False,
            })

        wdf = pd.DataFrame(weekly_res)
        val_records.append({
            "role": role_label,
            "rule_id": rule_spec.rule_id,
            "complexity": rule_spec.complexity,
            "val_w1_ret_med": float(wdf["w1_return"].median()),
            "val_w2_ret_med": float(wdf["w2_return"].median()),
            "val_w4_ret_med": float(wdf["w4_return"].median()),
            "val_exec_ret_med": float(wdf["executed_return"].median()),
            "val_stop8_pct": float(wdf["stop8_before_p20"].mean() * 100.0),
            "val_win_vs_l1_pct": float(wdf["win_vs_l1_exec"].mean() * 100.0),
        })

    val_df = pd.DataFrame(val_records)
    out_csv = out_dir / "historical_validation_evaluation.csv"
    val_df.to_csv(out_csv, index=False)
    logger.info(f"Saved historical validation evaluation CSV to {out_csv}")

    # Generate Markdown Report from the actual fixed-date calendar.
    manifest_label = manifest_path.resolve().relative_to(repo_root.resolve()).as_posix()
    report = f"""# Contaminated Historical Validation Audit Report

**测试性质**：历史验证集单向出表 (One-Way Historical Validation Disclosure)  
**Fixed calendar**：{CONTAMINATED_VALIDATION_START} 至 {CONTAMINATED_VALIDATION_END}（{len(validation_weeks)} 个 snapshot weeks）<br>
**输入来源**：`{manifest_label}` (SHA256: `{manifest_sha}`)<br>
**隔离原则**：本报告仅单向输出审计数据，**严禁反馈回演进引擎进行调参或规则重构**。

> [!WARNING]
> **方法论定位说明（Methodology Caveat）**：
> 鉴于前期研发探索已接触过第 31~40 周数据，本报告中的表现严格定性为 **“已被污染的历史验证集（Contaminated Historical Validation）”**，不能作为纯粹的盲测样本外证据。
> 真正的无偏样本外验证严格建立在 **2026-08-28 之后的实时前瞻 Shadow 跟测账本**。

---

## 一、Contaminated Historical Validation 表现总表

| 角色 / 规则 | 规则 ID | 复杂度 $C$ | W1 收益中位 | W2 收益中位 | W4 收益中位 | 全周期收益中位 | 止损发生率 | vs L1 胜率 | 定位与建议 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
"""
    for _, row in val_df.iterrows():
        advice = "生产继续保持 100% 冻结" if row["rule_id"] == "B0_BASELINE" else "研究观察候选 (Shadow)"
        report += (
            f"| **{row['role']}** | `{row['rule_id']}` | $C={row['complexity']}$ | "
            f"**{row['val_w1_ret_med']:+.2f}%** | {row['val_w2_ret_med']:+.2f}% | {row['val_w4_ret_med']:+.2f}% | "
            f"**{row['val_exec_ret_med']:+.2f}%** | {row['val_stop8_pct']:.1f}% | {row['val_win_vs_l1_pct']:.1f}% | {advice} |\n"
        )

    report += """
---

## 二、审计结论与后续行动

1. **生产基准零修改**：`dashboard/skill_industry_eps_known.py` 继续 100% 冻结不变。
2. **时序分期与影子跟测账本 (Forward Shadow Ledger)**：
   * **Pre-Freeze Replay (2026-08-14 与 2026-08-21)**：作为规则冻结前的回放测试周；
   * **Forward Shadow Kickoff (2026-08-28 起)**：正式启动纯净前瞻影子账本，并行跟踪冻结清单中预注册的 `B0_BASELINE`、`SIMPLER_PURE_FRESHNESS` 与 `SIMPLER_PURE_CLOSE_POS`。
"""
    out_md = out_dir / "CONTAMINATED_HISTORICAL_VALIDATION_REPORT.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Generated Historical Validation Audit Report at {out_md}")

    return val_df, report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    root_dir = Path(__file__).resolve().parent
    events_path = root_dir / "data" / "candidate_event_outcomes.parquet"
    weekly_path = root_dir / "data" / "candidate_weekly_outcomes.parquet"
    baseline_path = root_dir / "output" / "three_tier_weekly_comparison.csv"
    manifest_path = root_dir / "output" / "frozen_rules_manifest.json"
    out_dir = root_dir / "output"

    events_df = pd.read_parquet(events_path)
    weekly_outcomes_df = pd.read_parquet(weekly_path)
    baseline_df = pd.read_csv(baseline_path)

    run_historical_validation_unblind(manifest_path, events_df, weekly_outcomes_df, baseline_df, out_dir)
