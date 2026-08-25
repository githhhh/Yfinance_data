"""Historical Validation Verifier (Single-Direction Unblinding on Weeks 31~40).

This module performs one-way evaluation of frozen rules on Weeks 31~40.

IMPORTANT:
Weeks 31~40 have been exposed during prior research exploration and are formally designated
as 'Contaminated Historical Validation'. They do NOT serve as virgin out-of-sample evidence.
True immutable virgin out-of-sample testing begins with the forward shadow ledger on 2026-08-14+.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.b0_top3_quality_audit.skill_rule_engine import (
    RuleSpec,
    build_skill_rule_space,
    evaluate_rule_on_pool,
)
from backtest.b0_top3_quality_audit.three_tier_baseline import compute_portfolio_metrics

logger = logging.getLogger(__name__)


def run_historical_validation_unblind(
    manifest_path: Path,
    events_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, str]:
    """Load frozen manifest and evaluate on Weeks 31~40 in one direction."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_sha = manifest.get("manifest_sha256", "UNKNOWN")
    champions = manifest.get("champions", {})
    logger.info(f"Loaded frozen manifest (SHA256: {manifest_sha}) with {len(champions)} champions.")

    all_weeks = sorted(baseline_df["snapshot_date"].unique())
    validation_weeks = all_weeks[30:]  # Weeks 31~40 (10 weeks)
    logger.info(f"Evaluating {len(validation_weeks)} Historical Validation weeks ({validation_weeks[0]} to {validation_weeks[-1]})...")

    # Map of all rules to evaluate: B0 + Champions
    rule_map: dict[str, RuleSpec] = {r.rule_id: r for r in build_skill_rule_space()}
    rules_to_eval = [("PRODUCTION_BASELINE", "B0_BASELINE")]
    for role, champ in champions.items():
        rules_to_eval.append((role, champ["rule_id"]))

    # Precompute snapshots
    snapshot_data: dict[str, dict[str, Any]] = {}
    for snap_date in validation_weeks:
        snap_events = events_df[events_df["snapshot_date"] == snap_date].copy()
        snap_events["snapshot_date"] = snap_date
        snap_event_dict = {str(r["code"]): r.to_dict() for _, r in snap_events.iterrows()}
        snap_weekly_dict = {
            (str(w["code"]), int(w["holding_week_index"])): w.to_dict()
            for _, w in weekly_outcomes_df[weekly_outcomes_df["snapshot_date"] == snap_date].iterrows()
        }
        snapshot_data[snap_date] = {
            "events_df": snap_events,
            "event_dict": snap_event_dict,
            "weekly_dict": snap_weekly_dict,
        }

    b0_weekly_map = baseline_df.set_index("snapshot_date").to_dict(orient="index")
    val_records: list[dict[str, Any]] = []

    for role_label, rule_id in rules_to_eval:
        rule_spec = rule_map.get(rule_id)
        if not rule_spec:
            logger.warning(f"RuleSpec not found for {rule_id}")
            continue

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
            "rule_id": rule_id,
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

    # Generate Markdown Report
    report = f"""# Contaminated Historical Validation Audit Report (Weeks 31~40)

**测试性质**：历史验证集单向出表 (One-Way Historical Validation Disclosure)  
**样本窗口**：第 31~40 周 (2026-05-29 至 2026-08-07，共 10 周)  
**输入来源**：`frozen_rules_manifest.json` (SHA256: `{manifest_sha}`)  
**隔离原则**：本报告仅单向输出审计数据，**严禁反馈回演进引擎进行调参或规则重构**。

> [!WARNING]
> **方法论定位说明（Methodology Caveat）**：
> 鉴于前期研发探索已接触过第 31~40 周数据，本报告中的表现严格定性为 **“已被污染的历史验证集（Contaminated Historical Validation）”**，不能作为纯粹的盲测样本外证据。
> 真正的无偏样本外验证严格建立在 **2026-08-14 之后的实时前瞻 Shadow 跟测账本**。

---

## 一、历史验证集 (Weeks 31~40) 表现总表

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
2. **启动实时前瞻跟测 (Forward Shadow Ledger)**：
   * 从 2026-08-14 / 2026-08-21 周度复盘起，建立实时跟测账本，每周并行记录 `B0_BASELINE` 与 `SIMPLER_PURE_FRESHNESS` 的前瞻选股输出。
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
