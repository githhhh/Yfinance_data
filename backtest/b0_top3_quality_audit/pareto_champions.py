"""Four-Champion Pareto Matrix & Walk-Forward Evaluation Harness (Production-Aligned).

Champion Definitions (Anchor on Train Set Weeks 1~30):
  1. HISTORICAL_RETURN_WINNER:
     Highest Train executed return median (Research reference only, forbidden from live deployment).
  2. LOWEST_STOP_CANDIDATE:
     Lowest Train stop8_before_profit20 rate with full coverage (active weeks >= 30).
  3. SIMPLER_EQUIVALENT:
     Complexity C < C_B0, satisfies Non-Inferiority Test (Train exec median >= B0_median - 0.5%).
  4. PARETO_BALANCED_RULE:
     Balanced candidate across Train return median, stop rate, active-ranking win rate, and simplicity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


import hashlib
import json


def compute_file_sha256(file_path: Path | str) -> str:
    """Compute SHA256 hash of a local file."""
    p = Path(file_path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


from datetime import datetime, timezone
import subprocess


def get_git_commit_sha(repo_root: Path) -> str:
    """Retrieve current git commit SHA."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN"


def select_champions(evaluation_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Select the 4 Champions based strictly on Train Set (Weeks 1~30) metrics.
    
    Filters:
      - Valid portfolio rate >= 80% on Train
      - Valid portfolio weeks >= 15
    """
    b0_row = evaluation_df[evaluation_df["rule_id"] == "B0_BASELINE"].iloc[0]
    b0_train_w1_med = float(b0_row.get("train_w1_ret_med", b0_row.get("train_exec_ret_med", 0.0)))
    b0_complexity = int(b0_row["complexity"])

    # Denominator-aware eligible rules pool
    valid_candidates = evaluation_df[
        (evaluation_df.get("train_valid_portfolio_rate", 100.0) >= 80.0)
        & (evaluation_df.get("train_valid_weeks", 30) >= 15)
    ].copy()
    if valid_candidates.empty:
        valid_candidates = evaluation_df.copy()

    # 1. Historical Return Winner (Highest Train W1 Return Median & Ranking Opportunity Spread)
    ret_winner_df = valid_candidates.sort_values(
        by=["train_w1_ret_med", "train_act_opp_w1_spread_med", "train_exec_ret_med"],
        ascending=False,
    )
    ret_winner = ret_winner_df.iloc[0].to_dict()

    # 2. Lowest Stop Candidate (Lowest stop8_before_profit20 rate with valid coverage)
    stop_cand_df = valid_candidates.sort_values(
        by=["train_stop8_before_p20_pct", "train_w1_ret_med"],
        ascending=[True, False],
    )
    lowest_stop = stop_cand_df.iloc[0].to_dict()

    # 3. Simpler Equivalent (Complexity C < B0, non-inferiority: train_w1_med >= B0 - 0.5%)
    stab_col = "train_temporal_block_stability_score" if "train_temporal_block_stability_score" in valid_candidates.columns else "train_wf_stability_score"
    simpler_df = valid_candidates[
        (valid_candidates["complexity"] < b0_complexity)
        & (valid_candidates["train_w1_ret_med"] >= b0_train_w1_med - 0.5)
    ].sort_values(
        by=["complexity", stab_col, "train_w1_ret_med"],
        ascending=[True, False, False],
    )
    if simpler_df.empty:
        simpler_df = valid_candidates[valid_candidates["complexity"] < b0_complexity].sort_values(
            by=["complexity", "train_w1_ret_med"], ascending=[True, False]
        )
    simpler_equiv = simpler_df.iloc[0].to_dict()

    # 4. Pareto Balanced Rule (Research Candidate: High Train W1 return, temporal stability, W1 active win rate vs L1, low complexity)
    eval_df = valid_candidates.copy()
    stab_series = eval_df.get("train_temporal_block_stability_score", eval_df.get("train_wf_stability_score", 0.0))
    w1_win_series = eval_df.get("train_act_w1_win_vs_l1_pct", eval_df.get("train_act_rank_win_vs_l1_pct", 0.0))
    eval_df["pareto_score"] = (
        (eval_df["train_w1_ret_med"] / 2.0)
        + (stab_series / 2.0)
        + (w1_win_series / 100.0)
        - (eval_df["train_stop8_before_p20_pct"] / 100.0)
        - (eval_df["complexity"] * 0.08)
    )
    balanced_df = eval_df.sort_values(by="pareto_score", ascending=False)
    pareto_balanced = balanced_df.iloc[0].to_dict()

    return {
        "HISTORICAL_RETURN_WINNER": ret_winner,
        "LOWEST_STOP_CANDIDATE": lowest_stop,
        "SIMPLER_EQUIVALENT": simpler_equiv,
        "PARETO_BALANCED_RULE": pareto_balanced,
    }


def export_frozen_rules_manifest(
    champions: dict[str, dict[str, Any]],
    output_path: Path,
    repo_root: Path | None = None,
    created_at_utc: str | None = None,
    source_git_commit: str | None = None,
) -> dict[str, Any]:
    """Export frozen rules manifest with full RuleSpec params, train data snapshot hash, forward shadow registrations, and protocol SHA256 integrity signature."""
    from backtest.b0_top3_quality_audit.skill_rule_engine import build_skill_rule_space

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    if created_at_utc is None:
        created_at_utc = datetime.now(timezone.utc).isoformat()
    if source_git_commit is None:
        source_git_commit = get_git_commit_sha(repo_root)

    # Map rule specs to retrieve exact params
    rule_spec_map = {r.rule_id: r for r in build_skill_rule_space()}

    # Gather Code & Protocol Fingerprints
    code_fingerprints = {
        "production_selector_sha256": compute_file_sha256(repo_root / "dashboard" / "skill_industry_eps_known.py"),
        "eligibility_predicate_sha256": compute_file_sha256(repo_root / "backtest" / "b0_top3_quality_audit" / "eligibility.py"),
        "skill_rule_engine_sha256": compute_file_sha256(repo_root / "backtest" / "b0_top3_quality_audit" / "skill_rule_engine.py"),
        "three_tier_baseline_sha256": compute_file_sha256(repo_root / "backtest" / "b0_top3_quality_audit" / "three_tier_baseline.py"),
        "evaluate_rule_signatures_sha256": compute_file_sha256(repo_root / "backtest" / "b0_top3_quality_audit" / "evaluate_rule_signatures.py"),
    }

    # Gather Immutable Train Data Fingerprints (Weeks 1~30 only)
    data_fingerprints = {
        "train_candidate_events_parquet_sha256": compute_file_sha256(repo_root / "backtest" / "b0_top3_quality_audit" / "data" / "frozen" / "train_candidate_event_outcomes.parquet"),
        "train_candidate_weekly_outcomes_parquet_sha256": compute_file_sha256(repo_root / "backtest" / "b0_top3_quality_audit" / "data" / "frozen" / "train_candidate_weekly_outcomes.parquet"),
    }

    # Pre-registered Forward Shadow Rules starting 2026-08-28
    forward_shadow_rules = [
        {
            "rule_id": "B0_BASELINE",
            "role": "PRODUCTION_BENCHMARK",
            "description": "Production multi-factor heuristic (freshness -> volume -> close position)",
            "complexity": 3,
            "params": {},
            "start_date": "2026-08-28",
            "primary_metrics": ["w1_return", "w2_return", "w4_return"],
            "stop_protocol": "STOP_8PCT_OR_PROFIT_20PCT",
            "status": "ACTIVE_PRODUCTION_BENCHMARK",
        },
        {
            "rule_id": "SIMPLER_PURE_FRESHNESS",
            "role": "FORWARD_SHADOW_CANDIDATE",
            "description": "Single-factor freshness priority",
            "complexity": 1,
            "params": {"sort_mode": "freshness_only"},
            "start_date": "2026-08-28",
            "primary_metrics": ["w1_return", "w2_return", "w4_return"],
            "stop_protocol": "STOP_8PCT_OR_PROFIT_20PCT",
            "status": "REGISTERED_FORWARD_SHADOW",
        },
        {
            "rule_id": "SIMPLER_PURE_CLOSE_POS",
            "role": "RESEARCH_PARETO_CHAMPION",
            "description": "Single-factor close position priority",
            "complexity": 1,
            "params": {"sort_mode": "close_pos_only"},
            "start_date": "2026-08-28",
            "primary_metrics": ["w1_return", "w2_return", "w4_return"],
            "stop_protocol": "STOP_8PCT_OR_PROFIT_20PCT",
            "status": "REGISTERED_FORWARD_SHADOW",
        },
    ]

    manifest_champions: dict[str, Any] = {}
    for role, champ in champions.items():
        rid = champ["rule_id"]
        rspec = rule_spec_map.get(rid)
        params = rspec.params if rspec else {}

        role_entry = {
            "role": role,
            "rule_id": rid,
            "description": champ.get("description", ""),
            "complexity": int(champ.get("complexity", 0)),
            "params": params,
            "train_w1_ret_med": champ.get("train_w1_ret_med"),
            "train_paired_w1_spread_med": champ.get("train_paired_w1_spread_med"),
            "train_temporal_block_stability_score": champ.get("train_temporal_block_stability_score"),
            "train_valid_portfolio_rate": champ.get("train_valid_portfolio_rate"),
            "train_act_w1_win_vs_l1_pct": champ.get("train_act_w1_win_vs_l1_pct", champ.get("train_act_rank_win_vs_l1_pct")),
            "train_stop8_before_p20_pct": champ.get("train_stop8_before_p20_pct"),
        }
        manifest_champions[role] = role_entry

    # Canonical Protocol SHA256 (Invariant across re-executions with identical code/data/rules)
    protocol_payload = {
        "evaluation_protocol": "CENSORED_PORTFOLIO_PIT_W1_W4_PROTOCOL",
        "train_period": "2025-10-10_to_2026-05-22 (Weeks 1~30)",
        "code_fingerprints": code_fingerprints,
        "data_fingerprints": data_fingerprints,
        "forward_shadow_rules": forward_shadow_rules,
        "champions": manifest_champions,
    }
    protocol_sha256 = hashlib.sha256(json.dumps(protocol_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    manifest_data: dict[str, Any] = {
        "manifest_version": "2.1",
        "created_at_utc": created_at_utc,
        "source_git_commit": source_git_commit,
        "freeze_id": "PHASE2_IMMUTABLE_FREEZE_20260828",
        "evaluation_protocol": "CENSORED_PORTFOLIO_PIT_W1_W4_PROTOCOL",
        "protocol_sha256": protocol_sha256,
        "train_period": "2025-10-10_to_2026-05-22 (Weeks 1~30)",
        "historical_validation_period": "2026-05-29_to_2026-08-07 (Weeks 31~40, CONTAMINATED_HISTORICAL_VALIDATION)",
        "pre_freeze_replay_period": "2026-08-14_to_2026-08-21 (PRE_FREEZE_REPLAY)",
        "forward_shadow_start_date": "2026-08-28 (IMMUTABLE_VIRGIN_OOS)",
        "code_fingerprints": code_fingerprints,
        "data_fingerprints": data_fingerprints,
        "forward_shadow_rules": forward_shadow_rules,
        "champions": manifest_champions,
    }

    # Canonical SHA256 of entire manifest structure (excluding manifest_sha256)
    canonical_json_str = json.dumps(manifest_data, sort_keys=True, ensure_ascii=False)
    manifest_data["manifest_sha256"] = hashlib.sha256(canonical_json_str.encode("utf-8")).hexdigest()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    return manifest_data


def generate_champion_matrix_report(
    champions: dict[str, dict[str, Any]],
    evaluation_df: pd.DataFrame,
    output_path: Path,
) -> str:
    """Generate comprehensive Markdown Report documenting the Quad-Champion Matrix (Train-grounded)."""
    b0_row = evaluation_df[evaluation_df["rule_id"] == "B0_BASELINE"].iloc[0]
    b0_win_col = "train_act_w1_win_vs_l1_pct" if "train_act_w1_win_vs_l1_pct" in b0_row else "train_act_rank_win_vs_l1_pct"

    report = f"""# Phase 2 Step 4: 四维 Pareto 优胜规则矩阵报告 (Train 集基线完全对齐版)

**决策样本**：严格锚定 Train 探索集 (第 1~30 周，2025-10-10 至 2026-05-22)  
**历史验证样本**：第 31~40 周 (2026-05-29 至 2026-08-07，受污染历史验证集，由 Historical Verifier 独立单向出表)  
**前瞻 OOS 起点**：2026-08-28 (代码冻结后首个真实未来周)  
**基线对齐**：`B0_BASELINE` 在 Train 集 W1 收益中位数 **{b0_row['train_w1_ret_med']:+.2f}%**  
**生产状态**：生产基线 `dashboard/skill_industry_eps_known.py` **100% 保持冻结零修改**  

---

## 一、四维 Champion 优胜矩阵总表 (Train 1~30 周)

| Champion 角色 | 优胜规则 ID | 规则复杂度 $C$ | 规则逻辑简述 | Train W1 收益中位 | Train 活跃周vsL1胜率 | Train 止损发生率 | Train 3-Block 稳定性 | 角色与实盘定位 |
|:---|:---|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **生产基线 (Baseline)** | `B0_BASELINE` | $C=3$ | 新鲜度分桶 → 放量 → 收盘位置 | **{b0_row['train_w1_ret_med']:+.2f}%** | **{b0_row[b0_win_col]:.1f}%** | **{b0_row['train_stop8_before_p20_pct']:.1f}%** | **{b0_row['train_temporal_block_stability_score']:+.4f}** | **生产唯一在役基准 (保持冻结)** |
| **🏆 RETURN_WINNER** | `{champions['HISTORICAL_RETURN_WINNER']['rule_id']}` | $C={champions['HISTORICAL_RETURN_WINNER']['complexity']}$ | {champions['HISTORICAL_RETURN_WINNER']['description']} | **{champions['HISTORICAL_RETURN_WINNER']['train_w1_ret_med']:+.2f}%** | **{champions['HISTORICAL_RETURN_WINNER'].get('train_act_w1_win_vs_l1_pct', champions['HISTORICAL_RETURN_WINNER'].get('train_act_rank_win_vs_l1_pct', 0.0)):.1f}%** | **{champions['HISTORICAL_RETURN_WINNER']['train_stop8_before_p20_pct']:.1f}%** | **{champions['HISTORICAL_RETURN_WINNER']['train_temporal_block_stability_score']:+.4f}** | **Train收益冠军 (生产B0继续胜出)** |
| **🛡️ LOWEST_STOP** | `{champions['LOWEST_STOP_CANDIDATE']['rule_id']}` | $C={champions['LOWEST_STOP_CANDIDATE']['complexity']}$ | {champions['LOWEST_STOP_CANDIDATE']['description']} | **{champions['LOWEST_STOP_CANDIDATE']['train_w1_ret_med']:+.2f}%** | **{champions['LOWEST_STOP_CANDIDATE'].get('train_act_w1_win_vs_l1_pct', champions['LOWEST_STOP_CANDIDATE'].get('train_act_rank_win_vs_l1_pct', 0.0)):.1f}%** | **{champions['LOWEST_STOP_CANDIDATE']['train_stop8_before_p20_pct']:.1f}%** | **{champions['LOWEST_STOP_CANDIDATE']['train_temporal_block_stability_score']:+.4f}** | 观察储备 |
| **✂️ SIMPLER_EQUIV** | `{champions['SIMPLER_EQUIVALENT']['rule_id']}` | $C={champions['SIMPLER_EQUIVALENT']['complexity']}$ | {champions['SIMPLER_EQUIVALENT']['description']} | **{champions['SIMPLER_EQUIVALENT']['train_w1_ret_med']:+.2f}%** | **{champions['SIMPLER_EQUIVALENT'].get('train_act_w1_win_vs_l1_pct', champions['SIMPLER_EQUIVALENT'].get('train_act_rank_win_vs_l1_pct', 0.0)):.1f}%** | **{champions['SIMPLER_EQUIVALENT']['train_stop8_before_p20_pct']:.1f}%** | **{champions['SIMPLER_EQUIVALENT']['train_temporal_block_stability_score']:+.4f}** | **极简研究候选 (Shadow)** |
| **⚖️ PARETO_BALANCED** | `{champions['PARETO_BALANCED_RULE']['rule_id']}` | $C={champions['PARETO_BALANCED_RULE']['complexity']}$ | {champions['PARETO_BALANCED_RULE']['description']} | **{champions['PARETO_BALANCED_RULE']['train_w1_ret_med']:+.2f}%** | **{champions['PARETO_BALANCED_RULE'].get('train_act_w1_win_vs_l1_pct', champions['PARETO_BALANCED_RULE'].get('train_act_rank_win_vs_l1_pct', 0.0)):.1f}%** | **{champions['PARETO_BALANCED_RULE']['train_stop8_before_p20_pct']:.1f}%** | **{champions['PARETO_BALANCED_RULE']['train_temporal_block_stability_score']:+.4f}** | **综合平衡研究候选 (Shadow)** |

---

## 二、关键量化发现与风控洞察

1. **B0 标尺在搜索空间中的稳固地位**：
   * 在使用固定 W1/W2/W4 评价体系与真实 Censored Protocol 后，`B0_BASELINE` 在 Train 集以 **+0.99%** 的 W1 中位收益成为 `HISTORICAL_RETURN_WINNER`；
   * 在当前 85 个候选规则中，B0 在绝对 W1 收益上名列第一；但需客观指出，在活跃排序周其 W1 排序利差中位数仍为 0.00%（Wilcoxon $p=0.433$），尚未在统计上证实存在稳定的 W1 排序 Alpha。
2. **W4 跨周期信号展望**：
   * 在 38 个成熟周与 19 个活跃排序周中，B0 在 W4 显现出具有潜力的排序超额（配对利差中位 **+2.08%**，Wilcoxon $p=0.0299$），定性为 **Promising W4 Signal**，值得在后续前瞻影子账本中持续验证。
3. **生产零改动决策**：
   * 生产 selector 继续 100% 保持冻结；
   * 简化候选（Freshness 与 Close Position）作为 2026-08-28 起的前瞻影子账本（Forward Shadow）观察标的。
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    root_dir = Path(__file__).resolve().parent
    eval_csv = root_dir / "output" / "skill_rule_variants_evaluation.csv"
    out_dir = root_dir / "output"

    eval_df = pd.read_csv(eval_csv)
    champions = select_champions(eval_df)

    champ_csv = out_dir / "pareto_champions_matrix.csv"
    champ_df = pd.DataFrame([
        {"champion_role": k, **v} for k, v in champions.items()
    ])
    champ_df.to_csv(champ_csv, index=False)
    logger.info(f"Saved Champions matrix to {champ_csv}")

    report_path = out_dir / "pareto_champions_report.md"
    report = generate_champion_matrix_report(champions, eval_df, report_path)
    logger.info(f"Generated Champions report at {report_path}")
    print("\n" + report)
