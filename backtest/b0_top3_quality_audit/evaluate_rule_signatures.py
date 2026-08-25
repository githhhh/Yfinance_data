"""Evaluate deduplicated rule signatures across all 40 weeks with stratified analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.b0_top3_quality_audit.skill_rule_engine import (
    RuleSpec,
    build_skill_rule_space,
    deduplicate_rule_signatures,
    evaluate_rule_on_pool,
)
from backtest.b0_top3_quality_audit.three_tier_baseline import compute_portfolio_metrics

logger = logging.getLogger(__name__)


def evaluate_rules_on_train(
    events_train_df: pd.DataFrame,
    weekly_train_df: pd.DataFrame,
    baseline_train_df: pd.DataFrame,
    rules: list[RuleSpec],
    pick_limit: int = 3,
) -> pd.DataFrame:
    """Evaluate rule specifications strictly on the Train Set (Weeks 1~30) with physical isolation.
    
    This function has ZERO access to Weeks 31~40 and produces ONLY Train-specific metrics.
    """
    train_weeks = sorted(baseline_train_df["snapshot_date"].unique())

    snapshot_data: dict[str, dict[str, Any]] = {}
    for snap_date in train_weeks:
        snap_events = events_train_df[events_train_df["snapshot_date"] == snap_date].copy()
        snap_events["snapshot_date"] = snap_date
        snap_event_dict = {str(r["code"]): r.to_dict() for _, r in snap_events.iterrows()}
        snap_weekly_dict = {
            (str(w["code"]), int(w["holding_week_index"])): w.to_dict()
            for _, w in weekly_train_df[weekly_train_df["snapshot_date"] == snap_date].iterrows()
        }
        snapshot_data[snap_date] = {
            "events_df": snap_events,
            "event_dict": snap_event_dict,
            "weekly_dict": snap_weekly_dict,
        }

    b0_weekly_map = baseline_train_df.set_index("snapshot_date").to_dict(orient="index")
    records: list[dict[str, Any]] = []

    for rule in rules:
        weekly_records: list[dict[str, Any]] = []
        for snap_date in train_weeks:
            s_data = snapshot_data[snap_date]
            selected_codes = evaluate_rule_on_pool(s_data["events_df"], rule, pick_limit=pick_limit)
            m = compute_portfolio_metrics(selected_codes, s_data["event_dict"], s_data["weekly_dict"], snap_date)

            b0_info = b0_weekly_map.get(snap_date, {})
            l0_exec_med = b0_info.get("l0_executed_ret_med", np.nan)
            l1_exec_med = b0_info.get("l1_executed_ret_med", np.nan)
            l1_w1_med = b0_info.get("l1_w1_ret_med", np.nan)
            l1_w2_med = b0_info.get("l1_w2_ret_med", np.nan)
            l1_w4_med = b0_info.get("l1_w4_ret_med", np.nan)
            l1_pool_size = b0_info.get("l1_pool_size", 0)

            weekly_records.append({
                "snapshot_date": snap_date,
                "is_active_ranking": l1_pool_size >= 4,
                "picks_count": len(selected_codes),
                "is_portfolio_valid": m["is_portfolio_valid"],
                "invalid_reason": m["invalid_reason"],
                "executed_return": m["executed_return"],
                "w1_return": m["w1_return"],
                "w2_return": m["w2_return"],
                "w4_return": m["w4_return"],
                "stop8_before_profit20": m["stop8_before_profit20"],
                "b0_w1_ret": b0_info.get("l2_w1_ret", np.nan),
                "b0_w2_ret": b0_info.get("l2_w2_ret", np.nan),
                "b0_w4_ret": b0_info.get("l2_w4_ret", np.nan),
                "b0_exec_ret": b0_info.get("l2_executed_ret", np.nan),
                "win_vs_l1_w1": bool(m["w1_return"] > l1_w1_med) if not np.isnan(m["w1_return"]) and not np.isnan(l1_w1_med) else False,
                "win_vs_l1_w2": bool(m["w2_return"] > l1_w2_med) if not np.isnan(m["w2_return"]) and not np.isnan(l1_w2_med) else False,
                "win_vs_l1_w4": bool(m["w4_return"] > l1_w4_med) if not np.isnan(m["w4_return"]) and not np.isnan(l1_w4_med) else False,
                "win_vs_l1_exec": bool(m["executed_return"] > l1_exec_med) if not np.isnan(m["executed_return"]) and not np.isnan(l1_exec_med) else False,
            })

        wdf = pd.DataFrame(weekly_records)
        train_total_weeks = len(wdf)
        train_valid_weeks = int(wdf["is_portfolio_valid"].sum())
        train_valid_rate = float(train_valid_weeks / train_total_weeks * 100.0) if train_total_weeks > 0 else 0.0

        train_w1_med = float(wdf["w1_return"].median()) if not wdf.empty else np.nan
        train_w2_med = float(wdf["w2_return"].median()) if not wdf.empty else np.nan
        train_w4_med = float(wdf["w4_return"].median()) if not wdf.empty else np.nan
        train_exec_med = float(wdf["executed_return"].median()) if not wdf.empty else np.nan
        train_stop_rate = float(wdf["stop8_before_profit20"].mean() * 100.0) if not wdf.empty else np.nan
        train_full3_weeks = int((wdf["picks_count"] == 3).sum())
        train_win_vs_l1_w1 = float(wdf["win_vs_l1_w1"].mean() * 100.0) if not wdf.empty else np.nan
        train_win_vs_l1_exec = float(wdf["win_vs_l1_exec"].mean() * 100.0) if not wdf.empty else np.nan

        # Paired Deltas
        paired_df = wdf[wdf["is_portfolio_valid"] & wdf["b0_w1_ret"].notna()].copy()
        paired_w1 = paired_df["w1_return"] - paired_df["b0_w1_ret"]
        paired_w2 = paired_df["w2_return"] - paired_df["b0_w2_ret"]
        paired_w4 = paired_df["w4_return"] - paired_df["b0_w4_ret"]
        paired_exec = paired_df["executed_return"] - paired_df["b0_exec_ret"]

        # Active ranking opportunities (l1_pool >= 4)
        act_df = wdf[wdf["is_active_ranking"]]
        act_paired = act_df[act_df["is_portfolio_valid"] & act_df["b0_w1_ret"].notna()]
        act_w1_spread = act_paired["w1_return"] - act_paired["b0_w1_ret"]
        train_act_w1_win = float(act_df["win_vs_l1_w1"].mean() * 100.0) if not act_df.empty else np.nan
        train_act_w2_win = float(act_df["win_vs_l1_w2"].mean() * 100.0) if not act_df.empty else np.nan
        train_act_w4_win = float(act_df["win_vs_l1_w4"].mean() * 100.0) if not act_df.empty else np.nan
        train_act_exec_win = float(act_df["win_vs_l1_exec"].mean() * 100.0) if not act_df.empty else np.nan

        # 3-block temporal stability
        fold1 = wdf.iloc[:10]["w1_return"].dropna()
        fold2 = wdf.iloc[10:20]["w1_return"].dropna()
        fold3 = wdf.iloc[20:30]["w1_return"].dropna()
        f_meds = [
            float(fold1.median()) if not fold1.empty else 0.0,
            float(fold2.median()) if not fold2.empty else 0.0,
            float(fold3.median()) if not fold3.empty else 0.0,
        ]
        wf_mean = float(np.mean(f_meds))
        wf_std = float(np.std(f_meds))
        temporal_stability = round(wf_mean - 0.5 * wf_std, 4)

        records.append({
            "rule_id": rule.rule_id,
            "description": rule.description,
            "complexity": rule.complexity,
            "train_w1_ret_med": train_w1_med,
            "train_w2_ret_med": train_w2_med,
            "train_w4_ret_med": train_w4_med,
            "train_exec_ret_med": train_exec_med,
            "train_paired_w1_spread_med": float(paired_w1.median()) if not paired_w1.empty else np.nan,
            "train_paired_w2_spread_med": float(paired_w2.median()) if not paired_w2.empty else np.nan,
            "train_paired_w4_spread_med": float(paired_w4.median()) if not paired_w4.empty else np.nan,
            "train_paired_exec_spread_med": float(paired_exec.median()) if not paired_exec.empty else np.nan,
            "train_paired_win_rate_vs_b0": float((paired_w1 > 0).mean() * 100.0) if not paired_w1.empty else np.nan,
            "train_act_opp_w1_spread_med": float(act_w1_spread.median()) if not act_w1_spread.empty else np.nan,
            "train_act_opp_w1_ret_med": float(act_df["w1_return"].median()) if not act_df.empty else np.nan,
            "train_act_w1_win_vs_l1_pct": train_act_w1_win,
            "train_act_rank_win_vs_l1_pct": train_act_w1_win,
            "train_act_w2_win_vs_l1_pct": train_act_w2_win,
            "train_act_w4_win_vs_l1_pct": train_act_w4_win,
            "train_act_exec_win_vs_l1_pct": train_act_exec_win,
            "train_temporal_block_stability_score": temporal_stability,
            "train_temporal_min_block_ret": float(np.min(f_meds)),
            "train_valid_portfolio_rate": train_valid_rate,
            "train_valid_weeks": train_valid_weeks,
            "train_total_weeks": train_total_weeks,
            "train_full3_weeks": train_full3_weeks,
            "train_ranking_opportunity_weeks": len(act_df),
            "train_stop8_before_p20_pct": train_stop_rate,
            "train_win_vs_l1_pct": train_win_vs_l1_w1,
            "train_win_vs_l1_exec_pct": train_win_vs_l1_exec,
        })

    return pd.DataFrame(records)


def evaluate_all_rules(
    events_df: pd.DataFrame,
    weekly_outcomes_df: pd.DataFrame,
    weekly_baseline_df: pd.DataFrame,
    rules: list[RuleSpec],
    train_weeks: list[str],
    holdout_weeks: list[str],
    pick_limit: int = 3,
) -> pd.DataFrame:
    """Evaluate each rule specification across 40 weeks and compute metrics."""
    valid_weeks = sorted(weekly_baseline_df["snapshot_date"].unique())

    # Precompute snapshots to avoid repeated dataframe filtering
    snapshot_data: dict[str, dict[str, Any]] = {}
    for snap_date in valid_weeks:
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

    b0_weekly_map = weekly_baseline_df.set_index("snapshot_date").to_dict(orient="index")
    rule_summary_records: list[dict[str, Any]] = []

    for rule in rules:
        weekly_records: list[dict[str, Any]] = []

        for snap_date in valid_weeks:
            s_data = snapshot_data[snap_date]
            selected_codes = evaluate_rule_on_pool(s_data["events_df"], rule, pick_limit=pick_limit)
            m = compute_portfolio_metrics(selected_codes, s_data["event_dict"], s_data["weekly_dict"], snap_date)

            b0_info = b0_weekly_map.get(snap_date, {})
            l0_exec_med = b0_info.get("l0_executed_ret_med", np.nan)
            l1_exec_med = b0_info.get("l1_executed_ret_med", np.nan)
            l1_w1_med = b0_info.get("l1_w1_ret_med", np.nan)
            l1_w2_med = b0_info.get("l1_w2_ret_med", np.nan)
            l1_w4_med = b0_info.get("l1_w4_ret_med", np.nan)
            l1_pool_size = b0_info.get("l1_pool_size", 0)

            rec = {
                "snapshot_date": snap_date,
                "is_train": snap_date in train_weeks,
                "is_holdout": snap_date in holdout_weeks,
                "is_active_ranking": l1_pool_size >= 4,
                "picks_count": len(selected_codes),
                "is_portfolio_valid": m["is_portfolio_valid"],
                "invalid_reason": m["invalid_reason"],
                "executed_return": m["executed_return"],
                "w1_return": m["w1_return"],
                "w2_return": m["w2_return"],
                "w4_return": m["w4_return"],
                "stop8_before_profit20": m["stop8_before_profit20"],
                "stop_8_hit_ever": m["stop_8_hit_ever"],
                "l0_exec_med": l0_exec_med,
                "l1_exec_med": l1_exec_med,
                "b0_w1_ret": b0_info.get("l2_w1_ret", np.nan),
                "b0_w2_ret": b0_info.get("l2_w2_ret", np.nan),
                "b0_w4_ret": b0_info.get("l2_w4_ret", np.nan),
                "b0_exec_ret": b0_info.get("l2_executed_ret", np.nan),
                "win_vs_l0": bool(m["executed_return"] > l0_exec_med) if not np.isnan(m["executed_return"]) and not np.isnan(l0_exec_med) else False,
                "win_vs_l1_w1": bool(m["w1_return"] > l1_w1_med) if not np.isnan(m["w1_return"]) and not np.isnan(l1_w1_med) else False,
                "win_vs_l1_w2": bool(m["w2_return"] > l1_w2_med) if not np.isnan(m["w2_return"]) and not np.isnan(l1_w2_med) else False,
                "win_vs_l1_w4": bool(m["w4_return"] > l1_w4_med) if not np.isnan(m["w4_return"]) and not np.isnan(l1_w4_med) else False,
                "win_vs_l1_exec": bool(m["executed_return"] > l1_exec_med) if not np.isnan(m["executed_return"]) and not np.isnan(l1_exec_med) else False,
            }
            weekly_records.append(rec)

        wdf = pd.DataFrame(weekly_records)

        # 1. Train Set Metrics & Common Support vs B0 (Weeks 1~30)
        train_df = wdf[wdf["is_train"]]
        train_total_weeks = len(train_df)
        train_valid_weeks = int(train_df["is_portfolio_valid"].sum())
        train_valid_rate = float(train_valid_weeks / train_total_weeks * 100.0) if train_total_weeks > 0 else 0.0
        
        train_w1_med = float(train_df["w1_return"].median()) if not train_df.empty else np.nan
        train_w2_med = float(train_df["w2_return"].median()) if not train_df.empty else np.nan
        train_w4_med = float(train_df["w4_return"].median()) if not train_df.empty else np.nan
        train_exec_med = float(train_df["executed_return"].median()) if not train_df.empty else np.nan
        train_stop_rate = float(train_df["stop8_before_profit20"].mean() * 100.0) if not train_df.empty else np.nan
        train_full3_weeks = int((train_df["picks_count"] == 3).sum())
        train_win_vs_l1_w1 = float(train_df["win_vs_l1_w1"].mean() * 100.0) if not train_df.empty else np.nan
        train_win_vs_l1_exec = float(train_df["win_vs_l1_exec"].mean() * 100.0) if not train_df.empty else np.nan

        # Paired Common-Support Deltas vs B0 on Train Set
        paired_train_df = train_df[train_df["is_portfolio_valid"] & train_df["b0_w1_ret"].notna()].copy()
        paired_train_weeks = len(paired_train_df)
        paired_w1_spreads = paired_train_df["w1_return"] - paired_train_df["b0_w1_ret"]
        paired_w2_spreads = paired_train_df["w2_return"] - paired_train_df["b0_w2_ret"]
        paired_w4_spreads = paired_train_df["w4_return"] - paired_train_df["b0_w4_ret"]
        paired_exec_spreads = paired_train_df["executed_return"] - paired_train_df["b0_exec_ret"]

        train_paired_w1_spread_med = float(paired_w1_spreads.median()) if not paired_w1_spreads.empty else np.nan
        train_paired_w2_spread_med = float(paired_w2_spreads.median()) if not paired_w2_spreads.empty else np.nan
        train_paired_w4_spread_med = float(paired_w4_spreads.median()) if not paired_w4_spreads.empty else np.nan
        train_paired_exec_spread_med = float(paired_exec_spreads.median()) if not paired_exec_spreads.empty else np.nan
        train_paired_win_rate_vs_b0 = float((paired_w1_spreads > 0).mean() * 100.0) if not paired_w1_spreads.empty else np.nan

        # Active Ranking Opportunity Subset in Train Set (l1_pool >= 4)
        train_act_df = train_df[train_df["is_active_ranking"]]
        train_act_opp_weeks = len(train_act_df)
        train_act_w1_med = float(train_act_df["w1_return"].median()) if not train_act_df.empty else np.nan
        train_act_exec_med = float(train_act_df["executed_return"].median()) if not train_act_df.empty else np.nan
        train_act_w1_win = float(train_act_df["win_vs_l1_w1"].mean() * 100.0) if not train_act_df.empty else np.nan
        train_act_w2_win = float(train_act_df["win_vs_l1_w2"].mean() * 100.0) if not train_act_df.empty else np.nan
        train_act_w4_win = float(train_act_df["win_vs_l1_w4"].mean() * 100.0) if not train_act_df.empty else np.nan
        train_act_exec_win = float(train_act_df["win_vs_l1_exec"].mean() * 100.0) if not train_act_df.empty else np.nan

        paired_act_df = train_act_df[train_act_df["is_portfolio_valid"] & train_act_df["b0_w1_ret"].notna()]
        paired_act_w1_spreads = paired_act_df["w1_return"] - paired_act_df["b0_w1_ret"]
        train_act_opp_w1_spread_med = float(paired_act_w1_spreads.median()) if not paired_act_w1_spreads.empty else np.nan

        # 2. Train Internal 3-Fold Walk-Forward Stability Check (10-week sequential blocks)
        fold1_df = train_df.iloc[:10]
        fold2_df = train_df.iloc[10:20]
        fold3_df = train_df.iloc[20:30]
        
        f1_w1 = float(fold1_df["w1_return"].median()) if not fold1_df["w1_return"].dropna().empty else 0.0
        f2_w1 = float(fold2_df["w1_return"].median()) if not fold2_df["w1_return"].dropna().empty else 0.0
        f3_w1 = float(fold3_df["w1_return"].median()) if not fold3_df["w1_return"].dropna().empty else 0.0
        
        fold_medians = [f1_w1, f2_w1, f3_w1]
        wf_mean = float(np.mean(fold_medians))
        wf_std = float(np.std(fold_medians))
        # Stability score: reward higher mean, penalize high fold variance
        wf_stability_score = round(wf_mean - 0.5 * wf_std, 4)
        wf_min_fold = float(np.min(fold_medians))

        # 3. Full Sample (40 Weeks) Metrics
        full_exec_med = float(wdf["executed_return"].median())
        full_w1_med = float(wdf["w1_return"].median())
        full_w2_med = float(wdf["w2_return"].median())
        full_w4_med = float(wdf["w4_return"].median())
        full_stop_rate = float(wdf["stop8_before_profit20"].mean() * 100.0)
        full_active_weeks = int((wdf["picks_count"] >= 1).sum())
        full_full3_weeks = int((wdf["picks_count"] == 3).sum())
        full_win_vs_l0 = float(wdf["win_vs_l0"].mean() * 100.0)
        full_win_vs_l1 = float(wdf["win_vs_l1_w1"].mean() * 100.0)

        # 4. Historical Validation Set (Weeks 31~40, evaluated for disclosure)
        hold_df = wdf[wdf["is_holdout"]]
        hold_exec_med = float(hold_df["executed_return"].median()) if not hold_df.empty else np.nan
        hold_w1_med = float(hold_df["w1_return"].median()) if not hold_df.empty else np.nan
        hold_stop_rate = float(hold_df["stop8_before_profit20"].mean() * 100.0) if not hold_df.empty else np.nan
        hold_win_vs_l1 = float(hold_df["win_vs_l1_w1"].mean() * 100.0) if not hold_df.empty else np.nan

        rule_summary_records.append({
            "rule_id": rule.rule_id,
            "description": rule.description,
            "complexity": rule.complexity,
            # Primary Selection Anchors (Train Set Weeks 1~30)
            "train_w1_ret_med": train_w1_med,
            "train_w2_ret_med": train_w2_med,
            "train_w4_ret_med": train_w4_med,
            "train_paired_w1_spread_med": train_paired_w1_spread_med,
            "train_paired_w2_spread_med": train_paired_w2_spread_med,
            "train_paired_w4_spread_med": train_paired_w4_spread_med,
            "train_paired_win_rate_vs_b0": train_paired_win_rate_vs_b0,
            "train_act_opp_w1_spread_med": train_act_opp_w1_spread_med,
            "train_act_opp_w1_ret_med": train_act_w1_med,
            "train_act_w1_win_vs_l1_pct": train_act_w1_win,
            "train_act_rank_win_vs_l1_pct": train_act_w1_win,
            "train_act_w2_win_vs_l1_pct": train_act_w2_win,
            "train_act_w4_win_vs_l1_pct": train_act_w4_win,
            "train_act_exec_win_vs_l1_pct": train_act_exec_win,
            "train_temporal_block_stability_score": wf_stability_score,
            "train_temporal_min_block_ret": wf_min_fold,
            "train_valid_portfolio_rate": train_valid_rate,
            "train_valid_weeks": train_valid_weeks,
            "train_total_weeks": train_total_weeks,
            "train_full3_weeks": train_full3_weeks,
            "train_ranking_opportunity_weeks": train_act_opp_weeks,
            "train_stop8_before_p20_pct": train_stop_rate,
            # Secondary Descriptive Metrics (Train As-Of)
            "train_exec_ret_med": train_exec_med,
            "train_paired_exec_spread_med": train_paired_exec_spread_med,
            "train_act_rank_exec_med": train_act_exec_med,
            "train_win_vs_l1_pct": train_win_vs_l1_w1,
            "train_win_vs_l1_exec_pct": train_win_vs_l1_exec,
            # Full Sample (40 Weeks)
            "full_w1_ret_med": full_w1_med,
            "full_w2_ret_med": full_w2_med,
            "full_w4_ret_med": full_w4_med,
            "full_exec_ret_med": full_exec_med,
            "full_stop8_before_p20_pct": full_stop_rate,
            "full_win_vs_l0_pct": full_win_vs_l0,
            "full_win_vs_l1_pct": full_win_vs_l1,
            "full_active_weeks": full_active_weeks,
            "full_full3_weeks": full_full3_weeks,
            # Historical Validation Set (Weeks 31~40)
            "hist_val_w1_ret_med": hold_w1_med,
            "hist_val_exec_ret_med": hold_exec_med,
            "hist_val_stop8_before_p20_pct": hold_stop_rate,
            "hist_val_win_vs_l1_pct": hold_win_vs_l1,
        })

    return pd.DataFrame(rule_summary_records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    root_dir = Path(__file__).resolve().parent
    events_path = root_dir / "data" / "candidate_event_outcomes.parquet"
    weekly_path = root_dir / "data" / "candidate_weekly_outcomes.parquet"
    baseline_path = root_dir / "output" / "three_tier_weekly_comparison.csv"
    out_dir = root_dir / "output"

    events_df = pd.read_parquet(events_path)
    weekly_df = pd.read_parquet(weekly_path)
    baseline_df = pd.read_csv(baseline_path)

    all_weeks = sorted(baseline_df["snapshot_date"].unique())
    train_weeks = all_weeks[:30]
    holdout_weeks = all_weeks[30:]

    logger.info(f"Total Weeks: {len(all_weeks)} (Train: {len(train_weeks)} weeks, Holdout: {len(holdout_weeks)} weeks)")

    # 1. Generate full candidate rule space
    full_rules = build_skill_rule_space()
    logger.info(f"Generated {len(full_rules)} parameterized rule specifications.")

    # 2. Deduplicate signatures on Train set (<= 200 hard budget)
    deduped_rules, signature_map = deduplicate_rule_signatures(
        full_rules, events_df, train_weeks, pick_limit=3, budget_limit=200
    )
    logger.info(f"Deduplicated to {len(deduped_rules)} unique rule signatures on Train Set (Weeks 1~30).")

    # 3. Evaluate all deduplicated rules across all 40 weeks
    results_df = evaluate_all_rules(
        events_df, weekly_df, baseline_df, deduped_rules, train_weeks, holdout_weeks, pick_limit=3
    )

    out_csv = out_dir / "skill_rule_variants_evaluation.csv"
    results_df.to_csv(out_csv, index=False)
    logger.info(f"Saved rule evaluation results to {out_csv}")
