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
            l1_pool_size = b0_info.get("l1_pool_size", 0)

            rec = {
                "snapshot_date": snap_date,
                "is_train": snap_date in train_weeks,
                "is_holdout": snap_date in holdout_weeks,
                "is_active_ranking": l1_pool_size >= 4,
                "picks_count": len(selected_codes),
                "executed_return": m["executed_return"],
                "w1_return": m["w1_return"],
                "stop8_before_profit20": m["stop8_before_profit20"],
                "stop_8_hit_ever": m["stop_8_hit_ever"],
                "l0_exec_med": l0_exec_med,
                "l1_exec_med": l1_exec_med,
                "win_vs_l0": bool(m["executed_return"] > l0_exec_med) if not np.isnan(m["executed_return"]) and not np.isnan(l0_exec_med) else False,
                "win_vs_l1": bool(m["executed_return"] > l1_exec_med) if not np.isnan(m["executed_return"]) and not np.isnan(l1_exec_med) else False,
            }
            weekly_records.append(rec)

        wdf = pd.DataFrame(weekly_records)

        # 1. Full Sample (40 Weeks) Metrics
        full_exec_med = float(wdf["executed_return"].median())
        full_w1_med = float(wdf["w1_return"].median())
        full_stop_rate = float(wdf["stop8_before_profit20"].mean() * 100.0)
        full_active_weeks = int((wdf["picks_count"] >= 1).sum())
        full_full3_weeks = int((wdf["picks_count"] == 3).sum())
        full_win_vs_l0 = float(wdf["win_vs_l0"].mean() * 100.0)
        full_win_vs_l1 = float(wdf["win_vs_l1"].mean() * 100.0)

        # 2. Train Set (Weeks 1~30, strictly used for Champion Selection)
        train_df = wdf[wdf["is_train"]]
        train_exec_med = float(train_df["executed_return"].median()) if not train_df.empty else np.nan
        train_w1_med = float(train_df["w1_return"].median()) if not train_df.empty else np.nan
        train_stop_rate = float(train_df["stop8_before_profit20"].mean() * 100.0) if not train_df.empty else np.nan
        train_full3_weeks = int((train_df["picks_count"] == 3).sum())
        train_win_vs_l1 = float(train_df["win_vs_l1"].mean() * 100.0) if not train_df.empty else np.nan

        # 3. Active Ranking Subset in Train Set (l1_pool >= 4 on Train)
        train_act_df = train_df[train_df["is_active_ranking"]]
        train_act_exec_med = float(train_act_df["executed_return"].median()) if not train_act_df.empty else np.nan
        train_act_win_vs_l1 = float(train_act_df["win_vs_l1"].mean() * 100.0) if not train_act_df.empty else np.nan

        # 4. Sealed Final Holdout (Weeks 31~40, evaluated for disclosure)
        hold_df = wdf[wdf["is_holdout"]]
        hold_exec_med = float(hold_df["executed_return"].median()) if not hold_df.empty else np.nan
        hold_stop_rate = float(hold_df["stop8_before_profit20"].mean() * 100.0) if not hold_df.empty else np.nan
        hold_win_vs_l1 = float(hold_df["win_vs_l1"].mean() * 100.0) if not hold_df.empty else np.nan

        rule_summary_records.append({
            "rule_id": rule.rule_id,
            "description": rule.description,
            "complexity": rule.complexity,
            # Train Set (Weeks 1~30 - Selection Anchor)
            "train_exec_ret_med": train_exec_med,
            "train_w1_ret_med": train_w1_med,
            "train_stop8_before_p20_pct": train_stop_rate,
            "train_win_vs_l1_pct": train_win_vs_l1,
            "train_act_rank_exec_med": train_act_exec_med,
            "train_act_rank_win_vs_l1_pct": train_act_win_vs_l1,
            "train_full3_weeks": train_full3_weeks,
            # Full Sample (40 Weeks)
            "full_exec_ret_med": full_exec_med,
            "full_w1_ret_med": full_w1_med,
            "full_stop8_before_p20_pct": full_stop_rate,
            "full_win_vs_l0_pct": full_win_vs_l0,
            "full_win_vs_l1_pct": full_win_vs_l1,
            "full_active_weeks": full_active_weeks,
            "full_full3_weeks": full_full3_weeks,
            # Sealed Holdout (Weeks 31~40)
            "holdout_exec_ret_med": hold_exec_med,
            "holdout_stop8_before_p20_pct": hold_stop_rate,
            "holdout_win_vs_l1_pct": hold_win_vs_l1,
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
