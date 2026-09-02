from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import HORIZONS, add_forward_labels
from .metrics import (
    compare_metrics,
    evaluate_selection,
    included_big_losers,
    macro_average_summary,
    missed_big_winners,
    moving_block_bootstrap_delta,
    weekly_macro_table,
)
from .optimizer import select_all_weeks, two_stage_diagnostics
from .oracle import add_weekly_oracle_flags, oracle_projection
from .report import render_report
from .search_space import generate_core_rules, generate_evidence_ablations
from .selectors import (
    EVIDENCE_FAMILIES,
    primary_rule,
    rule_to_dict,
    select_review_variant,
)
from .utils import (
    content_hash,
    load_pools,
    load_price_cache,
    normalize_eps_pit,
    to_bool,
)
from .walk_forward import (
    choose_retrospective_champion,
    run_walk_forward,
    summarize_oos_stability,
)


POOL_ROOT = Path("backtest/ibd_skill_replay_pools")
PRICE_CACHE = Path("results_pkl/stock_data_290826_1d.pkl")
OUTPUT_DIR = Path("backtest/next_week_review_selection/output")
MIN_4W_COMPLETE_COVERAGE = 0.50
MIN_4W_COMPLETE_ROWS = 10


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run Next Week Review Selection research."
    )
    parser.add_argument("--pool-root", default=str(POOL_ROOT))
    parser.add_argument("--price-cache", default=str(PRICE_CACHE))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--min-train-weeks", type=int, default=20)
    parser.add_argument("--test-weeks", type=int, default=4)
    args = parser.parse_args(argv)

    outputs = run_research(
        pool_root=Path(args.pool_root),
        price_cache=Path(args.price_cache),
        output_dir=Path(args.output_dir),
        min_train_weeks=args.min_train_weeks,
        test_weeks=args.test_weeks,
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


def run_research(
    *,
    pool_root: Path = POOL_ROOT,
    price_cache: Path = PRICE_CACHE,
    output_dir: Path = OUTPUT_DIR,
    min_train_weeks: int = 20,
    test_weeks: int = 4,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    pools = load_pools(pool_root)
    prices = load_price_cache(price_cache)
    eps_path = pool_root / "signal_eps_pit.csv"
    eps = normalize_eps_pit(pd.read_csv(eps_path))

    panel = build_weekend_event_panel(pools, eps)
    panel = add_forward_labels(panel, prices)
    panel = add_weekly_oracle_flags(panel)
    evaluation_panel = mature_four_week_panel(panel)
    if evaluation_panel.empty:
        raise ValueError("No sufficiently mature 4W snapshot weeks for research evaluation")

    core = primary_rule()
    b0_selected = select_all_weeks(evaluation_panel, None)
    primary_selected = select_all_weeks(evaluation_panel, core)

    b0_metrics = evaluate_selection(
        evaluation_panel, b0_selected, variant="B0_ACTIONABLE_ONLY"
    )
    primary_metrics = compare_metrics(
        evaluate_selection(
            evaluation_panel, primary_selected, variant=core.name
        ),
        b0_metrics,
    )
    baseline_vs_primary = pd.DataFrame([b0_metrics, primary_metrics])

    b0_weekly = weekly_macro_table(
        evaluation_panel, b0_selected, variant="B0_ACTIONABLE_ONLY"
    )
    primary_weekly = weekly_macro_table(
        evaluation_panel, primary_selected, variant=core.name
    )
    weekly_macro = pd.concat([b0_weekly, primary_weekly], ignore_index=True)
    macro_summary = macro_average_summary(weekly_macro)
    bootstrap = moving_block_bootstrap_delta(
        b0_weekly,
        primary_weekly,
        metrics=[
            "opportunity_recall_1w",
            "tradable_big_winner_recall_mean_2_4w",
            "tradable_winner_capture_lift_mean_2_4w",
            "tradable_loser_capture_lift_mean_2_4w",
            "opp_severe_loser_exposure_mean_2_4w",
            "avg_watchlist_size",
        ],
    )

    core_rules = generate_core_rules()
    full_evidence_rules, core_grid, evidence_grid = two_stage_diagnostics(
        evaluation_panel, core_rules
    )

    (
        fold_results,
        wf_champions,
        wf_core_train,
        wf_evidence_train,
    ) = run_walk_forward(
        evaluation_panel,
        core_rules,
        min_train_weeks=min_train_weeks,
        test_weeks=test_weeks,
    )
    oos_stability = summarize_oos_stability(fold_results)

    candidate_rules: dict[str, object] = {}
    for rule in full_evidence_rules:
        candidate_rules[rule.name] = rule
    if not wf_evidence_train.empty:
        finalist_core_names = {
            str(name).split("_ALL")[0].split("_NO_")[0]
            for name in wf_evidence_train["variant"].dropna().astype(str)
        }
        for core_rule in core_rules:
            if core_rule.name in finalist_core_names:
                for rule in generate_evidence_ablations(core_rule):
                    candidate_rules[rule.name] = rule

    champion = choose_retrospective_champion(
        oos_stability,
        fold_results,
        list(candidate_rules.values()),
    )
    champion_status = (
        "RETROSPECTIVE_CANDIDATE"
        if champion is not None
        else "NO_STABLE_NEXT_WEEK_REVIEW_RULE"
    )
    champion_name = champion.name if champion is not None else ""

    missed_winners, included_losers = build_case_audits(
        evaluation_panel,
        core_rule=core,
        champion=champion,
    )

    latest_review = latest_review_list(panel, core)
    champion_latest = (
        latest_review_list(panel, champion)
        if champion is not None
        else pd.DataFrame()
    )
    extended = extended_exploratory_summary(panel)

    manifest = experiment_manifest(
        panel=panel,
        evaluation_panel=evaluation_panel,
        pool_root=pool_root,
        price_cache=price_cache,
        eps_path=eps_path,
        core_rule_count=len(core_rules),
        evidence_rule_count=len(full_evidence_rules),
        min_train_weeks=min_train_weeks,
        test_weeks=test_weeks,
        champion_status=champion_status,
        champion=champion,
    )
    data_audit = render_data_audit(panel, evaluation_panel, pools, eps)
    report = render_report(
        baseline_vs_primary=baseline_vs_primary,
        macro_summary=macro_summary,
        bootstrap=bootstrap,
        oos_stability=oos_stability,
        walk_forward_champions=wf_champions,
        champion_status=champion_status,
        champion_rule=champion_name or "n/a",
        extended_exploratory=extended,
    )

    outputs = {
        "data_audit.md": output_dir / "data_audit.md",
        "weekend_event_panel.csv": output_dir / "weekend_event_panel.csv",
        "winner_loser_oracle.csv": output_dir / "winner_loser_oracle.csv",
        "next_week_review_list.csv": output_dir / "next_week_review_list.csv",
        "champion_latest_review_list.csv": output_dir / "champion_latest_review_list.csv",
        "baseline_vs_primary.csv": output_dir / "baseline_vs_primary.csv",
        "weekly_macro_metrics.csv": output_dir / "weekly_macro_metrics.csv",
        "macro_summary.csv": output_dir / "macro_summary.csv",
        "week_block_bootstrap.csv": output_dir / "week_block_bootstrap.csv",
        "core_rule_grid.csv": output_dir / "core_rule_grid.csv",
        "evidence_ablation_grid.csv": output_dir / "evidence_ablation_grid.csv",
        "fold_level_results.csv": output_dir / "fold_level_results.csv",
        "walk_forward_champions.csv": output_dir / "walk_forward_champions.csv",
        "walk_forward_core_train.csv": output_dir / "walk_forward_core_train.csv",
        "walk_forward_evidence_train.csv": output_dir / "walk_forward_evidence_train.csv",
        "rule_stability.csv": output_dir / "rule_stability.csv",
        "missed_big_winners.csv": output_dir / "missed_big_winners.csv",
        "included_big_losers.csv": output_dir / "included_big_losers.csv",
        "extended_exploratory.csv": output_dir / "extended_exploratory.csv",
        "champion_rule.json": output_dir / "champion_rule.json",
        "experiment_manifest.yaml": output_dir / "experiment_manifest.yaml",
        "research_report.md": output_dir / "research_report.md",
    }

    outputs["data_audit.md"].write_text(data_audit, encoding="utf-8")
    panel.to_csv(outputs["weekend_event_panel.csv"], index=False)
    oracle_projection(panel).to_csv(outputs["winner_loser_oracle.csv"], index=False)
    latest_review.to_csv(outputs["next_week_review_list.csv"], index=False)
    champion_latest.to_csv(outputs["champion_latest_review_list.csv"], index=False)
    baseline_vs_primary.to_csv(outputs["baseline_vs_primary.csv"], index=False)
    weekly_macro.to_csv(outputs["weekly_macro_metrics.csv"], index=False)
    macro_summary.to_csv(outputs["macro_summary.csv"], index=False)
    bootstrap.to_csv(outputs["week_block_bootstrap.csv"], index=False)
    core_grid.to_csv(outputs["core_rule_grid.csv"], index=False)
    evidence_grid.to_csv(outputs["evidence_ablation_grid.csv"], index=False)
    fold_results.to_csv(outputs["fold_level_results.csv"], index=False)
    wf_champions.to_csv(outputs["walk_forward_champions.csv"], index=False)
    wf_core_train.to_csv(outputs["walk_forward_core_train.csv"], index=False)
    wf_evidence_train.to_csv(outputs["walk_forward_evidence_train.csv"], index=False)
    oos_stability.to_csv(outputs["rule_stability.csv"], index=False)
    missed_winners.to_csv(outputs["missed_big_winners.csv"], index=False)
    included_losers.to_csv(outputs["included_big_losers.csv"], index=False)
    extended.to_csv(outputs["extended_exploratory.csv"], index=False)
    outputs["champion_rule.json"].write_text(
        json.dumps(
            {
                "status": champion_status,
                "rule": rule_to_dict(champion) if champion is not None else None,
                "note": "retrospective research candidate; not production authorization",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    outputs["experiment_manifest.yaml"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["research_report.md"].write_text(report, encoding="utf-8")
    return {name: str(path) for name, path in outputs.items()}


def build_weekend_event_panel(
    pools: list[tuple[str, pd.DataFrame, Path]],
    eps: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for snapshot, pool, path in pools:
        frame = pool.copy()
        frame["snapshot_date"] = snapshot
        frame["pool_path"] = str(path)
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if raw.empty:
        return raw

    signal = raw[
        raw["signal"].map(to_bool).eq(True)
        & raw["ibd_candidate_rule"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    signal["code"] = signal["code"].astype(str).str.strip()
    signal["_source_row_order"] = np.arange(len(signal))
    signal = (
        signal.sort_values(["snapshot_date", "code", "_source_row_order"])
        .drop_duplicates(["snapshot_date", "code"], keep="first")
        .copy()
    )

    eps_key = eps[
        ["snapshot_date", "code", "pit_eps_state", "pit_eps_yoy_growth"]
    ].copy()
    eps_key["snapshot_date"] = eps_key["snapshot_date"].astype(str)
    eps_key["code"] = eps_key["code"].astype(str).str.strip()
    eps_key = eps_key.drop_duplicates(["snapshot_date", "code"], keep="first")

    panel = signal.merge(eps_key, on=["snapshot_date", "code"], how="left")
    panel["pit_eps_state"] = panel["pit_eps_state"].fillna("UNKNOWN")
    panel.loc[
        panel["pit_eps_state"].ne("VERIFIED"), "pit_eps_yoy_growth"
    ] = pd.NA
    return panel.reset_index(drop=True)


def mature_four_week_panel(
    panel: pd.DataFrame,
    *,
    min_complete_coverage: float = MIN_4W_COMPLETE_COVERAGE,
    min_complete_rows: int = MIN_4W_COMPLETE_ROWS,
) -> pd.DataFrame:
    """Keep weeks with enough complete snapshot-clock 4W rows."""
    if panel.empty:
        return panel.copy()
    stats = panel.groupby("snapshot_date")["forward_4w_censored"].agg(
        total="size",
        complete=lambda values: int(values.eq(False).sum()),
    )
    stats["coverage"] = stats["complete"] / stats["total"].clip(lower=1)
    mature_weeks = stats[
        stats["coverage"].ge(min_complete_coverage)
        & stats["complete"].ge(min_complete_rows)
    ].index.astype(str)
    return panel[
        panel["snapshot_date"].astype(str).isin(mature_weeks)
    ].copy()


def build_case_audits(
    panel: pd.DataFrame,
    *,
    core_rule,
    champion,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = [
        ("B0_ACTIONABLE_ONLY", select_all_weeks(panel, None)),
        (core_rule.name, select_all_weeks(panel, core_rule)),
    ]
    if champion is not None and champion.name != core_rule.name:
        variants.append((champion.name, select_all_weeks(panel, champion)))

    missed_parts = []
    loser_parts = []
    for name, selected in variants:
        missed = missed_big_winners(panel, selected)
        if not missed.empty:
            missed.insert(0, "audit_rule", name)
            missed_parts.append(missed)
        losers = included_big_losers(panel, selected)
        if not losers.empty:
            losers.insert(0, "audit_rule", name)
            loser_parts.append(losers)

    return (
        pd.concat(missed_parts, ignore_index=True)
        if missed_parts
        else pd.DataFrame(),
        pd.concat(loser_parts, ignore_index=True)
        if loser_parts
        else pd.DataFrame(),
    )


def latest_review_list(panel: pd.DataFrame, rule) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    latest = sorted(panel["snapshot_date"].astype(str).unique())[-1]
    week = panel[panel["snapshot_date"].astype(str).eq(latest)].copy()
    selected = select_review_variant(week, rule)
    columns = [
        "snapshot_date",
        "code",
        "selection_source",
        "ibd_entry_status",
        "ibd_candidate_rule",
        "ibd_candidate_price",
        "current_vs_ibd_candidate_pct",
        "_evidence_family_count",
        "review_reason",
        "pit_eps_yoy_growth",
        "volume_ratio",
        "dist_to_52w_high_pct",
        "pullback_v_is_dry",
        "sector",
        "industry",
    ]
    return selected[
        [column for column in columns if column in selected.columns]
    ].copy()


def extended_exploratory_summary(panel: pd.DataFrame) -> pd.DataFrame:
    extended = panel[
        panel["ibd_entry_status"].fillna("").astype(str).str.upper().eq("EXTENDED")
    ].copy()
    if extended.empty:
        return pd.DataFrame()

    vs = pd.to_numeric(
        extended["current_vs_ibd_candidate_pct"], errors="coerce"
    )
    extended["extension_bucket"] = pd.cut(
        vs,
        bins=[5.0, 10.0, 15.0, np.inf],
        labels=["+5_to_10", "+10_to_15", ">+15"],
        include_lowest=False,
        right=True,
    )
    rows = []
    for bucket, group in extended.groupby("extension_bucket", observed=True):
        complete = group[group["forward_1w_censored"].eq(False)]
        rows.append(
            {
                "extension_bucket": str(bucket),
                "rows": len(group),
                "evaluable_1w": len(complete),
                "retest_to_buy_zone_1w_rate": (
                    float(complete["review_opportunity_1w"].mean())
                    if len(complete)
                    else np.nan
                ),
                "median_snapshot_return_4w_pct": _median(
                    group[group["forward_4w_censored"].eq(False)],
                    "forward_4w_return_pct",
                ),
                "median_snapshot_mae_4w_pct": _median(
                    group[group["forward_4w_censored"].eq(False)],
                    "mae_4w_pct",
                ),
                "median_post_retest_return_4w_pct": _median(
                    group[group["opp_forward_4w_censored"].eq(False)],
                    "opp_forward_4w_return_pct",
                ),
                "median_post_retest_mae_4w_pct": _median(
                    group[group["opp_forward_4w_censored"].eq(False)],
                    "opp_mae_4w_pct",
                ),
            }
        )
    return pd.DataFrame(rows)


def render_data_audit(
    panel: pd.DataFrame,
    evaluation_panel: pd.DataFrame,
    pools: list[tuple[str, pd.DataFrame, Path]],
    eps: pd.DataFrame,
) -> str:
    weeks = (
        sorted(panel["snapshot_date"].astype(str).unique())
        if not panel.empty
        else []
    )
    lines = [
        "# Next Week Review Selection - Data Audit",
        "",
        f"- pool files loaded: {len(pools)}",
        f"- active-signal weeks: {len(weeks)}",
        f"- first snapshot: {weeks[0] if weeks else 'n/a'}",
        f"- last snapshot: {weeks[-1] if weeks else 'n/a'}",
        f"- active-signal events: {len(panel)}",
        f"- 4W-mature evaluation weeks: {evaluation_panel['snapshot_date'].astype(str).nunique()}",
        f"- 4W-mature evaluation events: {len(evaluation_panel)}",
        (
            f"- verified PIT EPS rows: {int(eps['pit_eps_state'].eq('VERIFIED').sum())}"
            if not eps.empty
            else "- verified PIT EPS rows: 0"
        ),
    ]
    for horizon in HORIZONS:
        complete = int(panel[f"forward_{horizon}_censored"].eq(False).sum())
        opp_complete = int(
            (
                panel["review_opportunity_1w"].eq(True)
                & panel[f"opp_forward_{horizon}_censored"].eq(False)
            ).sum()
        )
        lines.append(f"- complete snapshot-clock {horizon}: {complete}/{len(panel)}")
        lines.append(f"- complete opportunity-clock {horizon}: {opp_complete}")
    lines.extend(
        [
            "",
            "Guardrails:",
            "- Primary R1 has no Geometry hard reject.",
            "- Volume is one evidence family even if entry and weekly volume both confirm.",
            "- False/missing positive evidence is neutral.",
            "- Snapshot and opportunity clocks are kept separate.",
            "- Winner recall is capacity-normalized with selection coverage/capture lift.",
            "- Rule evolution is two-stage, not one 144-rule sweep.",
            "- Weekly macro metrics + paired moving-block bootstrap are reported.",
            "- C Rank and ATR are not used.",
            "",
        ]
    )
    return "\n".join(lines)


def experiment_manifest(
    *,
    panel: pd.DataFrame,
    evaluation_panel: pd.DataFrame,
    pool_root: Path,
    price_cache: Path,
    eps_path: Path,
    core_rule_count: int,
    evidence_rule_count: int,
    min_train_weeks: int,
    test_weeks: int,
    champion_status: str,
    champion,
) -> dict[str, object]:
    weeks = (
        sorted(panel["snapshot_date"].astype(str).unique())
        if not panel.empty
        else []
    )
    return {
        "study": "next_week_review_selection",
        "status": "retrospective_pre_registered_replay",
        "pool_root": str(pool_root),
        "price_cache": str(price_cache),
        "price_cache_sha256": content_hash(price_cache),
        "eps_pit": str(eps_path),
        "eps_pit_sha256": content_hash(eps_path),
        "active_signal_weeks": len(weeks),
        "mature_4w_evaluation_weeks": int(
            evaluation_panel["snapshot_date"].astype(str).nunique()
        ),
        "first_snapshot": weeks[0] if weeks else "",
        "last_snapshot": weeks[-1] if weeks else "",
        "horizons": HORIZONS,
        "mature_4w_gate": {
            "min_complete_coverage": MIN_4W_COMPLETE_COVERAGE,
            "min_complete_rows": MIN_4W_COMPLETE_ROWS,
        },
        "primary_rule": rule_to_dict(primary_rule()),
        "evidence_families": EVIDENCE_FAMILIES,
        "search": {
            "stage_1_core_rule_count": core_rule_count,
            "stage_2_full_sample_evidence_rule_count": evidence_rule_count,
            "stage_2_policy": "leave_one_evidence_family_out_only_around_core_finalists",
        },
        "walk_forward": {
            "min_train_weeks": min_train_weeks,
            "test_weeks": test_weeks,
            "expanding_window": True,
            "test_used_for_selection": False,
        },
        "champion_status": champion_status,
        "champion_rule": (
            rule_to_dict(champion) if champion is not None else None
        ),
        "c_rank_used": False,
        "atr_used": False,
        "extended_in_core_selector": False,
    }


def _median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if len(values) else np.nan


if __name__ == "__main__":
    main()
