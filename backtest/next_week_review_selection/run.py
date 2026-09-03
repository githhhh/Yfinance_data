from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .coverage import build_price_path_audits
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
from .sensitivity import setup_balanced_sensitivity
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
    adaptive_policy_status,
    convergent_static_candidate,
    run_walk_forward,
    summarize_adaptive_policy,
    summarize_oos_stability,
    summarize_rule_convergence,
)


POOL_ROOT = Path("backtest/ibd_skill_replay_pools")
PRICE_CACHE = Path("results_pkl/stock_data_290826_1d.pkl")
OUTPUT_DIR = Path("backtest/next_week_review_selection/output")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run Next Week Review Selection research v0.5."
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
    if panel.empty:
        raise ValueError("No active-signal events available")

    price_path_audits = build_price_path_audits(panel)

    # Primary R1 remains a fixed diagnostic baseline.
    core = primary_rule()
    b0_selected = select_all_weeks(panel, None)
    primary_selected = select_all_weeks(panel, core)

    b0_metrics = evaluate_selection(
        panel, b0_selected, variant="B0_ACTIONABLE_ONLY"
    )
    primary_metrics = compare_metrics(
        evaluate_selection(panel, primary_selected, variant=core.name),
        b0_metrics,
    )
    baseline_vs_primary = pd.DataFrame([b0_metrics, primary_metrics])

    b0_weekly = weekly_macro_table(
        panel, b0_selected, variant="B0_ACTIONABLE_ONLY"
    )
    primary_weekly = weekly_macro_table(
        panel, primary_selected, variant=core.name
    )
    weekly_macro = pd.concat([b0_weekly, primary_weekly], ignore_index=True)
    macro_summary = macro_average_summary(weekly_macro)

    bootstrap_metrics = ["opportunity_recall_1w", "avg_watchlist_size"]
    for horizon in ("2w", "3w", "4w"):
        bootstrap_metrics.extend(
            [
                f"tradable_big_winner_recall_{horizon}",
                f"tradable_winner_capture_lift_{horizon}",
                f"tradable_loser_capture_lift_{horizon}",
                f"opp_severe_loser_exposure_{horizon}",
            ]
        )
    bootstrap = moving_block_bootstrap_delta(
        b0_weekly,
        primary_weekly,
        metrics=bootstrap_metrics,
    )

    # Full-sample diagnostics are descriptive only. Formal selection happens
    # inside walk-forward train windows.
    core_rules = generate_core_rules()
    full_evidence_rules, core_grid, evidence_grid = two_stage_diagnostics(
        panel, core_rules
    )

    (
        fold_results,
        wf_champions,
        wf_core_train,
        wf_evidence_train,
        tail_exploratory,
        formal_oos_selections,
        fold_calendar,
    ) = run_walk_forward(
        panel,
        core_rules,
        min_train_weeks=min_train_weeks,
        test_weeks=test_weeks,
    )

    oos_stability = summarize_oos_stability(fold_results)
    adaptive_summary = summarize_adaptive_policy(fold_results)
    adaptive_status = adaptive_policy_status(adaptive_summary)
    convergence = summarize_rule_convergence(wf_champions)
    static_status, static_rule_name = convergent_static_candidate(
        convergence, oos_stability
    )

    formal_weeks = set(
        fold_calendar.loc[
            fold_calendar["phase"].eq("FORMAL_OOS"), "snapshot_date"
        ].astype(str)
    ) if not fold_calendar.empty else set()
    formal_panel = panel[
        panel["snapshot_date"].astype(str).isin(formal_weeks)
    ].copy()

    primary_setup_detail, primary_setup_summary = setup_balanced_sensitivity(
        panel,
        b0_selected,
        primary_selected,
        variant=core.name,
    )

    if not formal_oos_selections.empty and not formal_panel.empty:
        formal_b0 = formal_oos_selections[
            formal_oos_selections["evaluation_role"].eq("B0_ACTIONABLE_ONLY")
        ].copy()
        adaptive_selected = formal_oos_selections[
            formal_oos_selections["evaluation_role"].eq("TRAIN_CHAMPION")
        ].copy()
        adaptive_setup_detail, adaptive_setup_summary = setup_balanced_sensitivity(
            formal_panel,
            formal_b0,
            adaptive_selected,
            variant="ADAPTIVE_POLICY_FORMAL_OOS",
        )
    else:
        adaptive_setup_detail = pd.DataFrame()
        adaptive_setup_summary = pd.DataFrame()

    setup_balanced_summary = pd.concat(
        [primary_setup_summary, adaptive_setup_summary],
        ignore_index=True,
        sort=False,
    )

    all_rule_map = {rule.name: rule for rule in core_rules}
    for core_rule in core_rules:
        for rule in generate_evidence_ablations(core_rule):
            all_rule_map[rule.name] = rule
    for rule in full_evidence_rules:
        all_rule_map[rule.name] = rule

    static_rule = all_rule_map.get(static_rule_name)
    static_latest = (
        latest_review_list(panel, static_rule)
        if static_rule is not None
        else pd.DataFrame()
    )

    missed_winners, included_losers = build_case_audits(
        panel,
        core_rule=core,
        static_rule=static_rule,
    )
    latest_review = latest_review_list(panel, core)
    extended = extended_exploratory_summary(panel)

    formal_fold_count = (
        int(wf_champions["fold"].nunique())
        if not wf_champions.empty
        else 0
    )
    tail_week_count = int(
        fold_calendar["phase"].eq("TAIL_EXPLORATORY").sum()
    ) if not fold_calendar.empty else 0

    manifest = experiment_manifest(
        panel=panel,
        pool_root=pool_root,
        price_cache=price_cache,
        eps_path=eps_path,
        core_rule_count=len(core_rules),
        evidence_rule_count=len(full_evidence_rules),
        min_train_weeks=min_train_weeks,
        test_weeks=test_weeks,
        formal_fold_count=formal_fold_count,
        tail_week_count=tail_week_count,
        adaptive_status=adaptive_status,
        static_status=static_status,
        static_rule=static_rule,
    )
    data_audit = render_data_audit(
        panel,
        pools,
        eps,
        formal_fold_count=formal_fold_count,
        tail_week_count=tail_week_count,
    )
    report = render_report(
        baseline_vs_primary=baseline_vs_primary,
        macro_summary=macro_summary,
        bootstrap=bootstrap,
        coverage_summary=price_path_audits["price_path_coverage_summary.csv"],
        adaptive_summary=adaptive_summary,
        convergence=convergence,
        setup_balanced_summary=setup_balanced_summary,
        oos_stability=oos_stability,
        walk_forward_champions=wf_champions,
        tail_exploratory=tail_exploratory,
        adaptive_status=adaptive_status,
        static_status=static_status,
        static_rule=static_rule_name,
        extended_exploratory=extended,
    )

    outputs = {
        "data_audit.md": output_dir / "data_audit.md",
        "weekend_event_panel.csv": output_dir / "weekend_event_panel.csv",
        "winner_loser_oracle.csv": output_dir / "winner_loser_oracle.csv",
        "next_week_review_list.csv": output_dir / "next_week_review_list.csv",
        "static_candidate_latest_review_list.csv": output_dir / "static_candidate_latest_review_list.csv",
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
        "formal_oos_selections.csv": output_dir / "formal_oos_selections.csv",
        "fold_calendar.csv": output_dir / "fold_calendar.csv",
        "tail_exploratory.csv": output_dir / "tail_exploratory.csv",
        "rule_stability.csv": output_dir / "rule_stability.csv",
        "adaptive_policy_summary.csv": output_dir / "adaptive_policy_summary.csv",
        "rule_convergence.csv": output_dir / "rule_convergence.csv",
        "setup_sensitivity_primary.csv": output_dir / "setup_sensitivity_primary.csv",
        "setup_sensitivity_adaptive_oos.csv": output_dir / "setup_sensitivity_adaptive_oos.csv",
        "setup_balanced_summary.csv": output_dir / "setup_balanced_summary.csv",
        "missed_big_winners.csv": output_dir / "missed_big_winners.csv",
        "included_big_losers.csv": output_dir / "included_big_losers.csv",
        "extended_exploratory.csv": output_dir / "extended_exploratory.csv",
        "champion_rule.json": output_dir / "champion_rule.json",
        "experiment_manifest.yaml": output_dir / "experiment_manifest.yaml",
        "research_report.md": output_dir / "research_report.md",
    }
    for name in price_path_audits:
        outputs[name] = output_dir / name

    outputs["data_audit.md"].write_text(data_audit, encoding="utf-8")
    panel.to_csv(outputs["weekend_event_panel.csv"], index=False)
    oracle_projection(panel).to_csv(outputs["winner_loser_oracle.csv"], index=False)
    latest_review.to_csv(outputs["next_week_review_list.csv"], index=False)
    static_latest.to_csv(
        outputs["static_candidate_latest_review_list.csv"], index=False
    )
    baseline_vs_primary.to_csv(outputs["baseline_vs_primary.csv"], index=False)
    weekly_macro.to_csv(outputs["weekly_macro_metrics.csv"], index=False)
    macro_summary.to_csv(outputs["macro_summary.csv"], index=False)
    bootstrap.to_csv(outputs["week_block_bootstrap.csv"], index=False)
    core_grid.to_csv(outputs["core_rule_grid.csv"], index=False)
    evidence_grid.to_csv(outputs["evidence_ablation_grid.csv"], index=False)
    fold_results.to_csv(outputs["fold_level_results.csv"], index=False)
    wf_champions.to_csv(outputs["walk_forward_champions.csv"], index=False)
    wf_core_train.to_csv(outputs["walk_forward_core_train.csv"], index=False)
    wf_evidence_train.to_csv(
        outputs["walk_forward_evidence_train.csv"], index=False
    )
    formal_oos_selections.to_csv(
        outputs["formal_oos_selections.csv"], index=False
    )
    fold_calendar.to_csv(outputs["fold_calendar.csv"], index=False)
    tail_exploratory.to_csv(outputs["tail_exploratory.csv"], index=False)
    oos_stability.to_csv(outputs["rule_stability.csv"], index=False)
    adaptive_summary.to_csv(outputs["adaptive_policy_summary.csv"], index=False)
    convergence.to_csv(outputs["rule_convergence.csv"], index=False)
    primary_setup_detail.to_csv(
        outputs["setup_sensitivity_primary.csv"], index=False
    )
    adaptive_setup_detail.to_csv(
        outputs["setup_sensitivity_adaptive_oos.csv"], index=False
    )
    setup_balanced_summary.to_csv(
        outputs["setup_balanced_summary.csv"], index=False
    )
    missed_winners.to_csv(outputs["missed_big_winners.csv"], index=False)
    included_losers.to_csv(outputs["included_big_losers.csv"], index=False)
    extended.to_csv(outputs["extended_exploratory.csv"], index=False)
    for name, frame in price_path_audits.items():
        frame.to_csv(outputs[name], index=False)

    dominant = _dominant_convergence(convergence)
    outputs["champion_rule.json"].write_text(
        json.dumps(
            {
                "adaptive_policy_status": adaptive_status,
                "formal_oos_fold_count": formal_fold_count,
                "tail_exploratory_week_count": tail_week_count,
                "static_rule_status": static_status,
                "static_rule": (
                    rule_to_dict(static_rule)
                    if static_rule is not None
                    else None
                ),
                "dominant_convergence": dominant,
                "note": "retrospective research only; no production authorization",
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


def build_case_audits(
    panel: pd.DataFrame,
    *,
    core_rule,
    static_rule,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = [
        ("B0_ACTIONABLE_ONLY", select_all_weeks(panel, None)),
        (core_rule.name, select_all_weeks(panel, core_rule)),
    ]
    if static_rule is not None and static_rule.name != core_rule.name:
        variants.append(
            (static_rule.name, select_all_weeks(panel, static_rule))
        )

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
    if panel.empty or rule is None:
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
    pools: list[tuple[str, pd.DataFrame, Path]],
    eps: pd.DataFrame,
    *,
    formal_fold_count: int,
    tail_week_count: int,
) -> str:
    weeks = sorted(panel["snapshot_date"].astype(str).unique())
    lines = [
        "# Next Week Review Selection - Data Audit v0.5",
        "",
        f"- pool files loaded: {len(pools)}",
        f"- active-signal weeks: {len(weeks)}",
        f"- first snapshot: {weeks[0] if weeks else 'n/a'}",
        f"- last snapshot: {weeks[-1] if weeks else 'n/a'}",
        f"- active-signal events: {len(panel)}",
        f"- formal OOS folds: {formal_fold_count}",
        f"- tail exploratory weeks: {tail_week_count}",
        (
            f"- verified PIT EPS rows: {int(eps['pit_eps_state'].eq('VERIFIED').sum())}"
            if not eps.empty
            else "- verified PIT EPS rows: 0"
        ),
    ]
    for horizon in HORIZONS:
        complete = int(panel[f"forward_{horizon}_censored"].eq(False).sum())
        weeks_complete = int(
            panel.loc[
                panel[f"forward_{horizon}_censored"].eq(False),
                "snapshot_date",
            ]
            .astype(str)
            .nunique()
        )
        lines.append(
            f"- complete snapshot-clock {horizon}: {complete}/{len(panel)} across {weeks_complete} weeks"
        )
    lines.extend(
        [
            "",
            "Guardrails:",
            "- Every formal fold receives a provisional train-only champion.",
            "- Training stability is ranking-only, never an OOS admission veto.",
            "- Partial final test block is excluded from the formal verdict.",
            "- Behaviorally identical train rules are de-duplicated by selection signature.",
            "- Adaptive policy and exact-rule convergence are separate conclusions.",
            "- Setup-balanced sensitivity equal-weights eligible setup strata.",
            "- Horizon-aware as-of censoring and price-path audits remain active.",
            "- C Rank and ATR are not used.",
            "",
        ]
    )
    return "\n".join(lines)


def experiment_manifest(
    *,
    panel: pd.DataFrame,
    pool_root: Path,
    price_cache: Path,
    eps_path: Path,
    core_rule_count: int,
    evidence_rule_count: int,
    min_train_weeks: int,
    test_weeks: int,
    formal_fold_count: int,
    tail_week_count: int,
    adaptive_status: str,
    static_status: str,
    static_rule,
) -> dict[str, object]:
    weeks = sorted(panel["snapshot_date"].astype(str).unique())
    return {
        "study": "next_week_review_selection",
        "protocol_version": "0.5",
        "status": "retrospective_pre_registered_replay",
        "pool_root": str(pool_root),
        "price_cache": str(price_cache),
        "price_cache_sha256": content_hash(price_cache),
        "eps_pit": str(eps_path),
        "eps_pit_sha256": content_hash(eps_path),
        "active_signal_weeks": len(weeks),
        "first_snapshot": weeks[0] if weeks else "",
        "last_snapshot": weeks[-1] if weeks else "",
        "horizons": HORIZONS,
        "primary_rule": rule_to_dict(primary_rule()),
        "evidence_families": EVIDENCE_FAMILIES,
        "search": {
            "stage_1_core_rule_count": core_rule_count,
            "stage_2_full_sample_unique_evidence_rule_count": evidence_rule_count,
            "selection_signature_dedupe": True,
            "train_stability_is_hard_gate": False,
            "pareto_weighted_score": False,
        },
        "walk_forward": {
            "min_train_weeks": min_train_weeks,
            "test_weeks": test_weeks,
            "formal_full_block_only": True,
            "formal_oos_fold_count": formal_fold_count,
            "tail_exploratory_week_count": tail_week_count,
            "horizon_aware_asof_censoring": True,
            "test_used_for_selection": False,
        },
        "adaptive_policy_status": adaptive_status,
        "static_rule_status": static_status,
        "static_rule": (
            rule_to_dict(static_rule) if static_rule is not None else None
        ),
        "setup_balanced_sensitivity": True,
        "c_rank_used": False,
        "atr_used": False,
        "extended_in_core_selector": False,
    }


def _dominant_convergence(convergence: pd.DataFrame) -> dict[str, object]:
    if convergence.empty:
        return {}
    out: dict[str, object] = {}
    for level in ("EXACT_RULE", "STRUCTURE", "EVIDENCE_PROFILE"):
        rows = convergence[convergence["level"].eq(level)]
        if rows.empty:
            continue
        best = rows.sort_values(
            ["fold_count", "value"], ascending=[False, True]
        ).iloc[0]
        out[level.lower()] = {
            "value": str(best["value"]),
            "fold_count": int(best["fold_count"]),
            "fold_share": float(best["fold_share"]),
        }
    return out


def _median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if len(values) else np.nan


if __name__ == "__main__":
    main()
