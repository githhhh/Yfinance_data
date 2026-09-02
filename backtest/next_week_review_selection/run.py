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
    missed_big_winners,
)
from .optimizer import evaluate_rule_grid, select_all_weeks
from .oracle import add_weekly_oracle_flags, oracle_projection
from .report import render_report
from .search_space import generate_candidate_rules
from .selectors import (
    SUPPORT_KEYS,
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

    core_rule = primary_rule()
    b0_selected = select_all_weeks(evaluation_panel, None)
    primary_selected = select_all_weeks(evaluation_panel, core_rule)
    b0_metrics = evaluate_selection(
        evaluation_panel, b0_selected, variant="B0_ACTIONABLE_ONLY"
    )
    primary_metrics = compare_metrics(
        evaluate_selection(
            evaluation_panel, primary_selected, variant=core_rule.name
        ),
        b0_metrics,
    )
    baseline_vs_primary = pd.DataFrame([b0_metrics, primary_metrics])

    rules = generate_candidate_rules()
    full_rule_grid = evaluate_rule_grid(evaluation_panel, rules)
    fold_results, wf_champions, train_grid = run_walk_forward(
        evaluation_panel,
        rules,
        min_train_weeks=min_train_weeks,
        test_weeks=test_weeks,
    )
    oos_stability = summarize_oos_stability(fold_results)
    champion = choose_retrospective_champion(oos_stability, rules)

    champion_status = (
        "RETROSPECTIVE_CANDIDATE"
        if champion is not None
        else "NO_STABLE_NEXT_WEEK_REVIEW_RULE"
    )
    champion_name = champion.name if champion is not None else ""

    audit_rule = champion if champion is not None else core_rule
    audit_selected = select_all_weeks(evaluation_panel, audit_rule)
    missed_winners = missed_big_winners(evaluation_panel, audit_selected)
    included_losers = included_big_losers(evaluation_panel, audit_selected)
    if not missed_winners.empty:
        missed_winners.insert(0, "audit_rule", audit_rule.name)
    if not included_losers.empty:
        included_losers.insert(0, "audit_rule", audit_rule.name)

    latest_review = latest_review_list(panel, core_rule)
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
        rule_count=len(rules),
        min_train_weeks=min_train_weeks,
        test_weeks=test_weeks,
        champion_status=champion_status,
        champion=champion,
    )
    data_audit = render_data_audit(panel, evaluation_panel, pools, eps)
    report = render_report(
        baseline_vs_primary=baseline_vs_primary,
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
        "full_rule_grid.csv": output_dir / "full_rule_grid.csv",
        "fold_level_results.csv": output_dir / "fold_level_results.csv",
        "walk_forward_champions.csv": output_dir / "walk_forward_champions.csv",
        "walk_forward_train_grid.csv": output_dir / "walk_forward_train_grid.csv",
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
    full_rule_grid.to_csv(outputs["full_rule_grid.csv"], index=False)
    fold_results.to_csv(outputs["fold_level_results.csv"], index=False)
    wf_champions.to_csv(outputs["walk_forward_champions.csv"], index=False)
    train_grid.to_csv(outputs["walk_forward_train_grid.csv"], index=False)
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
    min_complete_coverage: float = 0.80,
) -> pd.DataFrame:
    """Use only snapshot weeks with broad 4W outcome coverage for optimization."""
    if panel.empty:
        return panel.copy()
    coverage = panel.groupby("snapshot_date")["forward_4w_censored"].apply(
        lambda values: float(values.eq(False).mean())
    )
    mature_weeks = coverage[coverage >= min_complete_coverage].index.astype(str)
    return panel[
        panel["snapshot_date"].astype(str).isin(mature_weeks)
    ].copy()


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
        "_support_count",
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
                "median_return_1w_pct": _median(
                    complete, "forward_1w_return_pct"
                ),
                "median_return_4w_pct": _median(
                    group[group["forward_4w_censored"].eq(False)],
                    "forward_4w_return_pct",
                ),
                "median_mae_4w_pct": _median(
                    group[group["forward_4w_censored"].eq(False)],
                    "mae_4w_pct",
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
        lines.append(f"- complete {horizon} outcomes: {complete}/{len(panel)}")
    lines.extend(
        [
            "",
            "Guardrails:",
            "- Weekend selector reads only frozen pool fields + verified PIT EPS.",
            "- Forward returns/MFE/MAE/oracle flags are evaluation-only.",
            "- ACTIONABLE baseline rows are never re-filtered by research variants.",
            "- EXTENDED is excluded from the core selector.",
            "- False/missing positive evidence is neutral.",
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
    rule_count: int,
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
        "primary_rule": rule_to_dict(primary_rule()),
        "search_rule_count": rule_count,
        "support_keys": SUPPORT_KEYS,
        "walk_forward": {
            "min_train_weeks": min_train_weeks,
            "test_weeks": test_weeks,
            "expanding_window": True,
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
