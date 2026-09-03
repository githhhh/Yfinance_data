from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .discriminative import (
    B0_NAME,
    FORMAL_HORIZONS,
    MAX_ATTENTION_MULTIPLIER,
    oos_policy_status,
    rule_to_dict,
    run_discriminative_walk_forward,
    select_refined_all_weeks,
    summarize_oos_policy,
)
from .labels import add_forward_labels
from .metrics import (
    included_big_losers,
    missed_big_winners,
    moving_block_bootstrap_delta,
    weekly_macro_table,
)
from .oracle import add_weekly_oracle_flags
from .run import build_weekend_event_panel
from .sensitivity import setup_balanced_sensitivity
from .utils import (
    content_hash,
    load_pools,
    load_price_cache,
    normalize_eps_pit,
)


POOL_ROOT = Path("backtest/ibd_skill_replay_pools")
PRICE_CACHE = Path("results_pkl/stock_data_290826_1d.pkl")
OUTPUT_DIR = Path("backtest/next_week_review_selection/output_v06")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic v0.6 discriminative study."
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

    study = run_discriminative_walk_forward(
        panel,
        min_train_weeks=min_train_weeks,
        test_weeks=test_weeks,
    )
    static_rule = study["static_rule"]
    fold_results = study["fold_results"]

    static_summary = summarize_oos_policy(
        fold_results, "STATIC_DISCOVERY_RULE"
    )
    adaptive_summary = summarize_oos_policy(
        fold_results, "ADAPTIVE_DISCRIMINATIVE_POLICY"
    )
    static_status = oos_policy_status(
        static_summary,
        static_rule_exists=static_rule is not None,
    )
    adaptive_status = oos_policy_status(
        adaptive_summary,
        static_rule_exists=True,
    )

    formal_selections = study["formal_selections"]
    formal_weeks = sorted(
        set(
            fold_results["test_start"].astype(str).tolist()
            + fold_results["test_end"].astype(str).tolist()
        )
    )
    actual_formal_snapshots = sorted(
        formal_selections["snapshot_date"].astype(str).unique()
    ) if not formal_selections.empty else []
    formal_panel = panel[
        panel["snapshot_date"].astype(str).isin(actual_formal_snapshots)
    ].copy()

    if static_rule is not None and not formal_panel.empty:
        static_selected = select_refined_all_weeks(formal_panel, static_rule)
        b0_selected = select_refined_all_weeks(formal_panel, None)

        static_weekly = weekly_macro_table(
            formal_panel,
            static_selected,
            variant=static_rule.name,
        )
        b0_weekly = weekly_macro_table(
            formal_panel,
            b0_selected,
            variant=B0_NAME,
        )
        bootstrap_metrics = ["opportunity_recall_1w", "avg_watchlist_size"]
        for horizon in FORMAL_HORIZONS:
            bootstrap_metrics.extend(
                [
                    f"tradable_winner_capture_lift_{horizon}",
                    f"tradable_loser_capture_lift_{horizon}",
                    f"opp_severe_loser_exposure_{horizon}",
                ]
            )
        static_bootstrap = moving_block_bootstrap_delta(
            b0_weekly,
            static_weekly,
            metrics=bootstrap_metrics,
        )
        setup_detail, setup_summary = setup_balanced_sensitivity(
            formal_panel,
            b0_selected,
            static_selected,
            variant="V06_STATIC_DISCOVERY_RULE",
        )
        missed = missed_big_winners(formal_panel, static_selected)
        included = included_big_losers(formal_panel, static_selected)
        if not missed.empty:
            missed.insert(0, "audit_rule", "V06_STATIC_DISCOVERY_RULE")
        if not included.empty:
            included.insert(0, "audit_rule", "V06_STATIC_DISCOVERY_RULE")
    else:
        static_weekly = pd.DataFrame()
        b0_weekly = pd.DataFrame()
        static_bootstrap = pd.DataFrame()
        setup_detail = pd.DataFrame()
        setup_summary = pd.DataFrame()
        missed = pd.DataFrame()
        included = pd.DataFrame()

    manifest = {
        "study": "next_week_review_selection_v06_discriminative",
        "status": "retrospective_reused_history_confirmation",
        "historical_oos_reused_from_prior_iterations": True,
        "production_authorization": False,
        "pool_root": str(pool_root),
        "price_cache": str(price_cache),
        "price_cache_sha256": content_hash(price_cache),
        "eps_pit": str(eps_path),
        "eps_pit_sha256": content_hash(eps_path),
        "min_train_weeks": min_train_weeks,
        "test_weeks": test_weeks,
        "max_attention_multiplier": MAX_ATTENTION_MULTIPLIER,
        "candidate_library_size": 25,
        "static_rule": rule_to_dict(static_rule),
        "static_status": static_status,
        "adaptive_status": adaptive_status,
        "formal_fold_count": int(fold_results["fold"].nunique()),
        "formal_snapshot_count": len(actual_formal_snapshots),
        "formal_span_endpoints": formal_weeks,
        "rd_agent_used": False,
        "ml_used": False,
        "c_rank_used": False,
        "atr_used": False,
    }

    report = _render_report(
        static_rule=static_rule,
        static_status=static_status,
        adaptive_status=adaptive_status,
        static_summary=static_summary,
        adaptive_summary=adaptive_summary,
        discovery_grid=study["discovery_grid"],
        choices=study["fold_choices"],
        convergence=study["adaptive_convergence"],
        setup_summary=setup_summary,
        bootstrap=static_bootstrap,
    )

    outputs = {
        "discovery_bucket_stats.csv": output_dir / "discovery_bucket_stats.csv",
        "discovery_interactions.csv": output_dir / "discovery_interactions.csv",
        "discovery_candidate_grid.csv": output_dir / "discovery_candidate_grid.csv",
        "train_candidate_grids.csv": output_dir / "train_candidate_grids.csv",
        "fold_choices.csv": output_dir / "fold_choices.csv",
        "fold_results.csv": output_dir / "fold_results.csv",
        "static_oos_summary.csv": output_dir / "static_oos_summary.csv",
        "adaptive_oos_summary.csv": output_dir / "adaptive_oos_summary.csv",
        "adaptive_convergence.csv": output_dir / "adaptive_convergence.csv",
        "formal_selections.csv": output_dir / "formal_selections.csv",
        "static_weekly_metrics.csv": output_dir / "static_weekly_metrics.csv",
        "b0_weekly_metrics.csv": output_dir / "b0_weekly_metrics.csv",
        "static_oos_bootstrap.csv": output_dir / "static_oos_bootstrap.csv",
        "static_setup_sensitivity.csv": output_dir / "static_setup_sensitivity.csv",
        "static_setup_summary.csv": output_dir / "static_setup_summary.csv",
        "static_missed_big_winners.csv": output_dir / "static_missed_big_winners.csv",
        "static_included_big_losers.csv": output_dir / "static_included_big_losers.csv",
        "tail_exploratory.csv": output_dir / "tail_exploratory.csv",
        "decision.json": output_dir / "decision.json",
        "manifest.json": output_dir / "manifest.json",
        "report.md": output_dir / "report.md",
    }

    study["discovery_buckets"].to_csv(
        outputs["discovery_bucket_stats.csv"], index=False
    )
    study["discovery_interactions"].to_csv(
        outputs["discovery_interactions.csv"], index=False
    )
    study["discovery_grid"].to_csv(
        outputs["discovery_candidate_grid.csv"], index=False
    )
    study["train_candidate_grids"].to_csv(
        outputs["train_candidate_grids.csv"], index=False
    )
    study["fold_choices"].to_csv(outputs["fold_choices.csv"], index=False)
    fold_results.to_csv(outputs["fold_results.csv"], index=False)
    static_summary.to_csv(outputs["static_oos_summary.csv"], index=False)
    adaptive_summary.to_csv(outputs["adaptive_oos_summary.csv"], index=False)
    study["adaptive_convergence"].to_csv(
        outputs["adaptive_convergence.csv"], index=False
    )
    formal_selections.to_csv(outputs["formal_selections.csv"], index=False)
    static_weekly.to_csv(outputs["static_weekly_metrics.csv"], index=False)
    b0_weekly.to_csv(outputs["b0_weekly_metrics.csv"], index=False)
    static_bootstrap.to_csv(outputs["static_oos_bootstrap.csv"], index=False)
    setup_detail.to_csv(outputs["static_setup_sensitivity.csv"], index=False)
    setup_summary.to_csv(outputs["static_setup_summary.csv"], index=False)
    missed.to_csv(outputs["static_missed_big_winners.csv"], index=False)
    included.to_csv(outputs["static_included_big_losers.csv"], index=False)
    study["tail_exploratory"].to_csv(
        outputs["tail_exploratory.csv"], index=False
    )

    outputs["decision.json"].write_text(
        json.dumps(
            {
                "static_status": static_status,
                "static_rule": rule_to_dict(static_rule),
                "adaptive_status": adaptive_status,
                "note": (
                    "retrospective reused-history confirmation only; "
                    "future sealed weeks are required for production authorization"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    outputs["manifest.json"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["report.md"].write_text(report, encoding="utf-8")
    return {name: str(path) for name, path in outputs.items()}


def _render_report(
    *,
    static_rule,
    static_status: str,
    adaptive_status: str,
    static_summary: pd.DataFrame,
    adaptive_summary: pd.DataFrame,
    discovery_grid: pd.DataFrame,
    choices: pd.DataFrame,
    convergence: pd.DataFrame,
    setup_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> str:
    selected = (
        discovery_grid.loc[discovery_grid["feasible"].eq(True)]
        .sort_values(
            [
                "pareto",
                "horizon_consistency_count",
                "tradable_winner_capture_lift_mean_2_4w_delta_vs_b0",
            ],
            ascending=[False, False, False],
            kind="mergesort",
        )
        .head(10)
    ) if not discovery_grid.empty else pd.DataFrame()

    return "\n".join(
        [
            "# Next Week Review Selection v0.6 — Deterministic Discriminative Study",
            "",
            "## Decision",
            f"- static status: {static_status}",
            f"- static rule: {static_rule.name if static_rule is not None else B0_NAME}",
            f"- adaptive status: {adaptive_status}",
            "- rd-agent: not used",
            "- production authorization: NO",
            "",
            "## Methodological warning",
            "The historical formal OOS weeks have already been observed in prior v0.5 research.",
            "v0.6 is therefore a disciplined retrospective confirmation, not a new sealed holdout.",
            "Any candidate still requires future sealed weeks before production use.",
            "",
            "## Static discovery rule — formal replay",
            _markdown(static_summary),
            "",
            "## Adaptive discriminative policy — formal replay",
            _markdown(adaptive_summary),
            "",
            "## Discovery feasible/Pareto candidates",
            _markdown(selected),
            "",
            "## Fold choices",
            _markdown(choices),
            "",
            "## Adaptive rule convergence",
            _markdown(convergence),
            "",
            "## Setup-balanced static sensitivity",
            _markdown(setup_summary),
            "",
            "## Static-rule moving-block bootstrap",
            _markdown(bootstrap),
            "",
            "## Guardrails",
            f"- attention multiplier cap: <= {MAX_ATTENTION_MULTIPLIER:.2f}x B0",
            "- fixed anchor: Near5 + UNCONFIRMED/BELOW_TRIGGER + >=2 evidence families + Geometry allow",
            "- candidate refinements use at most two coarse, interpretable PIT conditions",
            "- first 20 weeks choose one static discovery rule; it remains frozen across all formal replay folds",
            "- expanding-train adaptive refinement is secondary",
            "- no ML, no rd-agent, no C Rank, no ATR, no arbitrary decimal threshold search",
            "",
        ]
    )


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No data_"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_csv(index=False)


if __name__ == "__main__":
    main()
