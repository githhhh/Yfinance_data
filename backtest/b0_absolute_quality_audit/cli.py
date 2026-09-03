from __future__ import annotations

import argparse
import json

import pandas as pd

from .audit import materialize_core, summarize_core
from .config import (
    OUT,
    PROTOCOL_VERSION,
    RAW_MC_DRAWS,
    RAW_PRICE_COVERAGE_MIN_FOR_PRIMARY,
    ELIGIBLE_ENTRY_COVERAGE_MIN_FOR_PRIMARY,
)
from .data import sha256_file
from .report import write_report


CURRENT_STATE = OUT / "current_b0_state.csv"
B0_WEEKLY = OUT / "b0_weekly_quality.csv"
ELIGIBLE_RANDOM = OUT / "eligible_random_weekly.csv"
RAW_RANDOM = OUT / "raw_random_weekly.csv"
ELIGIBILITY = OUT / "eligibility_gate_weekly.csv"
RANKING = OUT / "ranking_weekly.csv"
RANK_BUCKET_ROWS = OUT / "rank_bucket_rows.csv"
RANK_BUCKET_SUMMARY = OUT / "rank_bucket_summary.csv"
SIMPLE_WEEKLY = OUT / "simple_baseline_weekly.csv"
SIMPLE_SUMMARY = OUT / "simple_baseline_summary.csv"
REJECTION_ROWS = OUT / "rejection_reason_rows.csv"
REJECTION_SUMMARY = OUT / "rejection_reason_summary.csv"
NONOVERLAP = OUT / "nonoverlap_offsets.csv"
HEALTH = OUT / "b0_health_summary.json"
MANIFEST = OUT / "run_manifest.json"
REPORT = OUT / "B0_ABSOLUTE_QUALITY_HEALTH_CHECK.md"


def _write_csv(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def materialize() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    core = materialize_core()
    summary = summarize_core(core)
    frames = core["frames"]
    manifest = dict(core["manifest"])

    panel = frames["panel"]
    state_cols = [
        "snapshot_date",
        "code",
        "current_b0_raw_rank",
        "current_b0_lane",
        "current_b0_eligible",
        "current_b0_reject_reasons",
        "current_b0_selected",
        "current_b0_pick_order",
        "w4_return_pct",
        "w4_stop8",
        "snapshot_w4_return_pct",
        "snapshot_w4_stop8",
        "snapshot_price_valid",
    ]
    _write_csv(panel[[c for c in state_cols if c in panel.columns]], CURRENT_STATE)
    _write_csv(frames["b0_weekly"], B0_WEEKLY)
    _write_csv(frames["eligible_random_weekly"], ELIGIBLE_RANDOM)
    _write_csv(frames["raw_random_weekly"], RAW_RANDOM)
    _write_csv(frames["eligibility_weekly"], ELIGIBILITY)
    _write_csv(frames["ranking_weekly"], RANKING)
    _write_csv(frames["rank_bucket_rows"], RANK_BUCKET_ROWS)
    _write_csv(summary["rank_bucket_summary"], RANK_BUCKET_SUMMARY)
    _write_csv(frames["simple_baseline_weekly"], SIMPLE_WEEKLY)
    _write_csv(summary["simple_baseline_summary"], SIMPLE_SUMMARY)
    _write_csv(frames["rejection_reason_rows"], REJECTION_ROWS)
    _write_csv(summary["rejection_reason_summary"], REJECTION_SUMMARY)
    _write_csv(summary["nonoverlap_offsets"], NONOVERLAP)

    HEALTH.write_text(
        json.dumps(summary["health"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    old_eligible_mismatch = None
    if "b0_eligible" in panel.columns:
        old = panel["b0_eligible"].fillna(False).astype(bool)
        old_eligible_mismatch = int((old != panel["current_b0_eligible"]).sum())

    old_selected_mismatch = None
    if "is_b0" in panel.columns:
        old = pd.to_numeric(panel["is_b0"], errors="coerce").fillna(0).astype(int).eq(1)
        old_selected_mismatch = int((old != panel["current_b0_selected"]).sum())

    manifest.update({
        "protocol_version": PROTOCOL_VERSION,
        "raw_random_mc_draws_when_not_exact": RAW_MC_DRAWS,
        "raw_price_coverage_min_for_primary": RAW_PRICE_COVERAGE_MIN_FOR_PRIMARY,
        "eligible_entry_coverage_min_for_primary": ELIGIBLE_ENTRY_COVERAGE_MIN_FOR_PRIMARY,
        "old_panel_b0_eligible_mismatch_rows": old_eligible_mismatch,
        "old_panel_is_b0_mismatch_rows": old_selected_mismatch,
        "outcome_semantics": {
            "entry_aligned_w4": (
                "Frozen candidate W4 return from Production-style entry; used only "
                "inside current B0-eligible universe."
            ),
            "snapshot_close_w4": (
                "Frozen close at/before snapshot to close at/before snapshot+28 calendar days; "
                "used for raw signal universe, gate opportunity-cost, simple rules and market benchmarks."
            ),
        },
        "raw_benchmark_definition": (
            "All frozen Review Universe rows (signal=True and non-empty ibd_candidate_rule); "
            "never conditioned on b0_eligible, B0 Lane, B0 rank or B0 reason/risk codes."
        ),
        "current_b0_definition": (
            "Recomputed directly from dashboard.skill_industry_eps_known.py on each frozen snapshot."
        ),
        "evidence_boundary": summary["health"]["evidence_boundary"],
    })

    write_report(
        REPORT,
        summary["health"],
        summary["nonoverlap_offsets"],
        summary["rank_bucket_summary"],
        summary["simple_baseline_summary"],
        summary["rejection_reason_summary"],
        manifest,
    )

    artifacts = [
        CURRENT_STATE,
        B0_WEEKLY,
        ELIGIBLE_RANDOM,
        RAW_RANDOM,
        ELIGIBILITY,
        RANKING,
        RANK_BUCKET_ROWS,
        RANK_BUCKET_SUMMARY,
        SIMPLE_WEEKLY,
        SIMPLE_SUMMARY,
        REJECTION_ROWS,
        REJECTION_SUMMARY,
        NONOVERLAP,
        HEALTH,
        REPORT,
    ]
    manifest["artifacts"] = {
        p.name: sha256_file(p)
        for p in artifacts
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    h = summary["health"]
    print("=== Current Production B0 Absolute Quality Audit ===")
    print(f"source={manifest['source_git_sha']}")
    print(f"snapshots={manifest['snapshot_count']} review_rows={manifest['review_rows']}")
    print(
        "absolute entry W4: "
        f"mean={h['absolute']['entry_aligned_w4']['mean']}, "
        f"median={h['absolute']['entry_aligned_w4']['median']}, "
        f"cvar10={h['absolute']['entry_aligned_w4']['cvar10']}"
    )
    print(
        "eligible random: "
        f"support={h['eligible_random']['support_weeks']}, "
        f"median_percentile={h['eligible_random']['median_weekly_percentile']}, "
        f"oracle_capture={h['eligible_random']['oracle_capture'].get('aggregate_capture_ratio')}"
    )
    print(
        "raw random: "
        f"support={h['raw_random_distinct1']['support_weeks']}, "
        f"median_percentile={h['raw_random_distinct1']['median_weekly_percentile']}, "
        f"oracle_capture={h['raw_random_distinct1']['oracle_capture'].get('aggregate_capture_ratio')}"
    )
    print(
        "gate: "
        f"winner_retention={h['eligibility']['winner_retention_rate']}, "
        f"rejected_winner_rate={h['eligibility']['rejected_winner_rate']}, "
        f"gate_lift={h['eligibility']['mean_gate_lift']}"
    )
    print(
        "ranking: "
        f"spearman_median={h['ranking']['weekly_spearman_median']}, "
        f"selected_minus_eligible={h['ranking']['b0_minus_eligible_mean_mean']}"
    )
    print("summary_policy=NO_ARBITRARY_PASS_FAIL_THRESHOLD")
    print(f"report={REPORT}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Current Production B0 absolute quality health check"
    )
    parser.add_argument("command", choices=["materialize"])
    args = parser.parse_args()
    if args.command == "materialize":
        materialize()


if __name__ == "__main__":
    main()
