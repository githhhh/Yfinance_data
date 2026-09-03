from __future__ import annotations

import argparse
import json

import pandas as pd

from .config import OUT, PROTOCOL_VERSION, YAHOO_DOWNLOAD_AUDIT_CSV, YAHOO_SUPPLEMENT_PARQUET
from .data import sha256_file
from .report_v11 import write_v11_report
from .v11 import materialize_v11_core, summarize_v11


CURRENT_STATE = OUT / "current_b0_state.csv"
B0_WEEKLY = OUT / "b0_weekly_quality.csv"
ELIGIBLE_RANDOM = OUT / "eligible_random_weekly.csv"
RAW_RANDOM = OUT / "raw_random_weekly.csv"
GATE_WEEKLY = OUT / "eligibility_gate_weekly.csv"
RANKING = OUT / "ranking_weekly.csv"
RANK_BUCKET_ROWS = OUT / "rank_bucket_rows.csv"
RANK_BUCKET_SUMMARY = OUT / "rank_bucket_summary.csv"
REJECTION_EVENTS = OUT / "rejection_events.csv"
EXCLUSIVE_REJECTION = OUT / "exclusive_rejection_summary.csv"
OVERLAP_REJECTION = OUT / "overlap_rejection_summary.csv"
REJECTION_COMBOS = OUT / "rejection_combinations.csv"
SIMPLE_WEEKLY = OUT / "simple_baseline_weekly.csv"
SIMPLE_SUMMARY = OUT / "simple_baseline_summary.csv"
CAPACITY_WEEKLY = OUT / "capacity_policy_weekly.csv"
CAPACITY_SUMMARY = OUT / "capacity_policy_summary.csv"
UNDERFILL_CAUSES = OUT / "underfill_cause_summary.csv"
MARKET_SUMMARY = OUT / "market_benchmark_summary.csv"
NONOVERLAP = OUT / "nonoverlap_offsets.csv"
HEALTH = OUT / "b0_health_summary.json"
MANIFEST = OUT / "run_manifest.json"
REPORT = OUT / "B0_ABSOLUTE_QUALITY_HEALTH_CHECK.md"


def _write_csv(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _clear_stale_output() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in OUT.iterdir():
        if path.is_file():
            path.unlink()


def materialize() -> None:
    _clear_stale_output()

    core = materialize_v11_core()
    summary = summarize_v11(core)
    frames = core["frames"]
    manifest = dict(core["manifest"])

    panel = frames["panel"]
    state_cols = [
        "snapshot_date",
        "code",
        "industry",
        "current_b0_raw_rank",
        "current_b0_lane",
        "current_b0_eligible",
        "current_b0_reject_reasons",
        "current_b0_selected",
        "current_b0_pick_order",
        "w4_return_pct",
        "w4_stop8",
        "next_open_w4_return_pct",
        "next_open_w4_stop8",
        "next_open_entry_date",
        "next_open_end_date",
        "next_open_price_valid",
        "next_open_invalid_reason",
    ]

    _write_csv(panel[[c for c in state_cols if c in panel.columns]], CURRENT_STATE)
    _write_csv(frames["b0_weekly"], B0_WEEKLY)
    _write_csv(frames["eligible_random_weekly"], ELIGIBLE_RANDOM)
    _write_csv(frames["raw_random_weekly"], RAW_RANDOM)
    _write_csv(frames["gate_weekly"], GATE_WEEKLY)
    _write_csv(frames["ranking_weekly"], RANKING)
    _write_csv(frames["rank_bucket_rows"], RANK_BUCKET_ROWS)
    _write_csv(summary["rank_bucket_summary"], RANK_BUCKET_SUMMARY)
    _write_csv(frames["rejection_events"], REJECTION_EVENTS)
    _write_csv(summary["exclusive_rejection_summary"], EXCLUSIVE_REJECTION)
    _write_csv(summary["overlap_rejection_summary"], OVERLAP_REJECTION)
    _write_csv(summary["rejection_combinations"], REJECTION_COMBOS)
    _write_csv(frames["simple_weekly"], SIMPLE_WEEKLY)
    _write_csv(summary["simple_summary"], SIMPLE_SUMMARY)
    _write_csv(frames["capacity_weekly"], CAPACITY_WEEKLY)
    _write_csv(summary["capacity_summary"], CAPACITY_SUMMARY)
    _write_csv(summary["underfill_cause_summary"], UNDERFILL_CAUSES)
    _write_csv(summary["market_summary"], MARKET_SUMMARY)
    _write_csv(summary["nonoverlap"], NONOVERLAP)

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
        "old_panel_b0_eligible_mismatch_rows": old_eligible_mismatch,
        "old_panel_is_b0_mismatch_rows": old_selected_mismatch,
        "ranking_headline_rule": (
            "Eligible random percentile headline uses only active-choice weeks "
            "with >1 feasible portfolio."
        ),
        "capacity_headline_rule": (
            "B0 original remains Production reference. Fill3 policies preserve all "
            "original picks and only fill unused slots."
        ),
        "reject_reason_rule": (
            "Exclusive single-reason summary isolates gate-specific candidates but is "
            "not causal proof; overlap summary is descriptive only."
        ),
        "simple_baseline_rule": (
            "Primary comparison requires full feature capacity and tradable next-open W4."
        ),
        "evidence_boundary": summary["health"]["evidence_boundary"],
    })

    write_v11_report(
        REPORT,
        health=summary["health"],
        raw_by_pick=summary["raw_by_pick_count"],
        capacity=summary["capacity_summary"],
        underfill_causes=summary["underfill_cause_summary"],
        exclusive_reject=summary["exclusive_rejection_summary"],
        overlap_reject=summary["overlap_rejection_summary"],
        reject_combos=summary["rejection_combinations"],
        rank_buckets=summary["rank_bucket_summary"],
        simple=summary["simple_summary"],
        market=summary["market_summary"],
        nonoverlap=summary["nonoverlap"],
        manifest=manifest,
    )

    artifacts = [
        CURRENT_STATE,
        B0_WEEKLY,
        ELIGIBLE_RANDOM,
        RAW_RANDOM,
        GATE_WEEKLY,
        RANKING,
        RANK_BUCKET_ROWS,
        RANK_BUCKET_SUMMARY,
        REJECTION_EVENTS,
        EXCLUSIVE_REJECTION,
        OVERLAP_REJECTION,
        REJECTION_COMBOS,
        SIMPLE_WEEKLY,
        SIMPLE_SUMMARY,
        CAPACITY_WEEKLY,
        CAPACITY_SUMMARY,
        UNDERFILL_CAUSES,
        MARKET_SUMMARY,
        NONOVERLAP,
        HEALTH,
        REPORT,
        YAHOO_SUPPLEMENT_PARQUET,
        YAHOO_DOWNLOAD_AUDIT_CSV,
    ]
    manifest["artifacts"] = {
        p.name: sha256_file(p)
        for p in artifacts
        if p.exists()
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    h = summary["health"]
    print("=== Current Production B0 Absolute Quality Audit v1.1 ===")
    print(f"source={manifest['source_git_sha']}")
    print(
        "ranking active-choice: "
        f"weeks={h['eligible_ranking_active_choice']['active_choice_weeks']}, "
        f"median_pct={h['eligible_ranking_active_choice']['median_percentile']}, "
        f"mean_edge={h['eligible_ranking_active_choice']['edge'].get('spread', {}).get('mean')}"
    )
    print(
        "raw next-open fixed3: "
        f"weeks={h['raw_fixed_capacity_next_open']['support_weeks']}, "
        f"median_pct={h['raw_fixed_capacity_next_open']['median_percentile']}, "
        f"mean_edge={h['raw_fixed_capacity_next_open']['edge'].get('spread', {}).get('mean')}"
    )
    print(
        "gate: "
        f"accept={h['gate']['accept_rate']}, "
        f"winner_enrichment={h['gate']['winner_enrichment_vs_random_selectivity']}, "
        f"loser_retention_relative={h['gate']['loser_retention_vs_random_selectivity']}"
    )
    print(f"capacity_summary={CAPACITY_SUMMARY}")
    print(f"market_summary={MARKET_SUMMARY}")
    print(f"report={REPORT}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Current Production B0 absolute quality health check v1.1"
    )
    parser.add_argument("command", choices=["materialize"])
    args = parser.parse_args()
    if args.command == "materialize":
        materialize()


if __name__ == "__main__":
    main()
