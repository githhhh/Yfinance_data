from __future__ import annotations

import argparse
import json

from .config import OUT
from .experiment import (
    build_event_summary,
    build_manifest,
    build_segments,
    collect_selection_events,
    evaluate_segments,
    load_panel,
    sha256_file,
)
from .report import write_report


SUMMARY_CSV = OUT / "comparison_summary.csv"
SUMMARY_JSON = OUT / "comparison_summary.json"
EVENTS_CSV = OUT / "selection_events.csv"
EVENTS_JSON = OUT / "selection_events.json"
EVENT_SUMMARY = OUT / "event_summary.json"
MANIFEST = OUT / "run_manifest.json"
REPORT = OUT / "TRACK_E_SOFT_LANE_REPORT.md"


def materialize() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    segments, split = build_segments(panel)
    summary = evaluate_segments(panel, segments)
    events = collect_selection_events(panel, split)
    event_summary = build_event_summary(events)
    manifest = build_manifest(panel, split)

    summary.to_csv(SUMMARY_CSV, index=False)
    SUMMARY_JSON.write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")
    events.to_csv(EVENTS_CSV, index=False)
    EVENTS_JSON.write_text(events.to_json(orient="records", indent=2), encoding="utf-8")
    EVENT_SUMMARY.write_text(
        json.dumps(event_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_report(REPORT, summary, events, event_summary, manifest)

    manifest["artifacts"] = {
        "comparison_summary_csv": sha256_file(SUMMARY_CSV),
        "comparison_summary_json": sha256_file(SUMMARY_JSON),
        "selection_events_csv": sha256_file(EVENTS_CSV),
        "selection_events_json": sha256_file(EVENTS_JSON),
        "event_summary": sha256_file(EVENT_SUMMARY),
        "report": sha256_file(REPORT),
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    retro = summary[summary["segment"] == "retrospective_all_40"].iloc[0]
    print("=== Track E v2 pairwise Lane audit complete ===")
    print(f"source={manifest['source_git_sha']}")
    print(
        "retrospective_all_40: "
        f"support={int(retro['support_weeks'])}, "
        f"mean_spread={retro['mean_spread']}, "
        f"median_spread={retro['median_spread']}, "
        f"cvar_delta={retro['cvar_delta']}"
    )
    print(
        "fresh/standard rank crossovers: "
        f"{event_summary['target_rank_crossover_mature_weeks']} mature / "
        f"{event_summary['target_rank_crossover_weeks']} total; "
        "Top3 swaps: "
        f"{event_summary['target_selection_swap_mature_weeks']} mature / "
        f"{event_summary['target_selection_swap_weeks']} total"
    )
    print(f"report={REPORT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Track E v2 isolated fresh-vs-standard Lane audit")
    parser.add_argument("command", choices=["materialize"])
    args = parser.parse_args()
    if args.command == "materialize":
        materialize()


if __name__ == "__main__":
    main()
