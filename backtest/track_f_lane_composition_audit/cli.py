from __future__ import annotations

import argparse
import json

from .config import OUT
from .experiment import (
    build_manifest,
    build_segments,
    build_taxonomy_rows,
    evaluate_policies,
    historical_support_decision,
    load_panel,
    parity_anchor_audit,
    route_bucket_weekly,
    route_pair_summary,
    selection_composition,
    sha256_file,
    taxonomy_matrix,
    taxonomy_summary,
)
from .report import write_report


EVAL_CSV = OUT / "policy_evaluations.csv"
EVAL_JSON = OUT / "policy_evaluations.json"
TAXONOMY_ROWS = OUT / "lane_taxonomy_rows.csv"
TAXONOMY_MATRIX = OUT / "lane_taxonomy_matrix.csv"
TAXONOMY_SUMMARY = OUT / "lane_taxonomy_summary.json"
SELECTIONS = OUT / "weekly_selection_composition.csv"
ROUTE_WEEKLY = OUT / "route_bucket_weekly.csv"
ROUTE_PAIR = OUT / "route_pair_summary.json"
PARITY_ANCHOR = OUT / "parity_anchor.json"
DECISION = OUT / "track_f_decision.json"
MANIFEST = OUT / "run_manifest.json"
REPORT = OUT / "TRACK_F_LANE_COMPOSITION_REPORT.md"


def materialize() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    segments, _ = build_segments(panel)
    taxonomy_rows = build_taxonomy_rows(panel)
    tax_matrix = taxonomy_matrix(taxonomy_rows)
    tax_summary = taxonomy_summary(taxonomy_rows)
    evaluation = evaluate_policies(panel, segments)
    selections = selection_composition(panel)
    route_weekly = route_bucket_weekly(panel, taxonomy_rows)
    route_pair = route_pair_summary(route_weekly)
    parity_anchor = parity_anchor_audit(panel)
    decision = historical_support_decision(evaluation)
    manifest = build_manifest(panel)

    if parity_anchor["mismatch_count"] != 0:
        raise RuntimeError(
            "Track F parity anchor failed: normalized CONFIRMED_PARITY_FALLBACK "
            f"does not reproduce Track C PULLBACK_PARITY on {parity_anchor['mismatch_count']} snapshots"
        )

    evaluation.to_csv(EVAL_CSV, index=False)
    EVAL_JSON.write_text(evaluation.to_json(orient="records", indent=2), encoding="utf-8")
    taxonomy_rows.to_csv(TAXONOMY_ROWS, index=False)
    tax_matrix.to_csv(TAXONOMY_MATRIX, index=False)
    TAXONOMY_SUMMARY.write_text(
        json.dumps(tax_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    selections.to_csv(SELECTIONS, index=False)
    route_weekly.to_csv(ROUTE_WEEKLY, index=False)
    ROUTE_PAIR.write_text(
        json.dumps(route_pair, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    PARITY_ANCHOR.write_text(
        json.dumps(parity_anchor, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    DECISION.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_report(
        REPORT,
        evaluation,
        tax_summary,
        route_pair,
        parity_anchor,
        decision,
        manifest,
    )

    manifest["artifacts"] = {
        "policy_evaluations_csv": sha256_file(EVAL_CSV),
        "policy_evaluations_json": sha256_file(EVAL_JSON),
        "lane_taxonomy_rows": sha256_file(TAXONOMY_ROWS),
        "lane_taxonomy_matrix": sha256_file(TAXONOMY_MATRIX),
        "lane_taxonomy_summary": sha256_file(TAXONOMY_SUMMARY),
        "weekly_selection_composition": sha256_file(SELECTIONS),
        "route_bucket_weekly": sha256_file(ROUTE_WEEKLY),
        "route_pair_summary": sha256_file(ROUTE_PAIR),
        "parity_anchor": sha256_file(PARITY_ANCHOR),
        "decision": sha256_file(DECISION),
        "report": sha256_file(REPORT),
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    primary = evaluation[
        (evaluation["role"] == "primary")
        & (evaluation["segment"] == "retrospective_track_d_40")
    ]
    print("=== Track F Lane taxonomy/composition audit complete ===")
    print(f"source={manifest['source_git_sha']}")
    print(
        "taxonomy: "
        f"constructive={tax_summary['constructive_pullback_rows']}, "
        f"eligible standard pullback={tax_summary['eligible_standard_pullback_rows']}, "
        f"eligible standard non-pullback={tax_summary['eligible_standard_non_pullback_rows']}"
    )
    print(
        "route pair: "
        f"support={route_pair['support_weeks']}, "
        f"mean={route_pair['mean_pullback_minus_non_pullback_w4']}, "
        f"median={route_pair['median_pullback_minus_non_pullback_w4']}"
    )
    for _, row in primary.iterrows():
        print(
            f"{row['policy_id']}: mean={row['mean_spread']}, "
            f"median={row['median_spread']}, stop={row['stop_delta_pct']}, "
            f"ruin={row['one_pick_ruins_delta_pct']}, coverage={row['slot_coverage_pct']}"
        )
    print(f"decision={decision['overall']} candidates={decision['shadow_candidates']}")
    print(f"report={REPORT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Track F Lane taxonomy/composition audit")
    parser.add_argument("command", choices=["materialize"])
    args = parser.parse_args()
    if args.command == "materialize":
        materialize()


if __name__ == "__main__":
    main()
