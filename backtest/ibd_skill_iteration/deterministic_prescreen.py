from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.ibd_skill_iteration.core import (
    ReasonedCandidate,
    build_reasoning_skill_picks,
    rank_non_actionable_alpha_radar,
    rank_non_actionable_pullback_scout,
    rank_reasoning_candidates,
    rank_signal_shadow_top3,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render deterministic IBD prescreen artifact from a pool CSV.")
    parser.add_argument("--pool", required=True, help="Pool CSV path.")
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--version", default="v3", choices=["v1", "v2", "v3"])
    parser.add_argument("--json-out", help="Optional JSON output path.")
    parser.add_argument("--markdown-out", help="Optional Markdown output path.")
    args = parser.parse_args(argv)

    pool = pd.read_csv(args.pool, encoding="utf-8-sig")
    artifact = build_prescreen_artifact(pool, snapshot_date=args.snapshot_date, version=args.version)
    artifact_json = json.dumps(artifact, ensure_ascii=False, indent=2, default=_json_default)
    markdown = render_prescreen_artifact_markdown(artifact)

    if args.json_out:
        Path(args.json_out).write_text(artifact_json + "\n", encoding="utf-8")
    if args.markdown_out:
        Path(args.markdown_out).write_text(markdown + "\n", encoding="utf-8")
    if not args.json_out and not args.markdown_out:
        sys.stdout.write(artifact_json + "\n")


SORT_KEY_LABELS = {
    "v1": [
        "clear_failure",
        "lane_order",
        "status_bucket",
        "fresh_bucket",
        "negative_evidence_count",
        "negative_entry_volume",
        "eps_missing_or_below_25",
        "weekly_volume_missing_or_below_1_3",
        "code",
        "row_index",
    ],
    "v2": [
        "clear_failure",
        "lane_order",
        "status_bucket",
        "negative_net_evidence",
        "risk_count",
        "fresh_bucket",
        "eps_missing_or_below_25",
        "weekly_volume_missing_or_below_1_3",
        "negative_entry_volume",
        "code",
        "row_index",
    ],
    "v3": [
        "clear_failure",
        "lane_order",
        "status_bucket",
        "negative_net_evidence",
        "risk_count",
        "fresh_bucket",
        "eps_missing_or_below_25",
        "weekly_volume_missing_or_below_1_3",
        "negative_entry_volume",
        "code",
        "row_index",
    ],
}


def build_prescreen_artifact(
    pool: pd.DataFrame,
    *,
    snapshot_date: str,
    version: str = "v3",
) -> dict[str, Any]:
    picks = build_reasoning_skill_picks(pool, snapshot_date=snapshot_date, version=version)
    actionable_raw = rank_reasoning_candidates(pool, universe="actionable", version=version)
    non_actionable = rank_non_actionable_alpha_radar(pool, version=version)
    pullback_scout = rank_non_actionable_pullback_scout(pool, version=version) if version == "v3" else []
    signal_shadow = rank_signal_shadow_top3(pool, version=version)
    labels = SORT_KEY_LABELS.get(version, SORT_KEY_LABELS["v1"])

    return {
        "schema_version": "ibd-prescreen-artifact-v1",
        "snapshot_date": snapshot_date,
        "version": version,
        "deterministic_contract": {
            "ordered_lists_are_authoritative": True,
            "models_must_not_reorder": True,
            "models_may_only_summarize_reason_codes": True,
        },
        "priority_top3": _rows([item for item in picks if item.final_group == "PRIORITY"][:3], labels),
        "actionable_raw_top5": _rows(actionable_raw[:5], labels),
        "alpha_radar_top5": _rows([item for item in picks if item.final_group == "ALPHA_RADAR"][:5], labels),
        "signal_shadow_top3": _rows(signal_shadow, labels),
        "non_actionable_alpha_radar_top10": _rows(non_actionable[:10], labels),
        "pullback_scout_top10": _rows(pullback_scout[:10], labels),
    }


def explain_pair_order(artifact: dict[str, Any], first: str, second: str, *, list_name: str) -> dict[str, Any]:
    rows = {str(row["code"]): row for row in artifact.get(list_name, [])}
    first_row = rows[first]
    second_row = rows[second]
    first_reasons = set(first_row["reason_codes"])
    second_reasons = set(second_row["reason_codes"])
    return {
        "list_name": list_name,
        "first": first,
        "second": second,
        "first_rank": first_row["rank"],
        "second_rank": second_row["rank"],
        "first_sort_key": first_row["sort_key"],
        "second_sort_key": second_row["sort_key"],
        "first_reason_codes": first_row["reason_codes"],
        "second_reason_codes": second_row["reason_codes"],
        "first_only_reasons": sorted(first_reasons - second_reasons),
        "second_only_reasons": sorted(second_reasons - first_reasons),
    }


def render_prescreen_artifact_markdown(artifact: dict[str, Any]) -> str:
    lines = [
        f"# Deterministic IBD Prescreen Artifact ({artifact['snapshot_date']}, {artifact['version']})",
        "",
        "Do not reorder these rows. The ordered lists are generated from the deterministic ranking artifact.",
        "",
    ]
    for key, title in [
        ("priority_top3", "Priority Top 3"),
        ("actionable_raw_top5", "Actionable Raw Top 5"),
        ("alpha_radar_top5", "Alpha Radar Top 5"),
        ("signal_shadow_top3", "Signal Shadow Top 3"),
        ("non_actionable_alpha_radar_top10", "Non-ACTIONABLE Alpha Radar Top 10"),
        ("pullback_scout_top10", "Pullback Scout Top 10"),
    ]:
        rows = artifact.get(key, [])
        lines.extend([f"## {title}", "", "| Rank | Code | Status | Lane | Reasons | Risks |", "|---:|---|---|---|---|---|"])
        if not rows:
            lines.append("|  |  |  |  |  |  |")
        for row in rows:
            lines.append(
                "| {rank} | {code} | {entry_status} | {lane} | {reasons} | {risks} |".format(
                    rank=row["rank"],
                    code=row["code"],
                    entry_status=row["entry_status"],
                    lane=row["lane"],
                    reasons=";".join(row["reason_codes"]),
                    risks=";".join(row["risk_codes"]),
                )
            )
        lines.append("")
    return "\n".join(lines)


def _rows(items: list[ReasonedCandidate], labels: list[str]) -> list[dict[str, Any]]:
    return [_row(rank, item, labels) for rank, item in enumerate(items, 1)]


def _row(rank: int, item: ReasonedCandidate, labels: list[str]) -> dict[str, Any]:
    return {
        "rank": rank,
        "code": item.code,
        "entry_status": item.entry_status,
        "lane": item.lane,
        "final_group": item.final_group,
        "industry": item.industry,
        "ibd_candidate_rule": item.feature_values.get("ibd_candidate_rule"),
        "sort_key_labels": labels,
        "sort_key": list(item.sort_key),
        "reason_codes": item.reason_codes,
        "risk_codes": item.risk_codes,
        "fields": {
            "current_vs_ibd_candidate_pct": item.feature_values.get("current_vs_ibd_candidate_pct"),
            "ibd_entry_volume_ratio": item.feature_values.get("ibd_entry_volume_ratio"),
            "ibd_entry_close_position": item.feature_values.get("ibd_entry_close_position"),
            "ibd_entry_breakout_range_ratio": item.feature_values.get("ibd_entry_breakout_range_ratio"),
            "eps_yoy_growth": item.feature_values.get("eps_yoy_growth"),
            "volume_ratio": item.feature_values.get("volume_ratio"),
            "dist_to_52w_high_pct": item.feature_values.get("dist_to_52w_high_pct"),
            "pullback_v_is_dry": item.feature_values.get("pullback_v_is_dry"),
        },
    }


def _json_default(value: object) -> object:
    if pd.isna(value):
        return None
    return str(value)


if __name__ == "__main__":
    main()
