from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_report(
    path: Path,
    evaluation: pd.DataFrame,
    taxonomy_summary: dict[str, Any],
    route_pair: dict[str, Any],
    parity_anchor: dict[str, Any],
    decision: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    focus_segment = "retrospective_track_d_40"
    focus = evaluation[evaluation["segment"] == focus_segment].copy()
    primary = focus[focus["role"] == "primary"].copy()
    secondary = focus[focus["role"] == "secondary"].copy()

    lines = [
        "# Track F — Lane Taxonomy & Composition Audit",
        "",
        "## Why this track exists",
        "",
        "Production B0 currently uses one Lane enum to represent several different concepts at once: "
        "setup route, evidence completeness, actionability context, and failure state. Track F does not "
        "change Production. It first decomposes Lane into orthogonal facts, then tests a small frozen "
        "set of composition policies against exact Production B0.",
        "",
        "## Orthogonal interpretation",
        "",
        "- setup_route: non_pullback / pullback",
        "- fresh_demand: near buy point AND entry volume >= 1.5x",
        "- follow_through: EPS >= 25% OR weekly volume >= 1.3x",
        "- quality_state: confirmed / standard / incomplete / failure",
        "",
        "For B0-eligible ACTIONABLE rows:",
        "",
        "- fresh_demand_alpha = confirmed + non_pullback route",
        "- constructive_pullback = confirmed + pullback route",
        "- standard_breakout = standard quality; route may be pullback or non_pullback",
        "- incomplete_evidence = incomplete quality",
        "- tail_risk = geometry failure",
        "",
        "The non-ACTIONABLE constructive_pullback context branch is recorded separately because it is "
        "a different semantic path and cannot enter Production Top3 eligibility.",
        "",
        "## Taxonomy audit",
        "",
        f"- Total review rows: **{taxonomy_summary['total_review_rows']}**",
        f"- B0-eligible rows: **{taxonomy_summary['b0_eligible_rows']}**",
        f"- constructive_pullback rows: **{taxonomy_summary['constructive_pullback_rows']}**",
        f"- constructive actionable-confirmed branch rows: "
        f"**{taxonomy_summary['constructive_actionable_confirmed_branch_rows']}**",
        f"- constructive non-actionable-context branch rows: "
        f"**{taxonomy_summary['constructive_non_actionable_context_branch_rows']}**",
        f"- constructive other rows: **{taxonomy_summary['constructive_other_rows']}**",
        f"- eligible standard_breakout rows from pullback route: "
        f"**{taxonomy_summary['eligible_standard_pullback_rows']}**",
        f"- eligible standard_breakout rows from non-pullback route: "
        f"**{taxonomy_summary['eligible_standard_non_pullback_rows']}**",
        "",
        "## Candidate-pool route diagnostic",
        "",
        "This is descriptive supporting evidence, not a portfolio promotion gate. On weeks where both "
        "confirmed route groups have fully mature W4 outcomes:",
        "",
        f"- paired support weeks: **{route_pair['support_weeks']}**",
        f"- mean pullback minus non-pullback W4: "
        f"**{_fmt(route_pair['mean_pullback_minus_non_pullback_w4'])} pp**",
        f"- median pullback minus non-pullback W4: "
        f"**{_fmt(route_pair['median_pullback_minus_non_pullback_w4'])} pp**",
        f"- positive-week ratio: **{_fmt(route_pair['positive_week_ratio'])}**",
        f"- mean stop-rate delta: **{_fmt(route_pair['mean_stop_delta_pct'])} pp**",
        "",
        "## Frozen primary policies vs Production B0",
        "",
        "All primary policies preserve Production B0 eligibility, symmetric dry semantics, 3-slot "
        "capital accounting, and distinct_1 industry dispersion.",
        "",
        "- CONFIRMED_PARITY_FALLBACK: confirmed pullback/non-pullback have equal route priority; "
        "standard/incomplete remain fallback quality tiers.",
        "- CONFIRMED_ONLY_TOP3: Top3 may come only from confirmed candidates of either route; no forced fill.",
        "- FCS_MAX1: max one confirmed_non_pullback, one confirmed_pullback, one standard; no forced fill.",
        "",
        f"### {focus_segment}",
        "",
        "| Policy | Mean Δ | Median Δ | CVaR Δ | Stop Δ pp | Ruin Δ pp | Coverage | Full Top3 | Jaccard | CI low | CI high |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, row in primary.iterrows():
        lines.append(
            f"| {row['policy_id']} | {_fmt(row['mean_spread'])} | {_fmt(row['median_spread'])} | "
            f"{_fmt(row['cvar_delta'])} | {_fmt(row['stop_delta_pct'], 2)} | "
            f"{_fmt(row['one_pick_ruins_delta_pct'], 2)} | {_fmt(row['slot_coverage_pct'], 2)} | "
            f"{_fmt(row['full_top3_rate_pct'], 2)} | {_fmt(row['jaccard_vs_b0'])} | "
            f"{_fmt(row['ci_low'])} | {_fmt(row['ci_high'])} |"
        )

    lines += [
        "",
        "## Secondary industry diagnostic",
        "",
        "These policies use identical Lane logic but remove distinct_1. They are diagnostic only; "
        "they must not be used to attribute a Lane effect because Lane composition and industry "
        "concentration both change relative to Production B0.",
        "",
        "| Policy | Mean Δ | Median Δ | CVaR Δ | Stop Δ pp | Ruin Δ pp | Coverage | Jaccard |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, row in secondary.iterrows():
        lines.append(
            f"| {row['policy_id']} | {_fmt(row['mean_spread'])} | {_fmt(row['median_spread'])} | "
            f"{_fmt(row['cvar_delta'])} | {_fmt(row['stop_delta_pct'], 2)} | "
            f"{_fmt(row['one_pick_ruins_delta_pct'], 2)} | {_fmt(row['slot_coverage_pct'], 2)} | "
            f"{_fmt(row['jaccard_vs_b0'])} |"
        )

    lines += [
        "",
        "## Integrity anchor",
        "",
        "Track F CONFIRMED_PARITY_FALLBACK is expected to reproduce the existing Track C "
        "PULLBACK_PARITY selector under distinct_1. This anchors the new orthogonal taxonomy to the "
        "previously tested structural policy.",
        "",
        f"- snapshots checked: **{parity_anchor['snapshot_count']}**",
        f"- pick mismatches: **{parity_anchor['mismatch_count']}**",
        "",
        "## Pre-registered historical support decision",
        "",
        f"- Overall: **{decision['overall']}**",
        f"- Historical shadow candidates: **{decision['shadow_candidates']}**",
        "",
        "This gate is retrospective-only. Passing it cannot promote Production; it only identifies a policy "
        "worth observing on future unseen weeks.",
        "",
        "## Evidence boundary",
        "",
        manifest["evidence_status"],
        "",
        "## Interpretation rule",
        "",
        "Track F is a mechanism/composition audit, not a search for the best historical policy. "
        "No thresholds are fitted and no policy is generated after seeing outcomes. A useful result "
        "must show coherent improvement across median/mean, downside and coverage without relying on "
        "one historical segment. Production B0 remains untouched.",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
