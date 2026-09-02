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
    summary: pd.DataFrame,
    events: pd.DataFrame,
    event_summary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    primary = summary[summary["comparator"] == "dry_neutral_hard_lane_control"].copy()
    production = summary[summary["comparator"] == "production_b0_reference"].copy()

    lines = [
        "# Track E v3 — Pairwise Standard-vs-Fresh Top3 Replacement",
        "",
        "## Research question",
        "",
        "Should an otherwise eligible standard_breakout be allowed to replace an already-selected "
        "fresh_demand_alpha when the standard candidate is unambiguously stronger on independent "
        "entry-quality axes?",
        "",
        "## Controlled design",
        "",
        "- Primary control: hard B0 Lane with dry=True reward and dry=False neutral.",
        "- Production B0 is reported separately as a reference, not used to attribute the Lane effect.",
        "- Challenger starts from the primary-control Top3.",
        "- constructive_pullback, incomplete_evidence, tail_risk, and every non-fresh selected slot are frozen.",
        "- Only an unselected standard_breakout may challenge a selected fresh_demand_alpha slot.",
        "- Replacement requires unweighted Pareto dominance:",
        "  - no more risk flags;",
        "  - no worse absolute distance to buy point;",
        "  - no weaker entry-volume ratio;",
        "  - at least one of those axes strictly better.",
        "- EPS>=25 and weekly-volume follow-through are excluded from the dominance test because they "
        "define fresh_demand_alpha itself; including them would make the test circular.",
        "- distinct_1 and portfolio capacity are preserved.",
        "",
        "This corrects Track E v1/v2, which altered global Lane ordering instead of isolating the "
        "specific Top3 replacement question.",
        "",
        "## Evidence boundary",
        "",
        "This is a post-Track-D hypothesis. Evidence through "
        f"{manifest['track_d_historical_end']} is retrospective mechanism evidence, not untouched OOS.",
        "",
        "## Primary comparison — challenger vs dry-neutral hard-Lane control",
        "",
        "| Segment | Support | Mean Δ | Median Δ | CVaR Δ | Stop Δ pp | Ruin Δ pp | Jaccard | CI low | CI high |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, row in primary.iterrows():
        lines.append(
            f"| {row['segment']} | {int(row['support_weeks'])} | "
            f"{_fmt(row['mean_spread'])} | {_fmt(row['median_spread'])} | "
            f"{_fmt(row['cvar_delta'])} | {_fmt(row['stop_delta_pct'], 2)} | "
            f"{_fmt(row['one_pick_ruins_delta_pct'], 2)} | "
            f"{_fmt(row['jaccard_vs_comparator'])} | {_fmt(row['ci_low'])} | {_fmt(row['ci_high'])} |"
        )

    lines += [
        "",
        "## Mechanism support",
        "",
        f"- Total snapshots: **{event_summary['total_snapshots']}**",
        f"- Weeks with at least one Pareto-valid standard-vs-fresh opportunity: "
        f"**{event_summary['opportunity_weeks']}**",
        f"- Pareto-valid candidate pairs: **{event_summary['opportunity_pairs']}**",
        f"- Weeks with an actual Top3 replacement: **{event_summary['actual_swap_weeks']}**",
        f"- Actual replacement pairs: **{event_summary['actual_swap_pairs']}**",
        f"- Mature replacement weeks: **{event_summary['mature_swap_weeks']}**",
        f"- Mature replacement pairs: **{event_summary['mature_swap_pairs']}**",
        f"- Replacement-pair mean W4 Δ: **{_fmt(event_summary['swap_pair_mean_w4_delta'])} pp**",
        f"- Replacement-pair median W4 Δ: **{_fmt(event_summary['swap_pair_median_w4_delta'])} pp**",
        f"- Replacement-pair positive ratio: **{_fmt(event_summary['swap_pair_positive_ratio'])}**",
        f"- Mean portfolio W4 Δ on membership-changed weeks: "
        f"**{_fmt(event_summary['changed_week_mean_portfolio_spread_vs_control_w4'])} pp**",
        "",
        "## Dry-neutral control vs Production B0 diagnostic",
        "",
        f"- Order-changed weeks: **{event_summary['dry_control_vs_production_order_changed_weeks']}**",
        f"- Membership-changed weeks: **{event_summary['dry_control_vs_production_membership_changed_weeks']}**",
        "",
        "The production-reference comparison is retained to show the net B0.1 effect, but the "
        "primary Lane conclusion must use the dry-neutral hard-Lane control above.",
        "",
        "## Production B0 reference",
        "",
        "| Segment | Support | Mean Δ | Median Δ | CVaR Δ | Stop Δ pp | Ruin Δ pp | Jaccard |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, row in production.iterrows():
        lines.append(
            f"| {row['segment']} | {int(row['support_weeks'])} | "
            f"{_fmt(row['mean_spread'])} | {_fmt(row['median_spread'])} | "
            f"{_fmt(row['cvar_delta'])} | {_fmt(row['stop_delta_pct'], 2)} | "
            f"{_fmt(row['one_pick_ruins_delta_pct'], 2)} | "
            f"{_fmt(row['jaccard_vs_comparator'])} |"
        )

    target = events[
        (events["target_selection_swap"] == True)
        & (events["mean_swap_pair_delta_w4"].notna())
    ].copy()
    lines += [
        "",
        "## Mature targeted swaps",
        "",
    ]
    if target.empty:
        lines.append(
            "No mature targeted swap occurred. In that case the experiment is non-identifying for "
            "outcome quality and should be reported as insufficient support rather than a win/loss."
        )
    else:
        lines += [
            "| Snapshot | Segment | Swap pairs | Portfolio Δ vs control |",
            "| --- | --- | --- | ---: |",
        ]
        for _, row in target.iterrows():
            lines.append(
                f"| {row['snapshot_date']} | {row['segment']} | "
                f"{row['swap_pairs_json']} | {_fmt(row['portfolio_spread_vs_control_w4'])} |"
            )

    lines += [
        "",
        "## Interpretation rule",
        "",
        "The result is only informative if Pareto-valid opportunities and actual replacements exist. "
        "If they do, judge the mechanism first on matched replacement-pair W4 deltas and then on "
        "portfolio mean/median/CVaR/stop behavior. Historical evidence can justify a forward shadow, "
        "but cannot by itself promote a production Lane change because the hypothesis was formed after Track D.",
        "",
        "Production B0 remains untouched.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
