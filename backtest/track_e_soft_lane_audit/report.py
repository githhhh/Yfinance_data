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
    lines = [
        "# Track E v2 — Isolated Fresh-vs-Standard Lane Audit",
        "",
        "## Fixed question",
        "",
        "Can a stronger standard_breakout outrank a weaker fresh_demand_alpha when its "
        "status/evidence/risk profile is better?",
        "",
        "## Controlled intervention",
        "",
        "- pullback_v_is_dry=True remains positive evidence.",
        "- pullback_v_is_dry=False is neutral.",
        "- Build the reward-only B0 ranking skeleton first.",
        "- Keep constructive_pullback, incomplete_evidence, and tail_risk at their exact skeleton positions.",
        "- Reorder only fresh_demand_alpha and standard_breakout candidates among the slots those two lanes already occupy.",
        "- Within those target slots compare status -> evidence/risk -> original Lane -> remaining B0 tie-breaks.",
        "- distinct_1, eligibility, and 3-slot capital accounting remain unchanged.",
        "",
        "This fixes Track E v1, where constructive_pullback was also softened and dominated every actual crossover.",
        "",
        "## Evidence boundary",
        "",
        "The hypothesis is post-Track-D. Evidence through "
        f"{manifest['track_d_historical_end']} is retrospective mechanism evidence, not untouched OOS.",
        "",
        "## Paired portfolio comparison vs Production B0",
        "",
        "| Segment | Support | Mean Δ | Median Δ | CVaR Δ | Stop Δ pp | Ruin Δ pp | Jaccard | CI low | CI high |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['segment']} | {int(row['support_weeks'])} | "
            f"{_fmt(row['mean_spread'])} | {_fmt(row['median_spread'])} | "
            f"{_fmt(row['cvar_delta'])} | {_fmt(row['stop_delta_pct'], 2)} | "
            f"{_fmt(row['one_pick_ruins_delta_pct'], 2)} | {_fmt(row['jaccard_vs_b0'])} | "
            f"{_fmt(row['ci_low'])} | {_fmt(row['ci_high'])} |"
        )

    lines += [
        "",
        "## Mechanism-trigger audit",
        "",
        f"- Total snapshots: **{event_summary['total_snapshots']}**",
        f"- Order-changed weeks: **{event_summary['order_changed_weeks']}**",
        f"- Top3 membership-changed weeks: **{event_summary['membership_changed_weeks']}**",
        f"- Mature membership-changed weeks: **{event_summary['mature_membership_changed_weeks']}**",
        f"- Fresh/standard rank-crossover weeks: **{event_summary['target_rank_crossover_weeks']}**",
        f"- Mature rank-crossover weeks: **{event_summary['target_rank_crossover_mature_weeks']}**",
        f"- Rank-crossover mean W4 pair Δ: **{_fmt(event_summary['target_rank_mean_pair_delta_w4'])} pp**",
        f"- Rank-crossover median W4 pair Δ: **{_fmt(event_summary['target_rank_median_pair_delta_w4'])} pp**",
        f"- Rank-crossover positive ratio: **{_fmt(event_summary['target_rank_positive_pair_ratio'])}**",
        f"- Top3 standard-in / fresh-out weeks: **{event_summary['target_selection_swap_weeks']}**",
        f"- Mature Top3 targeted swaps: **{event_summary['target_selection_swap_mature_weeks']}**",
        f"- Top3 targeted mean W4 pair Δ: **{_fmt(event_summary['target_selection_mean_pair_delta_w4'])} pp**",
        f"- Top3 targeted median W4 pair Δ: **{_fmt(event_summary['target_selection_median_pair_delta_w4'])} pp**",
        f"- Top3 targeted positive ratio: **{_fmt(event_summary['target_selection_positive_pair_ratio'])}**",
        f"- Mean portfolio W4 spread on membership-changed weeks: "
        f"**{_fmt(event_summary['membership_changed_mean_portfolio_spread_w4'])} pp**",
        "",
        "## Targeted selection swaps",
        "",
    ]

    target = events[
        (events["target_selection_swap"] == True)
        & (events["selection_pair_delta_w4"].notna())
    ].copy()
    if target.empty:
        lines.append("No mature Top3 standard-in / fresh-out event occurred.")
    else:
        lines += [
            "| Snapshot | Segment | Standard in | Fresh out | Pair W4 Δ | Portfolio W4 Δ |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
        for _, row in target.iterrows():
            lines.append(
                f"| {row['snapshot_date']} | {row['segment']} | "
                f"{row['incoming_standard_codes']} | {row['outgoing_fresh_codes']} | "
                f"{_fmt(row['selection_pair_delta_w4'])} | {_fmt(row['portfolio_spread_w4'])} |"
            )

    lines += [
        "",
        "## Interpretation rule",
        "",
        "Track E v2 is a mechanism audit, not an automatic production promotion. The primary question is "
        "first whether the intended fresh/standard crossover actually fires, then whether the resulting "
        "rank/selection substitutions improve W4 outcome without degrading downside metrics.",
        "",
        "Production B0 remains untouched.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
