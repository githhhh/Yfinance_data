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
    retro = summary[summary["segment"] == "retrospective_all_40"]
    locked = summary[summary["segment"] == "locked_forward_18"]
    confirm = summary[summary["segment"] == "confirmation_12"]
    post = summary[summary["segment"] == "post_track_d_shadow"]

    lines = [
        "# Track E — B0.1 Dry-Neutral + Soft Active-Lane Audit",
        "",
        "## Question",
        "",
        "Can a stronger standard_breakout outrank a weaker fresh_demand_alpha when evidence/risk is better, "
        "without removing the downgrade semantics of incomplete_evidence / tail_risk?",
        "",
        "The challenger is intentionally singular:",
        "",
        "- pullback_v_is_dry=True: keep positive evidence.",
        "- pullback_v_is_dry=False: neutral; no risk penalty.",
        "- fresh_demand_alpha, constructive_pullback, standard_breakout: soft hierarchy.",
        "- incomplete_evidence, tail_risk: still structurally downgraded.",
        "- distinct_1, eligibility, capital accounting and all other B0 semantics remain unchanged.",
        "",
        "## Interpretation boundary",
        "",
        "This hypothesis was formulated after reviewing Track D. Therefore all evidence through "
        f"{manifest['track_d_historical_end']} is retrospective mechanism evidence, not untouched OOS. "
        "Only later mature W4 snapshots may be described as post-Track-D shadow evidence.",
        "",
        "## Segment comparison vs Production B0",
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
        "## Decision-impact events",
        "",
        f"- Total panel snapshots: **{event_summary['total_snapshots']}**",
        f"- Weeks where Top3 changed: **{event_summary['selection_changed_weeks']}**",
        f"- Weeks with targeted standard_breakout in / fresh_demand_alpha out: "
        f"**{event_summary['target_standard_over_fresh_weeks']}**",
        f"- Mature targeted swap weeks: **{event_summary['target_mature_weeks']}**",
        f"- Targeted mean W4 pair delta: **{_fmt(event_summary['target_mean_pair_delta_w4'])} pp**",
        f"- Targeted median W4 pair delta: **{_fmt(event_summary['target_median_pair_delta_w4'])} pp**",
        f"- Targeted positive pair ratio: **{_fmt(event_summary['target_positive_pair_ratio'])}**",
        f"- Mean portfolio W4 spread on changed weeks: "
        f"**{_fmt(event_summary['changed_week_mean_portfolio_spread_w4'])} pp**",
        f"- Mature post-Track-D shadow weeks: **{event_summary['post_track_d_mature_weeks']}**",
        f"- Changed post-Track-D shadow weeks: **{event_summary['post_track_d_changed_weeks']}**",
        "",
        "## Evidence readout",
        "",
    ]

    if not retro.empty:
        r = retro.iloc[0]
        lines.append(
            "- Retrospective 40-snapshot portfolio effect: "
            f"mean Δ {_fmt(r['mean_spread'])} pp, median Δ {_fmt(r['median_spread'])} pp, "
            f"CVaR Δ {_fmt(r['cvar_delta'])} pp, stop Δ {_fmt(r['stop_delta_pct'], 2)} pp."
        )
    if not locked.empty:
        r = locked.iloc[0]
        lines.append(
            "- Track-D locked-forward 18-snapshot replay: "
            f"mean Δ {_fmt(r['mean_spread'])} pp, median Δ {_fmt(r['median_spread'])} pp, "
            f"CVaR Δ {_fmt(r['cvar_delta'])} pp."
        )
    if not confirm.empty:
        r = confirm.iloc[0]
        lines.append(
            "- Former Track-D confirmation 12-snapshot replay: "
            f"mean Δ {_fmt(r['mean_spread'])} pp, median Δ {_fmt(r['median_spread'])} pp, "
            f"CVaR Δ {_fmt(r['cvar_delta'])} pp. This is not untouched for Track E."
        )
    if not post.empty:
        r = post.iloc[0]
        lines.append(
            "- Post-Track-D shadow: "
            f"{int(r['support_weeks'])} mature paired weeks, mean Δ {_fmt(r['mean_spread'])} pp."
        )

    target = events[
        (events["target_standard_over_fresh"] == True)
        & (events["target_pair_delta_w4"].notna())
    ].copy()
    lines += [
        "",
        "## Targeted swap evidence",
        "",
    ]
    if target.empty:
        lines.append(
            "No mature Top3 event directly exercised standard_breakout in / "
            "fresh_demand_alpha out. The historical portfolio comparison alone cannot validate "
            "the exact crossover mechanism."
        )
    else:
        lines.append(
            "| Snapshot | Segment | Standard in | Fresh out | Pair W4 Δ | Portfolio W4 Δ |"
        )
        lines.append("| --- | --- | --- | --- | ---: | ---: |")
        for _, row in target.iterrows():
            lines.append(
                f"| {row['snapshot_date']} | {row['segment']} | "
                f"{row['incoming_standard_codes']} | {row['outgoing_fresh_codes']} | "
                f"{_fmt(row['target_pair_delta_w4'])} | {_fmt(row['portfolio_spread_w4'])} |"
            )

    lines += [
        "",
        "## Conclusion rule",
        "",
        "Track E does **not** mutate Production B0. Historical results can support or reject the "
        "soft-Lane mechanism as a forward-shadow candidate, but production change requires genuinely "
        "future mature observations because this hypothesis was created after Track D.",
        "",
        "The raw selection_events.csv is the primary audit artifact for determining whether the "
        "intended cross-Lane mechanism actually fired, rather than inferring mechanism quality only "
        "from aggregate portfolio returns.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
