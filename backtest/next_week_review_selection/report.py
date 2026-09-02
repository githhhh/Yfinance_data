from __future__ import annotations

import pandas as pd


def render_report(
    *,
    baseline_vs_primary: pd.DataFrame,
    macro_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    oos_stability: pd.DataFrame,
    walk_forward_champions: pd.DataFrame,
    champion_status: str,
    champion_rule: str,
    extended_exploratory: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# Next Week Review Selection Research",
            "",
            "Status: retrospective_pre_registered_replay",
            "",
            "## Core hypothesis",
            "B0 keeps every ACTIONABLE active signal. Primary R1 keeps B0 unchanged and adds",
            "Near-Buy-Point UNCONFIRMED / BELOW_TRIGGER candidates with >=1 independent",
            "positive evidence family. Geometry is NOT a hard reject in Primary R1.",
            "",
            "## B0 vs Primary R1 — micro aggregation",
            _markdown(baseline_vs_primary),
            "",
            "## Weekly macro aggregation",
            _markdown(macro_summary),
            "",
            "## Paired moving-block bootstrap",
            _markdown(bootstrap),
            "",
            "## Walk-forward train-selected champions",
            _markdown(walk_forward_champions),
            "",
            "## OOS stability",
            _markdown(oos_stability),
            "",
            "## Retrospective champion candidate",
            f"- status: {champion_status}",
            f"- rule: {champion_rule}",
            "",
            "## EXTENDED exploratory lane",
            _markdown(extended_exploratory),
            "",
            "## Guardrails",
            "- Snapshot clock asks whether a weekend candidate later became a winner.",
            "- Opportunity clock asks whether quality persisted AFTER a real Buy-Zone/confirmation event.",
            "- Winner recall is interpreted together with Selection Coverage and capture lift.",
            "- Rule evolution is two-stage: 24 structural rules, then evidence leave-one-out only around finalists.",
            "- C Rank, ATR and new technical indicators are excluded.",
            "- No production Skill/Futu/Dashboard change is authorized by this retrospective study.",
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
