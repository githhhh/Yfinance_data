from __future__ import annotations

import pandas as pd


def render_report(
    *,
    baseline_vs_primary: pd.DataFrame,
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
            "B0 keeps every ACTIONABLE active signal. R1 keeps B0 unchanged and only adds",
            "Near-Buy-Point UNCONFIRMED / BELOW_TRIGGER candidates with >=1 positive quality evidence.",
            "Missing/False evidence is neutral; EXTENDED is exploratory only.",
            "",
            "## B0 vs primary R1",
            _markdown(baseline_vs_primary),
            "",
            "## Walk-forward train-selected champions",
            _markdown(walk_forward_champions),
            "",
            "## OOS rule stability",
            _markdown(oos_stability.head(20)),
            "",
            "## Retrospective champion candidate",
            f"- status: {champion_status}",
            f"- rule: {champion_rule}",
            "",
            "## EXTENDED exploratory lane",
            _markdown(extended_exploratory),
            "",
            "## Guardrails",
            "- 1W measures whether the weekend list captured a next-week review opportunity.",
            "- 2W/3W/4W measure follow-through quality, winner capture and loser exposure.",
            "- Big winners/losers are defined within each snapshot's full active-signal universe.",
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
