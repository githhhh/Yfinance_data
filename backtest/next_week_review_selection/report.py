from __future__ import annotations

import pandas as pd


def render_report(
    *,
    baseline_vs_primary: pd.DataFrame,
    macro_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    coverage_summary: pd.DataFrame,
    adaptive_summary: pd.DataFrame,
    convergence: pd.DataFrame,
    setup_balanced_summary: pd.DataFrame,
    oos_stability: pd.DataFrame,
    walk_forward_champions: pd.DataFrame,
    tail_exploratory: pd.DataFrame,
    adaptive_status: str,
    static_status: str,
    static_rule: str,
    extended_exploratory: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# Next Week Review Selection Research",
            "",
            "Status: retrospective_pre_registered_replay_v0.5",
            "",
            "## Core question",
            "Can a train-selected supplemental admission rule capture more tradable winners",
            "per unit of review attention than ACTIONABLE-only, without proportionally",
            "increasing loser exposure?",
            "",
            "## Price-path coverage audit",
            _markdown(coverage_summary),
            "",
            "## B0 vs Primary R1",
            _markdown(baseline_vs_primary),
            "",
            "## Primary R1 weekly macro aggregation",
            _markdown(macro_summary),
            "",
            "## Primary R1 moving-block bootstrap",
            _markdown(bootstrap),
            "",
            "## Adaptive policy — formal OOS",
            _markdown(adaptive_summary),
            f"- adaptive status: {adaptive_status}",
            "",
            "## Train champion convergence",
            _markdown(convergence),
            f"- static-rule status: {static_status}",
            f"- static-rule candidate: {static_rule or 'n/a'}",
            "",
            "## Setup-balanced sensitivity",
            _markdown(setup_balanced_summary),
            "",
            "## Per-rule OOS stability",
            _markdown(oos_stability),
            "",
            "## Formal train-selected champions",
            _markdown(walk_forward_champions),
            "",
            "## Tail exploratory — excluded from formal verdict",
            _markdown(tail_exploratory),
            "",
            "## EXTENDED exploratory lane",
            _markdown(extended_exploratory),
            "",
            "## Guardrails",
            "- Every formal fold receives a train-only provisional champion; training stability cannot veto OOS entry.",
            "- Only full-size 4-week test blocks enter the formal OOS verdict; the final partial tail is exploratory.",
            "- Behaviorally identical rule parameterizations are de-duplicated by selected (snapshot, code) signature.",
            "- Adaptive-policy performance is aggregated across folds even when the chosen rule changes.",
            "- Setup-balanced sensitivity equal-weights eligible setup strata.",
            "- Price-path missingness, dual clocks, capacity lift, C Rank/ATR exclusions remain unchanged.",
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
