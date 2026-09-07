from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _md_table(df: pd.DataFrame, cols: list[str], max_rows: int = 12) -> str:
    if df.empty:
        return "_No rows._"
    available = [c for c in cols if c in df.columns]
    view = df[available].head(max_rows).copy()
    header = "| " + " | ".join(available) + " |"
    sep = "| " + " | ".join("---" for _ in available) + " |"
    rows = [header, sep]
    for _, row in view.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in available) + " |")
    return "\n".join(rows)


def write_final_report(
    path: Path,
    decision: dict[str, Any],
    mechanism_df: pd.DataFrame,
    failure_summary: dict[str, Any],
    forward_df: pd.DataFrame,
    split: dict[str, Any],
    request_budget: dict[str, Any],
    interpretations: list[dict[str, Any]],
    frozen_policy_count: int,
) -> None:
    component_df = pd.DataFrame(decision.get("component_verdicts", []))
    top = forward_df.copy()
    top["_confirm"] = top.get("confirmation_evaluated", False)
    top["_confirm_mean"] = pd.to_numeric(
        top.get("confirmation_mean_spread", pd.Series(index=top.index, dtype=float)),
        errors="coerce",
    )
    top["_screen_mean"] = pd.to_numeric(
        top.get("screen_mean_spread", pd.Series(index=top.index, dtype=float)),
        errors="coerce",
    )
    top = top.sort_values(
        ["_confirm", "b1_gate_pass", "_confirm_mean", "_screen_mean"],
        ascending=[False, False, False, False],
    )

    screening_blocks = len(split.get("screening_blocks", []))
    confirmation_blocks = len(split.get("confirmation_blocks", []))
    block_size = len(split["forward_blocks"][0]["snapshots"]) if split.get("forward_blocks") else 0

    lines = [
        "# Track D Final Report — B0 Mechanism Discovery & B1 Synthesis",
        "",
        "## Decision",
        f"**{decision['state']}**",
        "",
        decision.get("decision_basis", ""),
        "",
        "The terminal result is a production decision plus a rule-level mechanism map; "
        "Track D does not treat 'B0 was not beaten' as a sufficient conclusion.",
        "",
        "## Frozen research design",
        f"- Discovery train snapshots: {len(split['discovery_train'])}",
        f"- Purge snapshots: {len(split['purge'])}",
        f"- Screening: {screening_blocks} blocks x {block_size} weeks",
        f"- Untouched confirmation: {confirmation_blocks} blocks x {block_size} weeks",
        f"- Frozen RD-Agent DSL policies: {frozen_policy_count}",
        f"- DeepSeek request attempts used: {request_budget.get('attempts_used')} / {request_budget.get('hard_limit')}",
        "- Agent/Minimal-B0 candidates were shortlisted using screening blocks only; "
        "State A/B/C gates use untouched confirmation blocks.",
        "",
        "## B0 component verdicts — confirmation evidence",
        _md_table(
            component_df,
            [
                "component", "verdict", "confirmation_support_weeks",
                "knockout_mean_spread", "knockout_median_spread",
                "knockout_cvar_delta", "knockout_stop_delta_pct", "positive_block_ratio",
            ],
            20,
        ) if not component_df.empty else "_No component verdicts._",
        "",
        "## Locked forward leaders",
        _md_table(
            top,
            [
                "policy_id", "policy_kind", "screen_shortlisted",
                "screen_mean_spread", "confirmation_evaluated",
                "confirmation_mean_spread", "confirmation_median_spread",
                "confirmation_cvar_delta", "confirmation_stop_delta_pct",
                "confirmation_positive_block_ratio", "b1_gate_pass", "compression_gate_pass",
            ],
            18,
        ),
        "",
        "## B0 failure archaeology",
        f"- Labeled discovery-train cases: {failure_summary.get('case_count', 0)}",
        f"- Label counts: {json.dumps(failure_summary.get('labels', {}), ensure_ascii=False)}",
        "- Failure archaeology is hypothesis-generating only; it cannot directly promote a policy.",
        "",
        "## RD-Agent post-evaluation interpretations",
    ]

    for idx, item in enumerate(interpretations, 1):
        if not isinstance(item, dict):
            item = {
                "conclusion": str(item),
                "strongest_evidence": [],
                "contradictions": [],
                "unresolved": [],
            }
        lines.extend([
            f"### Reviewer {idx}",
            str(item.get("conclusion", "")),
            "",
            f"- Strongest evidence: {item.get('strongest_evidence', [])}",
            f"- Contradictions: {item.get('contradictions', [])}",
            f"- Unresolved: {item.get('unresolved', [])}",
            "",
        ])

    if decision.get("winner"):
        winner = decision["winner"]
        lines.extend([
            "## Selected forward-shadow candidate",
            f"- Policy: {winner.get('policy_id')}",
            f"- Confirmation mean spread: {winner.get('confirmation_mean_spread')}%",
            f"- Confirmation median spread: {winner.get('confirmation_median_spread')}%",
            f"- Confirmation CVaR delta: {winner.get('confirmation_cvar_delta')}%",
            f"- Confirmation stop delta: {winner.get('confirmation_stop_delta_pct')} pp",
            f"- Positive confirmation-block ratio: {winner.get('confirmation_positive_block_ratio')}",
            "",
        ])

    lines.extend([
        "## Interpretation boundary",
        "A historical winner may enter forward shadow only. Production B0 is not modified by Track D; "
        "a production replacement still requires genuinely future observations.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
