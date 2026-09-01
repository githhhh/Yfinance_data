from __future__ import annotations
import json
from pathlib import Path


def generate_blind_discovery_prompt(feature_manifest: dict) -> str:
    """Generate the outcome-blind research specification prompt for hypothesis discovery."""
    allowed_feats = [
        f"{k}: {v['as_of_semantics']} (type: {v['data_type']})"
        for k, v in feature_manifest["features"].items()
        if v.get("allowed_for_discovery") is True
    ]

    prompt = f"""# Track C Blind Policy Discovery Task

You are tasked with proposing candidate stock ranking and Top3 portfolio selection policies for US growth equities.
You must adhere strictly to the 3-Layer Decoupled Architecture:
1. `score_candidates(snapshot_df) -> pd.DataFrame`
2. `allocate_industries(scored_df) -> dict[str, int]` (returns industry quotas summing <= 3)
3. `pick_stocks(scored_df, industry_quotas) -> list[str]` (returns 0..3 selected codes)

## Allowed Feature Set (Point-in-Time Point Safe)
{json.dumps(allowed_feats, indent=2)}

## Research Families to Generate:
1. `industry_breadth`: Industry-first policies that rank industries by candidate concentration/breadth and allocate dynamic quotas (e.g. 2+1 or 1+1+1).
2. `continuous`: Multi-factor weighted continuous score rankers breaking lexicographic sorting.
3. `ltr`: Pairwise / Tree-based ranking models.
4. `portfolio`: Portfolio utility rankers maximizing stock quality minus industry concentration penalties.
5. `novel`: Domain-grounded heuristics combining volume, EPS, and base geometry in novel ways.

All proposed policies must output valid executable specifications.
"""
    return prompt
