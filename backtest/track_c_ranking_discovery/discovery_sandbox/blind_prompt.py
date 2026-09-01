from __future__ import annotations

import json
from typing import Any


FAMILY_PARAM_SCHEMA = {
    "industry_breadth": {
        "breadth_metric": "one of: actionable_count, volume_breadth, quality_and_count",
        "allow_dynamic_2_plus_1": "boolean",
        "min_breadth_for_2": "integer 2 or 3",
    },
    "continuous": {
        "weights": "object mapping 2..8 allowed numeric feature names to non-zero weights in [-8, 8]",
        "selector_mode": "one of: distinct_1, max_2_per_ind, pure_top3",
    },
    "linear_ranking": {
        "feature_subset": "array of 2..8 allowed numeric feature names",
        "regularization": "number in [0.25, 5.0]",
        "selector_mode": "one of: distinct_1, max_2_per_ind, pure_top3",
    },
    "portfolio": {
        "concentration_lambda": "number in [0.0, 5.0]",
        "stock_quality_metric": "one of: balanced, momentum_first",
    },
    "novel_heuristic": {
        "dry_weight": "number in [0.0, 6.0]",
        "base_depth_penalty": "number in [0.0, 6.0]",
        "volume_spike_bonus": "number in [0.0, 6.0]",
        "selector_mode": "one of: distinct_1, max_2_per_ind, pure_top3",
    },
}


def generate_blind_discovery_prompt(
    feature_manifest: dict[str, Any],
    family: str,
    budget: int,
    data_summary: dict[str, Any],
) -> str:
    """Build a family-scoped, outcome-blind policy-discovery prompt."""
    allowed_feats = {
        k: {
            "semantics": v["as_of_semantics"],
            "type": v["data_type"],
        }
        for k, v in feature_manifest["features"].items()
        if v.get("allowed_for_discovery") is True
    }
    if family not in FAMILY_PARAM_SCHEMA:
        raise ValueError(f"Unsupported discovery family: {family}")

    return f"""# Track C Blind RD-Agent Policy Discovery

You are proposing hypotheses BEFORE any forward return or stop outcome is revealed.
The input summary is strictly Train-only and ticker/date anonymized.

## Immutable constraints
- Never infer or invent ticker-specific or date-specific rules.
- Never use future returns, stop outcomes, B0 future performance, or post-snapshot data.
- The final portfolio capacity is 0..3 stocks; do not force fill-to-3.
- Policies must be generic and reproducible from PIT-safe snapshot features.
- Generate diverse hypotheses, not tiny coefficient perturbations of the same rule.
- Return JSON only.

## Family
{family}

## Maximum proposals
{budget}

## Allowed feature dictionary
{json.dumps(allowed_feats, ensure_ascii=False, indent=2)}

## Outcome-blind Train distribution summary
{json.dumps(data_summary, ensure_ascii=False, indent=2)}

## Required family parameter schema
{json.dumps(FAMILY_PARAM_SCHEMA[family], ensure_ascii=False, indent=2)}

## Exact output schema
{{
  "proposals": [
    {{
      "family": "{family}",
      "name": "short_unique_snake_case_name",
      "hypothesis": "concise economic / technical rationale",
      "params": {{
        "...": "must exactly follow the family parameter schema above"
      }}
    }}
  ]
}}

Generate up to {budget} meaningfully distinct proposals. Do not include prose outside the JSON object.
"""
