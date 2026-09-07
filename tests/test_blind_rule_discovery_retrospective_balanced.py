from __future__ import annotations

import json

import pandas as pd

from backtest.blind_rule_discovery.retrospective_ceiling_balanced import (
    _balanced_condition_pool,
    _score_all_single_conditions,
    balanced_rolling_walk_forward,
    balanced_search_rules,
)
from backtest.blind_rule_discovery.retrospective_ceiling import (
    _condition_key,
    _condition_mask,
    generate_conditions,
)


def interaction_history(quarters: int = 8) -> pd.DataFrame:
    """A/B have zero marginal lift but A AND B is perfect; decoys crowd global beams."""
    rows = []
    for q_index in range(quarters):
        period = f"{2023 + q_index // 4}Q{q_index % 4 + 1}"
        cells = [
            (0.9, 0.9, "winner"),
            (0.9, 0.1, "loser"),
            (0.1, 0.9, "loser"),
            (0.1, 0.1, "winner"),
        ]
        for a, b, primary in cells:
            for i in range(10):
                row = {
                    "period_quarter": period,
                    "Y_primary": primary,
                    "A": a,
                    "B": b,
                    "Y_12w_excess": 0.18 if (a == 0.9 and b == 0.9) else (0.02 if primary == "winner" else -0.06),
                    "Y_mae_12w": -0.03 if (a == 0.9 and b == 0.9) else -0.09,
                    "Y_mfe_12w": 0.30 if (a == 0.9 and b == 0.9) else 0.10,
                }
                # Decoys have slight marginal signal and can crowd a tiny global beam.
                for d in range(6):
                    if primary == "winner":
                        row[f"D{d}"] = 0.55 + 0.01 * d
                    else:
                        row[f"D{d}"] = 0.45 + 0.01 * d
                    # Alternate a few observations to keep decoys imperfect.
                    if i < 3:
                        row[f"D{d}"] = 1.0 - row[f"D{d}"]
                rows.append(row)
    return pd.DataFrame(rows)


def test_balanced_pool_keeps_every_feature_even_when_masks_are_identical():
    frame = interaction_history()
    features = ["A", "B", *[f"D{i}" for i in range(6)]]
    conditions = generate_conditions(frame, features, quantiles=(0.25, 0.5, 0.75))
    condition_masks = {_condition_key(c): _condition_mask(frame, c) for c in conditions}
    singles = _score_all_single_conditions(
        frame,
        conditions,
        condition_masks,
        min_selected=20,
        min_resolved=20,
        min_active_quarters=4,
        min_resolved_per_quarter=3,
    )
    pool = _balanced_condition_pool(singles, conditions_per_feature=1)
    assert {c["feature"] for c in pool} == set(features)


def test_balanced_pair_search_finds_zero_marginal_ab_interaction():
    frame = interaction_history()
    features = ["A", "B", *[f"D{i}" for i in range(6)]]
    scored, _ = balanced_search_rules(
        frame,
        features,
        quantiles=(0.25, 0.5, 0.75),
        conditions_per_feature=2,
        beam_width=2,
        max_conditions=2,
        max_clauses=1,
        dnf_clause_pool=0,
        min_selected=20,
        min_resolved=20,
        min_active_quarters=4,
        min_resolved_per_quarter=3,
    )
    ab = []
    for _, row in scored.loc[scored["condition_count"] == 2].iterrows():
        rule = json.loads(row["rule_json"])
        conditions = rule["clauses"][0]["all"]
        if {c["feature"] for c in conditions} == {"A", "B"}:
            ab.append(row)
    assert ab
    assert max(float(row["resolved_winner_rate"]) for row in ab) == 1.0
    assert max(float(row["winner_rate_lift"]) for row in ab) == 0.5


def test_balanced_rolling_uses_only_prior_quarters_and_rediscovers_interaction():
    frame = interaction_history()
    features = ["A", "B", *[f"D{i}" for i in range(6)]]
    rolling, usage = balanced_rolling_walk_forward(
        frame,
        features,
        min_train_quarters=4,
        conditions_per_feature=2,
        beam_width=4,
        min_selected=20,
        min_resolved=15,
        min_resolved_per_quarter=3,
    )
    assert rolling["test_quarter"].tolist() == ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]
    assert rolling["train_end_quarter"].tolist() == ["2023Q4", "2024Q1", "2024Q2", "2024Q3"]
    assert (rolling["test_winner_rate_lift"] > 0).all()
    assert "A" in set(usage["feature"])
    assert "B" in set(usage["feature"])
