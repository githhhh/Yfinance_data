from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.blind_rule_discovery.retrospective_ceiling import (
    _conditions_compatible,
    _write_best_rule,
    generate_conditions,
    pareto_frontier,
    rolling_walk_forward,
    search_rules,
)


def coupled_history(quarters: int = 8) -> pd.DataFrame:
    """A and B are weak alone; A>=0.9 AND B>=0.9 is the stable winner pocket."""
    rows = []
    for q_index in range(quarters):
        year = 2023 + q_index // 4
        quarter = q_index % 4 + 1
        period = f"{year}Q{quarter}"
        # 10 clean coupled winners.
        for _ in range(10):
            rows.append((period, "winner", 0.9, 0.9, 0.16, -0.03, 0.28))
        # High A alone is not enough.
        for _ in range(10):
            rows.append((period, "loser", 0.9, 0.1, -0.08, -0.11, 0.06))
        # High B alone is not enough.
        for _ in range(10):
            rows.append((period, "loser", 0.1, 0.9, -0.07, -0.10, 0.07))
        # Broad low/low universe: mostly losers, a few winners.
        for i in range(30):
            primary = "winner" if i < 5 else "loser"
            excess = 0.05 if primary == "winner" else -0.05
            rows.append((period, primary, 0.1, 0.1, excess, -0.08, 0.10))
    frame = pd.DataFrame(
        rows,
        columns=[
            "period_quarter",
            "Y_primary",
            "A",
            "B",
            "Y_12w_excess",
            "Y_mae_12w",
            "Y_mfe_12w",
        ],
    )
    return frame


def test_generate_conditions_has_both_directions_without_tautologies():
    frame = coupled_history(quarters=2)
    conditions = generate_conditions(frame, ["A"], quantiles=(0.25, 0.5, 0.75))
    keys = {(c["op"], c["threshold"]) for c in conditions}
    assert (">=", 0.9) in keys
    assert ("<=", 0.1) in keys
    assert (">=", 0.1) not in keys
    assert ("<=", 0.9) not in keys


def test_same_feature_interval_requires_opposite_compatible_bounds():
    assert _conditions_compatible(
        [
            {"feature": "A", "op": ">=", "threshold": 0.2},
            {"feature": "A", "op": "<=", "threshold": 0.8},
        ]
    )
    assert not _conditions_compatible(
        [
            {"feature": "A", "op": ">=", "threshold": 0.8},
            {"feature": "A", "op": "<=", "threshold": 0.2},
        ]
    )
    assert not _conditions_compatible(
        [
            {"feature": "A", "op": ">=", "threshold": 0.2},
            {"feature": "A", "op": ">=", "threshold": 0.8},
        ]
    )


def test_search_finds_coupled_rule_better_than_single_feature_rules():
    frame = coupled_history()
    scored, pareto, _ = search_rules(
        frame,
        ["A", "B"],
        quantiles=(0.25, 0.5, 0.75),
        beam_width=10,
        max_conditions=2,
        max_clauses=1,
        dnf_clause_pool=0,
        min_selected=20,
        min_resolved=20,
        min_active_quarters=4,
        min_resolved_per_quarter=3,
    )
    assert not scored.empty
    assert not pareto.empty

    two_condition = scored.loc[scored["condition_count"] == 2]
    single = scored.loc[scored["condition_count"] == 1]
    assert not two_condition.empty and not single.empty
    assert two_condition["resolved_winner_rate"].max() > single["resolved_winner_rate"].max()

    best_two = two_condition.sort_values("resolved_winner_rate", ascending=False).iloc[0]
    rule = json.loads(best_two["rule_json"])
    conditions = rule["clauses"][0]["all"]
    assert {c["feature"] for c in conditions} == {"A", "B"}
    assert best_two["resolved_winner_rate"] == 1.0
    assert best_two["winner_rate_lift"] > 0.5


def test_pareto_frontier_drops_strictly_dominated_rule():
    rows = pd.DataFrame(
        [
            {
                "rule_json": "strong",
                "robust_score": 0.9,
                "winner_rate_lift": 0.10,
                "excess_12w_p50": 0.10,
                "excess_12w_p25": 0.02,
                "mae_12w_p50": -0.05,
                "quarter_outperform_fraction": 0.8,
                "median_quarter_lift": 0.08,
                "coverage": 0.20,
            },
            {
                "rule_json": "dominated",
                "robust_score": 0.4,
                "winner_rate_lift": 0.05,
                "excess_12w_p50": 0.04,
                "excess_12w_p25": -0.02,
                "mae_12w_p50": -0.08,
                "quarter_outperform_fraction": 0.6,
                "median_quarter_lift": 0.04,
                "coverage": 0.10,
            },
        ]
    )
    frontier = pareto_frontier(rows)
    assert frontier["rule_json"].tolist() == ["strong"]


def test_rolling_walk_forward_researches_past_and_tests_next_quarter():
    frame = coupled_history(quarters=8)
    rolling, usage = rolling_walk_forward(
        frame,
        ["A", "B"],
        min_train_quarters=4,
        beam_width=10,
        min_selected=20,
        min_resolved=15,
        min_resolved_per_quarter=3,
    )
    assert len(rolling) == 4
    assert rolling["test_quarter"].tolist() == ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]
    assert rolling["train_end_quarter"].tolist() == ["2023Q4", "2024Q1", "2024Q2", "2024Q3"]
    assert (rolling["test_winner_rate_lift"] > 0).all()
    assert (rolling["test_resolved_winner_rate"] == 1.0).all()
    assert set(usage["feature"]) == {"A", "B"}


def test_best_rule_artifact_is_explicitly_non_holdout(tmp_path: Path):
    frame = coupled_history()
    scored, _, _ = search_rules(
        frame,
        ["A", "B"],
        quantiles=(0.25, 0.5, 0.75),
        beam_width=10,
        max_conditions=2,
        max_clauses=1,
        dnf_clause_pool=0,
        min_selected=20,
        min_resolved=20,
        min_active_quarters=4,
        min_resolved_per_quarter=3,
    )
    path = tmp_path / "best_rule.json"
    _write_best_rule(path, scored.iloc[0])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["research_mode"] == "retrospective_empirical_ceiling"
    assert payload["not_unseen_holdout"] is True
    assert payload["rule"]["version"] == 1
