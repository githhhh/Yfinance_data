from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from backtest.blind_rule_discovery.retrospective_ceiling import (
    _conditions_compatible,
    generate_conditions,
)
from backtest.blind_rule_discovery.retrospective_ceiling_exhaustive import (
    EvaluationContext,
    _supported,
    exhaustive_pair_search_rules,
    leave_one_quarter_out_research,
    rolling_walk_forward_research,
    summarize_rolling,
)


def interaction_history(quarters: int = 8) -> pd.DataFrame:
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
            for _ in range(10):
                rows.append(
                    {
                        "period_quarter": period,
                        "Y_primary": primary,
                        "A": a,
                        "B": b,
                        "Y_12w_excess": 0.18 if primary == "winner" else -0.06,
                        "Y_mae_12w": -0.03 if primary == "winner" else -0.10,
                        "Y_mfe_12w": 0.30 if primary == "winner" else 0.08,
                    }
                )
    return pd.DataFrame(rows)


def test_r3_exhausts_every_compatible_generated_pair():
    frame = interaction_history()
    quantiles = (0.25, 0.5, 0.75)
    conditions = generate_conditions(frame, ["A", "B"], quantiles=quantiles)
    expected = sum(
        1
        for left, right in itertools.combinations(conditions, 2)
        if _conditions_compatible([left, right])
    )
    scored, _, audit = exhaustive_pair_search_rules(
        frame,
        ["A", "B"],
        quantiles=quantiles,
        beam_width=1,
        max_conditions=2,
        max_clauses=1,
        dnf_clause_pool=0,
        min_selected=20,
        min_resolved=20,
        min_active_quarters=4,
        min_evaluated_quarters=4,
        min_evaluated_fraction=0.5,
        min_resolved_per_quarter=3,
    )
    assert audit["exact_compatible_pair_count"] == expected
    assert audit["supported_pair_count"] > 0
    assert (scored["condition_count"] == 2).any()
    assert scored["winner_rate_lift"].max() == 0.5


def test_supported_quarter_gate_rejects_active_but_unevaluable_rule():
    rows = []
    for q in range(6):
        quarter = f"2024Q{q % 4 + 1}-{q}"
        selected_in_quarter = 5 if q < 2 else 1
        for i in range(8):
            rows.append(
                {
                    "period_quarter": quarter,
                    "Y_primary": "winner" if i % 2 == 0 else "loser",
                    "Y_12w_excess": 0.05,
                    "Y_mae_12w": -0.05,
                    "Y_mfe_12w": 0.10,
                    "A": 1.0 if i < selected_in_quarter else 0.0,
                }
            )
    frame = pd.DataFrame(rows)
    context = EvaluationContext.from_frame(frame)
    mask = frame["A"].to_numpy() >= 1.0
    metrics = context.evaluate(mask, min_resolved_per_quarter=3)
    assert metrics["active_quarters"] == 6
    assert metrics["evaluated_quarters"] == 2
    assert metrics["evaluated_quarter_fraction"] == 2 / 6
    assert not _supported(
        metrics,
        min_selected=10,
        min_resolved=10,
        min_active_quarters=5,
        min_evaluated_quarters=4,
        min_evaluated_fraction=0.5,
    )


def test_true_loqo_research_excludes_held_quarter_and_tests_it():
    frame = interaction_history()
    result = leave_one_quarter_out_research(
        frame,
        ["A", "B"],
        beam_width=2,
        min_selected=20,
        min_resolved=20,
        min_resolved_per_quarter=3,
    )
    assert len(result) == 8
    assert not result["held_quarter_in_train"].any()
    assert (result["train_quarter_count"] == 7).all()
    assert (result["test_winner_rate_lift"] > 0).all()


def test_rolling_remains_past_only_with_exact_pair_search():
    frame = interaction_history()
    rolling, usage = rolling_walk_forward_research(
        frame,
        ["A", "B"],
        min_train_quarters=4,
        beam_width=2,
        min_selected=20,
        min_resolved=15,
        min_resolved_per_quarter=3,
    )
    assert rolling["test_quarter"].tolist() == ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]
    assert rolling["train_end_quarter"].tolist() == ["2023Q4", "2024Q1", "2024Q2", "2024Q3"]
    assert not rolling["test_quarter_in_train"].any()
    assert (rolling["test_winner_rate_lift"] > 0).all()
    assert {"A", "B"}.intersection(set(usage["feature"]))


def test_rolling_summary_counts_zero_selection_folds_and_pooled_baseline():
    rolling = pd.DataFrame(
        [
            {
                "test_selected_n": 10,
                "test_resolved_n": 8,
                "test_winner_n": 4,
                "test_universe_resolved_n": 20,
                "test_universe_winner_n": 6,
                "test_winner_rate_lift": 0.20,
                "test_excess_12w_p50": 0.05,
            },
            {
                "test_selected_n": 0,
                "test_resolved_n": 0,
                "test_winner_n": 0,
                "test_universe_resolved_n": 10,
                "test_universe_winner_n": 4,
                "test_winner_rate_lift": np.nan,
                "test_excess_12w_p50": np.nan,
            },
            {
                "test_selected_n": 6,
                "test_resolved_n": 5,
                "test_winner_n": 1,
                "test_universe_resolved_n": 10,
                "test_universe_winner_n": 3,
                "test_winner_rate_lift": -0.10,
                "test_excess_12w_p50": -0.02,
            },
        ]
    )
    summary = summarize_rolling(rolling)
    assert summary["folds"] == 3
    assert summary["evaluable_lift_folds"] == 2
    assert summary["zero_selection_fold_count"] == 1
    assert summary["zero_selection_fold_fraction"] == 1 / 3
    assert summary["positive_lift_evaluable_fraction"] == 0.5
    assert summary["positive_lift_all_fold_fraction"] == 1 / 3
    assert summary["pooled_resolved_winner_rate"] == 5 / 13
    assert summary["pooled_universe_resolved_winner_rate"] == 13 / 40
    assert summary["pooled_winner_rate_lift"] == 5 / 13 - 13 / 40
