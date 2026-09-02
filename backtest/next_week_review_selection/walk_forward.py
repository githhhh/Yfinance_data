from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .metrics import compare_metrics, evaluate_selection
from .optimizer import choose_training_champion, select_all_weeks
from .selectors import ReviewRule, primary_rule, rule_complexity


def run_walk_forward(
    panel: pd.DataFrame,
    core_rules: list[ReviewRule],
    *,
    min_train_weeks: int = 20,
    test_weeks: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Expanding-window two-stage walk-forward.

    Each fold chooses the structural finalist and evidence ablation on train only,
    freezes it, then evaluates only that champion plus the primary R1 on test.
    """
    weeks = sorted(panel["snapshot_date"].astype(str).unique())
    fold_rows = []
    champion_rows = []
    core_train_rows = []
    evidence_train_rows = []

    fold = 0
    train_end = min_train_weeks
    while train_end < len(weeks):
        test_end = min(train_end + test_weeks, len(weeks))
        train_weeks = weeks[:train_end]
        test_block = weeks[train_end:test_end]
        if not test_block:
            break

        fold += 1
        train = panel[
            panel["snapshot_date"].astype(str).isin(train_weeks)
        ].copy()
        test = panel[
            panel["snapshot_date"].astype(str).isin(test_block)
        ].copy()

        champion, core_diag, evidence_diag = choose_training_champion(
            train, core_rules
        )
        for stage, diag, sink in (
            ("core", core_diag, core_train_rows),
            ("evidence_ablation", evidence_diag, evidence_train_rows),
        ):
            if not diag.empty:
                tagged = diag.copy()
                tagged.insert(0, "fold", fold)
                tagged.insert(1, "stage", stage)
                tagged["train_end"] = train_weeks[-1]
                tagged["test_start"] = test_block[0]
                tagged["test_end"] = test_block[-1]
                sink.append(tagged)

        baseline_metrics = evaluate_selection(
            test,
            select_all_weeks(test, None),
            variant="B0_ACTIONABLE_ONLY",
        )

        primary = primary_rule()
        primary_metrics = compare_metrics(
            evaluate_selection(
                test,
                select_all_weeks(test, primary),
                variant=primary.name,
            ),
            baseline_metrics,
        )
        primary_metrics.update(
            {
                "fold": fold,
                "evaluation_role": "PRIMARY_R1",
                "train_start": train_weeks[0],
                "train_end": train_weeks[-1],
                "test_start": test_block[0],
                "test_end": test_block[-1],
                "rule_complexity": rule_complexity(primary),
            }
        )
        fold_rows.append(primary_metrics)

        if champion is not None:
            champion_metrics = compare_metrics(
                evaluate_selection(
                    test,
                    select_all_weeks(test, champion),
                    variant=champion.name,
                ),
                baseline_metrics,
            )
            champion_metrics.update(
                {
                    "fold": fold,
                    "evaluation_role": "TRAIN_CHAMPION",
                    "train_start": train_weeks[0],
                    "train_end": train_weeks[-1],
                    "test_start": test_block[0],
                    "test_end": test_block[-1],
                    "rule_complexity": rule_complexity(champion),
                }
            )
            fold_rows.append(champion_metrics)

        champion_rows.append(
            {
                "fold": fold,
                "train_start": train_weeks[0],
                "train_end": train_weeks[-1],
                "test_start": test_block[0],
                "test_end": test_block[-1],
                "champion_rule": (
                    champion.name
                    if champion is not None
                    else "NO_STABLE_CANDIDATE"
                ),
                "champion_rule_json": (
                    asdict(champion) if champion is not None else {}
                ),
            }
        )
        train_end = test_end

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(champion_rows),
        (
            pd.concat(core_train_rows, ignore_index=True)
            if core_train_rows
            else pd.DataFrame()
        ),
        (
            pd.concat(evidence_train_rows, ignore_index=True)
            if evidence_train_rows
            else pd.DataFrame()
        ),
    )


def summarize_oos_stability(fold_results: pd.DataFrame) -> pd.DataFrame:
    if fold_results.empty:
        return pd.DataFrame()
    rows = []
    for (role, variant), group in fold_results.groupby(
        ["evaluation_role", "variant"], sort=True
    ):
        rows.append(
            {
                "evaluation_role": role,
                "variant": variant,
                "folds": int(group["fold"].nunique()),
                "opportunity_positive_rate": _positive_rate(
                    group["opportunity_recall_1w_delta_vs_b0"]
                ),
                "tradable_winner_lift_nonnegative_rate": _nonnegative_rate(
                    group[
                        "tradable_winner_capture_lift_mean_2_4w_delta_vs_b0"
                    ]
                ),
                "tradable_loser_lift_nonworse_rate": _nonpositive_rate(
                    group[
                        "tradable_loser_capture_lift_mean_2_4w_delta_vs_b0"
                    ]
                ),
                "mean_opportunity_delta": _mean(
                    group["opportunity_recall_1w_delta_vs_b0"]
                ),
                "mean_tradable_winner_lift_delta": _mean(
                    group[
                        "tradable_winner_capture_lift_mean_2_4w_delta_vs_b0"
                    ]
                ),
                "mean_tradable_loser_lift_delta": _mean(
                    group[
                        "tradable_loser_capture_lift_mean_2_4w_delta_vs_b0"
                    ]
                ),
                "mean_incremental_opportunity_efficiency": _mean(
                    group["incremental_opportunities_per_added_review"]
                ),
                "mean_attention_delta": _mean(
                    group["avg_watchlist_size_delta_vs_b0"]
                ),
                "rule_complexity": int(group["rule_complexity"].iloc[0]),
            }
        )
    out = pd.DataFrame(rows)
    out["stability_floor"] = out[
        [
            "opportunity_positive_rate",
            "tradable_winner_lift_nonnegative_rate",
            "tradable_loser_lift_nonworse_rate",
        ]
    ].min(axis=1)
    return out.sort_values(
        [
            "evaluation_role",
            "stability_floor",
            "mean_tradable_winner_lift_delta",
            "mean_opportunity_delta",
            "mean_tradable_loser_lift_delta",
            "mean_attention_delta",
            "rule_complexity",
            "variant",
        ],
        ascending=[True, False, False, False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def choose_retrospective_champion(
    stability: pd.DataFrame,
    fold_results: pd.DataFrame,
    candidate_rules: list[ReviewRule],
) -> ReviewRule | None:
    """Require the same train-chosen rule to survive >=3 truly OOS folds."""
    if stability.empty or fold_results.empty:
        return None
    champions = stability[
        stability["evaluation_role"].eq("TRAIN_CHAMPION")
        & stability["folds"].ge(3)
    ].copy()
    if champions.empty:
        return None
    champions = champions.sort_values(
        [
            "stability_floor",
            "mean_tradable_winner_lift_delta",
            "mean_opportunity_delta",
            "mean_tradable_loser_lift_delta",
            "mean_attention_delta",
            "rule_complexity",
            "variant",
        ],
        ascending=[False, False, False, True, True, True, True],
        kind="mergesort",
    )
    best = champions.iloc[0]
    required = (
        best["stability_floor"],
        best["mean_opportunity_delta"],
        best["mean_tradable_winner_lift_delta"],
        best["mean_incremental_opportunity_efficiency"],
    )
    if not all(np.isfinite(float(value)) for value in required):
        return None
    if (
        float(best["stability_floor"]) < 0.5
        or float(best["mean_opportunity_delta"]) <= 0
        or float(best["mean_tradable_winner_lift_delta"]) < 0
        or float(best["mean_incremental_opportunity_efficiency"]) <= 0
    ):
        return None
    name = str(best["variant"])
    return next((rule for rule in candidate_rules if rule.name == name), None)


def _mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if len(numeric) else np.nan


def _positive_rate(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float((numeric > 0).mean()) if len(numeric) else 0.0


def _nonnegative_rate(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float((numeric >= 0).mean()) if len(numeric) else 0.0


def _nonpositive_rate(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float((numeric <= 0).mean()) if len(numeric) else 0.0
