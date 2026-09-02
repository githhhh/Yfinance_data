from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .metrics import compare_metrics, evaluate_selection
from .optimizer import choose_training_champion, select_all_weeks
from .selectors import ReviewRule, rule_complexity


def run_walk_forward(
    panel: pd.DataFrame,
    rules: list[ReviewRule],
    *,
    min_train_weeks: int = 20,
    test_weeks: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Expanding-window walk-forward with frozen test blocks."""
    weeks = sorted(panel["snapshot_date"].astype(str).unique())
    fold_rows = []
    champion_rows = []
    train_grid_rows = []

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

        champion, train_grid, _ = choose_training_champion(train, rules)
        train_grid = train_grid.copy()
        train_grid.insert(0, "fold", fold)
        train_grid["train_end"] = train_weeks[-1]
        train_grid["test_start"] = test_block[0]
        train_grid["test_end"] = test_block[-1]
        train_grid_rows.append(train_grid)

        baseline_metrics = evaluate_selection(
            test,
            select_all_weeks(test, None),
            variant="B0_ACTIONABLE_ONLY",
        )

        for rule in rules:
            candidate_metrics = evaluate_selection(
                test,
                select_all_weeks(test, rule),
                variant=rule.name,
            )
            row = compare_metrics(candidate_metrics, baseline_metrics)
            row.update(
                {
                    "fold": fold,
                    "train_start": train_weeks[0],
                    "train_end": train_weeks[-1],
                    "test_start": test_block[0],
                    "test_end": test_block[-1],
                    "selected_by_train": (
                        champion is not None and rule.name == champion.name
                    ),
                    "rule_complexity": rule_complexity(rule),
                }
            )
            fold_rows.append(row)

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
            pd.concat(train_grid_rows, ignore_index=True)
            if train_grid_rows
            else pd.DataFrame()
        ),
    )


def summarize_oos_stability(fold_results: pd.DataFrame) -> pd.DataFrame:
    if fold_results.empty:
        return pd.DataFrame()
    rows = []
    for variant, group in fold_results.groupby("variant", sort=True):
        rows.append(
            {
                "variant": variant,
                "folds": int(group["fold"].nunique()),
                "selected_by_train_count": int(group["selected_by_train"].sum()),
                "opportunity_positive_rate": _positive_rate(
                    group["opportunity_recall_1w_delta_vs_b0"]
                ),
                "winner_nonnegative_rate": _nonnegative_rate(
                    group["big_winner_recall_mean_2_4w_delta_vs_b0"]
                ),
                "loser_exclusion_nonworse_rate": _nonnegative_rate(
                    group["big_loser_exclusion_mean_2_4w_delta_vs_b0"]
                ),
                "severe_exposure_nonworse_rate": _nonpositive_rate(
                    group["severe_loser_exposure_mean_2_4w_delta_vs_b0"]
                ),
                "mean_opportunity_delta": _mean(
                    group["opportunity_recall_1w_delta_vs_b0"]
                ),
                "mean_winner_delta_2_4w": _mean(
                    group["big_winner_recall_mean_2_4w_delta_vs_b0"]
                ),
                "mean_loser_exclusion_delta_2_4w": _mean(
                    group["big_loser_exclusion_mean_2_4w_delta_vs_b0"]
                ),
                "mean_severe_exposure_delta_2_4w": _mean(
                    group["severe_loser_exposure_mean_2_4w_delta_vs_b0"]
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
            "winner_nonnegative_rate",
            "loser_exclusion_nonworse_rate",
            "severe_exposure_nonworse_rate",
        ]
    ].min(axis=1)
    return out.sort_values(
        [
            "stability_floor",
            "mean_winner_delta_2_4w",
            "mean_opportunity_delta",
            "mean_loser_exclusion_delta_2_4w",
            "mean_attention_delta",
            "rule_complexity",
            "variant",
        ],
        ascending=[False, False, False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def choose_retrospective_champion(
    stability: pd.DataFrame,
    rules: list[ReviewRule],
) -> ReviewRule | None:
    """Choose only a retrospective candidate, never a production rule."""
    if stability.empty:
        return None
    best = stability.iloc[0]
    if (
        float(best["stability_floor"]) < 0.5
        or float(best["mean_opportunity_delta"]) <= 0.0
        or float(best["mean_winner_delta_2_4w"]) < 0.0
    ):
        return None
    name = str(best["variant"])
    return next((rule for rule in rules if rule.name == name), None)


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
