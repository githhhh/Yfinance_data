from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import compare_metrics, evaluate_selection
from .selectors import (
    ReviewRule,
    rule_complexity,
    select_b0_actionable,
    select_review_variant,
)


def select_all_weeks(panel: pd.DataFrame, rule: ReviewRule | None) -> pd.DataFrame:
    chunks = []
    for snapshot, week in panel.groupby("snapshot_date", sort=True):
        selected = (
            select_b0_actionable(week)
            if rule is None
            else select_review_variant(week, rule)
        )
        if selected.empty:
            continue
        selected = selected.copy()
        selected["snapshot_date"] = str(snapshot)
        chunks.append(selected)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def evaluate_rule_grid(
    panel: pd.DataFrame,
    rules: list[ReviewRule],
) -> pd.DataFrame:
    baseline_selected = select_all_weeks(panel, None)
    baseline_metrics = evaluate_selection(
        panel, baseline_selected, variant="B0_ACTIONABLE_ONLY"
    )
    rows = []
    for rule in rules:
        selected = select_all_weeks(panel, rule)
        metrics = evaluate_selection(panel, selected, variant=rule.name)
        row = compare_metrics(metrics, baseline_metrics)
        row["rule_complexity"] = rule_complexity(rule)
        rows.append(row)
    return pd.DataFrame(rows)


def stability_table(
    panel: pd.DataFrame,
    rules: list[ReviewRule],
    *,
    blocks: int = 3,
) -> pd.DataFrame:
    weeks = sorted(panel["snapshot_date"].astype(str).unique())
    if not weeks:
        return pd.DataFrame()
    week_blocks = [
        list(block)
        for block in np.array_split(
            np.array(weeks, dtype=object), min(blocks, len(weeks))
        )
        if len(block)
    ]
    rows = []
    for rule in rules:
        block_rows = []
        for block_weeks in week_blocks:
            subset = panel[
                panel["snapshot_date"].astype(str).isin(block_weeks)
            ].copy()
            baseline = evaluate_selection(
                subset,
                select_all_weeks(subset, None),
                variant="B0_ACTIONABLE_ONLY",
            )
            candidate = evaluate_selection(
                subset,
                select_all_weeks(subset, rule),
                variant=rule.name,
            )
            block_rows.append(compare_metrics(candidate, baseline))
        frame = pd.DataFrame(block_rows)
        rows.append(
            {
                "variant": rule.name,
                "stability_blocks": len(frame),
                "opportunity_nonnegative_rate": _nonnegative_rate(
                    frame["opportunity_recall_1w_delta_vs_b0"]
                ),
                "winner_nonnegative_rate": _nonnegative_rate(
                    frame["big_winner_recall_mean_2_4w_delta_vs_b0"]
                ),
                "loser_exclusion_nonworse_rate": _nonnegative_rate(
                    frame["big_loser_exclusion_mean_2_4w_delta_vs_b0"]
                ),
                "severe_exposure_nonworse_rate": _nonpositive_rate(
                    frame["severe_loser_exposure_mean_2_4w_delta_vs_b0"]
                ),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["stability_floor"] = out[
            [
                "opportunity_nonnegative_rate",
                "winner_nonnegative_rate",
                "loser_exclusion_nonworse_rate",
                "severe_exposure_nonworse_rate",
            ]
        ].min(axis=1)
    return out


def choose_training_champion(
    panel: pd.DataFrame,
    rules: list[ReviewRule],
) -> tuple[ReviewRule | None, pd.DataFrame, pd.DataFrame]:
    grid = evaluate_rule_grid(panel, rules)
    stable = stability_table(panel, rules)
    merged = grid.merge(stable, on="variant", how="left")
    merged["pareto"] = pareto_mask(merged)
    candidates = merged[merged["pareto"].eq(True)].copy()
    if candidates.empty:
        candidates = merged.copy()
    candidates = candidates.sort_values(
        [
            "stability_floor",
            "big_winner_recall_mean_2_4w_delta_vs_b0",
            "opportunity_recall_1w_delta_vs_b0",
            "big_loser_exclusion_mean_2_4w_delta_vs_b0",
            "avg_watchlist_size_delta_vs_b0",
            "rule_complexity",
            "variant",
        ],
        ascending=[False, False, False, False, True, True, True],
        kind="mergesort",
    )
    if candidates.empty:
        return None, merged, stable

    winner_row = candidates.iloc[0]
    stable_enough = (
        _finite(winner_row.get("stability_floor"))
        and float(winner_row["stability_floor"]) >= (2.0 / 3.0)
        and _finite(winner_row.get("opportunity_recall_1w_delta_vs_b0"))
        and float(winner_row["opportunity_recall_1w_delta_vs_b0"]) > 0.0
        and _finite(winner_row.get("big_winner_recall_mean_2_4w_delta_vs_b0"))
        and float(winner_row["big_winner_recall_mean_2_4w_delta_vs_b0"]) >= 0.0
    )
    if not stable_enough:
        return None, merged, stable

    winner_name = str(winner_row["variant"])
    rule = next((item for item in rules if item.name == winner_name), None)
    return rule, merged, stable


def pareto_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    objectives = [
        ("opportunity_recall_1w_delta_vs_b0", True),
        ("big_winner_recall_mean_2_4w_delta_vs_b0", True),
        ("big_loser_exclusion_mean_2_4w_delta_vs_b0", True),
        ("severe_loser_exposure_mean_2_4w_delta_vs_b0", False),
        ("avg_watchlist_size_delta_vs_b0", False),
    ]
    mask = pd.Series(True, index=frame.index)
    for i in frame.index:
        dominated = False
        for j in frame.index:
            if i == j:
                continue
            no_worse = True
            strictly_better = False
            for column, maximize in objectives:
                a = _finite_value(frame.at[i, column], maximize)
                b = _finite_value(frame.at[j, column], maximize)
                if maximize:
                    if b < a:
                        no_worse = False
                        break
                    strictly_better |= b > a
                else:
                    if b > a:
                        no_worse = False
                        break
                    strictly_better |= b < a
            if no_worse and strictly_better:
                dominated = True
                break
        mask.at[i] = not dominated
    return mask


def _finite_value(value: object, maximize: bool) -> float:
    if _finite(value):
        return float(value)
    return -np.inf if maximize else np.inf


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _nonnegative_rate(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float((numeric >= 0).mean()) if len(numeric) else 0.0


def _nonpositive_rate(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float((numeric <= 0).mean()) if len(numeric) else 0.0
