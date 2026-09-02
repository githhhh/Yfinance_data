from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import compare_metrics, evaluate_selection
from .search_space import generate_evidence_ablations
from .selectors import ReviewRule, rule_complexity, select_b0_actionable, select_review_variant


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
    baseline = evaluate_selection(
        panel,
        select_all_weeks(panel, None),
        variant="B0_ACTIONABLE_ONLY",
    )
    rows = []
    for rule in rules:
        candidate = evaluate_selection(
            panel,
            select_all_weeks(panel, rule),
            variant=rule.name,
        )
        row = compare_metrics(candidate, baseline)
        row["rule_complexity"] = rule_complexity(rule)
        rows.append(row)
    return pd.DataFrame(rows)


def stability_table(
    panel: pd.DataFrame,
    rules: list[ReviewRule],
    *,
    blocks: int = 3,
) -> pd.DataFrame:
    """Direction stability across chronological training blocks."""
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
                "opportunity_positive_rate": _positive_rate(
                    frame["opportunity_recall_1w_delta_vs_b0"]
                ),
                "tradable_winner_lift_nonnegative_rate": _nonnegative_rate(
                    frame[
                        "tradable_winner_capture_lift_mean_2_4w_delta_vs_b0"
                    ]
                ),
                "tradable_loser_lift_nonworse_rate": _nonpositive_rate(
                    frame[
                        "tradable_loser_capture_lift_mean_2_4w_delta_vs_b0"
                    ]
                ),
                "incremental_efficiency_positive_rate": _positive_rate(
                    frame["incremental_opportunities_per_added_review"]
                ),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["stability_floor"] = out[
            [
                "opportunity_positive_rate",
                "tradable_winner_lift_nonnegative_rate",
                "tradable_loser_lift_nonworse_rate",
                "incremental_efficiency_positive_rate",
            ]
        ].min(axis=1)
    return out


def two_stage_diagnostics(
    panel: pd.DataFrame,
    core_rules: list[ReviewRule],
    *,
    max_finalists: int = 4,
) -> tuple[list[ReviewRule], pd.DataFrame, pd.DataFrame]:
    core_diag = _diagnostic_frame(panel, core_rules)
    finalists = _top_finalists(core_diag, core_rules, max_finalists=max_finalists)

    expanded: dict[str, ReviewRule] = {}
    for core_rule in finalists:
        for rule in generate_evidence_ablations(core_rule):
            expanded[rule.name] = rule
    evidence_rules = list(expanded.values())
    evidence_diag = (
        _diagnostic_frame(panel, evidence_rules)
        if evidence_rules
        else pd.DataFrame()
    )
    return evidence_rules, core_diag, evidence_diag


def choose_training_champion(
    panel: pd.DataFrame,
    core_rules: list[ReviewRule],
) -> tuple[ReviewRule | None, pd.DataFrame, pd.DataFrame]:
    """Two-stage train-only selection; no test data is touched."""
    evidence_rules, core_diag, evidence_diag = two_stage_diagnostics(
        panel, core_rules
    )
    if evidence_diag.empty:
        return None, core_diag, evidence_diag

    ranked = _rank_candidates(evidence_diag)
    if ranked.empty:
        return None, core_diag, evidence_diag

    best = ranked.iloc[0]
    required = [
        best.get("stability_floor"),
        best.get("opportunity_recall_1w_delta_vs_b0"),
        best.get("tradable_winner_capture_lift_mean_2_4w_delta_vs_b0"),
        best.get("incremental_opportunities_per_added_review"),
    ]
    if not all(_finite(value) for value in required):
        return None, core_diag, evidence_diag

    stable_enough = (
        float(best["stability_floor"]) >= (2.0 / 3.0)
        and float(best["opportunity_recall_1w_delta_vs_b0"]) > 0.0
        and float(
            best["tradable_winner_capture_lift_mean_2_4w_delta_vs_b0"]
        )
        >= 0.0
        and float(best["incremental_opportunities_per_added_review"]) > 0.0
    )
    if not stable_enough:
        return None, core_diag, evidence_diag

    name = str(best["variant"])
    return (
        next((rule for rule in evidence_rules if rule.name == name), None),
        core_diag,
        evidence_diag,
    )


def pareto_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    objectives = [
        ("opportunity_recall_1w_delta_vs_b0", True),
        ("tradable_winner_capture_lift_mean_2_4w_delta_vs_b0", True),
        ("tradable_loser_capture_lift_mean_2_4w_delta_vs_b0", False),
        ("incremental_opportunities_per_added_review", True),
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


def _diagnostic_frame(
    panel: pd.DataFrame,
    rules: list[ReviewRule],
) -> pd.DataFrame:
    grid = evaluate_rule_grid(panel, rules)
    stable = stability_table(panel, rules)
    merged = grid.merge(stable, on="variant", how="left")
    merged["pareto"] = pareto_mask(merged)
    return merged


def _top_finalists(
    diag: pd.DataFrame,
    rules: list[ReviewRule],
    *,
    max_finalists: int,
) -> list[ReviewRule]:
    if diag.empty:
        return []
    candidates = diag[diag["pareto"].eq(True)].copy()
    if candidates.empty:
        candidates = diag.copy()
    candidates = _rank_candidates(candidates).head(max_finalists)
    names = set(candidates["variant"].astype(str))
    return [rule for rule in rules if rule.name in names]


def _rank_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.sort_values(
        [
            "stability_floor",
            "tradable_winner_capture_lift_mean_2_4w_delta_vs_b0",
            "opportunity_recall_1w_delta_vs_b0",
            "tradable_loser_capture_lift_mean_2_4w_delta_vs_b0",
            "incremental_opportunities_per_added_review",
            "avg_watchlist_size_delta_vs_b0",
            "rule_complexity",
            "variant",
        ],
        ascending=[False, False, False, True, False, True, True, True],
        kind="mergesort",
    )


def _finite_value(value: object, maximize: bool) -> float:
    if _finite(value):
        return float(value)
    return -np.inf if maximize else np.inf


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _positive_rate(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float((numeric > 0).mean()) if len(numeric) else 0.0


def _nonnegative_rate(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float((numeric >= 0).mean()) if len(numeric) else 0.0


def _nonpositive_rate(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float((numeric <= 0).mean()) if len(numeric) else 0.0
