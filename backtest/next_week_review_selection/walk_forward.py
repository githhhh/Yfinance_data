from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .asof import panel_asof_cutoff, resolved_week_counts
from .metrics import compare_metrics, evaluate_selection
from .optimizer import (
    choose_training_champion,
    rule_evidence_profile,
    rule_structure_key,
    select_all_weeks,
)
from .selectors import ReviewRule, primary_rule, rule_complexity


def partition_walk_forward_weeks(
    weeks: list[str],
    *,
    min_train_weeks: int,
    test_weeks: int,
) -> tuple[list[dict[str, object]], list[str]]:
    """Return only full-size formal test blocks plus a non-formal tail."""
    formal: list[dict[str, object]] = []
    train_end = min_train_weeks
    fold = 0
    while train_end + test_weeks <= len(weeks):
        fold += 1
        formal.append(
            {
                "fold": fold,
                "train_weeks": weeks[:train_end],
                "test_weeks": weeks[train_end : train_end + test_weeks],
            }
        )
        train_end += test_weeks
    return formal, weeks[train_end:]


def run_walk_forward(
    panel: pd.DataFrame,
    core_rules: list[ReviewRule],
    *,
    min_train_weeks: int = 20,
    test_weeks: int = 4,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Horizon-aware OOS with a train-only provisional champion every fold.

    Formal stability uses only full test blocks. Remaining snapshot weeks are
    reported separately as TAIL_EXPLORATORY and never enter the formal verdict.
    """
    weeks = sorted(panel["snapshot_date"].astype(str).unique())
    formal_blocks, tail_weeks = partition_walk_forward_weeks(
        weeks,
        min_train_weeks=min_train_weeks,
        test_weeks=test_weeks,
    )

    fold_rows = []
    champion_rows = []
    core_train_rows = []
    evidence_train_rows = []
    oos_selection_rows = []
    calendar_rows = []

    for block in formal_blocks:
        fold = int(block["fold"])
        train_weeks = list(block["train_weeks"])
        test_block = list(block["test_weeks"])
        cutoff = test_block[0]

        train_raw = panel[
            panel["snapshot_date"].astype(str).isin(train_weeks)
        ].copy()
        train = panel_asof_cutoff(train_raw, cutoff)
        test = panel[
            panel["snapshot_date"].astype(str).isin(test_block)
        ].copy()

        champion, core_diag, evidence_diag = choose_training_champion(
            train, core_rules
        )
        if champion is None:
            raise RuntimeError(
                f"Fold {fold}: no provisional train champion could be selected"
            )

        for stage, diag, sink in (
            ("core", core_diag, core_train_rows),
            ("evidence_ablation", evidence_diag, evidence_train_rows),
        ):
            if not diag.empty:
                tagged = diag.copy()
                tagged.insert(0, "fold", fold)
                tagged.insert(1, "stage", stage)
                tagged["asof_cutoff"] = cutoff
                tagged["train_end"] = train_weeks[-1]
                tagged["test_start"] = test_block[0]
                tagged["test_end"] = test_block[-1]
                sink.append(tagged)

        baseline_selected = select_all_weeks(test, None)
        primary = primary_rule()
        primary_selected = select_all_weeks(test, primary)
        champion_selected = select_all_weeks(test, champion)

        baseline_metrics = evaluate_selection(
            test,
            baseline_selected,
            variant="B0_ACTIONABLE_ONLY",
        )
        for role, rule, selected in (
            ("PRIMARY_R1", primary, primary_selected),
            ("TRAIN_CHAMPION", champion, champion_selected),
        ):
            metrics = compare_metrics(
                evaluate_selection(test, selected, variant=rule.name),
                baseline_metrics,
            )
            metrics.update(
                {
                    "fold": fold,
                    "evaluation_role": role,
                    "formal_oos": True,
                    "asof_cutoff": cutoff,
                    "train_start": train_weeks[0],
                    "train_end": train_weeks[-1],
                    "test_start": test_block[0],
                    "test_end": test_block[-1],
                    "rule_complexity": rule_complexity(rule),
                }
            )
            fold_rows.append(metrics)

        for role, selected in (
            ("B0_ACTIONABLE_ONLY", baseline_selected),
            ("PRIMARY_R1", primary_selected),
            ("TRAIN_CHAMPION", champion_selected),
        ):
            oos_selection_rows.append(
                _selection_projection(
                    selected,
                    fold=fold,
                    role=role,
                    champion_rule=champion.name,
                )
            )

        resolved = resolved_week_counts(train)
        champion_rows.append(
            {
                "fold": fold,
                "formal_oos": True,
                "asof_cutoff": cutoff,
                "train_start": train_weeks[0],
                "train_end": train_weeks[-1],
                "test_start": test_block[0],
                "test_end": test_block[-1],
                "train_snapshot_weeks": len(train_weeks),
                "resolved_1w_train_weeks": resolved["1w"],
                "resolved_2w_train_weeks": resolved["2w"],
                "resolved_3w_train_weeks": resolved["3w"],
                "resolved_4w_train_weeks": resolved["4w"],
                "champion_rule": champion.name,
                "champion_structure": rule_structure_key(champion),
                "champion_evidence_profile": rule_evidence_profile(champion),
                "champion_rule_json": asdict(champion),
            }
        )
        for snapshot in test_block:
            calendar_rows.append(
                {
                    "phase": "FORMAL_OOS",
                    "fold": fold,
                    "snapshot_date": snapshot,
                }
            )

    tail_results, tail_calendar = _run_tail_exploratory(
        panel,
        core_rules,
        weeks=weeks,
        formal_blocks=formal_blocks,
        tail_weeks=tail_weeks,
    )
    calendar_rows.extend(tail_calendar)

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
        tail_results,
        (
            pd.concat(oos_selection_rows, ignore_index=True)
            if oos_selection_rows
            else pd.DataFrame()
        ),
        pd.DataFrame(calendar_rows),
    )


def summarize_oos_stability(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Per-rule OOS view; adaptive-policy aggregation is separate."""
    if fold_results.empty:
        return pd.DataFrame()
    rows = []
    for (role, variant), group in fold_results.groupby(
        ["evaluation_role", "variant"], sort=True
    ):
        rows.append(_stability_row(group, role=role, variant=variant))
    return pd.DataFrame(rows).sort_values(
        [
            "evaluation_role",
            "folds",
            "stability_floor",
            "mean_tradable_winner_lift_delta",
            "mean_tradable_loser_lift_delta",
            "variant",
        ],
        ascending=[True, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def summarize_adaptive_policy(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the train-selected rule as one adaptive OOS policy."""
    chosen = fold_results[
        fold_results["evaluation_role"].eq("TRAIN_CHAMPION")
    ].copy()
    if chosen.empty:
        return pd.DataFrame()
    row = _stability_row(
        chosen,
        role="ADAPTIVE_POLICY",
        variant="TRAIN_SELECTED_EACH_FOLD",
    )
    row["mean_incremental_opportunity_efficiency"] = _mean(
        chosen["incremental_opportunities_per_added_review"]
    )
    row["mean_attention_multiplier_vs_b0"] = _mean(
        chosen["attention_multiplier_vs_b0"]
    )
    row["median_attention_multiplier_vs_b0"] = _median(
        chosen["attention_multiplier_vs_b0"]
    )
    return pd.DataFrame([row])


def adaptive_policy_status(adaptive_summary: pd.DataFrame) -> str:
    if adaptive_summary.empty:
        return "INSUFFICIENT_OOS_HISTORY"
    row = adaptive_summary.iloc[0]
    if int(row["folds"]) < 3:
        return "INSUFFICIENT_OOS_HISTORY"

    required = [
        row["mean_opportunity_delta"],
        row["mean_tradable_winner_lift_delta"],
        row["mean_tradable_loser_lift_delta"],
        row["mean_incremental_opportunity_efficiency"],
    ]
    for horizon in ("2w", "3w", "4w"):
        required.extend(
            [
                row[f"mean_tradable_winner_lift_delta_{horizon}"],
                row[f"mean_tradable_loser_lift_delta_{horizon}"],
            ]
        )
    if not all(_finite(value) for value in required):
        return "NO_STABLE_ADAPTIVE_POLICY"

    horizon_consistent = all(
        float(row[f"mean_tradable_winner_lift_delta_{horizon}"]) >= 0
        and float(row[f"mean_tradable_loser_lift_delta_{horizon}"]) <= 0
        for horizon in ("2w", "3w", "4w")
    )
    stable = (
        float(row["opportunity_positive_rate"]) >= 0.60
        and float(row["tradable_winner_lift_nonnegative_rate"]) >= 0.60
        and float(row["tradable_loser_lift_nonworse_rate"]) >= 0.60
        and float(row["mean_opportunity_delta"]) > 0
        and float(row["mean_tradable_winner_lift_delta"]) >= 0
        and float(row["mean_tradable_loser_lift_delta"]) <= 0
        and float(row["mean_incremental_opportunity_efficiency"]) > 0
        and horizon_consistent
    )
    return (
        "RETROSPECTIVE_ADAPTIVE_CANDIDATE"
        if stable
        else "NO_STABLE_ADAPTIVE_POLICY"
    )


def summarize_rule_convergence(
    champion_rows: pd.DataFrame,
) -> pd.DataFrame:
    if champion_rows.empty:
        return pd.DataFrame()
    folds = int(champion_rows["fold"].nunique())
    rows = []
    for level, column in (
        ("EXACT_RULE", "champion_rule"),
        ("STRUCTURE", "champion_structure"),
        ("EVIDENCE_PROFILE", "champion_evidence_profile"),
    ):
        counts = champion_rows[column].astype(str).value_counts()
        for value, count in counts.items():
            rows.append(
                {
                    "level": level,
                    "value": value,
                    "fold_count": int(count),
                    "formal_folds": folds,
                    "fold_share": float(count) / folds if folds else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["level", "fold_count", "value"],
        ascending=[True, False, True],
        kind="mergesort",
    )


def convergent_static_candidate(
    convergence: pd.DataFrame,
    stability: pd.DataFrame,
) -> tuple[str, str]:
    """Conservative static-rule label based only on repeated train choices + OOS."""
    if convergence.empty or stability.empty:
        return "NO_CONVERGENT_STATIC_RULE", ""
    exact = convergence[
        convergence["level"].eq("EXACT_RULE")
    ].sort_values(["fold_count", "value"], ascending=[False, True])
    if exact.empty or int(exact.iloc[0]["fold_count"]) < 3:
        return "NO_CONVERGENT_STATIC_RULE", ""

    name = str(exact.iloc[0]["value"])
    row = stability[
        stability["evaluation_role"].eq("TRAIN_CHAMPION")
        & stability["variant"].eq(name)
    ]
    if row.empty:
        return "NO_CONVERGENT_STATIC_RULE", ""
    candidate = row.iloc[0]
    horizon_consistent = all(
        _finite(candidate[f"mean_tradable_winner_lift_delta_{horizon}"])
        and float(candidate[f"mean_tradable_winner_lift_delta_{horizon}"]) >= 0
        and _finite(candidate[f"mean_tradable_loser_lift_delta_{horizon}"])
        and float(candidate[f"mean_tradable_loser_lift_delta_{horizon}"]) <= 0
        for horizon in ("2w", "3w", "4w")
    )
    stable = (
        int(candidate["folds"]) >= 3
        and float(candidate["stability_floor"]) >= 0.60
        and _finite(candidate["mean_opportunity_delta"])
        and float(candidate["mean_opportunity_delta"]) > 0
        and _finite(candidate["mean_tradable_winner_lift_delta"])
        and float(candidate["mean_tradable_winner_lift_delta"]) >= 0
        and _finite(candidate["mean_tradable_loser_lift_delta"])
        and float(candidate["mean_tradable_loser_lift_delta"]) <= 0
        and horizon_consistent
    )
    return (
        ("RETROSPECTIVE_CONVERGENT_RULE_CANDIDATE", name)
        if stable
        else ("NO_CONVERGENT_STATIC_RULE", "")
    )


def _run_tail_exploratory(
    panel: pd.DataFrame,
    core_rules: list[ReviewRule],
    *,
    weeks: list[str],
    formal_blocks: list[dict[str, object]],
    tail_weeks: list[str],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if not tail_weeks:
        return pd.DataFrame(), []

    train_end = (
        len(formal_blocks[-1]["train_weeks"])
        + len(formal_blocks[-1]["test_weeks"])
        if formal_blocks
        else min(len(weeks), 20)
    )
    train_weeks = weeks[:train_end]
    cutoff = tail_weeks[0]
    train = panel_asof_cutoff(
        panel[panel["snapshot_date"].astype(str).isin(train_weeks)].copy(),
        cutoff,
    )
    tail = panel[
        panel["snapshot_date"].astype(str).isin(tail_weeks)
    ].copy()
    champion, _, _ = choose_training_champion(train, core_rules)
    if champion is None:
        return pd.DataFrame(), [
            {"phase": "TAIL_EXPLORATORY", "fold": pd.NA, "snapshot_date": week}
            for week in tail_weeks
        ]

    baseline = evaluate_selection(
        tail,
        select_all_weeks(tail, None),
        variant="B0_ACTIONABLE_ONLY",
    )
    rows = []
    for role, rule in (
        ("PRIMARY_R1", primary_rule()),
        ("TRAIN_CHAMPION", champion),
    ):
        metrics = compare_metrics(
            evaluate_selection(
                tail,
                select_all_weeks(tail, rule),
                variant=rule.name,
            ),
            baseline,
        )
        metrics.update(
            {
                "phase": "TAIL_EXPLORATORY",
                "evaluation_role": role,
                "asof_cutoff": cutoff,
                "train_start": train_weeks[0] if train_weeks else "",
                "train_end": train_weeks[-1] if train_weeks else "",
                "test_start": tail_weeks[0],
                "test_end": tail_weeks[-1],
                "tail_week_count": len(tail_weeks),
                "train_champion_rule": champion.name,
            }
        )
        rows.append(metrics)
    calendar = [
        {"phase": "TAIL_EXPLORATORY", "fold": pd.NA, "snapshot_date": week}
        for week in tail_weeks
    ]
    return pd.DataFrame(rows), calendar


def _selection_projection(
    selected: pd.DataFrame,
    *,
    fold: int,
    role: str,
    champion_rule: str,
) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    out = selected.copy()
    out.insert(0, "fold", fold)
    out.insert(1, "evaluation_role", role)
    out.insert(2, "train_champion_rule", champion_rule)
    columns = [
        "fold",
        "evaluation_role",
        "train_champion_rule",
        "snapshot_date",
        "code",
        "variant",
        "selection_source",
        "ibd_entry_status",
        "ibd_candidate_rule",
        "current_vs_ibd_candidate_pct",
        "review_reason",
    ]
    return out[[column for column in columns if column in out.columns]].copy()


def _stability_row(
    group: pd.DataFrame,
    *,
    role: str,
    variant: str,
) -> dict[str, object]:
    row = {
        "evaluation_role": role,
        "variant": variant,
        "folds": int(group["fold"].nunique()),
        "opportunity_positive_rate": _positive_rate(
            group["opportunity_recall_1w_delta_vs_b0"]
        ),
        "tradable_winner_lift_nonnegative_rate": _nonnegative_rate(
            group["tradable_winner_capture_lift_mean_2_4w_delta_vs_b0"]
        ),
        "tradable_loser_lift_nonworse_rate": _nonpositive_rate(
            group["tradable_loser_capture_lift_mean_2_4w_delta_vs_b0"]
        ),
        "mean_opportunity_delta": _mean(
            group["opportunity_recall_1w_delta_vs_b0"]
        ),
        "mean_tradable_winner_lift_delta": _mean(
            group["tradable_winner_capture_lift_mean_2_4w_delta_vs_b0"]
        ),
        "mean_tradable_loser_lift_delta": _mean(
            group["tradable_loser_capture_lift_mean_2_4w_delta_vs_b0"]
        ),
        "mean_incremental_opportunity_efficiency": _mean(
            group["incremental_opportunities_per_added_review"]
        ),
        "mean_attention_delta": _mean(
            group["avg_watchlist_size_delta_vs_b0"]
        ),
        "rule_complexity": int(
            pd.to_numeric(group["rule_complexity"], errors="coerce")
            .dropna()
            .iloc[0]
        ) if "rule_complexity" in group.columns and not pd.to_numeric(
            group["rule_complexity"], errors="coerce"
        ).dropna().empty else pd.NA,
    }
    for horizon in ("2w", "3w", "4w"):
        winner_col = f"tradable_winner_capture_lift_{horizon}_delta_vs_b0"
        loser_col = f"tradable_loser_capture_lift_{horizon}_delta_vs_b0"
        row[f"mean_tradable_winner_lift_delta_{horizon}"] = _mean(
            group[winner_col]
        )
        row[f"mean_tradable_loser_lift_delta_{horizon}"] = _mean(
            group[loser_col]
        )
        row[f"winner_lift_nonnegative_rate_{horizon}"] = _nonnegative_rate(
            group[winner_col]
        )
        row[f"loser_lift_nonworse_rate_{horizon}"] = _nonpositive_rate(
            group[loser_col]
        )

    row["stability_floor"] = min(
        float(row["opportunity_positive_rate"]),
        float(row["tradable_winner_lift_nonnegative_rate"]),
        float(row["tradable_loser_lift_nonworse_rate"]),
    )
    return row


def _mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if len(numeric) else np.nan


def _median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if len(numeric) else np.nan


def _positive_rate(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float((numeric > 0).mean()) if len(numeric) else 0.0


def _nonnegative_rate(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float((numeric >= 0).mean()) if len(numeric) else 0.0


def _nonpositive_rate(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float((numeric <= 0).mean()) if len(numeric) else 0.0


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
