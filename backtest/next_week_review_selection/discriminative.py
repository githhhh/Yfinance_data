from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .asof import panel_asof_cutoff
from .metrics import compare_metrics, evaluate_selection
from .selectors import (
    EVIDENCE_FAMILIES,
    PULLBACK_RULES,
    ReviewRule,
    select_b0_actionable,
    select_supplemental,
)
from .utils import (
    ZERO_TOL,
    is_nonnegative,
    is_nonpositive,
    is_positive,
    to_bool,
    to_float,
)
from .walk_forward import partition_walk_forward_weeks


B0_NAME = "B0_NO_EXPANSION"
MAX_ATTENTION_MULTIPLIER = 1.50
FORMAL_HORIZONS = ("2w", "3w", "4w")


@dataclass(frozen=True)
class RefinementRule:
    name: str
    conditions: tuple[str, ...]


def anchor_rule() -> ReviewRule:
    """The v0.5 structure that converged in 5/5 train folds.

    This is an anchor for v0.6 discrimination, not a production rule.
    """
    return ReviewRule(
        name="V06_ANCHOR_NEAR5_UB_E2_GA",
        near_below_pct=5.0,
        supplemental_statuses=("UNCONFIRMED", "BELOW_TRIGGER"),
        min_evidence_families=2,
        exclude_clear_geometry_failure=False,
        enabled_evidence_families=EVIDENCE_FAMILIES,
    )


def candidate_library() -> list[RefinementRule]:
    """Small, pre-defined, interpretable refinements.

    Thresholds are coarse domain buckets. No outcome-derived decimal thresholds,
    ML, C Rank, ATR, or automatic feature generation are allowed.
    """
    specs = [
        ("VS_ABS_LE_2", ("VS_ABS_LE_2",)),
        ("VS_ABS_LE_3", ("VS_ABS_LE_3",)),
        ("BASE_DEPTH_LE_33", ("BASE_DEPTH_LE_33",)),
        ("BASE_DEPTH_LE_50", ("BASE_DEPTH_LE_50",)),
        ("BASE_DURATION_7_65", ("BASE_DURATION_7_65",)),
        ("PULLBACK_DEPTH_LE_15", ("PULLBACK_DEPTH_LE_15",)),
        ("PULLBACK_DURATION_3_10", ("PULLBACK_DURATION_3_10",)),
        ("SETUP_PULLBACK_LIKE", ("SETUP_PULLBACK_LIKE",)),
        ("RS_WITHIN_10", ("RS_WITHIN_10",)),
        ("REQ_VOLUME", ("REQ_VOLUME",)),
        ("REQ_EPS", ("REQ_EPS",)),
        ("REQ_RS", ("REQ_RS",)),
        ("REQ_SUPPLY", ("REQ_SUPPLY",)),
        ("VS3_REQ_RS", ("VS_ABS_LE_3", "REQ_RS")),
        ("VS3_REQ_VOLUME", ("VS_ABS_LE_3", "REQ_VOLUME")),
        ("VS3_BASE50", ("VS_ABS_LE_3", "BASE_DEPTH_LE_50")),
        ("BASE50_DURATION", ("BASE_DEPTH_LE_50", "BASE_DURATION_7_65")),
        ("BASE50_REQ_RS", ("BASE_DEPTH_LE_50", "REQ_RS")),
        ("DURATION_REQ_RS", ("BASE_DURATION_7_65", "REQ_RS")),
        ("PB15_REQ_SUPPLY", ("PULLBACK_DEPTH_LE_15", "REQ_SUPPLY")),
        ("PULLBACK_SETUP_REQ_RS", ("SETUP_PULLBACK_LIKE", "REQ_RS")),
        ("PULLBACK_SETUP_REQ_SUPPLY", ("SETUP_PULLBACK_LIKE", "REQ_SUPPLY")),
        ("REQ_VOLUME_RS", ("REQ_VOLUME", "REQ_RS")),
        ("REQ_VOLUME_SUPPLY", ("REQ_VOLUME", "REQ_SUPPLY")),
        ("REQ_RS_SUPPLY", ("REQ_RS", "REQ_SUPPLY")),
    ]
    return [
        RefinementRule(name=f"V06_{name}", conditions=conditions)
        for name, conditions in specs
    ]


def select_refined(pool: pd.DataFrame, rule: RefinementRule | None) -> pd.DataFrame:
    """B0 ACTIONABLE + anchor supplemental candidates that pass refinement."""
    baseline = select_b0_actionable(pool)
    if rule is None:
        baseline = baseline.copy()
        baseline["variant"] = B0_NAME
        return baseline.reset_index(drop=True)

    supplemental = select_supplemental(pool, anchor_rule())
    if not supplemental.empty:
        mask = supplemental.apply(
            lambda row: all(_condition_passes(row, key) for key in rule.conditions),
            axis=1,
        )
        supplemental = supplemental.loc[mask].copy()
        supplemental["selection_source"] = "SUPPLEMENTAL_REFINED"
        supplemental["variant"] = rule.name
        supplemental["review_reason"] = supplemental["review_reason"].astype(str).map(
            lambda reason: f"{reason}|refine:{'+'.join(rule.conditions)}"
        )

    if baseline.empty:
        combined = supplemental.copy()
    elif supplemental.empty:
        combined = baseline.copy()
    else:
        combined = pd.concat([baseline, supplemental], ignore_index=True, sort=False)

    if combined.empty:
        return combined
    combined["variant"] = rule.name
    return (
        combined.sort_values("_source_row_order", kind="mergesort")
        .drop_duplicates(["snapshot_date", "code"], keep="first")
        .reset_index(drop=True)
    )


def select_refined_all_weeks(
    panel: pd.DataFrame,
    rule: RefinementRule | None,
) -> pd.DataFrame:
    chunks = []
    for snapshot, week in panel.groupby("snapshot_date", sort=True):
        selected = select_refined(week, rule)
        if selected.empty:
            continue
        selected = selected.copy()
        selected["snapshot_date"] = str(snapshot)
        chunks.append(selected)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def discovery_bucket_stats(discovery_panel: pd.DataFrame) -> pd.DataFrame:
    cohort = select_supplemental(discovery_panel, anchor_rule())
    if cohort.empty:
        return pd.DataFrame()

    work = _quality_labels(cohort)
    feature_buckets = _bucket_frame(work)
    rows = []
    for feature in feature_buckets.columns:
        values = feature_buckets[feature]
        for bucket, index in values.groupby(values, dropna=False, sort=True).groups.items():
            group = work.loc[index]
            rows.append(
                _bucket_summary(
                    group,
                    feature,
                    str(bucket),
                    total_rows=len(work),
                )
            )
    return pd.DataFrame(rows)


def discovery_interaction_stats(discovery_panel: pd.DataFrame) -> pd.DataFrame:
    cohort = select_supplemental(discovery_panel, anchor_rule())
    if cohort.empty:
        return pd.DataFrame()

    work = _quality_labels(cohort)
    buckets = _bucket_frame(work)
    pairs = (
        ("setup", "vs_buy_point"),
        ("setup", "base_depth"),
        ("vs_buy_point", "rs_distance"),
        ("base_depth", "dry_state"),
        ("pullback_depth", "dry_state"),
    )
    rows = []
    for left, right in pairs:
        keys = pd.DataFrame(
            {
                "left": buckets[left].astype(str),
                "right": buckets[right].astype(str),
            },
            index=work.index,
        )
        for (left_value, right_value), index in keys.groupby(
            ["left", "right"], sort=True
        ).groups.items():
            group = work.loc[index]
            row = _bucket_summary(
                group,
                f"{left} x {right}",
                f"{left_value} | {right_value}",
                total_rows=len(work),
            )
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_candidate_grid(
    train_panel: pd.DataFrame,
    rules: list[RefinementRule] | None = None,
) -> pd.DataFrame:
    rules = rules or candidate_library()
    baseline_selected = select_refined_all_weeks(train_panel, None)
    baseline = evaluate_selection(
        train_panel,
        baseline_selected,
        variant=B0_NAME,
    )

    rows = []
    for rule in rules:
        selected = select_refined_all_weeks(train_panel, rule)
        candidate = compare_metrics(
            evaluate_selection(train_panel, selected, variant=rule.name),
            baseline,
        )
        candidate["conditions"] = "+".join(rule.conditions)
        candidate["condition_count"] = len(rule.conditions)
        candidate["selection_signature"] = _selection_signature(selected)
        candidate["horizon_consistency_count"] = _horizon_consistency(candidate)
        block_count, block_rate = _train_block_consistency(train_panel, rule)
        candidate["train_consistent_blocks"] = block_count
        candidate["train_consistent_block_rate"] = block_rate
        rows.append(candidate)

    grid = pd.DataFrame(rows)
    if grid.empty:
        return grid

    representatives = {}
    group_sizes = {}
    for signature, group in grid.groupby("selection_signature", sort=True):
        names = list(group["variant"].astype(str))
        names.sort(
            key=lambda name: (
                int(grid.loc[grid["variant"].eq(name), "condition_count"].iloc[0]),
                name,
            )
        )
        representatives[str(signature)] = names[0]
        group_sizes[str(signature)] = len(names)

    grid["signature_group_size"] = grid["selection_signature"].map(group_sizes)
    grid["signature_representative_name"] = grid["selection_signature"].map(
        representatives
    )
    grid["signature_representative"] = (
        grid["variant"].astype(str)
        == grid["signature_representative_name"].astype(str)
    )
    grid["attention_cap_pass"] = pd.to_numeric(
        grid["attention_multiplier_vs_b0"], errors="coerce"
    ).le(MAX_ATTENTION_MULTIPLIER + ZERO_TOL)
    grid["train_quality_gate_pass"] = grid.apply(_train_quality_gate, axis=1)
    grid["feasible"] = (
        grid["signature_representative"]
        & grid["attention_cap_pass"]
        & grid["train_quality_gate_pass"]
    )
    grid["pareto"] = False
    feasible = grid.loc[grid["feasible"]].copy()
    if not feasible.empty:
        grid.loc[feasible.index, "pareto"] = _pareto_mask(feasible)
    return grid


def choose_train_refinement(
    train_panel: pd.DataFrame,
    rules: list[RefinementRule] | None = None,
) -> tuple[RefinementRule | None, pd.DataFrame]:
    rules = rules or candidate_library()
    grid = evaluate_candidate_grid(train_panel, rules)
    if grid.empty:
        return None, grid

    feasible = grid.loc[grid["feasible"]].copy()
    if feasible.empty:
        return None, grid

    frontier = feasible.loc[feasible["pareto"]].copy()
    ranked = _rank_grid(frontier if not frontier.empty else feasible)
    if ranked.empty:
        return None, grid

    name = str(ranked.iloc[0]["variant"])
    rule_map = {rule.name: rule for rule in rules}
    return rule_map.get(name), grid


def run_discriminative_walk_forward(
    panel: pd.DataFrame,
    *,
    min_train_weeks: int = 20,
    test_weeks: int = 4,
) -> dict[str, pd.DataFrame | RefinementRule | None]:
    weeks = sorted(panel["snapshot_date"].astype(str).unique())
    formal_blocks, tail_weeks = partition_walk_forward_weeks(
        weeks,
        min_train_weeks=min_train_weeks,
        test_weeks=test_weeks,
    )
    if not formal_blocks:
        raise ValueError("No formal OOS blocks available")

    library = candidate_library()
    first = formal_blocks[0]
    discovery_cutoff = list(first["test_weeks"])[0]
    discovery_train = panel_asof_cutoff(
        panel[
            panel["snapshot_date"].astype(str).isin(list(first["train_weeks"]))
        ].copy(),
        discovery_cutoff,
    )
    static_rule, discovery_grid = choose_train_refinement(
        discovery_train, library
    )

    fold_rows = []
    choice_rows = []
    train_grids = []
    selection_rows = []
    formal_calendar_rows = []

    for block in formal_blocks:
        fold = int(block["fold"])
        train_weeks = list(block["train_weeks"])
        test_block = list(block["test_weeks"])
        cutoff = test_block[0]
        train = panel_asof_cutoff(
            panel[panel["snapshot_date"].astype(str).isin(train_weeks)].copy(),
            cutoff,
        )
        test = panel[
            panel["snapshot_date"].astype(str).isin(test_block)
        ].copy()

        adaptive_rule, grid = choose_train_refinement(train, library)
        if not grid.empty:
            tagged = grid.copy()
            tagged.insert(0, "fold", fold)
            tagged.insert(1, "asof_cutoff", cutoff)
            train_grids.append(tagged)

        baseline_selected = select_refined_all_weeks(test, None)
        baseline = evaluate_selection(
            test,
            baseline_selected,
            variant=B0_NAME,
        )
        variants = (
            ("STATIC_DISCOVERY_RULE", static_rule),
            ("ADAPTIVE_DISCRIMINATIVE_POLICY", adaptive_rule),
        )
        for role, rule in variants:
            selected = select_refined_all_weeks(test, rule)
            metrics = compare_metrics(
                evaluate_selection(
                    test,
                    selected,
                    variant=rule.name if rule is not None else B0_NAME,
                ),
                baseline,
            )
            metrics.update(
                {
                    "fold": fold,
                    "evaluation_role": role,
                    "selected_rule": rule.name if rule is not None else B0_NAME,
                    "formal_oos": True,
                    "asof_cutoff": cutoff,
                    "train_start": train_weeks[0],
                    "train_end": train_weeks[-1],
                    "test_start": test_block[0],
                    "test_end": test_block[-1],
                    "condition_count": len(rule.conditions) if rule is not None else 0,
                }
            )
            fold_rows.append(metrics)
            if not selected.empty:
                projected = selected.copy()
                projected.insert(0, "fold", fold)
                projected.insert(1, "evaluation_role", role)
                projected.insert(
                    2,
                    "selected_rule",
                    rule.name if rule is not None else B0_NAME,
                )
                selection_rows.append(projected)

        for snapshot in test_block:
            formal_calendar_rows.append(
                {
                    "fold": fold,
                    "snapshot_date": snapshot,
                    "phase": "FORMAL_OOS",
                }
            )

        choice_rows.append(
            {
                "fold": fold,
                "asof_cutoff": cutoff,
                "train_start": train_weeks[0],
                "train_end": train_weeks[-1],
                "test_start": test_block[0],
                "test_end": test_block[-1],
                "static_rule": static_rule.name if static_rule is not None else B0_NAME,
                "adaptive_rule": adaptive_rule.name if adaptive_rule is not None else B0_NAME,
                "adaptive_conditions": (
                    "+".join(adaptive_rule.conditions)
                    if adaptive_rule is not None
                    else ""
                ),
            }
        )

    tail = _tail_exploratory(
        panel,
        weeks=weeks,
        formal_blocks=formal_blocks,
        tail_weeks=tail_weeks,
        static_rule=static_rule,
        library=library,
    )

    return {
        "discovery_panel": discovery_train,
        "discovery_grid": discovery_grid,
        "discovery_buckets": discovery_bucket_stats(discovery_train),
        "discovery_interactions": discovery_interaction_stats(discovery_train),
        "static_rule": static_rule,
        "fold_results": pd.DataFrame(fold_rows),
        "fold_choices": pd.DataFrame(choice_rows),
        "train_candidate_grids": (
            pd.concat(train_grids, ignore_index=True)
            if train_grids
            else pd.DataFrame()
        ),
        "formal_selections": (
            pd.concat(selection_rows, ignore_index=True)
            if selection_rows
            else pd.DataFrame()
        ),
        "formal_calendar": pd.DataFrame(formal_calendar_rows),
        "adaptive_convergence": _adaptive_convergence(
            pd.DataFrame(choice_rows)
        ),
        "tail_exploratory": tail,
    }


def summarize_oos_policy(
    fold_results: pd.DataFrame,
    role: str,
) -> pd.DataFrame:
    group = fold_results.loc[
        fold_results["evaluation_role"].eq(role)
    ].copy()
    if group.empty:
        return pd.DataFrame()

    row: dict[str, object] = {
        "evaluation_role": role,
        "folds": int(group["fold"].nunique()),
        "expanded_fold_rate": float(
            pd.to_numeric(
                group["avg_watchlist_size_delta_vs_b0"], errors="coerce"
            ).gt(0).mean()
        ),
        "opportunity_positive_rate": _positive_rate(
            group["opportunity_recall_1w_delta_vs_b0"]
        ),
        "winner_lift_nonnegative_rate": _nonnegative_rate(
            group["tradable_winner_capture_lift_mean_2_4w_delta_vs_b0"]
        ),
        "loser_lift_nonworse_rate": _nonpositive_rate(
            group["tradable_loser_capture_lift_mean_2_4w_delta_vs_b0"]
        ),
        "mean_opportunity_delta": _mean(
            group["opportunity_recall_1w_delta_vs_b0"]
        ),
        "mean_winner_lift_delta": _mean(
            group["tradable_winner_capture_lift_mean_2_4w_delta_vs_b0"]
        ),
        "mean_loser_lift_delta": _mean(
            group["tradable_loser_capture_lift_mean_2_4w_delta_vs_b0"]
        ),
        "mean_attention_multiplier_vs_b0": _mean(
            group["attention_multiplier_vs_b0"]
        ),
        "median_attention_multiplier_vs_b0": _median(
            group["attention_multiplier_vs_b0"]
        ),
        "mean_incremental_opportunities_per_added_review": _mean(
            group["incremental_opportunities_per_added_review"]
        ),
    }
    for horizon in FORMAL_HORIZONS:
        row[f"mean_winner_lift_delta_{horizon}"] = _mean(
            group[f"tradable_winner_capture_lift_{horizon}_delta_vs_b0"]
        )
        row[f"mean_loser_lift_delta_{horizon}"] = _mean(
            group[f"tradable_loser_capture_lift_{horizon}_delta_vs_b0"]
        )
        row[f"winner_nonnegative_rate_{horizon}"] = _nonnegative_rate(
            group[f"tradable_winner_capture_lift_{horizon}_delta_vs_b0"]
        )
        row[f"loser_nonworse_rate_{horizon}"] = _nonpositive_rate(
            group[f"tradable_loser_capture_lift_{horizon}_delta_vs_b0"]
        )
    return pd.DataFrame([row])


def oos_policy_status(
    summary: pd.DataFrame,
    *,
    static_rule_exists: bool,
) -> str:
    if summary.empty or int(summary.iloc[0]["folds"]) < 3:
        return "INSUFFICIENT_OOS_HISTORY"
    if not static_rule_exists and str(summary.iloc[0]["evaluation_role"]) == "STATIC_DISCOVERY_RULE":
        return "NO_DISCOVERY_REFINEMENT"

    row = summary.iloc[0]
    required = [
        row["mean_opportunity_delta"],
        row["mean_winner_lift_delta"],
        row["mean_loser_lift_delta"],
        row["mean_attention_multiplier_vs_b0"],
        row["mean_incremental_opportunities_per_added_review"],
    ]
    for horizon in FORMAL_HORIZONS:
        required.extend(
            [
                row[f"mean_winner_lift_delta_{horizon}"],
                row[f"mean_loser_lift_delta_{horizon}"],
            ]
        )
    if not all(_finite(value) for value in required):
        return "NO_STABLE_DISCRIMINATIVE_RULE"

    horizon_consistent = all(
        is_nonnegative(row[f"mean_winner_lift_delta_{horizon}"])
        and is_nonpositive(row[f"mean_loser_lift_delta_{horizon}"])
        for horizon in FORMAL_HORIZONS
    )
    stable = (
        float(row["expanded_fold_rate"]) >= 0.60
        and float(row["opportunity_positive_rate"]) >= 0.60
        and float(row["winner_lift_nonnegative_rate"]) >= 0.60
        and float(row["loser_lift_nonworse_rate"]) >= 0.60
        and is_positive(row["mean_opportunity_delta"])
        and is_nonnegative(row["mean_winner_lift_delta"])
        and is_nonpositive(row["mean_loser_lift_delta"])
        and float(row["mean_attention_multiplier_vs_b0"])
        <= MAX_ATTENTION_MULTIPLIER + ZERO_TOL
        and is_positive(row["mean_incremental_opportunities_per_added_review"])
        and horizon_consistent
    )
    return (
        "RETROSPECTIVE_DISCRIMINATIVE_CANDIDATE"
        if stable
        else "NO_STABLE_DISCRIMINATIVE_RULE"
    )


def rule_to_dict(rule: RefinementRule | None) -> dict[str, object] | None:
    return asdict(rule) if rule is not None else None


def _condition_passes(row: pd.Series, key: str) -> bool:
    value = to_float(row.get("current_vs_ibd_candidate_pct"))
    base_depth = _abs_float(row.get("base_depth_pct"))
    base_duration = to_float(row.get("base_duration_weeks"))
    pullback_depth = _abs_float(row.get("pullback_pct"))
    pullback_duration = to_float(row.get("pullback_duration_weeks"))
    dist = to_float(row.get("dist_to_52w_high_pct"))
    setup = str(row.get("ibd_candidate_rule", "") or "").strip()

    checks = {
        "VS_ABS_LE_2": value is not None and abs(value) <= 2.0,
        "VS_ABS_LE_3": value is not None and abs(value) <= 3.0,
        "BASE_DEPTH_LE_33": base_depth is not None and base_depth <= 33.0,
        "BASE_DEPTH_LE_50": base_depth is not None and base_depth <= 50.0,
        "BASE_DURATION_7_65": (
            base_duration is not None and 7.0 <= base_duration <= 65.0
        ),
        "PULLBACK_DEPTH_LE_15": (
            pullback_depth is not None and pullback_depth <= 15.0
        ),
        "PULLBACK_DURATION_3_10": (
            pullback_duration is not None and 3.0 <= pullback_duration <= 10.0
        ),
        "SETUP_PULLBACK_LIKE": setup in PULLBACK_RULES,
        "RS_WITHIN_10": dist is not None and dist > -10.0,
        "REQ_VOLUME": bool(row.get("_evidence_volume", False)),
        "REQ_EPS": bool(row.get("_evidence_eps", False)),
        "REQ_RS": bool(row.get("_evidence_rs_near_high", False)),
        "REQ_SUPPLY": bool(row.get("_evidence_supply_contraction", False)),
    }
    if key not in checks:
        raise KeyError(f"Unknown refinement condition: {key}")
    return bool(checks[key])


def _train_quality_gate(row: pd.Series) -> bool:
    required = (
        row.get("attention_multiplier_vs_b0"),
        row.get("added_evaluable_reviews_vs_b0"),
        row.get("incremental_opportunities_per_added_review"),
        row.get("opportunity_recall_1w_delta_vs_b0"),
        row.get("tradable_winner_capture_lift_mean_2_4w_delta_vs_b0"),
        row.get("tradable_loser_capture_lift_mean_2_4w_delta_vs_b0"),
    )
    if not all(_finite(value) for value in required):
        return False
    return (
        float(row["attention_multiplier_vs_b0"])
        <= MAX_ATTENTION_MULTIPLIER + ZERO_TOL
        and float(row["added_evaluable_reviews_vs_b0"]) > 0
        and is_positive(row["incremental_opportunities_per_added_review"])
        and is_positive(row["opportunity_recall_1w_delta_vs_b0"])
        and is_nonnegative(
            row["tradable_winner_capture_lift_mean_2_4w_delta_vs_b0"]
        )
        and is_nonpositive(
            row["tradable_loser_capture_lift_mean_2_4w_delta_vs_b0"]
        )
        and int(row["horizon_consistency_count"]) >= 2
        and float(row.get("train_consistent_block_rate", 0.0)) >= (2.0 / 3.0)
    )


def _horizon_consistency(row: pd.Series | dict[str, object]) -> int:
    score = 0
    for horizon in FORMAL_HORIZONS:
        winner = row.get(
            f"tradable_winner_capture_lift_{horizon}_delta_vs_b0"
        )
        loser = row.get(
            f"tradable_loser_capture_lift_{horizon}_delta_vs_b0"
        )
        if _finite(winner) and _finite(loser):
            if is_nonnegative(winner) and is_nonpositive(loser):
                score += 1
    return score


def _train_block_consistency(
    train_panel: pd.DataFrame,
    rule: RefinementRule,
    *,
    blocks: int = 3,
) -> tuple[int, float]:
    weeks = sorted(train_panel["snapshot_date"].astype(str).unique())
    if not weeks:
        return 0, 0.0
    week_blocks = [
        list(block)
        for block in np.array_split(
            np.array(weeks, dtype=object), min(blocks, len(weeks))
        )
        if len(block)
    ]
    consistent = 0
    evaluated = 0
    for block_weeks in week_blocks:
        subset = train_panel[
            train_panel["snapshot_date"].astype(str).isin(block_weeks)
        ].copy()
        baseline = evaluate_selection(
            subset,
            select_refined_all_weeks(subset, None),
            variant=B0_NAME,
        )
        candidate = compare_metrics(
            evaluate_selection(
                subset,
                select_refined_all_weeks(subset, rule),
                variant=rule.name,
            ),
            baseline,
        )
        needed = (
            candidate.get("opportunity_recall_1w_delta_vs_b0"),
            candidate.get(
                "tradable_winner_capture_lift_mean_2_4w_delta_vs_b0"
            ),
            candidate.get(
                "tradable_loser_capture_lift_mean_2_4w_delta_vs_b0"
            ),
        )
        if not all(_finite(value) for value in needed):
            continue
        evaluated += 1
        if (
            is_positive(candidate["opportunity_recall_1w_delta_vs_b0"])
            and is_nonnegative(
                candidate[
                    "tradable_winner_capture_lift_mean_2_4w_delta_vs_b0"
                ]
            )
            and is_nonpositive(
                candidate[
                    "tradable_loser_capture_lift_mean_2_4w_delta_vs_b0"
                ]
            )
        ):
            consistent += 1
    rate = consistent / evaluated if evaluated else 0.0
    return consistent, rate


def _pareto_mask(frame: pd.DataFrame) -> pd.Series:
    objectives = (
        ("tradable_winner_capture_lift_mean_2_4w_delta_vs_b0", True),
        ("tradable_loser_capture_lift_mean_2_4w_delta_vs_b0", False),
        ("incremental_opportunities_per_added_review", True),
        ("attention_multiplier_vs_b0", False),
    )
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
                    if b < a - ZERO_TOL:
                        no_worse = False
                        break
                    strictly_better |= b > a + ZERO_TOL
                else:
                    if b > a + ZERO_TOL:
                        no_worse = False
                        break
                    strictly_better |= b < a - ZERO_TOL
            if no_worse and strictly_better:
                dominated = True
                break
        mask.at[i] = not dominated
    return mask


def _rank_grid(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        [
            "train_consistent_block_rate",
            "horizon_consistency_count",
            "tradable_winner_capture_lift_mean_2_4w_delta_vs_b0",
            "tradable_loser_capture_lift_mean_2_4w_delta_vs_b0",
            "incremental_opportunities_per_added_review",
            "attention_multiplier_vs_b0",
            "condition_count",
            "variant",
        ],
        ascending=[False, False, False, True, False, True, True, True],
        kind="mergesort",
        na_position="last",
    )


def _selection_signature(selected: pd.DataFrame) -> str:
    if selected.empty:
        payload = "<EMPTY>"
    else:
        keys = (
            selected[["snapshot_date", "code"]]
            .astype(str)
            .drop_duplicates()
            .sort_values(["snapshot_date", "code"], kind="mergesort")
        )
        payload = "\n".join(
            f"{row.snapshot_date}|{row.code}"
            for row in keys.itertuples(index=False)
        )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _quality_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    resolved = pd.Series(0, index=out.index, dtype=int)
    winners = pd.Series(0, index=out.index, dtype=int)
    losers = pd.Series(0, index=out.index, dtype=int)
    returns = []

    for horizon in FORMAL_HORIZONS:
        complete = out[f"opp_forward_{horizon}_censored"].eq(False)
        resolved += complete.astype(int)
        winners += (
            complete & out[f"opp_big_winner_any_{horizon}"].eq(True)
        ).astype(int)
        losers += (
            complete & out[f"opp_big_loser_any_{horizon}"].eq(True)
        ).astype(int)
        returns.append(
            pd.to_numeric(
                out[f"opp_forward_{horizon}_return_pct"], errors="coerce"
            )
        )

    return_frame = pd.concat(returns, axis=1)
    out["_quality_resolved_horizons"] = resolved
    out["_persistent_winner"] = (
        out["review_opportunity_1w"].eq(True)
        & resolved.ge(2)
        & winners.ge(2)
    )
    out["_persistent_loser"] = (
        out["review_opportunity_1w"].eq(True)
        & resolved.ge(2)
        & losers.ge(2)
    )
    out["_quality_evaluable"] = (
        out["review_opportunity_1w"].eq(True) & resolved.ge(2)
    )
    out["_mean_opp_return_2_4w"] = return_frame.mean(axis=1, skipna=True)
    return out


def _bucket_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["status"] = (
        frame["ibd_entry_status"].fillna("<MISSING>").astype(str)
    )
    out["setup"] = (
        frame["ibd_candidate_rule"].fillna("<MISSING>").astype(str)
    )
    out["vs_buy_point"] = frame["current_vs_ibd_candidate_pct"].map(
        _bucket_vs
    )
    out["base_depth"] = frame["base_depth_pct"].map(_bucket_base_depth)
    out["base_duration"] = frame["base_duration_weeks"].map(
        _bucket_base_duration
    )
    out["pullback_depth"] = frame["pullback_pct"].map(
        _bucket_pullback_depth
    )
    out["pullback_duration"] = frame["pullback_duration_weeks"].map(
        _bucket_pullback_duration
    )
    out["entry_volume"] = frame["ibd_entry_volume_ratio"].map(
        _bucket_entry_volume
    )
    out["weekly_volume"] = frame["volume_ratio"].map(_bucket_weekly_volume)
    out["eps"] = frame.apply(_bucket_eps, axis=1)
    out["rs_distance"] = frame["dist_to_52w_high_pct"].map(
        _bucket_rs
    )
    out["dry_state"] = frame["pullback_v_is_dry"].map(_bucket_dry)
    out["evidence_count"] = frame["_evidence_family_count"].map(
        lambda value: f"E{int(value)}" if _finite(value) else "MISSING"
    )
    return out


def _bucket_summary(
    group: pd.DataFrame,
    feature: str,
    bucket: str,
    *,
    total_rows: int,
) -> dict[str, object]:
    total = len(group)
    quality = group.loc[group["_quality_evaluable"].eq(True)]
    return {
        "feature": feature,
        "bucket": bucket,
        "rows": total,
        "sample_share": total / total_rows if total_rows else np.nan,
        "opportunity_rate": (
            float(group["review_opportunity_1w"].mean())
            if total
            else np.nan
        ),
        "quality_evaluable_rows": len(quality),
        "persistent_winner_rate": (
            float(quality["_persistent_winner"].mean())
            if len(quality)
            else np.nan
        ),
        "persistent_loser_rate": (
            float(quality["_persistent_loser"].mean())
            if len(quality)
            else np.nan
        ),
        "winner_minus_loser": (
            float(quality["_persistent_winner"].mean())
            - float(quality["_persistent_loser"].mean())
            if len(quality)
            else np.nan
        ),
        "median_mean_opp_return_2_4w": (
            float(
                pd.to_numeric(
                    quality["_mean_opp_return_2_4w"], errors="coerce"
                ).median()
            )
            if len(quality)
            else np.nan
        ),
    }


def _adaptive_convergence(choices: pd.DataFrame) -> pd.DataFrame:
    if choices.empty:
        return pd.DataFrame()
    counts = choices["adaptive_rule"].astype(str).value_counts()
    total = len(choices)
    return pd.DataFrame(
        [
            {
                "rule": rule,
                "fold_count": int(count),
                "fold_share": float(count) / total if total else np.nan,
            }
            for rule, count in counts.items()
        ]
    )


def _tail_exploratory(
    panel: pd.DataFrame,
    *,
    weeks: list[str],
    formal_blocks: list[dict[str, object]],
    tail_weeks: list[str],
    static_rule: RefinementRule | None,
    library: list[RefinementRule],
) -> pd.DataFrame:
    if not tail_weeks:
        return pd.DataFrame()

    train_end = (
        len(formal_blocks[-1]["train_weeks"])
        + len(formal_blocks[-1]["test_weeks"])
    )
    train_weeks = weeks[:train_end]
    cutoff = tail_weeks[0]
    train = panel_asof_cutoff(
        panel[panel["snapshot_date"].astype(str).isin(train_weeks)].copy(),
        cutoff,
    )
    adaptive_rule, _ = choose_train_refinement(train, library)
    tail = panel[
        panel["snapshot_date"].astype(str).isin(tail_weeks)
    ].copy()
    baseline = evaluate_selection(
        tail,
        select_refined_all_weeks(tail, None),
        variant=B0_NAME,
    )

    rows = []
    for role, rule in (
        ("STATIC_DISCOVERY_RULE", static_rule),
        ("ADAPTIVE_DISCRIMINATIVE_POLICY", adaptive_rule),
    ):
        selected = select_refined_all_weeks(tail, rule)
        metrics = compare_metrics(
            evaluate_selection(
                tail,
                selected,
                variant=rule.name if rule is not None else B0_NAME,
            ),
            baseline,
        )
        metrics.update(
            {
                "phase": "TAIL_EXPLORATORY",
                "evaluation_role": role,
                "selected_rule": rule.name if rule is not None else B0_NAME,
                "test_start": tail_weeks[0],
                "test_end": tail_weeks[-1],
                "tail_week_count": len(tail_weeks),
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def _bucket_vs(value: object) -> str:
    number = to_float(value)
    if number is None:
        return "MISSING"
    if number < -2:
        return "[-5,-2)"
    if number < 0:
        return "[-2,0)"
    if number <= 2:
        return "[0,2]"
    return "(2,5]"


def _bucket_base_depth(value: object) -> str:
    number = _abs_float(value)
    if number is None:
        return "MISSING"
    if number <= 33:
        return "<=33"
    if number <= 50:
        return "33-50"
    return ">50"


def _bucket_base_duration(value: object) -> str:
    number = to_float(value)
    if number is None:
        return "MISSING"
    if number < 7:
        return "<7"
    if number <= 20:
        return "7-20"
    if number <= 65:
        return "21-65"
    return ">65"


def _bucket_pullback_depth(value: object) -> str:
    number = _abs_float(value)
    if number is None:
        return "MISSING"
    if number <= 8:
        return "<=8"
    if number <= 15:
        return "8-15"
    if number <= 25:
        return "15-25"
    return ">25"


def _bucket_pullback_duration(value: object) -> str:
    number = to_float(value)
    if number is None:
        return "MISSING"
    if number <= 3:
        return "<=3"
    if number <= 6:
        return "4-6"
    if number <= 10:
        return "7-10"
    return ">10"


def _bucket_entry_volume(value: object) -> str:
    number = to_float(value)
    if number is None:
        return "MISSING"
    return ">=1.5" if number >= 1.5 else "<1.5"


def _bucket_weekly_volume(value: object) -> str:
    number = to_float(value)
    if number is None:
        return "MISSING"
    if number < 1.0:
        return "<1.0"
    if number < 1.3:
        return "1.0-1.3"
    return ">=1.3"


def _bucket_eps(row: pd.Series) -> str:
    if str(row.get("pit_eps_state", "") or "").upper() != "VERIFIED":
        return "MISSING"
    number = to_float(row.get("pit_eps_yoy_growth"))
    if number is None:
        return "MISSING"
    return ">=25" if number >= 25 else "<25"


def _bucket_rs(value: object) -> str:
    number = to_float(value)
    if number is None:
        return "MISSING"
    if number > -5:
        return ">-5"
    if number > -10:
        return "(-10,-5]"
    return "<=-10"


def _bucket_dry(value: object) -> str:
    parsed = to_bool(value)
    if parsed is True:
        return "TRUE"
    if parsed is False:
        return "FALSE"
    return "MISSING"


def _abs_float(value: object) -> float | None:
    number = to_float(value)
    return abs(number) if number is not None else None


def _mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if len(numeric) else np.nan


def _median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if len(numeric) else np.nan


def _positive_rate(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float((numeric > ZERO_TOL).mean()) if len(numeric) else 0.0


def _nonnegative_rate(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float((numeric >= -ZERO_TOL).mean()) if len(numeric) else 0.0


def _nonpositive_rate(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float((numeric <= ZERO_TOL).mean()) if len(numeric) else 0.0


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _finite_value(value: object, maximize: bool) -> float:
    if _finite(value):
        return float(value)
    return -np.inf if maximize else np.inf
