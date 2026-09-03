from __future__ import annotations

import hashlib

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


def selection_signature(panel: pd.DataFrame, rule: ReviewRule) -> str:
    """Hash the actual selected (snapshot, code) set.

    Parameterizations that behave identically on the train sample are one
    effective hypothesis, not multiple independent rules.
    """
    selected = select_all_weeks(panel, rule)
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
    """Direction stability across chronological training blocks.

    Stability is a ranking input only. It is deliberately not an OOS admission
    gate in v0.5.
    """
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
    """Two-stage search with train-sample behavioral de-duplication."""
    core_diag = _diagnostic_frame(panel, core_rules)
    core_diag, unique_core_rules = _annotate_signatures(
        panel, core_rules, core_diag
    )
    finalists = _top_finalists(
        core_diag, unique_core_rules, max_finalists=max_finalists
    )

    expanded: dict[str, ReviewRule] = {}
    for core_rule in finalists:
        for rule in generate_evidence_ablations(core_rule):
            expanded[rule.name] = rule
    evidence_rules = list(expanded.values())
    if not evidence_rules:
        return unique_core_rules, core_diag, pd.DataFrame()

    evidence_diag = _diagnostic_frame(panel, evidence_rules)
    evidence_diag, unique_evidence_rules = _annotate_signatures(
        panel, evidence_rules, evidence_diag
    )
    return unique_evidence_rules, core_diag, evidence_diag


def choose_training_champion(
    panel: pd.DataFrame,
    core_rules: list[ReviewRule],
) -> tuple[ReviewRule | None, pd.DataFrame, pd.DataFrame]:
    """Always choose one train-only provisional champion when rules are evaluable.

    v0.5 intentionally removes the old train stability hard veto. OOS, not the
    training blocks, decides whether the adaptive policy is stable.
    """
    evidence_rules, core_diag, evidence_diag = two_stage_diagnostics(
        panel, core_rules
    )

    if not evidence_diag.empty:
        candidates = evidence_diag[
            evidence_diag["signature_representative"].eq(True)
        ].copy()
        frontier = candidates[candidates["pareto"].eq(True)].copy()
        ranked = _rank_candidates(frontier if not frontier.empty else candidates)
        if not ranked.empty:
            name = str(ranked.iloc[0]["variant"])
            rule = next(
                (item for item in evidence_rules if item.name == name),
                None,
            )
            if rule is not None:
                return rule, core_diag, evidence_diag

    core_candidates = core_diag[
        core_diag["signature_representative"].eq(True)
    ].copy()
    frontier = core_candidates[core_candidates["pareto"].eq(True)].copy()
    ranked = _rank_candidates(frontier if not frontier.empty else core_candidates)
    if ranked.empty:
        return None, core_diag, evidence_diag
    name = str(ranked.iloc[0]["variant"])
    unique_core = {
        rule.name: rule for rule in core_rules
    }
    return unique_core.get(name), core_diag, evidence_diag


def pareto_mask(frame: pd.DataFrame) -> pd.Series:
    """Pareto objectives for attention-efficient supplemental admission."""
    if frame.empty:
        return pd.Series(dtype=bool)
    objectives = [
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


def rule_structure_key(rule: ReviewRule) -> str:
    statuses = "+".join(rule.supplemental_statuses)
    geometry = "exclude" if rule.exclude_clear_geometry_failure else "allow"
    return (
        f"near={rule.near_below_pct:g}|statuses={statuses}|"
        f"minE={rule.min_evidence_families}|geometry={geometry}"
    )


def rule_evidence_profile(rule: ReviewRule) -> str:
    return "+".join(rule.enabled_evidence_families)


def _diagnostic_frame(
    panel: pd.DataFrame,
    rules: list[ReviewRule],
) -> pd.DataFrame:
    grid = evaluate_rule_grid(panel, rules)
    stable = stability_table(panel, rules)
    merged = grid.merge(stable, on="variant", how="left")
    merged["pareto"] = pareto_mask(merged)
    return merged


def _annotate_signatures(
    panel: pd.DataFrame,
    rules: list[ReviewRule],
    diag: pd.DataFrame,
) -> tuple[pd.DataFrame, list[ReviewRule]]:
    if diag.empty:
        return diag, []

    signature_by_name = {
        rule.name: selection_signature(panel, rule)
        for rule in rules
    }
    diag = diag.copy()
    diag["selection_signature"] = diag["variant"].map(signature_by_name)

    rule_map = {rule.name: rule for rule in rules}
    representatives: dict[str, str] = {}
    group_sizes: dict[str, int] = {}
    for signature, names in diag.groupby("selection_signature")["variant"]:
        candidate_names = list(names.astype(str))
        candidate_names.sort(
            key=lambda name: (
                rule_complexity(rule_map[name]),
                name,
            )
        )
        representatives[str(signature)] = candidate_names[0]
        group_sizes[str(signature)] = len(candidate_names)

    diag["signature_group_size"] = diag["selection_signature"].map(group_sizes)
    diag["signature_representative_name"] = diag["selection_signature"].map(
        representatives
    )
    diag["signature_representative"] = (
        diag["variant"].astype(str)
        == diag["signature_representative_name"].astype(str)
    )
    unique_rules = [
        rule_map[name]
        for name in sorted(set(representatives.values()))
    ]
    return diag, unique_rules


def _top_finalists(
    diag: pd.DataFrame,
    rules: list[ReviewRule],
    *,
    max_finalists: int,
) -> list[ReviewRule]:
    if diag.empty:
        return []
    candidates = diag[
        diag["signature_representative"].eq(True)
    ].copy()
    frontier = candidates[candidates["pareto"].eq(True)].copy()
    ranked = _rank_candidates(frontier if not frontier.empty else candidates)
    names = set(ranked.head(max_finalists)["variant"].astype(str))
    return [rule for rule in rules if rule.name in names]


def _rank_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministic Pareto tie-breaker; no arbitrary weighted score."""
    if frame.empty:
        return frame
    return frame.sort_values(
        [
            "stability_floor",
            "tradable_winner_capture_lift_mean_2_4w_delta_vs_b0",
            "tradable_loser_capture_lift_mean_2_4w_delta_vs_b0",
            "incremental_opportunities_per_added_review",
            "avg_watchlist_size_delta_vs_b0",
            "rule_complexity",
            "variant",
        ],
        ascending=[False, False, True, False, True, True, True],
        kind="mergesort",
        na_position="last",
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
