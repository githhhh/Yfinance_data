from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _fmt(v: Any, d: int = 3) -> str:
    if v is None:
        return "N/A"
    try:
        if pd.isna(v):
            return "N/A"
    except Exception:
        pass
    try:
        return f"{float(v):.{d}f}"
    except Exception:
        return str(v)


def _pct(v: Any, d: int = 1) -> str:
    if v is None:
        return "N/A"
    try:
        if pd.isna(v):
            return "N/A"
    except Exception:
        pass
    return f"{float(v) * 100.0:.{d}f}%"


def write_report(
    path: Path,
    *,
    outputs: dict[str, pd.DataFrame],
    manifest: dict[str, Any],
) -> None:
    labels = outputs["label_summary"]
    support = outputs["task_support_by_quarter"]
    numeric = outputs["numeric_feature_separation"]
    categorical = outputs["categorical_feature_separation"]
    models = outputs["model_summary"]
    pairs = outputs["pair_interactions"]
    tree = outputs["tree_permutation_importance"]
    reject = outputs["reject_reason_outcomes"]
    redundancy = manifest.get("redundancy", {})
    gaps = manifest.get("feature_gaps", {})

    lines = [
        "# B0 Error Atlas — False-Negative Recovery / False-Positive Veto",
        "",
        "## Purpose",
        "",
        "This track does **not** search for a replacement B1 and does not mutate Production.",
        "It asks whether the frozen PIT information can distinguish two concrete B0 errors:",
        "",
        "1. **False-negative recovery:** names B0 rejected/missed that later became clean large winners.",
        "2. **False-positive veto:** names B0 selected that later became path failures.",
        "",
        "Gate misses and eligible-but-unselected selector misses are analyzed separately.",
        "",
        "## Frozen path-aware labels",
        "",
        "- Clean big winner: terminal next-open W4 >= +20% and never touched Stop8.",
        "- Rebound big winner: terminal W4 >= +20% but touched Stop8 along the path.",
        "- Strict path failure: Stop8 occurs before Profit20, or terminal W4 <= -8%.",
        "- Same-day Stop8 and Profit20 is path-order ambiguous and excluded from binary tasks.",
        "",
        "Middle outcomes are deliberately excluded from the recovery/veto binary tasks.",
        "",
        "## Task support",
        "",
        "| Task | Rows | Weeks | Positive | Negative | Positive rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in labels.iterrows():
        if pd.isna(row.get("positive_rows")):
            continue
        lines.append(
            f"| {row['task']} | {int(row['rows'])} | {int(row['weeks'])} | "
            f"{int(row['positive_rows'])} | {int(row['negative_rows'])} | "
            f"{_pct(row['positive_rate'])} |"
        )

    lines += [
        "",
        "### Support by quarter",
        "",
        "| Task | Quarter | Rows | Weeks | Positive | Negative |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in support.iterrows():
        lines.append(
            f"| {row['task']} | {row['quarter']} | {int(row['rows'])} | "
            f"{int(row['weeks'])} | {int(row['positive_rows'])} | "
            f"{int(row['negative_rows'])} |"
        )

    lines += [
        "",
        "## Feature-information redundancy",
        "",
        f"- Numeric features considered: **{redundancy.get('numeric_features_considered', 'N/A')}**",
        f"- Numeric features with support: **{redundancy.get('numeric_features_with_support', 'N/A')}**",
        f"- Median pairwise |Spearman|: **{_fmt(redundancy.get('median_pairwise_abs_spearman'))}**",
        f"- 90th-percentile pairwise |Spearman|: **{_fmt(redundancy.get('p90_pairwise_abs_spearman'))}**",
        f"- Pairs with |Spearman| >= 0.85: **{redundancy.get('pairs_abs_spearman_ge_0_85', 'N/A')}**",
        f"- PCA components for 80/90/95% variance: "
        f"**{redundancy.get('pca_components_80pct', 'N/A')} / "
        f"{redundancy.get('pca_components_90pct', 'N/A')} / "
        f"{redundancy.get('pca_components_95pct', 'N/A')}**",
        "",
        "This section measures whether a large field count actually represents many independent information dimensions.",
        "",
        "## Strongest numeric separators — descriptive full-history coordinates",
        "",
        "These are **not** OOS claims. AUC separation is sign-agnostic: 0.5=no separation, 1.0=perfect separation.",
        "",
    ]

    if numeric.empty:
        lines.append("No numeric feature diagnostics.")
    else:
        for task in numeric["task"].drop_duplicates().tolist():
            lines += [f"### {task}", ""]
            top = (
                numeric[numeric["task"] == task]
                .sort_values(
                    ["auc_separation", "quarter_direction_stability", "abs_standardized_mean_diff"],
                    ascending=[False, False, False],
                )
                .head(12)
            )
            lines += [
                "| Feature | AUC separation | Direction | Std mean diff | Median diff | MI | Missing | Quarter direction stability |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
            for _, row in top.iterrows():
                lines.append(
                    f"| {row['feature']} | {_fmt(row['auc_separation'])} | "
                    f"{row['auc_direction']} | {_fmt(row['standardized_mean_diff'])} | "
                    f"{_fmt(row['median_diff'])} | {_fmt(row['mutual_information'])} | "
                    f"{_pct(row['missing_rate'])} | "
                    f"{_pct(row.get('quarter_direction_stability'))} |"
                )
            lines.append("")

    lines += [
        "## Categorical separators",
        "",
    ]
    if categorical.empty:
        lines.append("No categorical diagnostics.")
    else:
        top_cat = categorical.sort_values(
            ["mutual_information", "max_min_positive_rate_gap_min2"],
            ascending=[False, False],
        )
        lines += [
            "| Task | Feature | MI | Categories | Missing | Max-min rate gap (min2/category) |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for _, row in top_cat.iterrows():
            lines.append(
                f"| {row['task']} | {row['feature']} | "
                f"{_fmt(row['mutual_information'])} | {int(row['unique_categories'])} | "
                f"{_pct(row['missing_rate'])} | "
                f"{_pct(row['max_min_positive_rate_gap_min2'])} |"
            )

    lines += [
        "",
        "## Chronological model test",
        "",
        "Untuned models use expanding-quarter tests only; there is no random row split.",
        "RAW_ONLY excludes all B0-derived rank/lane/reject fields. B0_AUGMENTED adds them.",
        "",
    ]
    if models.empty:
        lines.append("No model had sufficient chronological class support.")
    else:
        lines += [
            "| Task | Feature set | Model | Folds | Mean AUC | Median AUC | Min AUC | Mean AP | Balanced acc | All folds >0.5 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for _, row in models.sort_values(
            ["task", "mean_roc_auc"], ascending=[True, False]
        ).iterrows():
            lines.append(
                f"| {row['task']} | {row['feature_set']} | {row['model']} | "
                f"{int(row['folds'])} | {_fmt(row['mean_roc_auc'])} | "
                f"{_fmt(row['median_roc_auc'])} | {_fmt(row['min_roc_auc'])} | "
                f"{_fmt(row['mean_average_precision'])} | "
                f"{_fmt(row['mean_balanced_accuracy'])} | "
                f"{row['all_fold_auc_above_0_5']} |"
            )

    lines += [
        "",
        "## Pair interactions — exploratory",
        "",
        "Pairs are selected from full-history univariate diagnostics, then evaluated chronologically. "
        "Therefore pair results are hypothesis-generation evidence, not untouched OOS proof.",
        "",
    ]
    if pairs.empty:
        lines.append("No pair scan had sufficient support.")
    else:
        top_pairs = pairs.sort_values(
            ["task", "pair_synergy_vs_best_single", "mean_pair_auc"],
            ascending=[True, False, False],
        ).groupby("task", sort=False).head(10)
        lines += [
            "| Task | A | B | Pair AUC | Best single AUC | Synergy | Folds |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for _, row in top_pairs.iterrows():
            lines.append(
                f"| {row['task']} | {row['feature_a']} | {row['feature_b']} | "
                f"{_fmt(row['mean_pair_auc'])} | {_fmt(row['best_single_cv_auc'])} | "
                f"{_fmt(row['pair_synergy_vs_best_single'])} | {int(row['folds'])} |"
            )

    lines += [
        "",
        "## Exploratory tree permutation importance",
        "",
        "Permutation importance is computed on chronological test quarters. "
        "It is used only to surface candidate mechanisms, not to define Production rules.",
        "",
    ]
    if tree.empty:
        lines.append("No tree-importance support.")
    else:
        top_tree = tree.sort_values(
            ["task", "mean_permutation_importance"],
            ascending=[True, False],
        ).groupby("task", sort=False).head(12)
        lines += [
            "| Task | Feature | Mean importance | Median | Min | Positive-fold rate | Folds |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, row in top_tree.iterrows():
            lines.append(
                f"| {row['task']} | {row['feature']} | "
                f"{_fmt(row['mean_permutation_importance'])} | "
                f"{_fmt(row['median_permutation_importance'])} | "
                f"{_fmt(row['min_permutation_importance'])} | "
                f"{_pct(row['positive_fold_rate'])} | {int(row['folds'])} |"
            )

    lines += [
        "",
        "## Reject-reason outcome map",
        "",
    ]
    if reject.empty:
        lines.append("No reject-reason rows.")
    else:
        lines += [
            "| Reason | Events | Weeks | Clean +20 | Rebound +20 | Path failure | Mean terminal | Median terminal | Mean MAE | Mean MFE |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, row in reject.sort_values("events", ascending=False).iterrows():
            lines.append(
                f"| {row['reason']} | {int(row['events'])} | {int(row['weeks'])} | "
                f"{_pct(row['clean_big_winner_rate'])} | "
                f"{_pct(row['rebound_big_winner_rate'])} | "
                f"{_pct(row['path_failure_rate'])} | "
                f"{_fmt(row['mean_terminal_return'], 2)}% | "
                f"{_fmt(row['median_terminal_return'], 2)}% | "
                f"{_fmt(row['mean_mae'], 2)}% | {_fmt(row['mean_mfe'], 2)}% |"
            )

    lines += [
        "",
        "## Remaining information gaps",
        "",
    ]
    for item in gaps.get("still_not_available_in_frozen_pool", []):
        lines.append(f"- {item}")

    lines += [
        "",
        "## Evidence boundary",
        "",
        "- All discovery is retrospective on reused history.",
        "- Chronological quarter tests reduce leakage but do not create untouched forward evidence.",
        "- Error tasks intentionally focus on tails and exclude ambiguous middle outcomes.",
        "- Features derived in this track use only data available on or before each snapshot.",
        "- No result in this report changes Production B0.",
        "",
        "## Provenance",
        "",
        f"- protocol_version: {manifest.get('protocol_version')}",
        f"- source_git_sha: {manifest.get('source_git_sha')}",
        f"- panel_hash: {manifest.get('panel_hash')}",
        f"- b0_state_hash: {manifest.get('b0_state_hash')}",
        f"- base_price_cache_hash: {manifest.get('base_price_cache_hash')}",
        f"- yahoo_supplement_hash: {manifest.get('yahoo_supplement_hash')}",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
