from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .analysis import (
    categorical_feature_separation,
    error_examples,
    feature_redundancy,
    numeric_feature_separation,
)
from .config import (
    B0_AUGMENTED_CATEGORICAL,
    B0_AUGMENTED_NUMERIC,
    DERIVED_CONTEXT_FEATURES,
    DERIVED_MARKET_FEATURES,
    DERIVED_PRICE_FEATURES,
    PROTOCOL_VERSION,
)
from .data import load_analysis_frame, load_frozen_prices
from .features import build_feature_frame
from .labels import add_path_labels, label_summary, task_frames
from .modeling import (
    evaluate_models,
    exploratory_numeric_tree_importance,
    pair_interaction_scan,
)


def _task_support_by_quarter(tasks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task_name, frame in tasks.items():
        if frame.empty:
            continue
        work = frame.copy()
        work["quarter"] = pd.PeriodIndex(
            pd.to_datetime(work["snapshot_date"]), freq="Q"
        ).astype(str)
        for quarter, group in work.groupby("quarter", sort=True):
            rows.append({
                "task": task_name,
                "quarter": quarter,
                "rows": int(len(group)),
                "weeks": int(group["snapshot_date"].nunique()),
                "positive_rows": int(group["target"].sum()),
                "negative_rows": int(len(group) - group["target"].sum()),
                "positive_rate": float(group["target"].mean()),
            })
    return pd.DataFrame(rows)


def _reason_outcome_summary(panel: pd.DataFrame) -> pd.DataFrame:
    valid = panel[
        (panel["path_valid"] == True)
        & (~panel["current_b0_eligible"].astype(bool))
    ].copy()
    rows: list[dict[str, Any]] = []

    for _, row in valid.iterrows():
        reasons = [
            r
            for r in str(row.get("current_b0_reject_reasons", "") or "").split("|")
            if r
        ]
        for reason in reasons:
            rows.append({
                "snapshot_date": row["snapshot_date"],
                "code": row["code"],
                "reason": reason,
                "clean_big_winner": bool(row["clean_big_winner"]),
                "rebound_big_winner": bool(row["rebound_big_winner"]),
                "strict_path_failure": (
                    np.nan
                    if pd.isna(row["strict_path_failure"])
                    else bool(row["strict_path_failure"])
                ),
                "terminal_return": float(row["next_open_w4_return_pct"]),
                "mae": float(row["path_mae_pct"]),
                "mfe": float(row["path_mfe_pct"]),
            })

    if not rows:
        return pd.DataFrame()
    exploded = pd.DataFrame(rows)
    return (
        exploded.groupby("reason", sort=False)
        .agg(
            events=("code", "size"),
            weeks=("snapshot_date", "nunique"),
            clean_big_winner_rate=("clean_big_winner", "mean"),
            rebound_big_winner_rate=("rebound_big_winner", "mean"),
            path_failure_rate=(
                "strict_path_failure",
                lambda s: float(pd.Series(s).dropna().astype(float).mean())
                if not pd.Series(s).dropna().empty
                else np.nan,
            ),
            mean_terminal_return=("terminal_return", "mean"),
            median_terminal_return=("terminal_return", "median"),
            mean_mae=("mae", "mean"),
            mean_mfe=("mfe", "mean"),
        )
        .reset_index()
    )


def feature_gap_manifest() -> dict[str, Any]:
    return {
        "already_added_in_this_track": [
            "pre-snapshot downside volatility / max down-day / gap-risk / skew / drawdown",
            "pre-snapshot close-position and down-volume structure",
            "SPY market momentum / volatility / drawdown regime",
            "cross-sectional and sector-relative pool context",
        ],
        "still_not_available_in_frozen_pool": [
            "sector-ETF relative strength and beta-adjusted residual returns",
            "sales/revenue growth and acceleration",
            "earnings-estimate revisions / surprise / next-quarter expectations",
            "institutional ownership / sponsorship change",
            "earnings date and other event/catalyst calendar",
            "news/event embeddings or offering/FDA/litigation flags",
            "true industry breadth outside the Review Universe",
        ],
        "interpretation": (
            "Failure to separate FN/FP after the added price-path and context features "
            "would be evidence that the current frozen pool still lacks independent "
            "predictive dimensions; it would not prove B0 globally optimal."
        ),
    }


def run_study() -> dict[str, Any]:
    panel, source_manifest = load_analysis_frame()
    prices = load_frozen_prices()

    panel = build_feature_frame(panel, prices)
    panel = add_path_labels(panel, prices)
    tasks = task_frames(panel)

    raw_numeric = list(source_manifest["raw_numeric_features"])
    raw_categorical = list(source_manifest["raw_categorical_features"])
    derived_numeric = list(
        DERIVED_PRICE_FEATURES + DERIVED_MARKET_FEATURES + DERIVED_CONTEXT_FEATURES
    )
    all_numeric_raw_only = sorted(set(raw_numeric + derived_numeric))
    all_categorical_raw_only = sorted(set(raw_categorical))

    numeric_sep, numeric_quarter = numeric_feature_separation(
        tasks,
        all_numeric_raw_only,
    )
    categorical_sep, categorical_levels = categorical_feature_separation(
        tasks,
        all_categorical_raw_only,
    )

    model_folds, model_summary = evaluate_models(
        tasks,
        raw_numeric,
        raw_categorical,
        derived_numeric,
    )
    pairs = pair_interaction_scan(tasks, numeric_sep)
    tree_importance = exploratory_numeric_tree_importance(
        tasks,
        all_numeric_raw_only,
    )

    redundancy_summary, high_corr_pairs = feature_redundancy(
        panel[panel["path_valid"] == True].copy(),
        all_numeric_raw_only,
    )

    missed, bad = error_examples(panel)

    outputs = {
        "analysis_frame": panel,
        "label_summary": label_summary(panel, tasks),
        "task_support_by_quarter": _task_support_by_quarter(tasks),
        "numeric_feature_separation": numeric_sep,
        "numeric_quarter_stability": numeric_quarter,
        "categorical_feature_separation": categorical_sep,
        "categorical_level_rates": categorical_levels,
        "model_fold_results": model_folds,
        "model_summary": model_summary,
        "pair_interactions": pairs,
        "tree_permutation_importance": tree_importance,
        "high_corr_pairs": high_corr_pairs,
        "missed_winner_examples": missed,
        "bad_selected_examples": bad,
        "reject_reason_outcomes": _reason_outcome_summary(panel),
    }

    manifest = {
        **source_manifest,
        "protocol_version": PROTOCOL_VERSION,
        "path_label_semantics": {
            "entry": "frozen next-open entry date from B0 absolute audit",
            "horizon": "entry date through frozen W4 end date",
            "clean_big_winner": "terminal W4 >= +20% and never touched Stop8",
            "rebound_big_winner": "terminal W4 >= +20% but touched Stop8",
            "strict_path_failure": (
                "Stop8 occurs before Profit20, or terminal W4 <= -8%; "
                "same-day Stop8+Profit20 ordering is ambiguous and excluded from binary tasks"
            ),
        },
        "feature_sets": {
            "RAW_ONLY_numeric": all_numeric_raw_only,
            "RAW_ONLY_categorical": all_categorical_raw_only,
            "B0_AUGMENTED_numeric": list(B0_AUGMENTED_NUMERIC),
            "B0_AUGMENTED_categorical": list(B0_AUGMENTED_CATEGORICAL),
        },
        "modeling_semantics": (
            "Frozen untuned logistic-L2 and shallow random forest. "
            "Chronological expanding-quarter evaluation only; no random row split."
        ),
        "pair_scan_semantics": (
            "Exploratory top-univariate numeric pairs, evaluated with the same "
            "chronological quarter folds. Pair list selection uses full-history "
            "univariate diagnostics and is therefore not untouched OOS evidence."
        ),
        "tree_importance_semantics": (
            "Exploratory numeric random-forest permutation importance on chronological "
            "test quarters; no SHAP dependency is required."
        ),
        "redundancy": redundancy_summary,
        "feature_gaps": feature_gap_manifest(),
    }

    return {
        "outputs": outputs,
        "manifest": manifest,
        "tasks": tasks,
    }
