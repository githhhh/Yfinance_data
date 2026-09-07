from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    B0_AUGMENTED_CATEGORICAL,
    B0_AUGMENTED_NUMERIC,
    MAX_PAIR_CANDIDATES,
    PERMUTATION_REPEATS,
    RANDOM_SEED,
    TOP_NUMERIC_FOR_PAIRS,
)


def quarter_key(series: pd.Series) -> pd.Series:
    return pd.PeriodIndex(pd.to_datetime(series), freq="Q").astype(str)


def chronological_quarter_splits(
    frame: pd.DataFrame,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    work = frame.copy()
    quarters = quarter_key(work["snapshot_date"])
    unique = sorted(quarters.unique().tolist())
    splits: list[tuple[str, np.ndarray, np.ndarray]] = []
    for q in unique[1:]:
        train = np.where(quarters < q)[0]
        test = np.where(quarters == q)[0]
        if len(train) < 10 or len(test) < 4:
            continue
        y_train = work.iloc[train]["target"].astype(int)
        y_test = work.iloc[test]["target"].astype(int)
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue
        splits.append((q, train, test))
    return splits


def _make_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    model_name: str,
) -> Pipeline:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])
    pre = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )

    if model_name == "logistic_l2":
        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_SEED,
        )
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            max_features="sqrt",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    else:
        raise RuntimeError(f"Unknown model: {model_name}")

    return Pipeline([("pre", pre), ("model", model)])


def evaluate_models(
    tasks: dict[str, pd.DataFrame],
    raw_numeric: list[str],
    raw_categorical: list[str],
    derived_numeric: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    feature_sets = {
        "RAW_ONLY": (
            sorted(set(raw_numeric + derived_numeric)),
            sorted(set(raw_categorical)),
        ),
        "B0_AUGMENTED": (
            sorted(set(raw_numeric + derived_numeric + list(B0_AUGMENTED_NUMERIC))),
            sorted(set(raw_categorical + list(B0_AUGMENTED_CATEGORICAL))),
        ),
    }

    for task_name, frame in tasks.items():
        if frame.empty or frame["target"].nunique() < 2:
            continue

        splits = chronological_quarter_splits(frame)
        for feature_set, (num, cat) in feature_sets.items():
            num = [c for c in num if c in frame.columns]
            cat = [c for c in cat if c in frame.columns]
            if not num and not cat:
                continue

            for model_name in ["logistic_l2", "random_forest"]:
                local_rows: list[dict[str, Any]] = []
                for test_quarter, train_idx, test_idx in splits:
                    train = frame.iloc[train_idx].copy()
                    test = frame.iloc[test_idx].copy()
                    pipe = _make_pipeline(num, cat, model_name=model_name)
                    pipe.fit(train[num + cat], train["target"].astype(int))
                    prob = pipe.predict_proba(test[num + cat])[:, 1]
                    pred = (prob >= 0.5).astype(int)
                    row = {
                        "task": task_name,
                        "feature_set": feature_set,
                        "model": model_name,
                        "test_quarter": test_quarter,
                        "train_rows": int(len(train)),
                        "test_rows": int(len(test)),
                        "test_positive_rate": float(test["target"].mean()),
                        "roc_auc": float(roc_auc_score(test["target"], prob)),
                        "average_precision": float(
                            average_precision_score(test["target"], prob)
                        ),
                        "balanced_accuracy": float(
                            balanced_accuracy_score(test["target"], pred)
                        ),
                    }
                    fold_rows.append(row)
                    local_rows.append(row)

                if local_rows:
                    r = pd.DataFrame(local_rows)
                    summary_rows.append({
                        "task": task_name,
                        "feature_set": feature_set,
                        "model": model_name,
                        "folds": int(len(r)),
                        "mean_roc_auc": float(r["roc_auc"].mean()),
                        "median_roc_auc": float(r["roc_auc"].median()),
                        "min_roc_auc": float(r["roc_auc"].min()),
                        "mean_average_precision": float(
                            r["average_precision"].mean()
                        ),
                        "mean_balanced_accuracy": float(
                            r["balanced_accuracy"].mean()
                        ),
                        "all_fold_auc_above_0_5": bool((r["roc_auc"] > 0.5).all()),
                    })

    return pd.DataFrame(fold_rows), pd.DataFrame(summary_rows)


def single_feature_cv_auc(
    frame: pd.DataFrame,
    feature: str,
) -> float | None:
    if feature not in frame.columns:
        return None
    splits = chronological_quarter_splits(frame)
    aucs: list[float] = []
    for _, train_idx, test_idx in splits:
        train = frame.iloc[train_idx]
        test = frame.iloc[test_idx]
        train_x = pd.to_numeric(train[feature], errors="coerce")
        test_x = pd.to_numeric(test[feature], errors="coerce")
        median = float(train_x.median()) if train_x.notna().any() else 0.0
        train_x = train_x.fillna(median).to_numpy().reshape(-1, 1)
        test_x = test_x.fillna(median).to_numpy().reshape(-1, 1)
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
        )
        model.fit(train_x, train["target"].astype(int))
        prob = model.predict_proba(test_x)[:, 1]
        aucs.append(float(roc_auc_score(test["target"], prob)))
    return None if not aucs else float(np.mean(aucs))


def pair_interaction_scan(
    tasks: dict[str, pd.DataFrame],
    numeric_separation: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for task_name, frame in tasks.items():
        if frame.empty or frame["target"].nunique() < 2:
            continue
        sep = numeric_separation[
            numeric_separation["task"] == task_name
        ].copy()
        if sep.empty:
            continue
        sep = sep.sort_values(
            ["auc_separation", "abs_standardized_mean_diff"],
            ascending=[False, False],
        )
        features = [
            f for f in sep["feature"].head(TOP_NUMERIC_FOR_PAIRS).tolist()
            if f in frame.columns
        ]
        pairs = list(itertools.combinations(features, 2))[:MAX_PAIR_CANDIDATES]

        single_auc = {
            feature: single_feature_cv_auc(frame, feature)
            for feature in features
        }

        splits = chronological_quarter_splits(frame)
        for a, b in pairs:
            aucs: list[float] = []
            for _, train_idx, test_idx in splits:
                train = frame.iloc[train_idx]
                test = frame.iloc[test_idx]
                med_a = pd.to_numeric(train[a], errors="coerce").median()
                med_b = pd.to_numeric(train[b], errors="coerce").median()
                med_a = 0.0 if pd.isna(med_a) else float(med_a)
                med_b = 0.0 if pd.isna(med_b) else float(med_b)

                x_train = np.column_stack([
                    pd.to_numeric(train[a], errors="coerce").fillna(med_a),
                    pd.to_numeric(train[b], errors="coerce").fillna(med_b),
                ])
                x_test = np.column_stack([
                    pd.to_numeric(test[a], errors="coerce").fillna(med_a),
                    pd.to_numeric(test[b], errors="coerce").fillna(med_b),
                ])
                model = Pipeline([
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=1000,
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ])
                model.fit(x_train, train["target"].astype(int))
                prob = model.predict_proba(x_test)[:, 1]
                aucs.append(float(roc_auc_score(test["target"], prob)))

            if not aucs:
                continue
            pair_auc = float(np.mean(aucs))
            best_single = max(
                [x for x in [single_auc.get(a), single_auc.get(b)] if x is not None],
                default=np.nan,
            )
            rows.append({
                "task": task_name,
                "feature_a": a,
                "feature_b": b,
                "folds": len(aucs),
                "mean_pair_auc": pair_auc,
                "feature_a_single_cv_auc": single_auc.get(a),
                "feature_b_single_cv_auc": single_auc.get(b),
                "best_single_cv_auc": best_single,
                "pair_synergy_vs_best_single": (
                    np.nan if not np.isfinite(best_single) else pair_auc - best_single
                ),
            })

    return pd.DataFrame(rows)


def exploratory_numeric_tree_importance(
    tasks: dict[str, pd.DataFrame],
    numeric_features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for task_name, frame in tasks.items():
        if frame.empty or frame["target"].nunique() < 2:
            continue
        features = [c for c in numeric_features if c in frame.columns]
        if not features:
            continue
        for test_quarter, train_idx, test_idx in chronological_quarter_splits(frame):
            train = frame.iloc[train_idx]
            test = frame.iloc[test_idx]
            medians = train[features].apply(pd.to_numeric, errors="coerce").median()
            x_train = train[features].apply(pd.to_numeric, errors="coerce").fillna(medians)
            x_test = test[features].apply(pd.to_numeric, errors="coerce").fillna(medians)
            y_train = train["target"].astype(int)
            y_test = test["target"].astype(int)

            model = RandomForestClassifier(
                n_estimators=500,
                max_depth=4,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                max_features="sqrt",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
            model.fit(x_train, y_train)
            perm = permutation_importance(
                model,
                x_test,
                y_test,
                scoring="roc_auc",
                n_repeats=PERMUTATION_REPEATS,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
            for feature, mean_imp, std_imp in zip(
                features,
                perm.importances_mean,
                perm.importances_std,
            ):
                rows.append({
                    "task": task_name,
                    "test_quarter": test_quarter,
                    "feature": feature,
                    "importance_mean": float(mean_imp),
                    "importance_std": float(std_imp),
                })

    if not rows:
        return pd.DataFrame()
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["task", "feature"], sort=False)
        .agg(
            folds=("test_quarter", "nunique"),
            mean_permutation_importance=("importance_mean", "mean"),
            median_permutation_importance=("importance_mean", "median"),
            min_permutation_importance=("importance_mean", "min"),
            positive_fold_rate=("importance_mean", lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )
    return summary
