from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif, mutual_info_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _std_mean_diff(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    var_pos = float(np.var(pos, ddof=1))
    var_neg = float(np.var(neg, ddof=1))
    pooled = np.sqrt(
        ((len(pos) - 1) * var_pos + (len(neg) - 1) * var_neg)
        / max(1, len(pos) + len(neg) - 2)
    )
    if not np.isfinite(pooled) or pooled == 0:
        return 0.0
    return float((np.mean(pos) - np.mean(neg)) / pooled)


def _numeric_auc(values: pd.Series, target: pd.Series) -> tuple[float, float, str]:
    x = pd.to_numeric(values, errors="coerce")
    mask = x.notna() & target.notna()
    if mask.sum() < 6 or target[mask].nunique() < 2:
        return np.nan, np.nan, "INSUFFICIENT"
    auc = float(roc_auc_score(target[mask].astype(int), x[mask]))
    separation = max(auc, 1.0 - auc)
    direction = "HIGH_POSITIVE" if auc >= 0.5 else "LOW_POSITIVE"
    return auc, separation, direction


def numeric_feature_separation(
    tasks: dict[str, pd.DataFrame],
    numeric_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    quarter_rows: list[dict[str, Any]] = []

    for task_name, frame in tasks.items():
        if frame.empty or frame["target"].nunique() < 2:
            continue
        work = frame.copy()
        work["quarter"] = pd.PeriodIndex(
            pd.to_datetime(work["snapshot_date"]), freq="Q"
        ).astype(str)

        for feature in numeric_features:
            if feature not in work.columns:
                continue
            x = pd.to_numeric(work[feature], errors="coerce")
            y = work["target"].astype(int)
            pos = x[y == 1].dropna().to_numpy(dtype=float)
            neg = x[y == 0].dropna().to_numpy(dtype=float)

            auc, sep, direction = _numeric_auc(x, y)
            mi = np.nan
            valid = x.notna()
            if valid.sum() >= 6 and y[valid].nunique() == 2:
                xv = x[valid].to_numpy().reshape(-1, 1)
                yv = y[valid].to_numpy()
                mi = float(
                    mutual_info_classif(
                        xv,
                        yv,
                        discrete_features=False,
                        random_state=20260903,
                    )[0]
                )

            rows.append({
                "task": task_name,
                "feature": feature,
                "rows": int(len(work)),
                "positive_rows": int((y == 1).sum()),
                "negative_rows": int((y == 0).sum()),
                "positive_nonnull": int(len(pos)),
                "negative_nonnull": int(len(neg)),
                "missing_rate": float(x.isna().mean()),
                "positive_mean": None if len(pos) == 0 else float(np.mean(pos)),
                "negative_mean": None if len(neg) == 0 else float(np.mean(neg)),
                "positive_median": None if len(pos) == 0 else float(np.median(pos)),
                "negative_median": None if len(neg) == 0 else float(np.median(neg)),
                "median_diff": (
                    None
                    if len(pos) == 0 or len(neg) == 0
                    else float(np.median(pos) - np.median(neg))
                ),
                "standardized_mean_diff": (
                    None if len(pos) < 2 or len(neg) < 2
                    else _std_mean_diff(pos, neg)
                ),
                "abs_standardized_mean_diff": (
                    None
                    if len(pos) < 2 or len(neg) < 2
                    else abs(_std_mean_diff(pos, neg))
                ),
                "auc_raw": auc,
                "auc_separation": sep,
                "auc_direction": direction,
                "mutual_information": mi,
            })

            for quarter, qf in work.groupby("quarter", sort=True):
                qx = pd.to_numeric(qf[feature], errors="coerce")
                qy = qf["target"].astype(int)
                qpos = qx[qy == 1].dropna().to_numpy(dtype=float)
                qneg = qx[qy == 0].dropna().to_numpy(dtype=float)
                qauc, qsep, qdirection = _numeric_auc(qx, qy)
                quarter_rows.append({
                    "task": task_name,
                    "quarter": quarter,
                    "feature": feature,
                    "rows": int(len(qf)),
                    "positive_rows": int((qy == 1).sum()),
                    "negative_rows": int((qy == 0).sum()),
                    "median_diff": (
                        None
                        if len(qpos) == 0 or len(qneg) == 0
                        else float(np.median(qpos) - np.median(qneg))
                    ),
                    "auc_raw": qauc,
                    "auc_separation": qsep,
                    "auc_direction": qdirection,
                })

    summary = pd.DataFrame(rows)
    quarter = pd.DataFrame(quarter_rows)
    if not summary.empty and not quarter.empty:
        stability_rows = []
        for (task, feature), group in quarter.groupby(["task", "feature"], sort=False):
            valid = group[group["auc_raw"].notna()].copy()
            overall_row = summary[
                (summary["task"] == task) & (summary["feature"] == feature)
            ].iloc[0]
            overall_auc = overall_row["auc_raw"]
            if pd.isna(overall_auc) or valid.empty:
                sign_stability = np.nan
            else:
                overall_high = float(overall_auc) >= 0.5
                sign_stability = float(
                    np.mean((valid["auc_raw"].astype(float) >= 0.5) == overall_high)
                )
            stability_rows.append({
                "task": task,
                "feature": feature,
                "quarter_support": int(len(valid)),
                "quarter_direction_stability": sign_stability,
                "quarter_min_auc_separation": (
                    None if valid.empty else float(valid["auc_separation"].min())
                ),
                "quarter_mean_auc_separation": (
                    None if valid.empty else float(valid["auc_separation"].mean())
                ),
            })
        stability = pd.DataFrame(stability_rows)
        summary = summary.merge(
            stability,
            on=["task", "feature"],
            how="left",
            validate="one_to_one",
        )
    return summary, quarter


def categorical_feature_separation(
    tasks: dict[str, pd.DataFrame],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []

    for task_name, frame in tasks.items():
        if frame.empty or frame["target"].nunique() < 2:
            continue
        y = frame["target"].astype(int)

        for feature in categorical_features:
            if feature not in frame.columns:
                continue
            x = frame[feature].fillna("(missing)").astype(str)
            mi = float(mutual_info_score(x, y))
            rates = (
                pd.DataFrame({"category": x, "target": y})
                .groupby("category", sort=False)
                .agg(rows=("target", "size"), positive_rate=("target", "mean"))
                .reset_index()
            )
            usable = rates[rates["rows"] >= 2].copy()
            rate_gap = (
                None
                if usable.empty
                else float(usable["positive_rate"].max() - usable["positive_rate"].min())
            )
            summary_rows.append({
                "task": task_name,
                "feature": feature,
                "rows": int(len(frame)),
                "missing_rate": float(
                    frame[feature].isna().mean()
                    if feature in frame.columns else 1.0
                ),
                "unique_categories": int(x.nunique()),
                "mutual_information": mi,
                "max_min_positive_rate_gap_min2": rate_gap,
            })

            for _, row in rates.iterrows():
                category_rows.append({
                    "task": task_name,
                    "feature": feature,
                    "category": str(row["category"]),
                    "rows": int(row["rows"]),
                    "positive_rate": float(row["positive_rate"]),
                })

    return pd.DataFrame(summary_rows), pd.DataFrame(category_rows)


def feature_redundancy(
    frame: pd.DataFrame,
    numeric_features: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    features = [f for f in numeric_features if f in frame.columns]
    if not features:
        return {}, pd.DataFrame()

    x = frame[features].apply(pd.to_numeric, errors="coerce")
    keep = [
        col for col in x.columns
        if x[col].notna().sum() >= max(10, int(0.25 * len(x)))
        and x[col].nunique(dropna=True) > 1
    ]
    x = x[keep]
    if x.empty:
        return {}, pd.DataFrame()

    corr = x.corr(method="spearman")
    pairs: list[dict[str, Any]] = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            value = corr.loc[a, b]
            if pd.notna(value) and abs(float(value)) >= 0.85:
                pairs.append({
                    "feature_a": a,
                    "feature_b": b,
                    "spearman": float(value),
                    "abs_spearman": abs(float(value)),
                })
    high_corr = pd.DataFrame(pairs)
    if not high_corr.empty:
        high_corr = high_corr.sort_values("abs_spearman", ascending=False)

    imp = SimpleImputer(strategy="median")
    scale = StandardScaler()
    z = scale.fit_transform(imp.fit_transform(x))
    pca = PCA().fit(z)
    cum = np.cumsum(pca.explained_variance_ratio_)

    def components_for(threshold: float) -> int:
        return int(np.searchsorted(cum, threshold) + 1)

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    abs_corr = upper.abs()
    summary = {
        "rows": int(len(x)),
        "numeric_features_considered": int(len(features)),
        "numeric_features_with_support": int(len(keep)),
        "median_pairwise_abs_spearman": (
            None if abs_corr.empty else float(abs_corr.median())
        ),
        "p90_pairwise_abs_spearman": (
            None if abs_corr.empty else float(abs_corr.quantile(0.90))
        ),
        "pairs_abs_spearman_ge_0_85": int(len(high_corr)),
        "pca_components_80pct": components_for(0.80),
        "pca_components_90pct": components_for(0.90),
        "pca_components_95pct": components_for(0.95),
    }
    return summary, high_corr


def error_examples(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = panel[panel["path_valid"] == True].copy()

    missed = valid[
        (~valid["current_b0_selected"].astype(bool))
        & (
            valid["clean_big_winner"].astype(bool)
            | valid["rebound_big_winner"].astype(bool)
        )
    ].copy()
    missed["miss_type"] = np.where(
        missed["current_b0_eligible"].astype(bool),
        "SELECTOR_MISS",
        "GATE_MISS",
    )
    missed = missed.sort_values(
        ["clean_big_winner", "next_open_w4_return_pct"],
        ascending=[False, False],
    )

    bad = valid[
        valid["current_b0_selected"].astype(bool)
        & valid["strict_path_failure"].fillna(False).astype(bool)
    ].copy()
    bad = bad.sort_values(
        ["path_mae_pct", "next_open_w4_return_pct"],
        ascending=[True, True],
    )
    return missed, bad
