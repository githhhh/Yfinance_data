from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from dashboard.skill_industry_eps_known import select_skill_industry_eps_known

from .config import FEATURE_MANIFEST_PATH


def _allowed_pit_columns(panel_df: pd.DataFrame) -> list[str]:
    manifest = json.loads(FEATURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    return [
        name for name, meta in manifest["features"].items()
        if meta.get("allowed_for_discovery") is True and name in panel_df.columns
    ]


def _anon(value: str, prefix: str) -> str:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def build_failure_archaeology(
    panel_df: pd.DataFrame,
    discovery_snapshots: list[str],
    horizon: str = "W4",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build anonymized, discovery-train-only B0 error cases.

    Outcomes are deliberately revealed here only after Phase 0 protocol freeze and
    only for the discovery-train prefix. These labels are never exposed from outer
    forward blocks before policy freeze.
    """
    ret_col = f"{horizon.lower()}_return_pct"
    stop_col = f"{horizon.lower()}_stop8"
    allowed = _allowed_pit_columns(panel_df)

    cases: list[dict[str, Any]] = []
    for snap in discovery_snapshots:
        s_df = panel_df[panel_df["snapshot_date"].astype(str) == str(snap)].copy()
        if s_df.empty:
            continue
        eligible = s_df[s_df["b0_eligible"].fillna(False).astype(bool)].copy()
        eligible[ret_col] = pd.to_numeric(eligible.get(ret_col), errors="coerce")
        eligible = eligible[eligible[ret_col].notna()].copy()
        if eligible.empty:
            continue

        picked = {x.code for x in select_skill_industry_eps_known(s_df, limit=3)}
        picked_rows = eligible[eligible["code"].astype(str).isin(picked)]
        picked_rets = picked_rows[ret_col].dropna()
        picked_median = float(picked_rets.median()) if not picked_rets.empty else 0.0
        worst_picked = float(picked_rets.min()) if not picked_rets.empty else 0.0

        for _, row in eligible.iterrows():
            code = str(row["code"])
            ret = float(row[ret_col])
            stop = bool(row.get(stop_col, False)) if not pd.isna(row.get(stop_col, False)) else False
            is_pick = code in picked

            labels: list[str] = []
            if is_pick and (ret <= -8.0 or stop):
                labels.append("false_positive_ruin")
            if is_pick and ret >= 8.0:
                labels.append("selected_winner")
            if (not is_pick) and ret >= 8.0 and ret >= picked_median + 5.0:
                labels.append("false_negative_big_winner")
            if (not is_pick) and ret >= worst_picked + 5.0:
                labels.append("false_negative_relative")
            if not labels:
                continue

            base = {
                "snapshot_anon": _anon(str(snap), "snap"),
                "entity_anon": _anon(f"{snap}:{code}", "entity"),
                "is_b0_pick": is_pick,
                "outcome_label": "|".join(sorted(set(labels))),
                "w4_return_bucket": (
                    "ruin" if ret <= -8.0 else
                    "loss" if ret < 0.0 else
                    "small_win" if ret < 8.0 else
                    "large_win"
                ),
            }
            for col in allowed:
                value = row.get(col)
                if isinstance(value, (np.generic,)):
                    value = value.item()
                if pd.isna(value):
                    value = None
                base[col] = value
            cases.append(base)

    case_df = pd.DataFrame(cases)
    if case_df.empty:
        return case_df, {"case_count": 0, "labels": {}, "numeric_contrasts": []}

    labels = case_df["outcome_label"].value_counts().to_dict()
    numeric = [
        c for c in allowed
        if c in case_df.columns and pd.api.types.is_numeric_dtype(case_df[c])
    ]

    contrasts = []
    false_neg = case_df[case_df["outcome_label"].str.contains("false_negative", na=False)]
    selected_win = case_df[case_df["outcome_label"].str.contains("selected_winner", na=False)]
    for col in numeric:
        a = pd.to_numeric(false_neg[col], errors="coerce").dropna()
        b = pd.to_numeric(selected_win[col], errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        pooled = pd.concat([a, b])
        scale = float(pooled.std(ddof=0))
        delta = float(a.median() - b.median())
        standardized = delta / scale if scale > 1e-9 else 0.0
        contrasts.append({
            "feature": col,
            "false_negative_median": round(float(a.median()), 6),
            "selected_winner_median": round(float(b.median()), 6),
            "median_delta": round(delta, 6),
            "standardized_delta": round(standardized, 4),
        })
    contrasts.sort(key=lambda x: abs(x["standardized_delta"]), reverse=True)

    example_features = [x["feature"] for x in contrasts[:10]]
    example_cols = [
        "snapshot_anon", "entity_anon", "is_b0_pick", "outcome_label", "w4_return_bucket",
        *[x for x in example_features if x in case_df.columns],
    ]
    examples = (
        case_df[example_cols]
        .head(30)
        .replace({np.nan: None})
        .to_dict(orient="records")
    )

    summary = {
        "case_count": int(len(case_df)),
        "labels": {str(k): int(v) for k, v in labels.items()},
        "numeric_contrasts": contrasts[:25],
        "case_examples": examples,
        "method_note": (
            "Discovery-train-only anonymized error archaeology. False negatives require "
            "future W4 >=8% and >=5pp above B0 selected median, or >=5pp above worst B0 pick."
        ),
    }
    return case_df, summary
