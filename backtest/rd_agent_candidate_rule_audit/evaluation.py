from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from .selectors import (
    SelectorConfig,
    atomic_selector_configs,
    compose_selector_config,
    select_all_weeks,
    selector_configs,
)
from .stats import RollingSplit, ci_from_samples, moving_time_block_bootstrap, week_block_bootstrap
from .utils import object_hash


ATOMIC_TRAIN_THRESHOLDS = {
    "min_mature_weeks": 8,
    "min_independent_8w_time_blocks": 3,
    "min_paired_weeks": 6,
    "min_mean_return_diff_pct": 0.5,
    "validated_ci_low_pct": 0.0,
    "max_stop_40d_diff_pp": 2.0,
    "min_profit_24_40d_diff_pp": -2.0,
    "min_coverage_ratio": 0.75,
    "max_industry_top_share_diff": 0.10,
}


RULE_FAMILY_BY_DIMENSION = {
    "status": "Status",
    "entry_valid": "Entry Valid",
    "close_trigger": "Close > Trigger",
    "fresh": "Fresh Zone",
    "volume": "Entry Volume",
    "geometry": "Geometry",
    "pullback_dry": "Pullback",
    "eps": "EPS",
    "industry_cover": "Industry",
    "topk": "TopK",
}


def propose_fold_candidates(
    panel: pd.DataFrame,
    split: RollingSplit,
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    dates = pd.to_datetime(panel["snapshot_date"])
    train = panel[dates.between(split.train_start, split.train_end)].copy()
    atomic_evidence = evaluate_atomic_training(
        train,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    proposed_configs = _generate_candidate_configs(atomic_evidence)
    baseline = select_all_weeks(train, selector_configs()["B0_PIT_VERIFIED"])
    composite_rows: list[dict[str, Any]] = []
    candidate_configs: list[SelectorConfig] = []
    for offset, config in enumerate(proposed_configs):
        comparison = compare_selected_outcomes(
            baseline,
            select_all_weeks(train, config),
            seed=seed + 10_000 + offset,
            bootstrap_iterations=bootstrap_iterations,
        )
        comparison.update(
            {
                "variant": config.name,
                "target_rule": "rule_set",
                "rule_family": "Rule Set",
                "source_atomic_variants": ";".join(config.source_atomic_variants),
            }
        )
        comparison["train_decision"] = _atomic_train_decision(comparison)
        composite_rows.append(comparison)
        if comparison["train_decision"] == "VALIDATED":
            candidate_configs.append(config)
    evidence = pd.concat(
        [atomic_evidence, pd.DataFrame(composite_rows)],
        ignore_index=True,
        sort=False,
    )
    frozen_records = [
        {
            "variant": config.name,
            "atomic_rules": list(config.source_atomic_variants),
            "config": asdict(config),
        }
        for config in candidate_configs
    ]
    if not frozen_records:
        frozen_records = [{"variant": "NO_STABLE_CANDIDATE", "atomic_rules": [], "config": None}]
    frozen_json = json.dumps(frozen_records, sort_keys=True, separators=(",", ":"))
    max_date = pd.to_datetime(train["snapshot_date"]).max() if not train.empty else split.train_end
    return {
        "train_start": str(split.train_start.date()),
        "train_end": str(split.train_end.date()),
        "train_evidence_max_date": str(pd.Timestamp(max_date).date()),
        "train_evidence": evidence,
        "train_evidence_json": evidence.to_json(orient="records", date_format="iso"),
        "candidate_configs": candidate_configs,
        "frozen_rules_json": frozen_json,
        "frozen_rule_hash": object_hash(frozen_records),
        "candidate_generation_status": "FROZEN_CANDIDATES" if candidate_configs else "NO_STABLE_CANDIDATE",
    }


def evaluate_atomic_training(
    train: pd.DataFrame,
    *,
    bootstrap_iterations: int,
    seed: int,
) -> pd.DataFrame:
    base_config = selector_configs()["B0_PIT_VERIFIED"]
    baseline = select_all_weeks(train, base_config)
    rows = []
    for offset, (name, config) in enumerate(atomic_selector_configs().items()):
        selected = select_all_weeks(train, config)
        comparison = compare_selected_outcomes(
            baseline,
            selected,
            seed=seed + offset,
            bootstrap_iterations=bootstrap_iterations,
        )
        comparison.update(
            {
                "variant": name,
                "target_rule": config.changed_rules[0],
                "rule_family": RULE_FAMILY_BY_DIMENSION[config.changed_rules[0]],
            }
        )
        comparison["train_decision"] = _atomic_train_decision(comparison)
        rows.append(comparison)
    return pd.DataFrame(rows)


def compare_selected_outcomes(
    baseline: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    seed: int,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    baseline_keys = _pick_keys(baseline)
    selected_keys = _pick_keys(selected)
    treatment_contrast = "OK" if baseline_keys != selected_keys else "NO_TREATMENT_CONTRAST"
    paired = _paired_week_returns(baseline, selected)
    samples = week_block_bootstrap(
        paired.rename(columns={"return_diff_pct": "effect"}),
        value_col="effect",
        seed=seed,
        iterations=bootstrap_iterations,
        time_block_weeks=8,
    )
    ci_low, ci_high = ci_from_samples(samples)
    baseline_complete = _complete_rows(baseline)
    selected_complete = _complete_rows(selected)
    mature_weeks = int(pd.to_datetime(selected_complete.get("snapshot_date", pd.Series(dtype=str))).nunique())
    return {
        "treatment_contrast": treatment_contrast,
        "selected_count": len(selected),
        "baseline_selected_count": len(baseline),
        "added": len(selected_keys - baseline_keys),
        "removed": len(baseline_keys - selected_keys),
        "rank_changed": _rank_changes(baseline, selected),
        "affected_weeks": _affected_weeks(baseline, selected),
        "mature_weeks": mature_weeks,
        "independent_8w_time_blocks": mature_weeks // 8,
        "complete_outcomes": len(selected_complete),
        "paired_weeks": len(paired),
        "mean_return_diff_pct": _finite_or_nan(paired["return_diff_pct"].mean() if not paired.empty else np.nan),
        "median_return_diff_pct": _finite_or_nan(paired["return_diff_pct"].median() if not paired.empty else np.nan),
        "return_ci_low_pct": ci_low,
        "return_ci_high_pct": ci_high,
        "stop_40d_diff_pp": _rate_diff_pp(selected_complete, baseline_complete, "stop_8_within_40d"),
        "profit_24_40d_diff_pp": _rate_diff_pp(selected_complete, baseline_complete, "profit_24_within_40d"),
        "coverage_ratio": len(selected_complete) / len(baseline_complete) if len(baseline_complete) else np.nan,
        "industry_top_share": _industry_top_share(selected),
        "baseline_industry_top_share": _industry_top_share(baseline),
        "industry_top_share_diff": _industry_top_share(selected) - _industry_top_share(baseline),
    }


def evaluate_frozen_config(
    test: pd.DataFrame,
    config: SelectorConfig,
    *,
    fold: int,
    split: RollingSplit,
    proposal: dict[str, Any],
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    baseline = select_all_weeks(test, selector_configs()["B0_PIT_VERIFIED"])
    selected = select_all_weeks(test, config)
    result = compare_selected_outcomes(
        baseline,
        selected,
        seed=seed,
        bootstrap_iterations=bootstrap_iterations,
    )
    paired = _paired_week_returns(baseline, selected)
    frozen_payload = {
        "variant": config.name,
        "atomic_rules": list(config.source_atomic_variants),
        "config": asdict(config),
    }
    result.update(
        {
            "fold": fold,
            "variant": config.name,
            "rule_family": RULE_FAMILY_BY_DIMENSION.get(config.changed_rules[0], "Rule Set")
            if len(config.changed_rules) == 1
            else "Rule Set",
            "train_start": str(split.train_start.date()),
            "train_end": str(split.train_end.date()),
            "test_start": str(split.test_start.date()),
            "test_end": str(split.test_end.date()),
            "evaluation_type": "blocked_retrospective_evaluation",
            "train_evidence_json": proposal["train_evidence_json"],
            "frozen_rules_json": json.dumps([frozen_payload], sort_keys=True, separators=(",", ":")),
            "frozen_rule_hash": object_hash(frozen_payload),
            "fold_direction": _fold_direction(paired),
            "worst_week_diff_pct": paired["return_diff_pct"].min() if not paired.empty else np.nan,
            "CVaR_20_diff_pct": _cvar(paired["return_diff_pct"]) if not paired.empty else np.nan,
            "frozen_rule_dimensions": ";".join(config.changed_rules),
            "frozen_rule_families": ";".join(
                RULE_FAMILY_BY_DIMENSION[dimension]
                for dimension in config.changed_rules
            ),
            "source_atomic_variants": ";".join(config.source_atomic_variants),
        }
    )
    return result


def summarize_blocked_results(fold_results: pd.DataFrame, *, seed: int, bootstrap_iterations: int) -> pd.DataFrame:
    if fold_results.empty:
        return pd.DataFrame()
    rows = []
    for variant, group in fold_results.groupby("variant", sort=True):
        effects = pd.to_numeric(group["mean_return_diff_pct"], errors="coerce").dropna()
        bootstrap_frame = group.loc[effects.index, ["fold"]].copy()
        bootstrap_frame["effect"] = effects
        samples = moving_time_block_bootstrap(
            bootstrap_frame,
            value_col="effect",
            order_col="fold",
            block_size=2,
            seed=seed,
            iterations=bootstrap_iterations,
        )
        ci_low, ci_high = ci_from_samples(samples)
        folds = int(group["fold"].nunique())
        rows.append(
            {
                "variant": variant,
                "folds": folds,
                "paired_weeks": int(pd.to_numeric(group["paired_weeks"], errors="coerce").sum()),
                "mean_oos_diff_pct": effects.mean() if len(effects) else np.nan,
                "median_oos_diff_pct": effects.median() if len(effects) else np.nan,
                "return_ci_low_pct": ci_low,
                "return_ci_high_pct": ci_high,
                "worst_fold_diff_pct": effects.min() if len(effects) else np.nan,
                "better_folds": int(group["fold_direction"].eq("better").sum()),
                "non_worse_folds": int(group["fold_direction"].isin(["better", "equal"]).sum()),
                "stop_40d_diff_pp": pd.to_numeric(group["stop_40d_diff_pp"], errors="coerce").mean(),
                "profit_24_40d_diff_pp": pd.to_numeric(group["profit_24_40d_diff_pp"], errors="coerce").mean(),
                "CVaR_20_diff_pct": pd.to_numeric(group["CVaR_20_diff_pct"], errors="coerce").mean(),
                "coverage_ratio": pd.to_numeric(group["coverage_ratio"], errors="coerce").mean(),
                "industry_top_share_diff": pd.to_numeric(group["industry_top_share_diff"], errors="coerce").mean(),
                "frozen_rule_hashes": int(group["frozen_rule_hash"].nunique()),
                "ci_time_block_weeks": 8,
                "ci_block_folds": 2,
                "required_rule_families": _union_semicolon_values(group["frozen_rule_families"]),
                "source_atomic_variants": _union_semicolon_values(group["source_atomic_variants"]),
                "portfolio_baseline_variant": f"B0_FOR_{variant}_BLOCKED",
                "evaluation_type": "blocked_retrospective_evaluation",
            }
        )
    return pd.DataFrame(rows)


def _generate_candidate_configs(evidence: pd.DataFrame) -> list[SelectorConfig]:
    if evidence.empty:
        return []
    validated = evidence[evidence["train_decision"].eq("VALIDATED")].copy()
    if validated.empty:
        return []
    validated = validated.sort_values(["mean_return_diff_pct", "return_ci_low_pct"], ascending=False)
    # One modification per rule family avoids contradictory configurations.
    validated = validated.drop_duplicates("target_rule", keep="first")
    names = validated["variant"].astype(str).tolist()
    candidates = [compose_selector_config("R1_ATOMIC_IMPROVEMENTS", names)]
    low_risk = validated[
        pd.to_numeric(validated["stop_40d_diff_pp"], errors="coerce").le(0)
        & pd.to_numeric(validated["industry_top_share_diff"], errors="coerce").le(0.05)
    ]
    if not low_risk.empty and low_risk["variant"].tolist() != names:
        candidates.append(compose_selector_config("R2_BALANCED_SOFT", low_risk["variant"].astype(str).tolist()))
    technical = validated[validated["target_rule"].isin(["fresh", "volume", "geometry", "pullback_dry"])]
    if not technical.empty:
        strongest = [str(technical.iloc[0]["variant"])]
        if strongest != names and all(config.name != "R3_MINIMAL_TECHNICAL" for config in candidates):
            candidates.append(compose_selector_config("R3_MINIMAL_TECHNICAL", strongest))
    return candidates[:3]


def _atomic_train_decision(row: dict[str, Any]) -> str:
    if row["treatment_contrast"] != "OK":
        return "NOT_IDENTIFIABLE"
    thresholds = ATOMIC_TRAIN_THRESHOLDS
    if (
        row["mature_weeks"] < thresholds["min_mature_weeks"]
        or row.get("independent_8w_time_blocks", 0) < thresholds["min_independent_8w_time_blocks"]
        or row["paired_weeks"] < thresholds["min_paired_weeks"]
    ):
        return "INSUFFICIENT_EVIDENCE"
    mean = row["mean_return_diff_pct"]
    ci_low = row["return_ci_low_pct"]
    ci_high = row["return_ci_high_pct"]
    risk_ok = (
        row["stop_40d_diff_pp"] <= thresholds["max_stop_40d_diff_pp"]
        and row["profit_24_40d_diff_pp"] >= thresholds["min_profit_24_40d_diff_pp"]
        and row["coverage_ratio"] >= thresholds["min_coverage_ratio"]
        and row["industry_top_share_diff"] <= thresholds["max_industry_top_share_diff"]
    )
    if pd.notna(mean) and pd.notna(ci_low) and mean >= thresholds["min_mean_return_diff_pct"] and ci_low > thresholds["validated_ci_low_pct"] and risk_ok:
        return "VALIDATED"
    if pd.notna(ci_high) and ci_high < 0:
        return "REJECTED"
    if not risk_ok:
        return "REJECTED"
    if pd.notna(mean) and mean > 0:
        return "PROMISING_NEEDS_CONFIRMATION"
    return "CONTEXT_ONLY"


def _complete_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "forward_8w_censored" not in frame.columns:
        return frame.iloc[0:0].copy()
    return frame[frame["forward_8w_censored"].eq(False)].copy()


def _paired_week_returns(baseline: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    b = _complete_rows(baseline).groupby("snapshot_date")["forward_8w_return_pct"].mean().rename("baseline_return_pct")
    s = _complete_rows(selected).groupby("snapshot_date")["forward_8w_return_pct"].mean().rename("selected_return_pct")
    paired = pd.concat([b, s], axis=1).dropna().reset_index()
    if not paired.empty:
        paired["return_diff_pct"] = paired["selected_return_pct"] - paired["baseline_return_pct"]
    else:
        paired["return_diff_pct"] = pd.Series(dtype=float)
    return paired


def _pick_keys(frame: pd.DataFrame) -> set[tuple[str, str, int]]:
    if frame.empty:
        return set()
    return {
        (str(row["snapshot_date"]), str(row["code"]), int(row["pick_order"]))
        for _, row in frame.iterrows()
    }


def _rank_changes(baseline: pd.DataFrame, selected: pd.DataFrame) -> int:
    if baseline.empty or selected.empty:
        return 0
    b = baseline.set_index(["snapshot_date", "code"])["pick_order"]
    s = selected.set_index(["snapshot_date", "code"])["pick_order"]
    common = b.index.intersection(s.index)
    return int((b.loc[common].astype(int) != s.loc[common].astype(int)).sum())


def _affected_weeks(baseline: pd.DataFrame, selected: pd.DataFrame) -> int:
    weeks = set(baseline.get("snapshot_date", pd.Series(dtype=str))).union(set(selected.get("snapshot_date", pd.Series(dtype=str))))
    affected = 0
    for week in weeks:
        b = _pick_keys(baseline[baseline["snapshot_date"].eq(week)])
        s = _pick_keys(selected[selected["snapshot_date"].eq(week)])
        affected += int(b != s)
    return affected


def _rate_diff_pp(selected: pd.DataFrame, baseline: pd.DataFrame, column: str) -> float:
    if column not in selected.columns or column not in baseline.columns or selected.empty or baseline.empty:
        return np.nan
    return float((selected[column].eq(True).mean() - baseline[column].eq(True).mean()) * 100.0)


def _industry_top_share(frame: pd.DataFrame) -> float:
    if frame.empty or "industry" not in frame.columns:
        return 0.0
    values = frame["industry"].fillna("UNKNOWN").astype(str)
    return float(values.value_counts(normalize=True).iloc[0]) if len(values) else 0.0


def _fold_direction(paired: pd.DataFrame) -> str:
    if paired.empty:
        return "no_contrast"
    effect = float(paired["return_diff_pct"].mean())
    if effect > 0:
        return "better"
    if effect < 0:
        return "worse"
    return "equal"


def _cvar(values: pd.Series, quantile: float = 0.2) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    cutoff = clean.quantile(quantile)
    return float(clean[clean <= cutoff].mean())


def _finite_or_nan(value: Any) -> float:
    try:
        return float(value) if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _union_semicolon_values(values: pd.Series) -> str:
    result = set()
    for value in values.dropna().astype(str):
        result.update(part.strip() for part in value.split(";") if part.strip())
    return ";".join(sorted(result))
