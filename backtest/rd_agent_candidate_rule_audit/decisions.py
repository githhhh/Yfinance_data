from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PARETO_THRESHOLDS = {
    "min_return_effect_pct": 0.5,
    "min_return_ci_low_pct": 0.0,
    "min_better_fold_ratio": 0.60,
    "min_non_worse_fold_ratio": 0.75,
    "min_evaluated_folds": 3,
    "min_fold_coverage_ratio": 0.60,
    "max_drawdown_worsening_pp": 2.0,
    "max_stop_40d_diff_pp": 2.0,
    "min_profit_24_40d_diff_pp": -2.0,
    "min_cvar_20_diff_pct": -2.0,
    "min_coverage_ratio": 0.75,
    "min_trade_ratio": 0.75,
    "min_trades": 10,
    "max_industry_top_share_diff": 0.10,
    "min_capacity_cost_nonnegative_ratio": 0.67,
    "required_capacity_cost_scenarios": 9,
}


RULE_DECISION_SPECS = {
    "Status": ("Status_ACTIONABLE_increment_controlled", "B0 status soft", -1),
    "Entry Valid": ("", "B0 no entry_valid", 1),
    "Close > Trigger": ("Close_nonnegative_vs_negative", "B0 close trigger soft", 1),
    "Fresh Zone": ("Fresh_0_5_vs_other", "B0 fresh continuous", 1),
    "Entry Volume": ("Volume_1_5_vs_other", "B0 volume route-specific", 1),
    "Geometry": ("Geometry_nonfailure_vs_failure", "B0 geometry soft", -1),
    "Pullback": ("Pullback_dry_PASS_vs_FAIL", "B0 pullback dry hard", 1),
    "EPS": ("EPS_verified_25_vs_verified_below", "B0 EPS >=25 hard", 1),
    "Industry": ("", "B0 no industry cover", -1),
    "TopK": ("", "B0 top1", 1),
}


def evaluate_pareto_candidates(
    oos: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    thresholds: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = dict(PARETO_THRESHOLDS)
    if thresholds:
        bars.update(thresholds)
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for _, candidate in oos.iterrows():
        variant = str(candidate.get("variant", ""))
        if variant in {"B0_PIT_VERIFIED", "B0_REPO_EXACT", "NO_STABLE_CANDIDATE"}:
            continue
        candidate_metric = _metric_row(metrics, variant)
        baseline_variant = str(candidate.get("portfolio_baseline_variant", "B0_PIT_VERIFIED") or "B0_PIT_VERIFIED")
        folds = max(int(_number(candidate.get("folds"), 0)), 1)
        better_ratio = _number(candidate.get("better_folds"), 0) / folds
        non_worse_ratio = _number(candidate.get("non_worse_folds"), 0) / folds
        fold_coverage_ratio = _number(candidate.get("fold_coverage_ratio"), 1.0)
        scenario = _scenario_comparison(metrics, variant, baseline_variant)
        drawdown_worsening = scenario["max_drawdown_worsening_pp"]
        trade_ratio = scenario["min_trade_ratio"]
        candidate_trades = _number(candidate_metric.get("trades") if candidate_metric else None, 0)
        criteria = [
            _criterion(variant, "return_effect", candidate.get("mean_oos_diff_pct"), ">=", bars["min_return_effect_pct"]),
            _criterion(variant, "return_confidence_interval", candidate.get("return_ci_low_pct"), ">", bars["min_return_ci_low_pct"]),
            _compound_criterion(
                variant,
                "fold_direction_stability",
                {
                    "folds": folds,
                    "better_ratio": better_ratio,
                    "non_worse_ratio": non_worse_ratio,
                    "fold_coverage_ratio": fold_coverage_ratio,
                },
                folds >= bars["min_evaluated_folds"]
                and better_ratio >= bars["min_better_fold_ratio"]
                and non_worse_ratio >= bars["min_non_worse_fold_ratio"]
                and fold_coverage_ratio >= bars["min_fold_coverage_ratio"],
                (
                    f"folds>={bars['min_evaluated_folds']};better>={bars['min_better_fold_ratio']};"
                    f"non_worse>={bars['min_non_worse_fold_ratio']};coverage>={bars['min_fold_coverage_ratio']}"
                ),
            ),
            _criterion(variant, "max_drawdown", drawdown_worsening, "<=", bars["max_drawdown_worsening_pp"]),
            _criterion(variant, "stop_8_within_40d", candidate.get("stop_40d_diff_pp"), "<=", bars["max_stop_40d_diff_pp"]),
            _criterion(variant, "profit_24_within_40d", candidate.get("profit_24_40d_diff_pp"), ">=", bars["min_profit_24_40d_diff_pp"]),
            _criterion(variant, "cvar_20", candidate.get("CVaR_20_diff_pct"), ">=", bars["min_cvar_20_diff_pct"]),
            _criterion(variant, "coverage", candidate.get("coverage_ratio"), ">=", bars["min_coverage_ratio"]),
            _compound_criterion(
                variant,
                "trade_count",
                {"trade_ratio": trade_ratio, "trades": candidate_trades},
                trade_ratio >= bars["min_trade_ratio"] and candidate_trades >= bars["min_trades"],
                f"ratio>={bars['min_trade_ratio']};trades>={bars['min_trades']}",
            ),
            _criterion(
                variant,
                "industry_concentration",
                candidate.get("industry_top_share_diff"),
                "<=",
                bars["max_industry_top_share_diff"],
            ),
            _compound_criterion(
                variant,
                "capacity_cost_robustness",
                {
                    "paired_scenarios": scenario["paired_scenarios"],
                    "nonnegative_return_ratio": scenario["nonnegative_return_ratio"],
                },
                scenario["paired_scenarios"] >= bars["required_capacity_cost_scenarios"]
                and scenario["nonnegative_return_ratio"] >= bars["min_capacity_cost_nonnegative_ratio"],
                (
                    f"paired_scenarios>={bars['required_capacity_cost_scenarios']};"
                    f"nonnegative_return_ratio>={bars['min_capacity_cost_nonnegative_ratio']}"
                ),
            ),
        ]
        rows.extend(criteria)
        failed = [row["criterion"] for row in criteria if not row["passed"]]
        decisions.append(
            {
                "variant": variant,
                "pareto_pass": bool(not failed),
                "passed_criteria": len(criteria) - len(failed),
                "total_criteria": len(criteria),
                "failed_criteria": ";".join(failed),
                "decision": "VALIDATED" if not failed else "REJECTED",
                "required_rule_families": str(candidate.get("required_rule_families", "") or ""),
                "source_atomic_variants": str(candidate.get("source_atomic_variants", "") or ""),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(decisions)


def machine_rule_decisions(
    evidence: pd.DataFrame,
    contrasts: pd.DataFrame,
    fold_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    families = ["Status", "Entry Valid", "Close > Trigger", "Fresh Zone", "Entry Volume", "Geometry", "Pullback", "EPS", "Industry", "TopK"]
    fold_results = fold_results if fold_results is not None else pd.DataFrame()
    rows = []
    for family in families:
        family_evidence = evidence[evidence["rule_family"].eq(family)] if "rule_family" in evidence.columns else pd.DataFrame()
        family_contrasts = contrasts[contrasts["rule_family"].eq(family)] if "rule_family" in contrasts.columns else pd.DataFrame()
        family_folds = fold_results[fold_results["rule_family"].eq(family)] if "rule_family" in fold_results.columns else pd.DataFrame()
        row = _decide_rule_family(family, family_evidence, family_contrasts, family_folds)
        rows.append(row)
    return pd.DataFrame(rows)


def rule_answer_lines(decisions: pd.DataFrame) -> list[str]:
    lines = []
    for _, row in decisions.iterrows():
        effect = row.get("mean_effect_pct")
        ci_low = row.get("ci_low_pct")
        ci_high = row.get("ci_high_pct")
        effect_text = "effect unavailable" if pd.isna(effect) else f"effect={float(effect):.3f}%, CI=[{_fmt(ci_low)}, {_fmt(ci_high)}]"
        blocker = str(row.get("blocker", "") or "")
        suffix = f"; {blocker}" if blocker else ""
        treatment = str(row.get("decision_contrast", "") or "fold-only atomic treatment")
        atomic = str(row.get("atomic_variant", "") or "none")
        lines.append(
            f"{row['rule_family']}: {row['evidence_status']} / {row['machine_role']}; "
            f"treatment={treatment}; atomic={atomic}; {effect_text}{suffix}."
        )
    return lines


def production_change_supported(rule_decisions: pd.DataFrame, pareto_decisions: pd.DataFrame) -> bool:
    if rule_decisions.empty or pareto_decisions.empty:
        return False
    supported_families = set(
        rule_decisions.loc[rule_decisions["production_change"].map(bool), "rule_family"].astype(str)
    )
    for _, candidate in pareto_decisions[pareto_decisions["pareto_pass"].map(bool)].iterrows():
        required = {
            value.strip()
            for value in str(candidate.get("required_rule_families", "") or "").split(";")
            if value.strip()
        }
        if required and required.issubset(supported_families):
            return True
    return False


def best_balanced_candidate(
    pareto_decisions: pd.DataFrame,
    oos: pd.DataFrame,
    metrics: pd.DataFrame,
) -> str:
    if pareto_decisions.empty:
        return "NO_STABLE_CANDIDATE"
    passed = pareto_decisions[pareto_decisions["pareto_pass"].map(bool)]
    if passed.empty:
        return "NO_STABLE_CANDIDATE"
    eligible = oos[oos["variant"].isin(passed["variant"])].copy()
    if eligible.empty:
        return "NO_STABLE_CANDIDATE"
    rankings = []
    for _, candidate in eligible.iterrows():
        variant = str(candidate["variant"])
        baseline_variant = str(candidate.get("portfolio_baseline_variant", "B0_PIT_VERIFIED") or "B0_PIT_VERIFIED")
        scenario = _scenario_comparison(metrics, variant, baseline_variant)
        candidate_metric = _metric_row(metrics, variant)
        baseline_metric = _metric_row(metrics, baseline_variant)
        rankings.append(
            {
                "variant": variant,
                "max_drawdown_worsening_pp": scenario["max_drawdown_worsening_pp"],
                "stop_40d_diff_pp": _number(candidate.get("stop_40d_diff_pp"), np.inf),
                "CVaR_20_diff_pct": _number(candidate.get("CVaR_20_diff_pct"), -np.inf),
                "industry_top_share_diff": _number(candidate.get("industry_top_share_diff"), np.inf),
                "min_trade_ratio": scenario["min_trade_ratio"],
                "portfolio_return_diff_pct": _number(candidate_metric.get("total_return_pct"), -np.inf)
                - _number(baseline_metric.get("total_return_pct"), np.inf),
                "return_ci_low_pct": _number(candidate.get("return_ci_low_pct"), -np.inf),
                "mean_oos_diff_pct": _number(candidate.get("mean_oos_diff_pct"), -np.inf),
            }
        )
    ranking = pd.DataFrame(rankings).sort_values(
        [
            "max_drawdown_worsening_pp",
            "stop_40d_diff_pp",
            "CVaR_20_diff_pct",
            "industry_top_share_diff",
            "min_trade_ratio",
            "portfolio_return_diff_pct",
            "return_ci_low_pct",
            "mean_oos_diff_pct",
        ],
        ascending=[True, True, False, True, False, False, False, False],
        na_position="last",
    )
    return str(ranking.iloc[0]["variant"])


def _decide_rule_family(
    family: str,
    evidence: pd.DataFrame,
    contrasts: pd.DataFrame,
    folds: pd.DataFrame,
) -> dict[str, Any]:
    decision_contrast, atomic_variant, fold_polarity = RULE_DECISION_SPECS[family]
    complete = int(pd.to_numeric(evidence.get("complete_8w", pd.Series(dtype=float)), errors="coerce").max()) if not evidence.empty else 0
    weeks_column = "mature_weeks" if "mature_weeks" in evidence.columns else "weeks"
    weeks = int(pd.to_numeric(evidence.get(weeks_column, pd.Series(dtype=float)), errors="coerce").max()) if not evidence.empty else 0
    registered_contrast = _registered_contrast(contrasts, decision_contrast)
    registered_folds = _registered_folds(folds, atomic_variant)
    fold_ok = registered_folds[registered_folds["treatment_contrast"].eq("OK")] if "treatment_contrast" in registered_folds.columns else registered_folds
    blocker = ""
    if not registered_contrast.empty:
        contrast_status = str(registered_contrast.iloc[0].get("status", ""))
        blocker = str(registered_contrast.iloc[0].get("blocker", "") or "")
        if contrast_status in {"RULE_NOT_IDENTIFIABLE", "NO_TREATMENT_CONTRAST"}:
            return _decision_row(
                family,
                "UNKNOWN",
                "NOT_IDENTIFIABLE",
                complete,
                weeks,
                blocker=blocker or "registered treatment is not identifiable",
                decision_contrast=decision_contrast,
                atomic_variant=atomic_variant,
                fold_polarity=fold_polarity,
            )
        if contrast_status == "INSUFFICIENT_EVIDENCE":
            return _decision_row(
                family,
                "UNKNOWN",
                "INSUFFICIENT_EVIDENCE",
                complete,
                weeks,
                blocker=blocker or "registered treatment has insufficient evidence",
                decision_contrast=decision_contrast,
                atomic_variant=atomic_variant,
                fold_polarity=fold_polarity,
            )
        row = registered_contrast.iloc[0]
        mean_effect = _number(row.get("mean_return_diff_pct"), np.nan)
        ci_low = _number(row.get("ci_low"), np.nan)
        ci_high = _number(row.get("ci_high"), np.nan)
        stop_diff = _number(row.get("stop_rate_diff"), np.nan)
        profit_diff = _number(row.get("profit_24_rate_diff"), np.nan)
        paired = int(_number(row.get("paired_week_routes"), 0))
    elif not fold_ok.empty:
        mean_raw = pd.to_numeric(fold_ok["mean_return_diff_pct"], errors="coerce").mean()
        low_raw = pd.to_numeric(fold_ok["return_ci_low_pct"], errors="coerce").min()
        high_raw = pd.to_numeric(fold_ok["return_ci_high_pct"], errors="coerce").max()
        mean_effect = fold_polarity * mean_raw
        ci_low = fold_polarity * (low_raw if fold_polarity > 0 else high_raw)
        ci_high = fold_polarity * (high_raw if fold_polarity > 0 else low_raw)
        stop_diff = fold_polarity * pd.to_numeric(fold_ok["stop_40d_diff_pp"], errors="coerce").mean() / 100.0
        profit_diff = fold_polarity * pd.to_numeric(fold_ok["profit_24_40d_diff_pp"], errors="coerce").mean() / 100.0
        paired = int(pd.to_numeric(fold_ok.get("paired_weeks", pd.Series(dtype=float)), errors="coerce").sum())
        complete = int(pd.to_numeric(fold_ok.get("complete_outcomes", pd.Series(dtype=float)), errors="coerce").sum())
        weeks = paired
    elif not registered_folds.empty:
        return _decision_row(
            family,
            "UNKNOWN",
            "NOT_IDENTIFIABLE",
            complete,
            weeks,
            blocker="registered atomic selector produced no treatment contrast",
            decision_contrast=decision_contrast,
            atomic_variant=atomic_variant,
            fold_polarity=fold_polarity,
        )
    else:
        return _decision_row(
            family,
            "UNKNOWN",
            "INSUFFICIENT_EVIDENCE",
            complete,
            weeks,
            blocker="registered treatment has no blocked retrospective folds",
            decision_contrast=decision_contrast,
            atomic_variant=atomic_variant,
            fold_polarity=fold_polarity,
        )

    directions = registered_folds.get("fold_direction", pd.Series(dtype=str)).astype(str)
    if fold_polarity < 0:
        directions = directions.replace({"better": "worse", "worse": "better"})
    better_folds = int(directions.eq("better").sum())
    evaluated_folds = int(directions.isin(["better", "worse", "equal"]).sum())
    better_ratio = better_folds / evaluated_folds if evaluated_folds else np.nan
    prospective = bool(registered_folds.get("evaluation_type", pd.Series(dtype=str)).eq("prospective_holdout").any())
    risk_observed = pd.notna(stop_diff) and pd.notna(profit_diff)
    risk_ok = risk_observed and stop_diff <= 0.02 and profit_diff >= -0.02

    if complete < 60 or weeks < 8 or paired < 5 or evaluated_folds < 3:
        status = "INSUFFICIENT_EVIDENCE"
        role = "UNKNOWN"
        blocker = "insufficient independent weeks, paired week-route contrasts, or retrospective folds"
    elif not risk_observed:
        status = "INSUFFICIENT_EVIDENCE"
        role = "UNKNOWN"
        blocker = "registered 40-day stop/profit risk evidence is missing"
    elif not risk_ok:
        status = "REJECTED"
        role = "UNKNOWN"
        blocker = "registered 40-day stop/profit risk bar failed"
    elif pd.notna(ci_high) and ci_high < 0:
        status = "REJECTED"
        role = "UNKNOWN"
        blocker = "confidence interval is entirely negative"
    elif pd.notna(mean_effect) and mean_effect > 0 and pd.notna(ci_low) and ci_low > 0 and risk_ok and (pd.isna(better_ratio) or better_ratio >= 0.60):
        status = "VALIDATED"
        role = _validated_role(family, stop_diff, profit_diff)
    elif pd.notna(mean_effect) and mean_effect > 0 and risk_ok and (pd.isna(better_ratio) or better_ratio >= 0.50):
        status = "PROMISING_NEEDS_CONFIRMATION"
        role = _promising_role(family)
        blocker = "confidence interval crosses zero or fold evidence is not strong enough"
    elif pd.notna(mean_effect) and mean_effect < 0:
        status = "REJECTED"
        role = "UNKNOWN"
        blocker = "registered contrast has negative mean effect"
    else:
        status = "CONTEXT_ONLY"
        role = "Context Only"
        blocker = "mixed or near-zero incremental evidence"
    return _decision_row(
        family,
        role,
        status,
        complete,
        weeks,
        mean_effect=mean_effect,
        ci_low=ci_low,
        ci_high=ci_high,
        stop_diff=stop_diff,
        profit_diff=profit_diff,
        paired=paired,
        evaluated_folds=evaluated_folds,
        better_folds=better_folds,
        blocker=blocker,
        production_change=status == "VALIDATED" and prospective,
        prospective_confirmed=prospective,
        decision_contrast=decision_contrast,
        atomic_variant=atomic_variant,
        fold_polarity=fold_polarity,
    )


def _registered_contrast(contrasts: pd.DataFrame, name: str) -> pd.DataFrame:
    if contrasts.empty or not name:
        return pd.DataFrame()
    if "contrast" not in contrasts.columns:
        return contrasts.head(1)
    return contrasts[contrasts["contrast"].eq(name)].head(1)


def _registered_folds(folds: pd.DataFrame, variant: str) -> pd.DataFrame:
    if folds.empty:
        return pd.DataFrame()
    if "variant" not in folds.columns:
        return folds
    return folds[folds["variant"].eq(variant)]


def _decision_row(
    family: str,
    role: str,
    status: str,
    complete: int,
    weeks: int,
    *,
    mean_effect: float = np.nan,
    ci_low: float = np.nan,
    ci_high: float = np.nan,
    stop_diff: float = np.nan,
    profit_diff: float = np.nan,
    paired: int = 0,
    evaluated_folds: int = 0,
    better_folds: int = 0,
    blocker: str = "",
    production_change: bool = False,
    prospective_confirmed: bool = False,
    decision_contrast: str = "",
    atomic_variant: str = "",
    fold_polarity: int = 1,
) -> dict[str, Any]:
    return {
        "rule_family": family,
        "machine_role": role,
        "evidence_status": status,
        "complete_outcomes": complete,
        "independent_weeks": weeks,
        "paired_week_routes": paired,
        "mean_effect_pct": mean_effect,
        "ci_low_pct": ci_low,
        "ci_high_pct": ci_high,
        "stop_rate_diff": stop_diff,
        "profit_24_rate_diff": profit_diff,
        "evaluated_folds": evaluated_folds,
        "better_folds": better_folds,
        "prospective_confirmed": prospective_confirmed,
        "decision_contrast": decision_contrast,
        "atomic_variant": atomic_variant,
        "fold_polarity": fold_polarity,
        "blocker": blocker,
        "production_change": bool(production_change),
    }


def _validated_role(family: str, stop_diff: float, profit_diff: float) -> str:
    if family in {"Geometry", "Pullback"}:
        return "Risk Flag" if pd.notna(stop_diff) and stop_diff <= 0 else "Route-specific"
    if family == "Status" and pd.notna(stop_diff) and stop_diff < 0 and (pd.isna(profit_diff) or profit_diff >= 0):
        return "Hard Eligibility"
    if family in {"Fresh Zone", "Entry Volume"}:
        return "Major Score"
    if family == "EPS":
        return "Minor Bonus"
    return "Context Only"


def _promising_role(family: str) -> str:
    if family in {"Geometry", "Pullback"}:
        return "Risk Flag" if family == "Geometry" else "Route-specific"
    if family in {"Fresh Zone", "Entry Volume", "Status"}:
        return "Major Score"
    if family == "EPS":
        return "Minor Bonus"
    return "Context Only"


def _criterion(variant: str, name: str, value: Any, operator: str, threshold: float) -> dict[str, Any]:
    numeric = _number(value, np.nan)
    if operator == ">=":
        passed = pd.notna(numeric) and numeric >= threshold
    elif operator == ">":
        passed = pd.notna(numeric) and numeric > threshold
    elif operator == "<=":
        passed = pd.notna(numeric) and numeric <= threshold
    else:
        raise ValueError(operator)
    return {
        "variant": variant,
        "criterion": name,
        "observed_value": numeric,
        "threshold": f"{operator}{threshold}",
        "passed": bool(passed),
        "reason": "PASS" if passed else "FAIL_OR_MISSING",
    }


def _compound_criterion(variant: str, name: str, value: dict[str, float], passed: bool, threshold: str) -> dict[str, Any]:
    return {
        "variant": variant,
        "criterion": name,
        "observed_value": str(value),
        "threshold": threshold,
        "passed": bool(passed),
        "reason": "PASS" if passed else "FAIL_OR_MISSING",
    }


def _metric_row(metrics: pd.DataFrame, variant: str) -> dict[str, Any]:
    if metrics.empty:
        return {}
    rows = metrics[
        metrics["variant"].eq(variant)
        & metrics["capacity"].eq(3)
        & metrics["cost_bps_per_side"].eq(10)
    ]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _scenario_comparison(metrics: pd.DataFrame, variant: str, baseline_variant: str) -> dict[str, float]:
    candidate = metrics[metrics["variant"].eq(variant)].copy()
    baseline = metrics[metrics["variant"].eq(baseline_variant)].copy()
    if candidate.empty or baseline.empty:
        return {
            "max_drawdown_worsening_pp": np.nan,
            "min_trade_ratio": np.nan,
            "nonnegative_return_ratio": np.nan,
            "paired_scenarios": 0,
        }
    paired = candidate.merge(
        baseline,
        on=["capacity", "cost_bps_per_side"],
        suffixes=("_candidate", "_baseline"),
    )
    if paired.empty:
        return {
            "max_drawdown_worsening_pp": np.nan,
            "min_trade_ratio": np.nan,
            "nonnegative_return_ratio": np.nan,
            "paired_scenarios": 0,
        }
    drawdown = (
        pd.to_numeric(paired["max_drawdown_pct_candidate"], errors="coerce").abs()
        - pd.to_numeric(paired["max_drawdown_pct_baseline"], errors="coerce").abs()
    )
    baseline_trades = pd.to_numeric(paired["trades_baseline"], errors="coerce")
    trade_ratio = pd.to_numeric(paired["trades_candidate"], errors="coerce") / baseline_trades.replace(0, np.nan)
    return_diff = (
        pd.to_numeric(paired["total_return_pct_candidate"], errors="coerce")
        - pd.to_numeric(paired["total_return_pct_baseline"], errors="coerce")
    )
    return {
        "max_drawdown_worsening_pp": float(drawdown.max()) if drawdown.notna().any() else np.nan,
        "min_trade_ratio": float(trade_ratio.min()) if trade_ratio.notna().any() else np.nan,
        "nonnegative_return_ratio": float(return_diff.ge(0).mean()) if return_diff.notna().any() else np.nan,
        "paired_scenarios": int(len(paired)),
    }


def _number(value: Any, default: float) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _fmt(value: Any) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.3f}"
