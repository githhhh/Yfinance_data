from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .labels import ExitPolicy, TradeLabelConfig, build_event_labels, normalize_eps_pit
from .evaluation import (
    ATOMIC_TRAIN_THRESHOLDS,
    evaluate_frozen_config,
    propose_fold_candidates as _propose_fold_candidates,
    summarize_blocked_results,
)
from .decisions import (
    PARETO_THRESHOLDS,
    best_balanced_candidate as _best_balanced_candidate,
    evaluate_pareto_candidates as _evaluate_pareto_candidates,
    machine_rule_decisions as _machine_rule_decisions,
    production_change_supported,
    rule_answer_lines as _rule_answer_lines,
)
from .portfolio import PortfolioConfig, portfolio_metrics, run_portfolio_backtest
from .selectors import (
    PULLBACK_RULES,
    atomic_selector_configs,
    audit_atomic_variant,
    audit_production_b0_replay_pools,
    enrich_features,
    select_all_weeks,
    selector_configs,
)
from .stats import ci_from_samples, make_rolling_splits, paired_week_route_bootstrap, week_block_bootstrap
from .utils import content_hash, next_bar_after, normalize_bars, object_hash, to_bool, to_float


POOL_ROOT = Path("backtest/ibd_skill_replay_pools")
PRICE_CACHE = Path("results_pkl/stock_data_230826_1d.pkl")
OUTPUT_DIR = Path("backtest/rd_agent_candidate_rule_audit/output")
SEED = 20260824


def propose_fold_candidates(panel: pd.DataFrame, split: Any, *, bootstrap_iterations: int) -> dict[str, Any]:
    return _propose_fold_candidates(
        panel,
        split,
        bootstrap_iterations=bootstrap_iterations,
        seed=SEED,
    )


def evaluate_pareto_candidates(oos: pd.DataFrame, metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _evaluate_pareto_candidates(oos, metrics)


def rule_answer_lines(decisions: pd.DataFrame) -> list[str]:
    return _rule_answer_lines(decisions)


def best_balanced_candidate(
    pareto_decisions: pd.DataFrame,
    oos: pd.DataFrame,
    metrics: pd.DataFrame,
) -> str:
    return _best_balanced_candidate(pareto_decisions, oos, metrics)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build candidate-event IBD rule audit.")
    parser.add_argument("--pool-root", default=str(POOL_ROOT))
    parser.add_argument("--price-cache", default=str(PRICE_CACHE))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--bootstrap-iterations", type=int, default=400)
    args = parser.parse_args(argv)
    outputs = run_audit(
        pool_root=Path(args.pool_root),
        price_cache=Path(args.price_cache),
        output_dir=Path(args.output_dir),
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


def run_audit(
    *,
    pool_root: Path = POOL_ROOT,
    price_cache: Path = PRICE_CACHE,
    output_dir: Path = OUTPUT_DIR,
    bootstrap_iterations: int = 400,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pools = load_pools(pool_root)
    prices = load_price_cache(price_cache)
    eps = normalize_eps_pit(pd.read_csv(pool_root / "signal_eps_pit.csv"))
    panel, duplicate_audit = build_candidate_event_panel(pools, eps, prices)
    evidence = candidate_rule_evidence(panel, bootstrap_iterations=bootstrap_iterations)
    contrasts = rule_treatment_contrast(panel, bootstrap_iterations=bootstrap_iterations)
    selections = build_all_selections(panel)
    production_b0_invariant = audit_production_b0_replay_pools(pool_root)
    atomic_invariant = b0_atomic_invariant_audit(panel)
    ablations = b0_atomic_ablation(selections, panel, atomic_invariant)
    fold_rules, oos, blocked_picks, freeze_registry = oos_results(
        panel,
        bootstrap_iterations=bootstrap_iterations,
    )
    full_baselines = selections[selections["variant"].isin(["B0_REPO_EXACT", "B0_PIT_VERIFIED"])]
    portfolio_selections = pd.concat([full_baselines, blocked_picks], ignore_index=True) if not blocked_picks.empty else full_baselines
    trade_ledger, equity_curves, metrics = portfolio_outputs(portfolio_selections, prices)
    exit_sensitivity = exit_policy_sensitivity(selections, prices, panel)
    coverage = candidate_label_coverage(panel)
    drift = pool_coverage_drift(pools, panel)
    decisions = machine_rule_decisions(evidence, contrasts, fold_rules)
    pareto_criteria, pareto_decisions = evaluate_pareto_candidates(oos, metrics)
    balanced = best_balanced_candidate(pareto_decisions, oos, metrics)
    skill_change = production_change_supported(decisions, pareto_decisions)
    hypotheses = hypothesis_registry()
    valuation_as_of = _price_cache_as_of(prices)
    valuation_start = _selection_valuation_start(portfolio_selections, prices, valuation_as_of)
    manifest = experiment_manifest(
        pool_root,
        price_cache,
        output_dir,
        pools,
        hypotheses,
        bootstrap_iterations=bootstrap_iterations,
        valuation_start=valuation_start,
        valuation_as_of=valuation_as_of,
        freeze_registry=freeze_registry,
    )
    data_audit = render_data_audit(pools, panel, duplicate_audit, drift, coverage)
    b0_diff = render_b0_diff(ablations, oos, metrics, pareto_criteria, pareto_decisions, balanced)
    report = render_report(
        panel,
        drift,
        evidence,
        contrasts,
        ablations,
        oos,
        metrics,
        exit_sensitivity,
        decisions,
        pareto_criteria,
        pareto_decisions,
        freeze_registry,
        balanced,
        skill_change,
    )
    acceptance = render_acceptance_summary(
        panel,
        production_b0_invariant,
        atomic_invariant,
        freeze_registry,
        pareto_decisions,
        metrics,
        balanced,
        skill_change,
    )

    outputs = {
        "data_audit.md": output_dir / "data_audit.md",
        "pool_coverage_drift.csv": output_dir / "pool_coverage_drift.csv",
        "candidate_event_panel.parquet": output_dir / "candidate_event_panel.parquet",
        "candidate_event_panel.csv": output_dir / "candidate_event_panel.csv",
        "candidate_label_coverage.csv": output_dir / "candidate_label_coverage.csv",
        "hypothesis_registry.jsonl": output_dir / "hypothesis_registry.jsonl",
        "candidate_rule_evidence.csv": output_dir / "candidate_rule_evidence.csv",
        "rule_treatment_contrast.csv": output_dir / "rule_treatment_contrast.csv",
        "b0_atomic_ablation.csv": output_dir / "b0_atomic_ablation.csv",
        "b0_production_invariant_audit.csv": output_dir / "b0_production_invariant_audit.csv",
        "b0_atomic_invariant_audit.csv": output_dir / "b0_atomic_invariant_audit.csv",
        "fold_level_rule_results.csv": output_dir / "fold_level_rule_results.csv",
        "rule_set_oos_results.csv": output_dir / "rule_set_oos_results.csv",
        "fold_rule_freeze_registry.csv": output_dir / "fold_rule_freeze_registry.csv",
        "portfolio_trade_ledger.csv": output_dir / "portfolio_trade_ledger.csv",
        "portfolio_equity_curves.csv": output_dir / "portfolio_equity_curves.csv",
        "portfolio_metrics.csv": output_dir / "portfolio_metrics.csv",
        "exit_policy_sensitivity.csv": output_dir / "exit_policy_sensitivity.csv",
        "machine_rule_decisions.csv": output_dir / "machine_rule_decisions.csv",
        "pareto_criteria.csv": output_dir / "pareto_criteria.csv",
        "pareto_rule_set_decisions.csv": output_dir / "pareto_rule_set_decisions.csv",
        "b0_vs_balanced_rule_diff.md": output_dir / "b0_vs_balanced_rule_diff.md",
        "rd_agent_candidate_rule_report.md": output_dir / "rd_agent_candidate_rule_report.md",
        "experiment_manifest.yaml": output_dir / "experiment_manifest.yaml",
        "acceptance_summary.md": output_dir / "acceptance_summary.md",
    }
    outputs["data_audit.md"].write_text(data_audit, encoding="utf-8")
    drift.to_csv(outputs["pool_coverage_drift.csv"], index=False)
    _write_panel(panel, outputs["candidate_event_panel.parquet"], outputs["candidate_event_panel.csv"])
    coverage.to_csv(outputs["candidate_label_coverage.csv"], index=False)
    outputs["hypothesis_registry.jsonl"].write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in hypotheses) + "\n",
        encoding="utf-8",
    )
    evidence.to_csv(outputs["candidate_rule_evidence.csv"], index=False)
    contrasts.to_csv(outputs["rule_treatment_contrast.csv"], index=False)
    ablations.to_csv(outputs["b0_atomic_ablation.csv"], index=False)
    production_b0_invariant.to_csv(outputs["b0_production_invariant_audit.csv"], index=False)
    atomic_invariant.to_csv(outputs["b0_atomic_invariant_audit.csv"], index=False)
    fold_rules.to_csv(outputs["fold_level_rule_results.csv"], index=False)
    oos.to_csv(outputs["rule_set_oos_results.csv"], index=False)
    freeze_registry.to_csv(outputs["fold_rule_freeze_registry.csv"], index=False)
    trade_ledger.to_csv(outputs["portfolio_trade_ledger.csv"], index=False)
    equity_curves.to_csv(outputs["portfolio_equity_curves.csv"], index=False)
    metrics.to_csv(outputs["portfolio_metrics.csv"], index=False)
    exit_sensitivity.to_csv(outputs["exit_policy_sensitivity.csv"], index=False)
    decisions.to_csv(outputs["machine_rule_decisions.csv"], index=False)
    pareto_criteria.to_csv(outputs["pareto_criteria.csv"], index=False)
    pareto_decisions.to_csv(outputs["pareto_rule_set_decisions.csv"], index=False)
    outputs["b0_vs_balanced_rule_diff.md"].write_text(b0_diff, encoding="utf-8")
    outputs["rd_agent_candidate_rule_report.md"].write_text(report, encoding="utf-8")
    outputs["experiment_manifest.yaml"].write_text(manifest, encoding="utf-8")
    outputs["acceptance_summary.md"].write_text(acceptance, encoding="utf-8")
    if skill_change:
        proposed = output_dir / "SKILL.proposed.md"
        proposed.write_text(render_skill_proposal(balanced, decisions, pareto_decisions), encoding="utf-8")
        outputs["SKILL.proposed.md"] = proposed
    else:
        proposed = output_dir / "SKILL.proposed.md"
        if proposed.exists():
            proposed.unlink()
    return {name: str(path) for name, path in outputs.items()}


def load_pools(pool_root: Path) -> list[tuple[str, pd.DataFrame, Path]]:
    pools = []
    for path in sorted(pool_root.glob("*/breakout_follow_pool.csv")):
        snapshot = path.parent.name
        frame = pd.read_csv(path)
        if "snapshot_date" not in frame.columns:
            frame["snapshot_date"] = snapshot
        frame["snapshot_date"] = frame["snapshot_date"].fillna(snapshot).astype(str)
        pools.append((snapshot, frame, path))
    return pools


def load_price_cache(path: Path) -> dict[str, pd.DataFrame]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    prices = {}
    for code, value in raw.items():
        prices[str(code)] = normalize_bars(value)
    return prices


def build_candidate_event_panel(
    pools: list[tuple[str, pd.DataFrame, Path]],
    eps: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for snapshot, pool, path in pools:
        frame = pool.copy()
        frame["snapshot_date"] = snapshot
        frame["pool_path"] = str(path)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    signal = raw[raw["signal"].map(to_bool).eq(True) & raw["ibd_candidate_rule"].fillna("").astype(str).str.strip().ne("")].copy()
    signal["code"] = signal["code"].astype(str).str.strip()
    signal["_source_row_order"] = np.arange(len(signal))
    duplicates = signal[signal.duplicated(["snapshot_date", "code"], keep=False)].copy()
    signal = signal.sort_values(["snapshot_date", "code", "_source_row_order"]).drop_duplicates(["snapshot_date", "code"], keep="first")

    audited_eps_columns = [
        "pit_eps_yoy_growth",
        "pit_eps_state",
        "source",
        "effective_date",
        "current_period",
    ]
    signal = signal.drop(columns=[column for column in audited_eps_columns if column in signal.columns])
    eps_key = eps[["snapshot_date", "code", *audited_eps_columns]].copy()
    eps_key["code"] = eps_key["code"].astype(str).str.strip()
    panel = signal.merge(eps_key, on=["snapshot_date", "code"], how="left")
    panel["pit_eps_state"] = panel["pit_eps_state"].fillna("UNKNOWN")
    panel.loc[panel["pit_eps_state"].ne("VERIFIED"), "pit_eps_yoy_growth"] = pd.NA
    panel = enrich_features(panel)
    panel = build_event_labels(panel, prices, TradeLabelConfig())
    panel["relative_8w_return_pct"] = panel["forward_8w_return_pct"] - panel.groupby(["snapshot_date", "ibd_candidate_rule"])[
        "forward_8w_return_pct"
    ].transform("median")
    return panel, duplicates


def pool_coverage_drift(pools: list[tuple[str, pd.DataFrame, Path]], panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    regimes = _coverage_regime_labels(pools)
    previous_rows: int | None = None
    previous_schema = ""
    for snapshot, pool, path in pools:
        mask = pool["signal"].map(to_bool).eq(True) if "signal" in pool else pd.Series(False, index=pool.index)
        signal = pool[mask]
        panel_week = panel[panel["snapshot_date"].astype(str).eq(snapshot)]
        schema_hash = object_hash(list(pool.columns))
        rows.append(
            {
                "snapshot_date": snapshot,
                "pool_rows": len(pool),
                "signal_rows": len(signal),
                "ACTIONABLE": int((signal.get("ibd_entry_status", pd.Series(dtype=str)).astype(str).str.upper() == "ACTIONABLE").sum()),
                "UNCONFIRMED": int((signal.get("ibd_entry_status", pd.Series(dtype=str)).astype(str).str.upper() == "UNCONFIRMED").sum()),
                "EXTENDED": int((signal.get("ibd_entry_status", pd.Series(dtype=str)).astype(str).str.upper() == "EXTENDED").sum()),
                "unique_signal_tickers": signal.get("code", pd.Series(dtype=str)).astype(str).nunique(),
                "routes": ";".join(
                    f"{k}:{v}" for k, v in signal.get("ibd_candidate_rule", pd.Series(dtype=str)).fillna("UNKNOWN").value_counts().sort_index().items()
                ),
                "geometry_complete_rate": _complete_rate(signal, ["ibd_entry_close_position", "ibd_entry_breakout_range_ratio"]),
                "volume_known_rate": signal.get("ibd_entry_volume_ratio", pd.Series(dtype=float)).map(to_float).notna().mean() if len(signal) else 0,
                "price_entry_available_rate": panel_week["entry_unavailable"].eq(False).mean() if len(panel_week) else 0,
                "forward_8w_complete_rate": panel_week["forward_8w_censored"].eq(False).mean() if len(panel_week) else 0,
                "pool_rows_change_pct": (
                    (len(pool) / previous_rows - 1.0) * 100.0
                    if previous_rows not in {None, 0}
                    else np.nan
                ),
                "schema_hash": schema_hash,
                "schema_changed": bool(previous_schema and schema_hash != previous_schema),
                "coverage_regime": regimes[snapshot],
                "coverage_regime_method": "three-segment log-row-count minimum-SSE change points",
                "pool_path": str(path),
            }
        )
        if len(pool):
            previous_rows = len(pool)
        previous_schema = schema_hash
    return pd.DataFrame(rows)


def _coverage_regime_labels(pools: list[tuple[str, pd.DataFrame, Path]]) -> dict[str, str]:
    labels = {snapshot: "empty_pool" for snapshot, pool, _ in pools if pool.empty}
    observed = [(snapshot, len(pool)) for snapshot, pool, _ in pools if not pool.empty]
    count = len(observed)
    if count < 9:
        labels.update({snapshot: "single_observed_regime" for snapshot, _ in observed})
        return labels
    minimum_segment = min(6, max(3, count // 4))
    values = np.log1p(np.array([rows for _, rows in observed], dtype=float))
    best: tuple[float, int, int] | None = None
    for first in range(minimum_segment, count - 2 * minimum_segment + 1):
        for second in range(first + minimum_segment, count - minimum_segment + 1):
            segments = (values[:first], values[first:second], values[second:])
            loss = float(sum(((segment - segment.mean()) ** 2).sum() for segment in segments))
            if best is None or loss < best[0]:
                best = (loss, first, second)
    if best is None:
        labels.update({snapshot: "single_observed_regime" for snapshot, _ in observed})
        return labels
    _, first, second = best
    regime_names = ("early_stable_low", "coverage_transition", "late_stable_high")
    for index, (snapshot, _) in enumerate(observed):
        regime = regime_names[0] if index < first else regime_names[1] if index < second else regime_names[2]
        labels[snapshot] = regime
    return labels


def candidate_label_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for snapshot, group in panel.groupby("snapshot_date"):
        rows.append(
            {
                "snapshot_date": snapshot,
                "signal_events": len(group),
                "entry_available": int(group["entry_unavailable"].eq(False).sum()),
                "forward_1w_complete": int(group["forward_1w_censored"].eq(False).sum()),
                "forward_3w_complete": int(group["forward_3w_censored"].eq(False).sum()),
                "forward_5w_complete": int(group["forward_5w_censored"].eq(False).sum()),
                "forward_8w_complete": int(group["forward_8w_censored"].eq(False).sum()),
                "eps_verified": int(group["pit_eps_state"].eq("VERIFIED").sum()),
                "eps_unverified_or_unknown": int(group["pit_eps_state"].ne("VERIFIED").sum()),
            }
        )
    return pd.DataFrame(rows)


def candidate_rule_evidence(panel: pd.DataFrame, *, bootstrap_iterations: int) -> pd.DataFrame:
    specs = [
        ("Status", "ibd_entry_status", lambda df: df["ibd_entry_status"].fillna("UNKNOWN").astype(str).str.upper()),
        ("Close > Trigger", "close_trigger_bin", lambda df: _close_groups(df["ibd_entry_close_vs_trigger_pct"])),
        ("Fresh Zone", "fresh_bin", lambda df: _fresh_groups(df["current_vs_ibd_candidate_pct"])),
        ("Entry Volume", "entry_volume_bin", lambda df: _volume_groups(df["ibd_entry_volume_ratio"])),
        ("Geometry", "geometry", lambda df: df["geometry"].fillna("UNKNOWN").astype(str)),
        ("Pullback", "pullback_v_is_dry", lambda df: df["pullback_dry_state"].fillna("UNKNOWN").astype(str)),
        ("EPS", "pit_eps_bin", lambda df: np.where(df["pit_eps_state"].eq("VERIFIED"), np.where(df["pit_eps_yoy_growth"].map(to_float) >= 25, "VERIFIED_>=25", "VERIFIED_<25"), df["pit_eps_state"])),
        ("Industry", "industry_present", lambda df: np.where(df["industry"].fillna("").astype(str).str.strip().ne(""), "KNOWN", "UNKNOWN")),
    ]
    rows: list[dict[str, Any]] = []
    for family, field, grouper in specs:
        tmp = panel.copy()
        tmp["_group"] = grouper(tmp)
        for value, group in tmp.groupby("_group", dropna=False):
            complete = group[group["forward_8w_censored"].eq(False)]
            samples = week_block_bootstrap(
                complete,
                value_col="forward_8w_return_pct",
                seed=SEED,
                iterations=bootstrap_iterations,
                time_block_weeks=8,
            )
            lo, hi = ci_from_samples(samples)
            rows.append(
                {
                    "rule_family": family,
                    "field": field,
                    "value": str(value),
                    "events": len(group),
                    "complete_8w": len(complete),
                    "weeks": group["snapshot_date"].nunique(),
                    "mean_8w_return_pct": complete["forward_8w_return_pct"].mean(),
                    "median_relative_8w_return_pct": complete["relative_8w_return_pct"].median(),
                    "mature_weeks": complete["snapshot_date"].nunique(),
                    "stop_8_40d_rate": complete["stop_8_within_40d"].mean() if len(complete) else np.nan,
                    "profit_24_40d_rate": complete["profit_24_within_40d"].mean() if len(complete) else np.nan,
                    "pattern_power_rate": _known_bool_rate(complete["pattern_power_trigger"]),
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                }
            )
    return pd.DataFrame(rows)


def rule_treatment_contrast(
    panel: pd.DataFrame,
    *,
    bootstrap_iterations: int,
    min_group_size: int = 20,
    min_weeks: int = 5,
) -> pd.DataFrame:
    route_keys = ["snapshot_date", "ibd_candidate_rule"]
    controlled = panel.copy()
    controlled["_control_fresh"] = _fresh_groups(controlled["current_vs_ibd_candidate_pct"])
    controlled["_control_volume"] = _volume_groups(controlled["ibd_entry_volume_ratio"])
    controlled["_control_geometry"] = controlled.get("geometry", pd.Series("UNKNOWN", index=controlled.index)).fillna("UNKNOWN").astype(str)
    treatments = [
        (
            "Status",
            "Status_ACTIONABLE_vs_other",
            panel,
            panel["ibd_entry_status"].astype(str).str.upper().eq("ACTIONABLE"),
            "",
            route_keys,
        ),
        (
            "Status",
            "Status_ACTIONABLE_increment_controlled",
            controlled,
            controlled["ibd_entry_status"].astype(str).str.upper().eq("ACTIONABLE"),
            "no same-week same-route Fresh/Volume/Geometry controlled overlap",
            [*route_keys, "_control_fresh", "_control_volume", "_control_geometry"],
        ),
        (
            "Close > Trigger",
            "Close_nonnegative_vs_negative",
            panel[panel["ibd_entry_close_vs_trigger_pct"].map(to_float).notna()],
            panel["ibd_entry_close_vs_trigger_pct"].map(to_float).ge(0),
            "negative group not observed",
            route_keys,
        ),
        (
            "Fresh Zone",
            "Fresh_0_5_vs_other",
            panel[panel["current_vs_ibd_candidate_pct"].map(to_float).notna()],
            panel["current_vs_ibd_candidate_pct"].map(to_float).between(0, 5, inclusive="both"),
            "",
            route_keys,
        ),
        (
            "Entry Volume",
            "Volume_1_5_vs_other",
            panel[panel["ibd_entry_volume_ratio"].map(to_float).notna()],
            panel["ibd_entry_volume_ratio"].map(to_float).ge(1.5),
            "below 1.5 group not observed",
            route_keys,
        ),
        (
            "Geometry",
            "Geometry_nonfailure_vs_failure",
            panel[panel["geometry"].ne("UNKNOWN")],
            ~panel["geometry"].isin(["Defensive Failure", "Squat / Upper Shadow"]),
            "",
            route_keys,
        ),
        (
            "Pullback",
            "Pullback_dry_PASS_vs_FAIL",
            panel[panel["ibd_candidate_rule"].isin(PULLBACK_RULES) & panel["pullback_dry_state"].isin(["PASS", "FAIL"])],
            panel["pullback_dry_state"].eq("PASS"),
            "",
            route_keys,
        ),
        (
            "Pullback",
            "Pullback_dry_PASS_vs_UNKNOWN",
            panel[panel["ibd_candidate_rule"].isin(PULLBACK_RULES) & panel["pullback_dry_state"].isin(["PASS", "UNKNOWN"])],
            panel["pullback_dry_state"].eq("PASS"),
            "",
            route_keys,
        ),
        (
            "Pullback",
            "Pullback_dry_FAIL_vs_UNKNOWN",
            panel[panel["ibd_candidate_rule"].isin(PULLBACK_RULES) & panel["pullback_dry_state"].isin(["FAIL", "UNKNOWN"])],
            panel["pullback_dry_state"].eq("FAIL"),
            "",
            route_keys,
        ),
        (
            "EPS",
            "EPS_verified_25_vs_verified_below",
            panel[panel["pit_eps_state"].eq("VERIFIED")],
            panel["pit_eps_yoy_growth"].map(to_float).ge(25),
            "",
            route_keys,
        ),
    ]
    for route in sorted(PULLBACK_RULES):
        route_frame = panel[
            panel["ibd_candidate_rule"].eq(route)
            & panel["pullback_dry_state"].isin(["PASS", "FAIL"])
        ]
        treatments.append(
            (
                "Pullback",
                f"Pullback_{route}_PASS_vs_FAIL",
                route_frame,
                panel["pullback_dry_state"].eq("PASS"),
                f"{route} PASS/FAIL comparison side not observed",
                route_keys,
            )
        )
    rows = []
    for family, name, applicable, mask, not_identifiable_reason, pair_keys in treatments:
        rows.append(
            _evaluate_treatment_contrast(
                family,
                name,
                applicable,
                mask,
                pair_keys=pair_keys,
                not_identifiable_reason=not_identifiable_reason,
                bootstrap_iterations=bootstrap_iterations,
                min_group_size=min_group_size,
                min_weeks=min_weeks,
            )
        )
    return pd.DataFrame(rows)


def _evaluate_treatment_contrast(
    family: str,
    name: str,
    applicable: pd.DataFrame,
    mask: pd.Series,
    *,
    pair_keys: list[str],
    not_identifiable_reason: str,
    bootstrap_iterations: int,
    min_group_size: int,
    min_weeks: int,
) -> dict[str, Any]:
    complete = applicable[applicable["forward_8w_censored"].eq(False)].copy()
    aligned_mask = mask.reindex(complete.index).fillna(False)
    treated = complete[aligned_mask]
    control = complete[~aligned_mask]
    paired = _paired_group_diffs(treated, control, pair_keys)
    if treated.empty or control.empty:
        status = "RULE_NOT_IDENTIFIABLE"
        blocker = not_identifiable_reason or "one registered comparison side is not observed"
    elif paired.empty and not_identifiable_reason:
        status = "RULE_NOT_IDENTIFIABLE"
        blocker = not_identifiable_reason
    elif len(treated) < min_group_size or len(control) < min_group_size or treated["snapshot_date"].nunique() < min_weeks or control["snapshot_date"].nunique() < min_weeks:
        status = "INSUFFICIENT_EVIDENCE"
        blocker = "insufficient treated/control count or independent weeks"
    elif paired.empty:
        status = "INSUFFICIENT_EVIDENCE"
        blocker = "no same-week same-route paired contrast"
    else:
        status = "OK"
        blocker = ""
    lo, hi = ci_from_samples(
        paired_week_route_bootstrap(
            paired,
            treated_col="treated_mean",
            control_col="control_mean",
            seed=SEED,
            iterations=bootstrap_iterations,
        )
    )
    return {
        "rule_family": family,
        "contrast": name,
        "applicable_events": len(applicable),
        "treated_complete": len(treated),
        "control_complete": len(control),
        "treated_weeks": treated["snapshot_date"].nunique(),
        "control_weeks": control["snapshot_date"].nunique(),
        "paired_week_routes": len(paired),
        "mean_return_diff_pct": paired["diff"].mean() if status == "OK" else np.nan,
        "stop_rate_diff": paired["stop_diff"].mean() if status == "OK" else np.nan,
        "profit_24_rate_diff": paired["profit_diff"].mean() if status == "OK" else np.nan,
        "power_rate_diff": paired["power_diff"].mean() if status == "OK" else np.nan,
        "ci_low": lo if status == "OK" else np.nan,
        "ci_high": hi if status == "OK" else np.nan,
        "status": status,
        "blocker": blocker,
        "pairing_keys": ";".join(pair_keys),
    }


def _paired_group_diffs(treated: pd.DataFrame, control: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    metrics = ["forward_8w_return_pct", "stop_8_within_40d", "profit_24_within_40d", "pattern_power_trigger"]
    treated_work = _contrast_metric_frame(treated, metrics)
    control_work = _contrast_metric_frame(control, metrics)
    treated_means = treated_work.groupby(keys)[metrics].mean().add_prefix("treated_")
    control_means = control_work.groupby(keys)[metrics].mean().add_prefix("control_")
    paired = pd.concat([treated_means, control_means], axis=1).dropna(
        subset=["treated_forward_8w_return_pct", "control_forward_8w_return_pct"]
    ).reset_index()
    if paired.empty:
        paired["diff"] = pd.Series(dtype=float)
        paired["stop_diff"] = pd.Series(dtype=float)
        paired["profit_diff"] = pd.Series(dtype=float)
        paired["power_diff"] = pd.Series(dtype=float)
        return paired
    paired = paired.rename(
        columns={
            "treated_forward_8w_return_pct": "treated_mean",
            "control_forward_8w_return_pct": "control_mean",
        }
    )
    paired["diff"] = paired["treated_mean"] - paired["control_mean"]
    paired["stop_diff"] = paired["treated_stop_8_within_40d"] - paired["control_stop_8_within_40d"]
    paired["profit_diff"] = paired["treated_profit_24_within_40d"] - paired["control_profit_24_within_40d"]
    paired["power_diff"] = paired["treated_pattern_power_trigger"] - paired["control_pattern_power_trigger"]
    return paired


def _contrast_metric_frame(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    work = frame.copy()
    work["forward_8w_return_pct"] = pd.to_numeric(work["forward_8w_return_pct"], errors="coerce")
    for column in metrics[1:]:
        work[column] = work[column].map(
            lambda raw: 1.0 if to_bool(raw) is True else 0.0 if to_bool(raw) is False else np.nan
        )
    return work


def _known_bool_rate(values: pd.Series) -> float:
    observed = values.map(to_bool).dropna()
    return float(observed.eq(True).mean()) if len(observed) else np.nan


def _close_groups(values: pd.Series) -> pd.Series:
    def classify(raw: object) -> str:
        value = to_float(raw)
        if value is None:
            return "UNKNOWN"
        if value < 0:
            return "<0"
        if value <= 1:
            return "0-1%"
        if value <= 3:
            return "1-3%"
        return ">3%"

    return values.map(classify)


def _fresh_groups(values: pd.Series) -> pd.Series:
    def classify(raw: object) -> str:
        value = to_float(raw)
        if value is None:
            return "UNKNOWN"
        if value < 0:
            return "<0"
        if value <= 2:
            return "0-2%"
        if value <= 5:
            return "2-5%"
        if value <= 10:
            return "5-10%"
        return ">10%"

    return values.map(classify)


def _volume_groups(values: pd.Series) -> pd.Series:
    def classify(raw: object) -> str:
        value = to_float(raw)
        if value is None:
            return "UNKNOWN"
        if value < 1.0:
            return "<1.0"
        if value < 1.3:
            return "1.0-1.3"
        if value < 1.5:
            return "1.3-1.5"
        if value <= 2.0:
            return "1.5-2.0"
        if value <= 3.0:
            return "2.0-3.0"
        return ">3.0"

    return values.map(classify)


def build_all_selections(panel: pd.DataFrame) -> pd.DataFrame:
    chunks = []
    for name, config in selector_configs().items():
        selected = select_all_weeks(panel, config)
        if not selected.empty:
            selected["variant"] = name
            chunks.append(selected)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def b0_atomic_invariant_audit(panel: pd.DataFrame) -> pd.DataFrame:
    base = selector_configs()["B0_PIT_VERIFIED"]
    rows = []
    audited_weeks = int(pd.to_datetime(panel["snapshot_date"]).nunique())
    for name, config in atomic_selector_configs().items():
        row = audit_atomic_variant(panel, base, config)
        row["audited_weeks"] = audited_weeks
        row["atomicity_status"] = "PASS" if row["non_target_trace_violations"] == 0 else "FAIL"
        rows.append(row)
    return pd.DataFrame(rows)


def b0_atomic_ablation(
    selections: pd.DataFrame,
    panel: pd.DataFrame,
    invariant: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    b0 = selections[selections["variant"].eq("B0_PIT_VERIFIED")]
    variants = ["B0_PIT_VERIFIED", *atomic_selector_configs().keys()]
    for variant in variants:
        sel = selections[selections["variant"].eq(variant)]
        added = removed = rank_changed = affected = complete = 0
        jaccards = []
        for snapshot in sorted(set(b0["snapshot_date"]).union(set(sel["snapshot_date"]))):
            b = b0[b0["snapshot_date"].eq(snapshot)]
            v = sel[sel["snapshot_date"].eq(snapshot)]
            bs = set(b["code"])
            vs = set(v["code"])
            added += len(vs - bs)
            removed += len(bs - vs)
            affected += int(bs != vs)
            rank_changed += sum(
                1 for code in bs & vs if int(b[b["code"].eq(code)]["pick_order"].iloc[0]) != int(v[v["code"].eq(code)]["pick_order"].iloc[0])
            )
            jaccards.append(len(bs & vs) / len(bs | vs) if bs | vs else 1.0)
        complete = len(sel[sel["forward_8w_censored"].eq(False)]) if "forward_8w_censored" in sel.columns else len(
            sel.merge(panel[["snapshot_date", "code", "forward_8w_censored"]], on=["snapshot_date", "code"], how="left").query("forward_8w_censored == False")
        )
        rows.append(
            {
                "variant": variant,
                "selected_count": len(sel),
                "added": added,
                "removed": removed,
                "rank_changed": rank_changed,
                "affected_weeks": affected,
                "Jaccard": float(np.mean(jaccards)) if jaccards else np.nan,
                "complete_outcome_count": complete,
                "treatment_contrast": "NO_TREATMENT_CONTRAST" if added + removed + rank_changed == 0 else "OK",
            }
        )
    result = pd.DataFrame(rows)
    if invariant is not None and not invariant.empty:
        result = result.merge(invariant, on=["variant", "treatment_contrast"], how="left")
    return result


def oos_results(
    panel: pd.DataFrame,
    *,
    bootstrap_iterations: int = 400,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits = make_rolling_splits(panel, test_weeks=4, embargo_weeks=8, min_train_weeks=8)
    rows: list[dict[str, Any]] = []
    pick_chunks: list[pd.DataFrame] = []
    registry_rows: list[dict[str, Any]] = []
    for split_id, split in enumerate(splits, 1):
        proposal = propose_fold_candidates(panel, split, bootstrap_iterations=bootstrap_iterations)
        registry_rows.append(
            {
                "fold": split_id,
                "train_start": proposal["train_start"],
                "train_end": proposal["train_end"],
                "test_start": str(split.test_start.date()),
                "test_end": str(split.test_end.date()),
                "train_evidence_max_date": proposal["train_evidence_max_date"],
                "train_evidence_json": proposal["train_evidence_json"],
                "frozen_rules_json": proposal["frozen_rules_json"],
                "frozen_rule_hash": proposal["frozen_rule_hash"],
                "candidate_generation_status": proposal["candidate_generation_status"],
                "evaluation_type": "blocked_retrospective_evaluation",
            }
        )
        dates = pd.to_datetime(panel["snapshot_date"])
        test = panel[dates.between(split.test_start, split.test_end)].copy()
        baseline_picks = select_all_weeks(test, selector_configs()["B0_PIT_VERIFIED"])
        if not baseline_picks.empty:
            baseline_picks["variant"] = "B0_PIT_VERIFIED_BLOCKED"
            baseline_picks["fold"] = split_id
            pick_chunks.append(baseline_picks)
        for config in atomic_selector_configs().values():
            rows.append(
                evaluate_frozen_config(
                    test,
                    config,
                    fold=split_id,
                    split=split,
                    proposal=proposal,
                    bootstrap_iterations=bootstrap_iterations,
                    seed=SEED + split_id,
                )
            )
        for config in proposal["candidate_configs"]:
            rows.append(
                evaluate_frozen_config(
                    test,
                    config,
                    fold=split_id,
                    split=split,
                    proposal=proposal,
                    bootstrap_iterations=bootstrap_iterations,
                    seed=SEED + 1000 + split_id,
                )
            )
            picks = select_all_weeks(test, config)
            if not baseline_picks.empty:
                matching_baseline = baseline_picks.copy()
                matching_baseline["variant"] = f"B0_FOR_{config.name}_BLOCKED"
                matching_baseline["fold"] = split_id
                pick_chunks.append(matching_baseline)
            if not picks.empty:
                picks["variant"] = config.name
                picks["fold"] = split_id
                pick_chunks.append(picks)
    fold = pd.DataFrame(rows)
    rule_sets = fold[fold["variant"].astype(str).str.startswith("R")] if not fold.empty else pd.DataFrame()
    summary = summarize_blocked_results(rule_sets, seed=SEED, bootstrap_iterations=bootstrap_iterations)
    if not summary.empty:
        summary["available_folds"] = len(splits)
        summary["fold_coverage_ratio"] = summary["folds"] / max(len(splits), 1)
    if summary.empty:
        summary = pd.DataFrame(
            [
                {
                    "variant": "NO_STABLE_CANDIDATE",
                    "folds": len(splits),
                    "paired_weeks": 0,
                    "mean_oos_diff_pct": np.nan,
                    "median_oos_diff_pct": np.nan,
                    "return_ci_low_pct": np.nan,
                    "return_ci_high_pct": np.nan,
                    "worst_fold_diff_pct": np.nan,
                    "better_folds": 0,
                    "non_worse_folds": 0,
                    "stop_40d_diff_pp": np.nan,
                    "profit_24_40d_diff_pp": np.nan,
                    "CVaR_20_diff_pct": np.nan,
                    "coverage_ratio": np.nan,
                    "industry_top_share_diff": np.nan,
                    "frozen_rule_hashes": 0,
                    "ci_time_block_weeks": 8,
                    "ci_block_folds": 2,
                    "available_folds": len(splits),
                    "fold_coverage_ratio": 0.0,
                    "required_rule_families": "",
                    "source_atomic_variants": "",
                    "portfolio_baseline_variant": "B0_PIT_VERIFIED_BLOCKED",
                    "evaluation_type": "blocked_retrospective_evaluation",
                }
            ]
        )
    blocked_picks = pd.concat(pick_chunks, ignore_index=True) if pick_chunks else pd.DataFrame()
    return fold, summary, blocked_picks, pd.DataFrame(registry_rows)


def portfolio_outputs(selections: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trade_rows = []
    equity_rows = []
    metric_rows = []
    variants = sorted(selections["variant"].dropna().astype(str).unique()) if not selections.empty else []
    valuation_as_of = _price_cache_as_of(prices)
    valuation_start = _selection_valuation_start(selections, prices, valuation_as_of)
    for variant in variants:
        picks = selections[selections["variant"].eq(variant)].copy()
        for capacity in (3, 6, 10):
            for cost in (0, 10, 25):
                cfg = PortfolioConfig(
                    capacity=capacity,
                    initial_capital=100_000.0,
                    cost_bps_per_side=cost,
                    valuation_start=valuation_start,
                    valuation_as_of=valuation_as_of,
                )
                trades, equity, events = run_portfolio_backtest(picks, prices, cfg)
                for frame in (trades, equity):
                    if not frame.empty:
                        frame["variant"] = variant
                        frame["capacity"] = capacity
                        frame["cost_bps_per_side"] = cost
                if not trades.empty:
                    trade_rows.append(trades)
                if not equity.empty:
                    equity_rows.append(equity)
                metrics = portfolio_metrics(equity, trades, initial_capital=cfg.initial_capital)
                entered_picks = (
                    trades[["snapshot_date", "code"]].merge(
                        picks[["snapshot_date", "code", "pattern_power_trigger"]],
                        on=["snapshot_date", "code"],
                        how="left",
                    )
                    if not trades.empty and "pattern_power_trigger" in picks.columns
                    else pd.DataFrame()
                )
                metrics.update(
                    {
                        "variant": variant,
                        "capacity": capacity,
                        "cost_bps_per_side": cost,
                        "trades": len(trades),
                        "skipped_capacity": int(events["event"].eq("capacity_skip").sum()) if not events.empty else 0,
                        "skipped_cash": int(events["event"].eq("cash_skip").sum()) if not events.empty else 0,
                        "repeat_signals_ignored": int(events["event"].eq("repeat_signal_ignored").sum()) if not events.empty else 0,
                        "stop_or_gap_stop": int(trades["exit_reason"].isin(["stop_loss", "gap_stop"]).sum()) if not trades.empty else 0,
                        "stop_exits": int(trades["exit_reason"].eq("stop_loss").sum()) if not trades.empty else 0,
                        "gap_stop_exits": int(trades["exit_reason"].eq("gap_stop").sum()) if not trades.empty else 0,
                        "profit_exits": int(trades["exit_reason"].astype(str).str.contains("profit").sum()) if not trades.empty else 0,
                        "pattern_power_triggers": int(entered_picks["pattern_power_trigger"].eq(True).sum()) if not entered_picks.empty else 0,
                        "trade_power_triggers": int(trades["power_trigger_date"].astype(str).str.len().gt(0).sum()) if not trades.empty else 0,
                        "censored_positions": int(trades["censored"].eq(True).sum()) if not trades.empty else 0,
                        "valuation_start": str(pd.Timestamp(valuation_start).date()) if valuation_start is not None else "",
                        "valuation_as_of": str(pd.Timestamp(valuation_as_of).date()) if valuation_as_of is not None else "",
                    }
                )
                metric_rows.append(metrics)
    return (
        pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame(),
        pd.concat(equity_rows, ignore_index=True) if equity_rows else pd.DataFrame(),
        pd.DataFrame(metric_rows),
    )


def exit_policy_sensitivity(selections: pd.DataFrame, prices: dict[str, pd.DataFrame], panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    picks = selections[selections["variant"].eq("B0_PIT_VERIFIED")]
    valuation_as_of = _price_cache_as_of(prices)
    valuation_start = _selection_valuation_start(picks, prices, valuation_as_of)
    policies = [
        ("ordinary_stop8_profit24_no_power_lock", ExitPolicy(stop_pct=8, profit_pct=24, enable_power_lock=False)),
        (
            "optimistic_same_day_profit_first_stop8_profit24",
            ExitPolicy(stop_pct=8, profit_pct=24, same_day_order="profit_first"),
        ),
        ("stop7_profit24", ExitPolicy(stop_pct=7, profit_pct=24)),
        ("stop7_5_profit24", ExitPolicy(stop_pct=7.5, profit_pct=24)),
        ("main_ibd_power_lock_stop8_profit24", ExitPolicy(stop_pct=8, profit_pct=24)),
        ("stop8_profit20", ExitPolicy(stop_pct=8, profit_pct=20)),
        ("stop8_profit22_5", ExitPolicy(stop_pct=8, profit_pct=22.5)),
        ("stop8_profit25", ExitPolicy(stop_pct=8, profit_pct=25)),
        ("post_lock_resume_profit", ExitPolicy(post_lock="resume_profit")),
        ("post_lock_week8_close", ExitPolicy(post_lock="week8_close")),
        ("post_lock_trend_exit", ExitPolicy(post_lock="trend_exit")),
        ("post_lock_mtm", ExitPolicy(post_lock="mark_to_market")),
    ]
    for name, policy in policies:
        trades, equity, _ = run_portfolio_backtest(
            picks,
            prices,
            PortfolioConfig(
                capacity=3,
                exit_policy=policy,
                valuation_start=valuation_start,
                valuation_as_of=valuation_as_of,
            ),
        )
        metrics = portfolio_metrics(equity, trades, initial_capital=100_000.0)
        metrics["exit_policy"] = name
        metrics["trades"] = len(trades)
        metrics["valuation_start"] = str(valuation_start.date()) if valuation_start is not None else ""
        metrics["valuation_as_of"] = str(valuation_as_of.date()) if valuation_as_of is not None else ""
        rows.append(metrics)
    return pd.DataFrame(rows)


def machine_rule_decisions(
    evidence: pd.DataFrame,
    contrasts: pd.DataFrame,
    fold_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return _machine_rule_decisions(evidence, contrasts, fold_results)


def hypothesis_registry() -> list[dict[str, Any]]:
    return [
        {
            "hypothesis_id": f"H{idx:03d}",
            "variant": name,
            "selector_config": asdict(cfg),
            "pre_registered_role": cfg.changed_rules or ("baseline",),
            "success_rule": "Must beat B0 on OOS direction and portfolio risk without worse stop/tail/concentration.",
            "period_name": "retrospective_final_test",
        }
        for idx, (name, cfg) in enumerate(selector_configs().items(), 1)
    ]


def experiment_manifest(
    pool_root: Path,
    price_cache: Path,
    output_dir: Path,
    pools: list[tuple[str, pd.DataFrame, Path]],
    hypotheses: list[dict[str, Any]],
    *,
    bootstrap_iterations: int,
    valuation_start: pd.Timestamp | None,
    valuation_as_of: pd.Timestamp | None,
    freeze_registry: pd.DataFrame,
) -> str:
    source_files = sorted(Path("backtest/rd_agent_candidate_rule_audit").glob("*.py"))
    manifest = {
        "base_commit": "3e73d887872a0f668b66cefe7811d09c7ae90b2b",
        "repair_baseline_commit": "199a0d892dc1aa7973bcad93a4180e9b3875e512",
        "head_commit_at_run": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "period_name": "retrospective_final_test",
        "data_status": "retrospective_inspected_data",
        "prospective_holdout": {
            "enabled": True,
            "unlock_rule": "future replay pool snapshots unlock only after complete 40 trading-day observation window",
            "embargo_weeks": 8,
        },
        "source_code_hash": object_hash({str(path): content_hash(path) for path in source_files}),
        "pool_file_content_hash": {str(path): content_hash(path) for _, _, path in pools},
        "eps_file_content_hash": content_hash(pool_root / "signal_eps_pit.csv"),
        "price_cache_content_hash": content_hash(price_cache),
        "selector_configs": {name: asdict(cfg) for name, cfg in selector_configs().items()},
        "B0_REPO_EXACT_invariant": "weekly code, pick order, reason_codes, and risk_codes must exactly match dashboard.skill_industry_eps_known",
        "B0_PIT_VERIFIED_eps_policy": "only pit_eps_state=VERIFIED and pit_eps_yoy_growth; production fallback disabled",
        "B0_ablation_configs": [name for name in selector_configs() if name.startswith("B0 ")],
        "label_windows_trading_days": {"1w": 5, "3w": 15, "5w": 25, "8w": 40},
        "stop_profit_parameters": {
            "main_stop_pct": 8,
            "main_profit_pct": 24,
            "main_power_lock_enabled": True,
            "ordinary_comparator_power_lock_enabled": False,
            "same_day_main_order": "stop_first",
            "same_day_sensitivity_order": "profit_first",
            "sensitivity_stop_pct": [7, 7.5, 8],
            "sensitivity_profit_pct": [20, 22.5, 24, 25],
        },
        "power_trigger_definition": "daily High >= ibd_candidate_price * 1.20 through third breakout-week Friday from valid ibd_entry_date",
        "walk_forward": {"test_weeks": 4, "embargo_weeks": 8, "min_train_weeks": 8},
        "blocked_retrospective_evaluation": {
            "train_select_freeze_test": True,
            "overlapping_outcome_window_weeks": 8,
            "oos_ci_moving_block_folds": 2,
            "candidate_generation_thresholds": ATOMIC_TRAIN_THRESHOLDS,
            "fold_rule_hashes": freeze_registry.get("frozen_rule_hash", pd.Series(dtype=str)).astype(str).tolist(),
            "fold_registry_hash": object_hash(freeze_registry.to_dict("records")),
        },
        "pareto_thresholds": PARETO_THRESHOLDS,
        "coverage_regime": {
            "method": "three-segment log-row-count minimum-SSE change points",
            "causal_interpretation": False,
        },
        "bootstrap": {"seed": SEED, "iterations": bootstrap_iterations, "block": "snapshot week"},
        "portfolio": {
            "capacities": [3, 6, 10],
            "cost_bps_per_side": [0, 10, 25],
            "initial_capital": 100000,
            "no_leverage": True,
            "uniform_valuation_start": str(valuation_start.date()) if valuation_start is not None else "",
            "uniform_valuation_as_of": str(valuation_as_of.date()) if valuation_as_of is not None else "",
        },
        "censoring_policy": "insufficient horizon stays censored; as_of_return_pct separate from complete forward returns",
        "hypothesis_hash": object_hash(hypotheses),
        "output_dir": str(output_dir),
    }
    return _simple_yaml(manifest)


def render_data_audit(
    pools: list[tuple[str, pd.DataFrame, Path]],
    panel: pd.DataFrame,
    duplicates: pd.DataFrame,
    drift: pd.DataFrame,
    coverage: pd.DataFrame,
) -> str:
    total_pool = sum(len(pool) for _, pool, _ in pools)
    nonempty = sum(1 for _, pool, _ in pools if len(pool))
    lines = [
        "# Candidate Event Rule Audit - Data Audit",
        "",
        f"- Pool directories with CSV: {len(pools)}",
        f"- Non-empty pool weeks: {nonempty}",
        f"- Pool raw rows: {total_pool}",
        f"- Signal ticker-week events after deterministic de-dup: {len(panel)}",
        f"- Unique signal tickers: {panel['code'].nunique()}",
        f"- Duplicate snapshot/code rows: {len(duplicates)}",
        f"- ACTIONABLE/UNCONFIRMED/EXTENDED: {_status_count(panel, 'ACTIONABLE')}/{_status_count(panel, 'UNCONFIRMED')}/{_status_count(panel, 'EXTENDED')}",
        f"- PIT EPS verified/blocked-or-unknown: {int(panel['pit_eps_state'].eq('VERIFIED').sum())}/{int(panel['pit_eps_state'].ne('VERIFIED').sum())}",
        f"- Complete 8w labels: {int(panel['forward_8w_censored'].eq(False).sum())}",
        "",
        "The local schema document is a migration pointer to the quant_trade SSOT. Field meanings not inferable from repository consumers remain schema blockers for production hard gates.",
        "",
        "Coverage regimes are descriptive minimum-SSE level-shift segments on log pool-row count. They are not evidence of a causal selector/config change; schema hashes and week-over-week coverage changes are reported separately.",
    ]
    return "\n".join(lines) + "\n"


def render_b0_diff(
    ablations: pd.DataFrame,
    oos: pd.DataFrame,
    metrics: pd.DataFrame,
    pareto_criteria: pd.DataFrame,
    pareto_decisions: pd.DataFrame,
    balanced: str,
) -> str:
    candidate_oos = oos[oos["variant"].eq(balanced)] if "variant" in oos.columns and not balanced.startswith("NO_STABLE") else pd.DataFrame()
    b0m = metrics[(metrics["variant"].eq("B0_PIT_VERIFIED")) & (metrics["capacity"].eq(3)) & (metrics["cost_bps_per_side"].eq(10))]
    cm = metrics[(metrics["variant"].eq(balanced)) & (metrics["capacity"].eq(3)) & (metrics["cost_bps_per_side"].eq(10))] if not balanced.startswith("NO_STABLE") else pd.DataFrame()
    return "\n".join(
        [
            "# B0 vs Balanced Rule Diff",
            "",
            f"Balanced candidate used for comparison: `{balanced}`.",
            "",
            "Candidate rules are generated independently inside each training fold and frozen before the blocked retrospective test window.",
            "",
            f"OOS row: {candidate_oos.to_dict('records')[:1]}",
            f"B0 portfolio metric: {b0m.to_dict('records')[:1]}",
            f"Candidate portfolio metric: {cm.to_dict('records')[:1]}",
            f"Pareto decision: {pareto_decisions[pareto_decisions['variant'].eq(balanced)].to_dict('records')[:1] if not pareto_decisions.empty else []}",
            f"Pareto criteria: {pareto_criteria[pareto_criteria['variant'].eq(balanced)].to_dict('records') if not pareto_criteria.empty else []}",
            "",
            "NO PRODUCTION SKILL CHANGE" if balanced.startswith("NO_STABLE") else "Any proposed change still requires prospective confirmation.",
        ]
    ) + "\n"


def render_report(
    panel: pd.DataFrame,
    drift: pd.DataFrame,
    evidence: pd.DataFrame,
    contrasts: pd.DataFrame,
    ablations: pd.DataFrame,
    oos: pd.DataFrame,
    metrics: pd.DataFrame,
    exit_sensitivity: pd.DataFrame,
    decisions: pd.DataFrame,
    pareto_criteria: pd.DataFrame,
    pareto_decisions: pd.DataFrame,
    freeze_registry: pd.DataFrame,
    balanced: str,
    skill_change: bool,
) -> str:
    b0_port = metrics[(metrics["variant"].eq("B0_PIT_VERIFIED")) & (metrics["capacity"].eq(3)) & (metrics["cost_bps_per_side"].eq(10))]
    best_port = metrics[(metrics["variant"].eq(balanced)) & (metrics["capacity"].eq(3)) & (metrics["cost_bps_per_side"].eq(10))]
    production_text = "PRODUCTION SKILL CHANGE SUPPORTED" if skill_change else "NO PRODUCTION SKILL CHANGE"
    generated_folds = int(freeze_registry["candidate_generation_status"].eq("FROZEN_CANDIDATES").sum()) if not freeze_registry.empty else 0
    no_candidate_folds = int(freeze_registry["candidate_generation_status"].eq("NO_STABLE_CANDIDATE").sum()) if not freeze_registry.empty else 0
    pattern_summary = _pattern_power_evidence(panel)
    ordinary_exit = _named_metric(exit_sensitivity, "exit_policy", "ordinary_stop8_profit24_no_power_lock")
    ibd_exit = _named_metric(exit_sensitivity, "exit_policy", "main_ibd_power_lock_stop8_profit24")
    max_paired = int(pd.to_numeric(contrasts.get("paired_week_routes", pd.Series(dtype=float)), errors="coerce").max()) if not contrasts.empty else 0
    improvement_source = (
        "No stable train-generated selector candidate exists, so selector improvement and selector/exit interaction are not identifiable; exit-policy sensitivity is reported separately."
        if balanced.startswith("NO_STABLE")
        else "The selected rule set passed the machine Pareto table; selector/exit interaction remains unestimated because exit-policy sensitivity is evaluated separately on B0."
    )
    answers = [
        f"Pool adequacy: {panel['snapshot_date'].nunique()} independent snapshot weeks, {int(panel['forward_8w_censored'].eq(False).sum())} complete 8-week events, and at most {max_paired} paired week-route contrasts; machine statuses below determine each rule's identifiability.",
        _rule_answer(decisions, "Status"),
        _rule_answer(decisions, "Close > Trigger"),
        _rule_answer(decisions, "Fresh Zone"),
        _rule_answer(decisions, "Entry Volume"),
        _rule_answer(decisions, "Geometry"),
        _rule_answer(decisions, "Pullback"),
        _rule_answer(decisions, "EPS"),
        _rule_answer(decisions, "Industry"),
        _rule_answer(decisions, "TopK"),
        f"Three-week pivot +20% is descriptive pattern evidence, distinct from entry-fill gain: {pattern_summary}.",
        f"IBD eight-week Power Lock versus ordinary -8%/+24%: ordinary={ordinary_exit}; power_lock={ibd_exit}.",
        f"Power Lock drawdown and capital occupancy use the same equity as-of and are read from max_drawdown_pct/cash_utilization in those two records: ordinary={ordinary_exit}; power_lock={ibd_exit}.",
        f"B0 portfolio credibility: explicit-ledger capacity 3 / 10 bps metric={b0_port.to_dict('records')[:1]}.",
        f"Most balanced machine result: {balanced}; Pareto={pareto_decisions.to_dict('records')}; portfolio={best_port.to_dict('records')[:1]}.",
        improvement_source,
        f"Production Skill decision: {production_text}; this value is computed from matching prospective rule-family support plus a fully passed Pareto candidate.",
        "Prospective holdout remains locked until future pools complete the 40-trading-day window; retrospective inspected data cannot satisfy that confirmation requirement.",
    ]
    lines = [
        "# Candidate-Level IBD Rule Audit and Portfolio Validation",
        "",
        "Period label: `retrospective_final_test`; data status: `retrospective_inspected_data`. This is not a sealed holdout.",
        "",
        "## Coverage",
        f"- Candidate signal events: {len(panel)}",
        f"- Unique tickers: {panel['code'].nunique()}",
        f"- Complete 8w outcomes: {int(panel['forward_8w_censored'].eq(False).sum())}",
        f"- Coverage regimes: {', '.join(sorted(drift['coverage_regime'].unique()))}",
        "",
        "## Rule Answers",
        *[f"{index}. {line}" for index, line in enumerate(answers, 1)],
        "",
        "## Rule Set Generation",
        f"- Folds with a train-generated frozen candidate: {generated_folds}",
        f"- Folds returning NO_STABLE_CANDIDATE: {no_candidate_folds}",
        "- Every evaluation row is labelled `blocked_retrospective_evaluation`; test outcomes do not participate in rule generation.",
        "",
        "## Power And Exit Evidence",
        f"- Pattern power events (pivot +20% within three breakout weeks): {int(panel['pattern_power_trigger'].eq(True).sum())}",
        f"- Trade power events (trigger after simulated entry): {int(panel['trade_power_trigger'].eq(True).sum())}",
        "- Pivot power and entry-fill +20% remain separate labels; selector evidence and exit-policy evidence remain separate tables.",
        "",
        "## Portfolio And Pareto",
        f"- B0 explicit-ledger metric (capacity 3, 10 bps/side): {b0_port.to_dict('records')[:1]}",
        f"- Balanced machine decision: `{balanced}`; matching metric: {best_port.to_dict('records')[:1]}",
        f"- Pareto decision: {pareto_decisions.to_dict('records')}",
        f"- Production decision: `{production_text}`.",
        "- Prospective holdout remains required before production modification.",
        "",
        "## Machine Decisions",
        decisions.to_markdown(index=False),
        "",
        "## B0 Atomic Ablation Snapshot",
        ablations.head(20).to_markdown(index=False),
        "",
        "## Rule Set OOS",
        oos.to_markdown(index=False) if not oos.empty else "No mature OOS folds with complete 8w labels.",
        "",
        "## Pareto Criteria",
        pareto_criteria.to_markdown(index=False) if not pareto_criteria.empty else "No train-generated candidate reached Pareto evaluation.",
        "",
        production_text,
    ]
    return "\n".join(lines) + "\n"


def render_acceptance_summary(
    panel: pd.DataFrame,
    production_b0_invariant: pd.DataFrame,
    atomic_invariant: pd.DataFrame,
    freeze_registry: pd.DataFrame,
    pareto_decisions: pd.DataFrame,
    metrics: pd.DataFrame,
    balanced: str,
    skill_change: bool,
) -> str:
    b0 = metrics[
        metrics["variant"].eq("B0_PIT_VERIFIED")
        & metrics["capacity"].eq(3)
        & metrics["cost_bps_per_side"].eq(10)
    ]
    lines = [
        "# Acceptance Summary",
        "",
        f"- Candidate events: {len(panel)}; independent snapshot weeks: {panel['snapshot_date'].nunique()}.",
        f"- Production B0 code/order mismatches: {int(production_b0_invariant['code_order_mismatches'].sum())}.",
        f"- Production B0 reason mismatches: {int(production_b0_invariant['reason_code_mismatches'].sum())}; risk mismatches: {int(production_b0_invariant['risk_code_mismatches'].sum())}.",
        f"- Atomic non-target trace violations: {int(atomic_invariant['non_target_trace_violations'].sum())}.",
        f"- Frozen fold hashes: {freeze_registry.get('frozen_rule_hash', pd.Series(dtype=str)).nunique()} across {len(freeze_registry)} folds.",
        f"- Balanced candidate: {balanced}.",
        f"- Pareto decisions: {pareto_decisions.to_dict('records')}.",
        f"- B0 capacity 3 / 10 bps metric: {b0.to_dict('records')[:1]}.",
        f"- Production Skill recommendation: {'CHANGE SUPPORTED' if skill_change else 'NO PRODUCTION SKILL CHANGE'}.",
    ]
    return "\n".join(lines) + "\n"


def _rule_answer(decisions: pd.DataFrame, family: str) -> str:
    row = decisions[decisions["rule_family"].eq(family)] if not decisions.empty else pd.DataFrame()
    return rule_answer_lines(row)[0] if not row.empty else f"{family}: INSUFFICIENT_EVIDENCE / UNKNOWN; effect unavailable."


def _named_metric(frame: pd.DataFrame, key: str, value: str) -> dict[str, Any]:
    if frame.empty or key not in frame.columns:
        return {}
    rows = frame[frame[key].eq(value)]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _pattern_power_evidence(panel: pd.DataFrame) -> dict[str, Any]:
    complete = panel[panel["forward_8w_censored"].eq(False)]
    rows: dict[str, Any] = {}
    known_state = complete["pattern_power_trigger"].map(to_bool)
    rows["unknown"] = {"events": int(known_state.isna().sum())}
    for state, group in complete[known_state.notna()].groupby(known_state.dropna().map(bool)):
        label = "triggered" if bool(state) else "not_triggered"
        rows[label] = {
            "events": len(group),
            "weeks": int(group["snapshot_date"].nunique()),
            "mean_8w_return_pct": float(group["forward_8w_return_pct"].mean()) if len(group) else np.nan,
            "stop_8_within_40d_rate": float(group["stop_8_within_40d"].eq(True).mean()) if len(group) else np.nan,
            "profit_24_within_40d_rate": float(group["profit_24_within_40d"].eq(True).mean()) if len(group) else np.nan,
        }
    return rows


def render_skill_proposal(balanced: str, decisions: pd.DataFrame, pareto_decisions: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Proposed Skill Change",
            "",
            f"Candidate: `{balanced}`.",
            "",
            decisions.to_markdown(index=False),
            "",
            pareto_decisions.to_markdown(index=False),
        ]
    ) + "\n"


def _write_panel(panel: pd.DataFrame, parquet_path: Path, csv_path: Path) -> None:
    try:
        panel.to_parquet(parquet_path, index=False)
    except Exception:
        panel.to_pickle(parquet_path)
    panel.to_csv(csv_path, index=False)


def _complete_rate(frame: pd.DataFrame, columns: list[str]) -> float:
    if frame.empty:
        return 0.0
    return float(frame[columns].apply(lambda col: col.map(to_float).notna()).all(axis=1).mean())


def _status_count(panel: pd.DataFrame, status: str) -> int:
    return int(panel["ibd_entry_status"].astype(str).str.upper().eq(status).sum())


def _price_cache_as_of(prices: dict[str, pd.DataFrame]) -> pd.Timestamp | None:
    dates = [pd.Timestamp(index) for frame in prices.values() for index in normalize_bars(frame).index]
    return max(dates) if dates else None


def _selection_valuation_start(
    selections: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    valuation_as_of: pd.Timestamp | None,
) -> pd.Timestamp | None:
    if selections.empty:
        return None
    normalized = {code: normalize_bars(frame) for code, frame in prices.items()}
    dates: list[pd.Timestamp] = []
    for _, pick in selections[["snapshot_date", "code"]].drop_duplicates().iterrows():
        signal_date = pd.to_datetime(pick["snapshot_date"], errors="coerce")
        if pd.isna(signal_date):
            continue
        entry = next_bar_after(
            normalized.get(str(pick["code"]), pd.DataFrame()),
            pd.Timestamp(signal_date),
        )
        if entry is not None and (valuation_as_of is None or entry[0] <= valuation_as_of):
            dates.append(pd.Timestamp(entry[0]))
    return min(dates) if dates else None


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return ""


def _simple_yaml(value: Any, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_simple_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {json.dumps(item, ensure_ascii=False, default=str)}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(_simple_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}- {json.dumps(item, ensure_ascii=False, default=str)}")
        return "\n".join(lines)
    return f"{prefix}{json.dumps(value, ensure_ascii=False, default=str)}"


if __name__ == "__main__":
    main()
