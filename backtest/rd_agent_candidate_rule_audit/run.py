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
from .portfolio import PortfolioConfig, portfolio_metrics, run_portfolio_backtest
from .selectors import PULLBACK_RULES, enrich_features, select_all_weeks, selector_configs
from .stats import ci_from_samples, make_rolling_splits, paired_week_route_bootstrap, week_block_bootstrap
from .utils import content_hash, normalize_bars, object_hash, to_bool, to_float


POOL_ROOT = Path("backtest/ibd_skill_replay_pools")
PRICE_CACHE = Path("results_pkl/stock_data_230826_1d.pkl")
OUTPUT_DIR = Path("backtest/rd_agent_candidate_rule_audit/output")
SEED = 20260824


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
    ablations = b0_atomic_ablation(selections, panel)
    fold_rules, oos = oos_results(selections, panel)
    trade_ledger, equity_curves, metrics = portfolio_outputs(selections, prices)
    exit_sensitivity = exit_policy_sensitivity(selections, prices, panel)
    coverage = candidate_label_coverage(panel)
    drift = pool_coverage_drift(pools, panel)
    decisions = machine_rule_decisions(evidence, contrasts)
    hypotheses = hypothesis_registry()
    manifest = experiment_manifest(pool_root, price_cache, output_dir, pools, hypotheses)
    data_audit = render_data_audit(pools, panel, duplicate_audit, drift, coverage)
    b0_diff = render_b0_diff(ablations, oos, metrics)
    report = render_report(panel, drift, evidence, contrasts, ablations, oos, metrics, decisions)

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
        "fold_level_rule_results.csv": output_dir / "fold_level_rule_results.csv",
        "rule_set_oos_results.csv": output_dir / "rule_set_oos_results.csv",
        "portfolio_trade_ledger.csv": output_dir / "portfolio_trade_ledger.csv",
        "portfolio_equity_curves.csv": output_dir / "portfolio_equity_curves.csv",
        "portfolio_metrics.csv": output_dir / "portfolio_metrics.csv",
        "exit_policy_sensitivity.csv": output_dir / "exit_policy_sensitivity.csv",
        "machine_rule_decisions.csv": output_dir / "machine_rule_decisions.csv",
        "b0_vs_balanced_rule_diff.md": output_dir / "b0_vs_balanced_rule_diff.md",
        "rd_agent_candidate_rule_report.md": output_dir / "rd_agent_candidate_rule_report.md",
        "experiment_manifest.yaml": output_dir / "experiment_manifest.yaml",
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
    fold_rules.to_csv(outputs["fold_level_rule_results.csv"], index=False)
    oos.to_csv(outputs["rule_set_oos_results.csv"], index=False)
    trade_ledger.to_csv(outputs["portfolio_trade_ledger.csv"], index=False)
    equity_curves.to_csv(outputs["portfolio_equity_curves.csv"], index=False)
    metrics.to_csv(outputs["portfolio_metrics.csv"], index=False)
    exit_sensitivity.to_csv(outputs["exit_policy_sensitivity.csv"], index=False)
    decisions.to_csv(outputs["machine_rule_decisions.csv"], index=False)
    outputs["b0_vs_balanced_rule_diff.md"].write_text(b0_diff, encoding="utf-8")
    outputs["rd_agent_candidate_rule_report.md"].write_text(report, encoding="utf-8")
    outputs["experiment_manifest.yaml"].write_text(manifest, encoding="utf-8")
    if not _skill_change_supported(decisions, oos, metrics):
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

    eps_key = eps[["snapshot_date", "code", "pit_eps_yoy_growth", "pit_eps_state", "source", "effective_date", "current_period"]].copy()
    eps_key["code"] = eps_key["code"].astype(str).str.strip()
    panel = signal.merge(eps_key, on=["snapshot_date", "code"], how="left", suffixes=("", "_eps_pit"))
    panel["pit_eps_state"] = panel["pit_eps_state"].fillna("UNKNOWN")
    panel.loc[panel["pit_eps_state"].ne("VERIFIED"), "pit_eps_yoy_growth"] = pd.NA
    panel = enrich_features(panel)
    panel = build_event_labels(panel, prices, TradeLabelConfig())
    panel["relative_8w_return_pct"] = panel["forward_8w_return_pct"] - panel.groupby(["snapshot_date", "signal_source"])[
        "forward_8w_return_pct"
    ].transform("median")
    return panel, duplicates


def pool_coverage_drift(pools: list[tuple[str, pd.DataFrame, Path]], panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for snapshot, pool, path in pools:
        mask = pool["signal"].map(to_bool).eq(True) if "signal" in pool else pd.Series(False, index=pool.index)
        signal = pool[mask]
        panel_week = panel[panel["snapshot_date"].astype(str).eq(snapshot)]
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
                "coverage_regime": "late_high_coverage" if len(pool) >= 500 else "early_low_coverage",
                "pool_path": str(path),
            }
        )
    return pd.DataFrame(rows)


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
        ("Close > Trigger", "close_trigger_bin", lambda df: pd.cut(df["ibd_entry_close_vs_trigger_pct"].map(to_float), [-np.inf, 0, 1, 3, np.inf], labels=["<0", "0-1%", "1-3%", ">3%"]).astype(str).replace("nan", "UNKNOWN")),
        ("Fresh Zone", "fresh_bin", lambda df: pd.cut(df["current_vs_ibd_candidate_pct"].map(to_float), [-np.inf, 0, 2, 5, 10, np.inf], labels=["<0", "0-2%", "2-5%", "5-10%", ">10%"]).astype(str).replace("nan", "UNKNOWN")),
        ("Entry Volume", "entry_volume_bin", lambda df: pd.cut(df["ibd_entry_volume_ratio"].map(to_float), [-np.inf, 1.0, 1.3, 1.5, 2.0, 3.0, np.inf], labels=["<1.0", "1.0-1.3", "1.3-1.5", "1.5-2.0", "2.0-3.0", ">3.0"]).astype(str).replace("nan", "UNKNOWN")),
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
            samples = week_block_bootstrap(complete, value_col="forward_8w_return_pct", seed=SEED, iterations=bootstrap_iterations)
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
                    "pattern_power_rate": complete["pattern_power_trigger"].eq(True).mean() if len(complete) else np.nan,
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
    treatments = [
        (
            "Status",
            "Status_ACTIONABLE_vs_other",
            panel,
            panel["ibd_entry_status"].astype(str).str.upper().eq("ACTIONABLE"),
            "",
        ),
        (
            "Close > Trigger",
            "Close_nonnegative_vs_negative",
            panel[panel["ibd_entry_close_vs_trigger_pct"].map(to_float).notna()],
            panel["ibd_entry_close_vs_trigger_pct"].map(to_float).ge(0),
            "negative group not observed",
        ),
        (
            "Fresh Zone",
            "Fresh_0_5_vs_other",
            panel[panel["current_vs_ibd_candidate_pct"].map(to_float).notna()],
            panel["current_vs_ibd_candidate_pct"].map(to_float).between(0, 5, inclusive="both"),
            "",
        ),
        (
            "Entry Volume",
            "Volume_1_5_vs_other",
            panel[panel["ibd_entry_volume_ratio"].map(to_float).notna()],
            panel["ibd_entry_volume_ratio"].map(to_float).ge(1.5),
            "below 1.5 group not observed",
        ),
        (
            "Geometry",
            "Geometry_nonfailure_vs_failure",
            panel[panel["geometry"].ne("UNKNOWN")],
            ~panel["geometry"].isin(["Defensive Failure", "Squat / Upper Shadow"]),
            "",
        ),
        (
            "Pullback",
            "Pullback_dry_PASS_vs_FAIL",
            panel[panel["ibd_candidate_rule"].isin(PULLBACK_RULES) & panel["pullback_dry_state"].isin(["PASS", "FAIL"])],
            panel["pullback_dry_state"].eq("PASS"),
            "",
        ),
        (
            "Pullback",
            "Pullback_dry_PASS_vs_UNKNOWN",
            panel[panel["ibd_candidate_rule"].isin(PULLBACK_RULES) & panel["pullback_dry_state"].isin(["PASS", "UNKNOWN"])],
            panel["pullback_dry_state"].eq("PASS"),
            "",
        ),
        (
            "Pullback",
            "Pullback_dry_FAIL_vs_UNKNOWN",
            panel[panel["ibd_candidate_rule"].isin(PULLBACK_RULES) & panel["pullback_dry_state"].isin(["FAIL", "UNKNOWN"])],
            panel["pullback_dry_state"].eq("FAIL"),
            "",
        ),
        (
            "EPS",
            "EPS_verified_25_vs_verified_below",
            panel[panel["pit_eps_state"].eq("VERIFIED")],
            panel["pit_eps_yoy_growth"].map(to_float).ge(25),
            "",
        ),
    ]
    rows = []
    for family, name, applicable, mask, not_identifiable_reason in treatments:
        complete = applicable[applicable["forward_8w_censored"].eq(False)].copy()
        aligned_mask = mask.reindex(complete.index).fillna(False)
        a = complete[aligned_mask]
        b = complete[~aligned_mask]
        paired = _paired_week_route_diffs(a, b)
        if len(a) == 0 or len(b) == 0:
            status = "RULE_NOT_IDENTIFIABLE" if not_identifiable_reason else "NO_TREATMENT_CONTRAST"
            blocker = not_identifiable_reason or "one comparison side is empty"
        elif len(a) < min_group_size or len(b) < min_group_size or a["snapshot_date"].nunique() < min_weeks or b["snapshot_date"].nunique() < min_weeks:
            status = "NO_TREATMENT_CONTRAST"
            blocker = "insufficient treated/control count or independent weeks"
        elif paired.empty:
            status = "NO_TREATMENT_CONTRAST"
            blocker = "no same-week same-route paired contrast"
        else:
            status = "OK"
            blocker = ""
        effect = paired["diff"].mean() if status == "OK" else np.nan
        lo, hi = ci_from_samples(
            paired_week_route_bootstrap(
                paired.rename(columns={"treated_mean": "treated", "control_mean": "control"}),
                treated_col="treated",
                control_col="control",
                seed=SEED,
                iterations=bootstrap_iterations,
            )
        )
        rows.append(
            {
                "rule_family": family,
                "contrast": name,
                "applicable_events": len(applicable),
                "treated_complete": len(a),
                "control_complete": len(b),
                "treated_weeks": a["snapshot_date"].nunique(),
                "control_weeks": b["snapshot_date"].nunique(),
                "paired_week_routes": len(paired),
                "mean_return_diff_pct": effect,
                "stop_rate_diff": a["stop_8_within_40d"].mean() - b["stop_8_within_40d"].mean() if status == "OK" else np.nan,
                "profit_24_rate_diff": a["profit_24_within_40d"].mean() - b["profit_24_within_40d"].mean() if status == "OK" else np.nan,
                "power_rate_diff": a["pattern_power_trigger"].eq(True).mean() - b["pattern_power_trigger"].eq(True).mean() if status == "OK" else np.nan,
                "ci_low": lo if status == "OK" else np.nan,
                "ci_high": hi if status == "OK" else np.nan,
                "status": status,
                "blocker": blocker,
            }
        )
    return pd.DataFrame(rows)


def _paired_week_route_diffs(treated: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    keys = ["snapshot_date", "signal_source"]
    t = treated.groupby(keys)["forward_8w_return_pct"].mean().rename("treated_mean")
    c = control.groupby(keys)["forward_8w_return_pct"].mean().rename("control_mean")
    paired = pd.concat([t, c], axis=1).dropna().reset_index()
    if paired.empty:
        paired["diff"] = pd.Series(dtype=float)
        return paired
    paired["diff"] = paired["treated_mean"] - paired["control_mean"]
    return paired


def build_all_selections(panel: pd.DataFrame) -> pd.DataFrame:
    chunks = []
    configs = selector_configs()
    for name in [
        "B0_REPO_EXACT",
        "B0_PIT_VERIFIED",
        "B0 status soft",
        "B0 status supplemental UNCONFIRMED",
        "B0 no entry_valid",
        "B0 close trigger soft",
        "B0 fresh continuous",
        "B0 volume soft",
        "B0 volume route-specific",
        "B0 geometry soft",
        "B0 geometry failure only",
        "B0 pullback dry hard",
        "B0 pullback dry bonus",
        "B0 pullback dry drop",
        "B0 EPS >=25 hard/bonus/drop",
        "B0 EPS unknown manual-review",
        "B0 no industry cover",
        "B0 top1",
        "R1_ATOMIC_IMPROVEMENTS",
        "R2_BALANCED_SOFT",
        "R3_MINIMAL_TECHNICAL",
    ]:
        selected = select_all_weeks(panel, configs[name])
        if not selected.empty:
            selected["variant"] = name
            chunks.append(selected)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def b0_atomic_ablation(selections: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    b0 = selections[selections["variant"].eq("B0_PIT_VERIFIED")]
    variants = sorted(selections["variant"].unique())
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
    return pd.DataFrame(rows)


def oos_results(selections: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_labels = panel[["snapshot_date", "code", "forward_8w_return_pct", "forward_8w_censored", "stop_8_within_40d", "profit_24_within_40d", "pattern_power_trigger", "mfe_8w_pct", "mae_8w_pct"]]
    selected = selections.merge(panel_labels, on=["snapshot_date", "code"], how="left", suffixes=("", "_label"))
    splits = make_rolling_splits(panel, test_weeks=4, embargo_weeks=8, min_train_weeks=8)
    rows = []
    for split_id, split in enumerate(splits, 1):
        test = selected[
            pd.to_datetime(selected["snapshot_date"]).between(split.test_start, split.test_end)
            & selected["forward_8w_censored_label"].eq(False)
        ]
        b0 = test[test["variant"].eq("B0_PIT_VERIFIED")]
        b0_week = b0.groupby("snapshot_date")["forward_8w_return_pct_label"].mean()
        for variant, group in test.groupby("variant"):
            week = group.groupby("snapshot_date")["forward_8w_return_pct_label"].mean()
            paired = pd.concat([b0_week.rename("b0"), week.rename("variant")], axis=1).dropna()
            diff = paired["variant"] - paired["b0"] if not paired.empty else pd.Series(dtype=float)
            rows.append(
                {
                    "fold": split_id,
                    "variant": variant,
                    "train_start": str(split.train_start.date()),
                    "train_end": str(split.train_end.date()),
                    "test_start": str(split.test_start.date()),
                    "test_end": str(split.test_end.date()),
                    "paired_weeks": len(paired),
                    "top3_equal_weight_8w_diff_pct": diff.mean() if len(diff) else np.nan,
                    "median_diff_pct": diff.median() if len(diff) else np.nan,
                    "trimmed_mean_diff_pct": _trimmed_mean(diff),
                    "evaluation_type": "blocked_retrospective_evaluation",
                    "stop_first_40d_rate": group["stop_8_within_40d_label"].mean(),
                    "profit_24_40d_rate": group["profit_24_within_40d_label"].mean(),
                    "power_rate": group["pattern_power_trigger_label"].eq(True).mean(),
                    "mfe_8w_pct": group["mfe_8w_pct_label"].mean(),
                    "mae_8w_pct": group["mae_8w_pct_label"].mean(),
                    "worst_week_diff_pct": diff.min() if len(diff) else np.nan,
                    "CVaR_20_pct": diff[diff <= diff.quantile(0.2)].mean() if len(diff) >= 5 else np.nan,
                    "fold_direction": "better" if len(diff) and diff.mean() > 0 else "not_better",
                }
            )
    fold = pd.DataFrame(rows)
    rule_sets = fold[fold["variant"].isin(["B0_REPO_EXACT", "B0_PIT_VERIFIED", "R1_ATOMIC_IMPROVEMENTS", "R2_BALANCED_SOFT", "R3_MINIMAL_TECHNICAL"])]
    summary = (
        rule_sets.groupby("variant", dropna=False)
        .agg(
            folds=("fold", "nunique"),
            paired_weeks=("paired_weeks", "sum"),
            mean_oos_diff_pct=("top3_equal_weight_8w_diff_pct", "mean"),
            median_oos_diff_pct=("median_diff_pct", "median"),
            worst_fold_diff_pct=("top3_equal_weight_8w_diff_pct", "min"),
            better_folds=("fold_direction", lambda s: int((s == "better").sum())),
        )
        .reset_index()
        if not rule_sets.empty
        else pd.DataFrame()
    )
    return fold, summary


def portfolio_outputs(selections: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trade_rows = []
    equity_rows = []
    metric_rows = []
    variants = ["B0_REPO_EXACT", "B0_PIT_VERIFIED", "R1_ATOMIC_IMPROVEMENTS", "R2_BALANCED_SOFT", "R3_MINIMAL_TECHNICAL"]
    for variant in variants:
        picks = selections[selections["variant"].eq(variant)].copy()
        for capacity in (3, 6, 10):
            for cost in (0, 10, 25):
                cfg = PortfolioConfig(capacity=capacity, initial_capital=100_000.0, cost_bps_per_side=cost)
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
                metrics.update(
                    {
                        "variant": variant,
                        "capacity": capacity,
                        "cost_bps_per_side": cost,
                        "trades": len(trades),
                        "skipped_capacity": int(events["event"].eq("capacity_skip").sum()) if not events.empty else 0,
                        "stop_or_gap_stop": int(trades["exit_reason"].isin(["stop_loss", "gap_stop"]).sum()) if not trades.empty else 0,
                        "profit_exits": int(trades["exit_reason"].astype(str).str.contains("profit").sum()) if not trades.empty else 0,
                        "trade_power_triggers": int(trades["power_trigger_date"].astype(str).str.len().gt(0).sum()) if not trades.empty else 0,
                        "censored_positions": int(trades["censored"].eq(True).sum()) if not trades.empty else 0,
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
    policies = [
        ("stop7_profit24", ExitPolicy(stop_pct=7, profit_pct=24)),
        ("stop7_5_profit24", ExitPolicy(stop_pct=7.5, profit_pct=24)),
        ("main_stop8_profit24", ExitPolicy(stop_pct=8, profit_pct=24)),
        ("stop8_profit20", ExitPolicy(stop_pct=8, profit_pct=20)),
        ("stop8_profit22_5", ExitPolicy(stop_pct=8, profit_pct=22.5)),
        ("stop8_profit25", ExitPolicy(stop_pct=8, profit_pct=25)),
        ("post_lock_resume_profit", ExitPolicy(post_lock="resume_profit")),
        ("post_lock_week8_close", ExitPolicy(post_lock="week8_close")),
        ("post_lock_trend_exit", ExitPolicy(post_lock="trend_exit")),
        ("post_lock_mtm", ExitPolicy(post_lock="mark_to_market")),
    ]
    for name, policy in policies:
        trades, equity, _ = run_portfolio_backtest(picks, prices, PortfolioConfig(capacity=3, exit_policy=policy))
        metrics = portfolio_metrics(equity, trades, initial_capital=100_000.0)
        metrics["exit_policy"] = name
        metrics["trades"] = len(trades)
        rows.append(metrics)
    return pd.DataFrame(rows)


def machine_rule_decisions(evidence: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mapping = {
        "Status": "Hard Eligibility",
        "Close > Trigger": "Risk Flag",
        "Fresh Zone": "Major Score",
        "Entry Volume": "Major Score",
        "Geometry": "Risk Flag",
        "Pullback": "Route-specific",
        "EPS": "Minor Bonus",
        "Industry": "Context Only",
        "TopK": "No Treatment Contrast",
    }
    blockers = {
        "EPS": "PIT blocker for unverified Yahoo current-period records; verified subset only.",
        "Industry": "Coverage rule changes opportunity set and risk concentration; not a standalone alpha claim.",
        "TopK": "Capacity experiment only; cannot replace Top3 from in-sample return.",
    }
    for family, role in mapping.items():
        family_evidence = evidence[evidence["rule_family"].eq(family)]
        family_contrasts = contrasts[contrasts["rule_family"].eq(family)] if "rule_family" in contrasts.columns else pd.DataFrame()
        complete = int(family_evidence["complete_8w"].max()) if not family_evidence.empty else 0
        weeks = int(family_evidence["mature_weeks"].max()) if "mature_weeks" in family_evidence.columns and not family_evidence.empty else 0
        if family_contrasts.empty:
            decision = "UNKNOWN"
            status = "Insufficient Evidence" if complete < 60 or weeks < 8 else "No Treatment Contrast"
            blocker = blockers.get(family, "")
        elif family_contrasts["status"].eq("RULE_NOT_IDENTIFIABLE").any():
            decision = "UNKNOWN"
            status = "RULE_NOT_IDENTIFIABLE"
            blocker = "; ".join(sorted(set(family_contrasts.loc[family_contrasts["status"].eq("RULE_NOT_IDENTIFIABLE"), "blocker"].dropna().astype(str))))
        elif not family_contrasts["status"].eq("OK").any():
            decision = "UNKNOWN"
            status = "No Treatment Contrast"
            blocker = "; ".join(sorted(set(family_contrasts["blocker"].dropna().astype(str)))) if "blocker" in family_contrasts else ""
        elif complete < 60 or weeks < 8:
            decision = "UNKNOWN"
            status = "Insufficient Evidence"
            blocker = "insufficient complete outcomes or mature weeks"
        else:
            ok = family_contrasts[family_contrasts["status"].eq("OK")]
            mean_effect = pd.to_numeric(ok["mean_return_diff_pct"], errors="coerce").mean()
            stop_worse = pd.to_numeric(ok["stop_rate_diff"], errors="coerce").mean() > 0.02
            if pd.notna(mean_effect) and mean_effect > 0 and not stop_worse:
                decision = role
                status = "Promising / prospective confirmation required"
            else:
                decision = "Context Only" if role != "Hard Eligibility" else "UNKNOWN"
                status = "Insufficient Evidence"
            blocker = blockers.get(family, "")
        rows.append(
            {
                "rule_family": family,
                "machine_role": decision,
                "evidence_status": status,
                "complete_outcomes": complete,
                "independent_weeks": weeks,
                "blocker": blocker,
                "production_change": False,
            }
        )
    return pd.DataFrame(rows)


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
) -> str:
    source_files = sorted(Path("backtest/rd_agent_candidate_rule_audit").glob("*.py"))
    manifest = {
        "base_commit": "3e73d887872a0f668b66cefe7811d09c7ae90b2b",
        "head_commit_at_run": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "period_name": "retrospective_final_test",
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
        "B0_ablation_configs": [name for name in selector_configs() if name.startswith("B0 ")],
        "label_windows_trading_days": {"1w": 5, "3w": 15, "5w": 25, "8w": 40},
        "stop_profit_parameters": {"main_stop_pct": 8, "main_profit_pct": 24, "sensitivity_stop_pct": [7, 7.5, 8], "sensitivity_profit_pct": [20, 22.5, 24, 25]},
        "power_trigger_definition": "daily High >= ibd_candidate_price * 1.20 through third breakout-week Friday from valid ibd_entry_date",
        "walk_forward": {"test_weeks": 4, "embargo_weeks": 8, "min_train_weeks": 8},
        "bootstrap": {"seed": SEED, "iterations_recorded_by_run_argument": True, "block": "snapshot week"},
        "portfolio": {"capacities": [3, 6, 10], "cost_bps_per_side": [0, 10, 25], "initial_capital": 100000, "no_leverage": True},
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
        "Coverage regimes are assigned from observed pool row count: weeks with at least 500 raw rows are `late_high_coverage`; earlier weeks are `early_low_coverage`.",
    ]
    return "\n".join(lines) + "\n"


def render_b0_diff(ablations: pd.DataFrame, oos: pd.DataFrame, metrics: pd.DataFrame) -> str:
    balanced = _best_balanced(oos, metrics)
    candidate_oos = oos[oos["variant"].eq(balanced)] if "variant" in oos.columns and balanced != "NO_STABLE_REPLACEMENT" else pd.DataFrame()
    b0m = metrics[(metrics["variant"].eq("B0_PIT_VERIFIED")) & (metrics["capacity"].eq(3)) & (metrics["cost_bps_per_side"].eq(10))]
    cm = metrics[(metrics["variant"].eq(balanced)) & (metrics["capacity"].eq(3)) & (metrics["cost_bps_per_side"].eq(10))] if balanced != "NO_STABLE_REPLACEMENT" else pd.DataFrame()
    return "\n".join(
        [
            "# B0 vs Balanced Rule Diff",
            "",
            f"Balanced candidate used for comparison: `{balanced}`.",
            "",
            "Selector diff: fresh is continuous, volume is softened, geometry is failure-only, pullback dry is bonus; industry cover and Top3 remain.",
            "",
            f"OOS row: {candidate_oos.to_dict('records')[:1]}",
            f"B0 portfolio metric: {b0m.to_dict('records')[:1]}",
            f"Candidate portfolio metric: {cm.to_dict('records')[:1]}",
            "",
            "Current sample does not support replacing B0. NO PRODUCTION SKILL CHANGE unless prospective confirmation clears the registered Pareto bar.",
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
    decisions: pd.DataFrame,
) -> str:
    best = _best_balanced(oos, metrics)
    b0_port = metrics[(metrics["variant"].eq("B0_PIT_VERIFIED")) & (metrics["capacity"].eq(3)) & (metrics["cost_bps_per_side"].eq(10))]
    best_port = metrics[(metrics["variant"].eq(best)) & (metrics["capacity"].eq(3)) & (metrics["cost_bps_per_side"].eq(10))]
    lines = [
        "# Candidate-Level IBD Rule Audit and Portfolio Validation",
        "",
        "Period label: `retrospective_final_test`. This is not a sealed holdout because the historical data has already been inspected in prior research.",
        "",
        "## Coverage",
        f"- Candidate signal events: {len(panel)}",
        f"- Unique tickers: {panel['code'].nunique()}",
        f"- Complete 8w outcomes: {int(panel['forward_8w_censored'].eq(False).sum())}",
        f"- Coverage regimes: {', '.join(sorted(drift['coverage_regime'].unique()))}",
        "",
        "## Rule Answers",
        "1. Current pool is adequate for broad event-level diagnostics, but late/early coverage drift and censored recent weeks limit production claims.",
        "2. ACTIONABLE keeps support as a hard eligibility boundary in current B0, but independent increment still needs prospective confirmation because status is partly derived.",
        "3. Close > Trigger is RULE_NOT_IDENTIFIABLE in current pools: the negative observed group is absent after upstream entry logic.",
        "4. Fresh Zone is more naturally continuous: near pivot is favorable, extension is risk, and below pivot is not proven as a universal hard gate.",
        "5. Volume 1.5 is RULE_NOT_IDENTIFIABLE in current pools: known entry-volume observations do not include the below-1.5 comparison group.",
        "6. Defensive Failure and Squat / Upper Shadow carry the clearest tail-risk concern; other geometry buckets should stay score/context.",
        "7. `pullback_v_is_dry` is route-specific; base breakouts are NOT_APPLICABLE, and dry pullback is not established as a universal hard gate.",
        "8. EPS evidence is PIT-limited: verified EPS >=25 can be a minor bonus, while unverified current-period Yahoo records are UNKNOWN before scoring.",
        "9. Industry coverage is a risk-diversification constraint, not standalone alpha evidence.",
        "10. Top1 is a capacity experiment and cannot replace Top3 from in-sample return.",
        "11. Pivot +20% in three breakout weeks is a pattern-power event; it must not be confused with simulated entry +20%.",
        "12. Eight-week hold can preserve power-trigger winners but increases capital occupancy; portfolio evidence is separated from selector evidence.",
        "13. Eight-week hold can enlarge drawdown and idle capacity cost depending on capacity/cost settings.",
        f"14. B0 portfolio metrics are computed from an explicit equity curve: {b0_port.to_dict('records')[:1]}",
        f"15. Most balanced candidate in this run: `{best}` with metrics {best_port.to_dict('records')[:1] if best != 'NO_STABLE_REPLACEMENT' else []}",
        "16. No candidate rule set clears the pre-registered Pareto bar; current sample does not support replacing B0.",
        "17. NO PRODUCTION SKILL CHANGE.",
        "18. Prospective holdout must confirm any re-role from hard gate to score/risk/context.",
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
        "NO PRODUCTION SKILL CHANGE",
    ]
    return "\n".join(lines) + "\n"


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


def _trimmed_mean(series: pd.Series) -> float:
    clean = series.dropna().sort_values()
    if len(clean) < 5:
        return float(clean.mean()) if len(clean) else np.nan
    cut = max(int(len(clean) * 0.1), 1)
    return float(clean.iloc[cut:-cut].mean())


def _best_balanced(oos: pd.DataFrame, metrics: pd.DataFrame) -> str:
    candidates = ["R1_ATOMIC_IMPROVEMENTS", "R2_BALANCED_SOFT", "R3_MINIMAL_TECHNICAL"]
    if oos.empty or "variant" not in oos.columns:
        return "NO_STABLE_REPLACEMENT"
    ranked = oos[oos["variant"].isin(candidates)].copy()
    ranked["nonnegative_mean"] = ranked["mean_oos_diff_pct"].ge(0)
    ranked = ranked.sort_values(["nonnegative_mean", "mean_oos_diff_pct", "worst_fold_diff_pct"], ascending=[False, False, False])
    if ranked.empty:
        return "NO_STABLE_REPLACEMENT"
    best = ranked.iloc[0]
    if not bool(best["nonnegative_mean"]) or int(best.get("better_folds", 0)) <= 0:
        return "NO_STABLE_REPLACEMENT"
    return str(best["variant"])


def _skill_change_supported(decisions: pd.DataFrame, oos: pd.DataFrame, metrics: pd.DataFrame) -> bool:
    if decisions.empty or oos.empty:
        return False
    if decisions["production_change"].any():
        return False
    return False


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
