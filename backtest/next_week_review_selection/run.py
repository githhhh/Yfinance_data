from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.rd_agent_candidate_rule_audit.labels import normalize_eps_pit
from backtest.rd_agent_candidate_rule_audit.run import load_pools, load_price_cache
from backtest.rd_agent_candidate_rule_audit.utils import content_hash, to_bool
from .labels import add_next_week_labels
from .selectors import (
    review_rules,
    select_attention_matched,
    select_b0_actionable,
    select_review_variant,
)


POOL_ROOT = Path("backtest/ibd_skill_replay_pools")
PRICE_CACHE = Path("results_pkl/stock_data_230826_1d.pkl")
OUTPUT_DIR = Path("backtest/next_week_review_selection/output")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run next-week weekend review selection research.")
    parser.add_argument("--pool-root", default=str(POOL_ROOT))
    parser.add_argument("--price-cache", default=str(PRICE_CACHE))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args(argv)
    outputs = run_research(
        pool_root=Path(args.pool_root),
        price_cache=Path(args.price_cache),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


def run_research(
    *,
    pool_root: Path = POOL_ROOT,
    price_cache: Path = PRICE_CACHE,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pools = load_pools(pool_root)
    prices = load_price_cache(price_cache)
    eps_path = pool_root / "signal_eps_pit.csv"
    eps = normalize_eps_pit(pd.read_csv(eps_path))
    panel = build_weekend_event_panel(pools, eps)
    panel = add_next_week_labels(panel, prices)

    selections = build_all_selections(panel)
    weekly = weekly_metrics(panel, selections)
    summary = overall_summary(panel, selections)
    frontier = attention_frontier(panel)
    blocked = blocked_validation_summary(panel)
    status = status_opportunity_summary(panel)
    risk_reward = risk_reward_summary(selections)
    opportunity_labels = opportunity_label_projection(panel)
    data_audit = render_data_audit(panel, pools)
    manifest = render_manifest(
        panel=panel,
        pool_root=pool_root,
        price_cache=price_cache,
        eps_path=eps_path,
    )
    report = render_report(panel, weekly, summary, frontier, blocked, status, risk_reward)

    outputs = {
        "data_audit.md": output_dir / "data_audit.md",
        "weekend_event_panel.csv": output_dir / "weekend_event_panel.csv",
        "opportunity_labels.csv": output_dir / "opportunity_labels.csv",
        "weekly_selection_counts.csv": output_dir / "weekly_selection_counts.csv",
        "baseline_vs_variants.csv": output_dir / "baseline_vs_variants.csv",
        "attention_frontier.csv": output_dir / "attention_frontier.csv",
        "blocked_validation.csv": output_dir / "blocked_validation.csv",
        "status_transition_summary.csv": output_dir / "status_transition_summary.csv",
        "risk_reward_summary.csv": output_dir / "risk_reward_summary.csv",
        "experiment_manifest.yaml": output_dir / "experiment_manifest.yaml",
        "research_report.md": output_dir / "research_report.md",
    }
    outputs["data_audit.md"].write_text(data_audit, encoding="utf-8")
    panel.to_csv(outputs["weekend_event_panel.csv"], index=False)
    opportunity_labels.to_csv(outputs["opportunity_labels.csv"], index=False)
    weekly.to_csv(outputs["weekly_selection_counts.csv"], index=False)
    summary.to_csv(outputs["baseline_vs_variants.csv"], index=False)
    frontier.to_csv(outputs["attention_frontier.csv"], index=False)
    blocked.to_csv(outputs["blocked_validation.csv"], index=False)
    status.to_csv(outputs["status_transition_summary.csv"], index=False)
    risk_reward.to_csv(outputs["risk_reward_summary.csv"], index=False)
    outputs["experiment_manifest.yaml"].write_text(manifest, encoding="utf-8")
    outputs["research_report.md"].write_text(report, encoding="utf-8")
    return {name: str(path) for name, path in outputs.items()}


def build_weekend_event_panel(
    pools: list[tuple[str, pd.DataFrame, Path]],
    eps: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for snapshot, pool, path in pools:
        frame = pool.copy()
        frame["snapshot_date"] = snapshot
        frame["pool_path"] = str(path)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if raw.empty:
        return raw

    signal = raw[
        raw["signal"].map(to_bool).eq(True)
        & raw["ibd_candidate_rule"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    signal["code"] = signal["code"].astype(str).str.strip()
    signal["_source_row_order"] = np.arange(len(signal))
    signal = (
        signal.sort_values(["snapshot_date", "code", "_source_row_order"])
        .drop_duplicates(["snapshot_date", "code"], keep="first")
        .copy()
    )

    audited_cols = ["pit_eps_yoy_growth", "pit_eps_state", "source", "effective_date", "current_period"]
    signal = signal.drop(columns=[column for column in audited_cols if column in signal.columns])
    eps_key = eps[["snapshot_date", "code", *audited_cols]].copy()
    eps_key["snapshot_date"] = eps_key["snapshot_date"].astype(str)
    eps_key["code"] = eps_key["code"].astype(str).str.strip()
    eps_key = eps_key.drop_duplicates(["snapshot_date", "code"], keep="first")
    panel = signal.merge(eps_key, on=["snapshot_date", "code"], how="left")
    panel["pit_eps_state"] = panel["pit_eps_state"].fillna("UNKNOWN")
    panel.loc[panel["pit_eps_state"].ne("VERIFIED"), "pit_eps_yoy_growth"] = pd.NA
    return panel.reset_index(drop=True)


def build_all_selections(panel: pd.DataFrame) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    rules = review_rules()
    for snapshot, group in panel.groupby("snapshot_date", sort=True):
        base = select_b0_actionable(group)
        if not base.empty:
            chunks.append(_selection_projection(base, snapshot))

        for rule in rules.values():
            selected = select_review_variant(group, rule)
            if not selected.empty:
                chunks.append(_selection_projection(selected, snapshot))

        matched = select_attention_matched(group, rules["R2_BALANCED"])
        if not matched.empty:
            chunks.append(_selection_projection(matched, snapshot))

        for cap in (10, 15, 20):
            capped = select_review_variant(group, rules["R2_BALANCED"], cap=cap)
            if not capped.empty:
                capped["variant"] = f"R2_BALANCED_CAP{cap}"
                chunks.append(_selection_projection(capped, snapshot))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _selection_projection(selected: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    cols = [
        "snapshot_date",
        "code",
        "variant",
        "review_reason",
        "_support_count",
        "_abs_vs_buy_point",
        "ibd_entry_status",
        "review_opportunity_5d",
        "opportunity_type",
        "forward_5d_return_pct",
        "mfe_5d_pct",
        "mae_5d_pct",
        "stop_8_within_5d",
        "forward_5d_censored",
    ]
    out = selected.copy()
    out["snapshot_date"] = snapshot
    return out[[column for column in cols if column in out.columns]].copy()


def _complete_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty or "forward_5d_censored" not in panel.columns:
        return panel.iloc[0:0].copy()
    return panel[panel["forward_5d_censored"].eq(False)].copy()


def weekly_metrics(panel: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    variants = _variant_names(selections)
    rows: list[dict[str, Any]] = []
    for snapshot, week_all in panel.groupby("snapshot_date", sort=True):
        week = _complete_panel(week_all)
        denom = int(week["review_opportunity_5d"].eq(True).sum())
        non_actionable_opps = int(
            (
                week["review_opportunity_5d"].eq(True)
                & week["ibd_entry_status"].fillna("").astype(str).str.upper().ne("ACTIONABLE")
            ).sum()
        )
        for variant in variants:
            selected_all = selections[
                selections["snapshot_date"].astype(str).eq(str(snapshot))
                & selections["variant"].eq(variant)
            ].copy()
            selected = selected_all[selected_all["forward_5d_censored"].eq(False)].copy() if not selected_all.empty else selected_all
            captured = int(selected["review_opportunity_5d"].eq(True).sum()) if not selected.empty else 0
            selected_non_actionable = selected[
                selected["ibd_entry_status"].fillna("").astype(str).str.upper().ne("ACTIONABLE")
            ] if not selected.empty else selected
            incremental = int(selected_non_actionable["review_opportunity_5d"].eq(True).sum()) if not selected_non_actionable.empty else 0
            rows.append(
                {
                    "snapshot_date": snapshot,
                    "variant": variant,
                    "watchlist_size": int(len(selected_all)),
                    "evaluable_watchlist_size": int(len(selected)),
                    "opportunities_available": denom,
                    "opportunities_captured": captured,
                    "capture_rate": captured / denom if denom else np.nan,
                    "non_actionable_opportunities_available": non_actionable_opps,
                    "non_actionable_opportunities_captured": incremental,
                    "opportunities_per_evaluable_review": captured / len(selected) if len(selected) else np.nan,
                    "median_5d_return_pct": _median(selected, "forward_5d_return_pct"),
                    "median_mfe_5d_pct": _median(selected, "mfe_5d_pct"),
                    "median_mae_5d_pct": _median(selected, "mae_5d_pct"),
                    "stop_8_within_5d_rate": _rate(selected, "stop_8_within_5d"),
                }
            )
    return pd.DataFrame(rows)


def overall_summary(panel: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    complete_panel = _complete_panel(panel)
    weekly = weekly_metrics(panel, selections)
    rows: list[dict[str, Any]] = []
    total_opps = int(complete_panel["review_opportunity_5d"].eq(True).sum())
    total_non_actionable = int(
        (
            complete_panel["review_opportunity_5d"].eq(True)
            & complete_panel["ibd_entry_status"].fillna("").astype(str).str.upper().ne("ACTIONABLE")
        ).sum()
    )
    for variant in _variant_names(selections):
        selected_all = selections[selections["variant"].eq(variant)].copy()
        selected = selected_all[selected_all["forward_5d_censored"].eq(False)].copy() if not selected_all.empty else selected_all
        captured = int(selected["review_opportunity_5d"].eq(True).sum()) if not selected.empty else 0
        incremental = int(
            (
                selected["review_opportunity_5d"].eq(True)
                & selected["ibd_entry_status"].fillna("").astype(str).str.upper().ne("ACTIONABLE")
            ).sum()
        ) if not selected.empty else 0
        w = weekly[weekly["variant"].eq(variant)]
        rows.append(
            {
                "variant": variant,
                "weeks": int(w["snapshot_date"].nunique()),
                "picks": int(len(selected_all)),
                "evaluable_picks": int(len(selected)),
                "avg_watchlist_size": float(w["watchlist_size"].mean()) if len(w) else np.nan,
                "median_watchlist_size": float(w["watchlist_size"].median()) if len(w) else np.nan,
                "p95_watchlist_size": float(w["watchlist_size"].quantile(0.95)) if len(w) else np.nan,
                "opportunity_capture_rate": captured / total_opps if total_opps else np.nan,
                "non_actionable_incremental_capture_rate": incremental / total_non_actionable if total_non_actionable else np.nan,
                "opportunities_per_evaluable_review": captured / len(selected) if len(selected) else np.nan,
                "median_5d_return_pct": _median(selected, "forward_5d_return_pct"),
                "median_mfe_5d_pct": _median(selected, "mfe_5d_pct"),
                "median_mae_5d_pct": _median(selected, "mae_5d_pct"),
                "stop_8_within_5d_rate": _rate(selected, "stop_8_within_5d"),
            }
        )
    return pd.DataFrame(rows).sort_values("variant").reset_index(drop=True)


def attention_frontier(panel: pd.DataFrame) -> pd.DataFrame:
    rule = review_rules()["R2_BALANCED"]
    complete_panel = _complete_panel(panel)
    total_opps = int(complete_panel["review_opportunity_5d"].eq(True).sum())
    all_weeks = sorted(panel["snapshot_date"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    for cap in (10, 15, 20):
        chunks = []
        counts = {week: 0 for week in all_weeks}
        for snapshot, week in panel.groupby("snapshot_date", sort=True):
            picked = select_review_variant(week, rule, cap=cap)
            counts[str(snapshot)] = len(picked)
            if not picked.empty:
                picked["snapshot_date"] = snapshot
                chunks.append(picked)
        selected_all = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        selected = selected_all[selected_all["forward_5d_censored"].eq(False)].copy() if not selected_all.empty else selected_all
        captured = int(selected["review_opportunity_5d"].eq(True).sum()) if not selected.empty else 0
        rows.append(
            {
                "variant": f"R2_BALANCED_CAP{cap}",
                "cap": cap,
                "avg_watchlist_size": float(np.mean(list(counts.values()))) if counts else 0.0,
                "median_watchlist_size": float(np.median(list(counts.values()))) if counts else 0.0,
                "opportunity_capture_rate": captured / total_opps if total_opps else np.nan,
                "opportunities_per_evaluable_review": captured / len(selected) if len(selected) else np.nan,
                "median_mae_5d_pct": _median(selected, "mae_5d_pct"),
            }
        )
    return pd.DataFrame(rows)


def blocked_validation_summary(panel: pd.DataFrame) -> pd.DataFrame:
    weeks = sorted(panel["snapshot_date"].astype(str).unique())
    if not weeks:
        return pd.DataFrame()
    cut = max(1, int(len(weeks) * 2 / 3))
    discovery_weeks = set(weeks[:cut])
    validation_weeks = set(weeks[cut:])
    rows: list[pd.DataFrame] = []
    for label, allowed in [("discovery", discovery_weeks), ("blocked_validation", validation_weeks)]:
        subset = panel[panel["snapshot_date"].astype(str).isin(allowed)].copy()
        if subset.empty:
            continue
        selections = build_all_selections(subset)
        summary = overall_summary(subset, selections)
        summary.insert(0, "split", label)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def status_opportunity_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    status_series = panel["ibd_entry_status"].fillna("").astype(str).str.upper()
    for status, group_all in panel.assign(_status=status_series).groupby("_status", sort=True):
        group = _complete_panel(group_all)
        rows.append(
            {
                "weekend_status": status,
                "rows": int(len(group_all)),
                "evaluable_rows": int(len(group)),
                "review_opportunities": int(group["review_opportunity_5d"].eq(True).sum()) if not group.empty else 0,
                "opportunity_rate": float(group["review_opportunity_5d"].eq(True).mean()) if len(group) else np.nan,
                "median_5d_return_pct": _median(group, "forward_5d_return_pct"),
                "median_mfe_5d_pct": _median(group, "mfe_5d_pct"),
                "median_mae_5d_pct": _median(group, "mae_5d_pct"),
            }
        )
    return pd.DataFrame(rows)


def risk_reward_summary(selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if selections.empty:
        return pd.DataFrame()
    for (variant, status), group_all in selections.groupby(
        ["variant", "ibd_entry_status"], sort=True, dropna=False
    ):
        group = group_all[group_all["forward_5d_censored"].eq(False)].copy()
        rows.append(
            {
                "variant": variant,
                "weekend_status": status,
                "picks": int(len(group_all)),
                "evaluable_picks": int(len(group)),
                "median_5d_return_pct": _median(group, "forward_5d_return_pct"),
                "median_mfe_5d_pct": _median(group, "mfe_5d_pct"),
                "median_mae_5d_pct": _median(group, "mae_5d_pct"),
                "p10_mae_5d_pct": _quantile(group, "mae_5d_pct", 0.10),
                "stop_8_within_5d_rate": _rate(group, "stop_8_within_5d"),
            }
        )
    return pd.DataFrame(rows)


def opportunity_label_projection(panel: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "snapshot_date",
        "code",
        "ibd_entry_status",
        "ibd_candidate_rule",
        "ibd_candidate_price",
        "current_vs_ibd_candidate_pct",
        "label_available",
        "forward_sessions",
        "review_opportunity_5d",
        "opportunity_type",
        "first_zone_date",
        "first_zone_close",
        "forward_5d_return_pct",
        "mfe_5d_pct",
        "mae_5d_pct",
        "stop_8_within_5d",
    ]
    return panel[[column for column in cols if column in panel.columns]].copy()


def render_data_audit(
    panel: pd.DataFrame,
    pools: list[tuple[str, pd.DataFrame, Path]],
) -> str:
    weeks = sorted(panel["snapshot_date"].astype(str).unique()) if not panel.empty else []
    complete = _complete_panel(panel)
    status_counts = (
        panel["ibd_entry_status"].fillna("UNKNOWN").astype(str).str.upper().value_counts().sort_index()
        if not panel.empty
        else pd.Series(dtype=int)
    )
    lines = [
        "# Next Week Review Selection - Data Audit",
        "",
        f"- replay pool directories read: {len(pools)}",
        f"- active-signal snapshot weeks: {len(weeks)}",
        f"- first active-signal snapshot: {weeks[0] if weeks else 'n/a'}",
        f"- last active-signal snapshot: {weeks[-1] if weeks else 'n/a'}",
        f"- active-signal events: {len(panel)}",
        f"- complete 5-session outcome rows: {len(complete)}",
        f"- complete 5-session coverage: {len(complete) / len(panel):.1%}" if len(panel) else "- complete 5-session coverage: n/a",
        "",
        "## Weekend status counts",
    ]
    for status, count in status_counts.items():
        lines.append(f"- {status}: {int(count)}")
    lines.extend(
        [
            "",
            "## Guardrails",
            "- Selection input is the frozen weekend pool plus verified PIT EPS only.",
            "- C Rank is not used by any research selector.",
            "- ATR and new technical indicators are not used.",
            "- Forward prices are labels only and never feed the weekend selector.",
            "",
        ]
    )
    return "\n".join(lines)


def render_manifest(
    *,
    panel: pd.DataFrame,
    pool_root: Path,
    price_cache: Path,
    eps_path: Path,
) -> str:
    rules = review_rules()
    weeks = sorted(panel["snapshot_date"].astype(str).unique()) if not panel.empty else []
    payload = {
        "study": "next_week_review_selection",
        "evaluation_status": "retrospective_pre_registered_replay",
        "pool_root": str(pool_root),
        "price_cache": str(price_cache),
        "price_cache_sha256": content_hash(price_cache) if price_cache.exists() else "",
        "eps_pit": str(eps_path),
        "eps_pit_sha256": content_hash(eps_path) if eps_path.exists() else "",
        "active_signal_weeks": len(weeks),
        "first_snapshot": weeks[0] if weeks else "",
        "last_snapshot": weeks[-1] if weeks else "",
        "baseline": "B0_ACTIONABLE_ONLY",
        "primary_rules": {
            name: {
                "near_below_pct": rule.near_below_pct,
                "extended_above_pct": rule.extended_above_pct,
                "min_support_count": rule.min_support_count,
            }
            for name, rule in rules.items()
        },
        "c_rank_used": False,
        "atr_used": False,
        "forward_sessions": 5,
    }
    # JSON is valid YAML 1.2; avoid adding a YAML dependency to the project.
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def render_report(
    panel: pd.DataFrame,
    weekly: pd.DataFrame,
    summary: pd.DataFrame,
    frontier: pd.DataFrame,
    blocked: pd.DataFrame,
    status: pd.DataFrame,
    risk_reward: pd.DataFrame,
) -> str:
    weeks = int(panel["snapshot_date"].astype(str).nunique()) if not panel.empty else 0
    signals = int(len(panel))
    available = int(panel["forward_5d_censored"].eq(False).sum()) if not panel.empty else 0
    return "\n".join(
        [
            "# Next Week Review Selection Research",
            "",
            "Status: retrospective_pre_registered_replay",
            "",
            "## Data",
            f"- snapshot weeks: {weeks}",
            f"- active-signal events: {signals}",
            f"- complete next-5-session labels: {available}",
            "",
            "## Baseline and primary variants",
            _markdown(summary),
            "",
            "## Attention frontier",
            _markdown(frontier),
            "",
            "## Weekend status opportunity conversion",
            _markdown(status),
            "",
            "## Risk / reward by selected status",
            _markdown(risk_reward),
            "",
            "## Discovery vs blocked validation",
            _markdown(blocked),
            "",
            "## Interpretation guardrails",
            "- B0 is ACTIONABLE-only Futu-review eligibility, not the Skill Top3.",
            "- C Rank is excluded from all selectors.",
            "- ATR and new technical indicators are excluded.",
            "- Review opportunity means current ACTIONABLE or a future 5-session close in the frozen Pivot +0% to +5% zone; it is not an automated buy signal.",
            "- Censored forward windows are excluded from capture and risk/reward denominators.",
            "- No production Skill/Futu/Dashboard change is authorized by this retrospective report alone.",
            "",
        ]
    )


def _variant_names(selections: pd.DataFrame) -> list[str]:
    if selections.empty:
        return [
            "B0_ACTIONABLE_ONLY",
            "R1_PATH",
            "R2_BALANCED",
            "R3_STRICT",
            "R2_BALANCED_ATTENTION_MATCHED",
            "R2_BALANCED_CAP10",
            "R2_BALANCED_CAP15",
            "R2_BALANCED_CAP20",
        ]
    return sorted(selections["variant"].dropna().astype(str).unique())


def _median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if len(values) else np.nan


def _quantile(frame: pd.DataFrame, column: str, q: float) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.quantile(q)) if len(values) else np.nan


def _rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = frame[column].dropna()
    return float(values.astype(bool).mean()) if len(values) else np.nan


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No data_"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_csv(index=False)


if __name__ == "__main__":
    main()
