from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from backtest.ibd_skill_replay.run_ytd_replay import _load_price_cache
from backtest.ibd_weekly_signal_oracle_eval import evaluate_weekly_signal_oracle as oracle
from backtest.ibd_weekly_signal_oracle_eval.price_cache import resolve_price_cache
from backtest.rd_agent_research_bench.backtrader_backtest import run_backtrader_variant_backtest
from backtest.rd_agent_research_bench.research import (
    intuitive_variant_summary,
    semiconductor_capture_audit,
    stop_loss_capital_backtest,
    summarize_variant_quality,
)


DEFAULT_OUTPUT_DIR = Path("backtest/rd_agent_research_bench/output")
EPS_MODE = "with_eps"
BASELINE_VARIANTS = [
    "skill_industry_eps_known",
    "signal_core_quality_eps_pass",
    "clean_eps_pass_no_dry_no_geom_caution",
]


@dataclass(frozen=True)
class SkepticalConfig:
    name: str
    cfg: dict[str, object]
    profile: dict[str, object]


def build_skeptical_configs() -> list[SkepticalConfig]:
    configs: list[SkepticalConfig] = []
    scopes = [
        ("act", False, False),
        ("sig", True, False),
        ("sigext", True, True),
    ]
    eps_gates = ["known", "pass25"]
    core_modes = [
        ("corestrict", False, False),
        ("coreentryloose", True, False),
        ("coreloose", True, True),
    ]
    geom_modes = [
        ("geomreject", False, False),
        ("geomallow", True, False),
        ("geomclean", False, True),
    ]
    freshness_options = [("freshstrict", False), ("freshallow", True)]
    buy_point_options = [("bpstrict", False), ("bpallow", True)]
    industry_options = [("ind", True), ("noind", False)]
    order_options = [("def", None), ("prox", "fresh_proximity")]
    topk_options = [1, 2, 3]

    for scope_label, allow_non_actionable, allow_extended in scopes:
        for eps_gate in eps_gates:
            for core_label, allow_entry_volume_gap, allow_without_volume_confirm in core_modes:
                for geom_label, allow_clear_geometry, exclude_geometry_caution in geom_modes:
                    for freshness_label, allow_freshness_missing in freshness_options:
                        for buy_point_label, allow_below_candidate_buy_point in buy_point_options:
                            for industry_label, industry_cover in industry_options:
                                for order_label, research_order in order_options:
                                    for topk in topk_options:
                                        name = "_".join(
                                            [
                                                "sk",
                                                scope_label,
                                                _eps_label(eps_gate),
                                                core_label,
                                                freshness_label,
                                                buy_point_label,
                                                geom_label,
                                                industry_label,
                                                order_label,
                                                f"top{topk}",
                                            ]
                                        )
                                        cfg: dict[str, object] = {
                                            "industry_cover": industry_cover,
                                            "max_picks": topk,
                                            "require_core_quality": True,
                                        }
                                        if allow_non_actionable:
                                            cfg["allow_non_actionable"] = True
                                        if allow_extended:
                                            cfg["allow_extended_from_buy_point"] = True
                                        if eps_gate == "known":
                                            cfg["require_eps_known"] = True
                                        elif eps_gate == "pass25":
                                            cfg["require_eps_pass"] = True
                                        else:
                                            raise ValueError(f"unsupported eps gate: {eps_gate}")
                                        if allow_entry_volume_gap:
                                            cfg["allow_entry_volume_missing"] = True
                                            cfg["allow_entry_volume_below_standard"] = True
                                        if allow_without_volume_confirm:
                                            cfg["allow_without_volume_confirm"] = True
                                        if allow_clear_geometry:
                                            cfg["allow_clear_geometry_failure"] = True
                                        if exclude_geometry_caution:
                                            cfg["exclude_geometry_caution"] = True
                                        if allow_freshness_missing:
                                            cfg["allow_freshness_missing"] = True
                                        if allow_below_candidate_buy_point:
                                            cfg["allow_below_candidate_buy_point"] = True
                                        if research_order:
                                            cfg["research_order"] = research_order
                                        profile = {
                                            "scope": scope_label,
                                            "signal_wide": allow_non_actionable,
                                            "allow_extended": allow_extended,
                                            "eps_gate": eps_gate,
                                            "core_mode": core_label,
                                            "allow_entry_volume_gap": allow_entry_volume_gap,
                                            "allow_without_volume_confirm": allow_without_volume_confirm,
                                            "geometry_mode": geom_label,
                                            "allow_clear_geometry_failure": allow_clear_geometry,
                                            "geometry_caution_hard_filter": exclude_geometry_caution,
                                            "freshness_mode": freshness_label,
                                            "allow_freshness_missing": allow_freshness_missing,
                                            "buy_point_mode": buy_point_label,
                                            "allow_below_candidate_buy_point": allow_below_candidate_buy_point,
                                            "industry_cover": industry_cover,
                                            "research_order": order_label,
                                            "proximity_reorder": research_order == "fresh_proximity",
                                            "max_picks": topk,
                                        }
                                        profile["rule_count"] = _rule_count(profile)
                                        configs.append(SkepticalConfig(name=name, cfg=cfg, profile=profile))
    return configs


def run_skeptical_search(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    price_cache: str | Path | None = None,
    initial_capital: float = 10000.0,
    stop_loss_pct: float = 8.0,
    backtrader_limit: int = 10000,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = build_skeptical_configs()
    variant_map = {name: dict(oracle.VARIANTS[name]) for name in BASELINE_VARIANTS}
    variant_map.update({config.name: config.cfg for config in configs})
    profiles = pd.DataFrame(
        [_baseline_profile(name) for name in BASELINE_VARIANTS]
        + [
            {
                "eps_mode": EPS_MODE,
                "variant": config.name,
                **config.profile,
            }
            for config in configs
        ]
    )

    with _temporary_variants(variant_map):
        universe, picks, weekly, _ = oracle.evaluate(True, price_cache=price_cache)

    quality_rows = [
        summarize_variant_quality(EPS_MODE, variant, universe, weekly, picks)
        for variant in variant_map
    ]
    abstain_rows = [
        abstain_quality_summary(
            weekly,
            eps_mode=EPS_MODE,
            baseline_variant="skill_industry_eps_known",
            variant=variant,
        )
        for variant in variant_map
    ]
    intuitive_rows = [
        intuitive_variant_summary(weekly, eps_mode=EPS_MODE, variant=variant)
        for variant in variant_map
    ]
    proxy_rows = [
        stop_loss_capital_backtest(
            picks,
            eps_mode=EPS_MODE,
            variant=variant,
            initial_capital=initial_capital,
            stop_loss_pct=-float(stop_loss_pct),
        )
        for variant in variant_map
    ]
    quality = pd.DataFrame(quality_rows)
    abstain = pd.DataFrame(abstain_rows)
    intuitive = pd.DataFrame(intuitive_rows)
    proxy = pd.DataFrame(proxy_rows)
    pick_profile = pick_composition(picks)
    summary = profiles.merge(quality, on=["eps_mode", "variant"], how="left")
    summary = summary.merge(abstain, on=["eps_mode", "variant"], how="left", suffixes=("", "_abstain"))
    summary = summary.merge(intuitive, on=["eps_mode", "variant"], how="left", suffixes=("", "_intuitive"))
    summary = summary.merge(
        proxy[["eps_mode", "variant", "final_equity", "total_return_pct"]].rename(
            columns={"final_equity": "final_equity_proxy", "total_return_pct": "total_return_pct_proxy"}
        ),
        on=["eps_mode", "variant"],
        how="left",
    )
    summary = summary.merge(pick_profile, on=["eps_mode", "variant"], how="left")
    for column in [
        "non_actionable_pick_rate",
        "extended_pick_rate",
        "clear_geometry_pick_rate",
        "freshness_missing_pick_rate",
        "below_candidate_buy_point_pick_rate",
    ]:
        summary[column] = pd.to_numeric(summary.get(column), errors="coerce").fillna(0.0)

    finalists = choose_backtrader_finalists(
        summary,
        named_variants=BASELINE_VARIANTS,
        limit=backtrader_limit,
    )
    prices = _load_price_cache(resolve_price_cache(price_cache))
    bt_rows = []
    trade_rows = []
    for variant in finalists:
        bt_summary, bt_trades, _ = run_backtrader_variant_backtest(
            picks,
            prices,
            eps_mode=EPS_MODE,
            variant=variant,
            initial_capital=initial_capital,
            stop_loss_pct=stop_loss_pct,
        )
        bt_rows.append(bt_summary)
        if not bt_trades.empty:
            bt_trades.insert(0, "variant", variant)
            bt_trades.insert(0, "eps_mode", EPS_MODE)
            trade_rows.extend(bt_trades.to_dict("records"))
    backtrader = pd.DataFrame(bt_rows)
    trades = pd.DataFrame(trade_rows)
    final_summary = summary.merge(backtrader, on=["eps_mode", "variant"], how="left", suffixes=("", "_bt"))
    decisions = skeptical_decision_matrix(final_summary)
    semiconductor = semiconductor_capture_audit(trades)
    report = render_skeptical_report(final_summary, decisions, semiconductor, finalists)

    outputs = {
        "profiles": output_dir / "skeptical_condition_profiles.csv",
        "quality": output_dir / "skeptical_condition_quality.csv",
        "abstain": output_dir / "skeptical_condition_abstain_quality.csv",
        "picks": output_dir / "skeptical_condition_picks.csv",
        "weekly": output_dir / "skeptical_condition_weekly.csv",
        "summary": output_dir / "skeptical_condition_summary.csv",
        "decisions": output_dir / "skeptical_condition_decisions.csv",
        "backtrader": output_dir / "skeptical_condition_backtrader_summary.csv",
        "trades": output_dir / "skeptical_condition_backtrader_trades.csv",
        "semiconductor": output_dir / "skeptical_condition_semiconductor_capture.csv",
        "report": output_dir / "skeptical_condition_report.md",
        "manifest": output_dir / "skeptical_condition_manifest.json",
    }
    profiles.to_csv(outputs["profiles"], index=False)
    quality.to_csv(outputs["quality"], index=False)
    abstain.to_csv(outputs["abstain"], index=False)
    picks.to_csv(outputs["picks"], index=False)
    weekly.to_csv(outputs["weekly"], index=False)
    final_summary.to_csv(outputs["summary"], index=False)
    decisions.to_csv(outputs["decisions"], index=False)
    backtrader.to_csv(outputs["backtrader"], index=False)
    trades.to_csv(outputs["trades"], index=False)
    semiconductor.to_csv(outputs["semiconductor"], index=False)
    outputs["report"].write_text(report, encoding="utf-8")
    outputs["manifest"].write_text(
        json.dumps(
            {
                "backend": (
                    "skeptical_condition_generator_plus_weekly_proxy_plus_backtrader_all"
                    if len(finalists) >= len(variant_map)
                    else "skeptical_condition_generator_plus_weekly_proxy_plus_backtrader_finalists"
                ),
                "eps_mode": EPS_MODE,
                "oracle_end_date": oracle.END_DATE,
                "initial_capital": initial_capital,
                "stop_loss_pct": stop_loss_pct,
                "config_count": len(configs),
                "variant_count": len(variant_map),
                "backtrader_finalist_count": len(finalists),
                "axes": [
                    "scope/actionability",
                    "extended",
                    "eps_gate",
                    "core_entry_volume_and_volume_confirm",
                    "freshness_missing",
                    "below_candidate_buy_point",
                    "clear_geometry_failure",
                    "geometry_caution",
                    "industry_cover",
                    "order",
                    "max_picks",
                ],
                "non_coupling_boundary": [
                    "no ticker-specific rule",
                    "no date-specific rule",
                    "no realized-return threshold inside selector",
                    "realized returns only rank completed experiments",
                ],
                "outputs": {name: str(path) for name, path in outputs.items() if name != "manifest"},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {name: str(path) for name, path in outputs.items()}


def abstain_quality_summary(
    weekly: pd.DataFrame,
    *,
    eps_mode: str,
    baseline_variant: str,
    variant: str,
) -> dict[str, object]:
    scoped = weekly[weekly["eps_mode"].astype(str).eq(eps_mode)].copy()
    baseline = scoped[scoped["variant"].astype(str).eq(baseline_variant)].copy()
    candidate = scoped[scoped["variant"].astype(str).eq(variant)].copy()
    baseline["avg_latest_return_pct"] = pd.to_numeric(baseline.get("avg_latest_return_pct"), errors="coerce")
    baseline["stop_8pct_count"] = pd.to_numeric(baseline.get("stop_8pct_count"), errors="coerce").fillna(0)
    candidate["avg_latest_return_pct"] = pd.to_numeric(candidate.get("avg_latest_return_pct"), errors="coerce")
    candidate["stop_8pct_count"] = pd.to_numeric(candidate.get("stop_8pct_count"), errors="coerce").fillna(0)
    baseline_weeks = set(baseline["snapshot_date"].astype(str))
    candidate_weeks = set(candidate["snapshot_date"].astype(str))
    overlap = sorted(baseline_weeks & candidate_weeks)
    missed = sorted(baseline_weeks - candidate_weeks)
    baseline_overlap = baseline[baseline["snapshot_date"].astype(str).isin(overlap)]
    candidate_overlap = candidate[candidate["snapshot_date"].astype(str).isin(overlap)]
    overlap_joined = candidate_overlap[["snapshot_date", "avg_latest_return_pct"]].merge(
        baseline_overlap[["snapshot_date", "avg_latest_return_pct"]],
        on="snapshot_date",
        how="inner",
        suffixes=("_candidate", "_baseline"),
    )
    missed_baseline = baseline[baseline["snapshot_date"].astype(str).isin(missed)]
    return {
        "eps_mode": eps_mode,
        "variant": variant,
        "baseline_variant": baseline_variant,
        "baseline_weeks": len(baseline_weeks),
        "candidate_weeks": len(candidate_weeks),
        "overlap_weeks": len(overlap),
        "missed_baseline_weeks": len(missed),
        "missed_baseline_avg_return_pct": _float(missed_baseline["avg_latest_return_pct"].mean()),
        "missed_baseline_median_return_pct": _float(missed_baseline["avg_latest_return_pct"].median()),
        "missed_baseline_stop_weeks": int(missed_baseline["stop_8pct_count"].gt(0).sum()),
        "overlap_avg_return_delta_pct": _float(
            (overlap_joined["avg_latest_return_pct_candidate"] - overlap_joined["avg_latest_return_pct_baseline"]).mean()
        ),
        "overlap_median_return_delta_pct": _float(
            (overlap_joined["avg_latest_return_pct_candidate"] - overlap_joined["avg_latest_return_pct_baseline"]).median()
        ),
    }


def pick_composition(picks: pd.DataFrame) -> pd.DataFrame:
    if picks.empty:
        return pd.DataFrame(columns=["eps_mode", "variant"])
    frame = picks.copy()
    frame["non_actionable_pick"] = ~frame["entry_status"].astype(str).eq("ACTIONABLE")
    frame["extended_pick"] = frame["entry_status"].astype(str).eq("EXTENDED")
    risk_codes = frame["risk_codes"].fillna("").astype(str)
    frame["clear_geometry_pick"] = risk_codes.str.contains("clear_geometry_failure", regex=False)
    frame["freshness_missing_pick"] = risk_codes.str.contains("freshness_missing", regex=False)
    frame["below_candidate_buy_point_pick"] = risk_codes.str.contains("below_candidate_buy_point", regex=False)
    grouped = frame.groupby(["eps_mode", "variant"], sort=True)
    return grouped.agg(
        non_actionable_picks=("non_actionable_pick", "sum"),
        extended_picks=("extended_pick", "sum"),
        clear_geometry_picks=("clear_geometry_pick", "sum"),
        freshness_missing_picks=("freshness_missing_pick", "sum"),
        below_candidate_buy_point_picks=("below_candidate_buy_point_pick", "sum"),
        total_picks=("code", "size"),
    ).reset_index().assign(
        non_actionable_pick_rate=lambda df: df["non_actionable_picks"] / df["total_picks"],
        extended_pick_rate=lambda df: df["extended_picks"] / df["total_picks"],
        clear_geometry_pick_rate=lambda df: df["clear_geometry_picks"] / df["total_picks"],
        freshness_missing_pick_rate=lambda df: df["freshness_missing_picks"] / df["total_picks"],
        below_candidate_buy_point_pick_rate=lambda df: df["below_candidate_buy_point_picks"] / df["total_picks"],
    )


def choose_backtrader_finalists(
    summary: pd.DataFrame,
    *,
    named_variants: list[str],
    limit: int,
) -> list[str]:
    summary = summary.copy()
    for column in [
        "non_actionable_pick_rate",
        "extended_pick_rate",
        "clear_geometry_pick_rate",
        "freshness_missing_pick_rate",
        "below_candidate_buy_point_pick_rate",
    ]:
        if column not in summary:
            summary[column] = 0.0
    all_variants = summary["variant"].astype(str).drop_duplicates().tolist()
    if limit >= len(all_variants):
        return all_variants
    variants: list[str] = []

    def add(names: list[str]) -> None:
        for name in names:
            if name not in variants:
                variants.append(name)

    add(named_variants)
    add(_top_names(summary[summary["non_actionable_pick_rate"].gt(0)], "final_equity_proxy", max(1, limit // 6), ascending=False))
    add(_top_names(summary[summary["extended_pick_rate"].gt(0)], "final_equity_proxy", max(1, limit // 8), ascending=False))
    add(_top_names(summary[summary["clear_geometry_pick_rate"].gt(0)], "final_equity_proxy", max(1, limit // 8), ascending=False))
    add(_top_names(summary[summary["freshness_missing_pick_rate"].gt(0)], "final_equity_proxy", max(1, limit // 10), ascending=False))
    add(_top_names(summary[summary["below_candidate_buy_point_pick_rate"].gt(0)], "final_equity_proxy", max(1, limit // 10), ascending=False))
    add(_top_names(summary, "final_equity_proxy", max(1, limit // 4), ascending=False))
    add(_top_names(summary, "median_week_avg_latest_return_pct", max(1, limit // 6), ascending=False))
    add(_top_names(summary, "min_week_avg_latest_return_pct", max(1, limit // 6), ascending=False))
    add(_top_names(summary, "pick_stop_rate", max(1, limit // 6), ascending=True))
    return variants[:limit]


def skeptical_decision_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[summary["variant"].astype(str).eq("skill_industry_eps_known")]
    if baseline.empty:
        return pd.DataFrame()
    base = baseline.iloc[0]
    rows = []
    bt_frame = summary[pd.to_numeric(summary.get("final_value"), errors="coerce").notna()].copy()
    for _, row in bt_frame.iterrows():
        variant = str(row["variant"])
        if variant == "skill_industry_eps_known":
            status = "baseline"
        else:
            return_ok = _num(row.get("final_value")) > _num(base.get("final_value"))
            drawdown_ok = _num(row.get("max_drawdown_pct")) >= _num(base.get("max_drawdown_pct"))
            stop_ok = _num(row.get("stop_events")) <= _num(base.get("stop_events"))
            median_ok = _num(row.get("median_week_avg_latest_return_pct")) >= _num(base.get("median_week_avg_latest_return_pct"))
            floor_ok = _num(row.get("min_week_avg_latest_return_pct")) >= _num(base.get("min_week_avg_latest_return_pct"))
            bottom_ok = _num(row.get("pick_bottom5_precision_bad")) <= _num(base.get("pick_bottom5_precision_bad"))
            pick_stop_ok = _num(row.get("pick_stop_rate")) <= _num(base.get("pick_stop_rate"))
            abstain_ok = (
                pd.isna(row.get("missed_baseline_avg_return_pct"))
                or _num(row.get("missed_baseline_avg_return_pct")) <= _num(base.get("median_week_avg_latest_return_pct"))
            )
            if return_ok and drawdown_ok and stop_ok and median_ok and floor_ok and bottom_ok and pick_stop_ok and abstain_ok:
                status = "robust_replacement"
            elif return_ok and drawdown_ok and stop_ok:
                status = "portfolio_tradeoff"
            elif return_ok:
                status = "return_only"
            else:
                status = "reject"
        rows.append(
            {
                "variant": variant,
                "skeptical_status": status,
                "final_value": row.get("final_value"),
                "total_return_pct": row.get("total_return_pct"),
                "max_drawdown_pct": row.get("max_drawdown_pct"),
                "stop_events": row.get("stop_events"),
                "input_picks": row.get("input_picks"),
                "rebalance_events": row.get("rebalance_events"),
                "median_week_avg_latest_return_pct": row.get("median_week_avg_latest_return_pct"),
                "min_week_avg_latest_return_pct": row.get("min_week_avg_latest_return_pct"),
                "pick_bottom5_precision_bad": row.get("pick_bottom5_precision_bad"),
                "pick_stop_rate": row.get("pick_stop_rate"),
                "missed_baseline_weeks": row.get("missed_baseline_weeks"),
                "missed_baseline_avg_return_pct": row.get("missed_baseline_avg_return_pct"),
                "overlap_avg_return_delta_pct": row.get("overlap_avg_return_delta_pct"),
                "non_actionable_pick_rate": row.get("non_actionable_pick_rate"),
                "extended_pick_rate": row.get("extended_pick_rate"),
                "clear_geometry_pick_rate": row.get("clear_geometry_pick_rate"),
                "freshness_missing_pick_rate": row.get("freshness_missing_pick_rate"),
                "below_candidate_buy_point_pick_rate": row.get("below_candidate_buy_point_pick_rate"),
                "rule_count": row.get("rule_count"),
                "scope": row.get("scope"),
                "eps_gate": row.get("eps_gate"),
                "core_mode": row.get("core_mode"),
                "geometry_mode": row.get("geometry_mode"),
                "freshness_mode": row.get("freshness_mode"),
                "buy_point_mode": row.get("buy_point_mode"),
                "industry_cover": row.get("industry_cover"),
                "research_order": row.get("research_order"),
                "max_picks": row.get("max_picks"),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["skeptical_status", "final_value", "max_drawdown_pct"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def render_skeptical_report(
    summary: pd.DataFrame,
    decisions: pd.DataFrame,
    semiconductor: pd.DataFrame,
    finalists: list[str],
) -> str:
    generated_count = summary["variant"].astype(str).nunique() if "variant" in summary else 0
    backtested_count = decisions["variant"].astype(str).nunique() if "variant" in decisions else 0
    if generated_count and backtested_count >= generated_count and len(finalists) >= generated_count:
        backtrader_scope = f"Backtrader is run on all {generated_count} variants: generated skeptical configs plus baselines."
    else:
        backtrader_scope = (
            f"Backtrader is run on {backtested_count} deterministic finalists selected from "
            f"{generated_count} generated schemes across multiple quality fronts."
        )
    lines = [
        "# Skeptical Condition Search Audit",
        "",
        "## Boundary",
        "",
        "- This run challenges ACTIONABLE-only, extended exclusion, clear-geometry rejection, geometry-caution filtering, freshness requirements, below-buy-point rejection, entry-volume requirements, volume-confirmation requirements, industry cover, ordering, EPS gate, and max recommendation slots.",
        "- Selectors still use only pool fields and deterministic ranks. Realized returns are used only after replay for evaluation and finalist selection.",
        "- All generated schemes are evaluated weekly/proxy. " + backtrader_scope,
        "",
        "## Conclusion",
        "",
    ]
    lines.extend(_conclusion_lines(summary, decisions))
    lines.extend(["", "## Backtrader Decisions", ""])
    show_cols = [
        "variant",
        "skeptical_status",
        "final_value",
        "total_return_pct",
        "max_drawdown_pct",
        "stop_events",
        "input_picks",
        "rebalance_events",
        "median_week_avg_latest_return_pct",
        "min_week_avg_latest_return_pct",
        "pick_bottom5_precision_bad",
        "pick_stop_rate",
        "missed_baseline_weeks",
        "missed_baseline_avg_return_pct",
        "non_actionable_pick_rate",
        "extended_pick_rate",
        "clear_geometry_pick_rate",
        "freshness_missing_pick_rate",
        "below_candidate_buy_point_pick_rate",
        "scope",
        "eps_gate",
        "core_mode",
        "geometry_mode",
        "freshness_mode",
        "buy_point_mode",
        "industry_cover",
        "research_order",
        "max_picks",
    ]
    lines.extend(_markdown_table(_round_frame(decisions[[col for col in show_cols if col in decisions.columns]].head(80))).splitlines())
    lines.extend(["", "## Weekly/Proxy Best", ""])
    proxy_best = summary.sort_values("final_equity_proxy", ascending=False).head(40)
    proxy_cols = [
        "variant",
        "final_equity_proxy",
        "total_return_pct_proxy",
        "median_week_avg_latest_return_pct",
        "min_week_avg_latest_return_pct",
        "pick_bottom5_precision_bad",
        "pick_stop_rate",
        "missed_baseline_weeks",
        "missed_baseline_avg_return_pct",
        "non_actionable_pick_rate",
        "extended_pick_rate",
        "clear_geometry_pick_rate",
        "freshness_missing_pick_rate",
        "below_candidate_buy_point_pick_rate",
        "scope",
        "eps_gate",
        "core_mode",
        "geometry_mode",
        "freshness_mode",
        "buy_point_mode",
        "industry_cover",
        "research_order",
        "max_picks",
    ]
    lines.extend(_markdown_table(_round_frame(proxy_best[[col for col in proxy_cols if col in proxy_best.columns]])).splitlines())
    lines.extend(["", "## Non-Actionable Front", ""])
    non_action = summary[summary["non_actionable_pick_rate"].gt(0)].sort_values("final_equity_proxy", ascending=False).head(30)
    lines.extend(_markdown_table(_round_frame(non_action[[col for col in proxy_cols if col in non_action.columns]])).splitlines())
    lines.extend(["", "## Semiconductor Capture", ""])
    if semiconductor.empty:
        lines.append("_empty_")
    else:
        lines.extend(_markdown_table(_round_frame(semiconductor.head(30))).splitlines())
    lines.extend(["", "## Backtrader Finalists", ""])
    lines.append(", ".join(finalists))
    return "\n".join(lines) + "\n"


def _conclusion_lines(summary: pd.DataFrame, decisions: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    baseline = decisions[decisions["variant"].astype(str).eq("skill_industry_eps_known")]
    robust = decisions[decisions["skeptical_status"].astype(str).eq("robust_replacement")]
    if baseline.empty:
        return ["- Baseline missing from decisions; cannot conclude."]
    base = baseline.iloc[0]
    if robust.empty:
        lines.append("- No robust replacement beat the baseline across Backtrader return, drawdown, stops, weekly median, weekly floor, Bottom5 exposure, pick stop rate, and abstain quality.")
    else:
        best = robust.sort_values("final_value", ascending=False).iloc[0]
        lines.append(f"- Robust replacement found: `{best['variant']}` final {float(best['final_value']):.2f}.")
    best_bt = decisions.sort_values("final_value", ascending=False).iloc[0]
    lines.append(
        f"- Baseline `skill_industry_eps_known`: final {float(base['final_value']):.2f}, return {float(base['total_return_pct']):.2f}%, max DD {float(base['max_drawdown_pct']):.2f}%, stops {int(base['stop_events'])}."
    )
    lines.append(
        f"- Best Backtrader scheme `{best_bt['variant']}`: final {float(best_bt['final_value']):.2f}, return {float(best_bt['total_return_pct']):.2f}%, max DD {float(best_bt['max_drawdown_pct']):.2f}%, stops {int(best_bt['stop_events'])}, status {best_bt['skeptical_status']}."
    )
    lines.extend(
        [
            _best_backtrader_challenge_line(
                decisions,
                column="non_actionable_pick_rate",
                label="non-ACTIONABLE",
            ),
            _best_backtrader_challenge_line(
                decisions,
                column="extended_pick_rate",
                label="extended",
            ),
            _best_backtrader_challenge_line(
                decisions,
                column="clear_geometry_pick_rate",
                label="clear-geometry-failure",
            ),
            _best_backtrader_challenge_line(
                decisions,
                column="freshness_missing_pick_rate",
                label="freshness-missing",
            ),
            _best_backtrader_challenge_line(
                decisions,
                column="below_candidate_buy_point_pick_rate",
                label="below-candidate-buy-point",
            ),
        ]
    )
    proxy_non_action = summary[summary["non_actionable_pick_rate"].gt(0)].sort_values("final_equity_proxy", ascending=False)
    if not proxy_non_action.empty:
        row = proxy_non_action.iloc[0]
        lines.append(
            f"- Proxy-only note: best non-ACTIONABLE proxy scheme `{row['variant']}` reached proxy final {float(row['final_equity_proxy']):.2f}, but replacement decisions above use Backtrader only."
        )
    return lines


def _best_backtrader_challenge_line(decisions: pd.DataFrame, *, column: str, label: str) -> str:
    if column not in decisions:
        return f"- {label} was not measured in Backtrader decisions."
    scoped = decisions[pd.to_numeric(decisions[column], errors="coerce").fillna(0).gt(0)].copy()
    if scoped.empty:
        return f"- {label} was allowed in the search, but no Backtradered top picks used it."
    row = scoped.sort_values("final_value", ascending=False).iloc[0]
    return (
        f"- {label} entered the experiment: best Backtrader {label} scheme `{row['variant']}` "
        f"final {float(row['final_value']):.2f}, return {float(row['total_return_pct']):.2f}%, "
        f"max DD {float(row['max_drawdown_pct']):.2f}%, stops {int(row['stop_events'])}, "
        f"pick rate {float(row[column]):.2%}, status {row['skeptical_status']}."
    )


@contextmanager
def _temporary_variants(variant_map: dict[str, dict[str, object]]):
    old_variants = dict(oracle.VARIANTS)
    oracle.VARIANTS.clear()
    oracle.VARIANTS.update(variant_map)
    try:
        yield
    finally:
        oracle.VARIANTS.clear()
        oracle.VARIANTS.update(old_variants)


def _baseline_profile(name: str) -> dict[str, object]:
    cfg = oracle.VARIANTS[name]
    return {
        "eps_mode": EPS_MODE,
        "variant": name,
        "scope": "sig" if cfg.get("allow_non_actionable") else "act",
        "signal_wide": bool(cfg.get("allow_non_actionable")),
        "allow_extended": bool(cfg.get("allow_extended_from_buy_point")),
        "eps_gate": "pass25" if cfg.get("require_eps_pass") else ("known" if cfg.get("require_eps_known") else "none"),
        "core_mode": "corestrict" if cfg.get("require_core_quality") else "none",
        "allow_entry_volume_gap": bool(cfg.get("allow_entry_volume_missing") or cfg.get("allow_entry_volume_below_standard")),
        "allow_without_volume_confirm": bool(cfg.get("allow_without_volume_confirm")),
        "geometry_mode": "geomclean" if cfg.get("exclude_geometry_caution") else "geomreject",
        "allow_clear_geometry_failure": bool(cfg.get("allow_clear_geometry_failure")),
        "geometry_caution_hard_filter": bool(cfg.get("exclude_geometry_caution")),
        "freshness_mode": "freshallow" if cfg.get("allow_freshness_missing") else "freshstrict",
        "allow_freshness_missing": bool(cfg.get("allow_freshness_missing")),
        "buy_point_mode": "bpallow" if cfg.get("allow_below_candidate_buy_point") else "bpstrict",
        "allow_below_candidate_buy_point": bool(cfg.get("allow_below_candidate_buy_point")),
        "industry_cover": bool(cfg.get("industry_cover")),
        "research_order": "prox" if cfg.get("research_order") == "fresh_proximity" else "def",
        "proximity_reorder": cfg.get("research_order") == "fresh_proximity",
        "max_picks": int(cfg.get("max_picks", 3)),
        "rule_count": 0,
    }


def _rule_count(profile: dict[str, object]) -> int:
    return int(profile["eps_gate"] != "none") + sum(
        bool(profile.get(key))
        for key in [
            "signal_wide",
            "allow_extended",
            "allow_entry_volume_gap",
            "allow_without_volume_confirm",
            "allow_clear_geometry_failure",
            "geometry_caution_hard_filter",
            "allow_freshness_missing",
            "allow_below_candidate_buy_point",
            "industry_cover",
            "proximity_reorder",
        ]
    ) + int(profile.get("max_picks") != 3)


def _eps_label(eps_gate: str) -> str:
    if eps_gate == "known":
        return "epsknown"
    if eps_gate == "pass25":
        return "epspass"
    raise ValueError(f"unsupported eps gate: {eps_gate}")


def _top_names(frame: pd.DataFrame, column: str, limit: int, *, ascending: bool) -> list[str]:
    if limit <= 0 or frame.empty or column not in frame:
        return []
    scoped = frame[pd.to_numeric(frame[column], errors="coerce").notna()].copy()
    if scoped.empty:
        return []
    return scoped.sort_values(column, ascending=ascending)["variant"].astype(str).head(limit).tolist()


def _float(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def _num(value: object) -> float:
    result = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(result) if pd.notna(result) else float("nan")


def _round_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=["float"]).columns:
        result[column] = result[column].round(6)
    return result


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    return frame.to_markdown(index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run skeptical condition search for IBD skill candidates.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--price-cache", default="")
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--stop-loss-pct", type=float, default=8.0)
    parser.add_argument("--backtrader-limit", type=int, default=10000)
    args = parser.parse_args(argv)
    outputs = run_skeptical_search(
        output_dir=Path(args.output_dir),
        price_cache=args.price_cache or None,
        initial_capital=args.initial_capital,
        stop_loss_pct=args.stop_loss_pct,
        backtrader_limit=args.backtrader_limit,
    )
    print(json.dumps(outputs, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
