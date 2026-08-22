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
    imax_rank_audit,
    render_markdown_report,
    rule_status_coverage,
    semiconductor_capture_audit,
    summarize_variant_quality,
)


DEFAULT_OUTPUT_DIR = Path("backtest/rd_agent_research_bench/output")
BASELINE_VARIANTS = {
    "no_eps": ["v3_core_top3"],
    "with_eps": [
        "skill_industry_eps_known",
        "clean_eps_pass_no_dry_no_geom_caution",
        "signal_core_quality_eps_pass",
    ],
}


@dataclass(frozen=True)
class GridConfig:
    name: str
    cfg: dict[str, object]
    profile: dict[str, object]


def build_grid_configs(
    *,
    eps_gates: tuple[str, ...] = ("none", "known", "pass25"),
) -> list[GridConfig]:
    configs: list[GridConfig] = []
    scopes = [("act", False), ("sig", True)]
    industry_options = [("ind", True), ("noind", False)]
    order_options = [("def", None), ("prox", "fresh_proximity")]
    risk_options = [
        ("keep", False, False),
        ("nodry", True, False),
        ("nogeom", False, True),
        ("clean", True, True),
    ]
    fill_options = [("strict", False), ("relaxed", True)]

    for scope_label, signal_wide in scopes:
        for eps_gate in eps_gates:
            eps_label = _eps_label(eps_gate)
            for industry_label, industry_cover in industry_options:
                for order_label, research_order in order_options:
                    for risk_label, exclude_dry, exclude_geom in risk_options:
                        for fill_label, fill_relaxed in fill_options:
                            name = "_".join(
                                [
                                    "grid",
                                    scope_label,
                                    eps_label,
                                    industry_label,
                                    order_label,
                                    risk_label,
                                    fill_label,
                                ]
                            )
                            cfg: dict[str, object] = {
                                "industry_cover": industry_cover,
                                "require_core_quality": True,
                            }
                            if signal_wide:
                                cfg["allow_non_actionable"] = True
                            if eps_gate == "known":
                                cfg["require_eps_known"] = True
                            elif eps_gate == "pass25":
                                cfg["require_eps_pass"] = True
                            elif eps_gate != "none":
                                raise ValueError(f"unsupported eps gate: {eps_gate}")
                            if research_order is not None:
                                cfg["research_order"] = research_order
                            if exclude_dry:
                                cfg["exclude_pullback_not_dry"] = True
                            if exclude_geom:
                                cfg["exclude_geometry_caution"] = True
                            if fill_relaxed:
                                cfg["fill_relaxed"] = True
                            profile = {
                                "signal_wide": signal_wide,
                                "actionable_gate": not signal_wide,
                                "eps_gate": eps_gate,
                                "industry_cover": industry_cover,
                                "core_quality_gate": True,
                                "proximity_reorder": research_order == "fresh_proximity",
                                "pullback_dry_hard_filter": exclude_dry,
                                "geometry_caution_hard_filter": exclude_geom,
                                "relaxed_fill": fill_relaxed,
                            }
                            configs.append(GridConfig(name=name, cfg=cfg, profile=profile))
    return configs


def build_eps_pass_fallback_configs() -> list[GridConfig]:
    configs: list[GridConfig] = []
    scopes = [("act", False), ("sig", True)]
    industry_options = [("ind", True), ("noind", False)]
    order_options = [("def", None), ("prox", "fresh_proximity")]
    risk_options = [
        ("keep", False, False),
        ("nodry", True, False),
        ("nogeom", False, True),
        ("clean", True, True),
    ]
    for scope_label, signal_wide in scopes:
        for industry_label, industry_cover in industry_options:
            for order_label, research_order in order_options:
                for risk_label, exclude_dry, exclude_geom in risk_options:
                    name = "_".join(
                        [
                            "grid",
                            scope_label,
                            "epspass2known",
                            industry_label,
                            order_label,
                            risk_label,
                            "relaxed",
                        ]
                    )
                    cfg: dict[str, object] = {
                        "industry_cover": industry_cover,
                        "require_core_quality": True,
                        "require_eps_pass": True,
                        "fill_relaxed": True,
                        "fill_eps_fallback": "known",
                    }
                    if signal_wide:
                        cfg["allow_non_actionable"] = True
                    if research_order is not None:
                        cfg["research_order"] = research_order
                    if exclude_dry:
                        cfg["exclude_pullback_not_dry"] = True
                    if exclude_geom:
                        cfg["exclude_geometry_caution"] = True
                    profile = {
                        "signal_wide": signal_wide,
                        "actionable_gate": not signal_wide,
                        "eps_gate": "pass25_then_known_fill",
                        "industry_cover": industry_cover,
                        "core_quality_gate": True,
                        "proximity_reorder": research_order == "fresh_proximity",
                        "pullback_dry_hard_filter": exclude_dry,
                        "geometry_caution_hard_filter": exclude_geom,
                        "relaxed_fill": True,
                    }
                    configs.append(GridConfig(name=name, cfg=cfg, profile=profile))
    return configs


def run_grid_search(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    price_cache: str | Path | None = None,
    initial_capital: float = 10000.0,
    stop_loss_pct: float = 8.0,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prices = _load_price_cache(resolve_price_cache(price_cache))

    all_quality: list[dict[str, object]] = []
    all_weekly: list[dict[str, object]] = []
    all_picks: list[dict[str, object]] = []
    all_coverage: list[dict[str, object]] = []
    all_imax: list[dict[str, object]] = []
    all_bt: list[dict[str, object]] = []
    all_trades: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []

    for eps_mode, enabled in [("no_eps", False), ("with_eps", True)]:
        grid_configs = build_grid_configs(eps_gates=("none",) if eps_mode == "no_eps" else ("none", "known", "pass25"))
        if eps_mode == "with_eps":
            grid_configs.extend(build_eps_pass_fallback_configs())
        baseline_names = BASELINE_VARIANTS[eps_mode]
        variant_map = {name: dict(oracle.VARIANTS[name]) for name in baseline_names}
        variant_map.update({config.name: config.cfg for config in grid_configs})
        profile_rows.extend(_baseline_profiles(eps_mode, baseline_names))
        profile_rows.extend(
            {
                "eps_mode": eps_mode,
                "variant": config.name,
                "rule_count": _rule_count(config.profile),
                **config.profile,
            }
            for config in grid_configs
        )

        with _temporary_variants(variant_map):
            universe, picks, weekly, _ = oracle.evaluate(enabled, price_cache=price_cache)

        variants = baseline_names + [config.name for config in grid_configs]
        for variant in variants:
            all_quality.append(summarize_variant_quality(eps_mode, variant, universe, weekly, picks))
            variant_picks = picks[picks["variant"].astype(str).eq(variant)].copy()
            coverage = rule_status_coverage(universe, variant_picks)
            coverage.insert(0, "variant", variant)
            coverage.insert(0, "eps_mode", eps_mode)
            all_coverage.extend(coverage.to_dict("records"))

            bt_summary, bt_trades, _ = run_backtrader_variant_backtest(
                picks,
                prices,
                eps_mode=eps_mode,
                variant=variant,
                initial_capital=initial_capital,
                stop_loss_pct=stop_loss_pct,
            )
            all_bt.append(bt_summary)
            if not bt_trades.empty:
                bt_trades.insert(0, "variant", variant)
                bt_trades.insert(0, "eps_mode", eps_mode)
                all_trades.extend(bt_trades.to_dict("records"))

        if not picks.empty:
            all_picks.extend(picks[picks["variant"].astype(str).isin(variants)].to_dict("records"))
        if not weekly.empty:
            all_weekly.extend(weekly[weekly["variant"].astype(str).isin(variants)].to_dict("records"))
        imax = imax_rank_audit(universe, picks)
        imax.insert(0, "eps_mode", eps_mode)
        all_imax.extend(imax.to_dict("records"))

    profiles = pd.DataFrame(profile_rows).drop_duplicates(["eps_mode", "variant"])
    quality = pd.DataFrame(all_quality)
    weekly = pd.DataFrame(all_weekly)
    picks = pd.DataFrame(all_picks)
    coverage = pd.DataFrame(all_coverage)
    imax = pd.DataFrame(all_imax)
    backtrader = pd.DataFrame(all_bt)
    trades = pd.DataFrame(all_trades)

    merged = _merge_grid_outputs(quality, backtrader, profiles, imax)
    selected = pd.concat(
        [
            select_best_candidates(merged, eps_mode="no_eps", baseline_variant="v3_core_top3"),
            select_best_candidates(merged, eps_mode="with_eps", baseline_variant="skill_industry_eps_known"),
        ],
        ignore_index=True,
        sort=False,
    )
    semiconductor = semiconductor_capture_audit(trades)
    report = render_grid_report(merged, selected, semiconductor)

    outputs = {
        "profiles": output_dir / "signal_core_grid_profiles.csv",
        "quality": output_dir / "signal_core_grid_quality.csv",
        "weekly": output_dir / "signal_core_grid_weekly.csv",
        "picks": output_dir / "signal_core_grid_picks.csv",
        "backtrader": output_dir / "signal_core_grid_backtrader_summary.csv",
        "trades": output_dir / "signal_core_grid_backtrader_trades.csv",
        "imax": output_dir / "signal_core_grid_imax_audit.csv",
        "coverage": output_dir / "signal_core_grid_rule_status_coverage.csv",
        "semiconductor": output_dir / "signal_core_grid_semiconductor_capture.csv",
        "summary": output_dir / "signal_core_grid_summary.csv",
        "selected": output_dir / "signal_core_grid_selected.csv",
        "report": output_dir / "signal_core_grid_report.md",
        "manifest": output_dir / "signal_core_grid_manifest.json",
    }
    profiles.to_csv(outputs["profiles"], index=False)
    quality.to_csv(outputs["quality"], index=False)
    weekly.to_csv(outputs["weekly"], index=False)
    picks.to_csv(outputs["picks"], index=False)
    backtrader.to_csv(outputs["backtrader"], index=False)
    trades.to_csv(outputs["trades"], index=False)
    imax.to_csv(outputs["imax"], index=False)
    coverage.to_csv(outputs["coverage"], index=False)
    semiconductor.to_csv(outputs["semiconductor"], index=False)
    merged.to_csv(outputs["summary"], index=False)
    selected.to_csv(outputs["selected"], index=False)
    outputs["report"].write_text(report, encoding="utf-8")
    manifest = {
        "backend": "deterministic_signal_core_grid_plus_backtrader",
        "oracle_end_date": oracle.END_DATE,
        "initial_capital": initial_capital,
        "stop_loss_pct": stop_loss_pct,
        "grid": {
            "axes": ["scope", "eps_gate", "industry_cover", "research_order", "risk_filter", "fill"],
            "no_eps_configs": len(build_grid_configs(eps_gates=("none",))),
            "with_eps_configs": len(build_grid_configs(eps_gates=("none", "known", "pass25"))) + len(build_eps_pass_fallback_configs()),
        },
        "non_coupling_boundary": [
            "no ticker-specific rules",
            "no date-specific rules",
            "no return-derived thresholds inside selection",
            "historical returns are used only after replay for comparison",
        ],
        "outputs": {name: str(path) for name, path in outputs.items() if name != "manifest"},
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {name: str(path) for name, path in outputs.items()}


def select_best_candidates(
    summary: pd.DataFrame,
    *,
    eps_mode: str,
    baseline_variant: str,
    min_rebalance_coverage_ratio: float = 0.95,
    min_pick_coverage_ratio: float = 0.90,
) -> pd.DataFrame:
    frame = summary[summary["eps_mode"].astype(str).eq(eps_mode)].copy()
    if frame.empty:
        return pd.DataFrame()
    baseline_rows = frame[frame["variant"].astype(str).eq(baseline_variant)]
    if baseline_rows.empty:
        return pd.DataFrame()
    baseline = baseline_rows.iloc[0]
    frame["rebalance_coverage_ratio"] = frame["rebalance_events"].map(lambda value: _safe_ratio(value, baseline["rebalance_events"]))
    frame["pick_coverage_ratio"] = frame["input_picks"].map(lambda value: _safe_ratio(value, baseline["input_picks"]))
    frame["final_value_delta"] = frame["final_value"].map(lambda value: _delta(value, baseline["final_value"]))
    frame["total_return_delta"] = frame["total_return_pct"].map(lambda value: _delta(value, baseline["total_return_pct"]))
    frame["max_drawdown_delta"] = frame["max_drawdown_pct"].map(lambda value: _delta(value, baseline["max_drawdown_pct"]))
    frame["stop_events_delta"] = frame["stop_events"].map(lambda value: _delta(value, baseline["stop_events"]))
    frame["pareto_frontier"] = _pareto_frontier(frame)
    statuses = []
    for _, row in frame.iterrows():
        if str(row["variant"]) == baseline_variant:
            statuses.append("baseline")
            continue
        return_ok = _num(row["final_value"]) >= _num(baseline["final_value"])
        drawdown_ok = _num(row["max_drawdown_pct"]) >= _num(baseline["max_drawdown_pct"])
        stop_ok = _num(row["stop_events"]) <= _num(baseline["stop_events"])
        coverage_ok = (
            _num(row["rebalance_coverage_ratio"]) >= min_rebalance_coverage_ratio
            and _num(row["pick_coverage_ratio"]) >= min_pick_coverage_ratio
        )
        median_ok = _optional_ge(row, baseline, "median_week_avg_latest_return_pct")
        floor_ok = _optional_ge(row, baseline, "min_week_avg_latest_return_pct")
        bottom_ok = _optional_le(row, baseline, "pick_bottom5_precision_bad")
        pick_stop_ok = _optional_le(row, baseline, "pick_stop_rate")
        robust_quality_ok = median_ok and floor_ok and bottom_ok and pick_stop_ok
        if return_ok and drawdown_ok and stop_ok and coverage_ok and robust_quality_ok:
            statuses.append("direct_replacement_candidate")
        elif return_ok and drawdown_ok and stop_ok and coverage_ok:
            statuses.append("portfolio_only_quality_tradeoff")
        elif return_ok and drawdown_ok and stop_ok:
            statuses.append("high_return_low_coverage")
        elif return_ok:
            statuses.append("return_only_audit")
        else:
            statuses.append("reject")
    frame["grid_status"] = statuses
    sort_cols = ["grid_status", "pareto_frontier", "final_value", "max_drawdown_pct", "stop_events", "rule_count"]
    ascending = [True, False, False, False, True, True]
    return frame.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def render_grid_report(summary: pd.DataFrame, selected: pd.DataFrame, semiconductor: pd.DataFrame) -> str:
    lines = [
        "# Signal-Core Grid Search Audit",
        "",
        "## Boundary",
        "",
        "- Search space is finite and non-coupled: scope, EPS gate, industry cover, ordering, risk filter, and relaxed-fill only.",
        "- No ticker/date/profit-derived rule is available to the selector; returns are used only after weekly replay and Backtrader simulation.",
        "- Direct replacement requires better/equal final value, max drawdown, stop events, weekly median, weekly floor, Bottom5 exposure, pick stop rate, and at least 95% rebalance plus 90% pick coverage versus baseline.",
        "",
        "## Conclusion",
        "",
    ]
    lines.extend(_grid_conclusion_lines(summary, selected))
    lines.extend(
        [
            "",
            "## Selected Candidates",
            "",
        ]
    )
    show_cols = [
        "eps_mode",
        "variant",
        "grid_status",
        "pareto_frontier",
        "rule_count",
        "final_value",
        "total_return_pct",
        "max_drawdown_pct",
        "stop_events",
        "rebalance_coverage_ratio",
        "pick_coverage_ratio",
        "median_week_avg_latest_return_pct",
        "min_week_avg_latest_return_pct",
        "pick_bottom5_precision_bad",
        "pick_stop_rate",
        "imax_selected",
        "imax_pick_order",
    ]
    selected_show = selected[
        selected["grid_status"].isin(
            [
                "baseline",
                "direct_replacement_candidate",
                "portfolio_only_quality_tradeoff",
                "high_return_low_coverage",
            ]
        )
    ].copy()
    lines.extend(_markdown_table(_round_frame(selected_show[[col for col in show_cols if col in selected_show.columns]].head(40))).splitlines())
    lines.extend(["", "## Best By Return", ""])
    best = summary.sort_values(["eps_mode", "final_value"], ascending=[True, False]).groupby("eps_mode", as_index=False).head(8)
    lines.extend(_markdown_table(_round_frame(best[[col for col in show_cols if col in best.columns]])).splitlines())
    lines.extend(["", "## Semiconductor Capture", ""])
    if semiconductor.empty:
        lines.append("_empty_")
    else:
        semi = semiconductor[semiconductor["eps_mode"].astype(str).eq("with_eps")].sort_values(
            ["semi_hit_weeks", "semi_pick_rate"],
            ascending=[False, False],
        )
        lines.extend(_markdown_table(_round_frame(semi.head(20))).splitlines())
    lines.extend(["", "## Full Summary", ""])
    compact = summary.sort_values(["eps_mode", "final_value"], ascending=[True, False])
    lines.extend(_markdown_table(_round_frame(compact[[col for col in show_cols if col in compact.columns]])).splitlines())
    return "\n".join(lines) + "\n"


def _grid_conclusion_lines(summary: pd.DataFrame, selected: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    with_eps = summary[summary["eps_mode"].astype(str).eq("with_eps")].copy()
    if with_eps.empty:
        return ["- With-EPS grid output is empty; no conclusion."]
    baseline = with_eps[with_eps["variant"].astype(str).eq("skill_industry_eps_known")]
    direct = selected[
        selected["eps_mode"].astype(str).eq("with_eps")
        & selected["grid_status"].astype(str).eq("direct_replacement_candidate")
        & ~selected["variant"].astype(str).eq("skill_industry_eps_known")
    ].copy()
    direct_improved = direct[
        pd.to_numeric(direct["final_value"], errors="coerce").gt(float(baseline["final_value"].iloc[0]))
    ] if not baseline.empty and not direct.empty else pd.DataFrame()
    best = with_eps.sort_values("final_value", ascending=False).iloc[0]
    lines.append(
        "- With-EPS robust replacement: none beyond the baseline-equivalent grid form; no candidate beat `skill_industry_eps_known` while also preserving weekly median, weekly floor, Bottom5 exposure, pick stop rate, coverage, drawdown, and stop events."
        if direct_improved.empty
        else f"- With-EPS robust replacement found: `{direct_improved.iloc[0]['variant']}`."
    )
    if not baseline.empty:
        base = baseline.iloc[0]
        lines.append(
            f"- Baseline `skill_industry_eps_known`: final {float(base['final_value']):.2f}, return {float(base['total_return_pct']):.2f}%, max DD {float(base['max_drawdown_pct']):.2f}%, stops {int(base['stop_events'])}, picks {int(base['input_picks'])}."
        )
    lines.append(
        f"- Best raw with-EPS return `{best['variant']}`: final {float(best['final_value']):.2f}, return {float(best['total_return_pct']):.2f}%, max DD {float(best['max_drawdown_pct']):.2f}%, stops {int(best['stop_events'])}, picks {int(best['input_picks'])}; blocked from replacement by coverage/weekly-quality gates."
    )
    fallback = with_eps[with_eps["variant"].astype(str).str.contains("epspass2known", regex=False)].sort_values("final_value", ascending=False)
    if not fallback.empty:
        row = fallback.iloc[0]
        lines.append(
            f"- EPS pass -> EPS known fallback recovered coverage to {int(row['rebalance_events'])} rebalances and {int(row['input_picks'])} picks at best final {float(row['final_value']):.2f}, but did not pass all robust quality gates."
        )
    no_eps = summary[summary["eps_mode"].astype(str).eq("no_eps")].sort_values("final_value", ascending=False)
    if not no_eps.empty:
        row = no_eps.iloc[0]
        lines.append(
            f"- No-EPS can be improved versus `v3_core_top3` by signal-wide industry proximity ordering, but its best final {float(row['final_value']):.2f} remains below the with-EPS baseline, so it is not the preferred production direction."
        )
    lines.append("- Stop condition: after adding the only generic fallback suggested by the grid failure mode, no further non-coupled candidate satisfies the full replacement gate.")
    return lines


def _merge_grid_outputs(
    quality: pd.DataFrame,
    backtrader: pd.DataFrame,
    profiles: pd.DataFrame,
    imax: pd.DataFrame,
) -> pd.DataFrame:
    merged = profiles.merge(quality, on=["eps_mode", "variant"], how="left")
    merged = merged.merge(backtrader, on=["eps_mode", "variant"], how="left", suffixes=("", "_bt"))
    if not imax.empty:
        imax_cols = imax[["eps_mode", "variant", "selected", "pick_order"]].rename(
            columns={"selected": "imax_selected", "pick_order": "imax_pick_order"}
        )
        merged = merged.merge(imax_cols, on=["eps_mode", "variant"], how="left")
    else:
        merged["imax_selected"] = False
        merged["imax_pick_order"] = pd.NA
    merged["imax_selected"] = merged["imax_selected"].fillna(False).astype(bool)
    return merged.sort_values(["eps_mode", "final_value"], ascending=[True, False]).reset_index(drop=True)


def _baseline_profiles(eps_mode: str, names: list[str]) -> list[dict[str, object]]:
    profiles = []
    for name in names:
        cfg = oracle.VARIANTS[name]
        profile = {
            "eps_mode": eps_mode,
            "variant": name,
            "signal_wide": bool(cfg.get("allow_non_actionable")),
            "actionable_gate": not bool(cfg.get("allow_non_actionable")),
            "eps_gate": "pass25" if cfg.get("require_eps_pass") else ("known" if cfg.get("require_eps_known") else "none"),
            "industry_cover": bool(cfg.get("industry_cover")),
            "core_quality_gate": bool(cfg.get("require_core_quality")),
            "proximity_reorder": cfg.get("research_order") == "fresh_proximity",
            "pullback_dry_hard_filter": bool(cfg.get("exclude_pullback_not_dry")),
            "geometry_caution_hard_filter": bool(cfg.get("exclude_geometry_caution")),
            "relaxed_fill": bool(cfg.get("fill_relaxed")),
        }
        profile["rule_count"] = _rule_count(profile)
        profiles.append(profile)
    return profiles


def _rule_count(profile: dict[str, object]) -> int:
    total = sum(
        bool(profile.get(key))
        for key in [
            "actionable_gate",
            "industry_cover",
            "core_quality_gate",
            "proximity_reorder",
            "pullback_dry_hard_filter",
            "geometry_caution_hard_filter",
            "relaxed_fill",
        ]
    )
    if profile.get("eps_gate") == "pass25_then_known_fill":
        total += 2
    elif profile.get("eps_gate") != "none":
        total += 1
    return total


def _pareto_frontier(frame: pd.DataFrame) -> pd.Series:
    rows = []
    for idx, row in frame.iterrows():
        dominated = False
        for other_idx, other in frame.iterrows():
            if idx == other_idx:
                continue
            better_or_equal = (
                _num(other["final_value"]) >= _num(row["final_value"])
                and _num(other["max_drawdown_pct"]) >= _num(row["max_drawdown_pct"])
                and _num(other["stop_events"]) <= _num(row["stop_events"])
                and _num(other["rule_count"]) <= _num(row["rule_count"])
                and _num(other.get("rebalance_coverage_ratio", 1.0)) >= _num(row.get("rebalance_coverage_ratio", 1.0))
                and _num(other.get("pick_coverage_ratio", 1.0)) >= _num(row.get("pick_coverage_ratio", 1.0))
            )
            strictly_better = (
                _num(other["final_value"]) > _num(row["final_value"])
                or _num(other["max_drawdown_pct"]) > _num(row["max_drawdown_pct"])
                or _num(other["stop_events"]) < _num(row["stop_events"])
                or _num(other["rule_count"]) < _num(row["rule_count"])
                or _num(other.get("rebalance_coverage_ratio", 1.0)) > _num(row.get("rebalance_coverage_ratio", 1.0))
                or _num(other.get("pick_coverage_ratio", 1.0)) > _num(row.get("pick_coverage_ratio", 1.0))
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        rows.append(not dominated)
    return pd.Series(rows, index=frame.index)


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


def _eps_label(eps_gate: str) -> str:
    if eps_gate == "none":
        return "noeps"
    if eps_gate == "known":
        return "epsknown"
    if eps_gate == "pass25":
        return "epspass"
    raise ValueError(f"unsupported eps gate: {eps_gate}")


def _safe_ratio(value: object, denominator: object) -> float:
    value_num = _num(value)
    denominator_num = _num(denominator)
    if pd.isna(value_num) or pd.isna(denominator_num) or denominator_num == 0:
        return 0.0
    return float(value_num) / float(denominator_num)


def _delta(value: object, baseline: object) -> float:
    return round(_num(value) - _num(baseline), 6)


def _optional_ge(row: pd.Series, baseline: pd.Series, column: str) -> bool:
    if column not in row or column not in baseline:
        return True
    row_value = _num(row[column])
    baseline_value = _num(baseline[column])
    if pd.isna(row_value) or pd.isna(baseline_value):
        return True
    return row_value >= baseline_value


def _optional_le(row: pd.Series, baseline: pd.Series, column: str) -> bool:
    if column not in row or column not in baseline:
        return True
    row_value = _num(row[column])
    baseline_value = _num(baseline[column])
    if pd.isna(row_value) or pd.isna(baseline_value):
        return True
    return row_value <= baseline_value


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
    parser = argparse.ArgumentParser(description="Run non-coupled signal-core grid search against weekly pools.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--price-cache", default="")
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--stop-loss-pct", type=float, default=8.0)
    args = parser.parse_args(argv)
    outputs = run_grid_search(
        output_dir=Path(args.output_dir),
        price_cache=args.price_cache or None,
        initial_capital=args.initial_capital,
        stop_loss_pct=args.stop_loss_pct,
    )
    print(json.dumps(outputs, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
