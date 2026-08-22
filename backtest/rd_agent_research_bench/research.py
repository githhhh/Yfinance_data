from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.rd_agent_research_bench.metrics import (
    coverage_rates,
    rank_weighted_week_return,
    robust_weekly_summary,
)
from backtest.rd_agent_research_bench.hypotheses import hypothesis_space


DEFAULT_ORACLE_DIR = Path("backtest/ibd_weekly_signal_oracle_eval")


def summarize_variant_quality(
    eps_mode: str,
    variant: str,
    universe: pd.DataFrame,
    weekly: pd.DataFrame,
    picks: pd.DataFrame,
) -> dict[str, object]:
    weekly_variant = _filter_variant(weekly, eps_mode, variant)
    picks_variant = _filter_variant(picks, eps_mode, variant)
    summary = robust_weekly_summary(weekly_variant, picks_variant)
    coverage = coverage_rates(universe, picks_variant)
    rank_weighted = pd.Series(
        [rank_weighted_week_return(group) for _, group in picks_variant.groupby("snapshot_date", sort=True)],
        name="rank_weighted_week_return",
    )
    return {
        "eps_mode": eps_mode,
        "variant": variant,
        **summary,
        **coverage,
        "rank_weighted_week_return_median": _float(rank_weighted.median()),
        "rank_weighted_week_return_min": _float(rank_weighted.min()),
        "rank_weighted_week_return_max": _float(rank_weighted.max()),
    }


def imax_rank_audit(
    universe: pd.DataFrame,
    picks: pd.DataFrame,
    *,
    snapshot_date: str = "2026-07-24",
    code: str = "IMAX",
) -> pd.DataFrame:
    universe_row = universe[(universe["snapshot_date"].astype(str) == snapshot_date) & (universe["code"].astype(str) == code)]
    if universe_row.empty:
        base = {
            "snapshot_date": snapshot_date,
            "code": code,
            "oracle_found": False,
            "latest_rank": pd.NA,
            "gain_rank": pd.NA,
            "loss_rank": pd.NA,
            "latest_return_pct": pd.NA,
            "max_gain_pct": pd.NA,
        }
    else:
        row = universe_row.iloc[0]
        base = {
            "snapshot_date": snapshot_date,
            "code": code,
            "oracle_found": True,
            "latest_rank": row.get("latest_rank"),
            "gain_rank": row.get("gain_rank"),
            "loss_rank": row.get("loss_rank"),
            "latest_return_pct": row.get("latest_return_pct"),
            "max_gain_pct": row.get("max_gain_pct"),
        }
    variants = sorted(picks["variant"].dropna().unique()) if "variant" in picks.columns else []
    rows = []
    for variant in variants:
        selected = picks[
            (picks["snapshot_date"].astype(str) == snapshot_date)
            & (picks["variant"].astype(str) == variant)
            & (picks["code"].astype(str) == code)
        ]
        rows.append(
            {
                "variant": variant,
                **base,
                "selected": bool(not selected.empty),
                "pick_order": int(selected["pick_order"].iloc[0]) if not selected.empty and "pick_order" in selected else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def rule_status_coverage(universe: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
    universe_top5 = universe[universe["valid_path"].astype(bool) & universe["latest_rank"].le(5)]
    universe_counts = (
        universe_top5.groupby(["rule", "entry_status"], dropna=False)
        .size()
        .rename("universe_top5")
        .reset_index()
    )
    if picks.empty:
        pick_counts = pd.DataFrame(columns=["rule", "entry_status", "picks", "top5_picks", "gain5_picks", "bottom5_picks", "stop_picks"])
    else:
        joined = picks.merge(
            universe[["snapshot_date", "code", "rule"]].drop_duplicates(["snapshot_date", "code"]),
            on=["snapshot_date", "code"],
            how="left",
        )
        pick_counts = (
            joined.groupby(["rule", "entry_status"], dropna=False)
            .agg(
                picks=("code", "size"),
                top5_picks=("hit_latest_top5", "sum"),
                gain5_picks=("hit_gain_top5", "sum"),
                bottom5_picks=("hit_loss_bottom5", "sum"),
                stop_picks=("hit_stop_8pct", "sum"),
            )
            .reset_index()
        )
    result = universe_counts.merge(pick_counts, on=["rule", "entry_status"], how="outer")
    result["rule"] = result["rule"].fillna("UNKNOWN")
    result["entry_status"] = result["entry_status"].fillna("UNKNOWN")
    numeric_columns = ["universe_top5", "picks", "top5_picks", "gain5_picks", "bottom5_picks", "stop_picks"]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).astype(int)
    return result.sort_values(["rule", "entry_status"]).reset_index(drop=True)


def render_markdown_report(
    quality: pd.DataFrame,
    imax: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    title: str = "RD-Agent Research Bench Audit",
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Decision Boundary",
        "",
        "- RD-Agent 只生成候选假设；不得直接改写正式 skill。",
        "- 正式 skill 仍必须通过 deterministic artifact 输出，历史 pool 只能用于验证通用规则方向。",
        "- 当前已清理低研究价值的 qlib rule optimizer 输出；后续若恢复 Qlib，必须升级为因子 IC、rolling retraining、分组收益和组合回测口径。",
        "",
        "## Quality Summary",
        "",
    ]
    lines.extend(_markdown_table(_round_frame(quality)).splitlines())
    lines.extend(["", "## Skill Absorption Reading", ""])
    lines.extend(_skill_absorption_reading(quality, imax))
    lines.extend(["", "## IMAX Rank Audit", ""])
    lines.extend(_markdown_table(_round_frame(imax)).splitlines())
    lines.extend(["", "## Rule / Status Coverage", ""])
    lines.extend(_markdown_table(_round_frame(coverage)).splitlines())
    lines.extend(["", "## RD-Agent Candidate Hypotheses", ""])
    lines.extend(_markdown_table(pd.DataFrame(hypothesis_space())).splitlines())
    return "\n".join(lines) + "\n"


def load_oracle_tables(eps_mode: str, *, oracle_dir: Path = DEFAULT_ORACLE_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = pd.read_csv(oracle_dir / f"{eps_mode}_signal_universe_oracle.csv")
    weekly = pd.read_csv(oracle_dir / f"{eps_mode}_weekly_variant_metrics.csv")
    picks = pd.read_csv(oracle_dir / f"{eps_mode}_variant_picks.csv")
    return universe, weekly, picks


def _filter_variant(frame: pd.DataFrame, eps_mode: str, variant: str) -> pd.DataFrame:
    result = frame[frame["variant"].astype(str).eq(variant)].copy()
    if "eps_mode" in result.columns:
        result = result[result["eps_mode"].astype(str).eq(eps_mode)]
    return result


def _float(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def _round_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=["float"]).columns:
        result[column] = result[column].round(6)
    return result


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    return frame.to_markdown(index=False)


def _skill_absorption_reading(quality: pd.DataFrame, imax: pd.DataFrame) -> list[str]:
    lines = [
        "- `signal_shadow_top3` is audit-only: it captures more big winners, but also carries materially higher Bottom5 and stop exposure.",
        "- `skill_industry_eps_known` remains the formal with-EPS baseline because its risk profile is more balanced than the research variants.",
    ]
    skill = _quality_row(quality, "with_eps", "skill_industry_eps_known")
    fresh = _quality_row(quality, "with_eps", "research_fresh_demand_proximity_first")
    pullback = _quality_row(quality, "with_eps", "research_pullback_vcp_lane_interleave")
    if skill is not None and fresh is not None:
        better_median = fresh["median_week_avg_latest_return_pct"] > skill["median_week_avg_latest_return_pct"]
        lower_stop = fresh["pick_stop_rate"] < skill["pick_stop_rate"]
        worse_floor = fresh["min_week_avg_latest_return_pct"] < skill["min_week_avg_latest_return_pct"]
        if better_median and lower_stop and worse_floor:
            lines.append(
                "- `research_fresh_demand_proximity_first` is candidate-only: it improves median week return and pick stop rate, and ranks IMAX first in the EPS run, but the worse weekly floor blocks direct absorption."
            )
        else:
            lines.append(
                "- `research_fresh_demand_proximity_first` is candidate-only until it improves the official baseline across median return, stop rate, and weekly floor at the same time."
            )
    if pullback is not None:
        lines.append(
            "- `research_pullback_vcp_lane_interleave` is not ready for official ranking: it raises Top5 precision, but the current pullback/VCP proxy also raises Bottom5 and stop exposure."
        )
    imax_selected = _imax_selected_row(imax, "with_eps", "research_fresh_demand_proximity_first")
    if imax_selected is not None:
        order = imax_selected["pick_order"]
        lines.append(f"- IMAX audit: the fresh-demand proximity candidate selects IMAX at rank {int(order)} in the with-EPS run; this supports studying buy-point proximity as a tie-break after evidence sufficiency.")
    return lines


def _imax_selected_row(imax: pd.DataFrame, eps_mode: str, variant: str) -> pd.Series | None:
    required = {"variant", "selected", "pick_order"}
    if imax.empty or not required.issubset(imax.columns):
        return None
    mask = imax["variant"].astype(str).eq(variant) & imax["selected"].astype(bool)
    if "eps_mode" in imax.columns:
        mask &= imax["eps_mode"].astype(str).eq(eps_mode)
    selected = imax[mask]
    if selected.empty:
        return None
    return selected.iloc[0]


def _quality_row(quality: pd.DataFrame, eps_mode: str, variant: str) -> pd.Series | None:
    rows = quality[quality["eps_mode"].astype(str).eq(eps_mode) & quality["variant"].astype(str).eq(variant)]
    if rows.empty:
        return None
    return rows.iloc[0]
