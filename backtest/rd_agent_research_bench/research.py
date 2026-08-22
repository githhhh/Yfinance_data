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
