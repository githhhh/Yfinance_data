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


def pair_outcome_audit(
    universe: pd.DataFrame,
    picks: pd.DataFrame,
    *,
    snapshot_date: str,
    codes: tuple[str, str] = ("BLFS", "IMAX"),
) -> pd.DataFrame:
    rows = []
    scoped_universe = universe[
        universe["snapshot_date"].astype(str).eq(str(snapshot_date))
        & universe["code"].astype(str).isin(codes)
    ].copy()
    if scoped_universe.empty:
        return pd.DataFrame()
    scoped_picks = picks[
        picks["snapshot_date"].astype(str).eq(str(snapshot_date))
        & picks["code"].astype(str).isin(codes)
    ].copy()
    latest_values = pd.to_numeric(scoped_universe["latest_return_pct"], errors="coerce")
    gain_values = pd.to_numeric(scoped_universe["max_gain_pct"], errors="coerce")
    best_latest = latest_values.max()
    best_gain = gain_values.max()
    by_code = {str(row["code"]): row for _, row in scoped_universe.iterrows()}
    for code in codes:
        if code not in by_code:
            continue
        row = by_code[code]
        peer_codes = [item for item in codes if item != code and item in by_code]
        peer_return = pd.NA
        if peer_codes:
            peer_return = to_numeric_or_na(by_code[peer_codes[0]].get("latest_return_pct"))
        latest_return = to_numeric_or_na(row.get("latest_return_pct"))
        pick_rows = scoped_picks[scoped_picks["code"].astype(str).eq(code)].sort_values(["variant", "pick_order"])
        variant_orders = ";".join(
            f"{pick['variant']}:{int(pick['pick_order'])}"
            for _, pick in pick_rows.iterrows()
            if pd.notna(pick.get("pick_order"))
        )
        rows.append(
            {
                "snapshot_date": snapshot_date,
                "code": code,
                "latest_return_pct": latest_return,
                "max_gain_pct": to_numeric_or_na(row.get("max_gain_pct")),
                "latest_rank": row.get("latest_rank"),
                "gain_rank": row.get("gain_rank"),
                "latest_return_delta_vs_peer": _delta(latest_return, peer_return),
                "best_latest_return_in_pair": bool(pd.notna(latest_return) and latest_return == best_latest),
                "best_max_gain_in_pair": bool(pd.notna(row.get("max_gain_pct")) and to_numeric_or_na(row.get("max_gain_pct")) == best_gain),
                "variant_orders": variant_orders,
                "reason_codes": _first_nonempty(pick_rows.get("reason_codes")),
                "risk_codes": _first_nonempty(pick_rows.get("risk_codes")),
            }
        )
    return pd.DataFrame(rows)


def intuitive_variant_summary(weekly: pd.DataFrame, *, eps_mode: str, variant: str) -> dict[str, object]:
    frame = _filter_variant(weekly, eps_mode, variant)
    frame = frame.copy()
    frame["avg_latest_return_pct"] = pd.to_numeric(frame.get("avg_latest_return_pct"), errors="coerce")
    frame["picks"] = pd.to_numeric(frame.get("picks"), errors="coerce")
    valid = frame[frame["avg_latest_return_pct"].notna()]
    weeks = int(len(frame))
    valid_weeks = int(len(valid))
    positive = int(valid["avg_latest_return_pct"].gt(0).sum())
    negative = int(valid["avg_latest_return_pct"].lt(0).sum())
    flat = int(valid["avg_latest_return_pct"].eq(0).sum())
    less_than_3 = int(valid["picks"].lt(3).sum()) if "picks" in valid else 0
    return {
        "eps_mode": eps_mode,
        "variant": variant,
        "weeks": weeks,
        "valid_return_weeks": valid_weeks,
        "missing_path_weeks": weeks - valid_weeks,
        "positive_weeks": positive,
        "negative_weeks": negative,
        "flat_weeks": flat,
        "positive_week_rate": positive / valid_weeks if valid_weeks else 0.0,
        "negative_week_rate": negative / valid_weeks if valid_weeks else 0.0,
        "weeks_with_less_than_3_picks": less_than_3,
    }


def absorption_candidate_matrix(
    quality: pd.DataFrame,
    *,
    eps_mode: str,
    baseline_variant: str,
    min_week_coverage_ratio: float = 0.70,
    min_pick_coverage_ratio: float = 0.60,
) -> pd.DataFrame:
    baseline = _quality_row(quality, eps_mode, baseline_variant)
    if baseline is None:
        return pd.DataFrame()
    frame = quality[quality["eps_mode"].astype(str).eq(eps_mode)].copy()
    rows = []
    for _, row in frame.iterrows():
        variant = str(row.get("variant"))
        if variant == baseline_variant:
            continue
        week_ratio = _safe_ratio(row.get("weeks"), baseline.get("weeks"))
        pick_ratio = _safe_ratio(row.get("picks"), baseline.get("picks"))
        median_ok = _leq_or_equal(baseline.get("median_week_avg_latest_return_pct"), row.get("median_week_avg_latest_return_pct"))
        floor_ok = _leq_or_equal(baseline.get("min_week_avg_latest_return_pct"), row.get("min_week_avg_latest_return_pct"))
        bottom_ok = _leq_or_equal(row.get("pick_bottom5_precision_bad"), baseline.get("pick_bottom5_precision_bad"))
        stop_ok = _leq_or_equal(row.get("pick_stop_rate"), baseline.get("pick_stop_rate"))
        coverage_ok = week_ratio >= min_week_coverage_ratio and pick_ratio >= min_pick_coverage_ratio
        if not coverage_ok:
            status = "reject_low_coverage"
        elif median_ok and floor_ok and bottom_ok and stop_ok:
            status = "candidate_absorbable"
        elif floor_ok and bottom_ok and stop_ok:
            status = "high_confidence_audit"
        elif row.get("median_week_avg_latest_return_pct", 0) > baseline.get("median_week_avg_latest_return_pct", 0):
            status = "audit_only"
        else:
            status = "reject_quality"
        rows.append(
            {
                "eps_mode": eps_mode,
                "baseline_variant": baseline_variant,
                "variant": variant,
                "absorption_status": status,
                "week_coverage_ratio": week_ratio,
                "pick_coverage_ratio": pick_ratio,
                "median_delta": _delta(row.get("median_week_avg_latest_return_pct"), baseline.get("median_week_avg_latest_return_pct")),
                "floor_delta": _delta(row.get("min_week_avg_latest_return_pct"), baseline.get("min_week_avg_latest_return_pct")),
                "bottom5_delta": _delta(row.get("pick_bottom5_precision_bad"), baseline.get("pick_bottom5_precision_bad")),
                "stop_delta": _delta(row.get("pick_stop_rate"), baseline.get("pick_stop_rate")),
                "decision_reason": _candidate_reason(status, coverage_ok, median_ok, floor_ok, bottom_ok, stop_ok),
            }
        )
    return pd.DataFrame(rows).sort_values(["absorption_status", "variant"]).reset_index(drop=True)


def stop_loss_capital_backtest(
    picks: pd.DataFrame,
    *,
    eps_mode: str,
    variant: str,
    initial_capital: float = 100000.0,
    stop_loss_pct: float = -8.0,
) -> dict[str, object]:
    frame = _filter_variant(picks, eps_mode, variant)
    input_picks = int(len(frame))
    if frame.empty:
        return {
            "eps_mode": eps_mode,
            "variant": variant,
            "initial_capital": float(initial_capital),
            "stop_loss_pct": float(stop_loss_pct),
            "weeks": 0,
            "input_picks": 0,
            "priced_picks": 0,
            "picks": 0,
            "final_equity": float(initial_capital),
            "total_return_pct": 0.0,
            "first_week_return_pct": pd.NA,
            "positive_weeks": 0,
            "negative_weeks": 0,
        }
    frame = frame.copy()
    frame["latest_return_pct"] = pd.to_numeric(frame.get("latest_return_pct"), errors="coerce")
    frame = frame[frame["latest_return_pct"].notna()]
    if frame.empty:
        return {
            "eps_mode": eps_mode,
            "variant": variant,
            "initial_capital": float(initial_capital),
            "stop_loss_pct": float(stop_loss_pct),
            "weeks": 0,
            "input_picks": input_picks,
            "priced_picks": 0,
            "picks": 0,
            "final_equity": float(initial_capital),
            "total_return_pct": 0.0,
            "first_week_return_pct": pd.NA,
            "positive_weeks": 0,
            "negative_weeks": 0,
        }
    frame["hit_stop_8pct"] = frame.get("hit_stop_8pct", False)
    frame["realized_return_pct"] = frame["latest_return_pct"]
    frame.loc[frame["hit_stop_8pct"].fillna(False).astype(bool), "realized_return_pct"] = float(stop_loss_pct)
    weekly_returns = (
        frame.groupby("snapshot_date", sort=True)["realized_return_pct"]
        .mean()
        .reset_index(name="weekly_return_pct")
    )
    equity = float(initial_capital)
    first_week_return = pd.NA
    for idx, row in weekly_returns.iterrows():
        weekly_return = float(row["weekly_return_pct"])
        if idx == 0:
            first_week_return = weekly_return
        equity *= 1.0 + weekly_return / 100.0
    return {
        "eps_mode": eps_mode,
        "variant": variant,
        "initial_capital": float(initial_capital),
        "stop_loss_pct": float(stop_loss_pct),
        "weeks": int(len(weekly_returns)),
        "input_picks": input_picks,
        "priced_picks": int(len(frame)),
        "picks": int(len(frame)),
        "final_equity": float(equity),
        "total_return_pct": (float(equity) / float(initial_capital) - 1.0) * 100.0 if initial_capital else 0.0,
        "first_week_return_pct": first_week_return,
        "positive_weeks": int(weekly_returns["weekly_return_pct"].gt(0).sum()),
        "negative_weeks": int(weekly_returns["weekly_return_pct"].lt(0).sum()),
    }


def backtrader_decision_matrix(
    summary: pd.DataFrame,
    *,
    eps_mode: str,
    baseline_variant: str,
    min_rebalance_coverage_ratio: float = 0.95,
    min_pick_coverage_ratio: float = 0.90,
) -> pd.DataFrame:
    baseline = _summary_row(summary, eps_mode, baseline_variant)
    if baseline is None:
        return pd.DataFrame()
    frame = summary[summary["eps_mode"].astype(str).eq(eps_mode)].copy()
    rows = []
    for _, row in frame.iterrows():
        variant = str(row.get("variant"))
        if variant == baseline_variant:
            continue
        rebalance_ratio = _safe_ratio(row.get("rebalance_events"), baseline.get("rebalance_events"))
        pick_ratio = _safe_ratio(row.get("input_picks"), baseline.get("input_picks"))
        return_ok = _leq_or_equal(baseline.get("final_value"), row.get("final_value"))
        drawdown_ok = _leq_or_equal(baseline.get("max_drawdown_pct"), row.get("max_drawdown_pct"))
        stop_ok = _leq_or_equal(row.get("stop_events"), baseline.get("stop_events"))
        coverage_ok = rebalance_ratio >= min_rebalance_coverage_ratio and pick_ratio >= min_pick_coverage_ratio
        if return_ok and drawdown_ok and stop_ok and coverage_ok:
            status = "direct_replacement_candidate"
        elif return_ok and drawdown_ok and stop_ok:
            status = "high_confidence_candidate"
        elif return_ok:
            status = "audit_only"
        else:
            status = "reject_backtrader"
        rows.append(
            {
                "eps_mode": eps_mode,
                "baseline_variant": baseline_variant,
                "variant": variant,
                "backtrader_status": status,
                "rebalance_coverage_ratio": rebalance_ratio,
                "pick_coverage_ratio": pick_ratio,
                "final_value_delta": _delta(row.get("final_value"), baseline.get("final_value")),
                "total_return_delta": _delta(row.get("total_return_pct"), baseline.get("total_return_pct")),
                "max_drawdown_delta": _delta(row.get("max_drawdown_pct"), baseline.get("max_drawdown_pct")),
                "stop_events_delta": _delta(row.get("stop_events"), baseline.get("stop_events")),
                "decision_reason": _backtrader_decision_reason(status, coverage_ok, return_ok, drawdown_ok, stop_ok),
            }
        )
    return pd.DataFrame(rows).sort_values(["backtrader_status", "variant"]).reset_index(drop=True)


def render_markdown_report(
    quality: pd.DataFrame,
    imax: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    pair_audit: pd.DataFrame | None = None,
    intuitive_summary: pd.DataFrame | None = None,
    absorption_matrix: pd.DataFrame | None = None,
    stop_loss_backtest: pd.DataFrame | None = None,
    backtrader_summary: pd.DataFrame | None = None,
    backtrader_decisions: pd.DataFrame | None = None,
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
    if intuitive_summary is not None:
        lines.extend(["", "## Intuitive Baseline Summary", ""])
        lines.extend(_markdown_table(_round_frame(intuitive_summary)).splitlines())
    if stop_loss_backtest is not None:
        lines.extend(["", "## Stop-Loss Capital Replay", ""])
        lines.append("- Replay lens: each snapshot allocates equal capital across selected picks; stopped picks realize the configured stop-loss return, then weekly returns compound from the initial capital.")
        lines.append("- This is not a live portfolio simulation: replay paths overlap and all path returns use the fixed oracle end date.")
        lines.extend(_markdown_table(_round_frame(stop_loss_backtest)).splitlines())
    if backtrader_summary is not None:
        lines.extend(["", "## Backtrader Weekly Rebalance Backtest", ""])
        lines.append("- Primary return evidence: Backtrader event-driven broker simulation with weekly rebalance, actual OHLC feeds, next tradable open execution, and per-position stop orders.")
        lines.extend(_markdown_table(_round_frame(backtrader_summary)).splitlines())
    if backtrader_decisions is not None:
        lines.extend(["", "## Backtrader Replacement Decision", ""])
        lines.append("- Direct replacement requires better/equal final value, drawdown, stop events, and similar rebalance/pick coverage versus the baseline.")
        lines.extend(_markdown_table(_round_frame(backtrader_decisions)).splitlines())
    if pair_audit is not None:
        lines.extend(["", "## BLFS / IMAX Pair Audit", ""])
        lines.extend(_markdown_table(_round_frame(pair_audit)).splitlines())
    lines.extend(["", "## Skill Absorption Reading", ""])
    lines.extend(_skill_absorption_reading(quality, imax))
    if absorption_matrix is not None:
        lines.extend(["", "## Candidate Absorption Matrix", ""])
        lines.extend(_markdown_table(_round_frame(absorption_matrix)).splitlines())
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


def to_numeric_or_na(value: object) -> float | object:
    result = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(result):
        return pd.NA
    return float(result)


def _delta(value: object, baseline: object) -> float | object:
    value_num = to_numeric_or_na(value)
    baseline_num = to_numeric_or_na(baseline)
    if pd.isna(value_num) or pd.isna(baseline_num):
        return pd.NA
    return round(float(value_num) - float(baseline_num), 6)


def _first_nonempty(series: pd.Series | None) -> str:
    if series is None:
        return ""
    for value in series:
        if pd.notna(value) and str(value).strip():
            return str(value)
    return ""


def _safe_ratio(value: object, denominator: object) -> float:
    value_num = to_numeric_or_na(value)
    denominator_num = to_numeric_or_na(denominator)
    if pd.isna(value_num) or pd.isna(denominator_num) or float(denominator_num) == 0:
        return 0.0
    return float(value_num) / float(denominator_num)


def _leq_or_equal(left: object, right: object) -> bool:
    left_num = to_numeric_or_na(left)
    right_num = to_numeric_or_na(right)
    if pd.isna(left_num) or pd.isna(right_num):
        return False
    return float(left_num) <= float(right_num)


def _candidate_reason(status: str, coverage_ok: bool, median_ok: bool, floor_ok: bool, bottom_ok: bool, stop_ok: bool) -> str:
    failed = []
    if not coverage_ok:
        failed.append("coverage")
    if not median_ok:
        failed.append("median")
    if not floor_ok:
        failed.append("weekly_floor")
    if not bottom_ok:
        failed.append("bottom5")
    if not stop_ok:
        failed.append("stop")
    if not failed:
        return f"{status}: non-coupled factor gates pass against baseline"
    return f"{status}: failed " + ",".join(failed)


def _skill_absorption_reading(quality: pd.DataFrame, imax: pd.DataFrame) -> list[str]:
    lines = [
        "- `signal_shadow_top3` is audit-only: it captures more big winners, but also carries materially higher Bottom5 and stop exposure.",
        "- `skill_industry_eps_known` remains the formal with-EPS baseline because its risk profile is more balanced than the research variants.",
    ]
    skill = _quality_row(quality, "with_eps", "skill_industry_eps_known")
    fresh = _quality_row(quality, "with_eps", "research_fresh_demand_proximity_first")
    pullback = _quality_row(quality, "with_eps", "research_pullback_vcp_lane_interleave")
    floor_guard = _quality_row(quality, "with_eps", "research_proximity_eps_known_floor_guard")
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
    if skill is not None and floor_guard is not None:
        floor_delta = _delta(floor_guard["min_week_avg_latest_return_pct"], skill["min_week_avg_latest_return_pct"])
        stop_delta = _delta(floor_guard["pick_stop_rate"], skill["pick_stop_rate"])
        pick_ratio = _safe_ratio(floor_guard["picks"], skill["picks"])
        lines.append(
            f"- `research_proximity_eps_known_floor_guard` is a high-confidence audit lane: weekly floor improves by {floor_delta} pct points and stop rate changes by {stop_delta}, but pick coverage is {pick_ratio:.2f}x of the formal baseline."
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


def _summary_row(summary: pd.DataFrame, eps_mode: str, variant: str) -> pd.Series | None:
    rows = summary[summary["eps_mode"].astype(str).eq(eps_mode) & summary["variant"].astype(str).eq(variant)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _backtrader_decision_reason(status: str, coverage_ok: bool, return_ok: bool, drawdown_ok: bool, stop_ok: bool) -> str:
    failed = []
    if not coverage_ok:
        failed.append("coverage")
    if not return_ok:
        failed.append("final_value")
    if not drawdown_ok:
        failed.append("drawdown")
    if not stop_ok:
        failed.append("stop_events")
    if not failed:
        return f"{status}: Backtrader return/risk/coverage gates pass"
    return f"{status}: failed " + ",".join(failed)
