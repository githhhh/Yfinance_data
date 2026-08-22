from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

sys.path.insert(0, str(Path.cwd()))

from backtest.ibd_skill_replay.core import compute_path_metrics, to_bool, to_float
from backtest.ibd_skill_replay.run_ytd_replay import _load_price_cache
from backtest.ibd_weekly_signal_oracle_eval.price_cache import resolve_price_cache
import eps_pit.lookup as eps_lookup


POOL_ROOT = Path("backtest/ibd_skill_replay_pools")
DEFAULT_END_DATE = "2026-08-14"


@dataclass(frozen=True)
class PortfolioRule:
    name: str
    weights: dict[str, float]
    penalties: dict[str, float] = field(default_factory=dict)
    require_actionable: bool = True
    industry_cap: bool = True
    require_eps_known: bool = False
    require_eps_pass: bool = False
    exclude_stop_risk_proxy: bool = False


@contextmanager
def eps_lookup_mode(enabled: bool):
    if enabled:
        yield
        return
    old = eps_lookup.get_signal_eps
    eps_lookup.get_signal_eps = lambda snapshot_date, code: None
    try:
        yield
    finally:
        eps_lookup.get_signal_eps = old


def qlib_backend_status() -> dict[str, object]:
    try:
        import qlib  # type: ignore
        from qlib.contrib.evaluate import risk_analysis  # noqa: F401
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    qlib.init()
    return {"available": True, "version": getattr(qlib, "__version__", "unknown"), "risk_analysis": True}


def require_qlib_backend() -> dict[str, object]:
    status = qlib_backend_status()
    if not status.get("available"):
        raise RuntimeError(f"Qlib backend is required but unavailable: {status.get('error')}")
    return status


def effective_eps(snapshot_date: str, code: str, raw_value: object, *, eps_enabled: bool) -> float | None:
    if not eps_enabled:
        return None
    value = to_float(raw_value)
    if value is not None:
        return value
    return eps_lookup.get_signal_eps(snapshot_date, code)


def load_signal_rows(
    *,
    eps_enabled: bool,
    pool_root: Path = POOL_ROOT,
    price_cache: str | Path | None = None,
    end_date: str = DEFAULT_END_DATE,
) -> pd.DataFrame:
    resolved_price_cache = resolve_price_cache(price_cache)
    prices = _load_price_cache(resolved_price_cache)
    rows: list[dict[str, object]] = []
    with eps_lookup_mode(eps_enabled):
        for pool_path in sorted(pool_root.glob("*/breakout_follow_pool.csv")):
            snapshot = pool_path.parent.name
            pool = pd.read_csv(pool_path, encoding="utf-8-sig")
            if pool.empty or "signal" not in pool.columns:
                continue
            signal = pool["signal"].astype(str).str.strip().str.lower().isin({"true", "1"})
            for row_index, row in pool[signal].iterrows():
                code = str(row.get("code", "")).strip()
                metrics = compute_path_metrics(
                    code=code,
                    snapshot_date=snapshot,
                    buy_price=to_float(row.get("ibd_candidate_price")),
                    snapshot_close=to_float(row.get("latest_close")),
                    price_bars=prices.get(code),
                    end_date=end_date,
                )
                eps = effective_eps(snapshot, code, row.get("eps_yoy_growth"), eps_enabled=eps_enabled)
                rows.append(
                    {
                        "snapshot_date": snapshot,
                        "code": code,
                        "row_index": row_index,
                        "industry": str(row.get("industry", "") or "").strip(),
                        "entry_status": str(row.get("ibd_entry_status", "") or "").strip().upper(),
                        "ibd_candidate_rule": str(row.get("ibd_candidate_rule", "") or "").strip(),
                        "current_vs_ibd_candidate_pct": to_float(row.get("current_vs_ibd_candidate_pct")),
                        "ibd_entry_volume_ratio": to_float(row.get("ibd_entry_volume_ratio")),
                        "ibd_entry_close_position": to_float(row.get("ibd_entry_close_position")),
                        "ibd_entry_breakout_range_ratio": to_float(row.get("ibd_entry_breakout_range_ratio")),
                        "volume_ratio": to_float(row.get("volume_ratio")),
                        "dist_to_52w_high_pct": to_float(row.get("dist_to_52w_high_pct")),
                        "pullback_v_is_dry": to_bool(row.get("pullback_v_is_dry")),
                        "eps_yoy_growth": eps,
                        "eps_state": "missing" if eps is None else ("pass_25" if eps >= 25 else "known_below_25"),
                        "latest_return_pct": metrics.latest_close_return_pct,
                        "max_gain_pct": metrics.max_gain_pct,
                        "max_drawdown_pct": metrics.max_drawdown_pct,
                        "hit_stop_8pct": metrics.hit_stop_8pct,
                        "path_source": metrics.source,
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return add_weekly_oracle_labels(frame)


def add_weekly_oracle_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["valid_path"] = pd.to_numeric(result["latest_return_pct"], errors="coerce").notna()
    valid = result[result["valid_path"]].copy()
    if valid.empty:
        result["latest_rank"] = pd.NA
        result["gain_rank"] = pd.NA
        result["loss_rank"] = pd.NA
        result["signal_valid_count"] = pd.NA
        return result
    valid = valid.sort_values(["snapshot_date", "latest_return_pct", "max_gain_pct", "code"], ascending=[True, False, False, True])
    valid["latest_rank"] = valid.groupby("snapshot_date").cumcount() + 1
    valid = valid.sort_values(["snapshot_date", "max_gain_pct", "latest_return_pct", "code"], ascending=[True, False, False, True])
    valid["gain_rank"] = valid.groupby("snapshot_date").cumcount() + 1
    valid = valid.sort_values(["snapshot_date", "latest_return_pct", "max_drawdown_pct", "code"], ascending=[True, True, True, True])
    valid["loss_rank"] = valid.groupby("snapshot_date").cumcount() + 1
    valid["signal_valid_count"] = valid.groupby("snapshot_date")["code"].transform("count")
    return result.merge(
        valid[["snapshot_date", "code", "latest_rank", "gain_rank", "loss_rank", "signal_valid_count"]],
        on=["snapshot_date", "code"],
        how="left",
    )


def build_qlib_panel(rows: pd.DataFrame) -> pd.DataFrame:
    frame = rows.copy()
    if frame.empty:
        return pd.DataFrame()
    frame["datetime"] = pd.to_datetime(frame["snapshot_date"])
    frame["instrument"] = frame["code"].astype(str)
    frame["$is_actionable"] = frame["entry_status"].astype(str).str.upper().eq("ACTIONABLE").astype(float)
    frame["$eps_known"] = frame["eps_state"].astype(str).ne("missing").astype(float)
    frame["$eps_pass_25"] = frame["eps_state"].astype(str).eq("pass_25").astype(float)
    frame["$eps_below_25"] = frame["eps_state"].astype(str).eq("known_below_25").astype(float)
    current_vs_candidate = _numeric_column(frame, "current_vs_ibd_candidate_pct")
    frame["$fresh_0_2"] = current_vs_candidate.between(0, 2).astype(float)
    frame["$fresh_0_5"] = current_vs_candidate.between(0, 5).astype(float)
    frame["$entry_volume_ratio"] = _numeric_column(frame, "ibd_entry_volume_ratio").fillna(0.0)
    frame["$weekly_volume_follow"] = _numeric_column(frame, "volume_ratio").ge(1.3).astype(float)
    frame["$near_52w_high"] = _numeric_column(frame, "dist_to_52w_high_pct").gt(-5.0).astype(float)
    frame["$dry_pullback"] = _object_column(frame, "pullback_v_is_dry").map(lambda value: 1.0 if value is True else 0.0)
    frame["$close_position"] = _numeric_column(frame, "ibd_entry_close_position").fillna(0.0)
    frame["$range_ratio"] = _numeric_column(frame, "ibd_entry_breakout_range_ratio").fillna(0.0)
    frame["label_return"] = pd.to_numeric(frame["latest_return_pct"], errors="coerce")
    frame["label_max_gain"] = pd.to_numeric(frame["max_gain_pct"], errors="coerce")
    frame["label_top5_return"] = _numeric_column(frame, "latest_rank").le(5).astype(float)
    frame["label_top5_gain"] = _numeric_column(frame, "gain_rank").le(5).astype(float)
    frame["label_bottom5"] = _numeric_column(frame, "loss_rank").le(5).astype(float)
    frame["label_stop_8pct"] = frame["hit_stop_8pct"].astype(bool).astype(float)
    columns = [
        "industry",
        "entry_status",
        "$is_actionable",
        "$eps_known",
        "$eps_pass_25",
        "$eps_below_25",
        "$fresh_0_2",
        "$fresh_0_5",
        "$entry_volume_ratio",
        "$weekly_volume_follow",
        "$near_52w_high",
        "$dry_pullback",
        "$close_position",
        "$range_ratio",
        "label_return",
        "label_max_gain",
        "label_top5_return",
        "label_top5_gain",
        "label_bottom5",
        "label_stop_8pct",
    ]
    return frame.set_index(["datetime", "instrument"]).sort_index()[columns]


def _object_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([pd.NA] * len(frame), index=frame.index)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(_object_column(frame, column), errors="coerce")


def filter_valid_return_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if panel.empty:
        return panel.copy(), {
            "signal_rows": 0,
            "valid_return_rows": 0,
            "missing_return_rows": 0,
            "signal_weeks": 0,
            "valid_return_weeks": 0,
        }
    valid_return = pd.to_numeric(panel["label_return"], errors="coerce").notna()
    filtered = panel[valid_return].copy()
    return filtered, {
        "signal_rows": int(len(panel)),
        "valid_return_rows": int(valid_return.sum()),
        "missing_return_rows": int((~valid_return).sum()),
        "signal_weeks": int(panel.index.get_level_values("datetime").nunique()),
        "valid_return_weeks": int(filtered.index.get_level_values("datetime").nunique()) if not filtered.empty else 0,
    }


def candidate_rules() -> list[PortfolioRule]:
    return [
        PortfolioRule(
            name="eps_known_balanced",
            weights={
                "$eps_known": 1.2,
                "$eps_pass_25": 0.4,
                "$fresh_0_5": 1.0,
                "$entry_volume_ratio": 0.6,
                "$weekly_volume_follow": 0.8,
                "$near_52w_high": 0.6,
            },
            require_eps_known=True,
        ),
        PortfolioRule(
            name="eps_pass_quality",
            weights={
                "$eps_pass_25": 1.5,
                "$fresh_0_5": 1.0,
                "$entry_volume_ratio": 0.6,
                "$weekly_volume_follow": 0.5,
                "$near_52w_high": 0.5,
            },
            require_eps_pass=True,
        ),
        PortfolioRule(
            name="technical_balanced",
            weights={
                "$fresh_0_5": 1.2,
                "$entry_volume_ratio": 0.8,
                "$weekly_volume_follow": 0.8,
                "$near_52w_high": 0.6,
                "$close_position": 0.4,
            },
        ),
        PortfolioRule(
            name="fresh_demand",
            weights={
                "$fresh_0_2": 1.5,
                "$entry_volume_ratio": 1.0,
                "$weekly_volume_follow": 0.8,
                "$near_52w_high": 0.4,
            },
        ),
        PortfolioRule(
            name="risk_conservative",
            weights={
                "$eps_known": 0.8,
                "$fresh_0_5": 1.0,
                "$entry_volume_ratio": 0.5,
                "$weekly_volume_follow": 0.5,
                "$near_52w_high": 0.5,
            },
            require_eps_known=False,
        ),
    ]


def score_frame(frame: pd.DataFrame, rule: PortfolioRule) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    for column, weight in rule.weights.items():
        _reject_future_label(column, rule)
        score = score + pd.to_numeric(frame.get(column, 0.0), errors="coerce").fillna(0.0) * weight
    for column, penalty in rule.penalties.items():
        _reject_future_label(column, rule)
        score = score - pd.to_numeric(frame.get(column, 0.0), errors="coerce").fillna(0.0) * penalty
    return score


def _reject_future_label(column: str, rule: PortfolioRule) -> None:
    if column.startswith("label_"):
        raise ValueError(f"Rule {rule.name} uses future label column {column} for portfolio scoring")


def select_portfolio(week_panel: pd.DataFrame, rule: PortfolioRule, *, top_k: int = 3) -> pd.DataFrame:
    frame = week_panel.copy()
    if frame.empty:
        return pd.DataFrame()
    if rule.require_actionable:
        frame = frame[frame["$is_actionable"].eq(1.0)]
    if rule.require_eps_known:
        frame = frame[frame["$eps_known"].eq(1.0)]
    if rule.require_eps_pass:
        frame = frame[frame["$eps_pass_25"].eq(1.0)]
    if frame.empty:
        return pd.DataFrame()
    frame["score"] = score_frame(frame, rule)
    frame = frame.reset_index().sort_values(["score", "instrument"], ascending=[False, True])
    selected = []
    covered = set()
    for _, row in frame.iterrows():
        industry_key = str(row.get("industry", "") or "").strip().lower()
        if rule.industry_cap and industry_key and industry_key in covered:
            continue
        selected.append(row)
        if rule.industry_cap and industry_key:
            covered.add(industry_key)
        if len(selected) == top_k:
            break
    return pd.DataFrame(selected)


def evaluate_rule_on_weeks(panel: pd.DataFrame, rule: PortfolioRule, weeks: Iterable[pd.Timestamp], *, top_k: int) -> dict[str, float]:
    rows = []
    for week in weeks:
        try:
            week_panel = panel.xs(week, level="datetime")
        except KeyError:
            continue
        picks = select_portfolio(week_panel, rule, top_k=top_k)
        if picks.empty:
            continue
        rows.append(
            {
                "snapshot_date": week,
                "avg_return": pd.to_numeric(picks["label_return"], errors="coerce").mean(),
                "top5_return": int(pd.to_numeric(picks["label_top5_return"], errors="coerce").sum()),
                "top5_gain": int(pd.to_numeric(picks["label_top5_gain"], errors="coerce").sum()),
                "bottom5": int(pd.to_numeric(picks["label_bottom5"], errors="coerce").sum()),
                "stops": int(pd.to_numeric(picks["label_stop_8pct"], errors="coerce").sum()),
            }
        )
    if not rows:
        return {
            "score": -999.0,
            "weeks": 0,
            "avg_return": 0.0,
            "top5_return_rate": 0.0,
            "top5_gain_rate": 0.0,
            "bottom5_rate": 1.0,
            "stop_rate": 1.0,
        }
    result = pd.DataFrame(rows)
    avg_return = float(result["avg_return"].mean())
    top5_return_rate = float(result["top5_return"].gt(0).mean())
    top5_gain_rate = float(result["top5_gain"].gt(0).mean())
    bottom5_rate = float(result["bottom5"].gt(0).mean())
    stop_rate = float(result["stops"].gt(0).mean())
    score = avg_return / 100.0 + top5_gain_rate * 0.5 + top5_return_rate * 0.25 - bottom5_rate * 1.5 - stop_rate
    return {
        "score": score,
        "weeks": float(len(result)),
        "avg_return": avg_return,
        "top5_return_rate": top5_return_rate,
        "top5_gain_rate": top5_gain_rate,
        "bottom5_rate": bottom5_rate,
        "stop_rate": stop_rate,
    }


def walk_forward_optimize(
    panel: pd.DataFrame,
    rules: list[PortfolioRule],
    *,
    min_train_weeks: int = 8,
    top_k: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel, _ = filter_valid_return_panel(panel)
    weeks = sorted(panel.index.get_level_values("datetime").unique())
    pick_rows = []
    chosen_rows = []
    rule_score_rows = []
    for pos, week in enumerate(weeks):
        if pos < min_train_weeks:
            continue
        train_weeks = weeks[:pos]
        scored = [(rule, evaluate_rule_on_weeks(panel, rule, train_weeks, top_k=top_k)) for rule in rules]
        for rule, train_metrics in scored:
            rule_score_rows.append(
                {
                    "snapshot_date": str(week.date()),
                    "candidate_rule": rule.name,
                    "train_score": train_metrics["score"],
                    "train_weeks": int(train_metrics["weeks"]),
                    "train_avg_return_pct": train_metrics["avg_return"],
                    "train_top5_return_week_rate": train_metrics["top5_return_rate"],
                    "train_top5_gain_week_rate": train_metrics["top5_gain_rate"],
                    "train_bottom5_week_rate": train_metrics["bottom5_rate"],
                    "train_stop_week_rate": train_metrics["stop_rate"],
                }
            )
        scored.sort(key=lambda item: (item[1]["score"], item[1]["avg_return"], item[0].name), reverse=True)
        best_rule, train_metrics = scored[0]
        week_panel = panel.xs(week, level="datetime")
        picks = select_portfolio(week_panel, best_rule, top_k=top_k)
        chosen_rows.append(
                {
                    "snapshot_date": str(week.date()),
                    "selected_rule": best_rule.name,
                    "train_score": train_metrics["score"],
                    "train_weeks": int(train_metrics["weeks"]),
                    "train_avg_return_pct": train_metrics["avg_return"],
                    "train_top5_return_week_rate": train_metrics["top5_return_rate"],
                    "train_top5_gain_week_rate": train_metrics["top5_gain_rate"],
                    "train_bottom5_week_rate": train_metrics["bottom5_rate"],
                    "train_stop_week_rate": train_metrics["stop_rate"],
                    "pick_count": len(picks),
                }
            )
        for order, (_, row) in enumerate(picks.iterrows(), 1):
            pick_rows.append(
                {
                    "snapshot_date": str(week.date()),
                    "selected_rule": best_rule.name,
                    "pick_order": order,
                    "code": row["instrument"],
                    "industry": row.get("industry", ""),
                    "score": row.get("score"),
                    "latest_return_pct": row.get("label_return"),
                    "max_gain_pct": row.get("label_max_gain"),
                    "top5_return": bool(row.get("label_top5_return")),
                    "top5_gain": bool(row.get("label_top5_gain")),
                    "bottom5": bool(row.get("label_bottom5")),
                    "hit_stop_8pct": bool(row.get("label_stop_8pct")),
                }
            )
    picks = pd.DataFrame(pick_rows)
    choices = pd.DataFrame(chosen_rows)
    rule_scores = pd.DataFrame(rule_score_rows)
    summary = summarize_strategy("walk_forward_best_rule", picks, choices)
    return picks, summary, choices, rule_scores


def qlib_risk_metrics(picks: pd.DataFrame) -> pd.DataFrame:
    if picks.empty:
        return pd.DataFrame()
    from qlib.contrib.evaluate import risk_analysis

    weekly_returns = (
        picks.groupby("snapshot_date")["latest_return_pct"]
        .mean()
        .dropna()
        .sort_index()
        .pipe(lambda series: pd.Series(series.to_numpy() / 100.0, index=pd.to_datetime(series.index)))
    )
    if weekly_returns.empty:
        return pd.DataFrame()
    analysis = risk_analysis(weekly_returns, freq="week")
    if isinstance(analysis, pd.Series):
        frame = analysis.rename("value").reset_index()
    else:
        frame = analysis.reset_index()
    return frame


def summarize_strategy(name: str, picks: pd.DataFrame, choices: pd.DataFrame | None = None) -> pd.DataFrame:
    if picks.empty:
        return pd.DataFrame(
            [{"strategy": name, "weeks": 0, "picks": 0, "avg_return_pct": 0.0, "median_week_return_pct": 0.0}]
        )
    weekly = picks.groupby("snapshot_date").agg(
        pick_count=("code", "count"),
        avg_return_pct=("latest_return_pct", "mean"),
        worst_return_pct=("latest_return_pct", "min"),
        top5_return_count=("top5_return", "sum"),
        top5_gain_count=("top5_gain", "sum"),
        bottom5_count=("bottom5", "sum"),
        stop_count=("hit_stop_8pct", "sum"),
    )
    row = {
        "strategy": name,
        "weeks": int(weekly.shape[0]),
        "picks": int(len(picks)),
        "avg_return_pct": float(weekly["avg_return_pct"].mean()),
        "median_week_return_pct": float(weekly["avg_return_pct"].median()),
        "median_worst_pick_return_pct": float(weekly["worst_return_pct"].median()),
        "top5_return_week_rate": float(weekly["top5_return_count"].gt(0).mean()),
        "top5_gain_week_rate": float(weekly["top5_gain_count"].gt(0).mean()),
        "bottom5_week_rate": float(weekly["bottom5_count"].gt(0).mean()),
        "stop_week_rate": float(weekly["stop_count"].gt(0).mean()),
        "unique_rules": int(choices["selected_rule"].nunique()) if choices is not None and not choices.empty else 0,
    }
    return pd.DataFrame([row])


def run_mode(
    eps_mode: str,
    *,
    output_dir: Path,
    top_k: int,
    min_train_weeks: int,
    price_cache: Path,
) -> dict[str, object]:
    eps_enabled = eps_mode == "with_eps"
    rows = load_signal_rows(eps_enabled=eps_enabled, price_cache=price_cache)
    panel = build_qlib_panel(rows)
    eval_panel, coverage = filter_valid_return_panel(panel)
    coverage["pool_file_weeks"] = len(list(POOL_ROOT.glob("*/breakout_follow_pool.csv")))
    coverage["training_window_weeks"] = min_train_weeks
    panel_path = output_dir / f"{eps_mode}_qlib_panel.csv"
    panel.reset_index().to_csv(panel_path, index=False)
    picks, summary, choices, rule_scores = walk_forward_optimize(
        eval_panel,
        candidate_rules(),
        min_train_weeks=min_train_weeks,
        top_k=top_k,
    )
    picks_path = output_dir / f"{eps_mode}_walk_forward_picks.csv"
    choices_path = output_dir / f"{eps_mode}_walk_forward_choices.csv"
    rule_scores_path = output_dir / f"{eps_mode}_walk_forward_rule_scores.csv"
    summary_path = output_dir / f"{eps_mode}_walk_forward_summary.csv"
    risk_path = output_dir / f"{eps_mode}_qlib_risk_analysis.csv"
    picks.to_csv(picks_path, index=False)
    choices.to_csv(choices_path, index=False)
    rule_scores.to_csv(rule_scores_path, index=False)
    risk = qlib_risk_metrics(picks)
    risk.to_csv(risk_path, index=False)
    summary.insert(0, "eps_mode", eps_mode)
    summary.to_csv(summary_path, index=False)
    return {
        "eps_mode": eps_mode,
        "panel_path": str(panel_path),
        "picks_path": str(picks_path),
        "choices_path": str(choices_path),
        "rule_scores_path": str(rule_scores_path),
        "summary_path": str(summary_path),
        "risk_path": str(risk_path),
        "coverage": coverage,
        "price_cache": str(price_cache),
        "summary": summary.iloc[0].to_dict() if not summary.empty else {},
    }


def render_comparison(results: list[dict[str, object]], backend: dict[str, object]) -> str:
    rows = [result["summary"] for result in results]
    summary = pd.DataFrame(rows)
    file_rows = pd.DataFrame(
        [
            {
                "eps_mode": result["eps_mode"],
                "panel": result["panel_path"],
                "weekly_choices": result["choices_path"],
                "weekly_rule_scores": result["rule_scores_path"],
                "weekly_picks": result["picks_path"],
                "risk": result["risk_path"],
            }
            for result in results
        ]
    )
    coverage_rows = pd.DataFrame([{"eps_mode": result["eps_mode"], **result["coverage"]} for result in results])
    rules = pd.DataFrame(
        [
            {
                "rule": rule.name,
                "requires_eps_known": rule.require_eps_known,
                "requires_eps_pass": rule.require_eps_pass,
                "features": ", ".join(rule.weights),
            }
            for rule in candidate_rules()
        ]
    )
    total_pool_weeks = coverage_rows["pool_file_weeks"].max() if not coverage_rows.empty else 0
    lines = [
        "# Qlib-Compatible Replay Pool Optimization",
        "",
        f"- Replay pool files: `{int(total_pool_weeks)}`",
        f"- Price cache: `{results[0].get('price_cache', 'unknown') if results else 'unknown'}`",
        f"- Qlib backend available: `{backend.get('available')}`",
        f"- Qlib version: `{backend.get('version', 'unknown')}`",
    ]
    lines.extend(
        [
            "- Data modes: no-EPS and with-EPS are optimized independently, then compared.",
            "- Walk-forward: each evaluation week selects the best rule using only prior weeks.",
            "- Leakage guard: portfolio scoring rejects any `label_` column; labels are only used after weekly picks are fixed.",
            "- Qlib usage: `qlib.init()` is required, and `qlib.contrib.evaluate.risk_analysis` computes risk metrics for each optimized weekly return series. If Qlib is unavailable this script stops instead of falling back.",
            "",
            "## Execution Steps",
            "",
            "1. Load every weekly `breakout_follow_pool.csv` under `backtest/ibd_skill_replay_pools/`.",
            "2. Build two datasets from the same replay pool history: `no_eps` disables EPS lookup, `with_eps` uses the supplemental point-in-time EPS lookup and assumes it is correct.",
            "3. Convert each dataset into a Qlib-style `datetime` / `instrument` panel with signal-time feature columns and future outcome label columns.",
            "4. For each evaluation week after the minimum training window, score every candidate rule on prior weeks only, then apply the best historical rule to that week.",
            "5. Evaluate that week's selected portfolio by realized return, top5 return hit, top5 gain hit, bottom5 loss hit, and 8% stop hit.",
            "6. Run Qlib `risk_analysis(freq=\"week\")` on the optimized weekly return series.",
            "",
            "## Data Coverage",
            "",
            "Rows without realized return are excluded from walk-forward optimization and portfolio statistics; they remain in the full panel for audit.",
            "",
        ]
    )
    lines.extend(coverage_rows.to_markdown(index=False).splitlines())
    lines.extend(
        [
            "",
            "## Summary",
            "",
        ]
    )
    lines.extend(summary.to_markdown(index=False).splitlines())
    lines.extend(["", "## Rule Space", ""])
    lines.extend(rules.to_markdown(index=False).splitlines())
    lines.extend(["", "## Output Files", ""])
    lines.extend(file_rows.to_markdown(index=False).splitlines())
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qlib-compatible optimizer for replay IBD signal pools.")
    parser.add_argument("--eps-mode", choices=["no_eps", "with_eps", "both"], default="both")
    parser.add_argument("--output-dir", default="backtest/ibd_weekly_signal_oracle_eval/qlib_optimizer")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-train-weeks", type=int, default=8)
    parser.add_argument("--price-cache", default=None)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = ["no_eps", "with_eps"] if args.eps_mode == "both" else [args.eps_mode]
    price_cache = resolve_price_cache(args.price_cache)
    backend = require_qlib_backend()
    results = [
        run_mode(mode, output_dir=output_dir, top_k=args.top_k, min_train_weeks=args.min_train_weeks, price_cache=price_cache)
        for mode in modes
    ]
    manifest = {
        "backend": backend,
        "eps_modes": modes,
        "top_k": args.top_k,
        "min_train_weeks": args.min_train_weeks,
        "price_cache": str(price_cache),
        "results": results,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "comparison_report.md").write_text(render_comparison(results, backend), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
