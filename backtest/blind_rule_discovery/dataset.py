"""Blind candidate surface, feature dossier, and purged time split."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .outcomes import (
    OutcomeConfig,
    _max_drawdown,
    _normalize_price_frame,
    evaluate_candidate_path,
    point_in_time_market_features,
)

DISCOVERY_FEATURE_ALLOWLIST = (
    "pullback_v_is_dry",
    "ibd_entry_volume_ratio",
    "ibd_entry_close_vs_trigger_pct",
    "ibd_entry_close_position",
    "ibd_entry_breakout_range_ratio",
    "current_vs_ibd_candidate_pct",
    "volume_ratio",
    "pct_above_ceiling",
    "touched_ema10_count",
    "mbox_count",
    "base_depth_pct",
    "base_mbox_count",
    "base_duration_weeks",
    "pullback_count",
    "pullback_duration_weeks",
    "pullback_pct",
    "pullback_pct_off_peak",
    "eps_yoy_growth",
    "dist_to_52w_high_pct",
)
BOOLEAN_DISCOVERY_FEATURES = {"pullback_v_is_dry"}
AGENT_OUTCOME_COLUMNS = {
    "sample_id", "period_month", "period_quarter", "Y_label", "Y_primary",
    "Y_stopped_out", "Y_recovered_after_stop", "Y_4w_return", "Y_8w_return",
    "Y_12w_return", "Y_4w_excess", "Y_8w_excess", "Y_12w_excess",
    "Y_mae_12w", "Y_mfe_12w",
}

def load_replay_candidates(replay_root: Path, *, signal_only: bool = True) -> pd.DataFrame:
    """Load every upstream signal candidate; never apply downstream selection."""
    frames: list[pd.DataFrame] = []
    for path in sorted(replay_root.glob("*/breakout_follow_pool.csv")):
        frame = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig")
        if "snapshot_date" not in frame.columns:
            frame["snapshot_date"] = path.parent.name
        frame["_source_pool"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no replay pools under {replay_root}")
    out = pd.concat(frames, ignore_index=True, sort=False)
    if signal_only and "signal" in out.columns:
        signal = out["signal"].astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes"})
        out = out.loc[signal].copy()
    if "code" not in out.columns or "snapshot_date" not in out.columns:
        raise ValueError("replay candidates require code and snapshot_date")
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["snapshot_date", "code"])
    return out.drop_duplicates(subset=["snapshot_date", "code"], keep="last").reset_index(drop=True)

def _bool_like_value(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes"}:
        return 1
    if normalized in {"0", "false", "f", "no"}:
        return 0
    return pd.NA

def _feature_value(name: str, value: Any) -> Any:
    if name in BOOLEAN_DISCOVERY_FEATURES:
        return _bool_like_value(value)
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

def assert_agent_surface_is_blind(columns: Iterable[str]) -> None:
    for name in columns:
        if name in AGENT_OUTCOME_COLUMNS or name.startswith("X") or name.startswith("M_"):
            continue
        raise ValueError(f"unexpected agent-facing column: {name}")

def build_blind_dataset(
    candidates: pd.DataFrame,
    price_map: Mapping[str, pd.DataFrame],
    spy_prices: pd.DataFrame,
    *,
    config: OutcomeConfig = OutcomeConfig(),
) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    """Return agent-facing anonymous data, private feature map, and reviewer rows."""
    feature_names = sorted(name for name in DISCOVERY_FEATURE_ALLOWLIST if name in candidates.columns)
    if not feature_names:
        raise ValueError("none of the explicit discovery features are present")
    aliases = {name: f"X{i:03d}" for i, name in enumerate(feature_names, start=1)}
    agent_rows: list[dict[str, Any]] = []
    reviewer_rows: list[dict[str, Any]] = []

    # Never preserve pool order: it may itself contain a legacy ranking signal.
    ordered = candidates.reset_index(drop=True).copy()
    ordered["_blind_sort_key"] = [
        hashlib.sha256(f"blind-v1|{row.code}|{row.snapshot_date}".encode()).hexdigest()
        for row in ordered[["code", "snapshot_date"]].itertuples(index=False)
    ]
    ordered = ordered.sort_values("_blind_sort_key").reset_index(drop=True)

    for seq, row in ordered.iterrows():
        code = str(row["code"])
        sample_id = f"S{seq + 1:06d}"
        prices = price_map.get(code)
        if prices is None or prices.empty:
            reviewer_rows.append({
                "sample_id": sample_id,
                "code": code,
                "signal_date": pd.Timestamp(row["snapshot_date"]),
                "label": "censored",
                "primary": "censored",
                "reason": "missing_price_data",
                "usable": False,
            })
            continue
        trigger = pd.to_numeric(pd.Series([row.get("ibd_trigger_price")]), errors="coerce").iloc[0]
        if pd.isna(trigger):
            trigger = pd.to_numeric(pd.Series([row.get("ibd_candidate_price")]), errors="coerce").iloc[0]
        outcome = evaluate_candidate_path(
            prices, row["snapshot_date"], trigger_price=trigger, spy_prices=spy_prices, config=config
        )
        if outcome.get("label") == "censored":
            reviewer_rows.append({
                "sample_id": sample_id,
                "code": code,
                "signal_date": pd.Timestamp(row["snapshot_date"]),
                **outcome,
                "usable": False,
            })
            continue
        entry_date = pd.Timestamp(outcome["entry_date"])
        agent: dict[str, Any] = {
            "sample_id": sample_id,
            "period_month": entry_date.strftime("%Y-%m"),
            "period_quarter": f"{entry_date.year}Q{entry_date.quarter}",
            "Y_label": outcome["label"],
            "Y_primary": outcome["primary"],
            "Y_stopped_out": int(bool(outcome["stopped_out"])),
            "Y_recovered_after_stop": int(bool(outcome["recovered_after_stop"])),
            "Y_4w_return": outcome["return_4w"],
            "Y_8w_return": outcome["return_8w"],
            "Y_12w_return": outcome["return_12w"],
            "Y_4w_excess": outcome["excess_4w"],
            "Y_8w_excess": outcome["excess_8w"],
            "Y_12w_excess": outcome["excess_12w"],
            "Y_mae_12w": outcome["mae_12w"],
            "Y_mfe_12w": outcome["mfe_12w"],
            **point_in_time_market_features(spy_prices, row["snapshot_date"]),
        }
        for original, alias in aliases.items():
            agent[alias] = _feature_value(original, row[original])
        agent_rows.append(agent)
        reviewer_rows.append(
            {
                "sample_id": sample_id,
                "code": code,
                "signal_date": pd.Timestamp(row["snapshot_date"]),
                **outcome,
                "usable": True,
            }
        )

    agent_df = pd.DataFrame(agent_rows)
    assert_agent_surface_is_blind(agent_df.columns)
    return agent_df, {alias: original for original, alias in aliases.items()}, pd.DataFrame(reviewer_rows)

def _period_key(ts: pd.Series, granularity: str) -> pd.Series:
    if granularity == "month":
        return ts.dt.strftime("%Y-%m")
    if granularity == "quarter":
        return ts.dt.to_period("Q").astype(str)
    raise ValueError(granularity)

def build_reviewer_market_context(reviewer_rows: pd.DataFrame, spy_prices: pd.DataFrame) -> pd.DataFrame:
    """Post-hoc month/quarter benchmark context; never copied to agent workspace."""
    if reviewer_rows.empty:
        return pd.DataFrame()
    spy = spy_prices if "date" in spy_prices.columns else _normalize_price_frame(spy_prices)
    entries = pd.to_datetime(reviewer_rows["entry_date"], errors="coerce").dropna()
    rows: list[dict[str, Any]] = []
    for granularity in ("month", "quarter"):
        periods = sorted(_period_key(entries.to_frame("d")["d"], granularity).unique())
        spy_period = _period_key(spy["date"], granularity)
        for period in periods:
            cur = spy.loc[spy_period == period]
            if cur.empty:
                continue
            rows.append(
                {
                    "granularity": granularity,
                    "period": period,
                    "spy_period_return": float(cur.iloc[-1]["Close"] / cur.iloc[0]["Open"] - 1.0),
                    "spy_period_max_drawdown": _max_drawdown(cur["Close"]),
                    "spy_sessions": int(len(cur)),
                }
            )
    return pd.DataFrame(rows)

def build_feature_dossier(agent_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Month/quarter distribution profiles over discovery data only; no averages."""
    if agent_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    feature_cols = [c for c in agent_df.columns if c.startswith("X") or c.startswith("M_")]
    summary_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for granularity, period_col in (("month", "period_month"), ("quarter", "period_quarter")):
        for period, period_df in agent_df.groupby(period_col, dropna=False):
            total = int(len(period_df))
            label_counts = period_df["Y_label"].value_counts(dropna=False).to_dict()
            primary_counts = period_df["Y_primary"].value_counts(dropna=False).to_dict()
            resolved = int(primary_counts.get("winner", 0)) + int(primary_counts.get("loser", 0))
            base = {
                "granularity": granularity,
                "period": str(period),
                "sample_n": total,
                "clean_winner_n": int(label_counts.get("clean_winner", 0)),
                "stopped_out_loser_n": int(label_counts.get("stopped_out_loser", 0)),
                "stop_out_then_winner_n": int(label_counts.get("stop_out_then_winner", 0)),
                "unresolved_n": int(label_counts.get("unresolved", 0)),
                "ambiguous_path_n": int(label_counts.get("ambiguous_path", 0)),
                "resolved_n": resolved,
                "resolved_winner_rate": (int(primary_counts.get("winner", 0)) / resolved) if resolved else None,
            }
            for target in ("Y_4w_excess", "Y_8w_excess", "Y_12w_excess", "Y_mae_12w", "Y_mfe_12w"):
                numeric = pd.to_numeric(period_df[target], errors="coerce").dropna()
                base[f"{target}_p25"] = numeric.quantile(0.25) if not numeric.empty else None
                base[f"{target}_p50"] = numeric.quantile(0.50) if not numeric.empty else None
                base[f"{target}_p75"] = numeric.quantile(0.75) if not numeric.empty else None
            summary_rows.append(base)

            for label, label_df in period_df.groupby("Y_primary", dropna=False):
                for feature in feature_cols:
                    numeric = pd.to_numeric(label_df[feature], errors="coerce").dropna()
                    feature_rows.append(
                        {
                            "granularity": granularity,
                            "period": str(period),
                            "primary_label": str(label),
                            "feature": feature,
                            "sample_n": int(len(label_df)),
                            "observed_n": int(len(numeric)),
                            "missing_n": int(len(label_df) - len(numeric)),
                            "p10": numeric.quantile(0.10) if not numeric.empty else None,
                            "p25": numeric.quantile(0.25) if not numeric.empty else None,
                            "p50": numeric.quantile(0.50) if not numeric.empty else None,
                            "p75": numeric.quantile(0.75) if not numeric.empty else None,
                            "p90": numeric.quantile(0.90) if not numeric.empty else None,
                        }
                    )
    return pd.DataFrame(summary_rows), pd.DataFrame(feature_rows)

def purged_chronological_holdout(
    agent_df: pd.DataFrame,
    reviewer_rows: pd.DataFrame,
    *,
    holdout_quarters: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], pd.Timestamp]:
    """Seal latest quarters and purge discovery rows whose 12w label crosses holdout."""
    if holdout_quarters < 1:
        raise ValueError("holdout_quarters must be >= 1")
    quarters = sorted(agent_df["period_quarter"].dropna().astype(str).unique())
    if len(quarters) <= holdout_quarters:
        raise ValueError("not enough quarters to create chronological holdout")
    sealed = quarters[-holdout_quarters:]
    first = pd.Period(sealed[0], freq="Q")
    holdout_start = first.start_time.normalize()

    holdout_mask = agent_df["period_quarter"].astype(str).isin(sealed)
    holdout = agent_df.loc[holdout_mask].reset_index(drop=True)
    pre_holdout = agent_df.loc[~holdout_mask].copy()

    exits = reviewer_rows.set_index("sample_id")["exit_date_12w"]
    pre_exit = pd.to_datetime(pre_holdout["sample_id"].map(exits), errors="coerce")
    embargo_mask = pre_exit >= holdout_start
    embargo = pre_holdout.loc[embargo_mask].reset_index(drop=True)
    discovery = pre_holdout.loc[~embargo_mask].reset_index(drop=True)
    if discovery.empty or holdout.empty:
        raise ValueError("purged chronological split produced an empty partition")
    return discovery, embargo, holdout, sealed, holdout_start


# Backwards-compatible reviewer-only diagnostic name.
build_market_context = build_reviewer_market_context
