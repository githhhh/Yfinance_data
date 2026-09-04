"""Executable-entry and causal outcome semantics for blind discovery."""
from __future__ import annotations

from dataclasses import dataclass
import math
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

TRADING_HORIZONS = {"4w": 20, "8w": 40, "12w": 60}

@dataclass(frozen=True)
class OutcomeConfig:
    stop_loss: float = -0.08
    winner_gain: float = 0.20
    minimum_sessions: int = 60
    entry_window_sessions: int = 5
    max_entry_extension: float = 0.05

def _normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(df.columns):
        raise ValueError(f"price frame missing columns: {sorted(required - set(df.columns))}")
    out = df.loc[:, ["Open", "High", "Low", "Close"]].copy()
    if "date" in df.columns:
        idx = pd.DatetimeIndex(pd.to_datetime(df["date"], errors="coerce"))
    else:
        idx = pd.DatetimeIndex(pd.to_datetime(df.index, errors="coerce"))
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    out["date"] = idx.normalize()
    for col in ["Open", "High", "Low", "Close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    adjustment_mode = "raw_no_adj_close"
    out["_adj_factor"] = 1.0
    if "Adj Close" in df.columns:
        adj = pd.to_numeric(df["Adj Close"], errors="coerce").reset_index(drop=True)
        raw_close = pd.to_numeric(df["Close"], errors="coerce").reset_index(drop=True)
        factor = adj / raw_close.where(raw_close != 0)
        # Preserve a continuous split/dividend-adjusted OHLC path while keeping
        # the source download itself unadjusted.
        factor_values = factor.to_numpy()
        for col in ["Open", "High", "Low", "Close"]:
            out[col] = pd.to_numeric(out[col], errors="coerce").to_numpy() * factor_values
        out["_adj_factor"] = factor_values
        adjustment_mode = "adj_close_factor"

    out = out.dropna(subset=["date", "Open", "High", "Low", "Close"]).sort_values("date").reset_index(drop=True)
    out.attrs["price_adjustment_mode"] = adjustment_mode
    return out

def load_price_pickle(path: Path, *, require_adjusted: bool = False) -> dict[str, pd.DataFrame]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    out: dict[str, pd.DataFrame] = {}
    unadjusted: list[str] = []
    for code, value in raw.items():
        if isinstance(value, dict) and {"index", "columns", "data"}.issubset(value):
            value = pd.DataFrame(index=value["index"], columns=value["columns"], data=value["data"])
        if isinstance(value, pd.DataFrame):
            normalized = _normalize_price_frame(value)
            out[str(code)] = normalized
            if normalized.attrs.get("price_adjustment_mode") != "adj_close_factor":
                unadjusted.append(str(code))
    if require_adjusted and unadjusted:
        sample = ",".join(unadjusted[:10])
        raise ValueError(
            f"outcome source lacks Adj Close for {len(unadjusted)} symbols ({sample}); "
            "refuse split-unsafe raw OHLC outcomes"
        )
    return out

def restrict_to_mature_outcome_quarters(
    candidates: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    *,
    minimum_sessions: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Exclude partial quarters whose quarter-end signals lack a full outcome window."""
    spy = benchmark_prices if "date" in benchmark_prices.columns else _normalize_price_frame(benchmark_prices)
    if len(spy) < minimum_sessions + 1:
        raise ValueError("benchmark history is too short to establish outcome maturity")
    maturity_cutoff = pd.Timestamp(spy.iloc[-(minimum_sessions + 1)]["date"]).normalize()
    quarter_end = pd.to_datetime(candidates["snapshot_date"], errors="coerce").dt.to_period("Q").dt.end_time.dt.normalize()
    mature_mask = quarter_end <= maturity_cutoff
    mature = candidates.loc[mature_mask].reset_index(drop=True)
    excluded = candidates.loc[~mature_mask].reset_index(drop=True)
    if mature.empty:
        raise ValueError("no fully mature outcome quarters remain")
    return mature, excluded, maturity_cutoff

def _first_event_date(window: pd.DataFrame, *, threshold: float, entry_price: float, side: str) -> pd.Timestamp | None:
    if side == "up":
        hits = window.loc[window["High"] >= entry_price * (1.0 + threshold), "date"]
    elif side == "down":
        hits = window.loc[window["Low"] <= entry_price * (1.0 + threshold), "date"]
    else:
        raise ValueError(side)
    return None if hits.empty else pd.Timestamp(hits.iloc[0])

def _benchmark_return(spy: pd.DataFrame, entry_date: pd.Timestamp, exit_date: pd.Timestamp) -> float | None:
    entry_rows = spy.loc[spy["date"] == entry_date]
    exit_rows = spy.loc[spy["date"] <= exit_date]
    if entry_rows.empty or exit_rows.empty:
        return None
    entry = float(entry_rows.iloc[0]["Open"])
    exit_close = float(exit_rows.iloc[-1]["Close"])
    return exit_close / entry - 1.0 if entry > 0 else None

def _adjusted_trigger_price(px: pd.DataFrame, signal_date: pd.Timestamp, trigger_price: float) -> float:
    history = px.loc[px["date"] <= signal_date]
    if "_adj_factor" in px.columns and not history.empty:
        factor = float(history.iloc[-1]["_adj_factor"])
    else:
        factor = 1.0
    return trigger_price * factor

def _resolve_executable_entry(
    px: pd.DataFrame,
    signal_date: pd.Timestamp,
    trigger_price: float,
    config: OutcomeConfig,
) -> dict[str, Any]:
    if not math.isfinite(trigger_price) or trigger_price <= 0:
        return {"reason": "missing_trigger_price"}
    trigger = _adjusted_trigger_price(px, signal_date, trigger_price)
    post = px.loc[px["date"] > signal_date].reset_index(drop=True)
    if post.empty:
        return {"reason": "no_executable_entry"}
    for i, bar in post.iloc[: config.entry_window_sessions].iterrows():
        open_px = float(bar["Open"])
        high_px = float(bar["High"])
        max_buy = trigger * (1.0 + config.max_entry_extension)
        if open_px >= trigger:
            if open_px <= max_buy:
                return {
                    "entry_index": int(i),
                    "entry_date": pd.Timestamp(bar["date"]),
                    "entry_price": open_px,
                    "entry_method": "gap_or_open",
                    "trigger_price_adjusted": trigger,
                }
            continue
        if high_px >= trigger:
            return {
                "entry_index": int(i),
                "entry_date": pd.Timestamp(bar["date"]),
                "entry_price": trigger,
                "entry_method": "intraday_trigger",
                "trigger_price_adjusted": trigger,
            }
    return {"reason": "no_entry_within_buy_zone_window"}

def evaluate_candidate_path(
    prices: pd.DataFrame,
    signal_date: str | pd.Timestamp,
    *,
    trigger_price: float | None = None,
    spy_prices: pd.DataFrame | None = None,
    config: OutcomeConfig = OutcomeConfig(),
) -> dict[str, Any]:
    """Resolve a causal buy point, then apply first-passage outcome semantics."""
    px = prices if "date" in prices.columns else _normalize_price_frame(prices)
    sig = pd.Timestamp(signal_date).tz_localize(None).normalize()
    try:
        raw_trigger = float(trigger_price) if trigger_price is not None else float("nan")
    except (TypeError, ValueError):
        raw_trigger = float("nan")
    entry = _resolve_executable_entry(px, sig, raw_trigger, config)
    if "entry_date" not in entry:
        return {"label": "censored", "primary": "censored", "reason": entry["reason"]}
    entry_date = pd.Timestamp(entry["entry_date"])
    entry_price = float(entry["entry_price"])
    post = px.loc[px["date"] >= entry_date].reset_index(drop=True)
    window = post.iloc[: config.minimum_sessions].copy()
    if len(window) < config.minimum_sessions or entry_price <= 0:
        return {
            "label": "censored",
            "primary": "censored",
            "reason": "insufficient_future_sessions",
            "entry_date": entry_date,
            "entry_price": entry_price,
            **entry,
        }

    target_date = _first_event_date(window, threshold=config.winner_gain, entry_price=entry_price, side="up")
    stop_date = _first_event_date(window, threshold=config.stop_loss, entry_price=entry_price, side="down")
    first_bar = window.iloc[0]
    entry_day_order_unknown = (
        entry["entry_method"] == "intraday_trigger"
        and float(first_bar["Low"]) <= entry_price * (1.0 + config.stop_loss)
    )

    if entry_day_order_unknown:
        label, primary, reason = "ambiguous_path", "ambiguous", "entry_day_stop_order_unknown"
    elif target_date is not None and stop_date is not None and target_date == stop_date:
        label, primary, reason = "ambiguous_path", "ambiguous", "same_bar_target_stop_order_unknown"
    elif target_date is not None and (stop_date is None or target_date < stop_date):
        label, primary, reason = "clean_winner", "winner", ""
    elif stop_date is not None and target_date is not None and stop_date < target_date:
        label, primary, reason = "stop_out_then_winner", "loser", ""
    elif stop_date is not None:
        label, primary, reason = "stopped_out_loser", "loser", ""
    else:
        label, primary, reason = "unresolved", "unresolved", ""

    result: dict[str, Any] = {
        "label": label,
        "primary": primary,
        "reason": reason,
        "entry_date": entry_date,
        "entry_price": entry_price,
        "entry_method": entry["entry_method"],
        "trigger_price_adjusted": entry["trigger_price_adjusted"],
        "target_date": target_date,
        "stop_date": stop_date,
        "stopped_out": stop_date is not None,
        "recovered_after_stop": label == "stop_out_then_winner",
        "mae_12w": float(window["Low"].min() / entry_price - 1.0),
        "mfe_12w": float(window["High"].max() / entry_price - 1.0),
    }
    spy = None
    if spy_prices is not None:
        spy = spy_prices if "date" in spy_prices.columns else _normalize_price_frame(spy_prices)
    for name, sessions in TRADING_HORIZONS.items():
        row = window.iloc[sessions - 1]
        exit_date = pd.Timestamp(row["date"])
        stock_return = float(row["Close"] / entry_price - 1.0)
        result[f"return_{name}"] = stock_return
        result[f"exit_date_{name}"] = exit_date
        benchmark = _benchmark_return(spy, entry_date, exit_date) if spy is not None else None
        result[f"benchmark_return_{name}"] = benchmark
        result[f"excess_{name}"] = None if benchmark is None else stock_return - benchmark
    return result

def _max_drawdown(close: pd.Series) -> float | None:
    values = pd.to_numeric(close, errors="coerce").dropna()
    if values.empty:
        return None
    running_peak = values.cummax()
    return float((values / running_peak - 1.0).min())

def point_in_time_market_features(spy_prices: pd.DataFrame, signal_date: str | pd.Timestamp) -> dict[str, Any]:
    """Broad-market features known by the signal close; never use future period data."""
    spy = spy_prices if "date" in spy_prices.columns else _normalize_price_frame(spy_prices)
    sig = pd.Timestamp(signal_date).tz_localize(None).normalize()
    hist = spy.loc[spy["date"] <= sig].reset_index(drop=True)
    out: dict[str, Any] = {}
    for name, sessions in TRADING_HORIZONS.items():
        if len(hist) >= sessions + 1:
            cur = hist.iloc[-(sessions + 1):]
            out[f"M_{name}_return"] = float(cur.iloc[-1]["Close"] / cur.iloc[0]["Close"] - 1.0)
            out[f"M_{name}_drawdown"] = _max_drawdown(cur["Close"])
        else:
            out[f"M_{name}_return"] = None
            out[f"M_{name}_drawdown"] = None
    trailing_52w = hist.iloc[-252:]
    if not trailing_52w.empty:
        high = float(trailing_52w["High"].max())
        close = float(trailing_52w.iloc[-1]["Close"])
        out["M_dist_52w_high"] = (close / high - 1.0) if high > 0 else None
    else:
        out["M_dist_52w_high"] = None
    return out
