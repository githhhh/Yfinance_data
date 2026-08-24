from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .labels import ExitPolicy
from .utils import breakout_week_friday, fmt_date, next_bar_after, normalize_bars, parse_date, pct, to_float


@dataclass(frozen=True)
class PortfolioConfig:
    capacity: int = 3
    initial_capital: float = 100_000.0
    cost_bps_per_side: float = 0.0
    exit_policy: ExitPolicy = ExitPolicy()


def run_portfolio_backtest(
    picks: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    config: PortfolioConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if picks.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    normalized = {code: normalize_bars(frame) for code, frame in prices.items()}
    ordered = picks.copy()
    ordered["snapshot_date"] = pd.to_datetime(ordered["snapshot_date"])
    if "pick_order" not in ordered.columns:
        ordered["pick_order"] = 1
    ordered = ordered.sort_values(["snapshot_date", "pick_order", "code"])

    cash = float(config.initial_capital)
    slot_capital = config.initial_capital / max(config.capacity, 1)
    trades: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    all_dates = _all_price_dates(normalized)

    for _, pick in ordered.iterrows():
        signal_date = pd.Timestamp(pick["snapshot_date"]).tz_localize(None)
        code = str(pick.get("code", "") or "").strip()
        _release_closed(active, signal_date)
        if _has_active(active, code, signal_date):
            events.append(_event(signal_date, code, "repeat_signal_ignored"))
            continue
        if len([p for p in active if not p["exit_date"] or pd.Timestamp(p["exit_date"]) > signal_date]) >= config.capacity:
            events.append(_event(signal_date, code, "capacity_skip"))
            continue
        bars = normalized.get(code, pd.DataFrame())
        entry_bar = next_bar_after(bars, signal_date)
        entry_open = to_float(entry_bar[1].get("Open")) if entry_bar is not None else None
        if entry_bar is None or entry_open is None:
            events.append(_event(signal_date, code, "entry_unavailable"))
            continue
        entry_date = entry_bar[0]
        gross_budget = min(slot_capital, cash)
        shares = gross_budget / (entry_open * (1.0 + config.cost_bps_per_side / 10_000.0))
        if shares <= 0:
            events.append(_event(signal_date, code, "cash_skip"))
            continue
        entry_cost = shares * entry_open
        entry_fee = entry_cost * config.cost_bps_per_side / 10_000.0
        cash -= entry_cost + entry_fee
        trade = _simulate_trade(pick, bars, entry_date, entry_open, shares, config.exit_policy)
        trade["entry_fee"] = round(entry_fee, 6)
        if trade["exit_fill_price"] is not None:
            exit_value = shares * trade["exit_fill_price"]
            exit_fee = exit_value * config.cost_bps_per_side / 10_000.0
            trade["exit_fee"] = round(exit_fee, 6)
            trade["net_pnl"] = round(exit_value - exit_fee - entry_cost - entry_fee, 6)
        else:
            trade["exit_fee"] = 0.0
            trade["net_pnl"] = pd.NA
        trades.append(trade)
        active.append(trade)
        events.append(_event(entry_date, code, "entry", price=entry_open))
        if trade["exit_date"]:
            events.append(_event(pd.Timestamp(trade["exit_date"]), code, trade["exit_reason"], price=trade["exit_fill_price"]))

    equity = _equity_curve(trades, normalized, config.initial_capital, config.cost_bps_per_side, all_dates)
    return pd.DataFrame(trades), equity, pd.DataFrame(events)


def portfolio_metrics(equity: pd.DataFrame, trades: pd.DataFrame, *, initial_capital: float) -> dict[str, Any]:
    if equity.empty:
        return {"total_return_pct": 0.0, "max_drawdown_pct": 0.0}
    curve = pd.to_numeric(equity["equity"], errors="coerce")
    total_return = (curve.iloc[-1] / initial_capital - 1.0) * 100.0
    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    daily_ret = curve.pct_change().dropna()
    years = max((pd.Timestamp(equity["date"].iloc[-1]) - pd.Timestamp(equity["date"].iloc[0])).days / 365.25, 1 / 365.25)
    cagr = ((curve.iloc[-1] / initial_capital) ** (1 / years) - 1.0) * 100.0
    downside = daily_ret[daily_ret < 0]
    return {
        "total_return_pct": round(float(total_return), 6),
        "CAGR_pct": round(float(cagr), 6),
        "max_drawdown_pct": round(float(drawdown.min() * 100.0), 6),
        "Calmar": round(float((cagr / abs(drawdown.min() * 100.0)) if drawdown.min() < 0 else np.nan), 6),
        "Sharpe": round(float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252)) if daily_ret.std() else np.nan, 6),
        "Sortino": round(float((daily_ret.mean() / downside.std()) * np.sqrt(252)) if downside.std() else np.nan, 6),
        "win_rate": round(float((pd.to_numeric(trades.get("return_pct", pd.Series(dtype=float)), errors="coerce") > 0).mean()), 6)
        if not trades.empty
        else 0.0,
        "average_holding_days": round(float(pd.to_numeric(trades.get("holding_days", pd.Series(dtype=float)), errors="coerce").mean()), 6)
        if not trades.empty
        else 0.0,
        "max_concurrent_positions": int(equity.get("open_positions", pd.Series([0])).max()),
    }


def _simulate_trade(
    pick: pd.Series,
    bars: pd.DataFrame,
    entry_date: pd.Timestamp,
    entry_price: float,
    shares: float,
    policy: ExitPolicy,
) -> dict[str, Any]:
    code = str(pick.get("code", "") or "").strip()
    pivot = to_float(pick.get("ibd_candidate_price"))
    anchor = parse_date(pick.get("ibd_entry_date"))
    stop = entry_price * (1.0 - policy.stop_pct / 100.0)
    profit = entry_price * (1.0 + policy.profit_pct / 100.0)
    power_price = pivot * (1.0 + policy.power_pct / 100.0) if pivot is not None else None
    power_deadline = breakout_week_friday(anchor, policy.power_weeks) if anchor is not None else None
    min_hold_until: pd.Timestamp | None = None
    power_date: pd.Timestamp | None = None
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason = ""
    high_water = entry_price
    window = bars[bars.index >= entry_date]
    last_close = entry_price
    for date, row in window.iterrows():
        date = pd.Timestamp(date)
        open_ = to_float(row.get("Open"))
        high = to_float(row.get("High"))
        low = to_float(row.get("Low"))
        close = to_float(row.get("Close"))
        if close is not None:
            last_close = close
        if high is not None:
            high_water = max(high_water, high)
        if open_ is not None and open_ <= stop:
            exit_date, exit_price, exit_reason = date, open_, "gap_stop"
            break
        stop_hit = low is not None and low <= stop
        power_hit = (
            power_price is not None
            and power_deadline is not None
            and date <= power_deadline
            and date >= entry_date
            and high is not None
            and high >= power_price
        )
        if stop_hit and policy.same_day_order == "stop_first":
            exit_date, exit_price, exit_reason = date, stop, "stop_loss"
            break
        if power_hit and power_date is None and anchor is not None:
            power_date = date
            min_hold_until = breakout_week_friday(anchor, policy.min_hold_weeks)
        if stop_hit:
            exit_date, exit_price, exit_reason = date, stop, "stop_loss"
            break
        locked = min_hold_until is not None and date <= min_hold_until
        if min_hold_until is not None and date >= min_hold_until and policy.post_lock == "week8_close" and close is not None:
            exit_date, exit_price, exit_reason = date, close, "week8_close"
            break
        if min_hold_until is not None and date > min_hold_until and policy.post_lock == "trend_exit":
            recent = window.loc[:date].tail(10)
            if len(recent) >= 5 and close is not None and close < float(recent["Close"].mean()):
                exit_date, exit_price, exit_reason = date, close, "trend_exit"
                break
        if min_hold_until is not None and date > min_hold_until and policy.post_lock == "mark_to_market":
            continue
        if not locked and high is not None and high >= profit:
            if min_hold_until is not None and date > min_hold_until and open_ is not None and open_ >= profit:
                exit_date, exit_price, exit_reason = date, open_, "profit_target_post_lock_gap"
            else:
                exit_date, exit_price, exit_reason = date, profit, "profit_target"
            break
    censored = exit_date is None
    mtm = last_close if censored else exit_price
    return {
        "variant": pick.get("variant", pick.get("selected_by", "")),
        "snapshot_date": fmt_date(parse_date(pick.get("snapshot_date"))),
        "code": code,
        "pick_order": pick.get("pick_order", 1),
        "entry_date": fmt_date(entry_date),
        "entry_fill_price": round(entry_price, 6),
        "shares": round(shares, 8),
        "position_cost": round(shares * entry_price, 6),
        "stop_price": round(stop, 6),
        "profit_target": round(profit, 6),
        "power_trigger_date": fmt_date(power_date),
        "minimum_hold_until": fmt_date(min_hold_until),
        "exit_date": fmt_date(exit_date),
        "exit_fill_price": None if exit_price is None else round(exit_price, 6),
        "exit_reason": "censored_mtm" if censored else exit_reason,
        "censored": censored,
        "mtm_price": round(float(mtm), 6) if mtm is not None else pd.NA,
        "return_pct": pct(mtm, entry_price),
        "holding_days": (pd.Timestamp(exit_date or window.index[-1]) - entry_date).days if not window.empty else 0,
        "high_water_pct": pct(high_water, entry_price),
    }


def _equity_curve(
    trades: list[dict[str, Any]],
    prices: dict[str, pd.DataFrame],
    initial_capital: float,
    cost_bps: float,
    dates: list[pd.Timestamp],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not dates:
        return pd.DataFrame([{"date": "", "cash": initial_capital, "market_value": 0.0, "equity": initial_capital, "open_positions": 0}])
    for date in dates:
        cash = initial_capital
        market_value = 0.0
        open_positions = 0
        for trade in trades:
            entry_date = parse_date(trade["entry_date"])
            exit_date = parse_date(trade["exit_date"])
            if entry_date is None or date < entry_date:
                continue
            cash -= float(trade["position_cost"]) + float(trade["entry_fee"])
            code = str(trade["code"])
            bars = prices.get(code, pd.DataFrame())
            px = _close_asof(bars, date)
            if exit_date is not None and date >= exit_date:
                exit_value = float(trade["shares"]) * float(trade["exit_fill_price"])
                cash += exit_value - float(trade["exit_fee"])
            else:
                open_positions += 1
                market_value += float(trade["shares"]) * (px if px is not None else float(trade["entry_fill_price"]))
        rows.append(
            {
                "date": fmt_date(date),
                "cash": round(max(cash, 0.0), 6),
                "market_value": round(market_value, 6),
                "equity": round(cash + market_value, 6),
                "open_positions": open_positions,
            }
        )
    return pd.DataFrame(rows)


def _close_asof(bars: pd.DataFrame, date: pd.Timestamp) -> float | None:
    window = bars[bars.index <= date]
    if window.empty:
        return None
    return to_float(window.iloc[-1].get("Close"))


def _all_price_dates(prices: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    dates = sorted({pd.Timestamp(idx) for frame in prices.values() for idx in frame.index})
    return dates


def _release_closed(active: list[dict[str, Any]], signal_date: pd.Timestamp) -> None:
    active[:] = [
        trade for trade in active if not trade["exit_date"] or pd.Timestamp(trade["exit_date"]) > signal_date
    ]


def _has_active(active: list[dict[str, Any]], code: str, signal_date: pd.Timestamp) -> bool:
    return any(trade["code"] == code and (not trade["exit_date"] or pd.Timestamp(trade["exit_date"]) > signal_date) for trade in active)


def _event(date: pd.Timestamp, code: str, event: str, price: float | None = None) -> dict[str, Any]:
    return {"date": fmt_date(date), "code": code, "event": event, "price": price}
