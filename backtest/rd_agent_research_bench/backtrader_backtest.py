from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import backtrader as bt
import pandas as pd


@dataclass(frozen=True)
class BacktraderResult:
    summary: dict[str, object]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame


class _PandasOHLCV(bt.feeds.PandasData):
    params = (
        ("datetime", None),
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
        ("openinterest", None),
    )


class WeeklyRebalanceStopStrategy(bt.Strategy):
    params = (
        ("schedule", None),
        ("stop_loss_pct", 8.0),
    )

    def __init__(self):
        self.schedule = self.p.schedule or {}
        self.stop_loss_pct = float(self.p.stop_loss_pct)
        self.executed_snapshots: set[pd.Timestamp] = set()
        self.stop_orders: dict[str, bt.Order] = {}
        self.trade_events: list[dict[str, object]] = []
        self.equity_rows: list[dict[str, object]] = []
        self.data_by_name = {data._name: data for data in self.datas}

    def next_open(self):
        current = pd.Timestamp(self.datas[0].datetime.date(0))
        pending = [snapshot for snapshot in self.schedule if snapshot < current and snapshot not in self.executed_snapshots]
        if pending:
            snapshot = max(pending)
            for old in pending:
                self.executed_snapshots.add(old)
            self._rebalance(current, snapshot)

    def next(self):
        current = self.datas[0].datetime.date(0)
        self.equity_rows.append(
            {
                "date": current.isoformat(),
                "value": float(self.broker.getvalue()),
                "cash": float(self.broker.getcash()),
            }
        )

    def notify_order(self, order):
        if order.status != order.Completed:
            return
        name = order.data._name
        if name == "__CASH__":
            return
        side = "buy" if order.isbuy() else "sell"
        event = order.info.get("event", "rebalance_buy" if order.isbuy() else "rebalance_sell")
        if not order.isbuy() and event == "stop":
            event = "stop"
            self.stop_orders.pop(name, None)
        self.trade_events.append(
            {
                "date": bt.num2date(order.executed.dt).date().isoformat(),
                "code": name,
                "event": event,
                "side": side,
                "size": float(order.executed.size),
                "price": float(order.executed.price),
                "value": float(order.executed.value),
                "commission": float(order.executed.comm),
                "portfolio_value": float(self.broker.getvalue()),
            }
        )
        if order.isbuy():
            self._replace_stop(order.data)

    def _rebalance(self, current: pd.Timestamp, snapshot: pd.Timestamp) -> None:
        target_codes = [code for code in self.schedule.get(snapshot, []) if code in self.data_by_name]
        target_codes = [code for code in target_codes if self._has_current_bar(self.data_by_name[code], current)]
        target_weight = 1.0 / len(target_codes) if target_codes else 0.0
        self._cancel_stops()
        for data in self.datas[1:]:
            if not self._has_current_bar(data, current):
                continue
            target = target_weight if data._name in target_codes else 0.0
            self.order_target_percent(data=data, target=target)
        self.trade_events.append(
            {
                "date": current.date().isoformat(),
                "code": "",
                "event": "rebalance",
                "side": "",
                "size": 0.0,
                "price": 0.0,
                "value": 0.0,
                "commission": 0.0,
                "portfolio_value": float(self.broker.getvalue()),
                "snapshot_date": snapshot.date().isoformat(),
                "target_codes": ",".join(target_codes),
            }
        )

    def _replace_stop(self, data) -> None:
        name = data._name
        position = self.getposition(data)
        if position.size <= 0:
            return
        old = self.stop_orders.pop(name, None)
        if old is not None and old.alive():
            self.cancel(old)
        stop_price = position.price * (1.0 - self.stop_loss_pct / 100.0)
        self.stop_orders[name] = self.sell(
            data=data,
            size=position.size,
            exectype=bt.Order.Stop,
            price=stop_price,
            event="stop",
        )

    def _cancel_stops(self) -> None:
        for order in list(self.stop_orders.values()):
            if order.alive():
                self.cancel(order)
        self.stop_orders.clear()

    @staticmethod
    def _has_current_bar(data, current: pd.Timestamp) -> bool:
        return pd.Timestamp(data.datetime.date(0)) == current


def run_backtrader_variant_backtest(
    picks: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    *,
    eps_mode: str,
    variant: str,
    initial_capital: float = 10000.0,
    stop_loss_pct: float = 8.0,
    commission: float = 0.0,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    scoped = _scoped_picks(picks, eps_mode=eps_mode, variant=variant)
    schedule = _build_schedule(scoped)
    symbols = sorted({code for codes in schedule.values() for code in codes})
    loaded_prices = {code: _normalize_bars(prices.get(code)) for code in symbols if prices.get(code) is not None}
    loaded_prices = {code: frame for code, frame in loaded_prices.items() if not frame.empty}
    if not schedule or not loaded_prices:
        summary = _empty_summary(eps_mode, variant, initial_capital, stop_loss_pct, len(scoped), len(symbols))
        return summary, pd.DataFrame(), pd.DataFrame()

    calendar = _calendar_feed(schedule, loaded_prices)
    cerebro = bt.Cerebro(cheat_on_open=True)
    cerebro.broker.setcash(float(initial_capital))
    cerebro.broker.setcommission(commission=float(commission))
    cerebro.adddata(_PandasOHLCV(dataname=calendar), name="__CASH__")
    for code, bars in loaded_prices.items():
        cerebro.adddata(_PandasOHLCV(dataname=bars), name=code)
    cerebro.addstrategy(WeeklyRebalanceStopStrategy, schedule=schedule, stop_loss_pct=stop_loss_pct)
    strategies = cerebro.run()
    strategy = strategies[0]
    trades = pd.DataFrame(strategy.trade_events)
    equity = pd.DataFrame(strategy.equity_rows)
    final_value = float(cerebro.broker.getvalue())
    summary = _summary(
        eps_mode=eps_mode,
        variant=variant,
        initial_capital=initial_capital,
        stop_loss_pct=stop_loss_pct,
        input_picks=len(scoped),
        requested_symbols=len(symbols),
        loaded_symbols=len(loaded_prices),
        final_value=final_value,
        trades=trades,
        equity=equity,
    )
    return summary, trades, equity


def _scoped_picks(picks: pd.DataFrame, *, eps_mode: str, variant: str) -> pd.DataFrame:
    if picks.empty:
        return picks.copy()
    frame = picks[picks["variant"].astype(str).eq(variant)].copy()
    if "eps_mode" in frame.columns:
        frame = frame[frame["eps_mode"].astype(str).eq(eps_mode)]
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
    frame["code"] = frame["code"].astype(str)
    return frame.sort_values(["snapshot_date", "pick_order", "code"])


def _build_schedule(picks: pd.DataFrame) -> dict[pd.Timestamp, list[str]]:
    schedule: dict[pd.Timestamp, list[str]] = {}
    if picks.empty:
        return schedule
    for snapshot, group in picks.groupby("snapshot_date", sort=True):
        schedule[pd.Timestamp(snapshot)] = list(dict.fromkeys(group.sort_values(["pick_order", "code"])["code"]))
    return schedule


def _normalize_bars(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    bars = frame.copy()
    if "Date" in bars.columns:
        index = pd.to_datetime(bars["Date"])
        bars = bars.drop(columns=["Date"])
    else:
        index = pd.to_datetime(bars.index)
    bars.index = index.tz_localize(None) if getattr(index, "tz", None) is not None else index
    rename = {column: str(column).title() for column in bars.columns}
    bars = bars.rename(columns=rename)
    for column in ["Open", "High", "Low", "Close"]:
        if column not in bars.columns:
            return pd.DataFrame()
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    if "Volume" not in bars.columns:
        bars["Volume"] = 0
    bars["Volume"] = pd.to_numeric(bars["Volume"], errors="coerce").fillna(0)
    return bars[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"]).sort_index()


def _calendar_feed(schedule: dict[pd.Timestamp, list[str]], prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    first_snapshot = min(schedule)
    last_date = max(frame.index.max() for frame in prices.values())
    dates = sorted({idx for frame in prices.values() for idx in frame.index if first_snapshot <= idx <= last_date})
    frame = pd.DataFrame(index=pd.DatetimeIndex(dates))
    frame["Open"] = 1.0
    frame["High"] = 1.0
    frame["Low"] = 1.0
    frame["Close"] = 1.0
    frame["Volume"] = 0
    return frame


def _summary(
    *,
    eps_mode: str,
    variant: str,
    initial_capital: float,
    stop_loss_pct: float,
    input_picks: int,
    requested_symbols: int,
    loaded_symbols: int,
    final_value: float,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
) -> dict[str, object]:
    return {
        "eps_mode": eps_mode,
        "variant": variant,
        "backtest_engine": "backtrader",
        "initial_capital": float(initial_capital),
        "stop_loss_pct": float(stop_loss_pct),
        "input_picks": int(input_picks),
        "requested_symbols": int(requested_symbols),
        "loaded_symbols": int(loaded_symbols),
        "final_value": float(final_value),
        "total_return_pct": (float(final_value) / float(initial_capital) - 1.0) * 100.0 if initial_capital else 0.0,
        "max_drawdown_pct": _max_drawdown_pct(equity),
        "trade_events": int(len(trades)),
        "stop_events": int(trades["event"].eq("stop").sum()) if "event" in trades else 0,
        "rebalance_events": int(trades["event"].eq("rebalance").sum()) if "event" in trades else 0,
        "start_date": equity["date"].iloc[0] if not equity.empty else "",
        "end_date": equity["date"].iloc[-1] if not equity.empty else "",
    }


def _empty_summary(
    eps_mode: str,
    variant: str,
    initial_capital: float,
    stop_loss_pct: float,
    input_picks: int,
    requested_symbols: int,
) -> dict[str, object]:
    return _summary(
        eps_mode=eps_mode,
        variant=variant,
        initial_capital=initial_capital,
        stop_loss_pct=stop_loss_pct,
        input_picks=input_picks,
        requested_symbols=requested_symbols,
        loaded_symbols=0,
        final_value=initial_capital,
        trades=pd.DataFrame(),
        equity=pd.DataFrame(),
    )


def _max_drawdown_pct(equity: pd.DataFrame) -> float:
    if equity.empty or "value" not in equity:
        return 0.0
    values = pd.to_numeric(equity["value"], errors="coerce").dropna()
    if values.empty:
        return 0.0
    drawdowns = values / values.cummax() - 1.0
    return float(drawdowns.min() * 100.0)
