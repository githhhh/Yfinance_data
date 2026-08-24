from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pandas as pd

from backtest.ibd_skill_replay.core import to_float


@dataclass(frozen=True)
class IBDTradeConfig:
    stop_loss_pct: float = 7.5
    profit_take_pct: float = 22.5
    power_trigger_pct: float = 20.0
    power_trigger_weeks: int = 3
    minimum_hold_weeks: int = 8
    same_day_priority: str = "conservative"
    post_lock_exit: str = "mark_to_market"


@dataclass
class Position:
    ticker: str
    signal_date: pd.Timestamp
    breakout_week: str
    signal_source: str
    pivot: float
    breakout_anchor_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_fill_price: float
    initial_stop: float
    highest_price: float
    power_trigger_date: pd.Timestamp | None = None
    minimum_hold_until: pd.Timestamp | None = None
    state: str = "OPEN"
    exit_date: pd.Timestamp | None = None
    exit_fill_price: float | None = None
    exit_reason: str = ""
    censored: bool = False

    def as_row(self) -> dict[str, Any]:
        ret = None
        if self.exit_fill_price is not None:
            ret = (self.exit_fill_price / self.entry_fill_price - 1.0) * 100.0
        return {
            "ticker": self.ticker,
            "signal_date": _fmt_date(self.signal_date),
            "breakout_week": self.breakout_week,
            "signal_source": self.signal_source,
            "pivot": self.pivot,
            "breakout_anchor_date": _fmt_date(self.breakout_anchor_date),
            "entry_date": _fmt_date(self.entry_date),
            "entry_fill_price": self.entry_fill_price,
            "initial_stop": self.initial_stop,
            "highest_price": self.highest_price,
            "power_trigger_date": _fmt_date(self.power_trigger_date),
            "minimum_hold_until": _fmt_date(self.minimum_hold_until),
            "state": self.state,
            "exit_date": _fmt_date(self.exit_date),
            "exit_fill_price": self.exit_fill_price,
            "exit_reason": self.exit_reason,
            "return_pct": ret,
            "censored": self.censored,
        }


def run_ibd_position_state_machine(
    picks: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    config: IBDTradeConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or IBDTradeConfig()
    if picks.empty:
        return pd.DataFrame(), pd.DataFrame()
    normalized_prices = {code: _normalize_bars(frame) for code, frame in prices.items()}
    pick_frame = picks.copy()
    pick_frame["snapshot_date"] = pd.to_datetime(pick_frame["snapshot_date"])
    if "pick_order" not in pick_frame.columns:
        pick_frame["pick_order"] = 1
    pick_frame = pick_frame.sort_values(["snapshot_date", "pick_order", "code"])

    positions: list[Position] = []
    events: list[dict[str, Any]] = []
    for _, pick in pick_frame.iterrows():
        code = str(pick.get("code", "")).strip()
        signal_date = pd.Timestamp(pick["snapshot_date"]).tz_localize(None)
        breakout_anchor = _parse_date_or_default(pick.get("ibd_entry_date"), signal_date)
        if _has_active_position(positions, code, signal_date):
            events.append(_event(signal_date, code, "repeat_signal_ignored", signal_date=signal_date))
            continue
        bars = normalized_prices.get(code, pd.DataFrame())
        pivot = to_float(pick.get("ibd_candidate_price"))
        if pivot is None:
            pivot = to_float(pick.get("pivot"))
        entry_bar = _next_bar_after(bars, signal_date)
        if entry_bar is None or pivot is None:
            events.append(_event(signal_date, code, "entry_missing", signal_date=signal_date))
            continue
        entry_date, entry_row = entry_bar
        entry_open = to_float(entry_row.get("Open"))
        if entry_open is None:
            events.append(_event(signal_date, code, "entry_missing", signal_date=signal_date))
            continue
        position = Position(
            ticker=code,
            signal_date=signal_date,
            breakout_week=_fmt_date(_week_start(breakout_anchor)),
            signal_source=str(pick.get("signal_source", pick.get("ibd_candidate_signal_source", "")) or ""),
            pivot=float(pivot),
            breakout_anchor_date=breakout_anchor,
            entry_date=entry_date,
            entry_fill_price=float(entry_open),
            initial_stop=float(entry_open) * (1.0 - float(cfg.stop_loss_pct) / 100.0),
            highest_price=float(entry_open),
        )
        positions.append(position)
        events.append(_event(entry_date, code, "entry", price=position.entry_fill_price, signal_date=signal_date))
        _advance_position(position, bars[bars.index >= entry_date], cfg, events)

    rows = [position.as_row() for position in positions]
    return pd.DataFrame(rows), pd.DataFrame(events)


def _advance_position(position: Position, bars: pd.DataFrame, cfg: IBDTradeConfig, events: list[dict[str, Any]]) -> None:
    stop = position.initial_stop
    profit_target = position.entry_fill_price * (1.0 + float(cfg.profit_take_pct) / 100.0)
    power_target = position.pivot * (1.0 + float(cfg.power_trigger_pct) / 100.0)
    for current, row in bars.iterrows():
        current = pd.Timestamp(current)
        open_ = to_float(row.get("Open"))
        high = to_float(row.get("High"))
        low = to_float(row.get("Low"))
        close = to_float(row.get("Close"))
        if high is not None:
            position.highest_price = max(position.highest_price, high)
        if open_ is not None and open_ <= stop:
            _close(position, current, open_, "gap_stop", events)
            return
        stop_hit = low is not None and low <= stop
        power_hit = high is not None and high >= power_target and _breakout_week_age(position.breakout_anchor_date, current) <= cfg.power_trigger_weeks
        if stop_hit and cfg.same_day_priority == "conservative":
            _close(position, current, stop, "stop_loss", events)
            return
        if power_hit and position.power_trigger_date is None:
            position.power_trigger_date = current
            position.minimum_hold_until = _minimum_hold_until(position.breakout_anchor_date, cfg.minimum_hold_weeks)
            events.append(_event(current, position.ticker, "power_trigger", price=power_target, signal_date=position.signal_date))
        if stop_hit:
            _close(position, current, stop, "stop_loss", events)
            return
        locked = position.minimum_hold_until is not None and current <= position.minimum_hold_until
        post_lock_mark = position.minimum_hold_until is not None and cfg.post_lock_exit == "mark_to_market"
        if not locked and not post_lock_mark and high is not None and high >= profit_target:
            _close(position, current, profit_target, "profit_take", events)
            return
        if close is not None:
            position.exit_fill_price = close
    position.censored = True


def _close(position: Position, date: pd.Timestamp, price: float, reason: str, events: list[dict[str, Any]]) -> None:
    position.state = "CLOSED"
    position.exit_date = date
    position.exit_fill_price = float(price)
    position.exit_reason = reason
    position.censored = False
    events.append(_event(date, position.ticker, reason, price=price, signal_date=position.signal_date))


def _has_active_position(positions: list[Position], code: str, signal_date: pd.Timestamp) -> bool:
    for position in positions:
        if position.ticker != code:
            continue
        if position.exit_date is None or position.exit_date > signal_date:
            return True
    return False


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
    bars = bars.rename(columns={column: str(column).title() for column in bars.columns})
    for column in ["Open", "High", "Low", "Close"]:
        if column not in bars.columns:
            return pd.DataFrame()
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    return bars.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()


def _next_bar_after(bars: pd.DataFrame, date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Series] | None:
    if bars.empty:
        return None
    window = bars[bars.index > date]
    if window.empty:
        return None
    idx = pd.Timestamp(window.index[0])
    return idx, window.iloc[0]


def _breakout_week_age(signal_date: pd.Timestamp, current: pd.Timestamp) -> int:
    return int(((_week_start(current) - _week_start(signal_date)).days // 7) + 1)


def _minimum_hold_until(signal_date: pd.Timestamp, weeks: int) -> pd.Timestamp:
    return _week_start(signal_date) + timedelta(days=(weeks - 1) * 7 + 4)


def _week_start(value: pd.Timestamp) -> pd.Timestamp:
    value = pd.Timestamp(value).tz_localize(None)
    return value.normalize() - timedelta(days=value.weekday())


def _parse_date_or_default(value: object, default: pd.Timestamp) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
        if pd.isna(parsed):
            return default
        return parsed.tz_localize(None) if parsed.tzinfo is not None else parsed
    except Exception:
        return default


def _event(date: pd.Timestamp, code: str, event: str, **extra: Any) -> dict[str, Any]:
    return {"date": _fmt_date(date), "code": code, "event": event, **{key: _fmt_date(val) if isinstance(val, pd.Timestamp) else val for key, val in extra.items()}}


def _fmt_date(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()
