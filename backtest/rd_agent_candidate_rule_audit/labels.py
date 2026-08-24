from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .utils import breakout_week_friday, fmt_date, next_bar_after, normalize_bars, parse_date, pct, to_bool, to_float, week_start


@dataclass(frozen=True)
class ExitPolicy:
    stop_pct: float = 8.0
    profit_pct: float = 24.0
    power_pct: float = 20.0
    power_weeks: int = 3
    min_hold_weeks: int = 8
    same_day_order: str = "stop_first"
    post_lock: str = "resume_profit"


@dataclass(frozen=True)
class TradeLabelConfig:
    exit_policy: ExitPolicy = ExitPolicy()
    horizons: tuple[int, ...] = (5, 15, 25, 40)
    mfe_mae_days_3w: int = 15
    mfe_mae_days_8w: int = 40


def normalize_eps_pit(eps: pd.DataFrame) -> pd.DataFrame:
    frame = eps.copy()
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        state = "UNKNOWN"
        value = to_float(row.get("eps_yoy_growth"))
        effective = parse_date(row.get("effective_date"))
        current_period = parse_date(row.get("current_period"))
        snapshot = parse_date(row.get("snapshot_date"))
        status = str(row.get("status", "") or "").strip().lower()
        if value is None or status in {"", "unresolved", "missing", "unknown"}:
            state = "UNKNOWN"
            value = None
        elif effective is not None and current_period is not None and effective.normalize() == current_period.normalize():
            state = "UNVERIFIED_AVAILABILITY"
            value = None
        elif effective is not None and snapshot is not None and effective > snapshot:
            state = "FUTURE_RECORD_BLOCKED"
            value = None
        else:
            state = "VERIFIED"
        out = row.to_dict()
        out["pit_eps_state"] = state
        out["pit_eps_yoy_growth"] = value
        rows.append(out)
    return pd.DataFrame(rows)


def classify_geometry(close_position: float | None, range_ratio: float | None) -> str:
    pos = to_float(close_position)
    rr = to_float(range_ratio)
    if pos is not None and not (0 <= pos <= 1):
        return "UNKNOWN"
    if rr is not None and rr <= 0:
        return "Defensive Failure"
    if pos is None or rr is None:
        return "UNKNOWN"
    trigger_pos = pos - rr
    if trigger_pos <= 0 and pos >= 0.80:
        return "Full-range Breakout"
    if pos >= 0.80 and rr >= 0.50:
        return "Strong Finish"
    if pos < 0.65:
        return "Squat / Upper Shadow"
    if (pos >= 0.80 and rr < 0.50) or (0.65 <= pos < 0.80 and rr >= 0.50):
        return "Constructive Breakout"
    return "Marginal Breakout"


def build_event_labels(events: pd.DataFrame, prices: dict[str, pd.DataFrame], config: TradeLabelConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        row = event.to_dict()
        code = str(event.get("code", "") or "").strip()
        snapshot = parse_date(event.get("snapshot_date"))
        bars = normalize_bars(prices.get(code))
        labels = _empty_labels()
        if snapshot is None or bars.empty:
            rows.append({**row, **labels})
            continue
        entry = next_bar_after(bars, snapshot)
        if entry is None:
            rows.append({**row, **labels})
            continue
        entry_date, entry_bar = entry
        entry_price = to_float(entry_bar.get("Open"))
        if entry_price is None:
            rows.append({**row, **labels})
            continue
        forward = bars[bars.index >= entry_date]
        labels.update(
            {
                "entry_fill_date": fmt_date(entry_date),
                "entry_fill_price": entry_price,
                "entry_unavailable": False,
                "as_of_return_pct": pct(to_float(forward.iloc[-1].get("Close")) if not forward.empty else None, entry_price),
            }
        )
        _add_forward_returns(labels, forward, entry_price, config.horizons)
        _add_mfe_mae(labels, forward, entry_price, "3w", config.mfe_mae_days_3w)
        _add_mfe_mae(labels, forward, entry_price, "8w", config.mfe_mae_days_8w)
        _add_touch_labels(labels, forward, entry_price, config.exit_policy)
        _add_power_labels(labels, event, bars, entry_date, entry_price, config.exit_policy)
        rows.append({**row, **labels})
    return pd.DataFrame(rows)


def _empty_labels() -> dict[str, Any]:
    labels: dict[str, Any] = {
        "entry_fill_date": "",
        "entry_fill_price": pd.NA,
        "entry_unavailable": True,
        "as_of_return_pct": pd.NA,
        "mfe_3w_pct": pd.NA,
        "mae_3w_pct": pd.NA,
        "mfe_8w_pct": pd.NA,
        "mae_8w_pct": pd.NA,
        "stop_8_touched": False,
        "stop_8_touched_date": "",
        "gap_stop_8": False,
        "realized_stop_loss_pct": pd.NA,
        "first_touch_8_20": "",
        "first_touch_8_24": "",
        "first_touch_40d_8_24": "",
        "same_day_path_ambiguous": False,
        "stop_8_within_15d": False,
        "stop_8_within_15d_date": "",
        "stop_8_within_40d": False,
        "stop_8_within_40d_date": "",
        "power_trigger_3w_from_pivot": pd.NA,
        "power_trigger_3w_from_pivot_date": "",
        "power_trigger_3w_max_gain_pct": pd.NA,
        "breakout_anchor_valid": False,
        "breakout_week_number_at_trigger": pd.NA,
        "gain_20_3w_from_entry": False,
        "gain_20_3w_from_entry_date": "",
        "gain_20_within_first_15_trading_days": False,
        "pattern_power_trigger": pd.NA,
        "trade_power_trigger": False,
    }
    for days, label in [(5, "1w"), (15, "3w"), (25, "5w"), (40, "8w")]:
        labels[f"forward_{label}_return_pct"] = pd.NA
        labels[f"forward_{label}_censored"] = True
    for target in ("20", "22_5", "24", "25"):
        labels[f"profit_{target}_touched"] = False
        labels[f"profit_{target}_touched_date"] = ""
        labels[f"profit_{target}_within_15d"] = False
        labels[f"profit_{target}_within_15d_date"] = ""
        labels[f"profit_{target}_within_40d"] = False
        labels[f"profit_{target}_within_40d_date"] = ""
    return labels


def _add_forward_returns(labels: dict[str, Any], forward: pd.DataFrame, entry_price: float, horizons: tuple[int, ...]) -> None:
    names = {5: "1w", 15: "3w", 25: "5w", 40: "8w"}
    for horizon in horizons:
        label = names[horizon]
        if len(forward) >= horizon:
            labels[f"forward_{label}_return_pct"] = pct(to_float(forward.iloc[horizon - 1].get("Close")), entry_price)
            labels[f"forward_{label}_censored"] = False


def _add_mfe_mae(labels: dict[str, Any], forward: pd.DataFrame, entry_price: float, suffix: str, days: int) -> None:
    window = forward.head(days)
    if window.empty:
        return
    labels[f"mfe_{suffix}_pct"] = pct(to_float(window["High"].max()), entry_price)
    labels[f"mae_{suffix}_pct"] = pct(to_float(window["Low"].min()), entry_price)


def _add_touch_labels(labels: dict[str, Any], forward: pd.DataFrame, entry_price: float, policy: ExitPolicy) -> None:
    stop_price = entry_price * (1.0 - policy.stop_pct / 100.0)
    targets = {20.0: "20", 22.5: "22_5", 24.0: "24", 25.0: "25"}
    first_20 = ""
    first_24 = ""
    first_40d_24 = ""
    for day_number, (date, bar) in enumerate(forward.iterrows(), 1):
        open_ = to_float(bar.get("Open"))
        high = to_float(bar.get("High"))
        low = to_float(bar.get("Low"))
        stop_gap = open_ is not None and open_ <= stop_price
        stop_hit = low is not None and low <= stop_price
        hit_targets = {target: high is not None and high >= entry_price * (1.0 + target / 100.0) for target in targets}
        if stop_hit and not labels["stop_8_touched"]:
            labels["stop_8_touched"] = True
            labels["stop_8_touched_date"] = fmt_date(pd.Timestamp(date))
            labels["gap_stop_8"] = bool(stop_gap)
            fill = open_ if stop_gap else stop_price
            labels["realized_stop_loss_pct"] = pct(fill, entry_price)
        if stop_hit and day_number <= 15 and not labels["stop_8_within_15d"]:
            labels["stop_8_within_15d"] = True
            labels["stop_8_within_15d_date"] = fmt_date(pd.Timestamp(date))
        if stop_hit and day_number <= 40 and not labels["stop_8_within_40d"]:
            labels["stop_8_within_40d"] = True
            labels["stop_8_within_40d_date"] = fmt_date(pd.Timestamp(date))
        for target, key in targets.items():
            if hit_targets[target] and not labels[f"profit_{key}_touched"]:
                labels[f"profit_{key}_touched"] = True
                labels[f"profit_{key}_touched_date"] = fmt_date(pd.Timestamp(date))
            if hit_targets[target] and day_number <= 15 and not labels[f"profit_{key}_within_15d"]:
                labels[f"profit_{key}_within_15d"] = True
                labels[f"profit_{key}_within_15d_date"] = fmt_date(pd.Timestamp(date))
            if hit_targets[target] and day_number <= 40 and not labels[f"profit_{key}_within_40d"]:
                labels[f"profit_{key}_within_40d"] = True
                labels[f"profit_{key}_within_40d_date"] = fmt_date(pd.Timestamp(date))
        if stop_hit and hit_targets[20.0]:
            labels["same_day_path_ambiguous"] = True
        if stop_hit and hit_targets[24.0]:
            labels["same_day_path_ambiguous"] = True
        if not first_20 and (stop_hit or hit_targets[20.0]):
            first_20 = "stop" if stop_hit and policy.same_day_order == "stop_first" else "profit"
        if not first_24 and (stop_hit or hit_targets[24.0]):
            first_24 = "stop" if stop_hit and policy.same_day_order == "stop_first" else "profit"
        if day_number <= 40 and not first_40d_24 and (stop_hit or hit_targets[24.0]):
            first_40d_24 = "stop" if stop_hit and policy.same_day_order == "stop_first" else "profit"
    labels["first_touch_8_20"] = first_20
    labels["first_touch_8_24"] = first_24
    labels["first_touch_40d_8_24"] = first_40d_24


def _add_power_labels(
    labels: dict[str, Any],
    event: pd.Series,
    bars: pd.DataFrame,
    entry_date: pd.Timestamp,
    entry_price: float,
    policy: ExitPolicy,
) -> None:
    pivot = to_float(event.get("ibd_candidate_price"))
    anchor = parse_date(event.get("ibd_entry_date"))
    snapshot = parse_date(event.get("snapshot_date"))
    if pivot is None or anchor is None or snapshot is None or anchor > snapshot:
        labels["power_trigger_3w_from_pivot"] = pd.NA
        labels["pattern_power_trigger"] = pd.NA
        return
    labels["breakout_anchor_valid"] = True
    deadline = breakout_week_friday(anchor, policy.power_weeks)
    pattern_window = bars[(bars.index >= anchor) & (bars.index <= deadline)]
    if not pattern_window.empty:
        max_high = to_float(pattern_window["High"].max())
        labels["power_trigger_3w_max_gain_pct"] = pct(max_high, pivot)
    trigger_price = pivot * (1.0 + policy.power_pct / 100.0)
    trigger_date = _first_high_touch(pattern_window, trigger_price)
    labels["power_trigger_3w_from_pivot"] = trigger_date is not None
    labels["pattern_power_trigger"] = trigger_date is not None
    if trigger_date is not None:
        labels["power_trigger_3w_from_pivot_date"] = fmt_date(trigger_date)
        labels["breakout_week_number_at_trigger"] = int(((week_start(trigger_date) - week_start(anchor)).days // 7) + 1)
        labels["trade_power_trigger"] = trigger_date >= entry_date
    entry_20_window = bars[(bars.index >= entry_date) & (bars.index <= deadline)]
    entry_touch = _first_high_touch(entry_20_window, entry_price * 1.20)
    labels["gain_20_3w_from_entry"] = entry_touch is not None
    first_15_entry_window = bars[bars.index >= entry_date].head(15)
    entry_15_touch = _first_high_touch(first_15_entry_window, entry_price * 1.20)
    labels["gain_20_within_first_15_trading_days"] = entry_15_touch is not None
    if entry_touch is not None:
        labels["gain_20_3w_from_entry_date"] = fmt_date(entry_touch)


def _first_high_touch(window: pd.DataFrame, level: float) -> pd.Timestamp | None:
    for date, row in window.iterrows():
        high = to_float(row.get("High"))
        if high is not None and high >= level:
            return pd.Timestamp(date)
    return None
