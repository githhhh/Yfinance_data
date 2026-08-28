from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import math
import pandas as pd

from backtest.replay_eps import get_replay_signal_eps


@dataclass(frozen=True)
class SnapshotMeta:
    snapshot_date: str
    commit: str
    commit_date: str
    row_count: int
    actionable_count: int
    comparable_schema: bool


@dataclass(frozen=True)
class PathMetrics:
    code: str
    source: str
    buy_price: float | None
    snapshot_close: float | None
    latest_close: float | None
    latest_close_return_pct: float | None
    max_gain_pct: float | None
    max_gain_date: str
    max_drawdown_pct: float | None
    max_drawdown_date: str
    hit_stop_8pct: bool
    stop_8pct_date: str


@dataclass
class ReplayItem:
    code: str
    raw_rank: int
    final_group: str
    industry: str
    sort_key: tuple


@dataclass(frozen=True)
class ReplayResult:
    selected: list[ReplayItem]
    raw_ranking: list[ReplayItem]


def to_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "<na>"}:
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def to_bool(value: object) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def choose_complete_week_snapshots(
    metas: Iterable[SnapshotMeta],
    *,
    start: date,
    end: date,
    excluded_snapshot: str | None = None,
) -> list[SnapshotMeta]:
    by_snapshot: dict[str, SnapshotMeta] = {}
    for meta in metas:
        snap = datetime.strptime(meta.snapshot_date, "%Y-%m-%d").date()
        if snap < start or snap > end:
            continue
        if excluded_snapshot and meta.snapshot_date == excluded_snapshot:
            continue
        if not meta.comparable_schema:
            continue
        previous = by_snapshot.get(meta.snapshot_date)
        if previous is None or meta.commit_date > previous.commit_date:
            by_snapshot[meta.snapshot_date] = meta
    return [by_snapshot[key] for key in sorted(by_snapshot)]


def _as_date_index(price_bars: pd.DataFrame) -> pd.DataFrame:
    bars = price_bars.copy()
    if "Date" in bars.columns:
        index = pd.to_datetime(bars["Date"])
    else:
        index = pd.to_datetime(bars.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    bars.index = index
    return bars.sort_index()


def compute_path_metrics(
    *,
    code: str,
    snapshot_date: str,
    buy_price: float | None,
    snapshot_close: float | None,
    price_bars: pd.DataFrame | None,
    end_date: str,
) -> PathMetrics:
    if buy_price is None or price_bars is None or price_bars.empty:
        return PathMetrics(code, "missing", buy_price, snapshot_close, None, None, None, "", None, "", False, "")

    source = str(price_bars.attrs.get("source", "daily_cache"))
    bars = _as_date_index(price_bars)
    window = bars[(bars.index >= pd.Timestamp(snapshot_date)) & (bars.index <= pd.Timestamp(end_date))]
    if window.empty:
        return PathMetrics(code, "missing", buy_price, snapshot_close, None, None, None, "", None, "", False, "")

    latest_close = to_float(window.iloc[-1].get("Close"))
    high_series = pd.to_numeric(window["High"], errors="coerce")
    low_series = pd.to_numeric(window["Low"], errors="coerce")
    high_idx = high_series.idxmax()
    low_idx = low_series.idxmin()
    max_high = to_float(high_series.loc[high_idx])
    min_low = to_float(low_series.loc[low_idx])
    stop_date = ""
    stop_level = buy_price * 0.92
    for idx, row in window.iterrows():
        low = to_float(row.get("Low"))
        if low is not None and low <= stop_level:
            stop_date = str(idx.date())
            break

    return PathMetrics(
        code=code,
        source=source,
        buy_price=buy_price,
        snapshot_close=snapshot_close,
        latest_close=latest_close,
        latest_close_return_pct=_pct(latest_close, buy_price),
        max_gain_pct=_pct(max_high, buy_price),
        max_gain_date=str(high_idx.date()),
        max_drawdown_pct=_pct(min_low, buy_price),
        max_drawdown_date=str(low_idx.date()),
        hit_stop_8pct=bool(stop_date),
        stop_8pct_date=stop_date,
    )


def _pct(value: float | None, base: float | None) -> float | None:
    if value is None or base is None or base == 0:
        return None
    return round((value / base - 1.0) * 100.0, 6)


def repair_pool_fields(pool: pd.DataFrame, prices: dict[str, pd.DataFrame], *, snapshot_date: str) -> pd.DataFrame:
    repaired = pool.copy()
    for column in [
        "latest_close_repair_method",
        "price_52_week_high_repair_method",
        "current_vs_ibd_candidate_pct_repair_method",
        "dist_to_52w_high_pct_repair_method",
        "repair_source_cutoff",
        "lookahead_risk",
    ]:
        if column not in repaired.columns:
            repaired[column] = ""

    cutoff = pd.Timestamp(snapshot_date)
    high_window_start = cutoff - pd.Timedelta(days=370)
    for idx, row in repaired.iterrows():
        code = str(row.get("code", "")).strip()
        bars = prices.get(code)
        if bars is None or bars.empty:
            continue
        bounded = _as_date_index(bars)
        bounded = bounded[bounded.index <= cutoff]
        if bounded.empty:
            continue
        last_row = bounded.iloc[-1]
        latest_close = to_float(row.get("latest_close"))
        if latest_close is None:
            latest_close = to_float(last_row.get("Close"))
            repaired.at[idx, "latest_close"] = latest_close
            repaired.at[idx, "latest_close_repair_method"] = "snapshot_bounded_close"
        high_52 = to_float(row.get("price_52_week_high"))
        if high_52 is None:
            high_window = bounded[bounded.index >= high_window_start]
            high_52 = to_float(pd.to_numeric(high_window["High"], errors="coerce").max())
            repaired.at[idx, "price_52_week_high"] = high_52
            repaired.at[idx, "price_52_week_high_repair_method"] = "snapshot_bounded_52w_high"
        candidate_price = to_float(row.get("ibd_candidate_price"))
        if to_float(row.get("current_vs_ibd_candidate_pct")) is None and latest_close is not None and candidate_price:
            repaired.at[idx, "current_vs_ibd_candidate_pct"] = round((latest_close / candidate_price - 1.0) * 100.0, 6)
            repaired.at[idx, "current_vs_ibd_candidate_pct_repair_method"] = "snapshot_close_vs_candidate"
        if to_float(row.get("dist_to_52w_high_pct")) is None and latest_close is not None and high_52:
            repaired.at[idx, "dist_to_52w_high_pct"] = round((latest_close / high_52 - 1.0) * 100.0, 6)
            repaired.at[idx, "dist_to_52w_high_pct_repair_method"] = "snapshot_close_vs_52w_high"
        if to_float(row.get("eps_yoy_growth")) is None:
            try:
                eps_val = get_replay_signal_eps(snapshot_date, code, allow_network=False)
                if eps_val is not None:
                    repaired.at[idx, "eps_yoy_growth"] = eps_val
                    repaired.at[idx, "eps_yoy_growth_repair_method"] = "pit_signal_supplement"
            except Exception:
                pass
        repaired.at[idx, "repair_source_cutoff"] = snapshot_date
        repaired.at[idx, "lookahead_risk"] = "none_for_price_repairs"
    return repaired


def select_current_skill_top3(pool: pd.DataFrame) -> ReplayResult:
    raw = []
    for row_idx, row in pool.iterrows():
        if not _is_actionable_signal(row):
            continue
        item = _current_item(row, row_idx)
        raw.append(item)
    raw.sort(key=lambda item: item.sort_key)
    for rank, item in enumerate(raw, 1):
        item.raw_rank = rank

    selected: list[ReplayItem] = []
    covered = set()
    for item in raw:
        row_obj = pool.loc[item.sort_key[-1]] if item.sort_key[-1] in pool.index else None
        eps = to_float(row_obj.get("eps_yoy_growth")) if row_obj is not None else None
        if eps is None and row_obj is not None:
            snap = str(row_obj.get("snapshot_date", "")).strip()
            if snap:
                try:
                    eps = get_replay_signal_eps(snap, item.code, allow_network=False)
                except Exception:
                    pass
        industry_key = item.industry.strip().lower()
        if not industry_key or eps is None:
            continue
        if industry_key in covered:
            continue
        selected.append(item)
        item.final_group = "PRIORITY"
        covered.add(industry_key)
        if len(selected) == 3:
            break

    for item in raw:
        if item.final_group == "PRIORITY":
            continue
        row_obj = pool.loc[item.sort_key[-1]] if item.sort_key[-1] in pool.index else None
        eps = to_float(row_obj.get("eps_yoy_growth")) if row_obj is not None else None
        if eps is None and row_obj is not None:
            snap = str(row_obj.get("snapshot_date", "")).strip()
            if snap:
                try:
                    eps = get_replay_signal_eps(snap, item.code, allow_network=False)
                except Exception:
                    pass
        if item.raw_rank <= 5 and eps is None:
            item.final_group = "WATCH"
        else:
            item.final_group = "OTHER"
    return ReplayResult(selected=selected, raw_ranking=raw)


def _current_item(row: pd.Series, row_idx: int) -> ReplayItem:
    tier, major_fail, major_unknown = _current_tier(row)
    geom = _current_geometry(row)
    cur = to_float(row.get("current_vs_ibd_candidate_pct"))
    fresh = cur is not None and 0 <= cur <= 2
    dist = to_float(row.get("dist_to_52w_high_pct"))
    if dist is None:
        minor7 = 1
    elif dist > -5:
        minor7 = 0
    else:
        minor7 = 2
    weekly = to_float(row.get("volume_ratio"))
    weekly_bonus = weekly is not None and weekly >= 1.3
    sort_key = (
        {"A": 0, "B": 1, "C": 2, "D": 3}[tier],
        major_fail,
        major_unknown,
        _CURRENT_GEOMETRY_ORDER[geom],
        0 if fresh else 1,
        minor7,
        0 if weekly_bonus else 1,
        str(row.get("code", "")),
        row_idx,
    )
    return ReplayItem(str(row.get("code", "")), 0, "OTHER", str(row.get("industry", "") or ""), sort_key)


def _current_tier(row: pd.Series) -> tuple[str, int, int]:
    critical_fail = 0
    critical_unknown = 0
    cur = to_float(row.get("current_vs_ibd_candidate_pct"))
    vol = to_float(row.get("ibd_entry_volume_ratio"))
    geom = _current_geometry(row)
    if cur is None:
        critical_unknown += 1
    elif not (0 <= cur <= 5):
        critical_fail += 1
    if vol is None:
        critical_unknown += 1
    elif vol < 1.5:
        critical_fail += 1
    if geom == "UNKNOWN":
        critical_unknown += 1
    elif geom in {"Squat / Upper Shadow", "Defensive Failure"}:
        critical_fail += 1

    rule = str(row.get("ibd_candidate_rule", "") or "")
    if rule in {"ceiling", "ceiling_breakout"}:
        depth = to_float(row.get("base_depth_pct"))
        duration = to_float(row.get("base_duration_weeks"))
    else:
        depth = to_float(row.get("pullback_pct"))
        duration = to_float(row.get("pullback_duration_weeks"))
    major_unknown = int(depth is None) + int(duration is None)
    major_fail = 0
    if rule not in {"ceiling", "ceiling_breakout"}:
        dry = to_bool(row.get("pullback_v_is_dry"))
        if dry is None:
            major_unknown += 1
        elif dry is False:
            major_fail += 1
    if critical_fail:
        return "D", major_fail, major_unknown
    if critical_unknown:
        return "C", major_fail, major_unknown
    if major_unknown:
        return "B", major_fail, major_unknown
    return "A", major_fail, major_unknown


_CURRENT_GEOMETRY_ORDER = {
    "Full-range Breakout": 0,
    "Strong Finish": 1,
    "Faded Gap": 2,
    "Constructive Breakout": 3,
    "Marginal Breakout": 4,
    "UNKNOWN": 5,
    "Squat / Upper Shadow": 6,
    "Defensive Failure": 7,
}


def _current_geometry(row: pd.Series) -> str:
    pos = to_float(row.get("ibd_entry_close_position"))
    rr = to_float(row.get("ibd_entry_breakout_range_ratio"))
    if pos is not None and not (0 <= pos <= 1):
        return "UNKNOWN"
    if rr is not None and rr <= 0:
        return "Defensive Failure"
    if pos is not None and pos < 0.65:
        return "Squat / Upper Shadow"
    if pos is None or rr is None:
        return "UNKNOWN"
    trigger_pos = pos - rr
    if trigger_pos <= 0 and pos >= 0.80:
        return "Full-range Breakout"
    if trigger_pos <= 0 and 0.65 <= pos < 0.80:
        return "Faded Gap"
    if trigger_pos > 0 and pos >= 0.80 and rr >= 0.50:
        return "Strong Finish"
    if trigger_pos > 0 and pos >= 0.80 and rr < 0.50:
        return "Constructive Breakout"
    if trigger_pos > 0 and 0.65 <= pos < 0.80 and rr >= 0.50:
        return "Constructive Breakout"
    return "Marginal Breakout"


def select_old_skill_proxy_top3(pool: pd.DataFrame) -> ReplayResult:
    raw = []
    for row_idx, row in pool.iterrows():
        if not _is_actionable_signal(row):
            continue
        raw.append(_old_item(row, row_idx))
    raw.sort(key=lambda item: item.sort_key)
    for rank, item in enumerate(raw, 1):
        item.raw_rank = rank

    selected = []
    sector_counts: dict[str, int] = {}
    sectors = [str(pool.loc[item.sort_key[-1]].get("sector", "") or "") for item in raw]
    crowded = {sector for sector in sectors if sector and sectors.count(sector) / max(len(sectors), 1) > 0.5}
    for item in raw:
        row = pool.loc[item.sort_key[-1]]
        if _old_critical_failures(row):
            continue
        sector = str(row.get("sector", "") or "")
        limit = 1 if sector in crowded else 2
        if sector and sector_counts.get(sector, 0) >= limit:
            continue
        item.final_group = "PRIORITY"
        selected.append(item)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected) == 3:
            break
    return ReplayResult(selected=selected, raw_ranking=raw)


def _old_item(row: pd.Series, row_idx: int) -> ReplayItem:
    eps = to_float(row.get("eps_yoy_growth"))
    if eps is None:
        snap = str(row.get("snapshot_date", "")).strip()
        code = str(row.get("code", "")).strip()
        if snap and code:
            try:
                eps = get_replay_signal_eps(snap, code, allow_network=False)
            except Exception:
                pass
    cur = to_float(row.get("current_vs_ibd_candidate_pct"))
    vol = to_float(row.get("ibd_entry_volume_ratio"))
    dist = to_float(row.get("dist_to_52w_high_pct"))
    weekly = to_float(row.get("volume_ratio"))
    sort_key = (
        0 if not _old_critical_failures(row) else 1,
        0 if eps is not None and eps > 0 else 1,
        0 if cur is not None and 0 <= cur <= 2 else 1,
        -(vol or -1),
        _OLD_GEOMETRY_ORDER[_old_geometry(row)],
        0 if dist is not None and dist > -5 else 1,
        0 if weekly is not None and weekly >= 1.3 else 1,
        str(row.get("code", "")),
        row_idx,
    )
    return ReplayItem(str(row.get("code", "")), 0, "OTHER", str(row.get("industry", "") or ""), sort_key)


_OLD_GEOMETRY_ORDER = {
    "Gap Breakout": 0,
    "Strong Finish": 1,
    "Constructive/Other": 2,
    "Squat / Upper Shadow": 3,
    "UNKNOWN": 4,
    "Defensive Failure": 5,
}


def _old_geometry(row: pd.Series) -> str:
    pos = to_float(row.get("ibd_entry_close_position"))
    rr = to_float(row.get("ibd_entry_breakout_range_ratio"))
    if rr is not None and rr <= 0:
        return "Defensive Failure"
    if pos is None or rr is None:
        return "UNKNOWN"
    if rr > 1:
        return "Gap Breakout"
    if pos >= 0.80 and rr >= 0.50:
        return "Strong Finish"
    if pos < 0.65:
        return "Squat / Upper Shadow"
    return "Constructive/Other"


def _old_critical_failures(row: pd.Series) -> list[str]:
    failures = []
    cur = to_float(row.get("current_vs_ibd_candidate_pct"))
    vol = to_float(row.get("ibd_entry_volume_ratio"))
    pos = to_float(row.get("ibd_entry_close_position"))
    rr = to_float(row.get("ibd_entry_breakout_range_ratio"))
    if cur is None or cur < 0 or cur > 2:
        failures.append("fresh")
    if vol is None or vol < 1.5:
        failures.append("volume")
    if rr is None or rr <= 0:
        failures.append("range")
    if pos is None or pos < 0.50:
        failures.append("close_position")
    return failures


def _is_actionable_signal(row: pd.Series) -> bool:
    return (
        to_bool(row.get("signal")) is True
        and bool(str(row.get("ibd_candidate_rule", "") or "").strip())
        and str(row.get("ibd_entry_status", "") or "").strip().upper() == "ACTIONABLE"
    )
