from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DERIVED_CONTEXT_FEATURES,
    DERIVED_MARKET_FEATURES,
    DERIVED_PRICE_FEATURES,
)


def _safe_pct_rank(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rank(pct=True, method="average")


def add_cross_sectional_context(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()

    grouped = out.groupby("snapshot_date", sort=False)
    if "mom_20" in out.columns:
        out["xs_mom20_pct"] = grouped["mom_20"].transform(_safe_pct_rank)
    else:
        out["xs_mom20_pct"] = np.nan

    if "rv_20" in out.columns:
        out["xs_rv20_pct"] = grouped["rv_20"].transform(_safe_pct_rank)
    else:
        out["xs_rv20_pct"] = np.nan

    if "ibd_entry_volume_ratio" in out.columns:
        out["xs_entry_volume_pct"] = grouped["ibd_entry_volume_ratio"].transform(
            _safe_pct_rank
        )
    else:
        out["xs_entry_volume_pct"] = np.nan

    if "dist_to_52w_high_pct" in out.columns:
        out["xs_dist52_pct"] = grouped["dist_to_52w_high_pct"].transform(
            _safe_pct_rank
        )
    else:
        out["xs_dist52_pct"] = np.nan

    sector = out.get("sector", pd.Series(index=out.index, dtype=object)).fillna("").astype(str)
    industry = out.get("industry", pd.Series(index=out.index, dtype=object)).fillna("").astype(str)
    sector_key = sector.where(sector.str.strip().ne(""), "(missing)")
    industry_key = industry.where(industry.str.strip().ne(""), "(missing)")
    out["_sector_key"] = sector_key
    out["_industry_key"] = industry_key

    out["sector_candidate_count"] = (
        out.groupby(["snapshot_date", "_sector_key"], sort=False)["code"]
        .transform("size")
        .astype(float)
    )
    out["industry_candidate_count"] = (
        out.groupby(["snapshot_date", "_industry_key"], sort=False)["code"]
        .transform("size")
        .astype(float)
    )

    if "mom_20" in out.columns:
        out["sector_mom20_median"] = (
            out.groupby(["snapshot_date", "_sector_key"], sort=False)["mom_20"]
            .transform("median")
        )
        out["mom20_minus_sector_median"] = (
            pd.to_numeric(out["mom_20"], errors="coerce")
            - pd.to_numeric(out["sector_mom20_median"], errors="coerce")
        )
    else:
        out["sector_mom20_median"] = np.nan
        out["mom20_minus_sector_median"] = np.nan

    if "is_actionable" in out.columns:
        actionable = pd.to_numeric(out["is_actionable"], errors="coerce")
    else:
        actionable = (
            out.get("ibd_entry_status", pd.Series(index=out.index, dtype=object))
            .astype(str)
            .str.upper()
            .eq("ACTIONABLE")
            .astype(float)
        )
    out["_actionable_numeric"] = actionable
    out["sector_actionable_share"] = (
        out.groupby(["snapshot_date", "_sector_key"], sort=False)["_actionable_numeric"]
        .transform("mean")
    )

    out = out.drop(columns=["_sector_key", "_industry_key", "_actionable_numeric"])
    return out


def _max_drawdown(close: np.ndarray) -> float:
    close = close[np.isfinite(close)]
    if len(close) < 2:
        return np.nan
    peaks = np.maximum.accumulate(close)
    dd = close / peaks - 1.0
    return float(np.min(dd))


def _window_price_features(g: pd.DataFrame, snapshot: pd.Timestamp) -> dict[str, float]:
    dates = g["date"].to_numpy(dtype="datetime64[ns]")
    idx = int(np.searchsorted(dates, np.datetime64(snapshot.to_datetime64()), side="right"))
    hist = g.iloc[:idx].copy()
    if hist.empty:
        return {name: np.nan for name in DERIVED_PRICE_FEATURES}

    hist = hist.tail(61).copy()
    close = pd.to_numeric(hist["close"], errors="coerce")
    open_ = pd.to_numeric(hist["open"], errors="coerce")
    high = pd.to_numeric(hist["high"], errors="coerce")
    low = pd.to_numeric(hist["low"], errors="coerce")
    volume = pd.to_numeric(hist["volume"], errors="coerce")

    ret = close.pct_change()
    prev_close = close.shift(1)
    gap = open_ / prev_close - 1.0
    day_range = high - low
    close_pos = (close - low) / day_range.replace(0, np.nan)

    def trailing(series: pd.Series, n: int) -> pd.Series:
        return series.tail(n).dropna()

    def downside_vol(n: int) -> float:
        vals = trailing(ret, n)
        vals = vals[vals < 0]
        if len(vals) < 2:
            return np.nan
        return float(np.sqrt(np.mean(np.square(vals))) * np.sqrt(252.0))

    def max_down_day(n: int) -> float:
        vals = trailing(ret, n)
        return np.nan if vals.empty else float(vals.min())

    def gap_down_freq(n: int) -> float:
        vals = trailing(gap, n)
        return np.nan if vals.empty else float((vals < 0).mean())

    def max_gap_down(n: int) -> float:
        vals = trailing(gap, n)
        return np.nan if vals.empty else float(vals.min())

    def ret_skew(n: int) -> float:
        vals = trailing(ret, n)
        return np.nan if len(vals) < 5 else float(vals.skew())

    def dd(n: int) -> float:
        vals = trailing(close, n)
        return np.nan if len(vals) < 2 else _max_drawdown(vals.to_numpy(dtype=float))

    pos20 = trailing(close_pos, 20)
    ret20 = ret.tail(20)
    vol20 = volume.tail(20)
    valid_down_vol = (
        (ret20 < 0)
        & vol20.notna()
        & np.isfinite(vol20)
    )
    total_volume = float(vol20.dropna().sum()) if not vol20.dropna().empty else np.nan
    down_volume = float(vol20[valid_down_vol].sum()) if valid_down_vol.any() else 0.0

    return {
        "pit_downside_vol_20": downside_vol(20),
        "pit_downside_vol_60": downside_vol(60),
        "pit_max_down_day_20": max_down_day(20),
        "pit_max_down_day_60": max_down_day(60),
        "pit_gap_down_freq_20": gap_down_freq(20),
        "pit_gap_down_freq_60": gap_down_freq(60),
        "pit_max_gap_down_20": max_gap_down(20),
        "pit_max_gap_down_60": max_gap_down(60),
        "pit_return_skew_20": ret_skew(20),
        "pit_return_skew_60": ret_skew(60),
        "pit_max_drawdown_20": dd(20),
        "pit_max_drawdown_60": dd(60),
        "pit_close_position_mean_20": (
            np.nan if pos20.empty else float(pos20.mean())
        ),
        "pit_low_close_frac_20": (
            np.nan if pos20.empty else float((pos20 < 0.30).mean())
        ),
        "pit_down_volume_share_20": (
            np.nan
            if not math.isfinite(total_volume) or total_volume <= 0
            else float(down_volume / total_volume)
        ),
    }


def add_pre_snapshot_price_features(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    out = panel.copy()
    price_groups = {
        str(code): g.sort_values("date").copy()
        for code, g in prices.groupby("code", sort=False)
    }

    rows: list[dict[str, Any]] = []
    for idx, row in out[["snapshot_date", "code"]].iterrows():
        code = str(row["code"])
        snapshot = pd.Timestamp(str(row["snapshot_date"]))
        g = price_groups.get(code)
        if g is None or g.empty:
            features = {name: np.nan for name in DERIVED_PRICE_FEATURES}
        else:
            features = _window_price_features(g, snapshot)
        features["_idx"] = idx
        rows.append(features)

    f = pd.DataFrame(rows).set_index("_idx").sort_index()
    for col in DERIVED_PRICE_FEATURES:
        out[col] = pd.to_numeric(f[col], errors="coerce")
    return out


def _market_features_for_snapshot(
    spy: pd.DataFrame,
    snapshot: pd.Timestamp,
) -> dict[str, float]:
    dates = spy["date"].to_numpy(dtype="datetime64[ns]")
    idx = int(np.searchsorted(dates, np.datetime64(snapshot.to_datetime64()), side="right"))
    hist = spy.iloc[:idx].copy()
    if hist.empty:
        return {name: np.nan for name in DERIVED_MARKET_FEATURES}

    close = pd.to_numeric(hist["close"], errors="coerce")
    ret = close.pct_change()

    def mom(n: int) -> float:
        vals = close.dropna()
        if len(vals) <= n:
            return np.nan
        return float(vals.iloc[-1] / vals.iloc[-n - 1] - 1.0)

    def rv(n: int) -> float:
        vals = ret.tail(n).dropna()
        if len(vals) < 2:
            return np.nan
        return float(vals.std(ddof=1) * np.sqrt(252.0))

    def dd(n: int) -> float:
        vals = close.tail(n).dropna()
        if len(vals) < 2:
            return np.nan
        return _max_drawdown(vals.to_numpy(dtype=float))

    return {
        "pit_spy_mom20": mom(20),
        "pit_spy_mom60": mom(60),
        "pit_spy_rv20": rv(20),
        "pit_spy_drawdown60": dd(60),
    }


def add_market_features(panel: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    spy = prices[prices["code"].astype(str) == "SPY"].sort_values("date").copy()
    if spy.empty:
        raise RuntimeError("SPY history missing from frozen prices")

    unique_snapshots = sorted(out["snapshot_date"].astype(str).unique().tolist())
    mapping = {
        snap: _market_features_for_snapshot(spy, pd.Timestamp(snap))
        for snap in unique_snapshots
    }
    for col in DERIVED_MARKET_FEATURES:
        out[col] = out["snapshot_date"].astype(str).map(
            {snap: values[col] for snap, values in mapping.items()}
        )
    return out


def build_feature_frame(panel: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    out = add_pre_snapshot_price_features(panel, prices)
    out = add_market_features(out, prices)
    out = add_cross_sectional_context(out)

    missing = sorted(
        (
            set(DERIVED_PRICE_FEATURES)
            | set(DERIVED_MARKET_FEATURES)
            | set(DERIVED_CONTEXT_FEATURES)
        )
        - set(out.columns)
    )
    if missing:
        raise RuntimeError(f"Derived feature construction incomplete: {missing}")
    return out
