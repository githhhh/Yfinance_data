from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.b0_top3_quality_audit.price_cache import _standardize_ohlcv_df
from backtest.b0_top3_quality_audit.ticker_resolution import resolve_symbol_for_provider
from data_providers.yahoo_provider import YahooDataProvider

from .config import (
    AUDIT_AS_OF_DATE,
    BENCHMARK_CODES,
    SNAPSHOT_FORWARD_DAYS,
    YAHOO_DOWNLOAD_AUDIT_CSV,
    YAHOO_SUPPLEMENT_PARQUET,
)


@dataclass(frozen=True)
class YahooSupplementResult:
    prices: pd.DataFrame
    supplement: pd.DataFrame
    audit: pd.DataFrame


def _normalize_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices.copy()
    if out.empty:
        return pd.DataFrame(
            columns=["code", "date", "open", "high", "low", "close", "volume", "source"]
        )
    out["code"] = out["code"].astype(str).str.upper().str.strip()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "source" not in out.columns:
        out["source"] = "unknown"
    out = out[
        ["code", "date", "open", "high", "low", "close", "volume", "source"]
    ]
    out = out[out["date"] <= pd.Timestamp(AUDIT_AS_OF_DATE)]
    return (
        out.drop_duplicates(["code", "date"], keep="first")
        .sort_values(["code", "date"])
        .reset_index(drop=True)
    )


def _event_frame(panel: pd.DataFrame, extra_codes: tuple[str, ...] = BENCHMARK_CODES) -> pd.DataFrame:
    events = panel[["snapshot_date", "code"]].drop_duplicates().copy()
    extras = pd.DataFrame(
        [
            (snapshot, code)
            for snapshot in sorted(panel["snapshot_date"].astype(str).unique().tolist())
            for code in extra_codes
        ],
        columns=["snapshot_date", "code"],
    )
    events = pd.concat([events, extras], ignore_index=True).drop_duplicates()
    events["snapshot_date"] = events["snapshot_date"].astype(str)
    events["code"] = events["code"].astype(str).str.upper().str.strip()
    return events


def _tradable_event_valid(code_prices: pd.DataFrame, snapshot: str) -> bool:
    if code_prices.empty:
        return False
    snap = pd.Timestamp(snapshot)
    asof = pd.Timestamp(AUDIT_AS_OF_DATE)

    g = code_prices.sort_values("date")
    dates = g["date"].to_numpy(dtype="datetime64[ns]")
    snap64 = np.datetime64(snap.to_datetime64())

    entry_idx = int(np.searchsorted(dates, snap64, side="right"))
    if entry_idx >= len(g):
        return False
    entry_date = pd.Timestamp(dates[entry_idx])
    target = entry_date + pd.Timedelta(days=SNAPSHOT_FORWARD_DAYS)
    if target > asof:
        return False

    target64 = np.datetime64(target.to_datetime64())
    end_idx = int(np.searchsorted(dates, target64, side="right") - 1)
    if end_idx < entry_idx:
        return False

    end_date = pd.Timestamp(dates[end_idx])
    if (entry_date - snap).days > 4:
        return False
    if (target - end_date).days > 4:
        return False

    entry_open = pd.to_numeric(pd.Series([g.iloc[entry_idx]["open"]]), errors="coerce").iloc[0]
    end_close = pd.to_numeric(pd.Series([g.iloc[end_idx]["close"]]), errors="coerce").iloc[0]
    return bool(
        pd.notna(entry_open)
        and pd.notna(end_close)
        and float(entry_open) > 0
    )


def find_codes_needing_yahoo(panel: pd.DataFrame, base_prices: pd.DataFrame) -> list[str]:
    """Download only symbols that cannot support at least one mature audit event.

    Benchmarks are always included so SPY/QQQ come from the same frozen Yahoo pull.
    """
    prices = _normalize_price_frame(base_prices)
    by_code = {code: g.copy() for code, g in prices.groupby("code", sort=False)}
    needed: set[str] = set(BENCHMARK_CODES)

    for _, event in _event_frame(panel).iterrows():
        snapshot = str(event["snapshot_date"])
        # Entry can occur up to four calendar days after snapshot. If even the
        # earliest possible next-open + 28d is not mature, no supplement can
        # make the outcome valid yet.
        earliest_target = (
            pd.Timestamp(snapshot)
            + pd.Timedelta(days=1)
            + pd.Timedelta(days=SNAPSHOT_FORWARD_DAYS)
        )
        if earliest_target > pd.Timestamp(AUDIT_AS_OF_DATE):
            continue
        code = str(event["code"])
        if not _tradable_event_valid(by_code.get(code, pd.DataFrame()), snapshot):
            needed.add(code)

    return sorted(needed)


def download_yahoo_supplement(
    panel: pd.DataFrame,
    base_prices: pd.DataFrame,
    *,
    provider: YahooDataProvider | None = None,
    supplement_path: Path = YAHOO_SUPPLEMENT_PARQUET,
    audit_path: Path = YAHOO_DOWNLOAD_AUDIT_CSV,
) -> YahooSupplementResult:
    """Use the repository's YahooDataProvider without mutating the shared base cache.

    Existing cache bars win on duplicate code/date. Yahoo is used only to fill gaps
    and to provide SPY/QQQ. Exact downloaded rows are persisted under this audit's
    output directory for provenance.
    """
    base = _normalize_price_frame(base_prices)
    needed_original = find_codes_needing_yahoo(panel, base_prices)
    resolver: dict[str, list[tuple[str, str]]] = {}
    for orig in needed_original:
        resolved, reason = resolve_symbol_for_provider(orig, provider="yahoo")
        resolver.setdefault(resolved, []).append((orig, reason))

    provider = provider or YahooDataProvider(batch_size=100, max_workers=8, max_retries=2)
    resolved_symbols = sorted(resolver)
    downloaded: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    if resolved_symbols:
        downloaded, failed = provider.download_batch_stocks(
            resolved_symbols,
            period="5y",
            interval="1d",
        )

    supplement_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []

    failed_set = set(failed)
    for resolved in resolved_symbols:
        mappings = resolver[resolved]
        raw = downloaded.get(resolved)
        for orig, reason in mappings:
            if raw is None or raw.empty:
                audit_rows.append({
                    "original_code": orig,
                    "resolved_code": resolved,
                    "resolution_reason": reason,
                    "status": "NO_PROVIDER_DATA" if resolved in failed_set else "EMPTY",
                    "bars": 0,
                    "first_date": "",
                    "last_date": "",
                })
                continue

            std = _standardize_ohlcv_df(raw, orig, source="yahoo_audit_v1_1")
            if std.empty:
                audit_rows.append({
                    "original_code": orig,
                    "resolved_code": resolved,
                    "resolution_reason": reason,
                    "status": "EMPTY_STANDARDIZED",
                    "bars": 0,
                    "first_date": "",
                    "last_date": "",
                })
                continue

            std["date"] = pd.to_datetime(std["date"])
            std = std[std["date"] <= pd.Timestamp(AUDIT_AS_OF_DATE)]
            supplement_parts.append(std)
            audit_rows.append({
                "original_code": orig,
                "resolved_code": resolved,
                "resolution_reason": reason,
                "status": "OK",
                "bars": int(len(std)),
                "first_date": "" if std.empty else str(std["date"].min().date()),
                "last_date": "" if std.empty else str(std["date"].max().date()),
            })

    supplement = (
        _normalize_price_frame(pd.concat(supplement_parts, ignore_index=True))
        if supplement_parts
        else _normalize_price_frame(pd.DataFrame())
    )

    missing_benchmarks = [
        code
        for code in BENCHMARK_CODES
        if supplement[supplement["code"] == code].empty
    ]
    if missing_benchmarks:
        raise RuntimeError(
            "Yahoo benchmark download failed for: "
            + ", ".join(missing_benchmarks)
        )

    supplement_path.parent.mkdir(parents=True, exist_ok=True)
    supplement.to_parquet(supplement_path, index=False, engine="pyarrow")

    audit = pd.DataFrame(audit_rows)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)

    if supplement.empty:
        merged = base.copy()
    elif base.empty:
        merged = supplement.copy()
    else:
        # Existing frozen cache is authoritative where populated; Yahoo fills
        # missing OHLCV cells and missing dates only.
        key = ["code", "date"]
        base_i = base.set_index(key)
        supp_i = supplement.set_index(key)
        merged = base_i.combine_first(supp_i).reset_index()
        merged = _normalize_price_frame(merged)
    return YahooSupplementResult(prices=merged, supplement=supplement, audit=audit)


def build_next_open_forward_returns(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    extra_codes: tuple[str, ...] = BENCHMARK_CODES,
) -> pd.DataFrame:
    """Tradable audit outcome: next-session open -> close at entry + 28d.

    Stop8 includes the entry session's daily low because entry occurs at that
    session's open. Outcomes are frozen at AUDIT_AS_OF_DATE.
    """
    events = _event_frame(panel, extra_codes)
    prices = _normalize_price_frame(prices)
    grouped = {code: g.copy() for code, g in prices.groupby("code", sort=False)}
    rows: list[dict[str, Any]] = []
    asof = pd.Timestamp(AUDIT_AS_OF_DATE)

    for _, event in events.iterrows():
        snapshot = str(event["snapshot_date"])
        code = str(event["code"])
        snap = pd.Timestamp(snapshot)
        g = grouped.get(code)

        record = {
            "snapshot_date": snapshot,
            "code": code,
            "next_open_w4_return_pct": np.nan,
            "next_open_w4_stop8": np.nan,
            "next_open_entry_date": None,
            "next_open_end_date": None,
            "next_open_price_valid": False,
            "next_open_invalid_reason": "",
        }

        if (
            snap
            + pd.Timedelta(days=1)
            + pd.Timedelta(days=SNAPSHOT_FORWARD_DAYS)
            > asof
        ):
            record["next_open_invalid_reason"] = "HORIZON_NOT_MATURE_AS_OF"
            rows.append(record)
            continue
        if g is None or g.empty:
            record["next_open_invalid_reason"] = "NO_PRICE_DATA"
            rows.append(record)
            continue

        dates = g["date"].to_numpy(dtype="datetime64[ns]")
        entry_idx = int(np.searchsorted(dates, np.datetime64(snap.to_datetime64()), side="right"))
        if entry_idx >= len(g):
            record["next_open_invalid_reason"] = "NO_NEXT_SESSION"
            rows.append(record)
            continue
        entry_date = pd.Timestamp(dates[entry_idx])
        target = entry_date + pd.Timedelta(days=SNAPSHOT_FORWARD_DAYS)
        if target > asof:
            record["next_open_invalid_reason"] = "HORIZON_NOT_MATURE_AS_OF"
            rows.append(record)
            continue
        end_idx = int(np.searchsorted(dates, np.datetime64(target.to_datetime64()), side="right") - 1)
        if end_idx < entry_idx:
            record["next_open_invalid_reason"] = "NO_END_BAR"
            rows.append(record)
            continue

        end_date = pd.Timestamp(dates[end_idx])
        if (entry_date - snap).days > 4:
            record["next_open_invalid_reason"] = "ENTRY_TOO_STALE"
            rows.append(record)
            continue
        if (target - end_date).days > 4:
            record["next_open_invalid_reason"] = "END_TOO_STALE"
            rows.append(record)
            continue

        entry_open = pd.to_numeric(pd.Series([g.iloc[entry_idx]["open"]]), errors="coerce").iloc[0]
        end_close = pd.to_numeric(pd.Series([g.iloc[end_idx]["close"]]), errors="coerce").iloc[0]
        if pd.isna(entry_open) or pd.isna(end_close) or float(entry_open) <= 0:
            record["next_open_invalid_reason"] = "INVALID_OHLC"
            rows.append(record)
            continue

        lows = pd.to_numeric(
            g.iloc[entry_idx : end_idx + 1]["low"],
            errors="coerce",
        ).dropna()
        stop8 = bool((lows <= float(entry_open) * 0.92).any()) if not lows.empty else False
        ret = (float(end_close) / float(entry_open) - 1.0) * 100.0

        record.update({
            "next_open_w4_return_pct": round(float(ret), 6),
            "next_open_w4_stop8": stop8,
            "next_open_entry_date": str(entry_date.date()),
            "next_open_end_date": str(end_date.date()),
            "next_open_price_valid": True,
            "next_open_invalid_reason": "",
        })
        rows.append(record)

    return pd.DataFrame(rows)



def spy_momentum_asof(
    prices: pd.DataFrame,
    snapshots: list[str],
    *,
    sessions: int = 20,
) -> pd.DataFrame:
    """PIT SPY momentum at each snapshot close, using only bars <= snapshot."""
    p = _normalize_price_frame(prices)
    spy = p[p["code"] == "SPY"].sort_values("date").copy()
    if spy.empty:
        raise RuntimeError("SPY price history unavailable for relative-momentum audit")

    spy["close"] = pd.to_numeric(spy["close"], errors="coerce")
    spy["spy_momentum"] = spy["close"] / spy["close"].shift(sessions) - 1.0
    events = pd.DataFrame({
        "snapshot_date": [str(x) for x in sorted(set(snapshots))]
    })
    events["snapshot_dt"] = pd.to_datetime(events["snapshot_date"])
    merged = pd.merge_asof(
        events.sort_values("snapshot_dt"),
        spy[["date", "spy_momentum"]].sort_values("date"),
        left_on="snapshot_dt",
        right_on="date",
        direction="backward",
        allow_exact_matches=True,
    )
    if (merged["date"] > merged["snapshot_dt"]).fillna(False).any():
        raise AssertionError("future SPY leakage in relative-momentum audit")
    return merged[["snapshot_date", "spy_momentum"]]
