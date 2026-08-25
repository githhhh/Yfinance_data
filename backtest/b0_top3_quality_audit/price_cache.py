"""Unified Daily Price Cache maintenance and incremental downloader."""

from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backtest.b0_top3_quality_audit.ticker_resolution import resolve_symbol_for_provider
from data_providers.yahoo_provider import YahooDataProvider

logger = logging.getLogger(__name__)

PRICE_PARQUET_COLUMNS = ["code", "date", "open", "high", "low", "close", "volume", "source"]


def compute_file_sha256(path: Path | str) -> str:
    """Compute sha256 hex digest of a file."""
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_existing_daily_pkl(pkl_path: Path | str = "results_pkl/stock_data_230826_1d.pkl") -> dict[str, pd.DataFrame]:
    """Read existing daily price pkl without modifying it."""
    import pickle

    p = Path(pkl_path)
    if not p.exists():
        logger.warning(f"Price pkl not found at {p}")
        return {}

    with open(p, "rb") as f:
        raw_dict = pickle.load(f)

    result: dict[str, pd.DataFrame] = {}
    for code, val in raw_dict.items():
        clean_code = str(code).strip().upper()
        if isinstance(val, pd.DataFrame):
            df = val.copy()
        elif isinstance(val, dict) and "data" in val and "columns" in val and "index" in val:
            df = pd.DataFrame(val["data"], index=val["index"], columns=val["columns"])
        else:
            continue

        # Standardize index to date strings YYYY-MM-DD
        if not df.empty:
            df = _standardize_ohlcv_df(df, clean_code, source="results_pkl")
            result[clean_code] = df

    return result


def _standardize_ohlcv_df(df: pd.DataFrame, code: str, source: str = "yahoo") -> pd.DataFrame:
    """Standardize OHLCV DataFrame into unified long table format."""
    out = df.copy()
    if "Date" in out.columns:
        date_series = pd.to_datetime(out["Date"])
    else:
        date_series = pd.to_datetime(out.index)

    if isinstance(date_series, pd.DatetimeIndex):
        if date_series.tz is not None:
            date_series = date_series.tz_localize(None)
        out["date"] = date_series.strftime("%Y-%m-%d")
    else:
        if getattr(date_series, "dt", None) is not None and date_series.dt.tz is not None:
            date_series = date_series.dt.tz_localize(None)
        out["date"] = date_series.dt.strftime("%Y-%m-%d") if hasattr(date_series, "dt") else [str(x)[:10] for x in date_series]
    out["code"] = str(code).strip().upper()

    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in out.columns:
            out[col_map[c]] = pd.to_numeric(out[c], errors="coerce")

    req_cols = ["open", "high", "low", "close", "volume"]
    for rc in req_cols:
        if rc not in out.columns:
            out[rc] = None

    out["source"] = source
    out = out.dropna(subset=["date", "close"])
    out = out[["code", "date", "open", "high", "low", "close", "volume", "source"]]
    out = out.drop_duplicates(subset=["code", "date"]).sort_values(["code", "date"]).reset_index(drop=True)
    return out


class DailyPriceCache:
    """Manages long-table daily prices in parquet format with incremental download and audit tracking."""

    def __init__(self, parquet_path: Path | str = "backtest/b0_top3_quality_audit/data/signal_daily_prices.parquet"):
        self.parquet_path = Path(parquet_path)
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        self.df: pd.DataFrame = self._load()

    def _load(self) -> pd.DataFrame:
        if self.parquet_path.exists():
            try:
                df = pd.read_parquet(self.parquet_path)
                if not df.empty and "date" in df.columns and "code" in df.columns:
                    return df.drop_duplicates(subset=["code", "date"]).sort_values(["code", "date"]).reset_index(drop=True)
            except Exception as e:
                logger.warning(f"Error loading {self.parquet_path}: {e}")
        return pd.DataFrame(columns=PRICE_PARQUET_COLUMNS)

    def save(self) -> None:
        """Save price DataFrame to parquet cleanly."""
        if not self.df.empty:
            self.df = (
                self.df.drop_duplicates(subset=["code", "date"])
                .sort_values(["code", "date"])
                .reset_index(drop=True)
            )
        self.df.to_parquet(self.parquet_path, index=False, engine="pyarrow")

    def get_prices_for_ticker(self, code: str) -> pd.DataFrame:
        """Get sorted daily price bars for a single ticker."""
        clean = str(code).strip().upper()
        if self.df.empty:
            return pd.DataFrame(columns=PRICE_PARQUET_COLUMNS)
        sub = self.df[self.df["code"] == clean]
        return sub.sort_values("date").reset_index(drop=True)

    def contains_ticker(self, code: str, start_date: str, end_date: str) -> bool:
        """Check if cache contains prices for ticker covering required date span."""
        sub = self.get_prices_for_ticker(code)
        if sub.empty:
            return False
        first_d = sub["date"].min()
        last_d = sub["date"].max()
        return bool(first_d <= start_date and last_d >= end_date)

    def append_bars(self, new_bars_df: pd.DataFrame) -> None:
        """Append and deduplicate new price bars."""
        if new_bars_df.empty:
            return
        if self.df.empty:
            self.df = new_bars_df.copy()
        else:
            self.df = pd.concat([self.df, new_bars_df], ignore_index=True)
        self.df = (
            self.df.drop_duplicates(subset=["code", "date"])
            .sort_values(["code", "date"])
            .reset_index(drop=True)
        )

    def build_or_update(
        self,
        ticker_master: pd.DataFrame,
        start_date: str = "2025-07-01",
        end_date: str = "2026-08-25",
        existing_pkl_path: Path | str = "results_pkl/stock_data_230826_1d.pkl",
        audit_csv_path: Path | str | None = "backtest/b0_top3_quality_audit/output/price_download_audit.csv",
        force_redownload: bool = False,
    ) -> Tuple[dict[str, dict[str, Any]], pd.DataFrame]:
        """Reconcile existing pkl, download missing tickers, and update cache and audit logs.
        
        Returns:
            (price_coverage_dict, audit_df)
        """
        audit_records: list[dict[str, Any]] = []
        coverage_dict: dict[str, dict[str, Any]] = {}

        # 1. Ingest existing pkl if cache is empty or needs baseline loading
        if self.df.empty and Path(existing_pkl_path).exists():
            logger.info(f"Ingesting baseline price pkl from {existing_pkl_path}...")
            pkl_data = load_existing_daily_pkl(existing_pkl_path)
            all_pkl_dfs = list(pkl_data.values())
            if all_pkl_dfs:
                combined_pkl = pd.concat(all_pkl_dfs, ignore_index=True)
                self.append_bars(combined_pkl)
                self.save()
                logger.info(f"Loaded {len(all_pkl_dfs)} tickers from baseline pkl.")

        # 2. Identify tickers needing download
        provider = YahooDataProvider(batch_size=100, max_workers=8, max_retries=2)
        needed_symbols: list[str] = []
        symbol_to_orig: dict[str, str] = {}

        for _, row in ticker_master.iterrows():
            orig_code = str(row["original_code"]).strip().upper()
            resolved_code = str(row["resolved_code"]).strip().upper() or orig_code
            first_req_date = str(row.get("first_snapshot_date") or start_date)

            # Check if we already have sufficient bars in cache
            sub = self.get_prices_for_ticker(orig_code)
            if sub.empty and resolved_code != orig_code:
                sub = self.get_prices_for_ticker(resolved_code)

            has_sufficient_data = False
            if not sub.empty and not force_redownload:
                min_d = sub["date"].min()
                max_d = sub["date"].max()
                # Accept if covers first snapshot date and has recent data
                if min_d <= first_req_date and max_d >= "2026-08-01":
                    has_sufficient_data = True
                    coverage_dict[orig_code] = {
                        "first_price_date": min_d,
                        "last_price_date": max_d,
                        "last_valid_close": float(sub.iloc[-1]["close"]),
                        "download_status": "OK" if max_d >= "2026-08-15" else "PARTIAL_HISTORY",
                        "retry_count": 0,
                        "notes": "from_cache",
                    }
                    audit_records.append({
                        "original_code": orig_code,
                        "resolved_code": resolved_code,
                        "action": "CACHE_HIT",
                        "bars_count": len(sub),
                        "first_price_date": min_d,
                        "last_price_date": max_d,
                        "status": "OK",
                        "source": sub.iloc[0]["source"] if "source" in sub.columns else "cache",
                    })

            if not has_sufficient_data:
                needed_symbols.append(resolved_code)
                symbol_to_orig[resolved_code] = orig_code

        # 3. Batch download missing symbols
        if needed_symbols:
            logger.info(f"Downloading {len(needed_symbols)} tickers via Yahoo provider...")
            downloaded, failed = provider.download_batch_stocks(
                needed_symbols, period="2y", interval="1d"
            )

            new_dfs: list[pd.DataFrame] = []
            for sym, raw_df in downloaded.items():
                orig_code = symbol_to_orig.get(sym, sym)
                std_df = _standardize_ohlcv_df(raw_df, orig_code, source="yahoo_download")
                if not std_df.empty:
                    new_dfs.append(std_df)
                    min_d = std_df["date"].min()
                    max_d = std_df["date"].max()
                    last_c = float(std_df.iloc[-1]["close"])
                    status = "OK" if max_d >= "2026-08-15" else "PARTIAL_HISTORY"
                    coverage_dict[orig_code] = {
                        "first_price_date": min_d,
                        "last_price_date": max_d,
                        "last_valid_close": last_c,
                        "download_status": status,
                        "retry_count": 0,
                        "notes": "downloaded_yahoo",
                    }
                    audit_records.append({
                        "original_code": orig_code,
                        "resolved_code": sym,
                        "action": "DOWNLOAD_SUCCESS",
                        "bars_count": len(std_df),
                        "first_price_date": min_d,
                        "last_price_date": max_d,
                        "status": status,
                        "source": "yahoo_download",
                    })

            for sym in failed:
                orig_code = symbol_to_orig.get(sym, sym)
                coverage_dict[orig_code] = {
                    "first_price_date": "",
                    "last_price_date": "",
                    "last_valid_close": None,
                    "download_status": "NO_PROVIDER_DATA",
                    "retry_count": 2,
                    "notes": "download_failed_yahoo",
                }
                audit_records.append({
                    "original_code": orig_code,
                    "resolved_code": sym,
                    "action": "DOWNLOAD_FAIL",
                    "bars_count": 0,
                    "first_price_date": "",
                    "last_price_date": "",
                    "status": "NO_PROVIDER_DATA",
                    "source": "yahoo_download",
                })

            if new_dfs:
                combined_new = pd.concat(new_dfs, ignore_index=True)
                self.append_bars(combined_new)
                self.save()
                logger.info(f"Successfully downloaded and saved {len(new_dfs)} new tickers.")

        audit_df = pd.DataFrame(audit_records)
        if audit_csv_path is not None:
            aud_p = Path(audit_csv_path)
            aud_p.parent.mkdir(parents=True, exist_ok=True)
            audit_df.to_csv(aud_p, index=False, encoding="utf-8-sig")

        return coverage_dict, audit_df
