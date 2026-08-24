import logging
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from eps_pit.providers.sec_yahoo_provider import SECYahooEPSProvider


class SignalEPSLookup:
    """Point-in-time EPS lookup and signal-only pool enrichment."""

    DEFAULT_CSV_PATH = "us/signal_eps_pit.csv"
    _eps_cache: Optional[Dict[Tuple[str, str], float]] = None
    _record_cache: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None
    _loaded_fingerprint: Optional[Tuple[str, float, int]] = None

    @classmethod
    def _normalize_ticker(cls, code: object) -> str:
        if code is None:
            return ""
        return str(code).strip().upper().replace(".", "-")

    @classmethod
    def _normalize_date(cls, date_val: object) -> str:
        if date_val is None:
            return ""
        return str(date_val).strip()[:10]

    @classmethod
    def _is_truthy(cls, value: object) -> bool:
        if value is None:
            return False
        try:
            if pd.isna(value):
                return False
        except Exception:
            pass
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y"}

    @classmethod
    def _compute_fingerprint(cls, path: str) -> Optional[Tuple[str, float, int]]:
        if not os.path.exists(path):
            return None
        try:
            st = os.stat(path)
            return (os.path.abspath(path), st.st_mtime, st.st_size)
        except OSError:
            return None

    @classmethod
    def load(cls, csv_path: Optional[str] = None) -> None:
        path = csv_path or cls.DEFAULT_CSV_PATH
        fp = cls._compute_fingerprint(path)

        if cls._eps_cache is not None and cls._loaded_fingerprint == fp:
            return

        if fp is None:
            cls._eps_cache = {}
            cls._record_cache = {}
            cls._loaded_fingerprint = fp
            return

        df = pd.read_csv(path)
        eps_map: Dict[Tuple[str, str], float] = {}
        rec_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for _, row in df.iterrows():
            snap = cls._normalize_date(row.get("snapshot_date"))
            sym = cls._normalize_ticker(row.get("code"))
            if not snap or not sym:
                continue

            raw_eps = row.get("eps_yoy_growth")
            if pd.notna(raw_eps):
                try:
                    eps_map[(snap, sym)] = float(raw_eps)
                except (TypeError, ValueError):
                    pass

            rec_map[(snap, sym)] = row.to_dict()

        cls._eps_cache = eps_map
        cls._record_cache = rec_map
        cls._loaded_fingerprint = fp

    @classmethod
    def clear_cache(cls) -> None:
        cls._eps_cache = None
        cls._record_cache = None
        cls._loaded_fingerprint = None

    @classmethod
    def get_eps(
        cls,
        snapshot_date: object,
        code: object,
        csv_path: Optional[str] = None,
    ) -> Optional[float]:
        cls.load(csv_path)
        snap = cls._normalize_date(snapshot_date)
        sym = cls._normalize_ticker(code)
        if cls._eps_cache is None:
            return None
        return cls._eps_cache.get((snap, sym))

    @classmethod
    def get_record(
        cls,
        snapshot_date: object,
        code: object,
        csv_path: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        cls.load(csv_path)
        snap = cls._normalize_date(snapshot_date)
        sym = cls._normalize_ticker(code)
        if cls._record_cache is None:
            return None
        return cls._record_cache.get((snap, sym))

    @classmethod
    def fetch_sec_yahoo_eps(
        cls,
        snapshot_date: object,
        codes: list[str],
    ) -> Dict[str, Dict[str, Any]]:
        snap = cls._normalize_date(snapshot_date)
        symbols = sorted({cls._normalize_ticker(code) for code in codes if cls._normalize_ticker(code)})
        if not snap or not symbols:
            return {}

        provider = SECYahooEPSProvider()
        results: Dict[str, Dict[str, Any]] = {}
        for symbol in symbols:
            try:
                record = provider.fetch_eps_yoy(symbol, snap)
            except Exception as exc:
                logging.warning("Signal EPS SEC/Yahoo refresh failed for %s: %s", symbol, exc)
                continue
            if record is not None:
                results[symbol] = record
        return results

    @classmethod
    def _apply_eps_record(cls, df: pd.DataFrame, idx: Any, record: Dict[str, Any]) -> None:
        try:
            eps_val = float(record["eps_yoy_growth"])
        except (KeyError, TypeError, ValueError):
            return
        df.at[idx, "eps_yoy_growth"] = eps_val
        df.at[idx, "eps_yoy_growth_source"] = record.get("source")

    @classmethod
    def enrich_pool(
        cls,
        pool_df: pd.DataFrame,
        snapshot_date: Optional[object] = None,
        csv_path: Optional[str] = None,
        stage2_path: Optional[str] = None,
        refresh_missing: bool = False,
    ) -> pd.DataFrame:
        if pool_df.empty:
            return pool_df.copy()

        cls.load(csv_path)
        df = pool_df.copy()

        if "eps_yoy_growth" not in df.columns:
            df["eps_yoy_growth"] = pd.NA
        if "eps_yoy_growth_source" not in df.columns:
            df["eps_yoy_growth_source"] = pd.NA

        default_snap = cls._normalize_date(snapshot_date) if snapshot_date else ""
        has_signal = "signal" in df.columns

        for idx, row in df.iterrows():
            if has_signal and not cls._is_truthy(row.get("signal")):
                continue
            curr_eps = row.get("eps_yoy_growth")
            if pd.notna(curr_eps):
                continue

            snap = cls._normalize_date(row.get("snapshot_date")) or default_snap
            sym = cls._normalize_ticker(row.get("code"))
            if not sym:
                continue

            eps_val = None
            if snap and cls._eps_cache:
                eps_val = cls._eps_cache.get((snap, sym))
            if eps_val is not None:
                df.at[idx, "eps_yoy_growth"] = eps_val
                df.at[idx, "eps_yoy_growth_source"] = "PIT"

        if refresh_missing and has_signal and "code" in df.columns:
            signal_mask = df["signal"].map(cls._is_truthy)
            missing_mask = signal_mask & df["eps_yoy_growth"].isna()
            missing_rows = df.loc[missing_mask]
            for snap, group in missing_rows.groupby(
                missing_rows["snapshot_date"].map(cls._normalize_date)
                if "snapshot_date" in missing_rows.columns
                else pd.Series([""] * len(missing_rows), index=missing_rows.index)
            ):
                snap = snap or default_snap
                missing_codes = sorted(
                    {
                        cls._normalize_ticker(code)
                        for code in group["code"].dropna().tolist()
                        if cls._normalize_ticker(code)
                    }
                )
                refreshed = cls.fetch_sec_yahoo_eps(snap, missing_codes)
                if not refreshed:
                    continue
                for idx, row in group.iterrows():
                    sym = cls._normalize_ticker(row.get("code"))
                    record = refreshed.get(sym)
                    if record is None:
                        continue
                    cls._apply_eps_record(df, idx, record)

        return df


def get_signal_eps(
    snapshot_date: object,
    code: object,
    csv_path: Optional[str] = None,
) -> Optional[float]:
    return SignalEPSLookup.get_eps(snapshot_date, code, csv_path=csv_path)


def enrich_pool_with_signal_eps(
    pool_df: pd.DataFrame,
    snapshot_date: Optional[object] = None,
    csv_path: Optional[str] = None,
    stage2_path: Optional[str] = None,
    refresh_missing: bool = False,
) -> pd.DataFrame:
    return SignalEPSLookup.enrich_pool(
        pool_df,
        snapshot_date=snapshot_date,
        csv_path=csv_path,
        stage2_path=stage2_path,
        refresh_missing=refresh_missing,
    )
