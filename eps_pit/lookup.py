import os
from typing import Dict, Any, Optional, Tuple
import pandas as pd


class SignalEPSLookup:
    """Point-in-Time Signal EPS Lookup & Enrichment Service.
    
    Provides O(1) in-memory lookup and dynamic dataframe enrichment for
    signal candidates across weekly replay snapshots when the base pool
    does not contain pre-filled eps_yoy_growth.
    """

    DEFAULT_CSV_PATH = "backtest/ibd_skill_replay_pools/signal_eps_pit.csv"
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

        if fp is None or not os.path.exists(path):
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
                    val = float(raw_eps)
                    eps_map[(snap, sym)] = val
                except (ValueError, TypeError):
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
        csv_path: Optional[str] = None
    ) -> Optional[float]:
        """Look up point-in-time EPS YoY growth for a specific snapshot and ticker."""
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
        csv_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Look up complete point-in-time provenance record."""
        cls.load(csv_path)
        snap = cls._normalize_date(snapshot_date)
        sym = cls._normalize_ticker(code)
        if cls._record_cache is None:
            return None
        return cls._record_cache.get((snap, sym))

    @classmethod
    def enrich_pool(
        cls,
        pool_df: pd.DataFrame,
        snapshot_date: Optional[object] = None,
        csv_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Enriches pool DataFrame by filling missing eps_yoy_growth for signal rows."""
        if pool_df.empty:
            return pool_df.copy()

        cls.load(csv_path)
        df = pool_df.copy()

        if "eps_yoy_growth" not in df.columns:
            df["eps_yoy_growth"] = None

        default_snap = cls._normalize_date(snapshot_date) if snapshot_date else ""

        # Identify rows needing EPS lookup
        for idx, row in df.iterrows():
            curr_eps = row.get("eps_yoy_growth")
            if pd.notna(curr_eps):
                continue

            snap = cls._normalize_date(row.get("snapshot_date")) or default_snap
            sym = cls._normalize_ticker(row.get("code"))
            if not snap or not sym:
                continue

            eps_val = cls._eps_cache.get((snap, sym)) if cls._eps_cache else None
            if eps_val is not None:
                df.at[idx, "eps_yoy_growth"] = eps_val

        return df


def get_signal_eps(snapshot_date: object, code: object, csv_path: Optional[str] = None) -> Optional[float]:
    """Convenience functional wrapper for SignalEPSLookup.get_eps."""
    return SignalEPSLookup.get_eps(snapshot_date, code, csv_path=csv_path)


def enrich_pool_with_signal_eps(
    pool_df: pd.DataFrame,
    snapshot_date: Optional[object] = None,
    csv_path: Optional[str] = None
) -> pd.DataFrame:
    """Convenience functional wrapper for SignalEPSLookup.enrich_pool."""
    return SignalEPSLookup.enrich_pool(pool_df, snapshot_date=snapshot_date, csv_path=csv_path)
