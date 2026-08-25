"""Ticker symbol resolution, normalization, and master catalog maintenance."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Valid download and lifecycle statuses
VALID_STATUSES = {
    "OK",
    "PARTIAL_HISTORY",
    "ALIAS_RESOLVED",
    "DELISTED_CONFIRMED",
    "POSSIBLE_DELISTED_UNVERIFIED",
    "INVALID_SYMBOL",
    "DOWNLOAD_ERROR_RETRYABLE",
    "NO_PROVIDER_DATA",
    "CORPORATE_ACTION_SUSPECT",
}

# Known Yahoo symbol normalization mappings
KNOWN_YAHOO_ALIASES: dict[str, str] = {
    "BRK.B": "BRK-B",
    "BRK/B": "BRK-B",
    "BRK.A": "BRK-A",
    "BRK/A": "BRK-A",
    "BF.B": "BF-B",
    "BF/B": "BF-B",
    "BF.A": "BF-A",
    "BF/A": "BF-A",
}


def resolve_symbol_for_provider(symbol: str, provider: str = "yahoo") -> tuple[str, str]:
    """Normalize symbol for specific data provider.
    
    Returns:
        (resolved_symbol, reason)
    """
    clean = str(symbol).strip().upper()
    if not clean:
        return "", "empty_symbol"

    if provider.lower() == "yahoo":
        if clean in KNOWN_YAHOO_ALIASES:
            return KNOWN_YAHOO_ALIASES[clean], f"known_alias_{clean}_to_{KNOWN_YAHOO_ALIASES[clean]}"
        # If contains dot (e.g. CWEN.A -> CWEN-A)
        if "." in clean and not clean.startswith("^"):
            alias = clean.replace(".", "-")
            return alias, f"dot_to_dash_{clean}_to_{alias}"
        if "/" in clean:
            alias = clean.replace("/", "-")
            return alias, f"slash_to_dash_{clean}_to_{alias}"

    return clean, "exact_match"


def build_ticker_master(
    events_df: pd.DataFrame,
    existing_master_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Build or update ticker_master from review universe events.
    
    Extracts unique tickers, computes snapshot statistics, and assigns initial download status.
    """
    if events_df.empty:
        master_df = pd.DataFrame(
            columns=[
                "original_code",
                "resolved_code",
                "first_snapshot_date",
                "last_snapshot_date",
                "signal_event_count",
                "first_price_date",
                "last_price_date",
                "last_valid_close",
                "provider",
                "download_status",
                "retry_count",
                "resolution_reason",
                "notes",
            ]
        )
        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            master_df.to_csv(out_p, index=False, encoding="utf-8-sig")
        return master_df

    # Aggregate stats per unique original code
    grouped = (
        events_df.groupby("code")
        .agg(
            first_snapshot_date=("snapshot_date", "min"),
            last_snapshot_date=("snapshot_date", "max"),
            signal_event_count=("snapshot_date", "count"),
        )
        .reset_index()
    )

    rows: list[dict[str, Any]] = []
    existing_map: dict[str, dict[str, Any]] = {}

    if existing_master_path and Path(existing_master_path).exists():
        try:
            prev_df = pd.read_csv(existing_master_path, encoding="utf-8-sig")
            for _, r in prev_df.iterrows():
                code_key = str(r["original_code"]).strip().upper()
                existing_map[code_key] = r.to_dict()
        except Exception as e:
            logger.warning(f"Could not load existing ticker master from {existing_master_path}: {e}")

    for _, row in grouped.iterrows():
        orig_code = str(row["code"]).strip().upper()
        resolved_code, reason = resolve_symbol_for_provider(orig_code, provider="yahoo")

        prev = existing_map.get(orig_code, {})

        entry = {
            "original_code": orig_code,
            "resolved_code": resolved_code,
            "first_snapshot_date": row["first_snapshot_date"],
            "last_snapshot_date": row["last_snapshot_date"],
            "signal_event_count": int(row["signal_event_count"]),
            "first_price_date": prev.get("first_price_date", ""),
            "last_price_date": prev.get("last_price_date", ""),
            "last_valid_close": prev.get("last_valid_close", None),
            "provider": prev.get("provider", "yahoo"),
            "download_status": prev.get("download_status", "DOWNLOAD_ERROR_RETRYABLE"),
            "retry_count": int(prev.get("retry_count", 0)),
            "resolution_reason": reason if not prev.get("resolution_reason") else prev.get("resolution_reason"),
            "notes": prev.get("notes", ""),
        }
        rows.append(entry)

    master_df = pd.DataFrame(rows)
    master_df = master_df.sort_values("original_code").reset_index(drop=True)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        master_df.to_csv(out_p, index=False, encoding="utf-8-sig")
        logger.info(f"Saved {len(master_df)} tickers to master catalog: {out_p}")

    return master_df


def update_ticker_master_with_prices(
    master_df: pd.DataFrame,
    price_coverage: dict[str, dict[str, Any]],
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Update ticker_master with actual price download results."""
    updated = master_df.copy()

    for idx, row in updated.iterrows():
        orig = str(row["original_code"]).strip().upper()
        resolved = str(row["resolved_code"]).strip().upper()
        cov = price_coverage.get(orig) or price_coverage.get(resolved)

        if cov is not None:
            updated.at[idx, "first_price_date"] = cov.get("first_price_date", "")
            updated.at[idx, "last_price_date"] = cov.get("last_price_date", "")
            updated.at[idx, "last_valid_close"] = cov.get("last_valid_close", None)
            updated.at[idx, "download_status"] = cov.get("download_status", "OK")
            updated.at[idx, "retry_count"] = int(cov.get("retry_count", row["retry_count"]))
            updated.at[idx, "notes"] = cov.get("notes", row["notes"])
        else:
            if not updated.at[idx, "download_status"] or updated.at[idx, "download_status"] == "OK":
                updated.at[idx, "download_status"] = "NO_PROVIDER_DATA"

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        updated.to_csv(out_p, index=False, encoding="utf-8-sig")

    return updated
