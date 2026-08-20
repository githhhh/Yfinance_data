import os
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


class TickerMapper:
    """Ticker Normalization & Mapping Engine."""

    SPECIAL_MAPPINGS = {
        "BRK.B": "BRK-B",
        "BF.B": "BF-B",
        "JW.A": "JW-A",
        "HEI.A": "HEI-A",
        "LEN.B": "LEN-B",
        "MOG.A": "MOG-A",
        "CW.EN": "CWEN",
    }

    def __init__(self, sec_provider=None):
        self.sec_provider = sec_provider

    def normalize_ticker(self, code: str) -> str:
        """Normalize ticker to standard format."""
        if not code or not isinstance(code, str):
            return ""
        c = code.strip().upper()
        # Direct special mapping
        if c in self.SPECIAL_MAPPINGS:
            return self.SPECIAL_MAPPINGS[c]
        # Replace dot with dash for Yahoo/SEC compatibility if class share
        if "." in c and not c.startswith("^"):
            # Check if suffix is a single letter (class share like BF.B -> BF-B)
            parts = c.split(".")
            if len(parts) == 2 and len(parts[1]) == 1:
                return f"{parts[0]}-{parts[1]}"
        return c

    def map_ticker(self, code: str) -> Dict[str, Any]:
        raw_code = str(code).strip().upper()
        norm_code = self.normalize_ticker(raw_code)
        
        cik = None
        if self.sec_provider:
            cik = self.sec_provider.get_cik(raw_code) or self.sec_provider.get_cik(norm_code)

        if raw_code == norm_code and cik:
            status = "EXACT"
            method = "EXACT_SEC_MATCH"
        elif cik:
            status = "NORMALIZED"
            method = "NORMALIZED_SEC_MATCH"
        else:
            # Check if known symbol
            status = "NORMALIZED" if raw_code != norm_code else "EXACT"
            method = "FALLBACK_YAHOO"

        return {
            "code": raw_code,
            "source_symbol": norm_code,
            "cik": cik,
            "mapping_method": method,
            "mapping_status": status,
            "notes": f"Mapped to {norm_code}" if raw_code != norm_code else "Standard symbol",
        }

    def build_mapping_table(self, tickers: List[str]) -> pd.DataFrame:
        records = [self.map_ticker(t) for t in sorted(set(tickers))]
        return pd.DataFrame(records)
