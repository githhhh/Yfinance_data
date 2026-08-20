import os
import datetime
from typing import Dict, Any, List, Optional
from eps_pit.providers.base import BaseFundamentalsProvider
from eps_pit.providers.sec_provider import SECProvider
from eps_pit.providers.yahoo_provider import YahooFundamentalsProvider
from eps_pit.providers.fmp_provider import FMPProvider


class CompositeFundamentalsProvider(BaseFundamentalsProvider):
    """Composite Provider orchestrating SEC EDGAR, FMP, and Yahoo Finance.
    
    Extracts As-Filed SEC dates for conservative PIT and Yahoo earnings release
    timestamps for release-based PIT.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(cache_dir=cache_dir)
        self.sec = SECProvider(cache_dir=self.cache_dir)
        self.yahoo = YahooFundamentalsProvider(cache_dir=self.cache_dir)
        self.fmp = FMPProvider(cache_dir=self.cache_dir)

    def fetch_quarterly_history(self, symbol: str) -> List[Dict[str, Any]]:
        # 1. Check FMP if configured
        if self.fmp.is_configured():
            fmp_data = self.fmp.fetch_quarterly_history(symbol)
            if fmp_data:
                return fmp_data

        # 2. Try SEC EDGAR (official, exact diluted EPS + filing date)
        sec_data = self.sec.fetch_quarterly_history(symbol)
        
        # 3. Fetch Yahoo data to supplement earnings_release_at and handle ADR/Foreign/latest quarters
        yahoo_data = self.yahoo.fetch_quarterly_history(symbol)
        
        if not sec_data and not yahoo_data:
            return []

        if not sec_data:
            # Foreign issuer / ADR (e.g. ASML, ARM, TSM) -> use Yahoo
            return yahoo_data

        if not yahoo_data:
            # Only SEC available
            return sec_data

        # Merge SEC and Yahoo:
        # SEC provides the core As-Filed history with official filing dates
        combined = list(sec_data)
        sec_periods = {r["report_period"] for r in sec_data if r.get("report_period")}
        sec_max_period = max(sec_periods) if sec_periods else ""

        # Map Yahoo release dates to SEC records
        for sec_rec in combined:
            sec_end = sec_rec.get("report_period")
            sec_filed = sec_rec.get("filing_date")
            
            matched_release = None
            for y_rec in yahoo_data:
                y_date = y_rec.get("report_period")
                if y_date and sec_end:
                    # Release date is around the report period end
                    try:
                        d_sec = datetime.date.fromisoformat(sec_end)
                        d_y = datetime.date.fromisoformat(y_date)
                        if 0 <= (d_y - d_sec).days <= 60:
                            matched_release = y_rec.get("earnings_release_at") or y_date
                            break
                    except Exception:
                        pass
            
            sec_rec["earnings_release_at"] = matched_release or sec_filed

        # Add newer quarters from Yahoo if SEC hasn't filed yet
        for y_rec in yahoo_data:
            y_date = y_rec.get("report_period")
            if y_date and y_date > sec_max_period:
                # Add newer quarter from Yahoo
                combined.append(y_rec)

        # Sort chronologically
        return sorted(combined, key=lambda x: (x.get("report_period") or "", x.get("filing_date") or ""))

    def fetch_earnings_events(self, symbol: str) -> List[Dict[str, Any]]:
        return self.yahoo.fetch_earnings_events(symbol) or self.sec.fetch_earnings_events(symbol)


class ProviderFactory:
    """Factory to obtain data providers."""

    @classmethod
    def get_provider(cls, name: str = "composite", cache_dir: Optional[str] = None) -> BaseFundamentalsProvider:
        name_lower = name.lower()
        if name_lower == "sec":
            return SECProvider(cache_dir=cache_dir)
        elif name_lower == "yahoo":
            return YahooFundamentalsProvider(cache_dir=cache_dir)
        elif name_lower == "fmp":
            return FMPProvider(cache_dir=cache_dir)
        elif name_lower == "composite":
            return CompositeFundamentalsProvider(cache_dir=cache_dir)
        else:
            raise ValueError(f"Unknown provider: {name}")
