import os
import json
import time
import datetime
from typing import Dict, Any, List, Optional
import requests
from eps_pit.providers.base import BaseFundamentalsProvider


class FMPProvider(BaseFundamentalsProvider):
    """Financial Modeling Prep (FMP) Provider.
    
    Reads FMP_API_KEY from environment.
    """

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None):
        super().__init__(cache_dir=cache_dir)
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        self.raw_fmp_dir = os.path.join(self.cache_dir, "FMP")
        os.makedirs(self.raw_fmp_dir, exist_ok=True)

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def fetch_quarterly_history(self, symbol: str) -> List[Dict[str, Any]]:
        raw_file = os.path.join(self.raw_fmp_dir, f"{symbol.upper()}.json")
        data = None

        if os.path.exists(raw_file):
            try:
                with open(raw_file, "r") as f:
                    data = json.load(f)
            except Exception:
                data = None

        if data is None:
            if not self.api_key:
                # Key not available
                return []

            url = f"{self.BASE_URL}/income-statement/{symbol}?period=quarter&apikey={self.api_key}"
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    with open(raw_file, "w") as f:
                        json.dump(data, f)
                else:
                    return []
            except Exception as e:
                print(f"[FMPProvider] Error fetching {symbol}: {e}")
                return []

        if not isinstance(data, list):
            return []

        records = []
        for row in data:
            date_end = row.get("date")
            filling_date = row.get("fillingDate")
            accepted_date = row.get("acceptedDate")
            period = row.get("period")  # Q1, Q2, Q3, Q4
            calendar_year = row.get("calendarYear")
            eps = row.get("epsdiluted")
            if eps is None:
                eps = row.get("eps")

            if eps is not None and date_end:
                records.append({
                    "code": symbol.upper(),
                    "source_symbol": symbol.upper(),
                    "fiscal_year": int(calendar_year) if calendar_year and str(calendar_year).isdigit() else None,
                    "fiscal_quarter": period,
                    "report_period": date_end,
                    "period_start": None,
                    "period_end": date_end,
                    "eps_diluted": float(eps),
                    "filing_date": filling_date,
                    "accepted_at": accepted_date,
                    "earnings_release_at": accepted_date or filling_date,
                    "source": "FMP",
                    "source_record_id": f"fmp_{date_end}_{period}",
                })

        return sorted(records, key=lambda x: x["report_period"])

    def fetch_earnings_events(self, symbol: str) -> List[Dict[str, Any]]:
        history = self.fetch_quarterly_history(symbol)
        events = []
        for h in history:
            events.append({
                "code": symbol.upper(),
                "report_period": h["report_period"],
                "fiscal_year": h["fiscal_year"],
                "fiscal_quarter": h["fiscal_quarter"],
                "eps_diluted": h["eps_diluted"],
                "event_date": h["earnings_release_at"] or h["filing_date"],
                "event_type": "FMP_EARNINGS",
                "source": "FMP",
            })
        return events
