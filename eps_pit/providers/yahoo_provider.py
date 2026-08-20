import os
import json
import time
import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import yfinance as yf
from eps_pit.providers.base import BaseFundamentalsProvider


class YahooFundamentalsProvider(BaseFundamentalsProvider):
    """Yahoo Finance Fundamentals & Earnings Release Provider.
    
    Provides quarterly earnings announcement timestamps and reported EPS.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(cache_dir=cache_dir)
        self.raw_yahoo_dir = os.path.join(self.cache_dir, "Yahoo")
        os.makedirs(self.raw_yahoo_dir, exist_ok=True)

    def fetch_quarterly_history(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch quarterly income stmt and earnings dates from Yahoo."""
        raw_file = os.path.join(self.raw_yahoo_dir, f"{symbol.upper()}.json")
        data = None

        if os.path.exists(raw_file):
            try:
                with open(raw_file, "r") as f:
                    data = json.load(f)
            except Exception:
                data = None

        if data is None:
            try:
                tk = yf.Ticker(symbol.upper().replace(".", "-"))
                
                # Fetch earnings dates
                ed_df = tk.get_earnings_dates(limit=32)
                ed_records = []
                if ed_df is not None and not ed_df.empty:
                    for idx, row in ed_df.iterrows():
                        rep_eps = row.get("Reported EPS")
                        if pd.notna(rep_eps):
                            ed_records.append({
                                "release_date": str(idx.strftime("%Y-%m-%d")),
                                "release_datetime": str(idx.isoformat()),
                                "reported_eps": float(rep_eps),
                                "estimate_eps": float(row.get("EPS Estimate")) if pd.notna(row.get("EPS Estimate")) else None,
                            })

                # Fetch quarterly income statement
                inc_df = tk.quarterly_income_stmt
                inc_records = []
                if inc_df is not None and not inc_df.empty:
                    eps_row = None
                    if "Diluted EPS" in inc_df.index:
                        eps_row = inc_df.loc["Diluted EPS"]
                    elif "Basic EPS" in inc_df.index:
                        eps_row = inc_df.loc["Basic EPS"]

                    if eps_row is not None:
                        for col_date, val in eps_row.items():
                            if pd.notna(val):
                                inc_records.append({
                                    "period_end": pd.to_datetime(col_date).strftime("%Y-%m-%d"),
                                    "eps_diluted": float(val),
                                })

                data = {
                    "symbol": symbol.upper(),
                    "earnings_dates": ed_records,
                    "income_stmt": inc_records,
                    "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                with open(raw_file, "w") as f:
                    json.dump(data, f)
            except Exception as e:
                print(f"[YahooFundamentalsProvider] Error fetching {symbol}: {e}")
                return []

        # Convert to standardized list
        records = []
        earnings_dates = data.get("earnings_dates", [])
        
        # Sort earnings dates chronologically
        sorted_ed = sorted(earnings_dates, key=lambda x: x["release_date"])
        
        for idx, ed in enumerate(sorted_ed):
            rep_eps = ed.get("reported_eps")
            if rep_eps is not None:
                records.append({
                    "code": symbol.upper(),
                    "source_symbol": symbol.upper(),
                    "fiscal_year": None,
                    "fiscal_quarter": None,
                    "report_period": ed.get("release_date"),
                    "period_start": None,
                    "period_end": ed.get("release_date"),
                    "eps_diluted": float(rep_eps),
                    "filing_date": ed.get("release_date"),
                    "accepted_at": ed.get("release_datetime"),
                    "earnings_release_at": ed.get("release_datetime") or ed.get("release_date"),
                    "source": "Yahoo",
                    "source_record_id": f"yahoo_{ed.get('release_date')}",
                })

        return records

    def fetch_earnings_events(self, symbol: str) -> List[Dict[str, Any]]:
        history = self.fetch_quarterly_history(symbol)
        events = []
        for h in history:
            events.append({
                "code": symbol.upper(),
                "report_period": h["report_period"],
                "eps_diluted": h["eps_diluted"],
                "event_date": h["earnings_release_at"] or h["filing_date"],
                "event_type": "EARNINGS_RELEASE",
                "source": "Yahoo",
            })
        return events
