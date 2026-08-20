import os
import json
import time
import datetime
from typing import Dict, Any, List, Optional
import requests
from eps_pit.providers.base import BaseFundamentalsProvider


class SECProvider(BaseFundamentalsProvider):
    """SEC EDGAR XBRL Company Facts Provider.
    
    Provides true As-Filed Point-in-Time historical quarterly 10-Q/10-K reports
    directly from the US Securities and Exchange Commission.
    """

    SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SEC_FACTS_URL_TMPL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    USER_AGENT = "QuantResearchBot/1.0 (quant_research@example.com)"

    def __init__(self, cache_dir: Optional[str] = None, rate_limit_sleep: float = 0.1):
        super().__init__(cache_dir=cache_dir)
        self.raw_sec_dir = os.path.join(self.cache_dir, "SEC")
        os.makedirs(self.raw_sec_dir, exist_ok=True)
        self.rate_limit_sleep = rate_limit_sleep
        self._cik_map: Optional[Dict[str, str]] = None
        self.headers = {"User-Agent": self.USER_AGENT}

    def _get_cik_map(self) -> Dict[str, str]:
        if self._cik_map is not None:
            return self._cik_map
        
        map_cache_file = os.path.join(self.cache_dir, "sec_company_tickers.json")
        if os.path.exists(map_cache_file):
            try:
                with open(map_cache_file, "r") as f:
                    data = json.load(f)
                    self._cik_map = {
                        v["ticker"].upper().replace("-", "."): str(v["cik_str"]).zfill(10)
                        for v in data.values()
                    }
                    # Also map with dash
                    for v in data.values():
                        self._cik_map[v["ticker"].upper()] = str(v["cik_str"]).zfill(10)
                    return self._cik_map
            except Exception:
                pass

        # Fetch from SEC
        try:
            r = requests.get(self.SEC_TICKERS_URL, headers=self.headers, timeout=15)
            if r.status_code == 200:
                data = r.json()
                with open(map_cache_file, "w") as f:
                    json.dump(data, f)
                self._cik_map = {}
                for v in data.values():
                    t = v["ticker"].upper()
                    cik = str(v["cik_str"]).zfill(10)
                    self._cik_map[t] = cik
                    self._cik_map[t.replace("-", ".")] = cik
                    self._cik_map[t.replace(".", "-")] = cik
                return self._cik_map
        except Exception as e:
            print(f"[SECProvider] Failed to fetch SEC tickers: {e}")
        
        self._cik_map = {}
        return self._cik_map

    def get_cik(self, symbol: str) -> Optional[str]:
        m = self._get_cik_map()
        sym = symbol.upper().strip()
        if sym in m:
            return m[sym]
        sym_dot = sym.replace("-", ".")
        if sym_dot in m:
            return m[sym_dot]
        sym_dash = sym.replace(".", "-")
        if sym_dash in m:
            return m[sym_dash]
        return None

    def fetch_quarterly_history(self, symbol: str) -> List[Dict[str, Any]]:
        raw_file = os.path.join(self.raw_sec_dir, f"{symbol.upper()}.json")
        facts = None
        
        # Load from raw cache if exists
        if os.path.exists(raw_file):
            try:
                with open(raw_file, "r") as f:
                    facts = json.load(f)
            except Exception:
                facts = None

        if facts is None:
            cik = self.get_cik(symbol)
            if not cik:
                return []
            
            url = self.SEC_FACTS_URL_TMPL.format(cik=cik)
            try:
                time.sleep(self.rate_limit_sleep)
                r = requests.get(url, headers=self.headers, timeout=15)
                if r.status_code == 200:
                    facts = r.json()
                    with open(raw_file, "w") as f:
                        json.dump(facts, f)
                else:
                    return []
            except Exception as e:
                print(f"[SECProvider] Error fetching facts for {symbol}: {e}")
                return []

        # Parse facts
        gaap = facts.get("facts", {}).get("us-gaap", {})
        # Prefer EarningsPerShareDiluted, then EarningsPerShareBasic
        eps_field = None
        for k in ["EarningsPerShareDiluted", "EarningsPerShareBasic"]:
            if k in gaap and "units" in gaap[k] and "USD/shares" in gaap[k]["units"]:
                eps_field = k
                break
        
        if not eps_field:
            return []

        entries = gaap[eps_field]["units"]["USD/shares"]
        
        # Extract quarterly rows
        # We need entries from 10-Q (or 10-K for Q4 if duration ~ 3 months or frame is Q4)
        records = []
        for e in entries:
            form = e.get("form", "")
            fp = e.get("fp", "")
            fy = e.get("fy")
            val = e.get("val")
            filed = e.get("filed")
            end = e.get("end")
            start = e.get("start")
            accn = e.get("accn")
            
            if val is None or not filed or not end:
                continue

            # Calculate duration in days if start is provided
            duration = None
            if start:
                try:
                    d_start = datetime.date.fromisoformat(start)
                    d_end = datetime.date.fromisoformat(end)
                    duration = (d_end - d_start).days
                except Exception:
                    pass

            # Filter for 3-month quarterly records (typical duration 60-120 days)
            # Or 10-Q filing where fp is Q1, Q2, Q3
            is_quarterly = False
            fiscal_quarter = None

            if form in ["10-Q", "10-Q/A"]:
                if duration is not None:
                    if 60 <= duration <= 125:
                        is_quarterly = True
                else:
                    is_quarterly = True
                if fp in ["Q1", "Q2", "Q3", "Q4"]:
                    fiscal_quarter = fp
            elif form in ["10-K", "10-K/A"]:
                # For 10-K, check if duration is ~3 months (Q4)
                if duration is not None and 60 <= duration <= 125:
                    is_quarterly = True
                    fiscal_quarter = "Q4" if fp == "FY" else fp
                elif fp == "Q4":
                    is_quarterly = True
                    fiscal_quarter = "Q4"

            if is_quarterly and fiscal_quarter:
                records.append({
                    "code": symbol.upper(),
                    "source_symbol": symbol.upper(),
                    "fiscal_year": int(fy) if fy is not None else None,
                    "fiscal_quarter": fiscal_quarter,
                    "report_period": end,
                    "period_start": start,
                    "period_end": end,
                    "eps_diluted": float(val),
                    "filing_date": filed,
                    "accepted_at": filed,
                    "earnings_release_at": None,
                    "source": "SEC",
                    "source_record_id": f"{accn}_{end}_{fiscal_quarter}" if accn else f"{filed}_{end}",
                })

        # Deduplicate by (report_period, fiscal_quarter) — NOT including fiscal_year.
        # SEC XBRL comparative-period entries restate prior-year quarterly EPS
        # under the current filing's fiscal_year. For the same (report_period, quarter),
        # keep the record with the smallest fiscal_year (the original filing) to
        # eliminate phantom comparative-period duplicates. On fiscal_year ties,
        # keep the latest filing_date (handles 10-Q/A amendments).
        dedup_dict = {}
        for r in records:
            k = (r["report_period"], r["fiscal_quarter"])
            if k not in dedup_dict:
                dedup_dict[k] = r
            else:
                existing = dedup_dict[k]
                if r["fiscal_year"] < existing["fiscal_year"]:
                    dedup_dict[k] = r
                elif r["fiscal_year"] == existing["fiscal_year"] and r["filing_date"] >= existing["filing_date"]:
                    dedup_dict[k] = r

        return sorted(list(dedup_dict.values()), key=lambda x: (x["report_period"], x["filing_date"]))

    def fetch_earnings_events(self, symbol: str) -> List[Dict[str, Any]]:
        # SEC provides filing dates, which serve as conservative PIT dates
        history = self.fetch_quarterly_history(symbol)
        events = []
        for h in history:
            events.append({
                "code": symbol.upper(),
                "report_period": h["report_period"],
                "fiscal_year": h["fiscal_year"],
                "fiscal_quarter": h["fiscal_quarter"],
                "eps_diluted": h["eps_diluted"],
                "event_date": h["filing_date"],
                "event_type": "SEC_FILING",
                "source": "SEC",
            })
        return events
