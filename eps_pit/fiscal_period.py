import datetime
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


class FiscalPeriodMatcher:
    """Matches fiscal quarters for YoY comparisons (e.g. FY2026 Q2 vs FY2025 Q2)."""

    @staticmethod
    def match_quarters(records: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]], str]]:
        """Given a chronologically sorted list of quarterly records for a single ticker,
        matches each record with its prior year same-quarter counterpart.
        
        Returns a list of tuples: (current_record, prior_year_record, match_status)
        where match_status is one of:
            - 'EXACT_FISCAL_MATCH': Same fiscal quarter (e.g. FY2026 Q2 vs FY2025 Q2)
            - 'DATE_CYCLE_MATCH': Inferred from ~365-day periodic cycle
            - 'NO_PRIOR_YEAR_QUARTER': No prior year data found (e.g. recent IPO)
            - 'AMBIGUOUS': Multiple conflicting matches or abnormal interval
        """
        if not records:
            return []

        # Sort chronologically by report_period / filing_date
        sorted_records = sorted(
            records,
            key=lambda r: (r.get("report_period") or "", r.get("filing_date") or "")
        )

        results = []

        # Build lookup tables
        # 1. By (fiscal_year, fiscal_quarter)
        fy_fp_map = {}
        # 2. By report_period date
        date_map = {}

        for r in sorted_records:
            fy = r.get("fiscal_year")
            fp = r.get("fiscal_quarter")
            rp = r.get("report_period")
            if fy and fp:
                fy_fp_map[(fy, fp)] = r
            if rp:
                date_map[rp] = r

        for curr in sorted_records:
            curr_fy = curr.get("fiscal_year")
            curr_fp = curr.get("fiscal_quarter")
            curr_rp = curr.get("report_period")

            prior = None
            status = "NO_PRIOR_YEAR_QUARTER"

            # Strategy 1: Exact Fiscal Year & Quarter Match
            if curr_fy and curr_fp:
                prior_key = (curr_fy - 1, curr_fp)
                if prior_key in fy_fp_map:
                    prior = fy_fp_map[prior_key]
                    status = "EXACT_FISCAL_MATCH"

            # Strategy 2: Date Cycle Matching (approx 340-390 days prior)
            if not prior and curr_rp:
                try:
                    curr_dt = datetime.date.fromisoformat(curr_rp)
                    best_cand = None
                    best_diff = 999
                    for prev_r in sorted_records:
                        prev_rp = prev_r.get("report_period")
                        if not prev_rp or prev_rp >= curr_rp:
                            continue
                        prev_dt = datetime.date.fromisoformat(prev_rp)
                        days_diff = (curr_dt - prev_dt).days
                        # Look for candidate around 365 days (340 to 395 days)
                        if 335 <= days_diff <= 395:
                            diff_from_year = abs(days_diff - 365)
                            if diff_from_year < best_diff:
                                best_diff = diff_from_year
                                best_cand = prev_r

                    if best_cand:
                        prior = best_cand
                        status = "DATE_CYCLE_MATCH"
                except Exception:
                    pass

            # Strategy 3: Index offset (4 quarters back) as fallback if exactly 4 quarters back is ~365 days
            if not prior:
                idx = sorted_records.index(curr)
                if idx >= 4:
                    cand = sorted_records[idx - 4]
                    c_dt_str = cand.get("report_period")
                    if curr_rp and c_dt_str:
                        try:
                            d1 = datetime.date.fromisoformat(curr_rp)
                            d2 = datetime.date.fromisoformat(c_dt_str)
                            if 300 <= (d1 - d2).days <= 430:
                                prior = cand
                                status = "DATE_CYCLE_MATCH"
                        except Exception:
                            pass

            results.append((curr, prior, status))

        return results
