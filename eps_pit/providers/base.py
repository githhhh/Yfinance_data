from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import os


class BaseFundamentalsProvider(ABC):
    """Abstract Base Class for financial fundamentals and PIT earnings providers."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join("outputs", "eps_pit_backfill", "cache", "raw")
        os.makedirs(self.cache_dir, exist_ok=True)

    @abstractmethod
    def fetch_quarterly_history(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch quarterly historical EPS and filings for a given symbol.
        
        Returns a list of dicts with standardized keys:
            - code: project standard ticker
            - source_symbol: provider's ticker symbol
            - fiscal_year: int / None
            - fiscal_quarter: str ('Q1', 'Q2', 'Q3', 'Q4')
            - report_period: str (YYYY-MM-DD)
            - period_start: Optional[str] (YYYY-MM-DD)
            - period_end: str (YYYY-MM-DD)
            - eps_diluted: float
            - filing_date: Optional[str] (YYYY-MM-DD)
            - accepted_at: Optional[str] (ISO datetime)
            - earnings_release_at: Optional[str] (ISO datetime or YYYY-MM-DD)
            - source: str (provider name)
            - source_record_id: Optional[str]
        """
        pass

    @abstractmethod
    def fetch_earnings_events(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch announced earnings release events with dates and reported EPS."""
        pass
