"""Point-in-Time EPS YoY Growth Backfill & Audit Module.

Provides robust, audit-compliant, point-in-time Diluted EPS YoY calculation,
caching, validation against TradingView reference data, backward-asof merging,
and dynamic signal EPS lookup for weekly pool snapshots.
"""

from eps_pit.growth import EPSGrowthCalculator
from eps_pit.fiscal_period import FiscalPeriodMatcher
from eps_pit.pit import PITTimelineEngine
from eps_pit.mapping import TickerMapper
from eps_pit.lookup import SignalEPSLookup, get_signal_eps, enrich_pool_with_signal_eps

__all__ = [
    "EPSGrowthCalculator",
    "FiscalPeriodMatcher",
    "PITTimelineEngine",
    "TickerMapper",
    "SignalEPSLookup",
    "get_signal_eps",
    "enrich_pool_with_signal_eps",
]

__version__ = "1.0.0"
