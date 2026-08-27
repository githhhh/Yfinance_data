from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from eps_pit.models import EPSMissingReason
from eps_pit.providers.pit_provider import (
    SECProvider,
    YahooFundamentalsProvider,
    calculate_latest_eps_yoy_diagnostic,
    date10,
)


class SECYahooEPSProvider:
    """Strict PIT fallback orchestration over SEC and Yahoo quarterly facts."""

    def __init__(self, cache_dir: Path | None = None):
        self.sec = SECProvider(cache_dir)
        self.yahoo = YahooFundamentalsProvider(cache_dir)

    def fetch_eps_yoy_detailed(
        self,
        symbol: str,
        snapshot_date: object,
    ) -> tuple[dict[str, Any] | None, EPSMissingReason | None]:
        self.yahoo.missing_release_periods = []
        sec_reason = EPSMissingReason.NO_QUARTERLY_EPS
        yahoo_reason = EPSMissingReason.NO_QUARTERLY_EPS
        sec_error: Exception | None = None
        yahoo_error: Exception | None = None

        try:
            sec_result, sec_reason = calculate_latest_eps_yoy_diagnostic(
                self.sec.fetch_quarterly_history(symbol), snapshot_date
            )
            if sec_result is not None:
                return sec_result, None
        except Exception as exc:
            sec_error = exc

        try:
            yahoo_result, yahoo_reason = calculate_latest_eps_yoy_diagnostic(
                self.yahoo.fetch_quarterly_history(symbol), snapshot_date
            )
            if yahoo_result is not None:
                return yahoo_result, None
        except Exception as exc:
            yahoo_error = exc

        provider_errors = [
            f"{name}: {exc}"
            for name, exc in (("SEC", sec_error), ("Yahoo", yahoo_error))
            if exc is not None
        ]
        if sec_error is not None and yahoo_error is not None:
            raise RuntimeError("; ".join(provider_errors))
        if sec_error is not None:
            logging.warning(
                "Signal EPS PIT SEC provider error for %s ignored after Yahoo returned %s: %s",
                symbol,
                yahoo_reason.value,
                sec_error,
            )
            return None, yahoo_reason
        if yahoo_error is not None:
            logging.warning(
                "Signal EPS PIT Yahoo provider error for %s ignored after SEC returned %s: %s",
                symbol,
                sec_reason.value,
                yahoo_error,
            )
            return None, sec_reason

        reasons = {sec_reason, yahoo_reason}
        if EPSMissingReason.PRIOR_YEAR_EPS_ZERO in reasons:
            return None, EPSMissingReason.PRIOR_YEAR_EPS_ZERO
        if EPSMissingReason.NO_PRIOR_YEAR_QUARTER in reasons:
            return None, EPSMissingReason.NO_PRIOR_YEAR_QUARTER

        snapshot = date10(snapshot_date)
        relevant_unverified = [
            period
            for period in self.yahoo.missing_release_periods
            if date10(period) and snapshot and date10(period) <= snapshot
        ]
        if relevant_unverified:
            return None, EPSMissingReason.NO_VERIFIED_YAHOO_RELEASE_DATE
        return None, EPSMissingReason.NO_QUARTERLY_EPS

    def fetch_eps_yoy(
        self,
        symbol: str,
        snapshot_date: object,
    ) -> dict[str, Any] | None:
        result, _ = self.fetch_eps_yoy_detailed(symbol, snapshot_date)
        return result
