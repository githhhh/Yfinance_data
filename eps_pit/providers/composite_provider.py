from __future__ import annotations

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
        provider_errors: list[str] = []
        self.yahoo.missing_release_periods = []
        sec_reason = EPSMissingReason.NO_QUARTERLY_EPS
        yahoo_reason = EPSMissingReason.NO_QUARTERLY_EPS

        try:
            sec_result, sec_reason = calculate_latest_eps_yoy_diagnostic(
                self.sec.fetch_quarterly_history(symbol), snapshot_date
            )
            if sec_result is not None:
                return sec_result, None
        except Exception as exc:
            provider_errors.append(f"SEC: {exc}")

        try:
            yahoo_result, yahoo_reason = calculate_latest_eps_yoy_diagnostic(
                self.yahoo.fetch_quarterly_history(symbol), snapshot_date
            )
            if yahoo_result is not None:
                return yahoo_result, None
        except Exception as exc:
            provider_errors.append(f"Yahoo: {exc}")

        # If neither provider resolved, any technical failure means we did not
        # establish completeness. Never downgrade it to a business-level miss.
        if len(provider_errors) == 2:
            raise RuntimeError("; ".join(provider_errors))
        if provider_errors:
            return None, EPSMissingReason.PROVIDER_ERROR

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
