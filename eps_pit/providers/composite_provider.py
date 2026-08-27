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
        *,
        allow_current_yahoo: bool = False,
        observation_date: object | None = None,
    ) -> tuple[dict[str, Any] | None, EPSMissingReason | None]:
        """Resolve Yahoo first, using SEC only as a fallback source."""
        self.yahoo.missing_release_periods = []
        yahoo_reason = EPSMissingReason.NO_QUARTERLY_EPS
        sec_reason = EPSMissingReason.NO_QUARTERLY_EPS
        yahoo_error: Exception | None = None
        sec_error: Exception | None = None

        try:
            yahoo_history = self.yahoo.fetch_quarterly_history(
                symbol,
                require_release_date=not allow_current_yahoo,
                observed_on=observation_date if allow_current_yahoo else None,
                refresh=allow_current_yahoo,
            )
            yahoo_result, yahoo_reason = calculate_latest_eps_yoy_diagnostic(
                yahoo_history, snapshot_date
            )
            if yahoo_result is not None:
                return yahoo_result, None
        except Exception as exc:
            yahoo_error = exc

        # Yahoo is the primary source. SEC is queried only when Yahoo could not
        # produce a numeric result, keeping SEC traffic low and preserving an
        # authoritative fallback for sparse or incomplete Yahoo fundamentals.
        try:
            sec_result, sec_reason = calculate_latest_eps_yoy_diagnostic(
                self.sec.fetch_quarterly_history(symbol), snapshot_date
            )
            if sec_result is not None:
                return sec_result, None
        except Exception as exc:
            sec_error = exc

        # A zero prior-year denominator is a semantic outcome rather than a
        # transport/coverage failure. If either provider reached it cleanly,
        # publishing EXPECTED_UNAVAILABLE is safe even if the other provider is
        # unavailable.
        if yahoo_error is None and yahoo_reason is EPSMissingReason.PRIOR_YEAR_EPS_ZERO:
            return None, yahoo_reason
        if sec_error is None and sec_reason is EPSMissingReason.PRIOR_YEAR_EPS_ZERO:
            return None, sec_reason

        provider_errors = [
            f"{name}: {exc}"
            for name, exc in (("Yahoo", yahoo_error), ("SEC", sec_error))
            if exc is not None
        ]
        if yahoo_error is not None and sec_error is not None:
            raise RuntimeError("; ".join(provider_errors))

        # A technical failure matters whenever no source resolved a value or
        # reached a terminal semantic outcome. Do not silently downgrade an
        # incomplete run into EXPECTED_UNAVAILABLE.
        if provider_errors:
            logging.warning(
                "Signal EPS PIT provider error for %s with no resolved fallback: %s",
                symbol,
                "; ".join(provider_errors),
            )
            return None, EPSMissingReason.PROVIDER_ERROR

        reasons = {yahoo_reason, sec_reason}
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
        *,
        allow_current_yahoo: bool = False,
        observation_date: object | None = None,
    ) -> dict[str, Any] | None:
        result, _ = self.fetch_eps_yoy_detailed(
            symbol,
            snapshot_date,
            allow_current_yahoo=allow_current_yahoo,
            observation_date=observation_date,
        )
        return result
