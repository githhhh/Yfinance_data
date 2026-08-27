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

        # PRIOR_YEAR_EPS_ZERO from SEC is a mathematical result from an
        # authoritative filing, not a source-coverage miss. Do not let a
        # secondary current-state provider manufacture a denominator that the
        # filed diluted EPS does not contain.
        if sec_error is None and sec_reason is EPSMissingReason.PRIOR_YEAR_EPS_ZERO:
            return None, sec_reason

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

        provider_errors = [
            f"{name}: {exc}"
            for name, exc in (("SEC", sec_error), ("Yahoo", yahoo_error))
            if exc is not None
        ]
        if sec_error is not None and yahoo_error is not None:
            raise RuntimeError("; ".join(provider_errors))

        # A technical provider failure matters whenever neither source resolved.
        # The other provider returning a business-level miss does not prove
        # completeness, so do not downgrade a transport/configuration failure
        # into EXPECTED_UNAVAILABLE.
        if provider_errors:
            logging.warning(
                "Signal EPS PIT provider error for %s with no resolved fallback: %s",
                symbol,
                "; ".join(provider_errors),
            )
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
