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
        """Resolve with a mode-specific source order.

        Historical/replay snapshots prefer SEC's filed companyfacts (bulk/cache
        first) and use Yahoo's release-dated reconstruction only as fallback.
        True LIVE snapshots prefer Yahoo's current observation and use SEC as
        fallback.
        """
        self.yahoo.missing_release_periods = []
        yahoo_reason = EPSMissingReason.NO_QUARTERLY_EPS
        sec_reason = EPSMissingReason.NO_QUARTERLY_EPS
        yahoo_error: Exception | None = None
        sec_error: Exception | None = None

        def fetch_yahoo() -> dict[str, Any] | None:
            nonlocal yahoo_reason, yahoo_error
            try:
                yahoo_history = self.yahoo.fetch_quarterly_history(
                    symbol,
                    require_release_date=not allow_current_yahoo,
                    observed_on=observation_date if allow_current_yahoo else None,
                    refresh=allow_current_yahoo,
                )
                result, yahoo_reason = calculate_latest_eps_yoy_diagnostic(
                    yahoo_history, snapshot_date
                )
                return result
            except Exception as exc:
                yahoo_error = exc
                return None

        def fetch_sec() -> dict[str, Any] | None:
            nonlocal sec_reason, sec_error
            try:
                sec_history = self.sec.fetch_quarterly_history(
                    symbol,
                    prefer_bulk=not allow_current_yahoo,
                )
                result, sec_reason = calculate_latest_eps_yoy_diagnostic(
                    sec_history, snapshot_date
                )
                return result
            except Exception as exc:
                sec_error = exc
                return None

        if allow_current_yahoo:
            yahoo_result = fetch_yahoo()
            if yahoo_result is not None:
                return yahoo_result, None
            if yahoo_error is None and yahoo_reason is EPSMissingReason.PRIOR_YEAR_EPS_ZERO:
                return None, yahoo_reason

            sec_result = fetch_sec()
            if sec_result is not None:
                return sec_result, None
            if sec_error is None and sec_reason is EPSMissingReason.PRIOR_YEAR_EPS_ZERO:
                return None, sec_reason
        else:
            sec_result = fetch_sec()
            if sec_result is not None:
                return sec_result, None
            if sec_error is None and sec_reason is EPSMissingReason.PRIOR_YEAR_EPS_ZERO:
                return None, sec_reason

            yahoo_result = fetch_yahoo()
            if yahoo_result is not None:
                return yahoo_result, None
            if yahoo_error is None and yahoo_reason is EPSMissingReason.PRIOR_YEAR_EPS_ZERO:
                return None, yahoo_reason

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
