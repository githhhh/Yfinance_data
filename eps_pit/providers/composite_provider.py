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
    latest_eps_pair_evidence,
    safe_float,
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
        sec_cik_hint: str | None = None,
    ) -> tuple[dict[str, Any] | None, EPSMissingReason | None]:
        """Resolve with mode-specific priority and zero-base reconciliation.

        Historical/replay:
        - SEC is authoritative primary.
        - A normal SEC numeric result returns immediately; Yahoo is not queried.
        - SEC PRIOR_YEAR_EPS_ZERO is a special semantic that triggers Yahoo
          Historical Event confirmation for the same current/prior periods.

        LIVE:
        - Yahoo current observation is primary.
        - SEC remains fallback.
        """
        self.yahoo.missing_release_periods = []
        yahoo_reason = EPSMissingReason.NO_QUARTERLY_EPS
        sec_reason = EPSMissingReason.NO_QUARTERLY_EPS
        yahoo_error: Exception | None = None
        sec_error: Exception | None = None
        yahoo_evidence: dict[str, Any] | None = None
        sec_evidence: dict[str, Any] | None = None

        def fetch_yahoo() -> dict[str, Any] | None:
            nonlocal yahoo_reason, yahoo_error, yahoo_evidence
            try:
                yahoo_history = self.yahoo.fetch_quarterly_history(
                    symbol,
                    require_release_date=not allow_current_yahoo,
                    observed_on=observation_date if allow_current_yahoo else None,
                    refresh=allow_current_yahoo,
                )
                yahoo_evidence = latest_eps_pair_evidence(
                    yahoo_history,
                    snapshot_date,
                )
                result, yahoo_reason = calculate_latest_eps_yoy_diagnostic(
                    yahoo_history,
                    snapshot_date,
                )
                return result
            except Exception as exc:
                yahoo_error = exc
                return None

        def fetch_sec() -> dict[str, Any] | None:
            nonlocal sec_reason, sec_error, sec_evidence
            try:
                sec_kwargs: dict[str, Any] = {
                    "prefer_bulk": not allow_current_yahoo,
                }
                if sec_cik_hint:
                    sec_kwargs["cik_hint"] = sec_cik_hint
                sec_history = self.sec.fetch_quarterly_history(
                    symbol,
                    **sec_kwargs,
                )
                sec_evidence = latest_eps_pair_evidence(
                    sec_history,
                    snapshot_date,
                )
                result, sec_reason = calculate_latest_eps_yoy_diagnostic(
                    sec_history,
                    snapshot_date,
                )
                return result
            except Exception as exc:
                sec_error = exc
                return None

        def evidence_fields(
            prefix: str,
            evidence: dict[str, Any] | None,
        ) -> dict[str, Any]:
            if not evidence:
                return {}
            return {
                f"{prefix}_current_eps": evidence.get("current_eps"),
                f"{prefix}_prior_year_eps": evidence.get("prior_year_eps"),
                f"{prefix}_current_period": evidence.get("current_period"),
                f"{prefix}_prior_year_period": evidence.get("prior_year_period"),
                f"{prefix}_effective_date": evidence.get("effective_date"),
                f"{prefix}_source_record_id": evidence.get("source_record_id"),
            }

        def zero_base_record(
            *,
            include_yahoo: bool,
        ) -> dict[str, Any] | None:
            if not sec_evidence:
                return None
            record = {
                "source": "SEC",
                "effective_date": sec_evidence.get("effective_date"),
                "current_eps": sec_evidence.get("current_eps"),
                "prior_year_eps": sec_evidence.get("prior_year_eps"),
                "current_period": sec_evidence.get("current_period"),
                "prior_year_period": sec_evidence.get("prior_year_period"),
                "growth_type": "ZERO_BASE",
                "calculation_method": "reported_zero_base",
                "sec_cik": sec_evidence.get("sec_cik"),
                "source_record_id": sec_evidence.get("source_record_id"),
                **evidence_fields("sec", sec_evidence),
            }
            if include_yahoo:
                record.update(evidence_fields("yahoo", yahoo_evidence))
            return record

        if allow_current_yahoo:
            yahoo_result = fetch_yahoo()
            if yahoo_result is not None:
                return yahoo_result, None
            if (
                yahoo_error is None
                and yahoo_reason is EPSMissingReason.PRIOR_YEAR_EPS_ZERO
            ):
                return None, yahoo_reason

            sec_result = fetch_sec()
            if sec_result is not None:
                return sec_result, None
            if sec_error is None and sec_reason is EPSMissingReason.PRIOR_YEAR_EPS_ZERO:
                return zero_base_record(include_yahoo=False), sec_reason
        else:
            sec_result = fetch_sec()
            if sec_result is not None:
                # Normal replay result: SEC alone is sufficient. Do not query
                # Yahoo merely for corroboration.
                return sec_result, None

            if sec_error is None and sec_reason is EPSMissingReason.PRIOR_YEAR_EPS_ZERO:
                # Special replay semantic: the SEC current/prior pair exists,
                # but the prior denominator is reported as zero. Query Yahoo's
                # release-dated Historical Event series only to determine
                # whether the same comparable periods contain a non-zero prior.
                yahoo_result = fetch_yahoo()

                if yahoo_result is not None and sec_evidence and yahoo_evidence:
                    same_periods = (
                        date10(sec_evidence.get("current_period"))
                        == date10(yahoo_evidence.get("current_period"))
                        and date10(sec_evidence.get("prior_year_period"))
                        == date10(yahoo_evidence.get("prior_year_period"))
                    )
                    sec_current = safe_float(sec_evidence.get("current_eps"))
                    yahoo_current = safe_float(yahoo_evidence.get("current_eps"))
                    current_consistent = (
                        sec_current is not None
                        and yahoo_current is not None
                        and abs(sec_current - yahoo_current)
                        <= max(0.01, abs(sec_current) * 0.10)
                    )
                    yahoo_prior = safe_float(yahoo_evidence.get("prior_year_eps"))

                    if same_periods and current_consistent and yahoo_prior not in {None, 0.0}:
                        reconciled = dict(yahoo_result)
                        reconciled.update(
                            {
                                "source": "SEC+YahooHistoricalEvent",
                                "calculation_method": "sec_zero_base_reconciled_yahoo_event",
                                "sec_cik": sec_evidence.get("sec_cik"),
                                **evidence_fields("sec", sec_evidence),
                                **evidence_fields("yahoo", yahoo_evidence),
                            }
                        )
                        return reconciled, None

                # Yahoo confirmation is absent, also zero-based, period-mismatched,
                # or materially inconsistent on the current quarter. SEC remains
                # the canonical historical fact and the semantic stays zero-base.
                return zero_base_record(include_yahoo=yahoo_evidence is not None), (
                    EPSMissingReason.PRIOR_YEAR_EPS_ZERO
                )

            yahoo_result = fetch_yahoo()
            if yahoo_result is not None:
                return yahoo_result, None
            if (
                yahoo_error is None
                and yahoo_reason is EPSMissingReason.PRIOR_YEAR_EPS_ZERO
            ):
                return None, yahoo_reason

        provider_errors = [
            f"{name}: {exc}"
            for name, exc in (("Yahoo", yahoo_error), ("SEC", sec_error))
            if exc is not None
        ]
        if yahoo_error is not None and sec_error is not None:
            raise RuntimeError("; ".join(provider_errors))

        primary_name = "Yahoo" if allow_current_yahoo else "SEC"
        primary_reason = yahoo_reason if allow_current_yahoo else sec_reason
        primary_error = yahoo_error if allow_current_yahoo else sec_error
        secondary_name = "SEC" if allow_current_yahoo else "Yahoo"
        secondary_error = sec_error if allow_current_yahoo else yahoo_error

        if primary_error is None and secondary_error is not None:
            logging.warning(
                "Signal EPS %s fallback failed for %s after clean %s outcome %s; "
                "preserving primary semantic result",
                secondary_name,
                symbol,
                primary_name,
                primary_reason.value,
            )
            return None, primary_reason

        if primary_error is not None:
            logging.warning(
                "Signal EPS PIT primary provider error for %s with no resolved fallback: %s",
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
        sec_cik_hint: str | None = None,
    ) -> dict[str, Any] | None:
        result, _ = self.fetch_eps_yoy_detailed(
            symbol,
            snapshot_date,
            allow_current_yahoo=allow_current_yahoo,
            observation_date=observation_date,
            sec_cik_hint=sec_cik_hint,
        )
        return result
