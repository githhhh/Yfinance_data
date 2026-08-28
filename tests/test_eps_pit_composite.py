from eps_pit import EPSMissingReason
from eps_pit.providers.composite_provider import SECYahooEPSProvider


def _resolved_yahoo_record():
    return {
        "report_period": "2026-06-30",
        "eps_diluted": 1.5,
        "earnings_release_at": "2026-08-01",
        "period_type": "quarter",
        "source": "Yahoo",
        "concept": "DilutedEPS",
        "unit": "USD/shares",
    }


def _prior_yahoo_record():
    return {
        "report_period": "2025-06-30",
        "eps_diluted": 1.0,
        "earnings_release_at": "2025-08-01",
        "period_type": "quarter",
        "source": "Yahoo",
        "concept": "DilutedEPS",
        "unit": "USD/shares",
    }


def test_single_provider_failure_plus_nonterminal_missing_is_provider_error(monkeypatch, caplog):
    provider = SECYahooEPSProvider()
    monkeypatch.setattr(
        provider.sec,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: (_ for _ in ()).throw(RuntimeError("SEC provider HTTP 500")),
    )
    monkeypatch.setattr(provider.yahoo, "fetch_quarterly_history", lambda symbol, **kwargs: [])

    with caplog.at_level("WARNING"):
        result, reason = provider.fetch_eps_yoy_detailed("TEST", "2026-08-21")

    assert result is None
    assert reason is EPSMissingReason.PROVIDER_ERROR
    assert "Signal EPS PIT primary provider error for TEST with no resolved fallback" in caplog.text


def test_single_provider_failure_plus_yahoo_prior_missing_is_provider_error(monkeypatch):
    provider = SECYahooEPSProvider()
    monkeypatch.setattr(
        provider.sec,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: (_ for _ in ()).throw(RuntimeError("SEC ticker map HTTP 403")),
    )
    monkeypatch.setattr(
        provider.yahoo,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: [_resolved_yahoo_record()],
    )

    result, reason = provider.fetch_eps_yoy_detailed("TEST", "2026-08-21")

    assert result is None
    assert reason is EPSMissingReason.PROVIDER_ERROR


def test_historical_clean_sec_missing_survives_yahoo_fallback_failure(monkeypatch):
    provider = SECYahooEPSProvider()
    monkeypatch.setattr(provider.sec, "fetch_quarterly_history", lambda symbol, **kwargs: [])
    monkeypatch.setattr(
        provider.yahoo,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: (_ for _ in ()).throw(RuntimeError("Yahoo service unavailable")),
    )

    result, reason = provider.fetch_eps_yoy_detailed("TEST", "2026-08-21")

    assert result is None
    assert reason is EPSMissingReason.NO_QUARTERLY_EPS


def test_other_provider_can_resolve_when_one_provider_fails(monkeypatch):
    provider = SECYahooEPSProvider()
    monkeypatch.setattr(
        provider.sec,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: (_ for _ in ()).throw(RuntimeError("SEC outage")),
    )
    monkeypatch.setattr(
        provider.yahoo,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: [_prior_yahoo_record(), _resolved_yahoo_record()],
    )

    result, reason = provider.fetch_eps_yoy_detailed("TEST", "2026-08-21")

    assert reason is None
    assert result["eps_yoy_growth"] == 50.0
    assert result["source"] == "Yahoo"


def test_dual_provider_failure_still_blocks_missing_result(monkeypatch):
    provider = SECYahooEPSProvider()
    monkeypatch.setattr(
        provider.sec,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: (_ for _ in ()).throw(RuntimeError("SEC outage")),
    )
    monkeypatch.setattr(
        provider.yahoo,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: (_ for _ in ()).throw(RuntimeError("Yahoo outage")),
    )

    try:
        provider.fetch_eps_yoy_detailed("TEST", "2026-08-21")
    except RuntimeError as exc:
        assert "SEC: SEC outage" in str(exc)
        assert "Yahoo: Yahoo outage" in str(exc)
    else:
        raise AssertionError("dual provider failure should raise")


def test_missing_yahoo_release_date_is_snapshot_scoped(monkeypatch):
    provider = SECYahooEPSProvider()
    monkeypatch.setattr(provider.sec, "fetch_quarterly_history", lambda symbol, **kwargs: [])

    def future_only(symbol, **kwargs):
        provider.yahoo.missing_release_periods = ["2027-06-30"]
        return []

    monkeypatch.setattr(provider.yahoo, "fetch_quarterly_history", future_only)
    result, reason = provider.fetch_eps_yoy_detailed("TEST", "2026-08-21")
    assert result is None
    assert reason is EPSMissingReason.NO_QUARTERLY_EPS


def test_relevant_missing_yahoo_release_date_is_explicit(monkeypatch):
    provider = SECYahooEPSProvider()
    monkeypatch.setattr(provider.sec, "fetch_quarterly_history", lambda symbol, **kwargs: [])

    def current_period(symbol, **kwargs):
        provider.yahoo.missing_release_periods = ["2026-06-30"]
        return []

    monkeypatch.setattr(provider.yahoo, "fetch_quarterly_history", current_period)
    result, reason = provider.fetch_eps_yoy_detailed("TEST", "2026-08-21")
    assert result is None
    assert reason is EPSMissingReason.NO_VERIFIED_YAHOO_RELEASE_DATE


def test_yahoo_zero_denominator_is_terminal_even_if_sec_failed(monkeypatch):
    provider = SECYahooEPSProvider()
    monkeypatch.setattr(
        provider.sec,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: (_ for _ in ()).throw(RuntimeError("SEC outage")),
    )
    zero_records = [
        {
            **_prior_yahoo_record(),
            "eps_diluted": 0.0,
        },
        _resolved_yahoo_record(),
    ]
    monkeypatch.setattr(
        provider.yahoo,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: zero_records,
    )

    result, reason = provider.fetch_eps_yoy_detailed("TEST", "2026-08-21")

    assert result is None
    assert reason is EPSMissingReason.PRIOR_YEAR_EPS_ZERO


def test_sec_zero_denominator_queries_yahoo_for_historical_confirmation(monkeypatch):
    provider = SECYahooEPSProvider()
    calls = []
    sec_records = [
        {
            "report_period": "2025-06-30",
            "start": "2025-04-01",
            "eps_diluted": 0.0,
            "filing_date": "2025-08-01",
            "period_type": "quarter",
            "source": "SEC",
            "concept": "EarningsPerShareDiluted",
            "unit": "USD/shares",
        },
        {
            "report_period": "2026-06-30",
            "start": "2026-04-01",
            "eps_diluted": 0.05,
            "filing_date": "2026-08-01",
            "period_type": "quarter",
            "source": "SEC",
            "concept": "EarningsPerShareDiluted",
            "unit": "USD/shares",
        },
    ]
    yahoo_records = [
        {
            "report_period": "2025-06-30",
            "eps_diluted": 0.0,
            "earnings_release_at": "2025-08-01",
            "period_type": "quarter",
            "source": "YahooHistoricalEvent",
            "concept": "DilutedEPS",
            "unit": "USD/shares",
        },
        {
            "report_period": "2026-06-30",
            "eps_diluted": 0.05,
            "earnings_release_at": "2026-08-01",
            "period_type": "quarter",
            "source": "YahooHistoricalEvent",
            "concept": "DilutedEPS",
            "unit": "USD/shares",
        },
    ]
    monkeypatch.setattr(
        provider.sec,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: sec_records,
    )

    def yahoo_fetch(symbol, **kwargs):
        calls.append(kwargs)
        return yahoo_records

    monkeypatch.setattr(provider.yahoo, "fetch_quarterly_history", yahoo_fetch)

    result, reason = provider.fetch_eps_yoy_detailed("TEST", "2026-08-21")

    assert reason is EPSMissingReason.PRIOR_YEAR_EPS_ZERO
    assert result["growth_type"] == "ZERO_BASE"
    assert result["sec_prior_year_eps"] == 0.0
    assert result["yahoo_prior_year_eps"] == 0.0
    assert len(calls) == 1


def test_historical_prefers_sec_and_never_calls_yahoo_when_sec_resolves(monkeypatch):
    provider = SECYahooEPSProvider()
    calls = []
    sec_records = [
        {
            "report_period": "2025-06-30",
            "start": "2025-04-01",
            "eps_diluted": 1.0,
            "filing_date": "2025-08-01",
            "period_type": "quarter",
            "source": "SEC",
            "concept": "EarningsPerShareDiluted",
            "unit": "USD/shares",
        },
        {
            "report_period": "2026-06-30",
            "start": "2026-04-01",
            "eps_diluted": 1.5,
            "filing_date": "2026-08-01",
            "period_type": "quarter",
            "source": "SEC",
            "concept": "EarningsPerShareDiluted",
            "unit": "USD/shares",
        },
    ]

    def sec_fetch(symbol, **kwargs):
        calls.append(("SEC", kwargs))
        return sec_records

    monkeypatch.setattr(provider.sec, "fetch_quarterly_history", sec_fetch)
    monkeypatch.setattr(
        provider.yahoo,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: (_ for _ in ()).throw(
            AssertionError("historical SEC success must not query Yahoo")
        ),
    )

    result, reason = provider.fetch_eps_yoy_detailed("TEST", "2026-08-21")

    assert reason is None
    assert result["source"] == "SEC"
    assert result["eps_yoy_growth"] == 50.0
    assert calls == [("SEC", {"prefer_bulk": True})]


def test_live_prefers_yahoo_and_never_calls_sec_when_yahoo_resolves(monkeypatch):
    provider = SECYahooEPSProvider()
    calls = []

    def yahoo_fetch(symbol, **kwargs):
        calls.append(("Yahoo", kwargs))
        return [_prior_yahoo_record(), _resolved_yahoo_record()]

    monkeypatch.setattr(provider.yahoo, "fetch_quarterly_history", yahoo_fetch)
    monkeypatch.setattr(
        provider.sec,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: (_ for _ in ()).throw(
            AssertionError("LIVE Yahoo success must not query SEC")
        ),
    )

    result, reason = provider.fetch_eps_yoy_detailed(
        "TEST",
        "2026-08-21",
        allow_current_yahoo=True,
        observation_date="2026-08-21",
    )

    assert reason is None
    assert result["source"] == "Yahoo"
    assert result["eps_yoy_growth"] == 50.0
    assert calls == [
        (
            "Yahoo",
            {
                "require_release_date": False,
                "observed_on": "2026-08-21",
                "refresh": True,
            },
        )
    ]


def test_live_clean_yahoo_missing_is_not_upgraded_by_sec_technical_failure(monkeypatch):
    provider = SECYahooEPSProvider()
    monkeypatch.setattr(
        provider.yahoo,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: [_resolved_yahoo_record()],
    )
    monkeypatch.setattr(
        provider.sec,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: (_ for _ in ()).throw(
            RuntimeError("SEC ticker map HTTP 403")
        ),
    )

    result, reason = provider.fetch_eps_yoy_detailed(
        "TEST",
        "2026-08-21",
        allow_current_yahoo=True,
        observation_date="2026-08-21",
    )

    assert result is None
    assert reason is EPSMissingReason.NO_PRIOR_YEAR_QUARTER


def test_live_yahoo_technical_failure_still_fails_closed_when_sec_cannot_resolve(monkeypatch):
    provider = SECYahooEPSProvider()
    monkeypatch.setattr(
        provider.yahoo,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: (_ for _ in ()).throw(
            RuntimeError("Yahoo service unavailable")
        ),
    )
    monkeypatch.setattr(
        provider.sec,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: [],
    )

    result, reason = provider.fetch_eps_yoy_detailed(
        "TEST",
        "2026-08-21",
        allow_current_yahoo=True,
        observation_date="2026-08-21",
    )

    assert result is None
    assert reason is EPSMissingReason.PROVIDER_ERROR


def test_historical_clean_sec_missing_is_not_upgraded_by_yahoo_failure(monkeypatch):
    provider = SECYahooEPSProvider()
    monkeypatch.setattr(
        provider.sec,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: [],
    )
    monkeypatch.setattr(
        provider.yahoo,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: (_ for _ in ()).throw(
            RuntimeError("Yahoo service unavailable")
        ),
    )

    result, reason = provider.fetch_eps_yoy_detailed("TEST", "2026-08-21")

    assert result is None
    assert reason is EPSMissingReason.NO_QUARTERLY_EPS


def test_replay_sec_zero_base_is_reconciled_by_same_period_yahoo_event(monkeypatch):
    provider = SECYahooEPSProvider()

    sec_records = [
        {
            "report_period": "2025-06-30",
            "start": "2025-04-01",
            "eps_diluted": 0.0,
            "filing_date": "2025-08-05",
            "period_type": "quarter",
            "source": "SEC",
            "concept": "EarningsPerShareDiluted",
            "unit": "USD/shares",
            "sec_cik": "0002100805",
            "source_record_id": "sec-prior",
        },
        {
            "report_period": "2026-06-30",
            "start": "2026-04-01",
            "eps_diluted": 0.05,
            "filing_date": "2026-08-05",
            "period_type": "quarter",
            "source": "SEC",
            "concept": "EarningsPerShareDiluted",
            "unit": "USD/shares",
            "sec_cik": "0002100805",
            "source_record_id": "sec-current",
        },
    ]
    yahoo_records = [
        {
            "report_period": "2025-06-30",
            "eps_diluted": -0.014205,
            "earnings_release_at": "2025-08-05",
            "period_type": "quarter",
            "source": "YahooHistoricalEvent",
            "concept": "DilutedEPS",
            "unit": "USD/shares",
            "source_record_id": "yahoo-prior",
        },
        {
            "report_period": "2026-06-30",
            "eps_diluted": 0.05,
            "earnings_release_at": "2026-08-05",
            "period_type": "quarter",
            "source": "YahooHistoricalEvent",
            "concept": "DilutedEPS",
            "unit": "USD/shares",
            "source_record_id": "yahoo-current",
        },
    ]

    monkeypatch.setattr(
        provider.sec,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: sec_records,
    )
    monkeypatch.setattr(
        provider.yahoo,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: yahoo_records,
    )

    result, reason = provider.fetch_eps_yoy_detailed("JAN", "2026-08-28")

    assert reason is None
    assert result["source"] == "SEC+YahooHistoricalEvent"
    assert result["current_period"] == "2026-06-30"
    assert result["prior_year_period"] == "2025-06-30"
    assert result["current_eps"] == 0.05
    assert result["prior_year_eps"] == -0.014205
    assert result["eps_yoy_growth"] == 451.9887363604
    assert result["growth_type"] == "TURNAROUND"
    assert result["sec_current_eps"] == 0.05
    assert result["sec_prior_year_eps"] == 0.0
    assert result["yahoo_current_eps"] == 0.05
    assert result["yahoo_prior_year_eps"] == -0.014205


def test_replay_sec_numeric_does_not_query_yahoo_for_reconciliation(monkeypatch):
    provider = SECYahooEPSProvider()
    sec_records = [
        {
            "report_period": "2025-06-30",
            "start": "2025-04-01",
            "eps_diluted": 1.0,
            "filing_date": "2025-08-05",
            "period_type": "quarter",
            "source": "SEC",
            "concept": "EarningsPerShareDiluted",
            "unit": "USD/shares",
        },
        {
            "report_period": "2026-06-30",
            "start": "2026-04-01",
            "eps_diluted": 1.5,
            "filing_date": "2026-08-05",
            "period_type": "quarter",
            "source": "SEC",
            "concept": "EarningsPerShareDiluted",
            "unit": "USD/shares",
        },
    ]
    monkeypatch.setattr(
        provider.sec,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: sec_records,
    )
    monkeypatch.setattr(
        provider.yahoo,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: (_ for _ in ()).throw(
            AssertionError("normal SEC replay result must not query Yahoo")
        ),
    )

    result, reason = provider.fetch_eps_yoy_detailed("TEST", "2026-08-28")

    assert reason is None
    assert result["source"] == "SEC"
    assert result["eps_yoy_growth"] == 50.0
    assert result["growth_type"] == "GROWTH"
    assert "yahoo_current_eps" not in result


def test_replay_sec_zero_base_stays_nonblocking_when_yahoo_confirmation_fails(
    monkeypatch,
):
    provider = SECYahooEPSProvider()
    sec_records = [
        {
            "report_period": "2025-06-30",
            "start": "2025-04-01",
            "eps_diluted": 0.0,
            "filing_date": "2025-08-05",
            "period_type": "quarter",
            "source": "SEC",
            "concept": "EarningsPerShareDiluted",
            "unit": "USD/shares",
            "source_record_id": "sec-prior",
        },
        {
            "report_period": "2026-06-30",
            "start": "2026-04-01",
            "eps_diluted": 0.05,
            "filing_date": "2026-08-05",
            "period_type": "quarter",
            "source": "SEC",
            "concept": "EarningsPerShareDiluted",
            "unit": "USD/shares",
            "source_record_id": "sec-current",
        },
    ]
    monkeypatch.setattr(
        provider.sec,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: sec_records,
    )
    monkeypatch.setattr(
        provider.yahoo,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: (_ for _ in ()).throw(RuntimeError("Yahoo outage")),
    )

    result, reason = provider.fetch_eps_yoy_detailed("JAN", "2026-08-28")

    assert reason is EPSMissingReason.PRIOR_YEAR_EPS_ZERO
    assert result["growth_type"] == "ZERO_BASE"
    assert result["sec_prior_year_eps"] == 0.0
