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
        lambda symbol: (_ for _ in ()).throw(RuntimeError("SEC provider HTTP 500")),
    )
    monkeypatch.setattr(provider.yahoo, "fetch_quarterly_history", lambda symbol: [])

    with caplog.at_level("WARNING"):
        result, reason = provider.fetch_eps_yoy_detailed("TEST", "2026-08-21")

    assert result is None
    assert reason is EPSMissingReason.PROVIDER_ERROR
    assert "Signal EPS PIT provider error for TEST with no resolved fallback" in caplog.text


def test_single_provider_failure_plus_yahoo_prior_missing_is_provider_error(monkeypatch):
    provider = SECYahooEPSProvider()
    monkeypatch.setattr(
        provider.sec,
        "fetch_quarterly_history",
        lambda symbol: (_ for _ in ()).throw(RuntimeError("SEC ticker map HTTP 403")),
    )
    monkeypatch.setattr(
        provider.yahoo,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: [_resolved_yahoo_record()],
    )

    result, reason = provider.fetch_eps_yoy_detailed("TEST", "2026-08-21")

    assert result is None
    assert reason is EPSMissingReason.PROVIDER_ERROR


def test_single_yahoo_provider_failure_plus_sec_nonterminal_missing_is_provider_error(monkeypatch):
    provider = SECYahooEPSProvider()
    monkeypatch.setattr(provider.sec, "fetch_quarterly_history", lambda symbol: [])
    monkeypatch.setattr(
        provider.yahoo,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: (_ for _ in ()).throw(RuntimeError("Yahoo service unavailable")),
    )

    result, reason = provider.fetch_eps_yoy_detailed("TEST", "2026-08-21")

    assert result is None
    assert reason is EPSMissingReason.PROVIDER_ERROR


def test_other_provider_can_resolve_when_one_provider_fails(monkeypatch):
    provider = SECYahooEPSProvider()
    monkeypatch.setattr(
        provider.sec,
        "fetch_quarterly_history",
        lambda symbol: (_ for _ in ()).throw(RuntimeError("SEC outage")),
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
        lambda symbol: (_ for _ in ()).throw(RuntimeError("SEC outage")),
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
    monkeypatch.setattr(provider.sec, "fetch_quarterly_history", lambda symbol: [])

    def future_only(symbol, **kwargs):
        provider.yahoo.missing_release_periods = ["2027-06-30"]
        return []

    monkeypatch.setattr(provider.yahoo, "fetch_quarterly_history", future_only)
    result, reason = provider.fetch_eps_yoy_detailed("TEST", "2026-08-21")
    assert result is None
    assert reason is EPSMissingReason.NO_QUARTERLY_EPS


def test_relevant_missing_yahoo_release_date_is_explicit(monkeypatch):
    provider = SECYahooEPSProvider()
    monkeypatch.setattr(provider.sec, "fetch_quarterly_history", lambda symbol: [])

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
        lambda symbol: (_ for _ in ()).throw(RuntimeError("SEC outage")),
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


def test_sec_zero_denominator_is_terminal_without_yahoo_call(monkeypatch):
    provider = SECYahooEPSProvider()
    sec_records = [
        {
            "report_period": "2025-06-30",
            "eps_diluted": 0.0,
            "filing_date": "2025-08-01",
            "period_type": "quarter",
            "source": "SEC",
            "concept": "EarningsPerShareDiluted",
            "unit": "USD/shares",
        },
        {
            "report_period": "2026-06-30",
            "eps_diluted": 0.05,
            "filing_date": "2026-08-01",
            "period_type": "quarter",
            "source": "SEC",
            "concept": "EarningsPerShareDiluted",
            "unit": "USD/shares",
        },
    ]
    monkeypatch.setattr(provider.sec, "fetch_quarterly_history", lambda symbol: sec_records)
    monkeypatch.setattr(
        provider.yahoo,
        "fetch_quarterly_history",
        lambda symbol, **kwargs: (_ for _ in ()).throw(
            AssertionError("SEC zero is terminal; Yahoo should not be queried")
        ),
    )

    result, reason = provider.fetch_eps_yoy_detailed("TEST", "2026-08-21")

    assert result is None
    assert reason is EPSMissingReason.PRIOR_YEAR_EPS_ZERO
