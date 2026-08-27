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


def test_partial_provider_failure_is_not_downgraded_to_expected_missing(monkeypatch):
    provider = SECYahooEPSProvider()
    monkeypatch.setattr(
        provider.sec,
        "fetch_quarterly_history",
        lambda symbol: (_ for _ in ()).throw(RuntimeError("SEC outage")),
    )
    monkeypatch.setattr(provider.yahoo, "fetch_quarterly_history", lambda symbol: [])

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
        lambda symbol: [_prior_yahoo_record(), _resolved_yahoo_record()],
    )

    result, reason = provider.fetch_eps_yoy_detailed("TEST", "2026-08-21")

    assert reason is None
    assert result["eps_yoy_growth"] == 50.0
    assert result["source"] == "Yahoo"


def test_missing_yahoo_release_date_is_snapshot_scoped(monkeypatch):
    provider = SECYahooEPSProvider()
    monkeypatch.setattr(provider.sec, "fetch_quarterly_history", lambda symbol: [])

    def future_only(symbol):
        provider.yahoo.missing_release_periods = ["2027-06-30"]
        return []

    monkeypatch.setattr(provider.yahoo, "fetch_quarterly_history", future_only)
    result, reason = provider.fetch_eps_yoy_detailed("TEST", "2026-08-21")
    assert result is None
    assert reason is EPSMissingReason.NO_QUARTERLY_EPS


def test_relevant_missing_yahoo_release_date_is_explicit(monkeypatch):
    provider = SECYahooEPSProvider()
    monkeypatch.setattr(provider.sec, "fetch_quarterly_history", lambda symbol: [])

    def current_period(symbol):
        provider.yahoo.missing_release_periods = ["2026-06-30"]
        return []

    monkeypatch.setattr(provider.yahoo, "fetch_quarterly_history", current_period)
    result, reason = provider.fetch_eps_yoy_detailed("TEST", "2026-08-21")
    assert result is None
    assert reason is EPSMissingReason.NO_VERIFIED_YAHOO_RELEASE_DATE
