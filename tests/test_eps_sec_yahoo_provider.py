import io
import os
import time
import zipfile

import pandas as pd
import pytest
import requests

from eps_pit import EPSMissingReason
from eps_pit.providers import pit_provider
from eps_pit.providers.sec_yahoo_provider import (
    SECProvider,
    TTLJSONCache,
    YahooFundamentalsProvider,
    calculate_latest_eps_yoy,
    calculate_latest_eps_yoy_diagnostic,
    select_visible_quarters,
)


@pytest.fixture(autouse=True)
def _reset_sec_blocked_hosts():
    pit_provider._SEC_BLOCKED_HOST_ERRORS.clear()
    yield
    pit_provider._SEC_BLOCKED_HOST_ERRORS.clear()


@pytest.fixture(autouse=True)
def _reset_sec_circuit_breakers():
    pit_provider._SEC_BLOCKED_HOST_ERRORS.clear()
    yield
    pit_provider._SEC_BLOCKED_HOST_ERRORS.clear()


def _quarter(period, eps, filed, *, start=None, fiscal_quarter=None, concept="EarningsPerShareDiluted", source="SEC", record_id=None):
    period_ts = pd.Timestamp(period)
    if start is None:
        start = (period_ts - pd.Timedelta(days=89)).strftime("%Y-%m-%d")
    return {
        "code": "TEST",
        "report_period": period,
        "start": start,
        "eps_diluted": eps,
        "filing_date": filed if source == "SEC" else None,
        "earnings_release_at": filed if source == "Yahoo" else None,
        "fiscal_quarter": fiscal_quarter or "",
        "concept": concept,
        "unit": "USD/shares",
        "period_type": "quarter",
        "source": source,
        "source_record_id": record_id or f"{source}_{period}_{filed}_{concept}",
    }


def _ytd(period, eps, filed, *, start, period_type, fiscal_quarter):
    return {
        "code": "TEST",
        "report_period": period,
        "start": start,
        "eps_diluted": eps,
        "filing_date": filed,
        "fiscal_quarter": fiscal_quarter,
        "concept": "EarningsPerShareDiluted",
        "unit": "USD/shares",
        "period_type": period_type,
        "source": "SEC",
        "source_record_id": f"{period_type}_{period}_{filed}",
    }


def test_latest_report_period_and_negative_math():
    records = [
        _quarter("2025-06-30", -1.0, "2025-08-01", fiscal_quarter="Q2"),
        _quarter("2026-06-30", 0.5, "2026-08-01", fiscal_quarter="Q2"),
    ]
    result = calculate_latest_eps_yoy(records, "2026-08-21")
    assert result["eps_yoy_growth"] == 150.0
    assert result["current_period"] == "2026-06-30"


def test_amendment_is_snapshot_first_and_old_amendment_cannot_become_current():
    records = [
        _quarter("2025-06-30", 1.0, "2025-08-01", fiscal_quarter="Q2"),
        _quarter("2026-06-30", 1.5, "2026-08-01", fiscal_quarter="Q2", record_id="q2-original"),
        _quarter("2026-06-30", 1.8, "2026-09-15", fiscal_quarter="Q2", record_id="q2-amend"),
        _quarter("2025-09-30", 2.0, "2025-11-01", fiscal_quarter="Q3"),
        _quarter("2026-09-30", 3.0, "2026-11-01", fiscal_quarter="Q3", record_id="q3-current"),
        _quarter("2025-06-30", 1.1, "2026-12-01", fiscal_quarter="Q2", record_id="old-amend"),
    ]
    before = calculate_latest_eps_yoy(records, "2026-08-21")
    after_q2_amend = calculate_latest_eps_yoy(records, "2026-09-20")
    after_q3 = calculate_latest_eps_yoy(records, "2026-12-10")
    assert before["current_eps"] == 1.5
    assert before["source_record_id"] == "q2-original"
    assert after_q2_amend["current_eps"] == 1.8
    assert after_q3["current_period"] == "2026-09-30"
    assert after_q3["source_record_id"] == "q3-current"


def test_reported_quarter_beats_ytd_for_same_period():
    records = [
        _quarter("2025-06-30", 0.8, "2025-08-01", fiscal_quarter="Q2"),
        _quarter("2026-06-30", 1.0, "2026-08-01", fiscal_quarter="Q2", record_id="reported"),
        _ytd("2026-06-30", 9.99, "2026-08-01", start="2026-01-01", period_type="ytd_6m", fiscal_quarter="Q2"),
    ]
    result = calculate_latest_eps_yoy(records, "2026-08-21")
    assert result["current_eps"] == 1.0
    assert result["source_record_id"] == "reported"
    assert result["calculation_method"] == "reported_quarter"


def test_q2_q3_and_q4_are_safely_derived_when_direct_quarter_missing():
    q2 = [
        _quarter("2025-03-31", 1.0, "2025-05-01", start="2025-01-01", fiscal_quarter="Q1"),
        _ytd("2025-06-30", 3.0, "2025-08-01", start="2025-01-01", period_type="ytd_6m", fiscal_quarter="Q2"),
        _quarter("2026-03-31", 1.5, "2026-05-01", start="2026-01-01", fiscal_quarter="Q1"),
        _ytd("2026-06-30", 4.5, "2026-08-01", start="2026-01-01", period_type="ytd_6m", fiscal_quarter="Q2"),
    ]
    q2_result = calculate_latest_eps_yoy(q2, "2026-08-21")
    assert (q2_result["current_eps"], q2_result["prior_year_eps"]) == (3.0, 2.0)
    assert q2_result["calculation_method"] == "derived_from_ytd"

    q3 = [
        _ytd("2025-06-30", 3.0, "2025-08-01", start="2025-01-01", period_type="ytd_6m", fiscal_quarter="Q2"),
        _ytd("2025-09-30", 5.0, "2025-11-01", start="2025-01-01", period_type="ytd_9m", fiscal_quarter="Q3"),
        _ytd("2026-06-30", 4.0, "2026-08-01", start="2026-01-01", period_type="ytd_6m", fiscal_quarter="Q2"),
        _ytd("2026-09-30", 7.0, "2026-11-01", start="2026-01-01", period_type="ytd_9m", fiscal_quarter="Q3"),
    ]
    q3_result = calculate_latest_eps_yoy(q3, "2026-11-10")
    assert (q3_result["current_eps"], q3_result["prior_year_eps"]) == (3.0, 2.0)
    assert q3_result["calculation_method"] == "derived_from_ytd"

    q4 = [
        _quarter("2025-03-31", 1.0, "2025-05-01", start="2025-01-01", fiscal_quarter="Q1"),
        _quarter("2025-06-30", 2.0, "2025-08-01", start="2025-04-01", fiscal_quarter="Q2"),
        _quarter("2025-09-30", 3.0, "2025-11-01", start="2025-07-01", fiscal_quarter="Q3"),
        _ytd("2025-12-31", 10.0, "2026-02-15", start="2025-01-01", period_type="fy", fiscal_quarter="FY"),
        _quarter("2026-03-31", 1.5, "2026-05-01", start="2026-01-01", fiscal_quarter="Q1"),
        _quarter("2026-06-30", 2.5, "2026-08-01", start="2026-04-01", fiscal_quarter="Q2"),
        _quarter("2026-09-30", 4.0, "2026-11-01", start="2026-07-01", fiscal_quarter="Q3"),
        _ytd("2026-12-31", 14.0, "2027-02-15", start="2026-01-01", period_type="fy", fiscal_quarter="FY"),
    ]
    q4_result = calculate_latest_eps_yoy(q4, "2027-02-20")
    assert (q4_result["current_eps"], q4_result["prior_year_eps"]) == (6.0, 4.0)
    assert q4_result["calculation_method"] == "derived_from_fy"


def test_ytd_is_never_used_directly_when_components_missing():
    records = [_ytd("2026-06-30", 99.0, "2026-08-01", start="2026-01-01", period_type="ytd_6m", fiscal_quarter="Q2")]
    assert select_visible_quarters(records, "2026-08-21") == []


def test_diluted_matches_diluted_and_zero_prior_is_explicit():
    records = [
        _quarter("2025-06-30", 1.0, "2025-08-01", fiscal_quarter="Q2", concept="EarningsPerShareDiluted"),
        _quarter("2025-06-30", 2.0, "2025-08-01", fiscal_quarter="Q2", concept="EarningsPerShareBasic"),
        _quarter("2026-06-30", 1.5, "2026-08-01", fiscal_quarter="Q2", concept="EarningsPerShareDiluted"),
    ]
    result = calculate_latest_eps_yoy(records, "2026-08-21")
    assert result["prior_year_eps"] == 1.0

    zero = [
        _quarter("2025-06-30", 0.0, "2025-08-01", fiscal_quarter="Q2"),
        _quarter("2026-06-30", 1.0, "2026-08-01", fiscal_quarter="Q2"),
    ]
    result, reason = calculate_latest_eps_yoy_diagnostic(zero, "2026-08-21")
    assert result is None
    assert reason is EPSMissingReason.PRIOR_YEAR_EPS_ZERO


def test_future_records_are_excluded_before_version_selection():
    records = [
        _quarter("2025-06-30", 1.0, "2025-08-01", fiscal_quarter="Q2"),
        _quarter("2026-06-30", 1.5, "2026-08-01", fiscal_quarter="Q2"),
    ]
    result, reason = calculate_latest_eps_yoy_diagnostic(records, "2026-07-31")
    assert result is None
    assert reason in {EPSMissingReason.NO_PRIOR_YEAR_QUARTER, EPSMissingReason.NO_QUARTERLY_EPS}


def test_sec_parser_preserves_versions_and_duration_metadata(tmp_path):
    facts = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": {
                    "units": {
                        "USD/shares": [
                            {"start": "2026-04-01", "end": "2026-06-30", "val": 1.5, "filed": "2026-08-01", "accepted": "2026-08-01T12:00:00", "form": "10-Q", "fp": "Q2", "fy": 2026, "accn": "original"},
                            {"start": "2026-04-01", "end": "2026-06-30", "val": 1.8, "filed": "2026-09-15", "accepted": "2026-09-15T12:00:00", "form": "10-Q/A", "fp": "Q2", "fy": 2026, "accn": "amendment"},
                            {"start": "2026-01-01", "end": "2026-06-30", "val": 2.5, "filed": "2026-08-01", "form": "10-Q", "fp": "Q2", "fy": 2026, "accn": "ytd"},
                        ]
                    }
                }
            }
        }
    }
    records = SECProvider(tmp_path)._parse_company_facts("TEST", facts)
    assert len(records) == 3
    assert [r["period_type"] for r in records].count("quarter") == 2
    assert [r["period_type"] for r in records].count("ytd_6m") == 1
    assert all(r["unit"] == "USD/shares" for r in records)


def test_sec_provider_uses_fixed_privacy_preserving_identity(tmp_path):
    provider = SECProvider(tmp_path)

    assert provider.headers["User-Agent"] == pit_provider.SEC_USER_AGENT
    assert provider.headers["User-Agent"].endswith("@users.noreply.github.com")
    assert provider.headers["Accept-Encoding"] == "gzip, deflate"


class _FakeSECResponse:
    def __init__(self, status_code, payload=None, headers=None, body=b""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self._body = body

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=1024 * 1024):
        if self._body:
            yield self._body

    def close(self):
        return None


def test_sec_retry_is_limited_to_retryable_statuses(monkeypatch, tmp_path):
    responses = [
        _FakeSECResponse(429),
        _FakeSECResponse(503),
        _FakeSECResponse(200, {"ok": True}),
    ]
    calls = []

    def fake_get(url, timeout, **kwargs):
        calls.append((url, timeout, kwargs))
        return responses.pop(0)

    provider = SECProvider(
        tmp_path,
        rate_limit_sleep=0,
        max_retries=2,
    )
    monkeypatch.setattr(provider.session, "get", fake_get)
    monkeypatch.setattr(pit_provider.time, "sleep", lambda seconds: None)

    assert provider._get_json("https://data.sec.gov/test", label="SEC test") == {"ok": True}
    assert len(calls) == 3


def test_sec_403_fails_immediately_without_blind_retry(monkeypatch, tmp_path):
    calls = []

    def fake_get(url, timeout, **kwargs):
        calls.append((url, timeout, kwargs))
        return _FakeSECResponse(403)

    provider = SECProvider(
        tmp_path,
        rate_limit_sleep=0,
        max_retries=2,
    )
    monkeypatch.setattr(provider.session, "get", fake_get)
    with pytest.raises(RuntimeError, match="SEC ticker map HTTP 403"):
        provider._get_json(provider.TICKERS_URL, label="SEC ticker map")

    assert len(calls) == 1


def test_sec_403_is_shared_across_providers_for_same_host(monkeypatch, tmp_path):
    calls = []

    first = SECProvider(tmp_path / "first", rate_limit_sleep=0, max_retries=2)
    monkeypatch.setattr(
        first.session,
        "get",
        lambda url, timeout, **kwargs: (
            calls.append(url) or _FakeSECResponse(403)
        ),
    )
    with pytest.raises(RuntimeError, match="SEC ticker map HTTP 403"):
        first._get_json(first.TICKERS_URL, label="SEC ticker map")

    second = SECProvider(tmp_path / "second", rate_limit_sleep=0, max_retries=2)
    monkeypatch.setattr(
        second.session,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("same blocked host must not be retried")
        ),
    )
    with pytest.raises(RuntimeError, match="SEC ticker map HTTP 403"):
        second._get_json(second.TICKERS_URL, label="SEC ticker map")

    assert calls == [first.TICKERS_URL]


def test_sec_403_does_not_block_other_sec_host(monkeypatch, tmp_path):
    blocked = SECProvider(tmp_path / "blocked", rate_limit_sleep=0, max_retries=0)
    monkeypatch.setattr(
        blocked.session,
        "get",
        lambda url, timeout, **kwargs: _FakeSECResponse(403),
    )
    with pytest.raises(RuntimeError, match="SEC ticker map HTTP 403"):
        blocked._get_json(blocked.TICKERS_URL, label="SEC ticker map")

    data_provider = SECProvider(tmp_path / "data", rate_limit_sleep=0, max_retries=0)
    calls = []
    monkeypatch.setattr(
        data_provider.session,
        "get",
        lambda url, timeout, **kwargs: (
            calls.append(url) or _FakeSECResponse(200, {"ok": True})
        ),
    )

    url = data_provider.FACTS_URL.format(cik="0000008203")
    assert data_provider._get_json(url, label="SEC companyfacts") == {"ok": True}
    assert calls == [url]


def test_sec_bulk_403_is_shared_for_www_host(monkeypatch, tmp_path):
    first = SECProvider(tmp_path / "first", rate_limit_sleep=0, max_retries=0)
    monkeypatch.setattr(
        first.session,
        "get",
        lambda url, timeout, **kwargs: _FakeSECResponse(403),
    )
    with pytest.raises(RuntimeError, match="SEC companyfacts bulk HTTP 403"):
        first._download_file(
            pit_provider.SEC_BULK_COMPANYFACTS_URL,
            first.bulk_companyfacts_zip,
            label="SEC companyfacts bulk",
        )

    second = SECProvider(tmp_path / "second", rate_limit_sleep=0, max_retries=0)
    monkeypatch.setattr(
        second.session,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("blocked www.sec.gov must not be retried")
        ),
    )
    with pytest.raises(RuntimeError, match="SEC companyfacts bulk HTTP 403"):
        second._download_file(
            pit_provider.SEC_BULK_COMPANYFACTS_URL,
            second.bulk_companyfacts_zip,
            label="SEC companyfacts bulk",
        )


def test_sec_403_circuit_breaker_is_shared_across_provider_instances(
    monkeypatch,
    tmp_path,
):
    first_calls = []
    second_calls = []

    first = SECProvider(tmp_path / "first", rate_limit_sleep=0, max_retries=0)
    second = SECProvider(tmp_path / "second", rate_limit_sleep=0, max_retries=0)

    monkeypatch.setattr(
        first.session,
        "get",
        lambda url, timeout, **kwargs: (
            first_calls.append(url) or _FakeSECResponse(403)
        ),
    )
    monkeypatch.setattr(
        second.session,
        "get",
        lambda url, timeout, **kwargs: (
            second_calls.append(url) or _FakeSECResponse(200, {"unexpected": True})
        ),
    )

    with pytest.raises(RuntimeError, match="SEC ticker map HTTP 403"):
        first._get_json(first.TICKERS_URL, label="SEC ticker map")

    with pytest.raises(RuntimeError, match="SEC ticker map HTTP 403"):
        second._get_json(second.TICKERS_URL, label="SEC ticker map")

    assert first_calls == [first.TICKERS_URL]
    assert second_calls == []


def test_yahoo_requires_verified_release_but_keeps_verified_history(monkeypatch, tmp_path):
    class NoReleaseTicker:
        def get_earnings_dates(self, limit=32):
            return pd.DataFrame()
        @property
        def quarterly_income_stmt(self):
            return pd.DataFrame({pd.Timestamp("2026-06-30"): [1.25]}, index=["Diluted EPS"])

    monkeypatch.setattr(pit_provider.yf, "Ticker", lambda symbol: NoReleaseTicker())
    provider = YahooFundamentalsProvider(tmp_path / "missing")
    assert provider.fetch_quarterly_history("TEST") == []
    assert provider.missing_release_periods == ["2026-06-30"]

    class StaleTicker:
        def get_earnings_dates(self, limit=32):
            index = pd.to_datetime(["2026-08-13", "2026-05-07", "2025-05-01"])
            return pd.DataFrame({"Reported EPS": [0.90, 0.27, -1.58]}, index=index)
        @property
        def quarterly_income_stmt(self):
            return pd.DataFrame({pd.Timestamp("2026-03-31"): [9.75], pd.Timestamp("2025-03-31"): [-1.58]}, index=["Diluted EPS"])

    monkeypatch.setattr(pit_provider.yf, "Ticker", lambda symbol: StaleTicker())
    records = YahooFundamentalsProvider(tmp_path / "stale").fetch_quarterly_history("ASND")
    assert [r["report_period"] for r in records] == ["2025-03-31", "2026-03-31"]
    assert all(r["eps_diluted"] != 0.90 for r in records)


def test_json_cache_ttl_invalidates_stale_file(tmp_path):
    path = tmp_path / "cache.json"
    TTLJSONCache.write(path, {"value": 1})
    stale_time = time.time() - 120
    os.utime(path, (stale_time, stale_time))
    assert TTLJSONCache(ttl_seconds=60).load(path) is None
    assert TTLJSONCache(ttl_seconds=3600).load(path) == {"value": 1}


def test_yahoo_current_observation_does_not_require_historical_release_dates(
    monkeypatch,
    tmp_path,
):
    class CurrentOnlyTicker:
        def get_earnings_dates(self, limit=32):
            raise AssertionError("LIVE current observation must not depend on earnings_dates")

        @property
        def quarterly_income_stmt(self):
            return pd.DataFrame(
                {
                    pd.Timestamp("2026-04-30"): [0.08],
                    pd.Timestamp("2025-04-30"): [-0.05],
                },
                index=["Diluted EPS"],
            )

    monkeypatch.setattr(pit_provider.yf, "Ticker", lambda symbol: CurrentOnlyTicker())
    provider = YahooFundamentalsProvider(tmp_path / "live")

    records = provider.fetch_quarterly_history(
        "ALOT",
        require_release_date=False,
        observed_on="2026-08-27",
        refresh=True,
    )

    assert [record["report_period"] for record in records] == [
        "2025-04-30",
        "2026-04-30",
    ]
    assert all(record["source"] == "YahooLiveObserved" for record in records)
    assert all(record["earnings_release_at"] == "2026-08-27" for record in records)

    result, reason = calculate_latest_eps_yoy_diagnostic(records, "2026-08-27")
    assert reason is None
    assert result["eps_yoy_growth"] == 260.0
    assert result["effective_date"] == "2026-08-27"


def test_yahoo_historical_reconstruction_still_requires_release_dates(
    monkeypatch,
    tmp_path,
):
    class NoHistoricalReleaseTicker:
        def get_earnings_dates(self, limit=32):
            return pd.DataFrame()

        @property
        def quarterly_income_stmt(self):
            return pd.DataFrame(
                {
                    pd.Timestamp("2026-04-30"): [0.08],
                    pd.Timestamp("2025-04-30"): [-0.05],
                },
                index=["Diluted EPS"],
            )

    monkeypatch.setattr(
        pit_provider.yf,
        "Ticker",
        lambda symbol: NoHistoricalReleaseTicker(),
    )
    provider = YahooFundamentalsProvider(tmp_path / "historical")

    records = provider.fetch_quarterly_history(
        "ALOT",
        require_release_date=True,
    )

    assert records == []
    assert provider.missing_release_periods == ["2026-04-30", "2025-04-30"]


def test_sec_historical_prefers_existing_bulk_without_network(monkeypatch, tmp_path):
    provider = SECProvider(
        tmp_path,
        rate_limit_sleep=0,
        bulk_cache_ttl_seconds=3600,
    )
    ticker_map = {
        "0": {"ticker": "ALOT", "cik_str": 8203},
    }
    TTLJSONCache.write(provider.cache_dir / "company_tickers.json", ticker_map)

    facts = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": {
                    "units": {
                        "USD/shares": [
                            {
                                "start": "2025-02-01",
                                "end": "2025-04-30",
                                "val": -0.05,
                                "filed": "2025-06-09",
                                "accepted": "2025-06-09T12:00:00",
                                "form": "10-Q",
                                "fp": "Q1",
                                "fy": 2026,
                                "accn": "prior",
                            },
                            {
                                "start": "2026-02-01",
                                "end": "2026-04-30",
                                "val": 0.08,
                                "filed": "2026-06-08",
                                "accepted": "2026-06-08T12:00:00",
                                "form": "10-Q",
                                "fp": "Q1",
                                "fy": 2027,
                                "accn": "current",
                            },
                        ]
                    }
                }
            }
        }
    }
    with zipfile.ZipFile(provider.bulk_companyfacts_zip, "w") as archive:
        archive.writestr("CIK0000008203.json", __import__("json").dumps(facts))

    monkeypatch.setattr(
        provider.session,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("existing historical bulk should avoid SEC network")
        ),
    )

    records = provider.fetch_quarterly_history("ALOT", prefer_bulk=True)
    result = calculate_latest_eps_yoy(records, "2026-08-21")

    assert result["eps_yoy_growth"] == 260.0
    assert result["source"] == "SEC"


def test_sec_bulk_prepare_downloads_at_most_once_per_provider(monkeypatch, tmp_path):
    provider = SECProvider(
        tmp_path,
        rate_limit_sleep=0,
        bulk_cache_ttl_seconds=0,
    )
    calls = []

    def fake_download(url, destination, *, label):
        calls.append((url, destination, label))
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("CIK0000008203.json", "{}")

    monkeypatch.setattr(provider, "_download_file", fake_download)

    first = provider.ensure_bulk_companyfacts()
    second = provider.ensure_bulk_companyfacts()

    assert first == second == provider.bulk_companyfacts_zip
    assert len(calls) == 1


def test_sec_bulk_refresh_failure_uses_stale_zip(monkeypatch, tmp_path):
    provider = SECProvider(
        tmp_path,
        rate_limit_sleep=0,
        bulk_cache_ttl_seconds=1,
    )
    with zipfile.ZipFile(provider.bulk_companyfacts_zip, "w") as archive:
        archive.writestr("CIK0000008203.json", "{}")
    stale_time = time.time() - 3600
    os.utime(provider.bulk_companyfacts_zip, (stale_time, stale_time))

    monkeypatch.setattr(
        provider,
        "_download_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("SEC companyfacts bulk HTTP 403")
        ),
    )

    assert provider.ensure_bulk_companyfacts() == provider.bulk_companyfacts_zip


def test_yahoo_rate_limit_error_uses_bounded_backoff(monkeypatch, tmp_path):
    calls = {"ticker": 0}
    sleeps = []

    class YFRateLimitError(Exception):
        pass

    class RateLimitedTicker:
        @property
        def quarterly_income_stmt(self):
            if calls["ticker"] < 3:
                raise YFRateLimitError("Too Many Requests")
            return pd.DataFrame(
                {
                    pd.Timestamp("2026-04-30"): [0.08],
                    pd.Timestamp("2025-04-30"): [-0.05],
                },
                index=["Diluted EPS"],
            )

    def fake_ticker(symbol):
        calls["ticker"] += 1
        return RateLimitedTicker()

    monkeypatch.setattr(pit_provider.yf, "Ticker", fake_ticker)
    monkeypatch.setattr(pit_provider.time, "sleep", lambda seconds: sleeps.append(seconds))

    provider = YahooFundamentalsProvider(
        tmp_path,
        rate_limit_sleep=0,
        max_rate_limit_retries=3,
    )
    records = provider.fetch_quarterly_history(
        "ALOT",
        require_release_date=False,
        observed_on="2026-08-27",
        refresh=True,
    )

    assert len(records) == 2
    assert calls["ticker"] == 3
    assert sleeps == [5.0, 15.0]


def _valid_zip_bytes(name="CIK0000008203.json", payload=b"{}"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, payload)
    return buffer.getvalue()


def test_sec_bulk_download_resumes_existing_partial_with_range(monkeypatch, tmp_path):
    provider = SECProvider(tmp_path, rate_limit_sleep=0, max_retries=0)
    destination = provider.bulk_companyfacts_zip
    partial = provider._partial_download_path(destination)
    payload = _valid_zip_bytes()
    split = max(1, len(payload) // 3)
    partial.write_bytes(payload[:split])
    calls = []

    def fake_get(url, timeout, stream, headers):
        calls.append(headers)
        assert headers == {"Range": f"bytes={split}-"}
        return _FakeSECResponse(
            206,
            headers={
                "Content-Range": f"bytes {split}-{len(payload) - 1}/{len(payload)}",
                "Content-Length": str(len(payload) - split),
            },
            body=payload[split:],
        )

    monkeypatch.setattr(provider.session, "get", fake_get)

    provider._download_file(
        pit_provider.SEC_BULK_COMPANYFACTS_URL,
        destination,
        label="SEC companyfacts bulk",
    )

    assert calls == [{"Range": f"bytes={split}-"}]
    assert destination.read_bytes() == payload
    assert not partial.exists()


def test_sec_bulk_download_restarts_if_server_ignores_range(monkeypatch, tmp_path):
    provider = SECProvider(tmp_path, rate_limit_sleep=0, max_retries=0)
    destination = provider.bulk_companyfacts_zip
    partial = provider._partial_download_path(destination)
    payload = _valid_zip_bytes()
    partial.write_bytes(b"stale-partial")
    calls = []

    def fake_get(url, timeout, stream, headers):
        calls.append(headers)
        assert headers == {"Range": f"bytes={len(b'stale-partial')}-"}
        return _FakeSECResponse(
            200,
            headers={"Content-Length": str(len(payload))},
            body=payload,
        )

    monkeypatch.setattr(provider.session, "get", fake_get)

    provider._download_file(
        pit_provider.SEC_BULK_COMPANYFACTS_URL,
        destination,
        label="SEC companyfacts bulk",
    )

    assert len(calls) == 1
    assert destination.read_bytes() == payload
    assert not partial.exists()


def test_sec_bulk_partial_survives_process_restart_and_resumes(monkeypatch, tmp_path):
    payload = _valid_zip_bytes()
    split = max(1, len(payload) // 2)
    destination = tmp_path / "sec" / pit_provider.SEC_BULK_COMPANYFACTS_FILENAME

    class InterruptedResponse(_FakeSECResponse):
        def iter_content(self, chunk_size=1024 * 1024):
            yield payload[:split]
            raise requests.ConnectionError("connection dropped")

    first = SECProvider(tmp_path, rate_limit_sleep=0, max_retries=0)
    monkeypatch.setattr(
        first.session,
        "get",
        lambda url, timeout, stream, headers: InterruptedResponse(
            200,
            headers={
                "Content-Length": str(len(payload)),
                "ETag": '"archive-v1"',
            },
        ),
    )

    with pytest.raises(requests.ConnectionError, match="connection dropped"):
        first._download_file(
            pit_provider.SEC_BULK_COMPANYFACTS_URL,
            destination,
            label="SEC companyfacts bulk",
        )

    partial = first._partial_download_path(destination)
    assert partial.exists()
    assert partial.read_bytes() == payload[:split]

    second = SECProvider(tmp_path, rate_limit_sleep=0, max_retries=0)
    calls = []

    def resumed_get(url, timeout, stream, headers):
        calls.append(headers)
        assert headers == {
            "Range": f"bytes={split}-",
            "If-Range": '"archive-v1"',
        }
        return _FakeSECResponse(
            206,
            headers={
                "Content-Range": f"bytes {split}-{len(payload) - 1}/{len(payload)}",
                "Content-Length": str(len(payload) - split),
            },
            body=payload[split:],
        )

    monkeypatch.setattr(second.session, "get", resumed_get)

    second._download_file(
        pit_provider.SEC_BULK_COMPANYFACTS_URL,
        destination,
        label="SEC companyfacts bulk",
    )

    assert calls == [
        {
            "Range": f"bytes={split}-",
            "If-Range": '"archive-v1"',
        }
    ]
    assert destination.read_bytes() == payload
    assert not partial.exists()
    assert not second._partial_metadata_path(destination).exists()


def test_sec_bulk_resume_restarts_when_remote_archive_changed(monkeypatch, tmp_path):
    provider = SECProvider(tmp_path, rate_limit_sleep=0, max_retries=0)
    destination = provider.bulk_companyfacts_zip
    partial = provider._partial_download_path(destination)
    metadata = provider._partial_metadata_path(destination)
    old_payload = _valid_zip_bytes(payload=b'{"old": true}')
    new_payload = _valid_zip_bytes(payload=b'{"new": true}')
    split = max(1, len(old_payload) // 3)
    partial.write_bytes(old_payload[:split])
    metadata.write_text(__import__("json").dumps({"if_range": '"archive-v1"'}))
    calls = []

    def fake_get(url, timeout, stream, headers):
        calls.append(headers)
        assert headers == {
            "Range": f"bytes={split}-",
            "If-Range": '"archive-v1"',
        }
        # If-Range validator no longer matches, so the server returns the full
        # new representation with HTTP 200.
        return _FakeSECResponse(
            200,
            headers={
                "Content-Length": str(len(new_payload)),
                "ETag": '"archive-v2"',
            },
            body=new_payload,
        )

    monkeypatch.setattr(provider.session, "get", fake_get)

    provider._download_file(
        pit_provider.SEC_BULK_COMPANYFACTS_URL,
        destination,
        label="SEC companyfacts bulk",
    )

    assert destination.read_bytes() == new_payload
    assert not partial.exists()
    assert not metadata.exists()
