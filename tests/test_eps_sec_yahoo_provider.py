import os
import time

import pandas as pd
import pytest

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
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


def test_sec_retry_is_limited_to_retryable_statuses(monkeypatch, tmp_path):
    responses = [
        _FakeSECResponse(429),
        _FakeSECResponse(503),
        _FakeSECResponse(200, {"ok": True}),
    ]
    calls = []

    def fake_get(url, headers, timeout):
        calls.append((url, headers, timeout))
        return responses.pop(0)

    monkeypatch.setattr(pit_provider.requests, "get", fake_get)
    monkeypatch.setattr(pit_provider.time, "sleep", lambda seconds: None)

    provider = SECProvider(
        tmp_path,
        rate_limit_sleep=0,
        max_retries=2,
    )
    assert provider._get_json("https://data.sec.gov/test", label="SEC test") == {"ok": True}
    assert len(calls) == 3


def test_sec_403_fails_immediately_without_blind_retry(monkeypatch, tmp_path):
    calls = []

    def fake_get(url, headers, timeout):
        calls.append((url, headers, timeout))
        return _FakeSECResponse(403)

    monkeypatch.setattr(pit_provider.requests, "get", fake_get)

    provider = SECProvider(
        tmp_path,
        rate_limit_sleep=0,
        max_retries=2,
        user_agent="Yfinance_data EPS PIT ops@example.com",
    )
    with pytest.raises(RuntimeError, match="SEC ticker map HTTP 403"):
        provider._get_json(provider.TICKERS_URL, label="SEC ticker map")

    assert len(calls) == 1


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
