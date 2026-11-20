"""Yahoo exception quarantine is precise, batch-scoped, and fail-closed."""
from collections import Counter

import pandas as pd
import pytest
from yfinance.exceptions import YFException, YFRateLimitError

import DataStore
import data_providers.yahoo_provider as yahoo_module
from data_providers.ohlcv_validation import DataIntegrityError
from data_providers.yahoo_failure_policy import (
    YahooFailureAction, classify_yahoo_exception,
)
from data_providers.yahoo_provider import YahooDataProvider


def _bar():
    return pd.DataFrame(
        {"Open": [10.0], "High": [11.0], "Low": [9.0],
         "Close": [10.5], "Volume": [1000]},
        index=pd.to_datetime(["2026-11-16"]),
    )


def _dividend_error(interval):
    return YFException(
        "The following 'Dividends' events are out-of-range, "
        f"did not expect with interval {interval}: DatetimeIndex("
        "['2022-03-14 00:00:00-04:00'], dtype='datetime64[ns, America/New_York]')"
    )


@pytest.mark.parametrize("interval", ["1d", "1wk"])
def test_exact_dividend_merge_exception_is_quarantined(interval):
    decision = classify_yahoo_exception(_dividend_error(interval), interval=interval)
    assert decision.action is YahooFailureAction.QUARANTINE
    assert decision.reason_code == "DIVIDEND_EVENT_OUT_OF_RANGE"


@pytest.mark.parametrize("error,interval", [
    (_dividend_error("1wk"), "1d"),
    (YFException("The following 'Stock Splits' events are out-of-range, did not expect with interval 1wk: DatetimeIndex("), "1wk"),
    (YFException("could not download"), "1wk"),
    (RuntimeError(str(_dividend_error("1wk"))), "1wk"),
    (YFRateLimitError(), "1wk"),
])
def test_nonallowlisted_errors_still_retry(error, interval):
    assert classify_yahoo_exception(error, interval=interval).action is YahooFailureAction.RETRY


def test_quarantine_skips_batch_recovery_but_preserves_valid_symbols(monkeypatch):
    calls = Counter()

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            calls[self.symbol] += 1
            if self.symbol == "NWE" and kwargs["interval"] == "1wk":
                raise _dividend_error("1wk")
            return _bar()

    monkeypatch.setattr(yahoo_module.yf, "Ticker", FakeTicker)
    provider = YahooDataProvider(max_retries=1, max_workers=2, recovery_max_workers=1)
    symbols = ["^GSPC", "^IXIC", "^DJI", "NWE", "MSFT"]
    data, failed = provider.download_batch_stocks(symbols, period="5y", interval="1wk")
    assert failed == ["NWE"]
    assert set(data) == set(symbols) - {"NWE"}
    assert calls["NWE"] == 1
    assert provider.quarantined_symbols == {"NWE"}
    assert provider.validation_failure_symbols == set()
    assert provider.allowed_failure_symbols == {"NWE"}
    assert "DIVIDEND_EVENT_OUT_OF_RANGE" in provider.failure_reasons["NWE"]

    filtered, excluded = DataStore.filter_downloaded_stock_data(
        data, failed=failed, expected_symbols=symbols,
        interval="1wk", provider_name="Yahoo",
        failure_reasons=provider.failure_reasons,
        allowed_failed_symbols=provider.allowed_failure_symbols,
    )
    assert excluded == ["NWE"]
    assert "NWE" not in filtered

    # The next independent 1D batch reattempts NWE instead of blacklisting it.
    daily, daily_failed = provider.download_batch_stocks(
        symbols, period="2y", interval="1d"
    )
    assert daily_failed == []
    assert "NWE" in daily
    assert calls["NWE"] == 2
    assert provider.quarantined_symbols == set()


def test_transport_failure_cannot_be_overwritten_by_dividend_error(monkeypatch):
    calls = Counter()

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            calls[self.symbol] += 1
            if self.symbol == "NWE":
                if calls[self.symbol] == 1:
                    raise RuntimeError("connection lost")
                raise _dividend_error("1wk")
            return _bar()

    monkeypatch.setattr(yahoo_module.yf, "Ticker", FakeTicker)
    provider = YahooDataProvider(max_retries=1, max_workers=2, recovery_max_workers=1)
    symbols = ["^GSPC", "^IXIC", "^DJI", "NWE", "MSFT"]
    data, failed = provider.download_batch_stocks(symbols, period="5y", interval="1wk")
    assert failed == ["NWE"]
    assert provider.quarantined_symbols == set()
    assert provider.allowed_failure_symbols == set()
    assert "connection lost" in provider.failure_reasons["NWE"]
    with pytest.raises(DataIntegrityError, match="non-validation download failures"):
        DataStore.filter_downloaded_stock_data(
            data, failed=failed, expected_symbols=symbols,
            interval="1wk", provider_name="Yahoo",
            allowed_failed_symbols=provider.allowed_failure_symbols,
        )


def test_required_index_cannot_be_quarantined_for_publication(monkeypatch):
    data = {s: _bar() for s in ["^IXIC", "^DJI", "MSFT"]}
    with pytest.raises(DataIntegrityError, match="market reference missing"):
        DataStore.filter_downloaded_stock_data(
            data, failed=["^GSPC"],
            expected_symbols=["^GSPC", *data], interval="1wk",
            provider_name="Yahoo", allowed_failed_symbols={"^GSPC"},
        )
