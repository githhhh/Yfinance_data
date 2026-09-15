import threading
import time

import pandas as pd
import pytest

import data_providers.yahoo_provider as yahoo_module
from data_providers.ohlcv_validation import DataIntegrityError
from data_providers.yahoo_provider import YahooDataProvider


def _invalid_frame():
    return pd.DataFrame(
        {
            "Open": [10.0],
            "High": [11.0],
            "Low": [9.0],
            "Close": [None],
            "Volume": [1000],
        },
        index=pd.to_datetime(["2026-09-14"]),
    )


def test_data_integrity_error_is_not_retried(monkeypatch):
    calls = []

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            calls.append(self.symbol)
            return _invalid_frame()

    monkeypatch.setattr(yahoo_module.yf, "Ticker", FakeTicker)

    provider = YahooDataProvider(max_retries=1)
    with pytest.raises(DataIntegrityError, match="null/non-numeric OHLCV"):
        provider.download_single_stock("BAD")

    assert calls == ["BAD"]


def test_batch_aborts_without_submitting_later_tickers(monkeypatch):
    calls = []
    barrier = threading.Barrier(2)

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            calls.append(self.symbol)
            if self.symbol in {"BAD", "OTHER"}:
                barrier.wait(timeout=1.0)

            if self.symbol == "BAD":
                return _invalid_frame()
            if self.symbol == "OTHER":
                time.sleep(0.05)
                raise RuntimeError("transient failure")
            raise AssertionError(f"unexpected download for {self.symbol}")

    monkeypatch.setattr(yahoo_module.yf, "Ticker", FakeTicker)

    provider = YahooDataProvider(
        batch_size=100,
        max_workers=2,
        max_retries=1,
    )

    with pytest.raises(DataIntegrityError, match="null/non-numeric OHLCV"):
        provider.download_batch_stocks(
            ["BAD", "OTHER", "LATER1", "LATER2"],
            period="2y",
            interval="1d",
        )

    assert calls.count("BAD") == 1
    assert calls.count("OTHER") == 1
    assert "LATER1" not in calls
    assert "LATER2" not in calls
