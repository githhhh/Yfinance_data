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
    symbol, frame = provider.download_single_stock("BAD")

    assert (symbol, frame) == ("BAD", None)
    assert calls == ["BAD"]
    assert provider.validation_failure_symbols == {"BAD"}
    assert "null/non-numeric OHLCV" in provider.failure_reasons["BAD"]


def test_batch_quarantines_bad_symbol_and_continues_later_tickers(monkeypatch):
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
            return pd.DataFrame(
                {"Open": [10.0], "High": [11.0], "Low": [9.0],
                 "Close": [10.5], "Volume": [1000]},
                index=pd.to_datetime(["2026-09-14"]),
            )

    monkeypatch.setattr(yahoo_module.yf, "Ticker", FakeTicker)
    provider = YahooDataProvider(batch_size=100, max_workers=2, max_retries=1)

    data, failed = provider.download_batch_stocks(
        ["BAD", "OTHER", "LATER1", "LATER2"], period="2y", interval="1d"
    )
    assert failed == ["BAD"]
    assert set(data) == {"OTHER", "LATER1", "LATER2"}
    assert calls.count("BAD") == 1  # no needless retry of deterministic bad data
    assert {"LATER1", "LATER2"} <= set(calls)
    assert provider.validation_failure_symbols == {"BAD"}
    assert "null/non-numeric OHLCV" in provider.failure_reasons["BAD"]

def test_yahoo_non_validation_failure_remains_fail_closed():
    import DataStore

    valid = pd.DataFrame(
        {"Open": [10.0], "High": [11.0], "Low": [9.0],
         "Close": [10.5], "Volume": [1000]},
        index=pd.to_datetime(["2026-09-14"]),
    )
    downloaded = {symbol: valid.copy() for symbol in ("^GSPC", "^IXIC", "^DJI", "MSFT")}
    with pytest.raises(DataIntegrityError, match="non-validation download failures"):
        DataStore.filter_downloaded_stock_data(
            downloaded, failed=["RATE_LIMITED"],
            expected_symbols=[*downloaded, "RATE_LIMITED"], interval="1d",
            provider_name="Yahoo", failure_reasons={"RATE_LIMITED": "HTTP 429"},
            allowed_failed_symbols=set(),
        )


def test_yahoo_filtered_symbols_only_published_with_reason(tmp_path, monkeypatch, capsys):
    import DataStore

    valid = pd.DataFrame(
        {"Open": [10.0], "High": [11.0], "Low": [9.0],
         "Close": [10.5], "Volume": [1000]},
        index=pd.to_datetime(["2026-09-14"]),
    )
    downloaded = {symbol: valid.copy() for symbol in ("^GSPC", "^IXIC", "^DJI", "MSFT")}
    result, excluded = DataStore.filter_downloaded_stock_data(
        downloaded, failed=["BAD"], expected_symbols=[*downloaded, "BAD"],
        interval="1d", provider_name="Yahoo",
        failure_reasons={"BAD": "DataIntegrityError: null OHLCV"},
        allowed_failed_symbols={"BAD"},
    )
    assert excluded == ["BAD"]
    assert "BAD: filtered; DataIntegrityError: null OHLCV" in capsys.readouterr().out
    path = tmp_path / "filtered.pkl"
    monkeypatch.setattr(DataStore, "get_stock_pkl_path", lambda interval: str(path))
    assert DataStore.save_stock_data(
        result, save_dir=str(tmp_path), interval="1d",
        expected_symbols=list(result),
    ) == str(path)
    assert set(DataStore.load_stock_data(str(path))) == set(downloaded)


def test_yahoo_filter_never_drops_required_market_index():
    import DataStore

    valid = pd.DataFrame(
        {"Open": [10.0], "High": [11.0], "Low": [9.0],
         "Close": [10.5], "Volume": [1000]},
        index=pd.to_datetime(["2026-09-14"]),
    )
    downloaded = {symbol: valid.copy() for symbol in ("^GSPC", "^IXIC", "MSFT")}
    with pytest.raises(DataIntegrityError, match="market reference missing or invalid"):
        DataStore.filter_downloaded_stock_data(
            downloaded, failed=["^DJI"],
            expected_symbols=[*downloaded, "^DJI"], interval="1d",
            provider_name="Yahoo", allowed_failed_symbols={"^DJI"},
        )

def test_no_equity_survivors_fail_closed_even_if_indices_are_valid():
    import DataStore

    valid = pd.DataFrame(
        {"Open": [10.0], "High": [11.0], "Low": [9.0],
         "Close": [10.5], "Volume": [1000]},
        index=pd.to_datetime(["2026-09-14"]),
    )
    downloaded = {symbol: valid.copy() for symbol in ("^GSPC", "^IXIC", "^DJI")}
    with pytest.raises(DataIntegrityError, match="no valid equity symbols"):
        DataStore.filter_downloaded_stock_data(
            downloaded, failed=["BAD"], expected_symbols=[*downloaded, "BAD"],
            interval="1d", provider_name="Yahoo", allowed_failed_symbols={"BAD"},
        )


def test_yahoo_recovered_symbol_clears_failure_reason_and_quarantine(monkeypatch):
    outcomes = iter([_invalid_frame(), pd.DataFrame(
        {"Open": [10.0], "High": [11.0], "Low": [9.0],
         "Close": [10.5], "Volume": [1000]},
        index=pd.to_datetime(["2026-09-14"]),
    )])

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            return next(outcomes).copy()

    monkeypatch.setattr(yahoo_module.yf, "Ticker", FakeTicker)
    provider = YahooDataProvider(max_retries=0)

    assert provider.download_single_stock("AAPL")[1] is None
    assert provider.validation_failure_symbols == {"AAPL"}
    assert provider.download_single_stock("AAPL")[1] is not None
    assert provider.validation_failure_symbols == set()
    assert provider.failure_reasons == {}


@pytest.mark.parametrize("interval", ["1d", "1wk"])
def test_yahoo_empty_histories_are_filtered_after_retries(monkeypatch, interval):
    import DataStore

    valid = pd.DataFrame(
        {"Open": [10.0], "High": [11.0], "Low": [9.0],
         "Close": [10.5], "Volume": [1000]},
        index=pd.to_datetime(["2026-09-14"]),
    )
    calls = {}

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            calls[self.symbol] = calls.get(self.symbol, 0) + 1
            if self.symbol == "EMPTY":
                return valid.iloc[:0].copy()
            return valid.copy()

    monkeypatch.setattr(yahoo_module.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(yahoo_module.time, "sleep", lambda _: None)
    provider = YahooDataProvider(max_workers=1, max_retries=1, batch_size=2)
    symbols = ["^GSPC", "^IXIC", "^DJI", "EMPTY", "MSFT"]
    data, failed = provider.download_batch_stocks(symbols, period="2y", interval=interval)

    assert failed == ["EMPTY"]
    assert calls["EMPTY"] == 4  # original per-symbol attempts + batch recovery
    assert provider.validation_failure_symbols == {"EMPTY"}
    assert "empty price history" in provider.failure_reasons["EMPTY"]
    filtered, excluded = DataStore.filter_downloaded_stock_data(
        data, failed=failed, expected_symbols=symbols, interval=interval,
        provider_name="Yahoo", failure_reasons=provider.failure_reasons,
        allowed_failed_symbols=provider.validation_failure_symbols,
    )
    assert set(filtered) == set(symbols) - {"EMPTY"}
    assert excluded == ["EMPTY"]


def test_yahoo_empty_after_transport_error_does_not_mask_outage(monkeypatch):
    import DataStore

    valid = pd.DataFrame(
        {"Open": [10.0], "High": [11.0], "Low": [9.0],
         "Close": [10.5], "Volume": [1000]},
        index=pd.to_datetime(["2026-09-14"]),
    )
    calls = {}

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            calls[self.symbol] = calls.get(self.symbol, 0) + 1
            if self.symbol == "BROKEN":
                if calls[self.symbol] == 1:
                    raise TimeoutError("upstream timeout")
                return valid.iloc[:0].copy()
            return valid.copy()

    monkeypatch.setattr(yahoo_module.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(yahoo_module.time, "sleep", lambda _: None)
    provider = YahooDataProvider(max_workers=1, max_retries=1, batch_size=5)
    symbols = ["^GSPC", "^IXIC", "^DJI", "BROKEN", "MSFT"]
    data, failed = provider.download_batch_stocks(symbols)
    assert failed == ["BROKEN"]
    assert "BROKEN" not in provider.validation_failure_symbols
    with pytest.raises(DataIntegrityError, match="non-validation download failures"):
        DataStore.filter_downloaded_stock_data(
            data, failed=failed, expected_symbols=symbols, interval="1d",
            provider_name="Yahoo", failure_reasons=provider.failure_reasons,
            allowed_failed_symbols=provider.validation_failure_symbols,
        )


def test_yahoo_empty_history_can_recover_in_original_batch_retry(monkeypatch):
    import DataStore

    valid = pd.DataFrame(
        {"Open": [10.0], "High": [11.0], "Low": [9.0],
         "Close": [10.5], "Volume": [1000]},
        index=pd.to_datetime(["2026-09-14"]),
    )
    attempts = {}

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            attempts[self.symbol] = attempts.get(self.symbol, 0) + 1
            if self.symbol == "RECOVER" and attempts[self.symbol] <= 2:
                return valid.iloc[:0].copy()
            return valid.copy()

    monkeypatch.setattr(yahoo_module.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(yahoo_module.time, "sleep", lambda _: None)
    provider = YahooDataProvider(max_workers=1, max_retries=1, batch_size=5)
    symbols = ["^GSPC", "^IXIC", "^DJI", "RECOVER", "MSFT"]
    data, failed = provider.download_batch_stocks(symbols)
    assert failed == []
    assert set(data) == set(symbols)
    assert attempts["RECOVER"] == 3
    assert provider.validation_failure_symbols == set()
    assert provider.failure_reasons == {}
    filtered, excluded = DataStore.filter_downloaded_stock_data(
        data, failed=failed, expected_symbols=symbols, interval="1d",
        provider_name="Yahoo", allowed_failed_symbols=provider.validation_failure_symbols,
    )
    assert set(filtered) == set(symbols) and not excluded


def test_yahoo_majority_empty_histories_cannot_publish_truncated_universe(monkeypatch):
    valid = pd.DataFrame(
        {"Open": [10.0], "High": [11.0], "Low": [9.0],
         "Close": [10.5], "Volume": [1000]},
        index=pd.to_datetime(["2026-09-14"]),
    )

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            if self.symbol.startswith("EMPTY"):
                return valid.iloc[:0].copy()
            return valid.copy()

    monkeypatch.setattr(yahoo_module.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(yahoo_module.time, "sleep", lambda _: None)
    provider = YahooDataProvider(max_workers=1, max_retries=0, batch_size=3)
    with pytest.raises(DataIntegrityError, match="suspected systemic data-quality outage"):
        provider.download_batch_stocks(
            ["^GSPC", "^IXIC", "^DJI", "MSFT", "EMPTY1", "EMPTY2", "EMPTY3"],
        )
    assert provider.validation_failure_symbols == {"EMPTY1", "EMPTY2", "EMPTY3"}


def test_yahoo_majority_malformed_ohlcv_cannot_silently_publish(monkeypatch):
    valid = pd.DataFrame(
        {"Open": [10.0], "High": [11.0], "Low": [9.0],
         "Close": [10.5], "Volume": [1000]},
        index=pd.to_datetime(["2026-09-14"]),
    )

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            data = valid.copy()
            if self.symbol.startswith("BAD"):
                data.loc[:, "Close"] = float("nan")
            return data

    monkeypatch.setattr(yahoo_module.yf, "Ticker", FakeTicker)
    provider = YahooDataProvider(max_workers=2, max_retries=0, batch_size=3)
    with pytest.raises(DataIntegrityError, match="suspected systemic data-quality outage"):
        provider.download_batch_stocks(
            ["^GSPC", "^IXIC", "^DJI", "MSFT", "BAD1", "BAD2", "BAD3"]
        )
    assert provider.validation_failure_symbols == {"BAD1", "BAD2", "BAD3"}
