import pandas as pd
from yfinance.exceptions import YFRateLimitError

import data_providers.yahoo_provider as yahoo_module
from data_providers.yahoo_provider import YahooDataProvider


def _sample_frame():
    return pd.DataFrame(
        {
            "Open": [10.0],
            "High": [11.0],
            "Low": [9.0],
            "Close": [10.5],
            "Volume": [1000],
        },
        index=pd.to_datetime(["2026-09-14"]),
    )


def _install_fake_clock(monkeypatch):
    now = [0.0]
    sleeps = []

    def fake_monotonic():
        return now[0]

    def fake_sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(yahoo_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(yahoo_module.time, "sleep", fake_sleep)
    return now, sleeps


def test_rate_limit_uses_exponential_backoff_before_circuit(monkeypatch):
    _, sleeps = _install_fake_clock(monkeypatch)
    outcomes = iter([YFRateLimitError(), YFRateLimitError(), _sample_frame()])

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome.copy()

    monkeypatch.setattr(yahoo_module.yf, "Ticker", FakeTicker)

    provider = YahooDataProvider(
        max_retries=2,
        rate_limit_threshold=99,
        rate_limit_backoff_base_seconds=2,
        rate_limit_backoff_max_seconds=30,
    )
    symbol, data = provider.download_single_stock("AAPL")

    assert symbol == "AAPL"
    assert data is not None
    assert sleeps == [2, 4]


def test_repeated_rate_limits_trip_shared_cooldown(monkeypatch):
    _, sleeps = _install_fake_clock(monkeypatch)
    outcomes = iter([YFRateLimitError(), YFRateLimitError(), _sample_frame()])

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome.copy()

    monkeypatch.setattr(yahoo_module.yf, "Ticker", FakeTicker)

    provider = YahooDataProvider(
        max_retries=0,
        rate_limit_threshold=2,
        rate_limit_window_seconds=10,
        rate_limit_cooldown_seconds=90,
        recovery_max_workers=2,
    )

    assert provider.download_single_stock("AAA")[1] is None
    assert provider.download_single_stock("BBB")[1] is None

    symbol, data = provider.download_single_stock("CCC")

    assert symbol == "CCC"
    assert data is not None
    assert provider._recovery_mode is True
    assert sleeps == [90]


def test_batch_retry_keeps_normal_workers_and_uses_low_concurrency(monkeypatch):
    provider = YahooDataProvider(
        batch_size=2,
        max_workers=8,
        max_retries=0,
        recovery_max_workers=2,
    )
    workers = []
    responses = iter(
        [
            ({"AAA": _sample_frame()}, ["BBB"]),
            ({"BBB": _sample_frame()}, []),
        ]
    )

    def fake_parallel(symbols, period, interval, *, max_workers):
        workers.append(max_workers)
        return next(responses)

    monkeypatch.setattr(provider, "_run_parallel_downloads", fake_parallel)
    monkeypatch.setattr(
        provider,
        "_retry_stale_latest_bars",
        lambda all_data, period, interval: [],
    )

    all_data, failed = provider.download_batch_stocks(["AAA", "BBB"])

    assert set(all_data) == {"AAA", "BBB"}
    assert failed == []
    assert workers == [8, 2]
