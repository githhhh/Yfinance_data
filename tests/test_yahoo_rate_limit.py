import threading
import time

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
        recovery_max_workers=2,
    )

    assert provider.download_single_stock("AAA")[1] is None
    assert provider.download_single_stock("BBB")[1] is None

    symbol, data = provider.download_single_stock("CCC")

    assert symbol == "CCC"
    assert data is not None
    assert provider._recovery_mode is True
    assert sleeps == [180]


def test_circuit_trip_drains_inflight_and_blocks_new_admission():
    provider = YahooDataProvider(
        max_retries=0,
        rate_limit_threshold=1,
        rate_limit_window_seconds=10,
        rate_limit_cooldown_seconds=0.05,
        recovery_max_workers=2,
    )

    first_entered = threading.Event()
    release_first = threading.Event()
    queued_entered = threading.Event()

    def hold_request():
        with provider._request_guard():
            first_entered.set()
            assert release_first.wait(1.0)

    def trip_circuit():
        provider._record_rate_limit()

    def queued_request():
        with provider._request_guard():
            queued_entered.set()

    first_thread = threading.Thread(target=hold_request)
    first_thread.start()
    assert first_entered.wait(1.0)

    trip_thread = threading.Thread(target=trip_circuit)
    trip_thread.start()

    deadline = time.monotonic() + 1.0
    while provider._cooldown_remaining() <= 0 and time.monotonic() < deadline:
        time.sleep(0.001)

    assert provider._cooldown_remaining() > 0
    assert trip_thread.is_alive()

    queued_thread = threading.Thread(target=queued_request)
    queued_thread.start()
    time.sleep(0.01)
    assert not queued_entered.is_set()

    release_first.set()
    first_thread.join(1.0)
    trip_thread.join(1.0)
    queued_thread.join(1.0)

    assert not first_thread.is_alive()
    assert not trip_thread.is_alive()
    assert not queued_thread.is_alive()
    assert queued_entered.is_set()


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