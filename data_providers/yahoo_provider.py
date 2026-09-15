import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from data_providers.base_provider import BaseDataProvider
from data_providers.ohlcv_validation import (
    DataIntegrityError,
    expected_latest_us_session,
    find_latest_bar_mismatches,
    reference_latest_dates,
    validate_ohlcv_frame,
)

BATCH_SIZE = 100
MAX_WORKERS = 8
MAX_RETRIES = 1

RATE_LIMIT_THRESHOLD = 3
RATE_LIMIT_WINDOW_SECONDS = 10.0
RATE_LIMIT_COOLDOWN_SECONDS = 90.0
RATE_LIMIT_BACKOFF_BASE_SECONDS = 2.0
RATE_LIMIT_BACKOFF_MAX_SECONDS = 30.0
RECOVERY_MAX_WORKERS = 2


class YahooDataProvider(BaseDataProvider):
    """基于 yfinance 的雅虎数据提供者，严格拒绝缺失或损坏的 OHLCV 数据。"""

    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        max_workers: int = MAX_WORKERS,
        max_retries: int = MAX_RETRIES,
        recovery_max_workers: int = RECOVERY_MAX_WORKERS,
        rate_limit_threshold: int = RATE_LIMIT_THRESHOLD,
        rate_limit_window_seconds: float = RATE_LIMIT_WINDOW_SECONDS,
        rate_limit_cooldown_seconds: float = RATE_LIMIT_COOLDOWN_SECONDS,
        rate_limit_backoff_base_seconds: float = RATE_LIMIT_BACKOFF_BASE_SECONDS,
        rate_limit_backoff_max_seconds: float = RATE_LIMIT_BACKOFF_MAX_SECONDS,
    ):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_retries = max_retries

        self.recovery_max_workers = max(1, min(max_workers, recovery_max_workers))
        self.rate_limit_threshold = max(1, rate_limit_threshold)
        self.rate_limit_window_seconds = max(0.0, rate_limit_window_seconds)
        self.rate_limit_cooldown_seconds = max(0.0, rate_limit_cooldown_seconds)
        self.rate_limit_backoff_base_seconds = max(
            0.0, rate_limit_backoff_base_seconds
        )
        self.rate_limit_backoff_max_seconds = max(
            self.rate_limit_backoff_base_seconds,
            rate_limit_backoff_max_seconds,
        )

        self._rate_limit_lock = threading.Lock()
        self._rate_limit_events = deque()
        self._cooldown_until = 0.0
        self._recovery_mode = False
        self._recovery_gate = threading.BoundedSemaphore(self.recovery_max_workers)

    def _cooldown_remaining(self) -> float:
        with self._rate_limit_lock:
            return max(0.0, self._cooldown_until - time.monotonic())

    def _wait_for_rate_limit_cooldown(self) -> None:
        while True:
            remaining = self._cooldown_remaining()
            if remaining <= 0:
                return
            time.sleep(remaining)

    def _request_guard(self):
        with self._rate_limit_lock:
            recovery_mode = self._recovery_mode
        return self._recovery_gate if recovery_mode else nullcontext()

    def _record_rate_limit(self) -> bool:
        """Record a Yahoo 429 burst and trip a shared cooldown when threshold is hit."""
        now = time.monotonic()
        tripped = False
        with self._rate_limit_lock:
            cutoff = now - self.rate_limit_window_seconds
            while self._rate_limit_events and self._rate_limit_events[0] < cutoff:
                self._rate_limit_events.popleft()
            self._rate_limit_events.append(now)

            if (
                len(self._rate_limit_events) >= self.rate_limit_threshold
                and self._cooldown_until <= now
            ):
                self._cooldown_until = now + self.rate_limit_cooldown_seconds
                self._recovery_mode = True
                tripped = True

        if tripped:
            print(
                "[Yahoo RateLimit] Circuit opened after repeated 429 responses; "
                f"pausing all workers for {self.rate_limit_cooldown_seconds:.0f}s "
                f"and limiting recovery concurrency to {self.recovery_max_workers}"
            )
        return tripped

    def _ensure_rate_limit_recovery_cooldown(self) -> None:
        """Before batch recovery, ensure a 429-affected run gets a real cooldown."""
        now = time.monotonic()
        with self._rate_limit_lock:
            cutoff = now - self.rate_limit_window_seconds
            while self._rate_limit_events and self._rate_limit_events[0] < cutoff:
                self._rate_limit_events.popleft()
            if not self._rate_limit_events:
                return
            if self._cooldown_until <= now:
                self._cooldown_until = now + self.rate_limit_cooldown_seconds
                self._recovery_mode = True
                should_log = True
            else:
                should_log = False

        if should_log:
            print(
                "[Yahoo RateLimit] Delaying batch recovery for "
                f"{self.rate_limit_cooldown_seconds:.0f}s before retrying "
                f"with {self.recovery_max_workers} workers"
            )
        self._wait_for_rate_limit_cooldown()

    def _rate_limit_backoff_seconds(self, retry_number: int) -> float:
        delay = self.rate_limit_backoff_base_seconds * (2 ** max(0, retry_number - 1))
        return min(delay, self.rate_limit_backoff_max_seconds)

    def download_single_stock(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> Tuple[str, Optional[pd.DataFrame]]:
        """抓取单只股票数据；429 使用共享熔断，其它失败按原策略重试。"""
        attempt = 0
        while attempt <= self.max_retries:
            self._wait_for_rate_limit_cooldown()
            try:
                with self._request_guard():
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(
                        period=period,
                        interval=interval,
                        # Raw Yahoo OHLC: split-adjusted, not dividend-adjusted.
                        # Exclude action-only rows before strict OHLCV validation.
                        auto_adjust=False,
                        actions=False,
                        keepna=False,
                        timeout=5,
                    )
                if not data.empty:
                    # 保留原始 Close，忽略 Adj Close，价格精度保持源数据原样。
                    req_cols = ["Open", "High", "Low", "Close", "Volume"]
                    if any(col not in data.columns for col in req_cols):
                        raise DataIntegrityError(
                            f"{symbol}: missing required OHLCV columns"
                        )
                    data = data[req_cols].copy()
                    validate_ohlcv_frame(symbol, data)
                    return symbol, data
                print(
                    f"[Yahoo] Empty history for {symbol} "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})"
                )
            except YFRateLimitError as e:
                print(
                    f"[Yahoo] Rate limited for {symbol} "
                    f"(attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                )
                self._record_rate_limit()
                attempt += 1
                if attempt <= self.max_retries:
                    if self._cooldown_remaining() <= 0:
                        delay = self._rate_limit_backoff_seconds(attempt)
                        print(
                            f"[Yahoo RateLimit] Backing off {delay:.1f}s "
                            f"before retrying {symbol}"
                        )
                        time.sleep(delay)
                    continue
                return symbol, None
            except DataIntegrityError as e:
                print(
                    f"[Yahoo] Invalid history for {symbol} "
                    f"(attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                )
            except Exception as e:
                print(
                    f"[Yahoo] Error downloading {symbol} "
                    f"(attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                )

            attempt += 1
            if attempt <= self.max_retries:
                # Preserve the existing retry cadence for non-rate-limit failures.
                time.sleep(0.5 * attempt)

        return symbol, None

    def _run_parallel_downloads(
        self,
        symbols: Sequence[str],
        period: str,
        interval: str,
        *,
        max_workers: int,
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        downloaded: Dict[str, pd.DataFrame] = {}
        failed: List[str] = []
        if not symbols:
            return downloaded, failed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(
                    self.download_single_stock, ticker, period, interval
                ): ticker
                for ticker in symbols
            }
            for future in as_completed(future_to_ticker):
                stock_code, data = future.result()
                if data is not None:
                    downloaded[stock_code] = data
                else:
                    failed.append(stock_code)

        return downloaded, failed

    def _redownload_symbols(
        self,
        symbols: Sequence[str],
        all_data: Dict[str, pd.DataFrame],
        period: str,
        interval: str,
    ) -> None:
        if not symbols:
            return

        # Recovery is intentionally lower-concurrency even if the normal path is 8.
        self._wait_for_rate_limit_cooldown()
        downloaded, _ = self._run_parallel_downloads(
            symbols,
            period,
            interval,
            max_workers=self.recovery_max_workers,
        )
        all_data.update(downloaded)

    def _retry_stale_latest_bars(
        self,
        all_data: Dict[str, pd.DataFrame],
        period: str,
        interval: str,
    ) -> List[str]:
        """Retry incomplete latest bars without trusting a single stale reference."""
        if interval not in {"1d", "1wk"}:
            return []

        reference_dates = reference_latest_dates(all_data)
        if not reference_dates:
            # Missing reference symbols are already represented in the normal failed
            # list; final batch validation will fail closed before PKL publication.
            return []

        if len(set(reference_dates.values())) != 1:
            newest_reference_date = max(reference_dates.values())
            lagging_references = [
                symbol
                for symbol, value in reference_dates.items()
                if value != newest_reference_date
            ]
            print(
                "[Yahoo Batch] Market references disagree; retrying only lagging "
                f"references first: {reference_dates}"
            )
            self._redownload_symbols(
                lagging_references, all_data, period=period, interval=interval
            )
            reference_dates = reference_latest_dates(all_data)
            if len(set(reference_dates.values())) != 1:
                failed_references = sorted(reference_dates)
                for symbol in failed_references:
                    all_data.pop(symbol, None)
                print(
                    "[Yahoo Batch] Market references still disagree after retry; "
                    f"failing closed: {reference_dates}"
                )
                return failed_references

        reference_date = next(iter(reference_dates.values()))

        # A consensus can still be globally stale. For daily data, compare it with
        # the latest regular US session that should already be fully closed.
        if interval == "1d":
            expected_session = expected_latest_us_session()
            if reference_date < expected_session:
                retry_symbols = list(all_data)
                print(
                    "[Yahoo Batch] Entire daily batch appears stale "
                    f"({reference_date} < {expected_session}); retrying all "
                    f"{len(retry_symbols)} symbols once"
                )
                self._redownload_symbols(
                    retry_symbols, all_data, period=period, interval=interval
                )
                reference_dates = reference_latest_dates(all_data)
                if (
                    not reference_dates
                    or len(set(reference_dates.values())) != 1
                    or next(iter(reference_dates.values())) != expected_session
                ):
                    failed_symbols = sorted(all_data)
                    all_data.clear()
                    print(
                        "[Yahoo Batch] Daily batch still does not reach latest "
                        f"completed session {expected_session}; failing closed"
                    )
                    return failed_symbols
                reference_date = expected_session

        mismatches = find_latest_bar_mismatches(all_data)
        if not mismatches:
            return []

        print(
            f"[Yahoo Batch] Retrying {len(mismatches)} symbols with stale latest bars "
            f"vs {reference_date}: {list(mismatches.items())[:10]}"
        )
        self._redownload_symbols(
            list(mismatches), all_data, period=period, interval=interval
        )

        remaining = find_latest_bar_mismatches(all_data)
        for symbol in remaining:
            all_data.pop(symbol, None)

        if remaining:
            print(
                f"[Yahoo Batch] Rejecting {len(remaining)} symbols still stale after retry: "
                f"{list(remaining.items())[:10]}"
            )
        return sorted(remaining)

    def download_batch_stocks(
        self, symbols: List[str], period: str = "1y", interval: str = "1d"
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """多线程抓取并严格校验；正常路径保持 8 并发，429 后降级恢复。"""
        all_data: Dict[str, pd.DataFrame] = {}
        failed: List[str] = []
        total = len(symbols)
        print(
            f"[Yahoo Batch] Starting download for {total} stocks, "
            f"batch size {self.batch_size}, workers {self.max_workers}"
        )
        overall_start = time.time()

        for batch_start in range(0, total, self.batch_size):
            batch = symbols[batch_start : batch_start + self.batch_size]
            print(
                f"[Yahoo Batch] Processing batch "
                f"{batch_start // self.batch_size + 1}: {len(batch)} stocks"
            )
            batch_start_time = time.time()

            downloaded, batch_failed = self._run_parallel_downloads(
                batch,
                period,
                interval,
                max_workers=self.max_workers,
            )
            all_data.update(downloaded)
            failed.extend(batch_failed)

            batch_end_time = time.time()
            print(
                f"[Yahoo Batch] Batch finished: Downloaded {len(downloaded)}, "
                f"Failed {len(batch_failed)} "
                f"(Time: {batch_end_time - batch_start_time:.2f}s)"
            )

        # Retry complete failures, but never immediately re-hit Yahoo with 8 workers.
        if failed:
            self._ensure_rate_limit_recovery_cooldown()
            print(
                f"[Yahoo Batch] Retrying {len(failed)} failed stocks "
                f"with {self.recovery_max_workers} workers..."
            )
            retry_start_time = time.time()
            retry_data, retry_failed = self._run_parallel_downloads(
                failed,
                period,
                interval,
                max_workers=self.recovery_max_workers,
            )
            all_data.update(retry_data)
            retry_end_time = time.time()
            print(
                f"[Yahoo Batch] Retry finished: "
                f"Recovered {len(failed) - len(retry_failed)}, "
                f"Still failed {len(retry_failed)} "
                f"(Time: {retry_end_time - retry_start_time:.2f}s)"
            )
            failed = retry_failed

        stale_failed = self._retry_stale_latest_bars(
            all_data, period=period, interval=interval
        )
        failed = sorted(set(failed).union(stale_failed))

        overall_end = time.time()
        print(
            f"[Yahoo Batch] Finished: {len(all_data)} downloaded, "
            f"{len(failed)} failed. "
            f"Total time: {overall_end - overall_start:.2f} seconds"
        )
        return all_data, failed

    def fetch_option_chain(self, symbol: str) -> Optional[Dict]:
        """Yahoo 期权链点查。"""
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            if not expirations:
                return None
            chain = ticker.option_chain(expirations[0])
            return {
                "symbol": symbol,
                "expiration": expirations[0],
                "calls": chain.calls,
                "puts": chain.puts,
            }
        except Exception as e:
            print(f"[Yahoo] fetch_option_chain failed for {symbol}: {e}")
            return None

    def fetch_quote(self, symbol: str) -> Optional[Dict]:
        """Yahoo 实时行情快照。"""
        try:
            ticker = yf.Ticker(symbol)
            fast_info = getattr(ticker, "fast_info", {})
            return {
                "symbol": symbol,
                "last_price": fast_info.get("lastPrice"),
                "previous_close": fast_info.get("previousClose"),
                "open": fast_info.get("open"),
                "day_high": fast_info.get("dayHigh"),
                "day_low": fast_info.get("dayLow"),
                "volume": fast_info.get("lastVolume"),
            }
        except Exception as e:
            print(f"[Yahoo] fetch_quote failed for {symbol}: {e}")
            return None
