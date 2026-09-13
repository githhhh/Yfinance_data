import time
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_providers.base_provider import BaseDataProvider
from data_providers.ohlcv_validation import (
    DataIntegrityError,
    MARKET_REFERENCE_SYMBOLS,
    expected_latest_us_session,
    find_latest_bar_mismatches,
    reference_latest_dates,
    validate_ohlcv_frame,
)

BATCH_SIZE = 100
MAX_WORKERS = 8
MAX_RETRIES = 1


class YahooDataProvider(BaseDataProvider):
    """基于 yfinance 的雅虎数据提供者，严格拒绝缺失或损坏的 OHLCV 数据。"""

    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        max_workers: int = MAX_WORKERS,
        max_retries: int = MAX_RETRIES,
    ):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_retries = max_retries

    def download_single_stock(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> Tuple[str, Optional[pd.DataFrame]]:
        """抓取单只股票数据；任何 OHLCV 缺失/异常均按下载失败重试。"""
        attempt = 0
        while attempt <= self.max_retries:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    period=period,
                    interval=interval,
                    # Raw Yahoo OHLC: split-adjusted, not dividend-adjusted.
                    # Keep yfinance source precision for downstream comparisons.
                    auto_adjust=False,
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
                time.sleep(0.5 * attempt)
        return symbol, None

    def _redownload_symbols(
        self,
        symbols: Sequence[str],
        all_data: Dict[str, pd.DataFrame],
        period: str,
        interval: str,
    ) -> None:
        if not symbols:
            return
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(
                    self.download_single_stock, ticker, period, interval
                ): ticker
                for ticker in symbols
            }
            for future in as_completed(future_to_ticker):
                stock_code, data = future.result()
                if data is not None:
                    all_data[stock_code] = data

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
        """多线程抓取并严格校验；缺失/异常/落后最新市场交易日均进入 failed。"""
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
            batch_success = 0
            batch_failed = 0

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_ticker = {
                    executor.submit(
                        self.download_single_stock, ticker, period, interval
                    ): ticker
                    for ticker in batch
                }
                for future in as_completed(future_to_ticker):
                    stock_code, data = future.result()
                    if data is not None:
                        all_data[stock_code] = data
                        batch_success += 1
                    else:
                        failed.append(stock_code)
                        batch_failed += 1

            batch_end_time = time.time()
            print(
                f"[Yahoo Batch] Batch finished: Downloaded {batch_success}, "
                f"Failed {batch_failed} "
                f"(Time: {batch_end_time - batch_start_time:.2f}s)"
            )

        # 重试完整失败的标的。download_single_stock 本身也有内部重试，这里保留
        # 原有 batch-level recovery，避免瞬时 Yahoo 故障直接污染发布流程。
        if failed:
            print(f"[Yahoo Batch] Retrying {len(failed)} failed stocks...")
            retry_failed: List[str] = []
            retry_start_time = time.time()
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_ticker = {
                    executor.submit(
                        self.download_single_stock, ticker, period, interval
                    ): ticker
                    for ticker in failed
                }
                for future in as_completed(future_to_ticker):
                    stock_code, data = future.result()
                    if data is not None:
                        all_data[stock_code] = data
                    else:
                        retry_failed.append(stock_code)
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
