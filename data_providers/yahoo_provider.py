import time
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_providers.base_provider import BaseDataProvider

BATCH_SIZE = 100
MAX_WORKERS = 8
MAX_RETRIES = 1


class YahooDataProvider(BaseDataProvider):
    """基于 yfinance 的雅虎数据提供者，100% 保持现有重试与线程池下载逻辑。"""

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
        """抓取单只股票数据（带重试与指数退避）。"""
        attempt = 0
        while attempt <= self.max_retries:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    rounding=True,
                    timeout=5,
                )
                if not data.empty:
                    # 确保关键列存在且精度为 round(2)
                    req_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
                    data = data[req_cols].round(2)
                    return symbol, data
            except Exception as e:
                print(f"[Yahoo] Error downloading {symbol} (attempt {attempt+1}): {e}")
            attempt += 1
            time.sleep(0.5 * attempt)
        return symbol, None

    def download_batch_stocks(
        self, symbols: List[str], period: str = "1y", interval: str = "1d"
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """多线程分批抓取股票数据，逻辑与现有 DataStore 保持完全一致。"""
        all_data: Dict[str, pd.DataFrame] = {}
        failed: List[str] = []
        total = len(symbols)
        print(
            f"[Yahoo Batch] Starting download for {total} stocks, batch size {self.batch_size}, workers {self.max_workers}"
        )
        overall_start = time.time()

        for batch_start in range(0, total, self.batch_size):
            batch = symbols[batch_start : batch_start + self.batch_size]
            print(
                f"[Yahoo Batch] Processing batch {batch_start // self.batch_size + 1}: {len(batch)} stocks"
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
                f"[Yahoo Batch] Batch finished: Downloaded {batch_success}, Failed {batch_failed} "
                f"(Time: {batch_end_time - batch_start_time:.2f}s)"
            )

        # 重试失败的标的
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
                f"Recovered {len(failed) - len(retry_failed)}, Still failed {len(retry_failed)} "
                f"(Time: {retry_end_time - retry_start_time:.2f}s)"
            )
            failed = retry_failed

        overall_end = time.time()
        print(
            f"[Yahoo Batch] Finished: {len(all_data)} downloaded, {len(failed)} failed. "
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
