import json
import os
import threading
import time
import pandas as pd
import requests
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_providers.base_provider import BaseDataProvider
from data_providers.ohlcv_validation import validate_ohlcv_frame


SCHWAB_API_BASE = "https://api.schwabapi.com"
SCHWAB_TOKEN_URL = f"{SCHWAB_API_BASE}/v1/oauth/token"
# The rest of the pipeline deliberately keeps Yahoo-compatible index keys.  The
# Schwab Trader API uses its own index symbols on the request boundary only.
SCHWAB_SYMBOL_ALIASES = {
    "^GSPC": "$SPX",
    "^IXIC": "$COMPX",
    "^DJI": "$DJI",
}


def _enum_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    return value


def _schwab_symbol(symbol: str) -> str:
    """Translate only provider-specific symbols; retain the caller's key."""
    normalized = str(symbol).strip().upper()
    # Schwab represents dual-class tickers with a slash (for example MOG/A),
    # while the repository's Yahoo-compatible keys use a hyphen (MOG-A).
    return SCHWAB_SYMBOL_ALIASES.get(normalized, normalized.replace("-", "/"))


class SchwabCredentials:
    """嘉信 API 凭证配置类 (支持参数显式指定或读取环境变量)。"""

    def __init__(
        self,
        app_key: Optional[str] = None,
        app_secret: Optional[str] = None,
        callback_url: Optional[str] = None,
        token_path: Optional[str] = None,
    ):
        self.app_key = app_key or os.environ.get("SCHWAB_APP_KEY", "")
        self.app_secret = app_secret or os.environ.get("SCHWAB_APP_SECRET", "")
        self.callback_url = (
            callback_url
            or os.environ.get("SCHWAB_CALLBACK_URL", "https://127.0.0.1")
        )
        self.token_path = (
            token_path or os.environ.get("SCHWAB_TOKEN_PATH", "token.json")
        )

    def is_valid(self) -> bool:
        """检查凭证及 Token 文件配置是否基本有效。"""
        return bool(self.app_key and self.app_secret) or os.path.exists(
            self.token_path
        )


class SchwabRawTokenClient:
    """Minimal Schwab REST client for OAuth token files created outside schwab-py."""

    def __init__(self, creds: SchwabCredentials, timeout: int = 30):
        self.creds = creds
        self.timeout = timeout

    def has_access_token(self) -> bool:
        return bool(self._load_token().get("access_token"))

    def get_quote(self, symbol: str) -> Dict:
        response = self._get(
            f"{SCHWAB_API_BASE}/marketdata/v1/{symbol}/quotes",
            params=None,
        )
        return response.json()

    def get_option_chain(self, symbol: str) -> Dict:
        response = self._get(
            f"{SCHWAB_API_BASE}/marketdata/v1/chains",
            params={"symbol": symbol},
        )
        return response.json()

    def get_price_history(
        self,
        symbol: str,
        period_type: Any = "year",
        period: Any = 1,
        frequency_type: Any = "daily",
        frequency: Any = 1,
        **_: Any,
    ) -> requests.Response:
        return self._get(
            f"{SCHWAB_API_BASE}/marketdata/v1/pricehistory",
            params={
                "symbol": symbol,
                "periodType": _enum_value(period_type),
                "period": _enum_value(period),
                "frequencyType": _enum_value(frequency_type),
                "frequency": _enum_value(frequency),
                "needExtendedHoursData": "false",
            },
        )

    def _get(self, url: str, params: Optional[Dict[str, Any]]) -> requests.Response:
        response = requests.get(
            url,
            headers=self._headers(),
            params=params,
            timeout=self.timeout,
        )
        if response.status_code == 401 and self._refresh_access_token():
            response = requests.get(
                url,
                headers=self._headers(),
                params=params,
                timeout=self.timeout,
            )
        response.raise_for_status()
        return response

    def _headers(self) -> Dict[str, str]:
        token = self._load_token().get("access_token", "")
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

    def _refresh_access_token(self) -> bool:
        token = self._load_token()
        refresh_token = token.get("refresh_token")
        if not refresh_token or not self.creds.app_key or not self.creds.app_secret:
            return False
        response = requests.post(
            SCHWAB_TOKEN_URL,
            auth=(self.creds.app_key, self.creds.app_secret),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "refresh_token", "refresh_token": refresh_token},
            timeout=self.timeout,
        )
        if not response.ok:
            return False
        updated = token.copy()
        updated.update(response.json())
        if "refresh_token" not in updated and refresh_token:
            updated["refresh_token"] = refresh_token
        self._save_token(updated)
        return bool(updated.get("access_token"))

    def _load_token(self) -> Dict[str, Any]:
        try:
            with open(self.creds.token_path, "r", encoding="utf-8") as f:
                token = json.load(f)
        except Exception:
            return {}
        return token if isinstance(token, dict) else {}

    def _save_token(self, token: Dict[str, Any]) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(self.creds.token_path, flags, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(token, f, indent=2)


class SchwabDataProvider(BaseDataProvider):
    """Schwab raw OHLCV provider: split-adjusted, not dividend-adjusted.

    The provider preserves vendor numeric precision and emits the canonical
    ``Open, High, Low, Close, Volume`` schema used by the existing PKL pipeline.
    Request starts are paced globally while bounded workers overlap network
    latency. Failed symbols get separate recovery rounds after the bulk pass.
    """

    def __init__(
        self,
        creds: Optional[SchwabCredentials] = None,
        client: Optional[Any] = None,
        batch_size: int = 100,
        max_workers: int = 8,
        max_retries: int = 1,
        rate_limit_sleep: float = 0.55,
        recovery_rounds: int = 2,
        recovery_sleep: float = 3.0,
    ):
        self.creds = creds or SchwabCredentials()
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.rate_limit_sleep = rate_limit_sleep
        self.recovery_rounds = recovery_rounds
        self.recovery_sleep = recovery_sleep
        self._client = client
        self._request_lock = threading.Lock()
        self._next_request_time = 0.0
        self._last_failure_reasons: Dict[str, str] = {}

    def _wait_for_request_slot(self) -> None:
        """Pace all workers, including retries, by request start time."""
        with self._request_lock:
            wait = self._next_request_time - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._next_request_time = time.monotonic() + self.rate_limit_sleep

    def _cooldown_after_rate_limit(self) -> None:
        with self._request_lock:
            self._next_request_time = max(
                self._next_request_time, time.monotonic() + 30.0
            )

    def _record_failure(self, symbol: str, reason: str) -> None:
        self._last_failure_reasons[symbol] = reason
        print(f"[Schwab] {symbol}: {reason}")

    @property
    def failure_reasons(self) -> Dict[str, str]:
        return dict(self._last_failure_reasons)

    @property
    def client(self) -> Any:
        """延迟加载 Schwab API Client 客户端。"""
        if self._client is None:
            self._client = self._init_client()
        return self._client

    def _init_client(self) -> Any:
        """使用 schwab-py 根据 token.json 与 API 凭证构建客户端。"""
        try:
            import schwab
        except ImportError:
            raise RuntimeError(
                "未在 Python 环境中找到 schwab-py 模块。请通过 `pip install schwab-py` 安装依赖。"
            )

        if not os.path.exists(self.creds.token_path):
            raise FileNotFoundError(
                f"嘉信 OAuth 授权 Token 文件不存在: {self.creds.token_path}。"
                "请先运行授权脚本生成 token.json，或配置文件路径。"
            )

        try:
            return schwab.auth.client_from_token_file(
                token_path=self.creds.token_path,
                api_key=self.creds.app_key,
                app_secret=self.creds.app_secret,
            )
        except Exception as e:
            raw_client = SchwabRawTokenClient(self.creds)
            if raw_client.has_access_token():
                return raw_client
            raise RuntimeError(f"初始化 Schwab API 客户端失败: {e}")

    def download_single_stock(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> Tuple[str, Optional[pd.DataFrame]]:
        """抓取单只标的 K 线历史数据并清洗对齐 Schema。"""
        attempt = 0
        while attempt <= self.max_retries:
            data = self._download_single_stock_once(symbol, period=period, interval=interval)
            if data is not None:
                self._last_failure_reasons.pop(symbol, None)
                return symbol, data
            attempt += 1
            if attempt <= self.max_retries:
                time.sleep(self.rate_limit_sleep * attempt)
        return symbol, None

    def _download_single_stock_once(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        try:
            self._wait_for_request_slot()
            schwab_symbol = _schwab_symbol(symbol)
            resp = self._request_price_history(schwab_symbol, period=period, interval=interval)

            if resp is None:
                self._record_failure(symbol, "empty API response")
                return None

            status_code = getattr(resp, "status_code", None)
            if isinstance(status_code, int) and status_code >= 400:
                if status_code == 429:
                    self._cooldown_after_rate_limit()
                self._record_failure(symbol, f"HTTP {status_code}")
                return None

            data_json = resp.json() if hasattr(resp, "json") and callable(resp.json) else resp

            if not isinstance(data_json, dict) or data_json.get("empty", False):
                self._record_failure(symbol, "empty or invalid history response")
                return None

            candles = data_json.get("candles", [])
            if not candles:
                self._record_failure(symbol, "no history candles")
                return None

            df = pd.DataFrame(candles)

            col_map = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
            df = df.rename(columns=col_map)

            if "datetime" in df.columns:
                df["Date"] = pd.to_datetime(df["datetime"], unit="ms", errors="coerce")
                df = df.set_index("Date")

            req_cols = ["Open", "High", "Low", "Close", "Volume"]
            if any(col not in df.columns for col in req_cols):
                self._record_failure(symbol, "missing OHLCV columns")
                return None

            df = df[req_cols].copy()

            for col in req_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(how="all")

            if df.empty:
                self._record_failure(symbol, "empty OHLCV rows")
                return None

            validate_ohlcv_frame(symbol, df)
            return df

        except Exception as e:
            if getattr(getattr(e, "response", None), "status_code", None) == 429:
                self._cooldown_after_rate_limit()
            self._record_failure(symbol, f"{type(e).__name__}: {e}")
            return None

    def _request_price_history(self, symbol: str, period: str, interval: str) -> Any:
        """Request the exact history shape needed by the existing PKL pipeline."""
        period_num = 1
        if period.endswith("y"):
            try:
                period_num = int(period[:-1])
            except ValueError:
                period_num = 1

        try:
            import schwab
        except ImportError:
            if not hasattr(self.client, "get_price_history"):
                return None
            frequency_type = "weekly" if interval == "1wk" else "daily"
            return self.client.get_price_history(
                symbol,
                period_type="year",
                period=period_num,
                frequency_type=frequency_type,
                frequency=1,
            )

        price_history = schwab.client.Client.PriceHistory
        if interval == "1wk":
            freq_type = price_history.FrequencyType.WEEKLY
            freq = price_history.Frequency.WEEKLY
        elif interval == "1d":
            freq_type = price_history.FrequencyType.DAILY
            freq = price_history.Frequency.DAILY
        else:
            print(f"[Schwab] Unsupported price-history interval: {interval}")
            return None

        period_map = {
            1: price_history.Period.ONE_YEAR,
            2: price_history.Period.TWO_YEARS,
            3: price_history.Period.THREE_YEARS,
            5: price_history.Period.FIVE_YEARS,
            10: price_history.Period.TEN_YEARS,
            15: price_history.Period.FIFTEEN_YEARS,
            20: price_history.Period.TWENTY_YEARS,
        }
        period_value = period_map.get(period_num)
        if period_value is None:
            print(f"[Schwab] Unsupported yearly price-history period: {period}")
            return None

        return self.client.get_price_history(
            symbol,
            period_type=price_history.PeriodType.YEAR,
            period=period_value,
            frequency_type=freq_type,
            frequency=freq,
        )

    def _download_pass(
        self, symbols: List[str], period: str, interval: str
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        data: Dict[str, pd.DataFrame] = {}
        failed: List[str] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for batch_start in range(0, len(symbols), self.batch_size):
                batch = symbols[batch_start : batch_start + self.batch_size]
                futures = {
                    executor.submit(self.download_single_stock, symbol, period, interval): symbol
                    for symbol in batch
                }
                for future in as_completed(futures):
                    stock_code, frame = future.result()
                    if frame is not None and not frame.empty:
                        data[stock_code] = frame
                    else:
                        failed.append(stock_code)
        return data, sorted(failed)

    def download_batch_stocks(
        self, symbols: List[str], period: str = "1y", interval: str = "1d"
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """批量抓取 Schwab K 线历史数据。"""
        all_data: Dict[str, pd.DataFrame] = {}
        failed: List[str] = []
        total = len(symbols)
        print(
            f"[Schwab Batch] Downloading {total} stocks (batch size {self.batch_size}, workers {self.max_workers})..."
        )
        start_time = time.time()
        # Build the OAuth client once before workers can race through lazy init.
        _ = self.client

        all_data, failed = self._download_pass(symbols, period, interval)
        for round_number in range(1, self.recovery_rounds + 1):
            if not failed:
                break
            print(f"[Schwab Batch] Recovery round {round_number}: {len(failed)} symbols")
            if self.recovery_sleep > 0:
                time.sleep(self.recovery_sleep * round_number)
            recovered, failed = self._download_pass(failed, period, interval)
            all_data.update(recovered)

        for symbol in failed:
            print(
                f"[Schwab Batch] Failed {symbol}: "
                f"{self._last_failure_reasons.get(symbol, 'no history returned')}"
            )

        elapsed = time.time() - start_time
        print(
            f"[Schwab Batch] Download complete. Success: {len(all_data)}, Failed: {len(failed)} (Time: {elapsed:.2f}s)"
        )
        return all_data, failed

    def fetch_quote(self, symbol: str) -> Optional[Dict]:
        """获取交易日盘中实时行情快照 (REST /marketdata/v1/quotes API)。"""
        try:
            schwab_symbol = _schwab_symbol(symbol)
            resp = self.client.get_quote(schwab_symbol)
            data = resp.json() if hasattr(resp, "json") and callable(resp.json) else resp
            if isinstance(data, dict) and schwab_symbol in data:
                return data[schwab_symbol]
            return data
        except Exception as e:
            print(f"[Schwab] fetch_quote failed for {symbol}: {e}")
            return None

    def fetch_option_chain(self, symbol: str) -> Optional[Dict]:
        """获取期权链数据 (REST /marketdata/v1/chains API)。"""
        try:
            schwab_symbol = _schwab_symbol(symbol)
            resp = self.client.get_option_chain(schwab_symbol)
            data = resp.json() if hasattr(resp, "json") and callable(resp.json) else resp
            return data
        except Exception as e:
            print(f"[Schwab] fetch_option_chain failed for {symbol}: {e}")
            return None
