import os
import time
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_providers.base_provider import BaseDataProvider


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


class SchwabDataProvider(BaseDataProvider):
    """基于 schwab-py 库实现的嘉信理财 (Charles Schwab) 数据提供者。"""

    def __init__(
        self,
        creds: Optional[SchwabCredentials] = None,
        client: Optional[Any] = None,
        batch_size: int = 50,
        max_workers: int = 4,
    ):
        self.creds = creds or SchwabCredentials()
        self.batch_size = batch_size
        self.max_workers = max_workers
        self._client = client

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
            # schwab-py 客户端加载
            return schwab.auth.client_from_token_file(
                token_path=self.creds.token_path,
                api_key=self.creds.app_key,
                app_secret=self.creds.app_secret,
            )
        except Exception as e:
            raise RuntimeError(f"初始化 Schwab API 客户端失败: {e}")

    def download_single_stock(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> Tuple[str, Optional[pd.DataFrame]]:
        """抓取单只标的 K 线历史数据并清洗对齐 Schema。"""
        try:
            # 兼容标准 yfinance 的符号格式 (例如 BRK-B 替换为 BRK.B 适配 Schwab)
            schwab_symbol = symbol.replace("-", ".")
            resp = self._request_price_history(schwab_symbol, period=period, interval=interval)
            
            if resp is None:
                return symbol, None
                
            data_json = resp.json() if hasattr(resp, "json") and callable(resp.json) else resp
            
            if not isinstance(data_json, dict) or data_json.get("empty", False):
                return symbol, None

            candles = data_json.get("candles", [])
            if not candles:
                return symbol, None

            df = pd.DataFrame(candles)
            
            # 列名重命名映射，强制与 yfinance / DataStore 输出 Schema 对齐
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
            for col in req_cols:
                if col not in df.columns:
                    df[col] = 0.0

            df = df[req_cols].copy()
            
            # 转换数值类型并强制 round(2)
            for col in req_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            df = df.round(2)
            df = df.dropna(how="all")

            if df.empty:
                return symbol, None

            return symbol, df

        except Exception as e:
            print(f"[Schwab] Error downloading {symbol}: {e}")
            return symbol, None

    def _request_price_history(self, symbol: str, period: str, interval: str) -> Any:
        """调用 Schwab Client 获取价格历史 response。"""
        try:
            import schwab
            # 判断频次类型
            if interval == "1wk":
                freq_type = schwab.client.Client.PriceHistory.FrequencyType.WEEKLY
                freq = schwab.client.Client.PriceHistory.Frequency.EVERY_WEEK
            else:
                freq_type = schwab.client.Client.PriceHistory.FrequencyType.DAILY
                freq = schwab.client.Client.PriceHistory.Frequency.DAILY

            # 判断周期
            period_type = schwab.client.Client.PriceHistory.PeriodType.YEAR
            period_num = 1
            if period.endswith("y"):
                try:
                    period_num = int(period[:-1])
                except ValueError:
                    period_num = 1

            resp = self.client.get_price_history(
                symbol,
                period_type=period_type,
                period=period_num,
                frequency_type=freq_type,
                frequency=freq,
            )
            return resp
        except AttributeError:
            # Mock 环境或无 Client 属性时回退
            if hasattr(self.client, "get_price_history"):
                return self.client.get_price_history(symbol)
            return None
        except Exception as e:
            print(f"[Schwab] API Request Error for {symbol}: {e}")
            return None

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

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(
                    self.download_single_stock, symbol, period, interval
                ): symbol
                for symbol in symbols
            }
            for future in as_completed(future_to_ticker):
                stock_code, data = future.result()
                if data is not None and not data.empty:
                    all_data[stock_code] = data
                else:
                    failed.append(stock_code)

        elapsed = time.time() - start_time
        print(
            f"[Schwab Batch] Download complete. Success: {len(all_data)}, Failed: {len(failed)} (Time: {elapsed:.2f}s)"
        )
        return all_data, failed

    def fetch_quote(self, symbol: str) -> Optional[Dict]:
        """获取交易日盘中实时行情快照 (REST /marketdata/v1/quotes API)。"""
        try:
            schwab_symbol = symbol.replace("-", ".")
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
            schwab_symbol = symbol.replace("-", ".")
            resp = self.client.get_option_chain(schwab_symbol)
            data = resp.json() if hasattr(resp, "json") and callable(resp.json) else resp
            return data
        except Exception as e:
            print(f"[Schwab] fetch_option_chain failed for {symbol}: {e}")
            return None
