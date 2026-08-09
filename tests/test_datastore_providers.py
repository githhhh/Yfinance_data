import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from data_providers.base_provider import BaseDataProvider
from data_providers.yahoo_provider import YahooDataProvider
from data_providers.schwab_provider import SchwabDataProvider, SchwabCredentials
from data_providers.factory import DataProviderFactory
import DataStore


class TestYahooDataProvider:
    """测试 YahooDataProvider 行情抓取、清洗与对齐契约。"""

    @patch("yfinance.Ticker")
    def test_download_single_stock_success(self, mock_ticker_cls):
        sample_df = pd.DataFrame(
            {
                "Open": [100.123, 102.456],
                "High": [105.789, 106.111],
                "Low": [99.001, 101.222],
                "Close": [103.555, 104.999],
                "Volume": [10000, 15000],
            },
            index=pd.date_range("2026-01-01", periods=2),
        )
        mock_instance = MagicMock()
        mock_instance.history.return_value = sample_df
        mock_ticker_cls.return_value = mock_instance

        provider = YahooDataProvider(max_retries=0)
        symbol, df = provider.download_single_stock("AAPL", period="1y", interval="1d")

        assert symbol == "AAPL"
        assert df is not None
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert df.loc[df.index[0], "Open"] == 100.12  # 验证 round(2)
        assert df.loc[df.index[0], "Close"] == 103.56

    @patch("yfinance.Ticker")
    def test_download_batch_stocks(self, mock_ticker_cls):
        sample_df = pd.DataFrame(
            {"Open": [10.0], "High": [11.0], "Low": [9.5], "Close": [10.5], "Volume": [1000]},
            index=pd.date_range("2026-01-01", periods=1),
        )
        mock_instance = MagicMock()
        mock_instance.history.return_value = sample_df
        mock_ticker_cls.return_value = mock_instance

        provider = YahooDataProvider(batch_size=2, max_workers=2, max_retries=0)
        all_data, failed = provider.download_batch_stocks(["AAPL", "MSFT"])

        assert len(all_data) == 2
        assert "AAPL" in all_data
        assert "MSFT" in all_data
        assert len(failed) == 0

    @patch("yfinance.Ticker")
    def test_fetch_quote_and_options(self, mock_ticker_cls):
        mock_instance = MagicMock()
        mock_instance.fast_info = {"lastPrice": 150.0, "previousClose": 148.0}
        mock_instance.options = ("2026-09-18",)
        chain_mock = MagicMock()
        chain_mock.calls = pd.DataFrame({"strike": [150]})
        chain_mock.puts = pd.DataFrame({"strike": [150]})
        mock_instance.option_chain.return_value = chain_mock
        mock_ticker_cls.return_value = mock_instance

        provider = YahooDataProvider()
        quote = provider.fetch_quote("AAPL")
        options = provider.fetch_option_chain("AAPL")

        assert quote is not None and quote["last_price"] == 150.0
        assert options is not None and options["expiration"] == "2026-09-18"


class TestSchwabDataProvider:
    """测试 SchwabDataProvider 行情抓取、数据清洗契约与错误处理。"""

    def test_schwab_credentials_validation(self):
        creds = SchwabCredentials(app_key="test_key", app_secret="test_secret")
        assert creds.is_valid() is True
        assert creds.app_key == "test_key"
        assert creds.callback_url == "https://127.0.0.1"

    def test_download_single_stock_mock_response(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "candles": [
                {
                    "open": 150.126,
                    "high": 155.888,
                    "low": 149.333,
                    "close": 154.555,
                    "volume": 5000000,
                    "datetime": 1672531200000,
                }
            ],
            "empty": False,
        }
        mock_client.get_price_history.return_value = mock_resp

        provider = SchwabDataProvider(client=mock_client)
        symbol, df = provider.download_single_stock("AAPL")

        assert symbol == "AAPL"
        assert df is not None
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert df.iloc[0]["Open"] == 150.13  # round(2)
        assert df.iloc[0]["High"] == 155.89  # round(2)
        assert df.iloc[0]["Volume"] == 5000000

    def test_download_batch_stocks(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "candles": [
                {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 100, "datetime": 1672531200000}
            ],
            "empty": False,
        }
        mock_client.get_price_history.return_value = mock_resp

        provider = SchwabDataProvider(client=mock_client, batch_size=2, max_workers=2)
        all_data, failed = provider.download_batch_stocks(["NVDA", "TSLA"])

        assert len(all_data) == 2
        assert "NVDA" in all_data
        assert "TSLA" in all_data
        assert len(failed) == 0

    def test_fetch_quote_and_options(self):
        mock_client = MagicMock()
        mock_client.get_quote.return_value = {"AAPL": {"lastPrice": 180.5}}
        mock_client.get_option_chain.return_value = {"symbol": "AAPL", "status": "SUCCESS"}

        provider = SchwabDataProvider(client=mock_client)
        quote = provider.fetch_quote("AAPL")
        options = provider.fetch_option_chain("AAPL")

        assert quote == {"lastPrice": 180.5}
        assert options == {"symbol": "AAPL", "status": "SUCCESS"}

    def test_missing_token_file_raises_error(self):
        creds = SchwabCredentials(token_path="non_existent_token.json")
        provider = SchwabDataProvider(creds=creds)
        with pytest.raises((FileNotFoundError, RuntimeError)):
            _ = provider.client


class TestDataProviderFactory:
    """测试 DataProviderFactory 工厂模式。"""

    def test_get_yahoo_provider(self):
        provider = DataProviderFactory.get_provider("yahoo")
        assert isinstance(provider, YahooDataProvider)

    def test_get_yahoo_provider_with_extra_cli_kwargs(self):
        """测试向 yahoo 提供者传入 CLI 默认凭证参数时能够自动过滤不报错。"""
        provider = DataProviderFactory.get_provider(
            "yahoo", app_key="key", app_secret="secret", token_path="token.json", callback_url=None
        )
        assert isinstance(provider, YahooDataProvider)

    def test_get_schwab_provider(self):
        mock_client = MagicMock()
        provider = DataProviderFactory.get_provider("schwab", client=mock_client)
        assert isinstance(provider, SchwabDataProvider)

    def test_invalid_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="未已知的数据源类型"):
            DataProviderFactory.get_provider("invalid_provider")



class TestLegacyBackwardCompatibility:
    """测试 DataStore 旧函数别名兼容性。"""

    @patch("yfinance.Ticker")
    def test_legacy_download_functions(self, mock_ticker_cls):
        sample_df = pd.DataFrame(
            {"Open": [50.0], "High": [52.0], "Low": [49.0], "Close": [51.0], "Volume": [500]},
            index=pd.date_range("2026-01-01", periods=1),
        )
        mock_instance = MagicMock()
        mock_instance.history.return_value = sample_df
        mock_ticker_cls.return_value = mock_instance

        symbol, df = DataStore.download_single_stock("AMD", period="1y", interval="1d")
        assert symbol == "AMD"
        assert df is not None

        all_data, failed = DataStore.download_batch_stocks(["AMD"])
        assert "AMD" in all_data
