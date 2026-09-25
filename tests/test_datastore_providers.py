import pytest
import pandas as pd
import threading
import time
from unittest.mock import MagicMock, patch
from data_providers.base_provider import BaseDataProvider
from data_providers.ohlcv_validation import DataIntegrityError
from data_providers.yahoo_provider import YahooDataProvider
from data_providers.schwab_provider import SchwabDataProvider, SchwabCredentials, SchwabRawTokenClient
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
                "Adj Close": [101.111, 102.222],
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
        assert df.loc[df.index[0], "Open"] == 100.123
        assert df.loc[df.index[0], "Close"] == 103.555
        assert "Adj Close" not in df.columns
        history_kwargs = mock_instance.history.call_args.kwargs
        assert history_kwargs["auto_adjust"] is False
        assert "rounding" not in history_kwargs

    @patch("yfinance.Ticker")
    def test_download_single_stock_rejects_missing_required_price_column(self, mock_ticker_cls):
        sample_df = pd.DataFrame(
            {
                "Open": [100.123],
                "High": [105.789],
                "Low": [99.001],
                "Volume": [10000],
            },
            index=pd.date_range("2026-01-01", periods=1),
        )
        mock_instance = MagicMock()
        mock_instance.history.return_value = sample_df
        mock_ticker_cls.return_value = mock_instance

        provider = YahooDataProvider(max_retries=0)
        with pytest.raises(DataIntegrityError, match="missing required OHLCV columns"):
            provider.download_single_stock("AAPL", period="1y", interval="1d")

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
        assert df.iloc[0]["Open"] == 150.126
        assert df.iloc[0]["High"] == 155.888
        assert df.iloc[0]["Volume"] == 5000000

    def test_download_single_stock_rejects_missing_required_price_column(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "candles": [
                {
                    "open": 150.126,
                    "high": 155.888,
                    "low": 149.333,
                    "volume": 5000000,
                    "datetime": 1672531200000,
                }
            ],
            "empty": False,
        }
        mock_client.get_price_history.return_value = mock_resp

        provider = SchwabDataProvider(client=mock_client, max_retries=0)
        symbol, df = provider.download_single_stock("AAPL")

        assert symbol == "AAPL"
        assert df is None

    def test_download_single_stock_filters_inconsistent_bar(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "candles": [
                {"open": 35.50, "high": 35.50, "low": 35.50, "close": 35.51,
                 "volume": 242833, "datetime": 1782450000000}
            ],
            "empty": False,
        }
        mock_client.get_price_history.return_value = mock_resp
        provider = SchwabDataProvider(client=mock_client, max_retries=0, rate_limit_sleep=0)

        symbol, frame = provider.download_single_stock("FNLC")

        assert symbol == "FNLC"
        assert frame is None
        assert "High<Close" in provider._last_failure_reasons["FNLC"]

    def test_inconsistent_vendor_bar_is_excluded_before_round_trip(self, tmp_path, monkeypatch):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "candles": [
                {"open": 35.50, "high": 35.50, "low": 35.50, "close": 35.51,
                 "volume": 242833, "datetime": 1782450000000}
            ],
            "empty": False,
        }
        mock_client.get_price_history.return_value = mock_resp
        provider = SchwabDataProvider(client=mock_client, max_retries=0, rate_limit_sleep=0)
        symbol, frame = provider.download_single_stock("FNLC")
        assert frame is None
        valid = pd.DataFrame(
            {"Open": [10.0], "High": [11.0], "Low": [9.0],
             "Close": [10.5], "Volume": [100]},
            index=pd.to_datetime(["2026-09-21"]),
        )
        batch = {reference: valid.copy() for reference in ("^GSPC", "^IXIC", "^DJI", "MSFT")}
        filtered, excluded = DataStore.filter_schwab_stock_data(
            batch, failed=[symbol], expected_symbols=[*batch, symbol],
            interval="1wk", failure_reasons=provider.failure_reasons,
        )
        assert excluded == ["FNLC"]

        output = tmp_path / "schwab.pkl"
        monkeypatch.setattr(DataStore, "get_stock_pkl_path", lambda interval: str(output))
        saved = DataStore.save_stock_data(
            filtered, save_dir=str(tmp_path), interval="1wk",
            expected_symbols=list(filtered),
        )
        assert saved == str(output)
        restored = DataStore.load_stock_data(saved)
        assert set(restored) == set(filtered)
        assert "FNLC" not in restored

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

    def test_download_single_stock_retries_transient_empty_response(self, monkeypatch):
        mock_client = MagicMock()
        empty_resp = MagicMock()
        empty_resp.json.return_value = {"candles": [], "empty": True}
        good_resp = MagicMock()
        good_resp.json.return_value = {
            "candles": [
                {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 100, "datetime": 1672531200000}
            ],
            "empty": False,
        }
        mock_client.get_price_history.side_effect = [empty_resp, good_resp]
        monkeypatch.setattr("data_providers.schwab_provider.time.sleep", lambda _: None)

        provider = SchwabDataProvider(client=mock_client, max_retries=1, rate_limit_sleep=0)
        symbol, df = provider.download_single_stock("AAPL")

        assert symbol == "AAPL"
        assert df is not None
        assert len(df) == 1
        assert mock_client.get_price_history.call_count == 2

    def test_download_batch_stocks_paces_requests_across_workers(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "candles": [
                {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 100, "datetime": 1672531200000}
            ],
            "empty": False,
        }
        starts = []
        lock = threading.Lock()
        active = 0
        peak_active = 0

        def get_history(*args, **kwargs):
            nonlocal active, peak_active
            with lock:
                starts.append(time.monotonic())
                active += 1
                peak_active = max(peak_active, active)
            time.sleep(0.06)
            with lock:
                active -= 1
            return mock_resp

        mock_client.get_price_history.side_effect = get_history
        provider = SchwabDataProvider(
            client=mock_client, batch_size=4, max_workers=4,
            max_retries=0, rate_limit_sleep=0.02, recovery_rounds=0,
        )
        all_data, failed = provider.download_batch_stocks(["AAPL", "MSFT", "NVDA", "TSLA"])

        assert set(all_data) == {"AAPL", "MSFT", "NVDA", "TSLA"}
        assert failed == []
        assert min(b - a for a, b in zip(starts, starts[1:])) >= 0.015
        assert peak_active >= 2  # Network waits overlap despite shared request pacing.

    def test_batch_retries_failed_symbols_after_other_downloads(self, capsys):
        mock_client = MagicMock()
        good_resp = MagicMock()
        good_resp.json.return_value = {
            "candles": [
                {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5,
                 "volume": 100, "datetime": 1672531200000}
            ],
            "empty": False,
        }
        attempts = {"AAPL": 0, "MSFT": 0}

        def get_history(symbol, **kwargs):
            attempts[symbol] += 1
            if symbol == "AAPL" and attempts[symbol] <= 2:
                raise ConnectionError("temporary disconnect")
            return good_resp

        mock_client.get_price_history.side_effect = get_history
        provider = SchwabDataProvider(
            client=mock_client, batch_size=2, max_workers=2,
            max_retries=1, rate_limit_sleep=0, recovery_rounds=1,
            recovery_sleep=0,
        )
        all_data, failed = provider.download_batch_stocks(["AAPL", "MSFT"])

        assert set(all_data) == {"AAPL", "MSFT"}
        assert failed == []
        assert attempts == {"AAPL": 3, "MSFT": 1}
        assert "temporary disconnect" in capsys.readouterr().out

    def test_batch_reports_last_reason_when_symbol_still_fails(self, capsys):
        mock_client = MagicMock()
        mock_client.get_price_history.side_effect = ConnectionError("persistent disconnect")
        provider = SchwabDataProvider(
            client=mock_client, max_retries=0, rate_limit_sleep=0,
            recovery_rounds=1, recovery_sleep=0,
        )

        all_data, failed = provider.download_batch_stocks(["AAPL"])

        assert all_data == {}
        assert failed == ["AAPL"]
        assert "Failed AAPL: ConnectionError: persistent disconnect" in capsys.readouterr().out

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

    def test_raw_token_client_fetches_quote_with_bearer_token(self, tmp_path, monkeypatch):
        token_path = tmp_path / "token.json"
        token_path.write_text('{"access_token": "access-token"}', encoding="utf-8")
        calls = []

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"AAPL": {"lastPrice": 180.5}}

            def raise_for_status(self):
                pass

        def fake_get(url, headers, params=None, timeout=30):
            calls.append({"url": url, "headers": headers, "params": params, "timeout": timeout})
            return FakeResponse()

        monkeypatch.setattr("data_providers.schwab_provider.requests.get", fake_get)

        client = SchwabRawTokenClient(SchwabCredentials(token_path=str(token_path)))

        assert client.get_quote("AAPL") == {"AAPL": {"lastPrice": 180.5}}
        assert calls[0]["headers"]["Authorization"] == "Bearer access-token"
        assert calls[0]["url"].endswith("/marketdata/v1/AAPL/quotes")

    def test_raw_token_client_fetches_price_history_with_download_params(self, tmp_path, monkeypatch):
        token_path = tmp_path / "token.json"
        token_path.write_text('{"access_token": "access-token"}', encoding="utf-8")
        calls = []

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"candles": [{"close": 10.0}]}

            def raise_for_status(self):
                pass

        def fake_get(url, headers, params=None, timeout=30):
            calls.append({"url": url, "headers": headers, "params": params, "timeout": timeout})
            return FakeResponse()

        monkeypatch.setattr("data_providers.schwab_provider.requests.get", fake_get)

        client = SchwabRawTokenClient(SchwabCredentials(token_path=str(token_path)))
        response = client.get_price_history(
            "AAPL",
            period_type="year",
            period=1,
            frequency_type="daily",
            frequency=1,
        )

        assert response.json() == {"candles": [{"close": 10.0}]}
        assert calls[0]["url"].endswith("/marketdata/v1/pricehistory")
        assert calls[0]["params"] == {
            "symbol": "AAPL",
            "periodType": "year",
            "period": 1,
            "frequencyType": "daily",
            "frequency": 1,
            "needExtendedHoursData": "false",
        }


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

    def test_results_pkl_round_trip_preserves_source_precision(self, tmp_path, monkeypatch):
        pkl_path = tmp_path / "stock_data_test_1d.pkl"
        monkeypatch.setattr(DataStore, "get_stock_pkl_path", lambda interval="1d": str(pkl_path))
        source_df = pd.DataFrame(
            {
                "Open": [100.123456],
                "High": [105.789123],
                "Low": [99.001987],
                "Close": [103.555123],
                "Volume": [10000],
            },
            index=pd.date_range("2026-01-01", periods=1),
        )

        saved_path = DataStore.save_stock_data({"AAPL": source_df}, save_dir=str(tmp_path), interval="1d")
        loaded = DataStore.load_stock_data(saved_path)

        assert loaded["AAPL"].iloc[0]["Open"] == pytest.approx(100.123456)
        assert loaded["AAPL"].iloc[0]["High"] == pytest.approx(105.789123)
        assert loaded["AAPL"].iloc[0]["Low"] == pytest.approx(99.001987)
        assert loaded["AAPL"].iloc[0]["Close"] == pytest.approx(103.555123)
