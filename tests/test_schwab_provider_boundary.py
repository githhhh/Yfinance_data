from unittest.mock import MagicMock

from data_providers.schwab_provider import SchwabDataProvider


def test_schwab_provider_owns_conservative_default_pacing():
    provider = SchwabDataProvider(client=MagicMock())

    assert provider.batch_size == 1
    assert provider.max_workers == 1
    assert provider.rate_limit_sleep == 0.55


def test_index_alias_is_used_only_at_schwab_request_boundary():
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "candles": [
            {
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100,
                "datetime": 1672531200000,
            }
        ],
        "empty": False,
    }
    mock_client.get_price_history.return_value = mock_resp

    provider = SchwabDataProvider(client=mock_client, max_retries=0)
    symbol, frame = provider.download_single_stock("^GSPC")

    assert symbol == "^GSPC"
    assert frame is not None
    assert mock_client.get_price_history.call_args.args[0] == "$SPX"
