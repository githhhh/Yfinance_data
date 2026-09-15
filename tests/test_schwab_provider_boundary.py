import sys
from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock

from data_providers.schwab_provider import (
    SchwabCredentials,
    SchwabDataProvider,
    SchwabRawTokenClient,
)


class _PeriodType(Enum):
    YEAR = "year"


class _Period(Enum):
    ONE_YEAR = 1
    TWO_YEARS = 2
    THREE_YEARS = 3
    FIVE_YEARS = 5
    TEN_YEARS = 10
    FIFTEEN_YEARS = 15
    TWENTY_YEARS = 20


class _FrequencyType(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"


class _Frequency(Enum):
    DAILY = 1
    WEEKLY = 1


def _fake_schwab_module():
    price_history = SimpleNamespace(
        PeriodType=_PeriodType,
        Period=_Period,
        FrequencyType=_FrequencyType,
        Frequency=_Frequency,
    )
    return SimpleNamespace(
        client=SimpleNamespace(Client=SimpleNamespace(PriceHistory=price_history))
    )


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


def test_formal_daily_history_maps_two_year_request_to_schwab_enums(monkeypatch):
    monkeypatch.setitem(sys.modules, "schwab", _fake_schwab_module())
    mock_client = MagicMock()
    provider = SchwabDataProvider(client=mock_client)

    provider._request_price_history("AAPL", period="2y", interval="1d")

    kwargs = mock_client.get_price_history.call_args.kwargs
    assert kwargs == {
        "period_type": _PeriodType.YEAR,
        "period": _Period.TWO_YEARS,
        "frequency_type": _FrequencyType.DAILY,
        "frequency": _Frequency.DAILY,
    }


def test_formal_weekly_history_maps_five_year_request_to_schwab_enums(monkeypatch):
    monkeypatch.setitem(sys.modules, "schwab", _fake_schwab_module())
    mock_client = MagicMock()
    provider = SchwabDataProvider(client=mock_client)

    provider._request_price_history("AAPL", period="5y", interval="1wk")

    kwargs = mock_client.get_price_history.call_args.kwargs
    assert kwargs == {
        "period_type": _PeriodType.YEAR,
        "period": _Period.FIVE_YEARS,
        "frequency_type": _FrequencyType.WEEKLY,
        "frequency": _Frequency.WEEKLY,
    }


def test_raw_token_client_serializes_schwab_enums_to_wire_values(tmp_path, monkeypatch):
    token_path = tmp_path / "token.json"
    token_path.write_text('{"access_token": "access-token"}', encoding="utf-8")
    calls = []

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"candles": []}

        def raise_for_status(self):
            pass

    def fake_get(url, headers, params=None, timeout=30):
        calls.append({"url": url, "headers": headers, "params": params, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("data_providers.schwab_provider.requests.get", fake_get)
    client = SchwabRawTokenClient(SchwabCredentials(token_path=str(token_path)))

    client.get_price_history(
        "AAPL",
        period_type=_PeriodType.YEAR,
        period=_Period.FIVE_YEARS,
        frequency_type=_FrequencyType.WEEKLY,
        frequency=_Frequency.WEEKLY,
    )

    assert calls[0]["params"] == {
        "symbol": "AAPL",
        "periodType": "year",
        "period": 5,
        "frequencyType": "weekly",
        "frequency": 1,
        "needExtendedHoursData": "false",
    }
