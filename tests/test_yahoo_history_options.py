import pandas as pd

import data_providers.yahoo_provider as yahoo_module
from data_providers.yahoo_provider import YahooDataProvider


def test_yahoo_history_excludes_corporate_action_only_rows(monkeypatch):
    captured = {}
    frame = pd.DataFrame(
        {
            "Open": [10.0],
            "High": [11.0],
            "Low": [9.0],
            "Close": [10.5],
            "Volume": [1000],
        },
        index=pd.to_datetime(["2026-09-11"]),
    )

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            captured.update(kwargs)
            return frame.copy()

    monkeypatch.setattr(yahoo_module.yf, "Ticker", FakeTicker)

    provider = YahooDataProvider(max_retries=0)
    symbol, data = provider.download_single_stock(
        "AVT", period="2y", interval="1d"
    )

    assert symbol == "AVT"
    assert data is not None
    assert captured["auto_adjust"] is False
    assert captured["actions"] is False
    assert captured["keepna"] is False
