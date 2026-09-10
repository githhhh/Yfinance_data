from __future__ import annotations

import time

import pandas as pd
import requests
import tradingview_screener

from eps_pit.providers.tradingview_provider import TradingViewEPSProvider


class _FakeColumn:
    def isin(self, values):
        return ("isin", tuple(values))

    def __eq__(self, value):
        return ("eq", value)


def test_retries_transient_tls_failures_three_times_before_succeeding(monkeypatch):
    attempts = []
    delays = []

    class FlakyQuery:
        def select(self, *args):
            return self

        def where(self, *args):
            return self

        def limit(self, value):
            return self

        def set_markets(self, market):
            return self

        def get_scanner_data(self):
            attempts.append(1)
            if len(attempts) <= 3:
                raise requests.exceptions.SSLError("unexpected EOF")
            return 1, pd.DataFrame(
                [
                    {
                        "name": "SUNC",
                        "exchange": "NASDAQ",
                        "earnings_per_share_diluted_yoy_growth_fq": 344.8,
                    }
                ]
            )

    monkeypatch.setattr(tradingview_screener, "Query", FlakyQuery)
    monkeypatch.setattr(tradingview_screener, "col", lambda name: _FakeColumn())
    monkeypatch.setattr(time, "sleep", delays.append)

    result = TradingViewEPSProvider().fetch_eps_yoy(["SUNC"])

    assert len(attempts) == 4
    assert delays == [1.0, 2.0, 4.0]
    assert result["SUNC"]["eps_yoy_growth"] == 344.8
