import importlib
import sys
import types

import pandas as pd


class _FakeColumn:
    def isin(self, _values):
        return self

    def has_none_of(self, _value):
        return self

    def empty(self):
        return self

    def __eq__(self, _other):
        return self

    def __ge__(self, _other):
        return self


def _install_fake_tradingview_module():
    fake_module = types.ModuleType("tradingview_screener")
    fake_module.Query = object
    fake_module.col = lambda _name: _FakeColumn()
    sys.modules.setdefault("tradingview_screener", fake_module)


def test_stage2_screener_exports_eps_and_52w_high_columns(tmp_path, monkeypatch):
    _install_fake_tradingview_module()
    stage2_screener = importlib.import_module("stage2_screener")

    class FakeQuery:
        select_calls = []
        frames = [
            pd.DataFrame(
                [
                    {
                        "name": "AAA",
                        "close": 20.0,
                        "SMA10|1W": 18.0,
                        "SMA40|1W": 15.0,
                        "sector": "Technology Services",
                        "industry": "Software",
                        "earnings_per_share_diluted_yoy_growth_fq": 42.5,
                        "price_52_week_high": 21.0,
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "name": "BBB",
                        "close": 30.0,
                        "SMA10|1W": 25.0,
                        "SMA40|1W": None,
                        "sector": "Health Technology",
                        "industry": "Biotechnology",
                        "earnings_per_share_diluted_yoy_growth_fq": -3.0,
                        "price_52_week_high": 31.0,
                    }
                ]
            ),
        ]

        def __init__(self):
            self._frame = self.frames.pop(0)

        def select(self, *columns):
            self.select_calls.append(columns)
            return self

        def where(self, *_conditions):
            return self

        def limit(self, _limit):
            return self

        def set_markets(self, _market):
            return self

        def get_scanner_data(self):
            return len(self._frame), self._frame

    monkeypatch.setattr(stage2_screener, "Query", FakeQuery)

    output_file = tmp_path / "stage2_whitelist.csv"
    total, df, tickers = stage2_screener._query_stage2(str(output_file), verbose=False)

    assert total == 2
    assert tickers == ["AAA", "BBB"]
    assert all("earnings_per_share_diluted_yoy_growth_fq" in call for call in FakeQuery.select_calls)
    assert all("price_52_week_high" in call for call in FakeQuery.select_calls)
    assert "eps_yoy_growth" in df.columns
    assert "earnings_per_share_diluted_yoy_growth_fq" not in df.columns
    assert "price_52_week_high" in df.columns
    assert df.loc[df["code"].eq("AAA"), "eps_yoy_growth"].item() == 42.5
    saved = pd.read_csv(output_file)
    assert {"code", "eps_yoy_growth", "price_52_week_high"}.issubset(saved.columns)
