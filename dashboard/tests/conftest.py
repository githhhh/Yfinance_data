import pytest

import yfinance_data
from eps_pit.lookup import SignalEPSLookup
from yfinance_data import pool_industry


@pytest.fixture
def isolated_bf_pool_data(tmp_path, monkeypatch):
    """Keep Pool publication tests independent of the checkout's data and Git history."""
    us_dir = tmp_path / "us"
    us_dir.mkdir()
    monkeypatch.setattr(yfinance_data, "DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(
        yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(us_dir / "breakout_follow_pool.csv"),
    )
    monkeypatch.setattr(
        yfinance_data, "BREAKOUT_FOLLOW_POOL_MIDWEEK_PATH",
        str(us_dir / "breakout_follow_pool_midweek.csv"),
    )
    monkeypatch.setattr(SignalEPSLookup, "DEFAULT_CSV_PATH", str(us_dir / "signal_eps_pit.csv"))
    monkeypatch.setattr(pool_industry, "_default_data_root", lambda: tmp_path)


@pytest.fixture(autouse=True)
def _disable_live_eps_network(monkeypatch):
    """Dashboard/contract tests must never depend on live external EPS state."""
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_tradingview_eps",
        staticmethod(lambda codes: {}),
    )
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_sec_yahoo_eps",
        staticmethod(
            lambda snapshot, codes: {
                code: {"missing_reason": "NO_QUARTERLY_EPS"} for code in codes
            }
        ),
    )
