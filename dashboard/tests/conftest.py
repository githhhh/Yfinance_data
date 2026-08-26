import pytest

from eps_pit.lookup import SignalEPSLookup


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
