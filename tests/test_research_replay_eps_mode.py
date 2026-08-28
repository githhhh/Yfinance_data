from __future__ import annotations

import pandas as pd

from backtest.replay_eps import get_replay_signal_eps, replay_signal_eps_lookup
from dashboard import skill_industry_eps_known as selector
from eps_pit import EPSResolveMode, EPSResult, EPSStatus
from eps_pit.lookup import SignalEPSLookup
from eps_pit.store import EPSPITStore


def _resolved(code: str, snapshot: str, value: float, source: str) -> EPSResult:
    return EPSResult(
        code=code,
        snapshot_date=snapshot,
        status=EPSStatus.RESOLVED,
        eps_yoy_growth=value,
        source=source,
        effective_date=snapshot,
    )


def test_research_replay_lookup_uses_replay_store_not_live_store(tmp_path, monkeypatch):
    live_path = tmp_path / "live.csv"
    replay_path = tmp_path / "replay.csv"
    snapshot = "2026-08-21"

    EPSPITStore(str(live_path)).upsert(_resolved("ABC", snapshot, 999.0, "LIVE_TEST"))
    EPSPITStore(str(replay_path)).upsert(_resolved("ABC", snapshot, 42.0, "REPLAY_TEST"))
    monkeypatch.setattr(SignalEPSLookup, "DEFAULT_CSV_PATH", str(live_path))
    monkeypatch.setattr(SignalEPSLookup, "DEFAULT_REPLAY_CSV_PATH", str(replay_path))

    assert get_replay_signal_eps(snapshot, "ABC", allow_network=False) == 42.0

    row = pd.Series(
        {
            "snapshot_date": snapshot,
            "code": "ABC",
            "eps_yoy_growth": pd.NA,
        }
    )
    with replay_signal_eps_lookup(allow_network=False):
        assert selector.row_eps(row, "ABC") == 42.0

    # Context restoration preserves production's normal LIVE default behavior.
    assert selector.row_eps(row, "ABC") == 999.0


def test_replay_lookup_never_calls_current_tradingview(tmp_path, monkeypatch):
    replay_path = tmp_path / "replay.csv"
    monkeypatch.setattr(SignalEPSLookup, "DEFAULT_REPLAY_CSV_PATH", str(replay_path))
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_tradingview_eps",
        staticmethod(lambda codes: (_ for _ in ()).throw(AssertionError("TradingView LIVE path forbidden"))),
    )
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_sec_yahoo_eps",
        staticmethod(
            lambda snapshot, codes, **kwargs: {
                codes[0]: {
                    "eps_yoy_growth": 33.0,
                    "source": "SEC",
                    "effective_date": snapshot,
                }
            }
        ),
    )

    assert get_replay_signal_eps(
        "2026-08-21",
        "ABC",
        allow_network=True,
    ) == 33.0


def test_replay_mode_constant_is_explicit():
    assert EPSResolveMode.REPLAY.value == "replay"
