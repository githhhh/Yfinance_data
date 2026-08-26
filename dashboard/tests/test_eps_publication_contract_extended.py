import pandas as pd
import pytest

import yfinance_data
from eps_pit import EPSMissingReason
from eps_pit.lookup import SignalEPSLookup


def _pool(code: object, snapshot_date: object = "2026-08-21") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "code": code,
                "snapshot_date": snapshot_date,
                "signal": True,
                "eps_yoy_growth": pd.NA,
            }
        ]
    )


def test_current_pool_does_not_publish_when_live_tradingview_fails_and_pit_cannot_resolve(
    tmp_path,
    monkeypatch,
):
    pool_path = tmp_path / "breakout_follow_pool.csv"
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(pool_path))
    monkeypatch.setattr(SignalEPSLookup, "DEFAULT_CSV_PATH", str(tmp_path / "pit.csv"))
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_tradingview_eps",
        staticmethod(lambda codes: (_ for _ in ()).throw(RuntimeError("TV outage"))),
    )
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_sec_yahoo_eps",
        staticmethod(
            lambda snapshot, codes: {
                codes[0]: {"missing_reason": EPSMissingReason.NO_QUARTERLY_EPS}
            }
        ),
    )

    with pytest.raises(RuntimeError, match="EPS provider failure"):
        yfinance_data.BreakoutFollowPoolRun.weekend().save_snapshot(_pool("ERR"))
    assert not pool_path.exists()


def test_current_pool_rejects_invalid_signal_snapshot_before_writing_anything(
    tmp_path,
    monkeypatch,
):
    pool_path = tmp_path / "breakout_follow_pool.csv"
    pit_path = tmp_path / "pit.csv"
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(pool_path))
    monkeypatch.setattr(SignalEPSLookup, "DEFAULT_CSV_PATH", str(pit_path))

    with pytest.raises(ValueError, match="snapshot_date"):
        yfinance_data.BreakoutFollowPoolRun.weekend().save_snapshot(
            _pool("BAD", snapshot_date="not-a-date")
        )

    assert not pool_path.exists()
    assert not pit_path.exists()
