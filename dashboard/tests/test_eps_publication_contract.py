from types import SimpleNamespace

import pandas as pd
import pytest

import yfinance_data
from eps_pit import EPSStatus
from eps_pit.lookup import SignalEPSLookup


def _pool(code: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "code": code,
                "snapshot_date": "2026-08-21",
                "signal": True,
                "eps_yoy_growth": pd.NA,
            }
        ]
    )


def test_current_pool_does_not_publish_on_eps_provider_error(tmp_path, monkeypatch):
    pool_path = tmp_path / "breakout_follow_pool.csv"
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(pool_path))
    monkeypatch.setattr(SignalEPSLookup, "DEFAULT_CSV_PATH", str(tmp_path / "signal_eps_pit.csv"))
    monkeypatch.setattr(SignalEPSLookup, "fetch_tradingview_eps", staticmethod(lambda codes: {}))
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_sec_yahoo_eps",
        staticmethod(lambda snapshot, codes, **kwargs: {codes[0]: {"missing_reason": "PROVIDER_ERROR"}}),
    )

    with pytest.raises(RuntimeError, match="BF Pool signal EPS provider failure: ERR"):
        yfinance_data.BreakoutFollowPoolRun.weekend().save_snapshot(_pool("ERR"))

    assert not pool_path.exists()


def test_expected_eps_unavailable_is_published_with_reason(tmp_path, monkeypatch):
    pool_path = tmp_path / "breakout_follow_pool.csv"
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(pool_path))
    monkeypatch.setattr(SignalEPSLookup, "DEFAULT_CSV_PATH", str(tmp_path / "signal_eps_pit.csv"))
    monkeypatch.setattr(SignalEPSLookup, "fetch_tradingview_eps", staticmethod(lambda codes: {}))
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_sec_yahoo_eps",
        staticmethod(
            lambda snapshot, codes, **kwargs: {
                codes[0]: {"missing_reason": "NO_PRIOR_YEAR_QUARTER"}
            }
        ),
    )

    yfinance_data.BreakoutFollowPoolRun.weekend().save_snapshot(_pool("IPO"))

    saved = pd.read_csv(pool_path)
    assert pd.isna(saved.loc[0, "eps_yoy_growth"])
    assert saved.loc[0, "eps_yoy_growth_status"] == EPSStatus.EXPECTED_UNAVAILABLE.value
    assert saved.loc[0, "eps_yoy_growth_missing_reason"] == "NO_PRIOR_YEAR_QUARTER"


def test_pool_commit_stages_pit_store_with_pool(tmp_path, monkeypatch):
    pool_path = tmp_path / "us" / "breakout_follow_pool.csv"
    pit_path = tmp_path / "us" / "signal_eps_pit.csv"
    pool_path.parent.mkdir(parents=True)
    pool_path.write_text("code\nABC\n")
    pit_path.write_text("snapshot_date,code,eps_yoy_growth\n2026-08-21,ABC,25\n")

    monkeypatch.setattr(yfinance_data, "DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(SignalEPSLookup, "DEFAULT_CSV_PATH", "us/signal_eps_pit.csv")
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:4] == ["git", "diff", "--cached", "--quiet"]:
            return SimpleNamespace(returncode=1, args=args)
        return SimpleNamespace(returncode=0, args=args)

    monkeypatch.setattr(yfinance_data.subprocess, "run", fake_run)

    yfinance_data._commit_pool(str(pool_path))

    add_call = calls[0]
    assert add_call[:2] == ["git", "add"]
    assert str(pool_path) in add_call
    assert str(pit_path) in add_call
