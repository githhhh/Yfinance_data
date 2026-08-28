import pandas as pd
import pytest

import yfinance_data
import eps_pit.lookup as eps_lookup
from eps_pit import EPSMissingReason, EPSResolveMode
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
    monkeypatch.setattr(eps_lookup, "current_eps_observation_date", lambda: "2026-08-21")
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_tradingview_eps",
        staticmethod(lambda codes: (_ for _ in ()).throw(RuntimeError("TV outage"))),
    )
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_sec_yahoo_eps",
        staticmethod(
            lambda snapshot, codes, **kwargs: {
                codes[0]: {"missing_reason": EPSMissingReason.NO_QUARTERLY_EPS}
            }
        ),
    )

    with pytest.raises(RuntimeError, match="EPS provider failure"):
        yfinance_data.BreakoutFollowPoolRun.weekend().save_snapshot(_pool("ERR"))
    assert not pool_path.exists()


def test_pool_publication_uses_live_mode_and_repo_pit_path_by_default(tmp_path, monkeypatch):
    pool_path = tmp_path / "breakout_follow_pool.csv"
    pit_path = tmp_path / "signal_eps_pit.csv"
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(pool_path))
    monkeypatch.setattr(SignalEPSLookup, "DEFAULT_CSV_PATH", str(pit_path))
    monkeypatch.setattr(eps_lookup, "current_eps_observation_date", lambda: "2026-08-28")
    calls = []

    def fake_enrich(pool, **kwargs):
        calls.append(kwargs)
        df = pool.copy()
        df["eps_yoy_growth"] = 260.0
        df["eps_yoy_growth_source"] = "YahooLiveObserved"
        df["eps_yoy_growth_status"] = "resolved"
        df["eps_yoy_growth_missing_reason"] = pd.NA
        df["eps_growth_type"] = "TURNAROUND"
        return df

    monkeypatch.setattr(yfinance_data, "enrich_pool_with_signal_eps", fake_enrich)

    yfinance_data.BreakoutFollowPoolRun.weekend().save_snapshot(
        _pool("ALOT", snapshot_date="2026-08-26")
    )

    assert calls == [
        {
            "refresh_missing": True,
            "mode": EPSResolveMode.LIVE,
            "csv_path": str(pit_path),
        }
    ]
    saved = pd.read_csv(pool_path)
    assert saved.loc[0, "eps_yoy_growth_source"] == "YahooLiveObserved"


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


def test_weekend_and_midweek_share_the_same_save_snapshot_implementation():
    weekend = yfinance_data.BreakoutFollowPoolRun.weekend()
    midweek = yfinance_data.BreakoutFollowPoolRun.midweek()

    assert weekend.save_snapshot.__func__ is midweek.save_snapshot.__func__


@pytest.mark.parametrize(
    ("factory_name", "path_attr"),
    [
        ("weekend", "BREAKOUT_FOLLOW_POOL_PATH"),
        ("midweek", "BREAKOUT_FOLLOW_POOL_MIDWEEK_PATH"),
    ],
)
def test_weekend_and_midweek_publish_complete_eps_pit_metadata(
    tmp_path,
    monkeypatch,
    factory_name,
    path_attr,
):
    pool_path = tmp_path / f"{factory_name}.csv"
    pit_path = tmp_path / "signal_eps_pit.csv"
    monkeypatch.setattr(yfinance_data, path_attr, str(pool_path))
    monkeypatch.setattr(SignalEPSLookup, "DEFAULT_CSV_PATH", str(pit_path))
    monkeypatch.setattr(eps_lookup, "current_eps_observation_date", lambda: "2026-08-27")
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_tradingview_eps",
        staticmethod(lambda codes: {}),
    )
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_sec_yahoo_eps",
        staticmethod(
            lambda snapshot, codes, **kwargs: {
                code: {"missing_reason": EPSMissingReason.NO_PRIOR_YEAR_QUARTER}
                for code in codes
            }
        ),
    )

    run = getattr(yfinance_data.BreakoutFollowPoolRun, factory_name)()
    run.save_snapshot(
        pd.DataFrame(
            [
                {
                    "code": "EXISTING",
                    "snapshot_date": "2026-08-27",
                    "signal": True,
                    "eps_yoy_growth": 25.5,
                },
                {
                    "code": "MISSING",
                    "snapshot_date": "2026-08-27",
                    "signal": True,
                    "eps_yoy_growth": pd.NA,
                },
                {
                    "code": "QUIET",
                    "snapshot_date": "2026-08-27",
                    "signal": False,
                    "eps_yoy_growth": pd.NA,
                },
            ]
        )
    )

    saved = pd.read_csv(pool_path)
    assert set(yfinance_data.EPS_PUBLICATION_COLUMNS).issubset(saved.columns)

    existing = saved.loc[saved["code"].eq("EXISTING")].iloc[0]
    assert existing["eps_yoy_growth"] == 25.5
    assert existing["eps_yoy_growth_source"] == "TV_STAGE2"
    assert existing["eps_yoy_growth_status"] == "resolved"
    assert pd.isna(existing["eps_yoy_growth_missing_reason"])

    missing = saved.loc[saved["code"].eq("MISSING")].iloc[0]
    assert pd.isna(missing["eps_yoy_growth"])
    assert missing["eps_yoy_growth_status"] == "expected_unavailable"
    assert missing["eps_yoy_growth_missing_reason"] == "NO_PRIOR_YEAR_QUARTER"

    run.ensure_current_snapshot()


@pytest.mark.parametrize(
    ("factory_name", "path_attr"),
    [
        ("weekend", "BREAKOUT_FOLLOW_POOL_PATH"),
        ("midweek", "BREAKOUT_FOLLOW_POOL_MIDWEEK_PATH"),
    ],
)
def test_weekend_and_midweek_fail_closed_if_enrichment_is_bypassed(
    tmp_path,
    monkeypatch,
    factory_name,
    path_attr,
):
    pool_path = tmp_path / f"{factory_name}.csv"
    monkeypatch.setattr(yfinance_data, path_attr, str(pool_path))
    monkeypatch.setattr(yfinance_data, "_enrich_signal_eps", lambda pool: pool.copy())

    run = getattr(yfinance_data.BreakoutFollowPoolRun, factory_name)()
    with pytest.raises(ValueError, match="EPS PIT publication 字段不完整"):
        run.save_snapshot(
            pd.DataFrame(
                [
                    {
                        "code": "LEGACY",
                        "snapshot_date": "2026-08-27",
                        "signal": True,
                        "eps_yoy_growth": 12.0,
                    }
                ]
            )
        )

    assert not pool_path.exists()


@pytest.mark.parametrize(
    ("factory_name", "path_attr"),
    [
        ("weekend", "BREAKOUT_FOLLOW_POOL_PATH"),
        ("midweek", "BREAKOUT_FOLLOW_POOL_MIDWEEK_PATH"),
    ],
)
def test_commit_rejects_legacy_pool_even_when_digest_matches(
    tmp_path,
    monkeypatch,
    factory_name,
    path_attr,
):
    pool_path = tmp_path / f"{factory_name}.csv"
    monkeypatch.setattr(yfinance_data, path_attr, str(pool_path))
    pd.DataFrame(
        [
            {
                "code": "LEGACY",
                "snapshot_date": "2026-08-27",
                "signal": True,
                "eps_yoy_growth": 12.0,
            }
        ]
    ).to_csv(pool_path, index=False)

    run = getattr(yfinance_data.BreakoutFollowPoolRun, factory_name)()
    run._published_digest = yfinance_data._snapshot_digest(str(pool_path))

    with pytest.raises(ValueError, match="EPS PIT publication 字段不完整"):
        run.commit()


def test_private_commit_helper_rejects_legacy_eps_before_git(tmp_path, monkeypatch):
    pool_path = tmp_path / "legacy.csv"
    pd.DataFrame(
        [
            {
                "code": "LEGACY",
                "snapshot_date": "2026-08-27",
                "signal": True,
                "eps_yoy_growth": 12.0,
            }
        ]
    ).to_csv(pool_path, index=False)

    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        raise AssertionError("git must not be reached for an invalid pool")

    monkeypatch.setattr(yfinance_data.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="EPS PIT publication 字段不完整"):
        yfinance_data._commit_pool(str(pool_path))

    assert calls == []


def test_publication_rejects_non_finite_resolved_eps(tmp_path, monkeypatch):
    pool_path = tmp_path / "breakout_follow_pool.csv"
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(pool_path))
    monkeypatch.setattr(
        yfinance_data,
        "_enrich_signal_eps",
        lambda pool: pool.assign(
            eps_yoy_growth=float("inf"),
            eps_yoy_growth_source="TEST",
            eps_yoy_growth_status="resolved",
            eps_yoy_growth_missing_reason=pd.NA,
            eps_growth_type="GROWTH",
        ),
    )

    with pytest.raises(ValueError, match="EPS PIT publication EPS 非有限数值"):
        yfinance_data.BreakoutFollowPoolRun.weekend().save_snapshot(
            pd.DataFrame(
                [
                    {
                        "code": "INF",
                        "snapshot_date": "2026-08-27",
                        "signal": True,
                        "eps_yoy_growth": 1.0,
                    }
                ]
            )
        )

    assert not pool_path.exists()
