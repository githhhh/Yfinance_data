import pandas as pd
import pytest

from eps_pit import EPSMissingReason, EPSResolveMode, EPSStatus
from eps_pit.lookup import SignalEPSLookup, enrich_pool_with_signal_eps, resolve_signal_eps
from eps_pit.store import EPSPITStore


def test_replay_mode_never_calls_current_tradingview(monkeypatch, tmp_path):
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_tradingview_eps",
        staticmethod(lambda codes: (_ for _ in ()).throw(AssertionError("TV forbidden in replay"))),
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

    result = resolve_signal_eps(
        "2026-08-21",
        "ABC",
        mode=EPSResolveMode.REPLAY,
        csv_path=str(tmp_path / "pit.csv"),
    )
    assert result.status is EPSStatus.EXPECTED_UNAVAILABLE


def test_live_tv_error_plus_pit_missing_is_provider_error(monkeypatch, tmp_path):
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

    result = resolve_signal_eps(
        "2026-08-21",
        "ABC",
        mode=EPSResolveMode.LIVE,
        observation_date="2026-08-21",
        csv_path=str(tmp_path / "pit.csv"),
    )
    assert result.status is EPSStatus.PROVIDER_ERROR
    assert result.missing_reason is EPSMissingReason.PROVIDER_ERROR
    assert EPSPITStore(str(tmp_path / "pit.csv")).get("2026-08-21", "ABC") is None


def test_live_tv_error_can_be_overridden_by_strict_pit_success(monkeypatch, tmp_path):
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
                codes[0]: {
                    "eps_yoy_growth": 50.0,
                    "source": "SEC",
                    "effective_date": "2026-08-01",
                }
            }
        ),
    )

    result = resolve_signal_eps(
        "2026-08-21",
        "ABC",
        mode=EPSResolveMode.LIVE,
        observation_date="2026-08-21",
        csv_path=str(tmp_path / "pit.csv"),
    )
    assert result.status is EPSStatus.RESOLVED
    assert result.eps_yoy_growth == 50.0


def test_live_tv_field_null_preserves_pit_business_missing_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_tradingview_eps",
        staticmethod(
            lambda codes: {
                codes[0]: {"missing_reason": EPSMissingReason.TV_FIELD_NULL}
            }
        ),
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

    result = resolve_signal_eps(
        "2026-08-21",
        "ABC",
        mode=EPSResolveMode.LIVE,
        observation_date="2026-08-21",
        csv_path=str(tmp_path / "pit.csv"),
    )

    assert result.status is EPSStatus.EXPECTED_UNAVAILABLE
    assert result.missing_reason is EPSMissingReason.NO_QUARTERLY_EPS


def test_refresh_disabled_is_not_mislabeled_expected_unavailable(tmp_path):
    result = resolve_signal_eps(
        "2026-08-21",
        "ABC",
        mode=EPSResolveMode.REPLAY,
        csv_path=str(tmp_path / "pit.csv"),
        allow_network=False,
    )
    assert result.status is EPSStatus.NOT_ATTEMPTED
    assert result.missing_reason is EPSMissingReason.REFRESH_DISABLED


def test_invalid_mode_snapshot_and_code_are_contract_errors(tmp_path):
    path = str(tmp_path / "pit.csv")
    with pytest.raises(TypeError, match="EPSResolveMode"):
        resolve_signal_eps("2026-08-21", "ABC", mode="replay", csv_path=path)
    with pytest.raises(ValueError, match="snapshot_date"):
        resolve_signal_eps("not-a-date", "ABC", mode=EPSResolveMode.REPLAY, csv_path=path)
    with pytest.raises(ValueError, match="code"):
        resolve_signal_eps("2026-08-21", pd.NA, mode=EPSResolveMode.REPLAY, csv_path=path)


def test_signal_rows_are_validated_before_any_partial_pit_write(monkeypatch, tmp_path):
    pit_path = tmp_path / "pit.csv"
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_tradingview_eps",
        staticmethod(
            lambda codes: {
                "GOOD": {"eps_yoy_growth": 10.0, "source": "TV_DIRECT"}
            }
        ),
    )
    pool = pd.DataFrame(
        [
            {"snapshot_date": "2026-08-21", "code": "GOOD", "signal": True, "eps_yoy_growth": pd.NA},
            {"snapshot_date": "bad-date", "code": "BAD", "signal": True, "eps_yoy_growth": pd.NA},
        ]
    )

    with pytest.raises(ValueError, match="snapshot_date"):
        enrich_pool_with_signal_eps(
            pool,
            csv_path=str(pit_path),
            refresh_missing=True,
            mode=EPSResolveMode.LIVE,
        observation_date="2026-08-21",
        )
    assert not pit_path.exists()


def test_live_batch_tradingview_failure_marks_all_unresolved_signals_provider_error(monkeypatch, tmp_path):
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_tradingview_eps",
        staticmethod(lambda codes: (_ for _ in ()).throw(RuntimeError("TV batch outage"))),
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
    pool = pd.DataFrame(
        [
            {"snapshot_date": "2026-08-21", "code": "A", "signal": True, "eps_yoy_growth": pd.NA},
            {"snapshot_date": "2026-08-21", "code": "B", "signal": True, "eps_yoy_growth": pd.NA},
        ]
    )

    enriched = enrich_pool_with_signal_eps(
        pool,
        csv_path=str(tmp_path / "pit.csv"),
        refresh_missing=True,
        mode=EPSResolveMode.LIVE,
        observation_date="2026-08-21",
    )
    assert set(enriched["eps_yoy_growth_status"]) == {EPSStatus.PROVIDER_ERROR.value}


def test_live_rerun_of_older_snapshot_never_calls_current_state_providers(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_tradingview_eps",
        staticmethod(
            lambda codes: (_ for _ in ()).throw(
                AssertionError("current TradingView forbidden for older snapshot")
            )
        ),
    )
    calls = []

    def fake_pit(snapshot, codes, **kwargs):
        calls.append((snapshot, codes, kwargs))
        return {
            codes[0]: {
                "eps_yoy_growth": 33.0,
                "source": "SEC",
                "effective_date": "2026-08-20",
            }
        }

    monkeypatch.setattr(SignalEPSLookup, "fetch_sec_yahoo_eps", staticmethod(fake_pit))

    result = resolve_signal_eps(
        "2026-08-26",
        "ABC",
        mode=EPSResolveMode.LIVE,
        observation_date="2026-08-27",
        csv_path=str(tmp_path / "pit.csv"),
    )

    assert result.status is EPSStatus.RESOLVED
    assert result.eps_yoy_growth == 33.0
    assert calls == [
        (
            "2026-08-26",
            ["ABC"],
            {"allow_current_yahoo": False, "observation_date": None},
        )
    ]


def test_true_live_snapshot_allows_current_yahoo_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_tradingview_eps",
        staticmethod(
            lambda codes: {
                codes[0]: {"missing_reason": EPSMissingReason.TV_FIELD_NULL}
            }
        ),
    )
    calls = []

    def fake_pit(snapshot, codes, **kwargs):
        calls.append((snapshot, codes, kwargs))
        return {
            codes[0]: {
                "eps_yoy_growth": 260.0,
                "source": "YahooLiveObserved",
                "effective_date": "2026-08-27",
            }
        }

    monkeypatch.setattr(SignalEPSLookup, "fetch_sec_yahoo_eps", staticmethod(fake_pit))

    result = resolve_signal_eps(
        "2026-08-27",
        "ALOT",
        mode=EPSResolveMode.LIVE,
        observation_date="2026-08-27",
        csv_path=str(tmp_path / "pit.csv"),
    )

    assert result.status is EPSStatus.RESOLVED
    assert result.source == "YahooLiveObserved"
    assert calls == [
        (
            "2026-08-27",
            ["ALOT"],
            {
                "allow_current_yahoo": True,
                "observation_date": "2026-08-27",
            },
        )
    ]


def test_live_existing_stage2_value_is_not_backdated_on_rerun(monkeypatch, tmp_path):
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_tradingview_eps",
        staticmethod(
            lambda codes: (_ for _ in ()).throw(
                AssertionError("stale Stage2 must not trigger current TV")
            )
        ),
    )
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_sec_yahoo_eps",
        staticmethod(
            lambda snapshot, codes, **kwargs: {
                codes[0]: {
                    "eps_yoy_growth": 42.0,
                    "source": "SEC",
                    "effective_date": "2026-08-20",
                }
            }
        ),
    )

    pool = pd.DataFrame(
        [
            {
                "snapshot_date": "2026-08-26",
                "code": "ABC",
                "signal": True,
                "eps_yoy_growth": 999.0,
            }
        ]
    )
    enriched = enrich_pool_with_signal_eps(
        pool,
        csv_path=str(tmp_path / "pit.csv"),
        refresh_missing=True,
        mode=EPSResolveMode.LIVE,
        observation_date="2026-08-27",
    )

    assert enriched.loc[0, "eps_yoy_growth"] == 42.0
    assert enriched.loc[0, "eps_yoy_growth_source"] == "SEC"


def test_replay_never_trusts_prefilled_pool_eps(monkeypatch, tmp_path):
    calls = []

    def fake_pit(snapshot, codes, **kwargs):
        calls.append((snapshot, codes, kwargs))
        return {
            codes[0]: {
                "eps_yoy_growth": 42.0,
                "source": "SEC",
                "effective_date": "2026-08-20",
                "sec_cik": "0000123456",
            }
        }

    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_tradingview_eps",
        staticmethod(
            lambda codes: (_ for _ in ()).throw(
                AssertionError("REPLAY must never call TradingView")
            )
        ),
    )
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_sec_yahoo_eps",
        staticmethod(fake_pit),
    )

    pool = pd.DataFrame(
        [
            {
                "snapshot_date": "2026-08-21",
                "code": "ABC",
                "signal": True,
                "eps_yoy_growth": 999.0,
                "eps_yoy_growth_source": "POOL_EXISTING",
            }
        ]
    )
    enriched = enrich_pool_with_signal_eps(
        pool,
        csv_path=str(tmp_path / "pit.csv"),
        refresh_missing=False,
        mode=EPSResolveMode.REPLAY,
        observation_date="2026-08-27",
    )

    assert enriched.loc[0, "eps_yoy_growth"] == 42.0
    assert enriched.loc[0, "eps_yoy_growth_source"] == "SEC"
    assert len(calls) == 1


def test_resolver_reuses_bound_sec_cik_hint(monkeypatch, tmp_path):
    store = EPSPITStore(str(tmp_path / "pit.csv"))
    store.upsert(
        __import__("eps_pit").EPSResult(
            code="ABC",
            snapshot_date="2026-08-20",
            status=EPSStatus.RESOLVED,
            eps_yoy_growth=10.0,
            source="SEC",
            effective_date="2026-08-01",
            sec_cik="0000123456",
        )
    )
    calls = []

    def fake_pit(snapshot, codes, **kwargs):
        calls.append(kwargs)
        return {
            codes[0]: {
                "eps_yoy_growth": 20.0,
                "source": "SEC",
                "effective_date": "2026-08-20",
                "sec_cik": "0000123456",
            }
        }

    monkeypatch.setattr(SignalEPSLookup, "fetch_sec_yahoo_eps", staticmethod(fake_pit))

    result = resolve_signal_eps(
        "2026-08-21",
        "ABC",
        mode=EPSResolveMode.REPLAY,
        csv_path=str(tmp_path / "pit.csv"),
        observation_date="2026-08-27",
    )

    assert result.status is EPSStatus.RESOLVED
    assert calls[0]["sec_cik_hints"] == {"ABC": "0000123456"}
