from pathlib import Path

import pandas as pd

from eps_pit import EPSResolveMode, EPSStatus
from eps_pit.lookup import SignalEPSLookup, enrich_pool_with_signal_eps, resolve_signal_eps
from eps_pit.store import EPSPITStore


def test_signal_eps_enrichment_uses_pit_then_sec_yahoo_for_signal_rows(tmp_path, monkeypatch):
    pit_path = tmp_path / "signal_eps_pit.csv"
    pd.DataFrame(
        [{"snapshot_date": "2026-08-14", "code": "PIT", "eps_yoy_growth": 31.5}]
    ).to_csv(pit_path, index=False)

    def fake_fetch(snapshot_date, codes, **kwargs):
        assert snapshot_date == "2026-08-14"
        assert codes in (["MISS"], ["SECY"])
        if codes == ["SECY"]:
            return {
                "SECY": {
                    "eps_yoy_growth": 42.0,
                    "source": "SEC",
                    "effective_date": "2026-08-01",
                    "current_eps": 1.42,
                    "prior_year_eps": 1.00,
                    "current_period": "2026-06-30",
                    "prior_year_period": "2025-06-30",
                }
            }
        return {"MISS": {"missing_reason": "NO_QUARTERLY_EPS"}}

    monkeypatch.setattr(SignalEPSLookup, "fetch_sec_yahoo_eps", staticmethod(fake_fetch))

    pool = pd.DataFrame(
        [
            {"snapshot_date": "2026-08-14", "code": "PIT", "signal": True, "eps_yoy_growth": pd.NA},
            {"snapshot_date": "2026-08-14", "code": "SECY", "signal": True, "eps_yoy_growth": pd.NA},
            {"snapshot_date": "2026-08-14", "code": "QUIET", "signal": False, "eps_yoy_growth": pd.NA},
            {"snapshot_date": "2026-08-14", "code": "MISS", "signal": True, "eps_yoy_growth": pd.NA},
        ]
    )

    enriched = enrich_pool_with_signal_eps(
        pool,
        csv_path=str(pit_path),
        refresh_missing=True,
        mode=EPSResolveMode.REPLAY,
    )

    assert enriched.loc[0, "eps_yoy_growth"] == 31.5
    assert enriched.loc[0, "eps_yoy_growth_source"] == "PIT"
    assert enriched.loc[1, "eps_yoy_growth"] == 42.0
    assert enriched.loc[1, "eps_yoy_growth_source"] == "SEC"
    assert enriched.loc[1, "eps_yoy_growth_status"] == EPSStatus.RESOLVED.value
    assert pd.isna(enriched.loc[2, "eps_yoy_growth"])
    assert pd.isna(enriched.loc[3, "eps_yoy_growth"])
    assert enriched.loc[3, "eps_yoy_growth_status"] == EPSStatus.EXPECTED_UNAVAILABLE.value


def test_signal_eps_enrichment_refreshes_only_missing_signal_rows(monkeypatch, tmp_path):
    requested_codes = []

    def fake_fetch(snapshot_date, codes, **kwargs):
        requested_codes.extend(codes)
        return {
            "MISS": {
                "eps_yoy_growth": 55.0,
                "source": "Yahoo",
                "effective_date": "2026-08-05",
                "current_eps": 1.55,
                "prior_year_eps": 1.00,
            }
        }

    monkeypatch.setattr(SignalEPSLookup, "fetch_sec_yahoo_eps", staticmethod(fake_fetch))

    pool = pd.DataFrame(
        [
            {"snapshot_date": "2026-08-21", "code": "MISS", "signal": True, "eps_yoy_growth": pd.NA},
            {"snapshot_date": "2026-08-21", "code": "EXISTING", "signal": True, "eps_yoy_growth": 12.0},
            {"snapshot_date": "2026-08-21", "code": "QUIET", "signal": False, "eps_yoy_growth": pd.NA},
        ]
    )

    enriched = enrich_pool_with_signal_eps(
        pool,
        csv_path=str(tmp_path / "signal_eps_pit.csv"),
        refresh_missing=True,
        mode=EPSResolveMode.REPLAY,
    )

    assert requested_codes == ["MISS"]
    assert enriched.loc[0, "eps_yoy_growth"] == 55.0
    assert enriched.loc[1, "eps_yoy_growth"] == 12.0
    assert enriched.loc[1, "eps_yoy_growth_source"] == "POOL_EXISTING"
    assert pd.isna(enriched.loc[2, "eps_yoy_growth"])


def test_signal_eps_enrichment_can_disable_direct_refresh(monkeypatch, tmp_path):
    def fake_fetch(snapshot_date, codes, **kwargs):
        raise AssertionError("direct refresh should be disabled")

    monkeypatch.setattr(SignalEPSLookup, "fetch_sec_yahoo_eps", staticmethod(fake_fetch))
    pool = pd.DataFrame(
        [{"snapshot_date": "2026-08-21", "code": "MISS", "signal": True, "eps_yoy_growth": pd.NA}]
    )

    enriched = enrich_pool_with_signal_eps(
        pool,
        csv_path=str(tmp_path / "missing_pit.csv"),
        refresh_missing=False,
        mode=EPSResolveMode.REPLAY,
    )

    assert pd.isna(enriched.loc[0, "eps_yoy_growth"])


def test_replay_mode_never_calls_tradingview(monkeypatch, tmp_path):
    def fail_tv(codes):
        raise AssertionError("REPLAY must never call current TradingView")

    monkeypatch.setattr(SignalEPSLookup, "fetch_tradingview_eps", staticmethod(fail_tv))
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_sec_yahoo_eps",
        staticmethod(lambda snapshot, codes, **kwargs: {codes[0]: {"missing_reason": "NO_QUARTERLY_EPS"}}),
    )

    result = resolve_signal_eps(
        "2026-08-21",
        "MISS",
        mode=EPSResolveMode.REPLAY,
        csv_path=str(tmp_path / "signal_eps_pit.csv"),
    )

    assert result.status is EPSStatus.EXPECTED_UNAVAILABLE


def test_live_mode_batches_tradingview_and_persists_exact_snapshot(monkeypatch, tmp_path):
    pit_path = tmp_path / "signal_eps_pit.csv"
    calls = []

    def fake_tv(codes):
        calls.append(codes)
        return {
            "TV": {
                "eps_yoy_growth": 88.0,
                "source": "TV_DIRECT",
                "calculation_method": "provider_reported_yoy",
            }
        }

    monkeypatch.setattr(SignalEPSLookup, "fetch_tradingview_eps", staticmethod(fake_tv))
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_sec_yahoo_eps",
        staticmethod(lambda snapshot, codes, **kwargs: (_ for _ in ()).throw(AssertionError("TV should resolve first"))),
    )

    pool = pd.DataFrame(
        [{"snapshot_date": "2026-08-21", "code": "TV", "signal": True, "eps_yoy_growth": pd.NA}]
    )
    enriched = enrich_pool_with_signal_eps(
        pool,
        csv_path=str(pit_path),
        refresh_missing=True,
        mode=EPSResolveMode.LIVE,
        observation_date="2026-08-21",
    )

    assert calls == [["TV"]]
    assert enriched.loc[0, "eps_yoy_growth"] == 88.0
    assert enriched.loc[0, "eps_yoy_growth_source"] == "TV_DIRECT"
    stored = EPSPITStore(str(pit_path)).get("2026-08-21", "TV")
    assert stored is not None and stored.eps_yoy_growth == 88.0


def test_persistent_store_is_reused_without_network(monkeypatch, tmp_path):
    pit_path = tmp_path / "signal_eps_pit.csv"
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_tradingview_eps",
        staticmethod(lambda codes: {"ABC": {"eps_yoy_growth": 25.0, "source": "TV_DIRECT"}}),
    )
    first = resolve_signal_eps(
        "2026-08-21",
        "ABC",
        mode=EPSResolveMode.LIVE,
        observation_date="2026-08-21",
        csv_path=str(pit_path),
    )
    assert first.eps_yoy_growth == 25.0

    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_tradingview_eps",
        staticmethod(lambda codes: (_ for _ in ()).throw(AssertionError("network must not be used"))),
    )
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_sec_yahoo_eps",
        staticmethod(lambda snapshot, codes, **kwargs: (_ for _ in ()).throw(AssertionError("network must not be used"))),
    )
    second = resolve_signal_eps(
        "2026-08-21",
        "ABC",
        mode=EPSResolveMode.REPLAY,
        csv_path=str(pit_path),
    )

    assert second.eps_yoy_growth == 25.0
    assert second.source == "TV_DIRECT"


def test_existing_upstream_eps_is_preserved_tagged_and_persisted(tmp_path):
    pit_path = tmp_path / "signal_eps_pit.csv"
    pool = pd.DataFrame(
        [{"snapshot_date": "2026-08-21", "code": "EXIST", "signal": True, "eps_yoy_growth": 37.5}]
    )

    enriched = enrich_pool_with_signal_eps(
        pool,
        csv_path=str(pit_path),
        refresh_missing=False,
        mode=EPSResolveMode.LIVE,
        observation_date="2026-08-21",
    )

    assert enriched.loc[0, "eps_yoy_growth"] == 37.5
    assert enriched.loc[0, "eps_yoy_growth_source"] == "TV_STAGE2"
    stored = EPSPITStore(str(pit_path)).get("2026-08-21", "EXIST")
    assert stored is not None
    assert stored.eps_yoy_growth == 37.5
    assert stored.source == "TV_STAGE2"


def test_provider_error_is_explicit_and_not_persisted(monkeypatch, tmp_path):
    pit_path = tmp_path / "signal_eps_pit.csv"
    monkeypatch.setattr(SignalEPSLookup, "fetch_tradingview_eps", staticmethod(lambda codes: {}))
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_sec_yahoo_eps",
        staticmethod(lambda snapshot, codes, **kwargs: (_ for _ in ()).throw(RuntimeError("temporary outage"))),
    )

    result = resolve_signal_eps(
        "2026-08-21",
        "ERR",
        mode=EPSResolveMode.LIVE,
        observation_date="2026-08-21",
        csv_path=str(pit_path),
    )

    assert result.status is EPSStatus.PROVIDER_ERROR
    assert EPSPITStore(str(pit_path)).get("2026-08-21", "ERR") is None


def test_expected_unavailable_is_not_persisted(monkeypatch, tmp_path):
    pit_path = tmp_path / "signal_eps_pit.csv"
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_sec_yahoo_eps",
        staticmethod(lambda snapshot, codes, **kwargs: {"MISS": {"missing_reason": "NO_PRIOR_YEAR_QUARTER"}}),
    )

    result = resolve_signal_eps(
        "2026-08-21",
        "MISS",
        mode=EPSResolveMode.REPLAY,
        csv_path=str(pit_path),
    )

    assert result.status is EPSStatus.EXPECTED_UNAVAILABLE
    assert EPSPITStore(str(pit_path)).get("2026-08-21", "MISS") is None


def test_eps_public_api_uses_enum_not_string_mode():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "eps_pit" / "lookup.py").read_text()

    assert "mode: EPSResolveMode" in text
    assert 'mode="live"' not in text
    assert 'mode="replay"' not in text
