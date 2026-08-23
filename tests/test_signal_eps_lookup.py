import pandas as pd

from pathlib import Path

from eps_pit.lookup import SignalEPSLookup, enrich_pool_with_signal_eps


def test_signal_eps_enrichment_uses_pit_then_sec_yahoo_for_signal_rows(tmp_path, monkeypatch):
    pit_path = tmp_path / "signal_eps_pit.csv"

    pd.DataFrame(
        [
            {
                "snapshot_date": "2026-08-14",
                "code": "PIT",
                "eps_yoy_growth": 31.5,
            },
        ]
    ).to_csv(pit_path, index=False)

    def fake_fetch(snapshot_date, codes):
        assert snapshot_date == "2026-08-14"
        assert codes == ["MISS", "SECY"]
        return {
            "SECY": {
                "eps_yoy_growth": 42.0,
                "source": "SEC",
                "effective_date": "2026-08-01",
                "current_eps": 1.42,
                "prior_year_eps": 1.00,
            }
        }

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
    )

    assert enriched.loc[0, "eps_yoy_growth"] == 31.5
    assert enriched.loc[0, "eps_yoy_growth_source"] == "PIT"
    assert enriched.loc[1, "eps_yoy_growth"] == 42.0
    assert enriched.loc[1, "eps_yoy_growth_source"] == "SEC"
    assert pd.isna(enriched.loc[2, "eps_yoy_growth"])
    assert pd.isna(enriched.loc[3, "eps_yoy_growth"])
    assert "eps_yoy_growth_repair_method" not in enriched.columns
    assert "eps_yoy_growth_effective_date" not in enriched.columns
    assert "eps_yoy_growth_current_eps" not in enriched.columns
    assert "eps_yoy_growth_prior_year_eps" not in enriched.columns


def test_signal_eps_enrichment_refreshes_only_missing_signal_rows_with_sec_yahoo(monkeypatch, tmp_path):
    requested_codes = []

    def fake_fetch(snapshot_date, codes):
        assert snapshot_date == "2026-08-21"
        requested_codes.extend(codes)
        return {
            "MISS": {
                "eps_yoy_growth": 55.0,
                "source": "Yahoo",
                "effective_date": "2026-08-05",
                "current_eps": 1.55,
                "prior_year_eps": 1.00,
            },
            "QUIET": {"eps_yoy_growth": 88.0, "source": "Yahoo"},
            "EXISTING": {"eps_yoy_growth": 99.0, "source": "Yahoo"},
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
        csv_path=str(tmp_path / "missing_pit.csv"),
        refresh_missing=True,
    )

    assert requested_codes == ["MISS"]
    assert enriched.loc[0, "eps_yoy_growth"] == 55.0
    assert enriched.loc[0, "eps_yoy_growth_source"] == "Yahoo"
    assert enriched.loc[1, "eps_yoy_growth"] == 12.0
    assert pd.isna(enriched.loc[2, "eps_yoy_growth"])
    assert "eps_yoy_growth_repair_method" not in enriched.columns


def test_signal_eps_enrichment_can_disable_direct_refresh(monkeypatch, tmp_path):
    def fake_fetch(snapshot_date, codes):
        raise AssertionError("direct refresh should be disabled")

    monkeypatch.setattr(SignalEPSLookup, "fetch_sec_yahoo_eps", staticmethod(fake_fetch))

    pool = pd.DataFrame(
        [{"snapshot_date": "2026-08-21", "code": "MISS", "signal": True, "eps_yoy_growth": pd.NA}]
    )

    enriched = enrich_pool_with_signal_eps(
        pool,
        csv_path=str(tmp_path / "missing_pit.csv"),
        refresh_missing=False,
    )

    assert pd.isna(enriched.loc[0, "eps_yoy_growth"])


def test_signal_eps_supplement_does_not_use_tradingview():
    repo_root = Path(__file__).resolve().parents[1]
    checked_files = [
        repo_root / "eps_pit" / "lookup.py",
        repo_root / "eps_pit" / "providers" / "sec_yahoo_provider.py",
    ]
    text = "\n".join(path.read_text() for path in checked_files if path.exists())

    assert "tradingview" not in text.lower()
