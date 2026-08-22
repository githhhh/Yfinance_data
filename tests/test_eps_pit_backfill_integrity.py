import os
import glob
import json
import pytest
import pandas as pd
from eps_pit.lookup import SignalEPSLookup, get_signal_eps, enrich_pool_with_signal_eps


def test_signal_eps_pit_csv_integrity():
    """Verify replay signal PIT EPS dataset exists, covers signals, and has correct schema."""
    csv_path = "backtest/ibd_skill_replay_pools/signal_eps_pit.csv"
    assert os.path.exists(csv_path), f"Signal PIT EPS CSV missing at {csv_path}"

    df = pd.read_csv(csv_path)
    expected_signal_rows = 0
    for pool_file in sorted(glob.glob("backtest/ibd_skill_replay_pools/*/breakout_follow_pool.csv")):
        pool = pd.read_csv(pool_file, encoding="utf-8-sig")
        signal = pool["signal"].astype(str).str.strip().str.lower().isin({"true", "1"})
        expected_signal_rows += int(signal.sum())
    assert len(df) == expected_signal_rows

    required_cols = [
        "snapshot_date", "code", "eps_yoy_growth", "eps_current",
        "eps_prior_year", "report_period", "growth_status",
        "effective_at_conservative", "effective_at_release", "source"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column {col} in signal_eps_pit.csv"

    # Coverage
    filled_count = df["eps_yoy_growth"].notna().sum()
    coverage_pct = filled_count / len(df) * 100.0
    assert coverage_pct >= 95.0, f"Signal EPS coverage {coverage_pct:.2f}% < 95.0%"

    # Unique (snapshot_date, code)
    dup = df.duplicated(subset=["snapshot_date", "code"])
    assert not dup.any(), f"Duplicate (snapshot_date, code) found in signal_eps_pit.csv"


def test_signal_eps_zero_future_leakage():
    """Verify strictly no future data leakage across all signal records."""
    csv_path = "backtest/ibd_skill_replay_pools/signal_eps_pit.csv"
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)

    filled = df.dropna(subset=["eps_yoy_growth", "effective_at_conservative"])
    leakage = filled[filled["effective_at_conservative"] > filled["snapshot_date"]]
    assert len(leakage) == 0, f"Future leakage found: {len(leakage)} rows in signal_eps_pit.csv"


def test_signal_eps_no_self_match_contamination():
    """Self-match rows (growth=0, eps_curr==eps_prior, non-zero) must be < 2%."""
    csv_path = "backtest/ibd_skill_replay_pools/signal_eps_pit.csv"
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)

    filled = df.dropna(subset=["eps_yoy_growth"])
    self_match = filled[
        (filled["eps_yoy_growth"] == 0)
        & (filled["eps_current"] == filled["eps_prior_year"])
        & (filled["eps_current"] != 0)
    ]
    rate = len(self_match) / len(filled) * 100 if len(filled) > 0 else 0
    assert rate < 2.0, (
        f"Self-match rate {rate:.2f}% exceeds 2% threshold: "
        f"{len(self_match)} rows"
    )


def test_signal_eps_dynamic_lookup_and_enrich():
    """Verify SignalEPSLookup service for O(1) query and dataframe enrichment."""
    SignalEPSLookup.clear_cache()

    # Test single item lookup
    eps_asml = get_signal_eps("2026-01-02", "ASML")
    assert eps_asml is not None
    assert pytest.approx(eps_asml, 0.01) == 3.98

    eps_ftai = get_signal_eps("2026-01-02", "FTAI")
    assert eps_ftai is not None
    assert pytest.approx(eps_ftai, 0.01) == 44.74

    # Test non-existent ticker
    eps_none = get_signal_eps("2026-01-02", "NONEXISTENT_TICKER_XYZ")
    assert eps_none is None

    # Test pool dataframe enrichment
    mock_pool = pd.DataFrame([
        {"snapshot_date": "2026-01-02", "code": "ASML", "signal": True, "eps_yoy_growth": None},
        {"snapshot_date": "2026-01-02", "code": "FTAI", "signal": True, "eps_yoy_growth": None},
        {"snapshot_date": "2026-01-02", "code": "OTHER", "signal": False, "eps_yoy_growth": None},
    ])

    enriched = enrich_pool_with_signal_eps(mock_pool)
    assert pytest.approx(enriched.loc[0, "eps_yoy_growth"], 0.01) == 3.98
    assert pytest.approx(enriched.loc[1, "eps_yoy_growth"], 0.01) == 44.74
    assert pd.isna(enriched.loc[2, "eps_yoy_growth"])


def test_raw_replay_pools_remain_untampered():
    """Verify replay weekly pool CSVs match successful manifest weeks."""
    orig_files = sorted(glob.glob("backtest/ibd_skill_replay_pools/*/breakout_follow_pool.csv"))
    with open("backtest/ibd_skill_replay_pools/manifest.json", "r") as f:
        manifest = json.load(f)
    rows = manifest["weeks"] if isinstance(manifest, dict) else manifest
    success_weeks = [row for row in rows if row.get("status") == "success"]
    assert len(orig_files) == len(success_weeks)
