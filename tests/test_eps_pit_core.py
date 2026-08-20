import pytest
import pandas as pd
import numpy as np
from eps_pit.growth import EPSGrowthCalculator
from eps_pit.fiscal_period import FiscalPeriodMatcher
from eps_pit.pit import PITTimelineEngine
from eps_pit.mapping import TickerMapper


def test_known_positive_eps_case():
    """Verify standard positive EPS growth calculation (1.00 -> 1.25 => 25.0%)."""
    growth, status, is_calc = EPSGrowthCalculator.calculate(1.25, 1.00)
    assert is_calc is True
    assert status == "NORMAL_POSITIVE"
    assert pytest.approx(growth, 0.001) == 25.0


def test_special_eps_cases():
    """Verify special EPS cases classification."""
    # Loss to profit (turnaround)
    growth, status, is_calc = EPSGrowthCalculator.calculate(0.50, -0.50)
    assert is_calc is True
    assert status == "LOSS_TO_PROFIT"
    assert pytest.approx(growth, 0.001) == 200.0

    # Profit to loss
    growth, status, is_calc = EPSGrowthCalculator.calculate(-0.25, 1.00)
    assert is_calc is True
    assert status == "PROFIT_TO_LOSS"
    assert pytest.approx(growth, 0.001) == -125.0

    # Loss narrowing (-1.00 -> -0.40)
    growth, status, is_calc = EPSGrowthCalculator.calculate(-0.40, -1.00)
    assert is_calc is True
    assert status == "LOSS_NARROWING"
    assert pytest.approx(growth, 0.001) == 60.0

    # Loss widening (-0.40 -> -1.00)
    growth, status, is_calc = EPSGrowthCalculator.calculate(-1.00, -0.40)
    assert is_calc is True
    assert status == "LOSS_WIDENING"
    assert pytest.approx(growth, 0.001) == -150.0

    # Zero base
    growth, status, is_calc = EPSGrowthCalculator.calculate(1.00, 0.00)
    assert is_calc is False
    assert status == "ZERO_BASE"
    assert growth is None


def test_fiscal_quarter_matching():
    """Verify matching pairs use the identical fiscal quarter from prior fiscal year."""
    records = [
        {"fiscal_year": 2024, "fiscal_quarter": "Q1", "report_period": "2024-03-31", "eps_diluted": 1.0, "filing_date": "2024-05-01"},
        {"fiscal_year": 2024, "fiscal_quarter": "Q2", "report_period": "2024-06-30", "eps_diluted": 1.1, "filing_date": "2024-08-01"},
        {"fiscal_year": 2024, "fiscal_quarter": "Q3", "report_period": "2024-09-30", "eps_diluted": 1.2, "filing_date": "2024-11-01"},
        {"fiscal_year": 2024, "fiscal_quarter": "Q4", "report_period": "2024-12-31", "eps_diluted": 1.3, "filing_date": "2025-02-01"},
        {"fiscal_year": 2025, "fiscal_quarter": "Q1", "report_period": "2025-03-31", "eps_diluted": 1.5, "filing_date": "2025-05-01"},
        {"fiscal_year": 2025, "fiscal_quarter": "Q2", "report_period": "2025-06-30", "eps_diluted": 1.6, "filing_date": "2025-08-01"},
    ]
    matched = FiscalPeriodMatcher.match_quarters(records)
    
    # 2025 Q1 should match 2024 Q1
    q1_2025 = [m for m in matched if m[0]["fiscal_year"] == 2025 and m[0]["fiscal_quarter"] == "Q1"][0]
    assert q1_2025[1] is not None
    assert q1_2025[1]["fiscal_year"] == 2024
    assert q1_2025[1]["fiscal_quarter"] == "Q1"
    assert q1_2025[2] == "EXACT_FISCAL_MATCH"

    # 2025 Q2 should match 2024 Q2
    q2_2025 = [m for m in matched if m[0]["fiscal_year"] == 2025 and m[0]["fiscal_quarter"] == "Q2"][0]
    assert q2_2025[1]["fiscal_year"] == 2024
    assert q2_2025[1]["fiscal_quarter"] == "Q2"


def test_no_future_leakage_and_rejection():
    """Verify that effective_at > snapshot_date is strictly rejected by merge_asof."""
    events = pd.DataFrame([
        {
            "code": "TEST",
            "report_period": "2026-03-31",
            "fiscal_year": 2026,
            "fiscal_quarter": "Q1",
            "eps_current": 2.0,
            "eps_prior_year": 1.0,
            "eps_yoy_growth": 100.0,
            "growth_status": "NORMAL_POSITIVE",
            "effective_at_conservative": "2026-04-15",
            "effective_at_release": "2026-04-10",
            "effective_date_method": "SEC_FILING",
            "source": "SEC",
        }
    ])

    # Snapshot before filing date (e.g. 2026-04-05) -> must be NaN (future data rejected)
    snap_before = pd.DataFrame([
        {"code": "TEST", "snapshot_date": "2026-04-05", "close": 100.0}
    ])
    patched_before, prov_before = PITTimelineEngine.merge_asof_snapshot(
        snap_before, events, snapshot_date_col="snapshot_date", pit_mode="conservative"
    )
    assert pd.isna(patched_before.loc[0, "eps_yoy_growth"])

    # Snapshot on/after filing date (e.g. 2026-04-20) -> must be filled with 100.0
    snap_after = pd.DataFrame([
        {"code": "TEST", "snapshot_date": "2026-04-20", "close": 105.0}
    ])
    patched_after, prov_after = PITTimelineEngine.merge_asof_snapshot(
        snap_after, events, snapshot_date_col="snapshot_date", pit_mode="conservative"
    )
    assert patched_after.loc[0, "eps_yoy_growth"] == 100.0


def test_original_rows_and_order_preserved():
    """Verify input row count and row order are 100% preserved."""
    events = pd.DataFrame([
        {
            "code": "AAPL",
            "eps_yoy_growth": 25.0,
            "growth_status": "NORMAL_POSITIVE",
            "effective_at_conservative": "2026-01-15",
            "effective_at_release": "2026-01-15",
            "source": "SEC",
        },
        {
            "code": "NVDA",
            "eps_yoy_growth": 150.0,
            "growth_status": "NORMAL_POSITIVE",
            "effective_at_conservative": "2026-01-15",
            "effective_at_release": "2026-01-15",
            "source": "SEC",
        }
    ])

    snap = pd.DataFrame([
        {"code": "NVDA", "snapshot_date": "2026-02-01", "signal": True, "custom_metric": 42},
        {"code": "AAPL", "snapshot_date": "2026-02-01", "signal": False, "custom_metric": 99},
        {"code": "UNKNOWN", "snapshot_date": "2026-02-01", "signal": True, "custom_metric": 10},
    ])

    patched, prov = PITTimelineEngine.merge_asof_snapshot(snap, events, snapshot_date_col="snapshot_date")
    assert len(patched) == 3
    assert list(patched["code"]) == ["NVDA", "AAPL", "UNKNOWN"]
    assert list(patched["custom_metric"]) == [42, 99, 10]
    assert patched.loc[0, "eps_yoy_growth"] == 150.0
    assert patched.loc[1, "eps_yoy_growth"] == 25.0
    assert pd.isna(patched.loc[2, "eps_yoy_growth"])


def test_ticker_normalization():
    """Verify ticker normalization for class shares and dot/dash conversion."""
    mapper = TickerMapper()
    assert mapper.normalize_ticker("BRK.B") == "BRK-B"
    assert mapper.normalize_ticker("BF.B") == "BF-B"
    assert mapper.normalize_ticker("AAPL") == "AAPL"
    assert mapper.normalize_ticker("moga") == "MOGA"
