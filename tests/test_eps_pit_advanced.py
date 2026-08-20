import os
import pytest
import pandas as pd
import numpy as np
from eps_pit.growth import EPSGrowthCalculator
from eps_pit.fiscal_period import FiscalPeriodMatcher
from eps_pit.pit import PITTimelineEngine
from eps_pit.audit import ReplayPoolAuditor


def test_asof_merge_multiple_dates_progression():
    """Verify that as snapshots progress across time, merge_asof selects the correct sequential quarter."""
    events = pd.DataFrame([
        {
            "code": "AAPL",
            "fiscal_year": 2025,
            "fiscal_quarter": "Q4",
            "report_period": "2025-09-30",
            "eps_yoy_growth": 10.0,
            "effective_at_conservative": "2025-10-31",
            "effective_at_release": "2025-10-30",
            "growth_status": "NORMAL_POSITIVE",
            "source": "SEC",
        },
        {
            "code": "AAPL",
            "fiscal_year": 2026,
            "fiscal_quarter": "Q1",
            "report_period": "2025-12-31",
            "eps_yoy_growth": 20.0,
            "effective_at_conservative": "2026-01-30",
            "effective_at_release": "2026-01-29",
            "growth_status": "NORMAL_POSITIVE",
            "source": "SEC",
        },
        {
            "code": "AAPL",
            "fiscal_year": 2026,
            "fiscal_quarter": "Q2",
            "report_period": "2026-03-31",
            "eps_yoy_growth": 30.0,
            "effective_at_conservative": "2026-05-01",
            "effective_at_release": "2026-04-30",
            "growth_status": "NORMAL_POSITIVE",
            "source": "SEC",
        },
    ])

    # Snap 1: 2025-12-01 -> should match Q4 2025 (growth = 10.0)
    snap1 = pd.DataFrame([{"code": "AAPL", "snapshot_date": "2025-12-01"}])
    p1, _ = PITTimelineEngine.merge_asof_snapshot(snap1, events, snapshot_date_col="snapshot_date")
    assert p1.loc[0, "eps_yoy_growth"] == 10.0

    # Snap 2: 2026-02-15 -> should match Q1 2026 (growth = 20.0)
    snap2 = pd.DataFrame([{"code": "AAPL", "snapshot_date": "2026-02-15"}])
    p2, _ = PITTimelineEngine.merge_asof_snapshot(snap2, events, snapshot_date_col="snapshot_date")
    assert p2.loc[0, "eps_yoy_growth"] == 20.0

    # Snap 3: 2026-04-20 -> before 2026-05-01 filing -> should still match Q1 2026 (growth = 20.0)
    snap3 = pd.DataFrame([{"code": "AAPL", "snapshot_date": "2026-04-20"}])
    p3, _ = PITTimelineEngine.merge_asof_snapshot(snap3, events, snapshot_date_col="snapshot_date", pit_mode="conservative")
    assert p3.loc[0, "eps_yoy_growth"] == 20.0

    # Snap 4: 2026-05-15 -> after 2026-05-01 filing -> should match Q2 2026 (growth = 30.0)
    snap4 = pd.DataFrame([{"code": "AAPL", "snapshot_date": "2026-05-15"}])
    p4, _ = PITTimelineEngine.merge_asof_snapshot(snap4, events, snapshot_date_col="snapshot_date", pit_mode="conservative")
    assert p4.loc[0, "eps_yoy_growth"] == 30.0


def test_near_zero_base_eps_handling():
    """Verify that near-zero base EPS (< 0.005) is flagged as NEAR_ZERO_BASE."""
    growth, status, is_calc = EPSGrowthCalculator.calculate(0.10, 0.001)
    assert is_calc is True
    assert status == "NEAR_ZERO_BASE"
    assert growth > 5000.0


def test_dual_mode_timestamp_difference():
    """Verify differences between conservative (SEC filing) and release (earnings release) dates."""
    events = pd.DataFrame([
        {
            "code": "XYZ",
            "eps_yoy_growth": 50.0,
            "effective_at_conservative": "2026-05-05",  # Filed Tuesday
            "effective_at_release": "2026-04-30",       # Released previous Thursday
            "growth_status": "NORMAL_POSITIVE",
            "source": "SEC",
        }
    ])

    snap_midweek = pd.DataFrame([{"code": "XYZ", "snapshot_date": "2026-05-01"}])
    
    # Conservative mode: not available on 2026-05-01
    p_cons, _ = PITTimelineEngine.merge_asof_snapshot(snap_midweek, events, snapshot_date_col="snapshot_date", pit_mode="conservative")
    assert pd.isna(p_cons.loc[0, "eps_yoy_growth"])

    # Release mode: available on 2026-05-01
    p_rel, _ = PITTimelineEngine.merge_asof_snapshot(snap_midweek, events, snapshot_date_col="snapshot_date", pit_mode="release")
    assert p_rel.loc[0, "eps_yoy_growth"] == 50.0
