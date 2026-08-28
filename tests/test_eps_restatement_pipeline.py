from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backtest.b0_top3_quality_audit.research_windows import (
    contaminated_validation_dates,
    train_dates,
)
from backtest.b0_top3_quality_audit.run_eps_restatement import (
    _explicit_eligible,
    _restate_candidate_event_outcomes,
)
from backtest.ibd_skill_replay_pools.recalibrate_eps_pit import (
    _assert_non_eps_unchanged,
)


def test_fixed_research_windows_do_not_shift_when_weeks_are_missing():
    dates = [
        "2025-10-10",
        "2026-05-22",
        "2026-05-29",
        # deliberately omit a middle validation week
        "2026-08-07",
        "2026-08-14",
    ]
    assert train_dates(dates) == {"2025-10-10", "2026-05-22"}
    assert contaminated_validation_dates(dates) == {"2026-05-29", "2026-08-07"}


def test_eps_recalibration_allows_only_eps_column_changes():
    before = pd.DataFrame(
        [
            {
                "code": "ABC",
                "signal": True,
                "latest_close": 10.123456789,
                "eps_yoy_growth": 20.0,
                "eps_yoy_growth_source": "OLD",
            }
        ]
    )
    after = before.copy()
    after["eps_yoy_growth"] = 42.0
    after["eps_yoy_growth_source"] = "SEC"
    after["eps_yoy_growth_status"] = "resolved"
    after["eps_yoy_growth_repair_method"] = pd.NA
    _assert_non_eps_unchanged(before, after, Path("pool.csv"))

    bad = after.copy()
    bad["latest_close"] = 11.0
    with pytest.raises(AssertionError):
        _assert_non_eps_unchanged(before, bad, Path("pool.csv"))


def test_event_restatement_updates_eps_but_preserves_frozen_outcomes(tmp_path):
    path = tmp_path / "candidate_event_outcomes.parquet"
    old = pd.DataFrame(
        [
            {
                "event_id": "2026-01-01_ABC_0",
                "snapshot_date": "2026-01-01",
                "code": "ABC",
                "eps_yoy_growth": 10.0,
                "entry_status": "ENTRY_OK",
                "entry_open": 100.0,
                "week1_close_return_pct": 5.0,
                "stop_8_hit_ever": False,
            }
        ]
    )
    old.to_parquet(path, index=False)
    new_events = pd.DataFrame(
        [
            {
                "event_id": "2026-01-01_ABC_0",
                "snapshot_date": "2026-01-01",
                "code": "ABC",
                "eps_yoy_growth": 55.0,
                "eps_yoy_growth_source": "SEC",
                "eps_yoy_growth_status": "resolved",
            }
        ]
    )

    result = _restate_candidate_event_outcomes(new_events, path)

    assert result.loc[0, "eps_yoy_growth"] == 55.0
    assert result.loc[0, "eps_yoy_growth_source"] == "SEC"
    assert result.loc[0, "entry_open"] == 100.0
    assert result.loc[0, "week1_close_return_pct"] == 5.0
    assert result.loc[0, "stop_8_hit_ever"] == False


def test_explicit_e0_eligibility_uses_corrected_eps_fact_only():
    row = pd.Series(
        {
            "signal": True,
            "ibd_candidate_rule": "ceiling",
            "ibd_entry_status": "ACTIONABLE",
            "ibd_entry_close_position": 0.8,
            "ibd_entry_breakout_range_ratio": 0.4,
            "current_vs_ibd_candidate_pct": 2.0,
            "industry": "Software",
        }
    )
    assert _explicit_eligible(row, 12.0) is True
    assert _explicit_eligible(row, None) is False
    missing_industry = row.copy()
    missing_industry["industry"] = pd.NA
    assert _explicit_eligible(missing_industry, 12.0) is False
