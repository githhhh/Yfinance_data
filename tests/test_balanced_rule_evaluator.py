import pandas as pd

from backtest.rd_agent_research_bench.balanced_rule_evaluator import (
    SelectorConfig,
    evaluate_candidate,
    select_by_config,
)


def test_base_signal_marks_pullback_dry_not_applicable_not_fail():
    row = pd.Series(_row("AAA", rule="ceiling", dry=False))

    evaluated = evaluate_candidate(row, row_index=0)

    dry_check = evaluated["checks"]["pullback_dry"]
    assert dry_check["state"] == "NOT_APPLICABLE"
    assert "pullback_not_dry" not in evaluated["risk_flags"]


def test_pullback_dry_modes_compare_hard_minor_and_drop():
    frame = pd.DataFrame(
        [
            _row("DRY", rule="ceiling_pullback", dry=True),
            _row("WET", rule="ceiling_pullback", dry=False),
        ]
    )

    hard = select_by_config(frame, SelectorConfig(name="hard", pullback_dry_mode="hard", top_n=3, industry_cover=False))
    minor = select_by_config(frame, SelectorConfig(name="minor", pullback_dry_mode="minor", top_n=3, industry_cover=False))
    drop = select_by_config(frame, SelectorConfig(name="drop", pullback_dry_mode="drop", top_n=3, industry_cover=False))

    assert list(hard["code"]) == ["DRY"]
    assert set(minor["code"]) == {"DRY", "WET"}
    assert set(drop["code"]) == {"DRY", "WET"}
    wet_minor = minor[minor["code"].eq("WET")].iloc[0]
    wet_drop = drop[drop["code"].eq("WET")].iloc[0]
    assert "pullback_not_dry" in wet_minor["risk_flags"]
    assert "pullback_not_dry" not in wet_drop["risk_flags"]


def _row(code, *, rule="ceiling", dry=True):
    return {
        "snapshot_date": "2026-01-02",
        "code": code,
        "signal": True,
        "ibd_entry_status": "ACTIONABLE",
        "ibd_entry_valid": 1,
        "ibd_candidate_rule": rule,
        "ibd_candidate_price": 100.0,
        "latest_close": 101.0,
        "current_vs_ibd_candidate_pct": 1.0,
        "ibd_entry_volume_ratio": 2.0,
        "ibd_entry_close_position": 0.9,
        "ibd_entry_breakout_range_ratio": 0.7,
        "volume_ratio": 1.5,
        "dist_to_52w_high_pct": -2.0,
        "pullback_v_is_dry": dry,
        "eps_yoy_growth": 30.0,
        "industry": "Software",
        "signal_source": "ceiling_breakout",
    }
