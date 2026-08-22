import pandas as pd

from backtest.ibd_skill_iteration.core import rank_reasoning_candidates
from backtest.ibd_weekly_signal_oracle_eval.evaluate_weekly_signal_oracle import VARIANTS, variant_items


def test_signal_shadow_variant_selects_from_all_signal_statuses():
    pool = pd.DataFrame(
        [
            _row("EXT1", "EXTENDED", 6.0, 2.2, 1.8),
            _row("ACT1", "ACTIONABLE", 1.0, 1.8, 1.5),
            _row("UNC1", "UNCONFIRMED", 1.5, 1.7, 1.4),
        ]
    )
    ranked = rank_reasoning_candidates(pool, universe="review", version="v3")

    selected = variant_items(ranked, pool, enabled=True, cfg=VARIANTS["signal_shadow_top3"])

    assert len(selected) == 3
    assert {item.entry_status for item in selected} == {"ACTIONABLE", "EXTENDED", "UNCONFIRMED"}


def test_research_fresh_proximity_variant_prefers_imax_style_buy_point_closeness():
    pool = pd.DataFrame(
        [
            _row("BLFS", "ACTIONABLE", 1.2, 11.1, 3.1),
            _row("IMAX", "ACTIONABLE", 0.5, 5.8, 2.2),
            _row("NWFL", "ACTIONABLE", 2.7, 1.9, 1.2),
        ]
    )
    ranked = rank_reasoning_candidates(pool, universe="review", version="v3")

    selected = variant_items(ranked, pool, enabled=True, cfg=VARIANTS["research_fresh_demand_proximity_first"])

    assert [item.code for item in selected[:2]] == ["IMAX", "BLFS"]


def test_research_pullback_interleave_can_promote_high_quality_constructive_pullback():
    pool = pd.DataFrame(
        [
            _row("CEIL1", "ACTIONABLE", 1.2, 2.0, 1.4, rule="ceiling"),
            _row("PULL1", "ACTIONABLE", 1.0, 2.2, 1.8, rule="ceiling_pullback", dry=True),
            _row("CEIL2", "ACTIONABLE", 1.5, 1.8, 1.5, rule="ceiling"),
        ]
    )
    ranked = rank_reasoning_candidates(pool, universe="review", version="v3")

    selected = variant_items(ranked, pool, enabled=True, cfg=VARIANTS["research_pullback_vcp_lane_interleave"])

    assert "PULL1" in [item.code for item in selected[:2]]


def _row(
    code: str,
    status: str,
    current_vs_buy: float,
    entry_volume: float,
    weekly_volume: float,
    *,
    rule: str = "ceiling",
    dry: bool | None = None,
) -> dict[str, object]:
    return {
        "snapshot_date": "2026-07-24",
        "code": code,
        "signal": True,
        "ibd_candidate_rule": rule,
        "ibd_entry_status": status,
        "current_vs_ibd_candidate_pct": current_vs_buy,
        "ibd_entry_volume_ratio": entry_volume,
        "ibd_entry_close_position": 0.9,
        "ibd_entry_breakout_range_ratio": 0.7,
        "dist_to_52w_high_pct": -1.0,
        "volume_ratio": weekly_volume,
        "eps_yoy_growth": 30.0,
        "industry": f"{code} Industry",
        "pullback_v_is_dry": dry,
    }
