from __future__ import annotations

import pandas as pd

from backtest.ibd_skill_iteration.deterministic_prescreen import (
    build_prescreen_artifact,
    render_prescreen_artifact_markdown,
)
from backtest.ibd_skill_iteration.core import rank_signal_shadow_top3


def test_clean_replay_pool_can_render_deterministic_prescreen_artifact():
    pool = pd.read_csv("backtest/ibd_skill_replay_pools/2026-07-24/breakout_follow_pool.csv")

    artifact = build_prescreen_artifact(pool, snapshot_date="2026-07-24", version="v3")
    markdown = render_prescreen_artifact_markdown(artifact)

    assert artifact["deterministic_contract"]["ordered_lists_are_authoritative"] is True
    assert len(artifact["priority_top3"]) <= 3
    assert artifact["alpha_radar_top5"]
    assert artifact["pullback_scout_top10"]
    assert len(artifact["signal_shadow_top3"]) <= 3
    assert "Do not reorder these rows" in markdown
    assert "Signal Shadow Top 3" in markdown


def test_shadow_portfolio_top3_is_a_separate_audit_layer():
    pool = pd.DataFrame(
        [
            _signal_row("NEAR", "ACTIONABLE", 0.5, 2.0, 1.7, -1.0, "ceiling"),
            _signal_row("FAR", "ACTIONABLE", 4.0, 2.0, 1.7, -1.0, "ceiling"),
            _signal_row("LOWEPS", "ACTIONABLE", 0.4, 2.0, 1.7, -1.0, "ceiling") | {"eps_yoy_growth": 12.0},
            _signal_row("UNC", "UNCONFIRMED", 0.3, 2.0, 1.7, -1.0, "ceiling"),
        ]
    )

    artifact = build_prescreen_artifact(pool, snapshot_date="2026-07-24", version="v3")
    markdown = render_prescreen_artifact_markdown(artifact)

    shadow = artifact["shadow_portfolio_top3"]
    assert [row["code"] for row in shadow] == ["NEAR", "FAR"]
    assert all(row["entry_status"] == "ACTIONABLE" for row in shadow)
    assert all(row["final_group"] == "SHADOW_PORTFOLIO" for row in shadow)
    assert "Shadow Portfolio Top 3" in markdown
    assert "Formal Baseline" in markdown


def test_signal_shadow_top3_can_select_any_signal_status_without_expanding_priority():
    pool = pd.DataFrame(
        [
            _signal_row("EXT1", "EXTENDED", 6.0, 2.2, 1.8, -1.0, "ceiling"),
            _signal_row("UNC1", "UNCONFIRMED", 1.2, 1.8, 1.6, -2.0, "ceiling_pullback"),
            _signal_row("ACT1", "ACTIONABLE", 1.5, 1.7, 1.5, -3.0, "ceiling"),
            _signal_row("BAD1", "UNCONFIRMED", -1.0, 2.0, 1.7, -1.0, "ceiling"),
        ]
    )

    shadow = rank_signal_shadow_top3(pool, version="v3")

    assert len(shadow) <= 3
    assert {item.code for item in shadow} >= {"EXT1", "UNC1"}
    assert all(item.final_group == "SIGNAL_SHADOW" for item in shadow)
    assert all(item.entry_status in {"ACTIONABLE", "EXTENDED", "UNCONFIRMED"} for item in shadow)
    assert "BAD1" not in {item.code for item in shadow}


def _signal_row(
    code: str,
    status: str,
    current_vs_buy: float,
    entry_volume: float,
    weekly_volume: float,
    dist_to_high: float,
    rule: str,
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
        "dist_to_52w_high_pct": dist_to_high,
        "volume_ratio": weekly_volume,
        "eps_yoy_growth": 30.0,
        "pullback_v_is_dry": True,
        "industry": f"{code} Industry",
    }
