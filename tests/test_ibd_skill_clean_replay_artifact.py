from __future__ import annotations

import pandas as pd

from backtest.ibd_skill_iteration.deterministic_prescreen import (
    build_prescreen_artifact,
    render_prescreen_artifact_markdown,
)


def test_clean_replay_pool_can_render_deterministic_prescreen_artifact():
    pool = pd.read_csv("backtest/ibd_skill_replay_pools/2026-07-24/breakout_follow_pool.csv")

    artifact = build_prescreen_artifact(pool, snapshot_date="2026-07-24", version="v3")
    markdown = render_prescreen_artifact_markdown(artifact)

    assert artifact["deterministic_contract"]["ordered_lists_are_authoritative"] is True
    assert len(artifact["priority_top3"]) <= 3
    assert artifact["alpha_radar_top5"]
    assert artifact["pullback_scout_top10"]
    assert "Do not reorder these rows" in markdown
