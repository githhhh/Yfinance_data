from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.blind_rule_discovery.experiment import (
    MAX_RESEARCH_SECONDS,
    build_blind_dataset,
    build_feature_dossier,
    build_market_context,
    chronological_holdout,
    evaluate_candidate_path,
    run_research_command,
    write_agent_workspace,
)


def prices(rows):
    df = pd.DataFrame(rows, columns=["date", "Open", "High", "Low", "Close"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def daily_path(*, stop_first=False, target_first=False):
    dates = pd.bdate_range("2026-01-05", periods=61)
    rows = []
    for i, date in enumerate(dates):
        high, low, close = 105.0, 95.0, 100.0
        if stop_first and i == 2:
            low = 91.0
        if target_first and i == 2:
            high = 121.0
        if stop_first and i == 5:
            high = 121.0
        if target_first and i == 5:
            low = 91.0
        rows.append((date, 100.0, high, low, close))
    return prices(rows)


def spy_path():
    dates = pd.bdate_range("2026-01-05", periods=61)
    return prices([(d, 100.0, 101.0, 99.0, 100.0 + i * 0.1) for i, d in enumerate(dates)])


def test_entry_is_next_session_open_and_stop_then_rally_is_not_clean_winner():
    result = evaluate_candidate_path(daily_path(stop_first=True), "2026-01-02", spy_prices=spy_path())
    assert result["entry_date"] == pd.Timestamp("2026-01-05")
    assert result["entry_price"] == 100.0
    assert result["label"] == "stop_out_then_winner"


def test_target_before_stop_is_clean_winner():
    result = evaluate_candidate_path(daily_path(target_first=True), "2026-01-02", spy_prices=spy_path())
    assert result["label"] == "clean_winner"


def test_same_bar_stop_and_target_is_ambiguous_not_winner():
    frame = daily_path()
    frame.loc[2, "High"] = 121.0
    frame.loc[2, "Low"] = 91.0
    result = evaluate_candidate_path(frame, "2026-01-02", spy_prices=spy_path())
    assert result["label"] == "ambiguous_path"


def test_agent_surface_is_anonymous_and_excludes_prior_decision_artifacts(tmp_path: Path):
    candidates = pd.DataFrame(
        {
            "code": ["AAA", "BBB"],
            "snapshot_date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
            "raw_numeric": [1.0, 2.0],
            "pullback_v_is_dry": ["True", "False"],
            "rank_C_continuous": [1, 2],
            "some_score": [99.0, 1.0],
            "ibd_candidate_rule": ["x", "y"],
            "is_priority": [True, False],
        }
    )
    agent, feature_map, reviewer = build_blind_dataset(
        candidates,
        {"AAA": daily_path(target_first=True), "BBB": daily_path(stop_first=True)},
        spy_path(),
    )
    assert len(agent) == 2
    allowed = {
        "sample_id",
        "period_month",
        "period_quarter",
        "Y_label",
        "Y_stopped_out",
        "Y_4w_return",
        "Y_8w_return",
        "Y_12w_return",
        "Y_4w_excess",
        "Y_8w_excess",
        "Y_12w_excess",
        "Y_mae_12w",
        "Y_mfe_12w",
    }
    assert all(c.startswith("X") or c in allowed for c in agent.columns)
    mapped_names = set(feature_map.values())
    assert "raw_numeric" in mapped_names
    assert "pullback_v_is_dry" in mapped_names
    assert "rank_C_continuous" not in mapped_names
    assert "some_score" not in mapped_names
    assert "is_priority" not in mapped_names
    workspace = write_agent_workspace(agent, tmp_path)
    prompt = (workspace / "prompt.md").read_text()
    surface = prompt + (workspace / "samples.csv").read_text()
    assert "B0" not in surface
    assert "rank_C_continuous" not in surface
    assert "pullback_v_is_dry" not in surface
    assert not reviewer.empty


def test_dossier_is_monthly_and_quarterly_distributional_not_average():
    candidates = pd.DataFrame(
        {
            "code": ["AAA", "BBB"],
            "snapshot_date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
            "feature": [1.0, 2.0],
        }
    )
    agent, _, reviewer = build_blind_dataset(
        candidates,
        {"AAA": daily_path(target_first=True), "BBB": daily_path(stop_first=True)},
        spy_path(),
    )
    market = build_market_context(reviewer, spy_path())
    summary, profile = build_feature_dossier(agent, market)
    assert set(summary["granularity"]) == {"month", "quarter"}
    assert set(profile["granularity"]) == {"month", "quarter"}
    joined = " ".join(summary.columns).lower() + " " + " ".join(profile.columns).lower()
    assert "mean" not in joined
    assert "avg" not in joined
    assert "p50" in joined
    assert "spy_period_return" in summary.columns


def test_research_timeout_is_hard_capped_to_one_hour(monkeypatch, tmp_path: Path):
    seen = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)
    run_research_command(["research"], tmp_path, timeout_seconds=99999)
    assert seen["timeout"] == MAX_RESEARCH_SECONDS == 3600


def test_chronological_holdout_seals_latest_quarters():
    rows = []
    quarters = ["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2"]
    for i, quarter in enumerate(quarters):
        rows.append(
            {
                "sample_id": str(i),
                "period_quarter": quarter,
                "period_month": "x",
                "Y_label": "loser",
            }
        )
    df = pd.DataFrame(rows)
    discovery, holdout, sealed = chronological_holdout(df, holdout_quarters=2)
    assert sealed == ["2025Q1", "2025Q2"]
    assert set(holdout["period_quarter"]) == set(sealed)
    assert not set(discovery["period_quarter"]) & set(sealed)


def test_agent_workspace_can_include_only_public_market_context(tmp_path: Path):
    agent = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "period_month": "2026-01",
                "period_quarter": "2026Q1",
                "Y_label": "loser",
                "Y_stopped_out": 1,
                "Y_4w_return": 0.0,
                "Y_8w_return": 0.0,
                "Y_12w_return": 0.0,
                "Y_4w_excess": -0.01,
                "Y_8w_excess": -0.02,
                "Y_12w_excess": -0.03,
                "Y_mae_12w": -0.10,
                "Y_mfe_12w": 0.05,
                "X001": 1.0,
            }
        ]
    )
    market = pd.DataFrame(
        [
            {
                "granularity": "month",
                "period": "2026-01",
                "spy_period_return": 0.02,
                "spy_period_max_drawdown": -0.03,
                "spy_sessions": 20,
            }
        ]
    )
    workspace = write_agent_workspace(agent, tmp_path, market)
    public = pd.read_csv(workspace / "market_context.csv")
    assert "M_market_return" in public.columns
    assert "spy_period_return" not in public.columns
