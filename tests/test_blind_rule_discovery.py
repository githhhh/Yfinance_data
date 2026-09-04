from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

from backtest.blind_rule_discovery.experiment import (
    DISCOVERY_FEATURE_ALLOWLIST,
    MAX_RESEARCH_SECONDS,
    apply_rule,
    build_blind_dataset,
    build_feature_dossier,
    evaluate_candidate_path,
    evaluate_frozen_rule,
    freeze_rule_artifact,
    point_in_time_market_features,
    purged_chronological_holdout,
    load_price_pickle,
    restrict_to_mature_outcome_quarters,
    run_research_command,
    validate_rule_artifact,
    validate_rule_support,
    write_agent_workspace,
)


def prices(rows):
    df = pd.DataFrame(rows, columns=["date", "Open", "High", "Low", "Close"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def daily_path(*, stop_first=False, target_first=False, neither=False, periods=100):
    dates = pd.bdate_range("2024-01-02", periods=periods)
    rows = []
    for i, date in enumerate(dates):
        high, low, close = (105.0, 95.0, 100.0)
        if neither:
            high, low = 110.0, 94.0
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


def spy_path(periods=700):
    dates = pd.bdate_range("2023-01-02", periods=periods)
    return prices([(d, 100.0, 101.0 + i * 0.01, 99.0, 100.0 + i * 0.05) for i, d in enumerate(dates)])


def candidate_row(code, snapshot_date, value=1.0):
    return {
        "code": code,
        "snapshot_date": pd.Timestamp(snapshot_date),
        "pullback_v_is_dry": True,
        "ibd_entry_volume_ratio": value,
        "ibd_trigger_price": 100.0,
        "ibd_candidate_price": 100.0,
        "C_continuous": 99.0,
        "rank_C_continuous": 1,
        "ibd_entry_status": "ACTIONABLE",
        "is_priority": True,
    }


def shifted_path(start, *, mode="winner", periods=100):
    dates = pd.bdate_range(start, periods=periods)
    rows = []
    for i, date in enumerate(dates):
        high, low, close = 105.0, 95.0, 100.0
        if mode == "winner" and i == 2:
            high = 121.0
        if mode == "loser" and i == 2:
            low = 91.0
        if mode == "recovery":
            if i == 2:
                low = 91.0
            if i == 5:
                high = 121.0
        rows.append((date, 100.0, high, low, close))
    return prices(rows)


def test_real_entry_and_stop_then_rally_is_loser():
    result = evaluate_candidate_path(daily_path(stop_first=True), "2024-01-01", trigger_price=100.0, spy_prices=spy_path())
    assert result["entry_date"] == pd.Timestamp("2024-01-02")
    assert result["entry_price"] == 100.0
    assert result["label"] == "stop_out_then_winner"
    assert result["primary"] == "loser"


def test_target_before_stop_is_winner():
    result = evaluate_candidate_path(daily_path(target_first=True), "2024-01-01", trigger_price=100.0, spy_prices=spy_path())
    assert result["label"] == "clean_winner"
    assert result["primary"] == "winner"


def test_neither_boundary_is_unresolved_not_loser():
    result = evaluate_candidate_path(daily_path(neither=True), "2024-01-01", trigger_price=100.0, spy_prices=spy_path())
    assert result["label"] == "unresolved"
    assert result["primary"] == "unresolved"


def test_same_bar_is_ambiguous():
    frame = daily_path()
    frame.loc[2, "High"] = 121.0
    frame.loc[2, "Low"] = 91.0
    result = evaluate_candidate_path(frame, "2024-01-01", trigger_price=100.0, spy_prices=spy_path())
    assert result["label"] == "ambiguous_path"
    assert result["primary"] == "ambiguous"


def test_explicit_allowlist_blocks_b0_artifacts_and_anonymizes():
    candidates = pd.DataFrame([candidate_row("AAA", "2024-01-01")])
    agent, feature_map, reviewer = build_blind_dataset(
        candidates,
        {"AAA": daily_path(target_first=True)},
        spy_path(),
    )
    mapped = set(feature_map.values())
    assert "pullback_v_is_dry" in mapped
    assert "ibd_entry_volume_ratio" in mapped
    assert "C_continuous" not in mapped
    assert "rank_C_continuous" not in mapped
    assert "ibd_entry_status" not in mapped
    assert "is_priority" not in mapped
    assert all(name in DISCOVERY_FEATURE_ALLOWLIST for name in mapped)
    assert all(c.startswith(("X", "M_")) or c.startswith("Y_") or c in {"sample_id", "period_month", "period_quarter"} for c in agent.columns)
    assert not reviewer.empty


def test_market_features_are_point_in_time_only():
    spy = spy_path()
    a = point_in_time_market_features(spy, "2023-08-01")
    changed = spy.copy()
    changed.loc[changed["date"] > pd.Timestamp("2023-08-01"), "Close"] = 9999.0
    b = point_in_time_market_features(changed, "2023-08-01")
    assert a == b
    assert "M_4w_return" in a and "M_12w_drawdown" in a


def test_dossier_is_monthly_quarterly_and_no_means():
    candidates = pd.DataFrame([
        candidate_row("AAA", "2024-01-01", 1.0),
        candidate_row("BBB", "2024-01-01", 2.0),
    ])
    agent, _, _ = build_blind_dataset(
        candidates,
        {"AAA": daily_path(target_first=True), "BBB": daily_path(stop_first=True)},
        spy_path(),
    )
    summary, profile = build_feature_dossier(agent)
    assert set(summary["granularity"]) == {"month", "quarter"}
    assert set(profile["granularity"]) == {"month", "quarter"}
    joined = " ".join(summary.columns).lower() + " " + " ".join(profile.columns).lower()
    assert "mean" not in joined and "avg" not in joined
    assert "p50" in joined


def test_purged_holdout_removes_training_outcomes_crossing_holdout_start():
    rows = []
    review = []
    quarters = ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1", "2024Q2", "2024Q3", "2024Q4"]
    for i, q in enumerate(quarters):
        period = pd.Period(q, freq="Q")
        sid = f"s{i}"
        entry = period.start_time + pd.Timedelta(days=10)
        exit12 = entry + pd.Timedelta(days=84)
        rows.append({"sample_id": sid, "period_quarter": q, "period_month": entry.strftime("%Y-%m"), "Y_primary": "loser"})
        review.append({"sample_id": sid, "exit_date_12w": exit12})
    review[5]["exit_date_12w"] = pd.Timestamp("2024-07-15")
    agent = pd.DataFrame(rows)
    reviewer = pd.DataFrame(review)
    discovery, embargo, holdout, sealed, start = purged_chronological_holdout(agent, reviewer, holdout_quarters=2)
    assert sealed == ["2024Q3", "2024Q4"]
    assert start == pd.Timestamp("2024-07-01")
    assert "s5" in set(embargo["sample_id"])
    assert "s5" not in set(discovery["sample_id"])
    assert set(holdout["period_quarter"]) == set(sealed)


def test_workspace_contains_no_private_artifacts(tmp_path: Path):
    agent = pd.DataFrame([{
        "sample_id": "a", "period_month": "2024-01", "period_quarter": "2024Q1",
        "Y_label": "unresolved", "Y_primary": "unresolved", "Y_stopped_out": 0,
        "Y_recovered_after_stop": 0, "Y_4w_return": 0.0, "Y_8w_return": 0.0, "Y_12w_return": 0.0,
        "Y_4w_excess": 0.0, "Y_8w_excess": 0.0, "Y_12w_excess": 0.0,
        "Y_mae_12w": -0.01, "Y_mfe_12w": 0.01, "M_4w_return": 0.0, "X001": 1.0,
    }])
    workspace = write_agent_workspace(agent, tmp_path)
    assert {p.name for p in workspace.iterdir()} == {"samples.csv", "prompt.md"}
    surface = (workspace / "samples.csv").read_text() + (workspace / "prompt.md").read_text()
    assert "C_continuous" not in surface and "B0" not in surface


def test_research_refuses_unsandboxed_and_caps_timeout(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "agent_workspace"
    workspace.mkdir()
    (workspace / "samples.csv").write_text("sample_id,X001\na,1\n")
    (workspace / "prompt.md").write_text("x")
    with pytest.raises(RuntimeError, match="physical sandbox"):
        run_research_command(["research"], workspace, sandbox_prefix=None)

    seen = {}
    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen.update(kwargs)
        class Result:
            returncode = 0
            stdout = ""
            stderr = ""
        return Result()
    monkeypatch.setattr("subprocess.run", fake_run)
    run_research_command(["research"], workspace, sandbox_prefix=["sandbox-exec", "{workspace}", "--"], timeout_seconds=99999)
    assert seen["timeout"] == MAX_RESEARCH_SECONDS == 3600
    assert seen["cmd"][0] == "sandbox-exec"
    assert "blind_rule_agent_" in seen["cmd"][1]


def test_rule_validator_rejects_y_date_unknown_and_excess_complexity(tmp_path: Path):
    cols = ["X001", "M_4w_return", "Y_primary", "period_quarter"]
    bad = tmp_path / "rule.json"
    bad.write_text(json.dumps({"version": 1, "clauses": [{"all": [{"feature": "Y_primary", "op": ">=", "threshold": 1}]}]}))
    with pytest.raises(ValueError, match="forbidden"):
        validate_rule_artifact(bad, cols)
    bad.write_text(json.dumps({"version": 1, "clauses": [{"all": [{"feature": "period_quarter", "op": ">=", "threshold": 1}]}]}))
    with pytest.raises(ValueError, match="forbidden"):
        validate_rule_artifact(bad, cols)
    bad.write_text(json.dumps({"version": 1, "clauses": [{"all": [{"feature": "X999", "op": ">=", "threshold": 1}]}]}))
    with pytest.raises(ValueError, match="unknown"):
        validate_rule_artifact(bad, cols)


def test_valid_rule_freezes_and_holdout_evaluates(tmp_path: Path):
    rule_path = tmp_path / "rule.json"
    rule = {"version": 1, "clauses": [{"all": [{"feature": "X001", "op": ">=", "threshold": 1.5}]}]}
    rule_path.write_text(json.dumps(rule))
    manifest = freeze_rule_artifact(rule_path, tmp_path, agent_columns=["X001", "Y_primary"])
    assert manifest["validated_condition_count"] == 1
    df = pd.DataFrame([
        {"period_quarter": "2025Q1", "X001": 2.0, "Y_primary": "winner", "Y_12w_excess": .2, "Y_mae_12w": -.03, "Y_mfe_12w": .25},
        {"period_quarter": "2025Q1", "X001": 1.0, "Y_primary": "loser", "Y_12w_excess": -.1, "Y_mae_12w": -.1, "Y_mfe_12w": .05},
    ])
    mask = apply_rule(rule, df)
    assert mask.tolist() == [True, False]
    report = evaluate_frozen_rule(rule, df)
    all_row = report.loc[report["period"] == "ALL"].iloc[0]
    assert all_row["selected_n"] == 1
    assert all_row["winner_n"] == 1
    assert all_row["resolved_winner_rate"] == 1.0


def test_partial_outcome_quarter_is_excluded():
    spy = spy_path(periods=700)
    cutoff = pd.Timestamp(spy.iloc[-61]["date"])
    mature_q = (cutoff - pd.offsets.QuarterEnd(1)).to_period("Q")
    immature_q = cutoff.to_period("Q")
    candidates = pd.DataFrame([
        candidate_row("AAA", mature_q.start_time + pd.Timedelta(days=5)),
        candidate_row("BBB", immature_q.start_time + pd.Timedelta(days=5)),
    ])
    mature, excluded, maturity_cutoff = restrict_to_mature_outcome_quarters(candidates, spy, minimum_sessions=60)
    assert maturity_cutoff == cutoff.normalize()
    assert len(mature) == 1
    assert len(excluded) == 1


def test_input_pool_order_does_not_control_sample_ids():
    rows = [candidate_row("AAA", "2024-01-01"), candidate_row("BBB", "2024-01-01")]
    price_map = {"AAA": daily_path(target_first=True), "BBB": daily_path(stop_first=True)}
    a, _, ra = build_blind_dataset(pd.DataFrame(rows), price_map, spy_path())
    b, _, rb = build_blind_dataset(pd.DataFrame(list(reversed(rows))), price_map, spy_path())
    map_a = dict(zip(ra["code"], ra["sample_id"]))
    map_b = dict(zip(rb["code"], rb["sample_id"]))
    assert map_a == map_b
    assert set(a["sample_id"]) == set(b["sample_id"])


def test_missing_price_is_recorded_as_censored_attrition():
    candidates = pd.DataFrame([candidate_row("MISSING", "2024-01-01")])
    agent, _, reviewer = build_blind_dataset(candidates, {}, spy_path())
    assert agent.empty
    assert reviewer.iloc[0]["reason"] == "missing_price_data"
    assert reviewer.iloc[0]["usable"] == False


def test_rule_support_rejects_tiny_or_single_period_rule():
    rule = {"version": 1, "clauses": [{"all": [{"feature": "X001", "op": ">=", "threshold": 1.0}]}]}
    df = pd.DataFrame([
        {"X001": 1.0, "period_quarter": "2024Q1", "Y_primary": "winner"} for _ in range(5)
    ])
    with pytest.raises(ValueError, match="selects only"):
        validate_rule_support(rule, df, min_selected=20, min_active_quarters=3)


def test_holdout_report_includes_market_context_and_universe_baseline():
    rule = {"version": 1, "clauses": [{"all": [{"feature": "X001", "op": ">=", "threshold": 1.5}]}]}
    df = pd.DataFrame([
        {"period_quarter": "2025Q1", "X001": 2.0, "Y_primary": "winner", "Y_12w_excess": .2, "Y_mae_12w": -.03, "Y_mfe_12w": .25},
        {"period_quarter": "2025Q1", "X001": 1.0, "Y_primary": "loser", "Y_12w_excess": -.1, "Y_mae_12w": -.1, "Y_mfe_12w": .05},
    ])
    market = pd.DataFrame([{
        "granularity": "quarter", "period": "2025Q1", "spy_period_return": .04, "spy_period_max_drawdown": -.06
    }])
    report = evaluate_frozen_rule(rule, df, market_context=market)
    q = report.loc[report["period"] == "2025Q1"].iloc[0]
    assert q["spy_period_return"] == .04
    assert q["universe_resolved_winner_rate"] == .5
    assert q["resolved_winner_rate"] == 1.0


def test_adj_close_factor_handles_datetime_index_and_split(tmp_path: Path):
    idx = pd.bdate_range("2024-01-02", periods=3)
    df = pd.DataFrame({
        "Open": [100.0, 100.0, 50.0],
        "High": [102.0, 102.0, 51.0],
        "Low": [98.0, 98.0, 49.0],
        "Close": [100.0, 100.0, 50.0],
        "Adj Close": [50.0, 50.0, 50.0],
    }, index=idx)
    path = tmp_path / "px.pkl"
    with path.open("wb") as f:
        pickle.dump({"AAA": df}, f)
    loaded = load_price_pickle(path, require_adjusted=True)["AAA"]
    assert loaded.attrs["price_adjustment_mode"] == "adj_close_factor"
    assert loaded["Close"].tolist() == [50.0, 50.0, 50.0]
    assert loaded["Open"].tolist() == [50.0, 50.0, 50.0]


def test_unadjusted_price_pickle_is_rejected_when_required(tmp_path: Path):
    idx = pd.bdate_range("2024-01-02", periods=3)
    df = pd.DataFrame({"Open": [1,1,1], "High": [1,1,1], "Low": [1,1,1], "Close": [1,1,1]}, index=idx)
    path = tmp_path / "px.pkl"
    with path.open("wb") as f:
        pickle.dump({"AAA": df}, f)
    with pytest.raises(ValueError, match="split-unsafe"):
        load_price_pickle(path, require_adjusted=True)


def test_consumed_holdout_blocks_same_output_root(tmp_path: Path):
    from backtest.blind_rule_discovery.runner import _prepare_output_root
    (tmp_path / "holdout_consumed.json").write_text("{}")
    with pytest.raises(RuntimeError, match="already consumed"):
        _prepare_output_root(tmp_path)


def test_true_buy_point_cross_from_below_uses_trigger_not_future_pullback():
    dates = pd.bdate_range("2024-01-02", periods=70)
    rows = []
    for i, d in enumerate(dates):
        o, h, l, c = 98.0, 99.0, 97.0, 98.0
        if i == 1:
            h, c = 101.0, 100.5
        if i == 3:
            h = 121.0
        rows.append((d, o, h, l, c))
    result = evaluate_candidate_path(prices(rows), "2024-01-01", trigger_price=100.0, spy_prices=spy_path())
    assert result["entry_date"] == pd.Timestamp(dates[1])
    assert result["entry_price"] == 100.0
    assert result["entry_method"] == "intraday_trigger"


def test_extended_gap_is_not_bought_on_same_day_pullback():
    dates = pd.bdate_range("2024-01-02", periods=70)
    rows = []
    for i, d in enumerate(dates):
        if i == 0:
            rows.append((d, 108.0, 110.0, 101.0, 103.0))
        else:
            rows.append((d, 108.0, 110.0, 106.0, 108.0))
    result = evaluate_candidate_path(prices(rows), "2024-01-01", trigger_price=100.0, spy_prices=spy_path())
    assert result["label"] == "censored"
    assert result["reason"] == "no_entry_within_buy_zone_window"


def test_missing_trigger_is_censored_not_fallback_open():
    result = evaluate_candidate_path(daily_path(target_first=True), "2024-01-01", trigger_price=None, spy_prices=spy_path())
    assert result["label"] == "censored"
    assert result["reason"] == "missing_trigger_price"


def test_intraday_entry_day_stop_order_is_ambiguous():
    dates = pd.bdate_range("2024-01-02", periods=70)
    rows = [(dates[0], 98.0, 101.0, 90.0, 100.0)]
    rows += [(d, 100.0, 105.0, 95.0, 100.0) for d in dates[1:]]
    result = evaluate_candidate_path(prices(rows), "2024-01-01", trigger_price=100.0, spy_prices=spy_path())
    assert result["label"] == "ambiguous_path"
    assert result["reason"] == "entry_day_stop_order_unknown"


def test_research_rejects_fake_or_workspace_agnostic_sandbox(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "agent_workspace"
    workspace.mkdir()
    (workspace / "samples.csv").write_text("sample_id,X001\na,1\n")
    (workspace / "prompt.md").write_text("x")
    with pytest.raises(RuntimeError, match="unsupported sandbox"):
        run_research_command(["research"], workspace, sandbox_prefix=["env", "{workspace}"])
    with pytest.raises(RuntimeError, match=r"reference \{workspace\}"):
        run_research_command(["research"], workspace, sandbox_prefix=["sandbox-exec", "-p", "deny default"])
