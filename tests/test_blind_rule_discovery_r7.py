import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def core():
    return importlib.import_module("backtest.blind_rule_discovery.r7_economics")


def rows():
    return pd.DataFrame({
        "code": list("ABCD"), "snapshot_date": pd.to_datetime(["2024-04-05"] * 4),
        "entry_date": pd.to_datetime(["2024-04-08"] * 4),
        "exit_date_w4": pd.to_datetime(["2024-05-03"] * 4),
        "return_w4": [-.10, .30, .04, -.02],
        "stop_first_3w": [1, 0, 0, 0], "fast_winner_3w": [0, 1, 0, 0],
        "unresolved_3w": [0, 0, 1, 0], "ambiguous_3w": [0, 0, 0, 1],
    })


def test_cash_accounting_decomposition_includes_unresolved_and_ambiguous():
    metrics, parts = core().account_week(rows(), np.array([1, 1, 0, 0], bool), "w4", 20)
    assert metrics["baseline_net"] == pytest.approx(.053)
    assert metrics["veto_cash_net"] == pytest.approx(.004)
    assert metrics["random_cash_net"] == pytest.approx(.0265)
    assert metrics["incremental_vs_random"] == pytest.approx(-.0225)
    assert metrics["avoided_gross_loss"] == pytest.approx(.025)
    assert metrics["foregone_gross_gain"] == pytest.approx(.075)
    assert metrics["saved_cost"] == pytest.approx(.001)
    assert sum(p["cash_delta_contribution"] for p in parts) == pytest.approx(metrics["cash_delta"])
    assert {p["path_label"] for p in parts} == set(core().LABELS)
    assert sum(p["baseline_n"] for p in parts) == 4


def test_exact_matched_n_random_expectation_and_no_shrink_or_reinvestment():
    import itertools
    df = rows()
    observed = []
    for flagged in itertools.combinations(range(4), 2):
        mask = np.array([i in flagged for i in range(4)])
        m, _ = core().account_week(df, mask, "w4", 20)
        observed.append(m["veto_cash_net"])
    m, _ = core().account_week(df, np.array([1, 1, 0, 0], bool), "w4", 20)
    assert np.mean(observed) == pytest.approx(m["random_cash_net"])
    all_out, _ = core().account_week(df, np.ones(4, bool), "w4", 20)
    assert all_out["veto_cash_net"] == 0
    assert all_out["retained_mean"] is None
    none, _ = core().account_week(df, np.zeros(4, bool), "w4", 20)
    assert none["cash_delta"] == 0
    assert none["incremental_vs_random"] == 0


def test_nonoverlap_is_shared_baseline_w4_schedule_and_not_outcome_selected():
    df = pd.concat([rows(), rows().iloc[[0]]], ignore_index=True)
    df.loc[4, "snapshot_date"] = pd.Timestamp("2024-04-12")
    df.loc[4, "entry_date"] = pd.Timestamp("2024-04-15")
    df.loc[4, "exit_date_w4"] = pd.Timestamp("2024-05-10")
    admitted = core().baseline_admission(df)
    assert admitted.tolist() == [True, True, True, True, False]
    df.loc[4, "return_w4"] = 9.
    assert np.array_equal(admitted, core().baseline_admission(df))
    df.loc[4, "entry_date"] = pd.Timestamp("2024-05-03")
    assert not core().baseline_admission(df)[4]  # cannot reenter before closing same day
    df.loc[4, "entry_date"] = pd.Timestamp("2024-05-06")
    assert core().baseline_admission(df)[4]


def test_missing_features_not_claimed_to_be_safe_and_empty_quarters_visible():
    df = synthetic()
    calendar = [str(q) for q in pd.period_range("2022Q4", periods=9, freq="Q")]
    df.loc[df.snapshot_date >= "2024-04-01", "pullback_pct"] = np.nan
    frozen, flags = core().freeze_families(df, calendar)
    first = next(x for x in frozen if x["quarter"] == "2024Q2" and x["policy"] == "deep_pullback")
    assert first["rule"] is not None
    flagged, known = flags[("2024Q2", "deep_pullback")]
    assert not flagged.any() and not known.any()
    assert {x["quarter"] for x in frozen} == set(calendar[6:])


def test_family_thresholds_purged_and_do_not_see_test_returns_or_features():
    df = synthetic()
    calendar = [str(q) for q in pd.period_range("2022Q4", periods=8, freq="Q")]
    first, _ = core().freeze_families(df, calendar)
    changed = df.copy()
    future = changed.snapshot_date >= "2024-04-01"
    changed.loc[future, ["pullback_pct", "return_w4"]] = 9999.
    second, _ = core().freeze_families(changed, calendar)
    assert [r for r in first if r["quarter"] == "2024Q2"] == [r for r in second if r["quarter"] == "2024Q2"]
    assert set(r["policy"] for r in first) == set(core().FAMILIES)
    assert "high_entry_extension" not in core().FAMILIES


def synthetic(periods=8, weeks=4):
    result = []
    for quarter in pd.period_range("2022Q4", periods=periods, freq="Q"):
        for week in range(weeks):
            day = quarter.start_time + pd.Timedelta(days=(4-quarter.start_time.dayofweek) % 7 + 7 * week)
            for i in range(24):
                label = i % 4
                row = dict(code=f"S{i}", snapshot_date=day,
                    entry_date=day + pd.Timedelta(days=1),
                    pullback_pct=-float(i + 1), pct_above_ceiling=float(i),
                    current_vs_ibd_candidate_pct=float(i),
                    stop_first_3w=int(label == 0), fast_winner_3w=int(label == 1),
                    unresolved_3w=int(label == 2), ambiguous_3w=int(label == 3))
                for n in range(1, 5):
                    row[f"exit_date_w{n}"] = day + pd.Timedelta(days=7 * n)
                    row[f"return_w{n}"] = [-.10, .30, .04, -.02][label] * n / 4
                result.append(row)
    return pd.DataFrame(result)


def fixture_files(tmp_path, frame=None):
    from backtest.blind_rule_discovery.r6_runner import load_inputs
    from backtest.blind_rule_discovery.r6_stability import canonical, digest, fit_rule, purged_before, assess, apply_rule
    from backtest.blind_rule_discovery.pipeline_contract import sha256_file
    df = synthetic() if frame is None else frame
    samples, metadata = tmp_path / "samples.csv", tmp_path / "metadata.json"
    df.to_csv(samples, index=False)
    metadata.write_text(json.dumps({"replay_dataset_sha256": "a" * 64,
        "usable_trigger_entries": len(df),
        "entry_quarters": sorted(df.entry_date.dt.to_period("Q").astype(str).unique())}))
    projected, manifest = load_inputs(samples, metadata, sha256_file(samples))
    anchor = tmp_path / "r6"
    anchor.mkdir()
    rules, folds = [], []
    for q in manifest["calendar"][6:]:
        past = purged_before(projected, pd.Period(q, freq="Q").start_time)
        surface = past[["snapshot_date", "code", "exit_date_w3", *manifest["features"]]].to_csv(index=False)
        for source in ("rdagent", "simple"):
            proposal = dict(name="frozen", hypothesis="synthetic only",
                expression={"op": "raw", "feature": "pullback_pct"}, tail="low", quantile=.2)
            rule = fit_rule(proposal, past)
            rules.append(dict(fold=q, source=source, rule=rule, status="FROZEN",
                training_surface_sha256=hashlib.sha256(surface.encode()).hexdigest()))
            test = projected.loc[projected.snapshot_date.dt.to_period("Q").astype(str) == q]
            folds.append(dict(quarter=q, source=source, **assess(test, apply_rule(rule, past, test))))
    lock = dict(config={"min_train_quarters": 6}, test_feedback_used=False, rules=rules)
    (anchor / "frozen_rules.json").write_text(canonical(lock))
    (anchor / "summary.json").write_text(canonical(dict(frozen_sha256=digest(lock), folds=folds)))
    (anchor / "input_manifest.json").write_text(canonical(manifest))
    return samples, metadata, anchor


def test_end_to_end_offline_readonly_complete_matrix_and_repeatability(tmp_path, monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: pytest.fail("network forbidden"))
    runner = importlib.import_module("backtest.blind_rule_discovery.r7_runner")
    samples, metadata, anchor = fixture_files(tmp_path)
    protected = [samples, metadata, *anchor.iterdir()]
    before = {p: p.read_bytes() for p in protected}
    out = tmp_path / "run"
    runner.run(samples, metadata, anchor, out, cost_bps=20)
    assert before == {p: p.read_bytes() for p in protected}
    assert (out / "COMPLETE.json").exists()
    report = (out / "R7_REPORT.md").read_text()
    for phrase in ("KEEP PRODUCTION FROZEN", "NOT AN UNTOUCHED HOLDOUT", "not portfolio P&L",
                   "unresolved", "Matched-N", "W3 diagnostic", "cash", "RD-Agent calls: 0"):
        assert phrase in report
    summary = pd.read_csv(out / "economic_summary.csv")
    assert set(summary.policy) == set(core().FAMILIES) | {"r6_rdagent", "r6_simple"}
    assert set(summary.panel) == {"all_entries", "nonoverlap_w4"}
    assert set(summary.horizon) == {"w1", "w2", "w3", "w4"}
    assert len(summary) == 48
    assert (out / "decision_matrix.csv").exists()
    assert (out / "label_contributions.csv").exists()
    assert (out / "quarterly_summary.csv").exists()
    other = tmp_path / "repeat"
    runner.run(samples, metadata, anchor, other, cost_bps=20)
    for name in ("economic_summary.csv", "weekly_economics.csv", "decision_matrix.csv", "frozen_rules.json"):
        assert (out / name).read_bytes() == (other / name).read_bytes()
    with pytest.raises(ValueError, match="fresh"):
        runner.run(samples, metadata, anchor, out, cost_bps=20)


@pytest.mark.parametrize("problem", ["hash", "return", "temporal", "threshold", "cost"])
def test_invalid_inputs_fail_without_success_artifact(tmp_path, problem):
    runner = importlib.import_module("backtest.blind_rule_discovery.r7_runner")
    samples, metadata, anchor = fixture_files(tmp_path)
    if problem in {"return", "temporal"}:
        df = pd.read_csv(samples)
        df.loc[0, "return_w4" if problem == "return" else "exit_date_w4"] = np.nan if problem == "return" else "2020-01-01"
        df.to_csv(samples, index=False)
        mpath = anchor / "input_manifest.json"
        m = json.loads(mpath.read_text())
        m["samples_sha256"] = hashlib.sha256(samples.read_bytes()).hexdigest()
        mpath.write_text(json.dumps(m))
    elif problem == "hash":
        samples.write_text(samples.read_text() + "\n")
    elif problem == "threshold":
        from backtest.blind_rule_discovery.r6_stability import digest
        lockpath = anchor / "frozen_rules.json"
        lock = json.loads(lockpath.read_text())
        lock["rules"][0]["rule"]["threshold"] += 1
        lockpath.write_text(json.dumps(lock))
        summarypath = anchor / "summary.json"
        summary = json.loads(summarypath.read_text())
        summary["frozen_sha256"] = digest(lock)
        summarypath.write_text(json.dumps(summary))
    out = tmp_path / "bad"
    with pytest.raises(ValueError):
        runner.run(samples, metadata, anchor, out, cost_bps=-1 if problem == "cost" else 20)
    assert not (out / "COMPLETE.json").exists()


def test_equal_week_aggregation_not_candidate_weighted_and_uncertainty_reproducible():
    weekly = pd.DataFrame({"snapshot_date": pd.date_range("2024-01-05", periods=16, freq="7D"),
        "quarter": ["2024Q1"] * 13 + ["2024Q2"] * 3,
        "incremental_vs_random": [-.01] * 8 + [.03] * 8,
        "cash_delta": [.02] * 16, "n": [1] * 8 + [100] * 8})
    a = core().stability(weekly)
    assert a["incremental_mean"] == pytest.approx(.01)
    assert a == core().stability(weekly)
    assert a["calendar_quarters_observed"] == 2
    assert a["independent_confirmation"] is False


def test_leave_one_ticker_out_matches_bruteforce_without_refitting():
    first = rows().assign(_flagged=[True, True, False, False])
    second = rows().assign(_flagged=[False, True, False, False])
    second["snapshot_date"] += pd.Timedelta(days=7)
    df = pd.concat([first, second], ignore_index=True)
    result = core().ticker_sensitivity(df, "w4")
    for row in result:
        reduced = df.loc[df.code != row["code"]]
        actual = [core().account_week(g, g._flagged.to_numpy(), "w4", 0)[0]["incremental_vs_random"]
                  for _, g in reduced.groupby("snapshot_date")]
        assert row["incremental_without_ticker"] == pytest.approx(np.mean(actual))


def test_cost_cancels_from_exact_random_increment_but_not_cash_delta():
    mask = np.array([1, 0, 0, 0], bool)
    gross, _ = core().account_week(rows(), mask, "w4", 0)
    net, _ = core().account_week(rows(), mask, "w4", 35)
    assert net["incremental_vs_random"] == pytest.approx(gross["incremental_vs_random"])
    assert net["cash_delta"] - gross["cash_delta"] == pytest.approx(.25*.0035)


def test_preflight_never_writes_and_output_cannot_overlap_inputs(tmp_path):
    runner = importlib.import_module("backtest.blind_rule_discovery.r7_runner")
    samples, metadata, anchor = fixture_files(tmp_path)
    output = tmp_path / "preflight"
    assert runner.run(samples, metadata, anchor, output, cost_bps=20, preflight=True)["protocol"]["rdagent_calls"] == 0
    assert not output.exists()
    with pytest.raises(ValueError, match="disjoint"):
        runner.run(samples, metadata, anchor, anchor / "output", cost_bps=20)


def test_holiday_week_snapshots_use_week_identity_not_weekday():
    weekly = pd.DataFrame({"snapshot_date": pd.to_datetime(["2024-03-22", "2024-03-28", "2024-04-05"]),
        "quarter": ["2024Q1", "2024Q1", "2024Q2"], "incremental_vs_random": [.01, .02, .03]})
    assert core().stability(weekly)["incremental_mean"] == pytest.approx(.02)


def test_all_unknown_test_weeks_are_accounted_but_not_zero_alpha_evidence():
    df = synthetic()
    df.loc[df.snapshot_date >= "2024-04-01", "pullback_pct"] = np.nan
    calendar = [str(q) for q in pd.period_range("2022Q4", periods=9, freq="Q")]
    frozen, flags = core().freeze_families(df, calendar)
    out = core().evaluate(df, calendar, frozen, flags, 20)
    summary = out["economic_summary"].query("policy == 'deep_pullback'")
    assert summary.incremental_mean.isna().all()
    assert (summary.available_weeks == 0).all()
    assert (summary.observed_weeks > 0).all()
    q = out["quarterly_summary"].query("policy == 'deep_pullback'")
    assert set(q.status) == {"TEST_FEATURE_UNAVAILABLE", "EMPTY_TEST_QUARTER"}


def test_loader_accepts_upstream_holiday_snapshot_and_preserves_hashes(tmp_path):
    runner = importlib.import_module("backtest.blind_rule_discovery.r7_runner")
    df = synthetic()
    holiday = df.iloc[[0]].copy()
    holiday["snapshot_date"] = pd.Timestamp("2024-03-28")
    holiday["entry_date"] = pd.Timestamp("2024-04-01")
    for n in range(1, 5):
        holiday[f"exit_date_w{n}"] = pd.Timestamp("2024-04-01") + pd.Timedelta(days=7*n)
    df = pd.concat([df, holiday], ignore_index=True)
    samples, metadata, anchor = fixture_files(tmp_path, df)
    manifest = runner.run(samples, metadata, anchor, tmp_path / "no-write", cost_bps=20, preflight=True)
    assert manifest["samples_sha256"] == hashlib.sha256(samples.read_bytes()).hexdigest()


def test_ticker_sensitivity_singleton_week_is_missing_not_zero_return():
    df = rows().iloc[[0]].assign(_flagged=True)
    result = core().ticker_sensitivity(df, "w4")
    assert result[0]["incremental_without_ticker"] is None


def test_no_champion_or_data_rebuild_interface_and_completion_hashes(tmp_path):
    import inspect
    runner = importlib.import_module("backtest.blind_rule_discovery.r7_runner")
    assert set(inspect.signature(runner.run).parameters) == {"samples", "metadata", "r6_dir", "output", "cost_bps", "preflight"}
    samples, metadata, anchor = fixture_files(tmp_path)
    output = tmp_path / "audit"
    runner.run(samples, metadata, anchor, output, cost_bps=20)
    complete = json.loads((output / "COMPLETE.json").read_text())
    assert complete["rdagent_calls"] == 0 and not complete["production_change"]
    for name, sha in complete["output_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == sha


def test_favorable_synthetic_economics_still_never_authorize_production():
    df = synthetic(periods=10, weeks=12)
    for horizon in core().HORIZONS:
        df[f"return_{horizon}"] = np.where(df.pullback_pct <= -20, -.10, .08)
    calendar = [str(q) for q in pd.period_range("2022Q4", periods=10, freq="Q")]
    frozen, flags = core().freeze_families(df, calendar)
    tables = core().evaluate(df, calendar, frozen, flags, 20)
    result = tables["decision_matrix"].set_index("policy").loc["deep_pullback"]
    assert result.verdict == "HISTORICAL_ECONOMIC_DIRECTION"
    assert not result.production_change and not result.prospective_confirmation
    assert result.realized_stop_pnl == "NOT_IDENTIFIABLE_FROM_TERMINAL_RETURN_CSV"
