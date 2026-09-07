import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest.blind_rule_discovery.r6_stability import (
    Config, assess, discover, evaluate_expression, purged_before, validate_proposal,
    write_report,
)
from backtest.blind_rule_discovery.r6_agent import RDAgentProposer
from backtest.blind_rule_discovery.r6_runner import load_inputs
from backtest.blind_rule_discovery.pipeline_contract import sha256_file


def sample_frame():
    rows = []
    for q in pd.period_range("2022Q4", periods=10, freq="Q"):
        for week in range(4):
            date = q.start_time + pd.Timedelta(days=7 * week)
            for i in range(24):
                stop, fast = int(i < 8), int(i >= 18)
                rows.append(dict(
                    snapshot_date=date, code=f"S{i}", entry_date=date + pd.Timedelta(days=1),
                    exit_date_w3=date + pd.Timedelta(days=22), pullback_pct=float(-i),
                    volume_ratio=float(i % 5), ambiguous_3w=0, stop_first_3w=stop,
                    fast_winner_3w=fast, unresolved_3w=1-stop-fast,
                ))
    return pd.DataFrame(rows)


def proposal(feature="pullback_pct"):
    return dict(name="risk", hypothesis="Shallow pullback has synthetic stop risk.",
                expression={"op": "raw", "feature": feature}, tail="high", quantile=0.8)


def test_only_pit_expressions_allowed_and_complexity_bounded():
    validate_proposal(proposal(), ["pullback_pct"])
    for field in ("stop_first_3w", "M_8w_drawdown", "entry_extension_pct", "pick_order", "code"):
        with pytest.raises(ValueError):
            validate_proposal(proposal(field), ["pullback_pct", field])
    bad = proposal()
    bad["expression"] = {"op": "eval", "code": "1"}
    with pytest.raises(ValueError):
        validate_proposal(bad, ["pullback_pct"])


def test_ecdf_fit_uses_past_only_and_preserves_missing():
    fit = pd.DataFrame({"pullback_pct": [1., 2., 3.]})
    future = pd.DataFrame({"pullback_pct": [2., 1e9, np.nan]})
    actual = evaluate_expression({"op": "train_percentile", "feature": "pullback_pct"}, fit, future)
    assert actual[0] == pytest.approx(0.5)
    assert actual[1] == 1.0
    assert np.isnan(actual[2])


def test_purge_uses_label_availability_and_signal_time():
    df = sample_frame().iloc[:3].copy()
    df["snapshot_date"] = pd.to_datetime(["2023-03-01", "2023-03-02", "2023-04-01"])
    df["exit_date_w3"] = pd.to_datetime(["2023-03-30", "2023-04-01", "2023-03-30"])
    assert len(purged_before(df, pd.Timestamp("2023-04-01"))) == 1


def test_same_snapshot_contrast_avoids_quarter_composition_lift():
    df = pd.DataFrame(dict(snapshot_date=["2024-01-01"]*10+["2024-02-01"]*10,
        ambiguous_3w=[0]*20, stop_first_3w=[1]*10+[0]*10,
        fast_winner_3w=[0]*20, unresolved_3w=[0]*10+[1]*10))
    mask = np.array([True]*9+[False]+[True]+[False]*9)
    result = assess(df, mask)
    assert result["matched_stop_lift"] == 0
    assert result["stop_rate"] == 0.9
    assert result["baseline_stop_rate"] == 0.5


def test_ambiguity_unresolved_and_winner_cost_remain_visible():
    df = sample_frame().iloc[:24].copy()
    df.loc[0, "ambiguous_3w"] = 1
    result = assess(df, np.array([True]*12+[False]*12))
    assert result["selected_n"] == 12 and result["evaluable_n"] == 11
    assert result["ambiguous_n"] == 1
    assert result["stop_capture"] == 1
    assert result["winner_loss"] == 0
    assert result["unresolved_n"] == 4


def test_generator_freezes_before_outer_evaluation_and_retains_empty_quarters(tmp_path):
    frame = sample_frame()
    calls = []
    def agent(payload):
        calls.append(payload)
        return {"proposals": [proposal()]}
    cfg = Config(rounds=2, min_train_quarters=6, min_inner_quarters=2,
                 min_selected=5, min_complement=5, min_snapshots=2)
    calendar = [str(q) for q in pd.period_range("2022Q4", periods=11, freq="Q")]
    result = discover(frame, ["pullback_pct", "volume_ratio"], calendar, agent, cfg, tmp_path)
    assert len(result["folds"]) == 10  # agent + simple comparator, including empty test quarter
    assert result["folds"][-1]["test_n"] == 0
    assert calls and all("test_results" not in call for call in calls)
    assert all("12w" not in json.dumps(call) for call in calls)
    frozen = json.loads((tmp_path / "frozen_rules.json").read_text())
    assert frozen["test_feedback_used"] is False
    report = write_report(result, tmp_path)
    assert "KEEP PRODUCTION FROZEN" in report.read_text()
    assert "NOT AN UNTOUCHED HOLDOUT" in report.read_text()
    assert "winner_loss" in report.read_text()


def test_changing_outer_future_cannot_change_first_fold_proposal_or_rule(tmp_path):
    frame = sample_frame()
    cfg = Config(rounds=1, min_train_quarters=6, min_inner_quarters=2,
                 min_selected=5, min_complement=5, min_snapshots=2)
    calendar = [str(q) for q in pd.period_range("2022Q4", periods=7, freq="Q")]
    first = discover(frame, ["pullback_pct"], calendar,
                     lambda _: {"proposals": [proposal()]}, cfg, tmp_path / "first")
    assert first["frozen"][0]["rule"] is not None
    changed = frame.copy()
    future = changed.snapshot_date >= pd.Period(calendar[-1], freq="Q").start_time
    changed.loc[future, "stop_first_3w"] = 0
    changed.loc[future, "pullback_pct"] = 999
    second = discover(changed, ["pullback_pct"], calendar,
                      lambda _: {"proposals": [proposal()]}, cfg, tmp_path / "second")
    assert first["frozen"] == second["frozen"]


def test_provider_failure_cannot_publish_a_successful_research_report(tmp_path):
    def fail(_):
        raise RuntimeError("provider unavailable")
    with pytest.raises(RuntimeError, match="provider unavailable"):
        discover(sample_frame(), ["pullback_pct"],
                 [str(q) for q in pd.period_range("2022Q4", periods=8, freq="Q")],
                 fail, Config(), tmp_path)
    assert not (tmp_path / "R6_REPORT.md").exists()


def test_budget_counts_failed_attempts_caches_success_and_survives_new_client(tmp_path):
    calls = []
    def transport(system, prompt):
        calls.append(prompt)
        return '{"proposals": []}' if len(calls) == 1 else "invalid json"
    kwargs = dict(ledger_path=tmp_path/"ledger.json", cache_dir=tmp_path/"cache",
                  prior_used=42, total_budget=44, run_cap=2, transport=transport)
    agent = RDAgentProposer(**kwargs)
    assert agent({"fold": "A"}) == {"proposals": []}
    assert agent({"fold": "A"}) == {"proposals": []}
    assert len(calls) == 1
    with pytest.raises(RuntimeError):
        agent({"fold": "B"})
    assert agent.snapshot()["accounted_total"] == 44
    resumed = RDAgentProposer(**kwargs)
    assert resumed({"fold": "A"}) == {"proposals": []}
    with pytest.raises(RuntimeError):
        resumed({"fold": "C"})
    assert len(calls) == 2
    assert resumed.snapshot()["failed_attempts"] == 1


def test_input_population_mismatch_and_tampering_fail_closed(tmp_path):
    df = sample_frame()
    samples, metadata = tmp_path/"samples.csv", tmp_path/"metadata.json"
    df.to_csv(samples, index=False)
    meta = {"replay_dataset_sha256": "a"*64, "usable_trigger_entries": len(df),
            "entry_quarters": sorted(df.entry_date.dt.to_period("Q").astype(str).unique())}
    metadata.write_text(json.dumps(meta))
    actual, manifest = load_inputs(samples, metadata, sha256_file(samples))
    assert len(actual) == len(df)
    assert manifest["features"] == ["volume_ratio", "pullback_pct"]
    with pytest.raises(ValueError, match="SHA256"):
        load_inputs(samples, metadata, "b"*64)
    meta["usable_trigger_entries"] -= 1
    metadata.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="count"):
        load_inputs(samples, metadata, sha256_file(samples))


def test_unsupported_and_abstaining_folds_cannot_be_dropped(tmp_path):
    frame = sample_frame()
    frame["fast_winner_3w"] = 0
    result = discover(frame, ["pullback_pct"],
                      [str(q) for q in pd.period_range("2022Q4", periods=10, freq="Q")],
                      lambda _: {"proposals": []}, Config(), tmp_path)
    assert len(result["folds"]) == 8
    assert all(r["rule_status"] == "NO_SUPPORTED_RULE" for r in result["folds"])
    assert all(r["selected_n"] == 0 for r in result["folds"])


def test_generated_report_does_not_select_using_unsupported_positive_folds(tmp_path):
    result = discover(sample_frame(), ["pullback_pct"],
                      [str(q) for q in pd.period_range("2022Q4", periods=10, freq="Q")],
                      lambda _: {"proposals": [proposal()]}, Config(rounds=1), tmp_path)
    assert all(item["rule"] is None or item["inner_evidence"]["supported_quarters"] >= 3
               for item in result["frozen"])
    text = write_report(result, tmp_path).read_text()
    assert "no automatic promotion" in text
    assert "high winner_loss" in text


def test_official_backend_adapter_counts_each_underlying_completion(tmp_path, monkeypatch):
    import importlib.metadata
    import dotenv
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "test-version")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setenv("RD_AGENT_MODEL", "deepseek/test-model")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-placeholder")
    monkeypatch.setenv("DEEPSEEK_API_BASE", "https://example.invalid")
    modules = {name: ModuleType(name) for name in (
        "rdagent", "rdagent.oai", "rdagent.oai.backend", "rdagent.oai.backend.base",
        "rdagent.oai.backend.litellm", "rdagent.log")}
    backend = modules["rdagent.oai.backend.litellm"]
    modules["rdagent.oai.backend"].litellm = backend
    modules["rdagent.oai.backend.base"].LLM_SETTINGS = SimpleNamespace(max_retry=10)
    backend.LITELLM_SETTINGS = SimpleNamespace(chat_model="old", chat_max_tokens=10, chat_stream=True)
    calls = []
    def completion(**kwargs):
        assert kwargs["max_retries"] == kwargs["num_retries"] == 0
        calls.append(kwargs)
        return '{"proposals": []}'
    backend.completion = completion
    class FakeBackend:
        def __init__(self, **kwargs):
            assert not any(kwargs.values())
        def build_messages_and_create_chat_completion(self, **kwargs):
            backend.completion(messages=[])
            return backend.completion(messages=[])
    backend.LiteLLMAPIBackend = FakeBackend
    # Match rdagent==0.8.0 on the data machine: the logger has no `debug` method.
    modules["rdagent.log"].rdagent_logger = SimpleNamespace(
        **{method: lambda *a, **kw: None for method in ("info", "warning", "error", "log_object")})
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    agent = RDAgentProposer(ledger_path=tmp_path/"ledger.json", cache_dir=tmp_path/"cache")
    agent({"fold": "2024Q1"})
    assert len(calls) == 2
    assert agent.snapshot()["accounted_total"] == 44
    assert agent.snapshot()["rdagent_version"] == "test-version"
