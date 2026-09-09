import importlib.metadata
import sys
from types import ModuleType, SimpleNamespace

import dotenv
import numpy as np
import pandas as pd
import pytest

from backtest.blind_rule_discovery.r8_agent import R8AgentProposer
from backtest.blind_rule_discovery.r8_atlas import (
    AtlasConfig,
    aggregate_feature_stability,
    class_profiles,
    cliffs_delta,
    discover_interactions,
    inner_interaction_evidence,
    quarter_feature_contrasts,
    quintile_surfaces,
    train_percentile,
    validate_interaction,
)


def test_cliffs_delta_and_train_percentile_ties():
    assert cliffs_delta([3, 4], [1, 2]) == pytest.approx(1.0)
    assert cliffs_delta([1, 2], [3, 4]) == pytest.approx(-1.0)
    assert cliffs_delta([1, 2], [1, 2]) == pytest.approx(0.0)
    pct = train_percentile([1, 1, 2, 3], np.array([1.0, 2.0, np.nan]))
    assert pct[0] == pytest.approx(0.25)
    assert pct[1] == pytest.approx(0.625)
    assert np.isnan(pct[2])


def test_predeclared_feature_stability_requires_effect_zeros_and_loo_sign():
    rows = []
    for i in range(8):
        rows.append(dict(quarter=f"Q{i}", feature="x", status="SUPPORTED",
                         matched_percentile_gap_median=.12 + i*.001,
                         cliffs_delta_winner_minus_stop=.20,
                         low_tail_log_odds_winner_vs_stop=-.2,
                         high_tail_log_odds_winner_vs_stop=.3,
                         winner_missing_fraction=.1, stop_missing_fraction=.1))
    out = aggregate_feature_stability(pd.DataFrame(rows), AtlasConfig())
    assert out.iloc[0].stability_label == "CONSISTENT_WINNER_HIGH"
    changed = pd.DataFrame(rows)
    changed.loc[:2, "matched_percentile_gap_median"] = -.5
    assert aggregate_feature_stability(changed, AtlasConfig()).iloc[0].stability_label == "MIXED_OR_WEAK"
    zeros = pd.DataFrame(rows)
    zeros.loc[:2, "matched_percentile_gap_median"] = 0.0
    assert aggregate_feature_stability(zeros, AtlasConfig()).iloc[0].direction_consistency == pytest.approx(5/8)


def synthetic(periods=10, weeks=4, each_class=6):
    rows = []
    for q_idx, quarter in enumerate(pd.period_range("2022Q4", periods=periods, freq="Q")):
        for week in range(weeks):
            day = quarter.start_time + pd.Timedelta(days=(4-quarter.start_time.dayofweek) % 7 + 7*week)
            for klass, label in enumerate(("fast_winner_3w", "stop_first_3w", "unresolved_3w", "ambiguous_3w")):
                for i in range(each_class):
                    winner = label == "fast_winner_3w"
                    base = 10.0 if winner else 0.0
                    rows.append({
                        "code": f"{q_idx}-{week}-{klass}-{i}",
                        "snapshot_date": day,
                        "entry_date": day + pd.Timedelta(days=1),
                        "exit_date_w3": day + pd.Timedelta(days=21),
                        "fast_winner_3w": int(label == "fast_winner_3w"),
                        "stop_first_3w": int(label == "stop_first_3w"),
                        "unresolved_3w": int(label == "unresolved_3w"),
                        "ambiguous_3w": int(label == "ambiguous_3w"),
                        "pullback_pct": base + i/10 + q_idx/100,
                        "volume_ratio": base + (each_class-i)/10 + q_idx/100,
                        "current_vs_ibd_candidate_pct": base + i/20,
                    })
    return pd.DataFrame(rows)


def test_class_profiles_keep_all_four_labels_and_quintiles_are_past_fit():
    df = synthetic()
    calendar = [str(q) for q in pd.period_range("2022Q4", periods=10, freq="Q")]
    profiles = class_profiles(df, ["pullback_pct"], calendar)
    assert set(profiles.path_class) == {"fast_winner_3w", "stop_first_3w", "unresolved_3w", "ambiguous_3w"}
    cfg = AtlasConfig()
    first = quintile_surfaces(df, ["pullback_pct"], calendar, cfg)
    changed = df.copy()
    test_q = calendar[6]
    changed.loc[changed.snapshot_date.dt.to_period("Q").astype(str) == test_q, "pullback_pct"] += 1000
    second = quintile_surfaces(changed, ["pullback_pct"], calendar, cfg)
    a = first.loc[first.quarter == test_q, ["bin", "train_q20", "train_q40", "train_q60", "train_q80"]]
    b = second.loc[second.quarter == test_q, ["bin", "train_q20", "train_q40", "train_q60", "train_q80"]]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_quarter_contrast_is_winner_high_and_uses_matched_weeks():
    df = synthetic()
    calendar = [str(q) for q in pd.period_range("2022Q4", periods=10, freq="Q")]
    out = quarter_feature_contrasts(df, ["pullback_pct"], calendar, AtlasConfig())
    supported = out.loc[out.status == "SUPPORTED"]
    assert len(supported) == 4
    assert (supported.cliffs_delta_winner_minus_stop > 0).all()
    assert (supported.matched_percentile_gap_median > 0).all()
    assert (supported.matched_snapshots >= 3).all()


def proposal():
    return {
        "name": "winner_structure_combo",
        "hypothesis": "Two independent PIT dimensions jointly mark winner structure; falsified if next-quarter target lift is non-positive.",
        "expression": {
            "op": "product",
            "left": {"op": "train_percentile", "feature": "pullback_pct"},
            "right": {"op": "train_percentile", "feature": "volume_ratio"},
        },
        "target": "winner",
        "tail": "high",
        "quantile": 0.8,
    }


def test_interaction_contract_requires_exactly_two_features_and_extreme_tail():
    p = proposal()
    features = ["pullback_pct", "volume_ratio", "current_vs_ibd_candidate_pct"]
    assert validate_interaction(p, features) == {"pullback_pct", "volume_ratio"}
    bad_single = {**p, "expression": {"op": "raw", "feature": "pullback_pct"}}
    with pytest.raises(ValueError, match="exactly two distinct"):
        validate_interaction(bad_single, features)
    bad_tail = {**p, "quantile": 0.2}
    with pytest.raises(ValueError, match="q20-low or q80-high"):
        validate_interaction(bad_tail, features)
    bad_three = {**p, "expression": {"op": "product",
        "left": {"op": "product", "left": {"op": "raw", "feature": "pullback_pct"},
                 "right": {"op": "raw", "feature": "volume_ratio"}},
        "right": {"op": "raw", "feature": "current_vs_ibd_candidate_pct"}}}
    with pytest.raises(ValueError, match="exactly two distinct"):
        validate_interaction(bad_three, features)


def test_inner_interaction_uses_purged_past_and_has_support():
    df = synthetic()
    calendar = [str(q) for q in pd.period_range("2022Q4", periods=10, freq="Q")]
    cutoff = pd.Period(calendar[6], freq="Q").start_time
    past = df.loc[(df.snapshot_date < cutoff) & (df.exit_date_w3 < cutoff)].reset_index(drop=True)
    evidence = inner_interaction_evidence(proposal(), past, calendar[:6], AtlasConfig())
    assert evidence["supported_quarters"] >= 3
    assert evidence["median_matched_target_lift"] > 0


def test_discovery_freezes_before_outer_and_keeps_complete_trace(tmp_path):
    df = synthetic()
    calendar = [str(q) for q in pd.period_range("2022Q4", periods=10, freq="Q")]
    calls = []

    def proposer(payload):
        calls.append((payload["fold"], payload["round"]))
        return {"proposals": [proposal()]} if payload["round"] == 0 else {"proposals": []}

    out = tmp_path / "r8"
    out.mkdir()
    lock, outer, stability, recurrence, traces = discover_interactions(
        df, ["pullback_pct", "volume_ratio"], calendar, proposer, AtlasConfig(), out)
    assert lock["test_feedback_used"] is False
    assert len(traces) == len(calls)
    assert (out / "interaction_frozen.json").exists()
    assert (out / "discovery_trace.jsonl").exists()
    assert not outer.empty
    assert outer.supported.all()
    assert (outer.matched_target_lift > 0).all()
    assert not recurrence.empty
    if not stability.empty:
        assert set(stability["name"]) == {"winner_structure_combo"}


def test_agent_cache_identity_includes_exact_model_and_all_attempts_are_metered(tmp_path):
    ledger = tmp_path / "ledger.json"
    cache = tmp_path / "cache"
    calls = []

    def transport(system, prompt):
        calls.append((system, prompt))
        return '{"proposals": []}'

    a = R8AgentProposer(ledger_path=ledger, cache_dir=cache, model="provider/model-a",
                        prior_used_floor=132, transport=transport)
    assert a({"fold": "2025Q1", "round": 0}) == {"proposals": []}
    assert a.snapshot()["attempts_used"] == 1
    a2 = R8AgentProposer(ledger_path=ledger, cache_dir=cache, model="provider/model-a",
                         prior_used_floor=132, transport=transport)
    assert a2({"fold": "2025Q1", "round": 0}) == {"proposals": []}
    assert len(calls) == 1
    b = R8AgentProposer(ledger_path=ledger, cache_dir=cache, model="other/model-a",
                        prior_used_floor=132, transport=transport)
    assert b({"fold": "2025Q1", "round": 0}) == {"proposals": []}
    assert len(calls) == 2
    assert b.snapshot()["attempts_used"] == 2


def test_official_backend_contract_retries_original_prompt_with_240s_streaming(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.8.0-test")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-placeholder")
    monkeypatch.setenv("DEEPSEEK_API_BASE", "https://example.invalid")
    modules = {name: ModuleType(name) for name in (
        "rdagent", "rdagent.oai", "rdagent.oai.backend", "rdagent.oai.backend.base",
        "rdagent.oai.backend.litellm", "rdagent.log")}
    backend = modules["rdagent.oai.backend.litellm"]
    modules["rdagent.oai.backend"].litellm = backend
    llm_settings = SimpleNamespace(max_retry=10)
    modules["rdagent.oai.backend.base"].LLM_SETTINGS = llm_settings
    backend.LITELLM_SETTINGS = SimpleNamespace(chat_model="old", chat_max_tokens=3000, chat_stream=False)
    calls = []

    def completion(**kwargs):
        assert kwargs["timeout"] == 240
        assert kwargs["max_retries"] == kwargs["num_retries"] == 0
        calls.append(kwargs)
        return ("partial", "length") if len(calls) == 1 else ('{"proposals": []}', "stop")

    backend.completion = completion

    class FakeBackend:
        def __init__(self, **kwargs):
            assert not any(kwargs.values())

        def _create_chat_completion_inner_function(self, messages, response_format=None, **kwargs):
            assert response_format is None
            assert backend.LITELLM_SETTINGS.chat_max_tokens == 8192
            assert backend.LITELLM_SETTINGS.chat_stream is True
            return backend.completion(messages=messages)

        def build_messages_and_create_chat_completion(self, user_prompt, system_prompt=None, **kwargs):
            assert llm_settings.max_retry == 3
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            last = None
            for _ in range(llm_settings.max_retry):
                try:
                    return self._create_chat_completion_auto_continue(messages=messages)
                except RuntimeError as exc:
                    last = exc
            raise last

    backend.LiteLLMAPIBackend = FakeBackend
    modules["rdagent.log"].rdagent_logger = SimpleNamespace(
        info=lambda *a, **kw: None, warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None, log_object=lambda *a, **kw: None)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    agent = R8AgentProposer(ledger_path=tmp_path / "ledger.json", cache_dir=tmp_path / "cache",
                            model="deepseek/test-model", prior_used_floor=132)
    assert agent({"fold": "2025Q1", "round": 0}) == {"proposals": []}
    assert len(calls) == 2
    assert calls[0]["messages"] == calls[1]["messages"]
    assert all(m["role"] != "assistant" for m in calls[1]["messages"])
    assert agent.snapshot()["accounted_total_floor"] == 134
    assert agent.snapshot()["cache_identity_includes_model"] is True


def test_no_feature_prefilter_all_requested_features_are_published():
    df = synthetic()
    calendar = [str(q) for q in pd.period_range("2022Q4", periods=10, freq="Q")]
    features = ["pullback_pct", "volume_ratio"]
    contrasts = quarter_feature_contrasts(df, features, calendar, AtlasConfig())
    assert set(contrasts.feature) == set(features)
    assert set(aggregate_feature_stability(contrasts, AtlasConfig()).feature) == set(features)
