import json
from pathlib import Path

import pandas as pd
import pytest

import backtest.blind_rule_discovery.r8_agent as agent_module
import backtest.blind_rule_discovery.r8_runner as runner
from backtest.blind_rule_discovery.r8_runner import nonoverlap_w3_frame


def test_nonoverlap_w3_is_shared_outcome_independent_issuer_schedule():
    frame = pd.DataFrame({
        "code": ["A", "A", "A", "A", "B"],
        "snapshot_date": pd.to_datetime([
            "2025-01-03", "2025-01-10", "2025-01-24", "2025-01-31", "2025-01-10"
        ]),
        "entry_date": pd.to_datetime([
            "2025-01-06", "2025-01-13", "2025-01-27", "2025-02-03", "2025-01-13"
        ]),
        "exit_date_w3": pd.to_datetime([
            "2025-01-27", "2025-02-03", "2025-02-17", "2025-02-24", "2025-02-03"
        ]),
        "fast_winner_3w": [1, 0, 0, 1, 0],
        "stop_first_3w": [0, 1, 1, 0, 1],
        "unresolved_3w": [0, 0, 0, 0, 0],
        "ambiguous_3w": [0, 0, 0, 0, 0],
        "pullback_pct": [-5.0, -20.0, -40.0, -2.0, -10.0],
    })

    admitted = nonoverlap_w3_frame(frame)
    assert admitted.index.tolist() == [0, 1, 2]
    assert admitted.code.tolist() == ["A", "A", "B"]
    assert admitted.entry_date.tolist() == [
        pd.Timestamp("2025-01-06"), pd.Timestamp("2025-02-03"), pd.Timestamp("2025-01-13")
    ]

    changed = frame.copy()
    changed[["fast_winner_3w", "stop_first_3w"]] = changed[["stop_first_3w", "fast_winner_3w"]].to_numpy()
    changed["pullback_pct"] *= 1000
    rerun = nonoverlap_w3_frame(changed)
    assert rerun[["code", "snapshot_date", "entry_date", "exit_date_w3"]].equals(
        admitted[["code", "snapshot_date", "entry_date", "exit_date_w3"]]
    )


def _bound_fixture(tmp_path: Path):
    samples = tmp_path / "samples.csv"
    metadata = tmp_path / "metadata.json"
    r6 = tmp_path / "r6"
    r6.mkdir()
    samples.write_text("sample")
    metadata.write_text("metadata")
    bound = []
    for name in ("input_manifest.json", "frozen_rules.json", "summary.json"):
        path = r6 / name
        path.write_text(name)
        bound.append(path)
    frame = pd.DataFrame({
        "code": ["A"],
        "snapshot_date": pd.to_datetime(["2025-01-03"]),
        "entry_date": pd.to_datetime(["2025-01-06"]),
        "exit_date_w3": pd.to_datetime(["2025-01-27"]),
        "fast_winner_3w": [1],
        "stop_first_3w": [0],
        "unresolved_3w": [0],
        "ambiguous_3w": [0],
        "pullback_pct": [1.0],
    })
    observed = {
        "samples_sha256": "a" * 64,
        "metadata_sha256": "b" * 64,
        "source_replay_dataset_sha256": "c" * 64,
        "sample_rows": 1,
        "snapshot_weeks": 1,
        "unique_tickers": 1,
        "calendar": ["2025Q1"],
        "features": ["pullback_pct"],
        "population": "fixture",
    }
    return samples, metadata, r6, bound, frame, observed


def test_r8a_is_offline_and_publishes_without_model_or_ledger(tmp_path, monkeypatch):
    samples, metadata, r6, bound, frame, observed = _bound_fixture(tmp_path)
    monkeypatch.delenv("RD_AGENT_MODEL", raising=False)
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    monkeypatch.setattr(runner, "load_bound_inputs",
                        lambda *a: (frame, observed, {}, bound, 132, "frozen-sha"))
    monkeypatch.setattr(runner, "_execution_head", lambda: "head")
    monkeypatch.setattr(runner, "class_profiles",
                        lambda *a, **k: pd.DataFrame({"feature": ["pullback_pct"], "path_class": ["fast_winner_3w"]}))
    monkeypatch.setattr(runner, "quintile_surfaces",
                        lambda *a, **k: pd.DataFrame({"feature": ["pullback_pct"], "bin": [1]}))
    monkeypatch.setattr(runner, "quarter_feature_contrasts",
                        lambda *a, **k: pd.DataFrame({"feature": ["pullback_pct"], "status": ["INSUFFICIENT_SUPPORT"]}))
    monkeypatch.setattr(runner, "aggregate_feature_stability",
                        lambda *a, **k: pd.DataFrame({
                            "feature": ["pullback_pct"], "stability_label": ["INSUFFICIENT_EVIDENCE"],
                            "supported_quarters": [0], "direction_consistency": [None],
                            "median_matched_percentile_gap": [None], "median_cliffs_delta": [None],
                        }))

    output = tmp_path / "atlas"
    runner.run_atlas(samples, metadata, r6, output)
    complete = json.loads((output / "COMPLETE.json").read_text())
    assert complete["status"] == "ATLAS_COMPLETE"
    assert complete["rdagent_calls"] == 0
    assert not (output / "FAILED.json").exists()
    assert (output / "feature_stability.csv").exists()
    assert set(pd.read_csv(output / "feature_stability.csv").panel) == {"all_entries", "nonoverlap_w3"}


def test_existing_r8_ledger_preserves_prior_floor_and_failed_attempts(tmp_path):
    ledger = tmp_path / "r8_requests.json"
    ledger.write_text(json.dumps({"hard_limit": 868, "attempts_used": 2}))
    assert runner._resolve_prior_floor(ledger, 132, None) == 132
    assert json.loads(ledger.read_text())["attempts_used"] == 2
    with pytest.raises(ValueError, match="conflicts"):
        runner._resolve_prior_floor(ledger, 132, 140)


def test_agent_failure_cannot_invalidate_completed_atlas(tmp_path, monkeypatch):
    samples, metadata, r6, bound, frame, observed = _bound_fixture(tmp_path)
    atlas = tmp_path / "atlas"
    atlas.mkdir()
    atlas_manifest = atlas / "input_manifest.json"
    atlas_complete = atlas / "COMPLETE.json"
    atlas_manifest.write_text("atlas-manifest")
    atlas_complete.write_text("atlas-complete")
    atlas_bytes = {p: p.read_bytes() for p in (atlas_manifest, atlas_complete)}

    monkeypatch.setattr(runner, "load_bound_inputs",
                        lambda *a: (frame, observed, {}, bound, 132, "frozen-sha"))
    monkeypatch.setattr(runner, "_load_atlas_anchor",
                        lambda *a: ({"protocol_sha256": "atlas-protocol"}, atlas_complete))
    monkeypatch.setattr(runner, "_model_from_env", lambda: "provider/model")
    monkeypatch.setattr(runner, "_backend_version", lambda: "0.8.0-test")
    monkeypatch.setattr(runner, "_execution_head", lambda: "head")
    monkeypatch.setattr(runner, "discover_interactions_compact",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("proxy timeout")))

    class FakeProposer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def snapshot(self):
            return {"attempts_used": 2, "failed_attempts": 2, "successful_calls": 0}

    monkeypatch.setattr(agent_module, "R8AgentProposer", FakeProposer)
    output = tmp_path / "agent"
    ledger = tmp_path / "ledger.json"
    cache = tmp_path / "cache"
    ledger.write_text(json.dumps({"hard_limit": 868, "attempts_used": 2}))
    cache.mkdir()

    with pytest.raises(RuntimeError, match="proxy timeout"):
        runner.run_agent(samples, metadata, r6, atlas, output, ledger, cache)

    assert {p: p.read_bytes() for p in (atlas_manifest, atlas_complete)} == atlas_bytes
    failed = json.loads((output / "FAILED.json").read_text())
    assert failed["status"] == "AGENT_FAILED"
    assert failed["atlas_invalidated"] is False
