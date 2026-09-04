from __future__ import annotations

from pathlib import Path

import pytest

from backtest.blind_rule_discovery.experiment import MAX_RESEARCH_SECONDS, run_research_command


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "agent_workspace"
    workspace.mkdir()
    (workspace / "samples.csv").write_text("sample_id,X001\na,1\n", encoding="utf-8")
    (workspace / "prompt.md").write_text("x", encoding="utf-8")
    return workspace


def test_sandbox_preflight_rejects_wrapper_that_can_read_outside_workspace(monkeypatch, tmp_path: Path):
    workspace = _workspace(tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))

        class Result:
            returncode = 97
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="sandbox isolation preflight failed"):
        run_research_command(
            ["research"],
            workspace,
            sandbox_prefix=["sandbox-exec", "-D", "WORKSPACE={workspace}", "--"],
        )

    assert len(calls) == 1
    assert "/bin/sh" in calls[0][0]


def test_sandbox_preflight_passes_before_agent_and_runtime_is_hard_capped(monkeypatch, tmp_path: Path):
    workspace = _workspace(tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)
    run_research_command(
        ["research"],
        workspace,
        sandbox_prefix=["sandbox-exec", "-D", "WORKSPACE={workspace}", "--"],
        timeout_seconds=999999,
    )

    assert len(calls) == 2
    probe_cmd, probe_kwargs = calls[0]
    agent_cmd, agent_kwargs = calls[1]
    assert "/bin/sh" in probe_cmd
    assert probe_kwargs["timeout"] <= 30
    assert agent_cmd[-1] == "research"
    assert agent_kwargs["timeout"] == MAX_RESEARCH_SECONDS == 3600
