from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.blind_rule_discovery.experiment import MAX_RESEARCH_SECONDS, write_agent_workspace
from backtest.blind_rule_discovery.runner import RESEARCH_BUDGET_POLICY, _effective_research_seconds


def test_workspace_prompt_treats_research_budget_as_ceiling(tmp_path: Path):
    agent = pd.DataFrame([{"sample_id": "S000001", "X001": 1.0}])
    workspace = write_agent_workspace(agent, tmp_path)
    prompt = (workspace / "prompt.md").read_text(encoding="utf-8")

    assert "research timeout is a safety ceiling, not a target" in prompt
    assert "Stop early once a compact rule has stable repeated-period evidence" in prompt
    assert "Do not continue searching merely because time, model calls, or compute budget remain" in prompt
    assert "exhaustive threshold search" in prompt
    assert "500" not in prompt


def test_effective_research_seconds_is_a_hard_ceiling_not_a_target():
    assert _effective_research_seconds(1800) == 1800
    assert _effective_research_seconds(MAX_RESEARCH_SECONDS) == MAX_RESEARCH_SECONDS == 3600
    assert _effective_research_seconds(99999) == MAX_RESEARCH_SECONDS
    assert _effective_research_seconds(0) == 1
    assert _effective_research_seconds(-10) == 1
    assert RESEARCH_BUDGET_POLICY == "hard_ceiling_stop_early_on_convergence"
