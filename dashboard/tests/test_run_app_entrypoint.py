from __future__ import annotations

import sys

import pytest

from dashboard import run_app


def test_run_app_forwards_midweek_inputs_to_streamlit(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(run_app, "is_port_in_use", lambda port: False)
    monkeypatch.setattr(run_app.os, "chdir", lambda path: captured.update(cwd=path))

    def fake_execvp(executable, command):
        captured["executable"] = executable
        captured["command"] = command
        raise RuntimeError("exec intercepted")

    monkeypatch.setattr(run_app.os, "execvp", fake_execvp)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dashboard/run_app.py",
            "--csv",
            "us/breakout_follow_pool.csv",
            "--midweek-csv",
            "us/breakout_follow_pool_midweek.csv",
            "--window-date",
            "2026-07-30",
            "--server-port",
            "8517",
            "--headless",
        ],
    )

    with pytest.raises(RuntimeError, match="exec intercepted"):
        run_app.main()

    command = captured["command"]
    assert command[-6:] == [
        "--csv",
        str(run_app.Path.cwd() / "us" / "breakout_follow_pool.csv"),
        "--midweek-csv",
        str(run_app.Path.cwd() / "us" / "breakout_follow_pool_midweek.csv"),
        "--window-date",
        "2026-07-30",
    ]
    assert command[command.index("--server.port") + 1] == "8517"
