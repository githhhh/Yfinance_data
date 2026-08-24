from pathlib import Path
import subprocess
import sys

import pandas as pd

from tools.refresh_replay_signal_eps import rebuild_replay_signal_eps


class FakeProvider:
    def __init__(self):
        self.calls = []

    def fetch_eps_yoy(self, symbol, snapshot_date):
        self.calls.append((symbol, snapshot_date))
        records = {
            ("GOOD", "2026-08-07"): {
                "eps_yoy_growth": 42.5,
                "source": "SEC",
                "effective_date": "2026-08-01",
                "current_eps": 1.425,
                "prior_year_eps": 1.0,
                "current_period": "2026-06-30",
                "prior_year_period": "2025-06-30",
            }
        }
        return records.get((symbol, snapshot_date))


def _write_week(pool_dir: Path, snapshot_date: str, rows: list[dict]) -> Path:
    week_dir = pool_dir / snapshot_date
    week_dir.mkdir(parents=True)
    path = week_dir / "breakout_follow_pool.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_rebuild_replay_signal_eps_overwrites_unaudited_signal_values(tmp_path):
    pool_dir = tmp_path / "replay_pools"
    path = _write_week(
        pool_dir,
        "2026-08-07",
        [
            {
                "snapshot_date": "2026-08-07",
                "code": "GOOD",
                "signal": True,
                "eps_yoy_growth": 999.0,
                "eps_yoy_growth_source": "unaudited",
            },
            {
                "snapshot_date": "2026-08-07",
                "code": "MISS",
                "signal": True,
                "eps_yoy_growth": 888.0,
                "eps_yoy_growth_source": "unaudited",
            },
            {
                "snapshot_date": "2026-08-07",
                "code": "QUIET",
                "signal": False,
                "eps_yoy_growth": 777.0,
                "eps_yoy_growth_source": "unaudited",
            },
        ],
    )

    provider = FakeProvider()
    summary = rebuild_replay_signal_eps(pool_dir, provider=provider)

    updated = pd.read_csv(path)
    assert updated.loc[0, "eps_yoy_growth"] == 42.5
    assert updated.loc[0, "eps_yoy_growth_source"] == "SEC"
    assert pd.isna(updated.loc[1, "eps_yoy_growth"])
    assert pd.isna(updated.loc[1, "eps_yoy_growth_source"])
    assert updated.loc[2, "eps_yoy_growth"] == 777.0
    assert updated.loc[2, "eps_yoy_growth_source"] == "unaudited"

    signal_eps = pd.read_csv(pool_dir / "signal_eps_pit.csv")
    assert signal_eps["code"].tolist() == ["GOOD", "MISS"]
    assert signal_eps.loc[0, "eps_yoy_growth"] == 42.5
    assert signal_eps.loc[0, "source"] == "SEC"
    assert signal_eps.loc[1, "status"] == "unresolved"

    assert summary["signal_rows"] == 2
    assert summary["filled_rows"] == 1
    assert summary["unresolved_rows"] == 1
    assert provider.calls == [("GOOD", "2026-08-07"), ("MISS", "2026-08-07")]


def test_refresh_replay_signal_eps_cli_runs_from_tools_path(tmp_path):
    pool_dir = tmp_path / "empty_replay_pools"
    pool_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "tools/refresh_replay_signal_eps.py",
            "--pool-dir",
            str(pool_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert '"signal_rows": 0' in result.stdout
