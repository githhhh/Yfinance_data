from __future__ import annotations

import inspect
import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

from backtest.latest_quant_trade_replay import SnapshotWeek
from backtest.blind_rule_discovery import replay_builder as rb


def test_parallel_replay_uses_spawned_processes_never_threads():
    source = inspect.getsource(rb)
    assert "ProcessPoolExecutor" in source
    assert "ThreadPoolExecutor" not in source
    assert 'mp.get_context("spawn")' in source
    assert 1 <= rb.DEFAULT_WORKERS <= 6
    # Top-level callables must remain pickleable under macOS spawn.
    pickle.dumps(rb._init_replay_worker)
    pickle.dumps(rb._run_week_worker)


def test_stateless_quant_trade_contract_is_required(tmp_path: Path):
    strategy = tmp_path / "strategy"
    strategy.mkdir()
    run_context = strategy / "run_context.py"
    run_context.write_text(
        "class RunContext:\n"
        "    @classmethod\n"
        "    def replay(cls, week_date):\n"
        "        return cls()\n",
        encoding="utf-8",
    )
    contract = rb.assert_quant_trade_replay_stateless(tmp_path)
    assert contract["state_mode"] == "stateless_independent_weeks"
    assert contract["replay_parameters"] == ["week_date"]

    run_context.write_text(
        "class RunContext:\n"
        "    @classmethod\n"
        "    def replay(cls, week_date, old_pool=None):\n"
        "        return cls()\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="stateless quant_trade"):
        rb.assert_quant_trade_replay_stateless(tmp_path)


def test_stateless_contract_rejects_hidden_replay_old_pool_field(tmp_path: Path):
    strategy = tmp_path / "strategy"
    strategy.mkdir()
    (strategy / "run_context.py").write_text(
        "class RunContext:\n"
        "    replay_old_pool: set\n"
        "    @classmethod\n"
        "    def replay(cls, week_date):\n"
        "        return cls()\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="cross-week state"):
        rb.assert_quant_trade_replay_stateless(tmp_path)


def test_no_clean_reuses_only_compatible_success_checkpoint(tmp_path: Path):
    week = SnapshotWeek("2024-01-05", "2024-01-05")
    week_dir = tmp_path / week.snapshot_date
    week_dir.mkdir()
    (week_dir / "breakout_follow_pool.csv").write_text("code,signal\nAAA,True\n")
    metadata = {
        "snapshot_date": week.snapshot_date,
        "expected_last_trading_day": week.expected_last_trading_day,
        "status": "success",
        "data_source_mode": "research_full_history_bundle",
        "daily_pkl_sha256": "daily",
        "weekly_pkl_sha256": "weekly",
        "quant_trade_commit": "qt",
    }
    (week_dir / "metadata.json").write_text(json.dumps(metadata))

    reused = rb._completed_week_metadata(
        tmp_path,
        week,
        daily_sha="daily",
        weekly_sha="weekly",
        quant_trade_commit="qt",
    )
    assert reused is not None

    assert rb._completed_week_metadata(
        tmp_path,
        week,
        daily_sha="changed",
        weekly_sha="weekly",
        quant_trade_commit="qt",
    ) is None
    assert rb._completed_week_metadata(
        tmp_path,
        week,
        daily_sha="daily",
        weekly_sha="weekly",
        quant_trade_commit="changed",
    ) is None


def test_worker_task_uses_process_local_bundles_and_empty_cross_week_state(monkeypatch):
    daily = {"AAA": pd.DataFrame({"Close": [1.0]})}
    weekly = {"AAA": pd.DataFrame({"Close": [1.0]})}
    rb._WORKER_DAILY_DATA = daily
    rb._WORKER_WEEKLY_DATA = weekly
    rb._WORKER_CONFIG = {
        "daily_pkl": "/tmp/daily.pkl",
        "weekly_pkl": "/tmp/weekly.pkl",
        "daily_sha": "d",
        "weekly_sha": "w",
        "output_root": "/tmp/output",
        "quant_trade_path": "/tmp/quant_trade",
        "quant_trade_env": "",
        "yfinance_data_path": "/tmp/yfinance",
        "quant_trade_commit": "qt",
    }
    captured = {}

    def fake_run_one_week(**kwargs):
        captured.update(kwargs)
        return {"status": "success", "snapshot_date": kwargs["snapshot_date"]}

    monkeypatch.setattr(rb, "run_one_week", fake_run_one_week)
    row = rb._run_week_worker(("2024-01-05", "2024-01-05"))
    assert row["status"] == "success"
    assert captured["daily_data"] is daily
    assert captured["weekly_data"] is weekly
    assert captured["replay_old_pool"] == set()
    assert captured["replay_old_pool_source"] == "stateless_parallel_independent_week"
    assert list(inspect.signature(rb._run_week_worker).parameters) == ["task"]


def test_worker_eps_caches_merge_without_shared_writer(tmp_path: Path):
    final_cache = tmp_path / "eps.csv"
    pd.DataFrame(
        [{"snapshot_date": "2024-01-05", "code": "AAA", "eps_yoy_growth": 10.0, "retrieved_at": "a"}]
    ).to_csv(final_cache, index=False)
    worker_dir = tmp_path / "workers"
    worker_dir.mkdir()
    pd.DataFrame(
        [
            {"snapshot_date": "2024-01-05", "code": "AAA", "eps_yoy_growth": 10.0, "retrieved_at": "b"},
            {"snapshot_date": "2024-01-12", "code": "BBB", "eps_yoy_growth": 20.0, "retrieved_at": "b"},
        ]
    ).to_csv(worker_dir / "eps_pit_1.csv", index=False)
    pd.DataFrame(
        [{"snapshot_date": "2024-01-19", "code": "CCC", "eps_yoy_growth": 30.0, "retrieved_at": "c"}]
    ).to_csv(worker_dir / "eps_pit_2.csv", index=False)

    count = rb._merge_worker_eps_caches(worker_dir, final_cache)
    merged = pd.read_csv(final_cache, dtype={"code": str})
    assert count == 2
    assert list(zip(merged["snapshot_date"], merged["code"])) == [
        ("2024-01-05", "AAA"),
        ("2024-01-12", "BBB"),
        ("2024-01-19", "CCC"),
    ]


def test_worker_eps_cache_merge_fails_on_conflicting_same_key(tmp_path: Path):
    final_cache = tmp_path / "eps.csv"
    worker_dir = tmp_path / "workers"
    worker_dir.mkdir()
    pd.DataFrame(
        [{"snapshot_date": "2024-01-05", "code": "AAA", "eps_yoy_growth": 10.0}]
    ).to_csv(worker_dir / "eps_pit_1.csv", index=False)
    pd.DataFrame(
        [{"snapshot_date": "2024-01-05", "code": "AAA", "eps_yoy_growth": 11.0}]
    ).to_csv(worker_dir / "eps_pit_2.csv", index=False)
    with pytest.raises(RuntimeError, match="conflicting worker EPS PIT records"):
        rb._merge_worker_eps_caches(worker_dir, final_cache)
