from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from backtest.blind_rule_discovery.outcomes import RESEARCH_PRICE_MODE
from backtest.blind_rule_discovery.pipeline_contract import (
    replay_dataset_digest,
    sha256_file,
    validate_replay_preflight,
)
from backtest.blind_rule_discovery.runner import _validate_split_safe_basis


def _write_one_week_replay(tmp_path: Path, *, declared_floor: int, quarters: int) -> tuple[Path, Path]:
    daily = tmp_path / "research_daily.pkl"
    daily.write_bytes(b"exact-daily")
    replay = tmp_path / "replay"
    week = replay / "2025-01-03"
    week.mkdir(parents=True)
    (week / "breakout_follow_pool.csv").write_text(
        "code,signal\nAAA,True\n", encoding="utf-8"
    )
    digest, _ = replay_dataset_digest(replay)
    (replay / "research_replay_preflight.json").write_text(
        json.dumps(
            {
                "price_adjustment_mode": RESEARCH_PRICE_MODE,
                "benchmark_code": "SPY",
                "warmup_failed_weeks": 0,
                "failed_weeks": 0,
                "analysis_weeks_expected": 1,
                "analysis_weeks_persisted": 1,
                "successful_quarters": quarters,
                "minimum_required_quarters": declared_floor,
                "daily_pkl_sha256": sha256_file(daily),
                "replay_dataset_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    return replay, daily


def test_stage1_cannot_ignore_replay_builders_stricter_quarter_floor(tmp_path: Path):
    replay, daily = _write_one_week_replay(tmp_path, declared_floor=14, quarters=13)
    with pytest.raises(ValueError, match="need 14"):
        validate_replay_preflight(replay, daily_pkl=daily, required_quarters=12)


def test_stage1_requires_every_mature_candidate_to_exist_in_exact_price_bundle():
    spy = pd.DataFrame()
    spy.attrs["price_adjustment_mode"] = RESEARCH_PRICE_MODE
    aaa = pd.DataFrame()
    aaa.attrs["price_adjustment_mode"] = RESEARCH_PRICE_MODE
    with pytest.raises(ValueError, match="missing from outcome prices"):
        _validate_split_safe_basis(
            {"AAA": aaa},
            {"SPY": spy},
            benchmark_code="SPY",
            candidate_codes={"AAA", "MISSING"},
        )
