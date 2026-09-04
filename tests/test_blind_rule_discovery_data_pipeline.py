from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

from backtest.blind_rule_discovery.outcomes import (
    RESEARCH_PRICE_MODE,
    is_split_safe_price_frame,
    load_price_pickle,
)
from backtest.blind_rule_discovery.replay_builder import (
    MIN_ANALYSIS_QUARTERS,
    MIN_BUNDLE_COVERAGE,
    _assert_research_bundle,
    assert_history_coverage,
    enumerate_snapshot_weeks_from_benchmark,
    validate_price_manifest,
)
from backtest.blind_rule_discovery.research_data import (
    collect_research_universe,
    daily_to_weekly,
)
from backtest.latest_quant_trade_replay.runner import sha256_file


def _daily(start: str, periods: int = 5, *, interval: str | None = None) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=periods)
    frame = pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(periods)],
            "High": [101.0 + i for i in range(periods)],
            "Low": [99.0 + i for i in range(periods)],
            "Close": [100.5 + i for i in range(periods)],
            "Volume": [1000 + i for i in range(periods)],
        },
        index=idx,
    )
    frame.attrs["price_adjustment_mode"] = RESEARCH_PRICE_MODE
    if interval is not None:
        frame.attrs["interval"] = interval
    return frame


def _write_bundle_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    daily_path = tmp_path / "research_daily.pkl"
    weekly_path = tmp_path / "research_weekly.pkl"
    with daily_path.open("wb") as handle:
        pickle.dump({"SPY": _daily("2017-01-03", interval="1d")}, handle)
    with weekly_path.open("wb") as handle:
        pickle.dump({"SPY": _daily("2017-01-03", interval="1wk")}, handle)
    manifest = {
        "schema_version": 1,
        "provider": "Yahoo Finance via yfinance",
        "yfinance_version": "test",
        "price_adjustment_mode": RESEARCH_PRICE_MODE,
        "daily_interval": "1d",
        "weekly_interval": "1wk",
        "weekly_source": "direct_yahoo_1wk_not_daily_resample",
        "auto_adjust": False,
        "repair": True,
        "rounding": False,
        "coverage": 1.0,
        "benchmark_codes_downloaded": ["SPY"],
        "daily_sha256": sha256_file(daily_path),
        "weekly_sha256": sha256_file(weekly_path),
    }
    manifest_path = tmp_path / "price_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return daily_path, weekly_path, manifest_path


def test_research_price_mode_does_not_apply_dividend_adj_close_factor(tmp_path: Path):
    frame = _daily("2024-01-02", periods=3)
    frame["Adj Close"] = [50.0, 51.0, 52.0]
    original_close = frame["Close"].tolist()
    path = tmp_path / "research.pkl"
    with path.open("wb") as handle:
        pickle.dump({"AAA": frame}, handle)

    loaded = load_price_pickle(path, require_adjusted=True)["AAA"]
    assert loaded["Close"].tolist() == original_close
    assert loaded.attrs["price_adjustment_mode"] == RESEARCH_PRICE_MODE
    assert is_split_safe_price_frame(loaded)


def test_legacy_adj_close_path_still_supported(tmp_path: Path):
    idx = pd.bdate_range("2024-01-02", periods=3)
    frame = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 50.0],
            "High": [102.0, 102.0, 51.0],
            "Low": [98.0, 98.0, 49.0],
            "Close": [100.0, 100.0, 50.0],
            "Adj Close": [50.0, 50.0, 50.0],
        },
        index=idx,
    )
    path = tmp_path / "legacy.pkl"
    with path.open("wb") as handle:
        pickle.dump({"AAA": frame}, handle)
    loaded = load_price_pickle(path, require_adjusted=True)["AAA"]
    assert loaded.attrs["price_adjustment_mode"] == "adj_close_factor"
    assert loaded["Close"].tolist() == [50.0, 50.0, 50.0]


def test_daily_to_weekly_labels_actual_last_trading_session():
    idx = pd.to_datetime(["2024-03-25", "2024-03-26", "2024-03-27", "2024-03-28"])
    frame = pd.DataFrame(
        {
            "Open": [100, 101, 102, 103],
            "High": [101, 102, 103, 104],
            "Low": [99, 100, 101, 102],
            "Close": [100.5, 101.5, 102.5, 103.5],
            "Volume": [10, 20, 30, 40],
        },
        index=idx,
    )
    frame.attrs["price_adjustment_mode"] = RESEARCH_PRICE_MODE
    weekly = daily_to_weekly(frame)
    assert weekly.index.tolist() == [pd.Timestamp("2024-03-28")]
    assert weekly.iloc[0]["Open"] == 100
    assert weekly.iloc[0]["Close"] == 103.5
    assert weekly.iloc[0]["Volume"] == 100
    assert weekly.attrs["price_adjustment_mode"] == RESEARCH_PRICE_MODE


def test_snapshot_calendar_uses_actual_benchmark_sessions_not_hardcoded_holidays():
    idx = pd.to_datetime(
        [
            "2024-03-25", "2024-03-26", "2024-03-27", "2024-03-28",
            "2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04", "2024-04-05",
        ]
    )
    benchmark = pd.DataFrame({"Close": range(len(idx))}, index=idx)
    weeks = enumerate_snapshot_weeks_from_benchmark(
        benchmark, start_date="2024-03-01", end_date="2024-04-05"
    )
    assert [week.snapshot_date for week in weeks] == ["2024-03-28", "2024-04-05"]


def test_research_bundle_preflight_rejects_unverified_price_basis():
    verified = _daily("2024-01-02")
    unverified = verified.copy()
    unverified.attrs.clear()
    with pytest.raises(ValueError, match="verified price mode"):
        _assert_research_bundle({"SPY": verified, "AAA": unverified}, benchmark_code="SPY")


def test_research_bundle_preflight_rejects_wrong_interval():
    frame = _daily("2024-01-02", interval="1wk")
    with pytest.raises(ValueError, match="wrong interval"):
        _assert_research_bundle(
            {"SPY": frame}, benchmark_code="SPY", expected_interval="1d"
        )


def test_price_manifest_hash_must_match_exact_bundle(tmp_path: Path):
    daily_path, weekly_path, manifest_path = _write_bundle_files(tmp_path)
    manifest = validate_price_manifest(
        manifest_path, daily_pkl=daily_path, weekly_pkl=weekly_path
    )
    assert manifest["price_adjustment_mode"] == RESEARCH_PRICE_MODE

    with daily_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="daily pkl SHA256"):
        validate_price_manifest(
            manifest_path, daily_pkl=daily_path, weekly_pkl=weekly_path
        )


def test_price_manifest_rejects_wrong_price_semantics(tmp_path: Path):
    daily_path, weekly_path, manifest_path = _write_bundle_files(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["price_adjustment_mode"] = "unknown"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="price_adjustment_mode"):
        validate_price_manifest(
            manifest_path, daily_pkl=daily_path, weekly_pkl=weekly_path
        )


def test_price_manifest_rejects_low_coverage_even_if_builder_was_overridden(tmp_path: Path):
    daily_path, weekly_path, manifest_path = _write_bundle_files(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["coverage"] = MIN_BUNDLE_COVERAGE - 0.01
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="below canonical minimum"):
        validate_price_manifest(
            manifest_path, daily_pkl=daily_path, weekly_pkl=weekly_path
        )


def test_history_preflight_requires_five_year_context_before_warmup():
    short = _daily("2020-01-02", periods=700)
    with pytest.raises(ValueError, match="starts too late"):
        assert_history_coverage(
            short,
            warmup_start="2022-07-01",
            analysis_end="2022-12-30",
            min_lookback_days=5 * 365,
        )

    long = _daily("2017-01-03", periods=1600)
    coverage = assert_history_coverage(
        long,
        warmup_start="2022-07-01",
        analysis_end="2022-12-30",
        min_lookback_days=5 * 365,
    )
    assert coverage["first_session"] == "2017-01-03"


def test_collect_research_universe_unions_strategy_replay_seed_and_benchmarks(tmp_path: Path):
    (tmp_path / "us").mkdir()
    pd.DataFrame({"code": ["AAA"]}).to_csv(tmp_path / "us" / "breakout_follow_pool.csv", index=False)
    replay_root = tmp_path / "replay"
    (replay_root / "2025-01-03").mkdir(parents=True)
    pd.DataFrame({"code": ["BBB"]}).to_csv(
        replay_root / "2025-01-03" / "breakout_follow_pool.csv", index=False
    )
    seed = tmp_path / "seed.pkl"
    with seed.open("wb") as handle:
        pickle.dump({"CCC": _daily("2024-01-02")}, handle)

    universe = collect_research_universe(
        data_root=tmp_path, replay_roots=[replay_root], seed_pkl=seed
    )
    assert {"AAA", "BBB", "CCC", "SPY", "^GSPC"}.issubset(set(universe))


def test_canonical_replay_capacity_is_fourteen_quarters():
    assert MIN_ANALYSIS_QUARTERS == 14
    assert MIN_BUNDLE_COVERAGE == 0.98
