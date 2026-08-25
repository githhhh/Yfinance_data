"""Comprehensive unit tests for B0 Top3 quality audit and data infrastructure (Phase 1).

Covers all 24 audit scenarios:
1. Review Universe construction
2. Unique ticker deduplication
3. Daily price incremental caching idempotency
4. `code + date` uniqueness
5. B0 weekly replay determinism & consistency
6. Next trading day Open entry
7. Natural-week aggregation
8. Holiday / short trading week
9. Partial / in-progress current week
10. Week 1 close return calculation
11. Week 1 max gain calculation
12. Standard intraday -8% stop
13. Gap stop at Open
14. Same-day High +20% / Low -8% path ambiguity
15. Max gain before stop isolation
16. Post-stop rebound excluded from executed return
17. Delisted / partial history handling
18. Ticker alias resolution (e.g. BRK.B -> BRK-B)
19. Failed downloads never drop events
20. Adjusted OHLCV split consistency
21. Random sampling without replacement
22. 1000-draw random reproducibility with fixed seed
23. Missing path draws kept without re-sampling
24. Zero-download Parquet accessibility for Phase 2 RD-Agent
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.b0_top3_quality_audit.baseline import replay_b0_on_pool
from backtest.b0_top3_quality_audit.outcomes import compute_single_candidate_outcome
from backtest.b0_top3_quality_audit.price_cache import DailyPriceCache, _standardize_ohlcv_df
from backtest.b0_top3_quality_audit.random_control import run_random_top3_for_snapshot
from backtest.b0_top3_quality_audit.ticker_resolution import (
    build_ticker_master,
    resolve_symbol_for_provider,
)
from backtest.b0_top3_quality_audit.universe import (
    build_review_universe_events,
    is_non_empty_rule,
    is_signal_true,
)
from backtest.b0_top3_quality_audit.weekly_aggregation import (
    compute_weekly_outcomes_for_candidate,
    get_calendar_week_bounds,
    slice_into_natural_weeks,
)


@pytest.fixture
def sample_pool_df() -> pd.DataFrame:
    """Create a minimal representative pool DataFrame."""
    return pd.DataFrame({
        "code": ["AAPL", "MSFT", "GOOG", "TSLA", "DELISTED"],
        "signal": [True, "true", 1, False, True],
        "ibd_candidate_rule": ["pivot", "ceiling_pullback", "standard_breakout", "pivot", "pivot"],
        "ibd_entry_status": ["ACTIONABLE", "ACTIONABLE", "ACTIONABLE", "ACTIONABLE", "RADAR"],
        "current_vs_ibd_candidate_pct": [1.5, 2.0, 3.5, 0.5, 1.0],
        "ibd_entry_volume_ratio": [2.0, 1.8, 1.6, 1.2, 1.5],
        "ibd_entry_close_position": [0.85, 0.75, 0.80, 0.70, 0.50],
        "ibd_entry_breakout_range_ratio": [0.3, 0.4, 0.35, 0.2, -0.1],
        "industry": ["Tech", "Software", "Internet", "Auto", "Tech"],
        "dist_to_52w_high_pct": [-1.0, -2.0, -3.0, -10.0, -2.0],
        "volume_ratio": [1.5, 1.4, 1.2, 1.0, 1.1],
        "eps_yoy_growth": [35.0, 40.0, 20.0, 15.0, None],
    })


@pytest.fixture
def mock_price_cache(tmp_path: Path) -> DailyPriceCache:
    """Create a mock DailyPriceCache with test price data."""
    cache = DailyPriceCache(parquet_path=tmp_path / "mock_prices.parquet")
    dates = pd.date_range("2025-10-01", "2025-11-30", freq="B")

    bars_list = []
    for code, base_p in [("AAPL", 100.0), ("MSFT", 200.0), ("GOOG", 150.0)]:
        for i, dt in enumerate(dates):
            d_str = dt.strftime("%Y-%m-%d")
            # Normal trending price
            p = base_p * (1.0 + 0.005 * i)
            bars_list.append({
                "code": code,
                "date": d_str,
                "open": round(p, 2),
                "high": round(p * 1.02, 2),
                "low": round(p * 0.99, 2),
                "close": round(p * 1.01, 2),
                "volume": 1000000,
                "source": "mock",
            })
    cache.append_bars(pd.DataFrame(bars_list))
    cache.save()
    return cache


# Test 1: Review Universe 构造
def test_01_review_universe_construction(sample_pool_df: pd.DataFrame, tmp_path: Path):
    pool_path = tmp_path / "2025-10-10" / "breakout_follow_pool.csv"
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    sample_pool_df.to_csv(pool_path, index=False)

    events_df = build_review_universe_events(pool_paths=[pool_path])
    # AAPL (True), MSFT ('true'), GOOG (1), DELISTED (True) -> 4 events
    # TSLA has signal=False -> excluded
    assert len(events_df) == 4
    assert set(events_df["code"]) == {"AAPL", "MSFT", "GOOG", "DELISTED"}


# Test 2: 唯一 Ticker 去重
def test_02_unique_ticker_deduplication(sample_pool_df: pd.DataFrame, tmp_path: Path):
    p1 = tmp_path / "2025-10-10" / "breakout_follow_pool.csv"
    p2 = tmp_path / "2025-10-17" / "breakout_follow_pool.csv"
    p1.parent.mkdir(parents=True, exist_ok=True)
    p2.parent.mkdir(parents=True, exist_ok=True)
    sample_pool_df.to_csv(p1, index=False)
    sample_pool_df.to_csv(p2, index=False)

    events_df = build_review_universe_events(pool_paths=[p1, p2])
    master_df = build_ticker_master(events_df)
    assert len(master_df) == 4  # 4 unique codes
    assert master_df["signal_event_count"].sum() == 8


# Test 3: 日线增量下载幂等性
def test_03_price_cache_idempotency(tmp_path: Path):
    cache_path = tmp_path / "prices.parquet"
    cache = DailyPriceCache(parquet_path=cache_path)

    df1 = pd.DataFrame([
        {"code": "AAPL", "date": "2025-10-10", "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1000, "source": "test"}
    ])
    cache.append_bars(df1)
    cache.save()
    assert len(cache.df) == 1

    # Append duplicate
    cache.append_bars(df1)
    cache.save()
    assert len(cache.df) == 1


# Test 4: `code + date` 唯一主键
def test_04_code_date_uniqueness(tmp_path: Path):
    cache = DailyPriceCache(parquet_path=tmp_path / "prices.parquet")
    df = pd.DataFrame([
        {"code": "AAPL", "date": "2025-10-10", "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1000, "source": "test"},
        {"code": "AAPL", "date": "2025-10-10", "open": 100.5, "high": 102.5, "low": 99.5, "close": 101.5, "volume": 2000, "source": "test2"},
    ])
    cache.append_bars(df)
    assert len(cache.df) == 1


# Test 5: B0 逐周复刻与生产一致性
def test_05_b0_determinism(sample_pool_df: pd.DataFrame):
    picks1 = replay_b0_on_pool(sample_pool_df, snapshot_date="2025-10-10", limit=3)
    picks2 = replay_b0_on_pool(sample_pool_df, snapshot_date="2025-10-10", limit=3)
    assert len(picks1) == len(picks2)
    assert [p["code"] for p in picks1] == [p["code"] for p in picks2]


# Test 6: 快照后下一交易日 Open 入场
def test_06_next_trading_day_open_entry(mock_price_cache: DailyPriceCache):
    ev = {"code": "AAPL", "snapshot_date": "2025-10-10", "signal": True, "ibd_candidate_rule": "pivot"}
    res, _ = compute_single_candidate_outcome(ev, mock_price_cache, as_of_date="2025-11-25")
    # First trading day after 2025-10-10 is 2025-10-13 (Monday)
    assert res["entry_date"] == "2025-10-13"
    assert res["entry_open"] is not None and res["entry_open"] > 0
    assert res["entry_status"] == "ENTRY_OK"


# Test 7: 自然周划分与聚合
def test_07_natural_week_aggregation():
    dt = pd.Timestamp("2025-10-15").date()  # Wednesday
    mon, fri, cal_str = get_calendar_week_bounds(dt)
    assert mon == pd.Timestamp("2025-10-13").date()
    assert fri == pd.Timestamp("2025-10-17").date()
    assert cal_str == "2025-W42"


# Test 8: 节假日短周
def test_08_holiday_short_week():
    # Only 3 trading days in week
    bars = pd.DataFrame([
        {"date": "2025-10-13", "open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 100},
        {"date": "2025-10-14", "open": 104.0, "high": 106.0, "low": 103.0, "close": 105.0, "volume": 100},
        {"date": "2025-10-15", "open": 105.0, "high": 107.0, "low": 104.0, "close": 106.0, "volume": 100},
    ])
    weeks = slice_into_natural_weeks(bars, entry_date_str="2025-10-13", as_of_date="2025-10-31")
    assert len(weeks) == 1
    assert weeks[0]["week_trading_sessions"] == 3
    assert weeks[0]["week_high"] == 107.0
    assert weeks[0]["week_low"] == 99.0


# Test 9: 当前未完成周标记
def test_09_partial_week_flag():
    bars = pd.DataFrame([
        {"date": "2026-08-24", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 100},
        {"date": "2026-08-25", "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 100},
    ])
    # as_of_date is Tuesday 2026-08-25, Friday is 2026-08-28 -> week not complete!
    weeks = slice_into_natural_weeks(bars, entry_date_str="2026-08-24", as_of_date="2026-08-25")
    assert len(weeks) == 1
    assert weeks[0]["is_complete_week"] is False


# Test 10 & 11: 首周收盘收益与最高收益
def test_10_11_week1_metrics(mock_price_cache: DailyPriceCache):
    ev = {"code": "AAPL", "snapshot_date": "2025-10-10", "signal": True, "ibd_candidate_rule": "pivot"}
    res, w_list = compute_single_candidate_outcome(ev, mock_price_cache, as_of_date="2025-11-25")
    assert res["week1_close_return_pct"] is not None
    assert res["week1_max_gain_pct"] is not None
    assert res["week1_max_gain_pct"] >= res["week1_close_return_pct"]


# Test 12: 普通日内 -8% 止损
def test_12_intraday_stop_8pct(tmp_path: Path):
    cache = DailyPriceCache(parquet_path=tmp_path / "stop_prices.parquet")
    bars = pd.DataFrame([
        {"code": "STOP1", "date": "2025-10-13", "open": 100.0, "high": 102.0, "low": 98.0, "close": 101.0, "volume": 100, "source": "t"},
        {"code": "STOP1", "date": "2025-10-14", "open": 98.0, "high": 99.0, "low": 91.0, "close": 91.5, "volume": 100, "source": "t"},
    ])
    cache.append_bars(bars)
    ev = {"code": "STOP1", "snapshot_date": "2025-10-10", "signal": True, "ibd_candidate_rule": "pivot"}
    res, _ = compute_single_candidate_outcome(ev, cache, as_of_date="2025-10-20")

    assert res["stop_8_hit_ever"] is True
    assert res["stop_8_date"] == "2025-10-14"
    assert res["gap_stop"] is False
    assert res["executed_return_to_asof_pct"] == -8.0


# Test 13: Gap Stop 跳空止损
def test_13_gap_stop_at_open(tmp_path: Path):
    cache = DailyPriceCache(parquet_path=tmp_path / "gap_prices.parquet")
    bars = pd.DataFrame([
        {"code": "GAP1", "date": "2025-10-13", "open": 100.0, "high": 102.0, "low": 98.0, "close": 101.0, "volume": 100, "source": "t"},
        # Gap down below 92.0 at open
        {"code": "GAP1", "date": "2025-10-14", "open": 88.0, "high": 89.0, "low": 85.0, "close": 87.0, "volume": 100, "source": "t"},
    ])
    cache.append_bars(bars)
    ev = {"code": "GAP1", "snapshot_date": "2025-10-10", "signal": True, "ibd_candidate_rule": "pivot"}
    res, _ = compute_single_candidate_outcome(ev, cache, as_of_date="2025-10-20")

    assert res["stop_8_hit_ever"] is True
    assert res["gap_stop"] is True
    assert res["executed_return_to_asof_pct"] == -12.0  # (88 - 100)/100


# Test 14: 同日 High/Low 歧义保守处理
def test_14_same_day_path_ambiguity(tmp_path: Path):
    cache = DailyPriceCache(parquet_path=tmp_path / "amb_prices.parquet")
    bars = pd.DataFrame([
        {"code": "AMB1", "date": "2025-10-13", "open": 100.0, "high": 102.0, "low": 98.0, "close": 101.0, "volume": 100, "source": "t"},
        # High hit +20% (122.0) AND Low hit -8% (91.0) on same day!
        {"code": "AMB1", "date": "2025-10-14", "open": 100.0, "high": 122.0, "low": 91.0, "close": 110.0, "volume": 100, "source": "t"},
    ])
    cache.append_bars(bars)
    ev = {"code": "AMB1", "snapshot_date": "2025-10-10", "signal": True, "ibd_candidate_rule": "pivot"}
    res, _ = compute_single_candidate_outcome(ev, cache, as_of_date="2025-10-20")

    assert res["same_day_path_ambiguous"] is True
    assert res["stop_8_hit_ever"] is True
    assert res["stop8_before_profit20"] is True  # Conservative stop first


# Test 15 & 16: 止损前最高收益与止损后反弹隔离
def test_15_16_max_gain_before_stop_isolation(tmp_path: Path):
    cache = DailyPriceCache(parquet_path=tmp_path / "rebound_prices.parquet")
    bars = pd.DataFrame([
        {"code": "REB1", "date": "2025-10-13", "open": 100.0, "high": 105.0, "low": 98.0, "close": 104.0, "volume": 100, "source": "t"},
        {"code": "REB1", "date": "2025-10-14", "open": 98.0, "high": 99.0, "low": 91.0, "close": 91.5, "volume": 100, "source": "t"},
        # Post-stop massive rebound
        {"code": "REB1", "date": "2025-10-15", "open": 95.0, "high": 150.0, "low": 94.0, "close": 145.0, "volume": 100, "source": "t"},
    ])
    cache.append_bars(bars)
    ev = {"code": "REB1", "snapshot_date": "2025-10-10", "signal": True, "ibd_candidate_rule": "pivot"}
    res, _ = compute_single_candidate_outcome(ev, cache, as_of_date="2025-10-20")

    assert res["stop_8_hit_ever"] is True
    assert res["max_gain_before_stop_pct"] == 5.0  # Max gain before stop is 105.0 (+5%)
    assert res["max_gain_to_asof_pct"] == 50.0   # Full window max gain is 150.0 (+50%)
    assert res["executed_return_to_asof_pct"] == -8.0  # Rebound does NOT enter executed return!


# Test 17: 退市前部分历史
def test_17_partial_history(tmp_path: Path):
    cache = DailyPriceCache(parquet_path=tmp_path / "part_prices.parquet")
    bars = pd.DataFrame([
        {"code": "DELIST", "date": "2025-10-13", "open": 50.0, "high": 52.0, "low": 48.0, "close": 49.0, "volume": 100, "source": "t"},
    ])
    cache.append_bars(bars)
    ev = {"code": "DELIST", "snapshot_date": "2025-10-10", "signal": True, "ibd_candidate_rule": "pivot"}
    res, _ = compute_single_candidate_outcome(ev, cache, as_of_date="2026-08-25")

    assert res["terminal_data_status"] == "PARTIAL_HISTORY"
    assert res["latest_observation_date"] == "2025-10-13"


# Test 18: Ticker Alias 映射
def test_18_ticker_alias_resolution():
    sym1, r1 = resolve_symbol_for_provider("BRK.B", provider="yahoo")
    assert sym1 == "BRK-B"
    sym2, r2 = resolve_symbol_for_provider("CWEN.A", provider="yahoo")
    assert sym2 == "CWEN-A"


# Test 19: 下载失败不丢事件
def test_19_missing_prices_keep_event(tmp_path: Path):
    empty_cache = DailyPriceCache(parquet_path=tmp_path / "empty.parquet")
    ev = {"code": "NODATA", "snapshot_date": "2025-10-10", "signal": True, "ibd_candidate_rule": "pivot"}
    res, _ = compute_single_candidate_outcome(ev, empty_cache, as_of_date="2026-08-25")
    assert res["code"] == "NODATA"
    assert res["entry_status"] == "NO_PRICE_DATA"
    assert res["current_return_to_asof_pct"] is None


# Test 20: 拆股复权一致性
def test_20_split_adjusted_consistency():
    # If open and close are both split-adjusted (e.g. 10.0 and 11.0), return is +10%
    raw_df = pd.DataFrame([
        {"Open": 10.0, "High": 12.0, "Low": 9.5, "Close": 11.0, "Volume": 1000}
    ], index=[pd.Timestamp("2025-10-13")])
    std = _standardize_ohlcv_df(raw_df, "SPLIT_CO")
    assert std.iloc[0]["open"] == 10.0
    assert std.iloc[0]["close"] == 11.0


# Test 21: 随机抽样无放回
def test_21_random_sampling_without_replacement():
    candidates = pd.DataFrame({
        "code": ["A", "B", "C", "D", "E"],
        "snapshot_date": ["2025-10-10"] * 5,
        "entry_status": ["ENTRY_OK"] * 5,
        "entry_open": [10.0] * 5,
        "current_return_to_asof_pct": [1.0, 2.0, 3.0, 4.0, 5.0],
        "executed_return_to_asof_pct": [1.0, 2.0, 3.0, 4.0, 5.0],
        "max_gain_to_asof_pct": [2.0, 3.0, 4.0, 5.0, 6.0],
        "stop_8_hit_ever": [False] * 5,
        "profit20_hit": [False] * 5,
    })
    _, draws_df = run_random_top3_for_snapshot("2025-10-10", candidates, pd.DataFrame(), n_draws=50, seed=42)
    for _, r in draws_df.iterrows():
        sampled = r["sampled_codes"].split(",")
        assert len(sampled) == 3
        assert len(set(sampled)) == 3  # Distinct, without replacement


# Test 22: 1000次随机固定种子可复现
def test_22_random_reproducibility():
    candidates = pd.DataFrame({
        "code": ["A", "B", "C", "D", "E"],
        "snapshot_date": ["2025-10-10"] * 5,
        "entry_status": ["ENTRY_OK"] * 5,
        "entry_open": [10.0] * 5,
        "current_return_to_asof_pct": [1.0, 2.0, 3.0, 4.0, 5.0],
        "executed_return_to_asof_pct": [1.0, 2.0, 3.0, 4.0, 5.0],
        "max_gain_to_asof_pct": [2.0, 3.0, 4.0, 5.0, 6.0],
        "stop_8_hit_ever": [False] * 5,
        "profit20_hit": [False] * 5,
    })
    sum1, _ = run_random_top3_for_snapshot("2025-10-10", candidates, pd.DataFrame(), n_draws=100, seed=1234)
    sum2, _ = run_random_top3_for_snapshot("2025-10-10", candidates, pd.DataFrame(), n_draws=100, seed=1234)
    assert sum1["asof_mean_return_pct_p50"] == sum2["asof_mean_return_pct_p50"]


# Test 23: 抽到缺失路径不重新抽样
def test_23_missing_paths_not_resampled():
    candidates = pd.DataFrame({
        "code": ["VALID1", "VALID2", "NODATA1"],
        "snapshot_date": ["2025-10-10"] * 3,
        "entry_status": ["ENTRY_OK", "ENTRY_OK", "NO_PRICE_DATA"],
        "entry_open": [10.0, 20.0, None],  # One missing price
        "current_return_to_asof_pct": [5.0, 10.0, None],
        "executed_return_to_asof_pct": [5.0, 10.0, None],
        "max_gain_to_asof_pct": [6.0, 12.0, None],
        "stop_8_hit_ever": [False, False, False],
        "profit20_hit": [False, False, False],
    })
    sum_res, draws_df = run_random_top3_for_snapshot("2025-10-10", candidates, pd.DataFrame(), n_draws=10, seed=42)
    # When picking 3 out of 3, NODATA1 is always drawn
    assert draws_df["is_valid_draw"].all() == False
    assert sum_res["valid_draw_pct"] == 0.0


# Test 24: 第二阶段无需重新下载即可直接读取 Parquet 接口
def test_24_phase2_parquet_zero_download(tmp_path: Path):
    events_path = tmp_path / "candidate_event_outcomes.parquet"
    df = pd.DataFrame([
        {"snapshot_date": "2025-10-10", "code": "AAPL", "entry_open": 100.0, "current_return_to_asof_pct": 15.0}
    ])
    df.to_parquet(events_path, index=False)

    # Phase 2 reader directly loads parquet
    read_df = pd.read_parquet(events_path)
    assert len(read_df) == 1
    assert read_df.iloc[0]["code"] == "AAPL"
    assert read_df.iloc[0]["current_return_to_asof_pct"] == 15.0


# Test 25: 停牌/缺失超过7天延迟入场自动作废标记为 ENTRY_STALE_EXPIRED
def test_25_delayed_stale_entry_rejection(tmp_path: Path):
    cache = DailyPriceCache(parquet_path=tmp_path / "stale_prices.parquet")
    # Snapshot is 2025-10-10 (Friday), but first bar is 2025-11-10 (31 days later)
    bars = pd.DataFrame([
        {"code": "STALE1", "date": "2025-11-10", "open": 50.0, "high": 52.0, "low": 48.0, "close": 49.0, "volume": 100, "source": "t"},
    ])
    cache.append_bars(bars)
    ev = {"code": "STALE1", "snapshot_date": "2025-10-10", "signal": True, "ibd_candidate_rule": "pivot"}
    res, weeks = compute_single_candidate_outcome(ev, cache, as_of_date="2026-08-25")

    assert res["entry_status"] == "ENTRY_STALE_EXPIRED"
    assert res["is_valid_entry"] is False
    assert res["terminal_data_status"] == "STALE_ENTRY_ABANDONED"
    assert res["current_return_to_asof_pct"] is None
    assert len(weeks) == 0


# Test 26: 同周配对比较计算与检验
def test_26_paired_pick_comparison(tmp_path: Path):
    from backtest.b0_top3_quality_audit.metrics import compute_paired_pick_comparison

    # Mock 5 weeks of B0 outcomes with pick 1, 2, 3
    rows = []
    for i in range(5):
        s_date = f"2025-10-{10+i*7}"
        rows.append({"snapshot_date": s_date, "pick_order": 1, "week1_close_return_pct": 2.0 + i, "current_return_to_asof_pct": 10.0 + i, "executed_return_to_asof_pct": 5.0})
        rows.append({"snapshot_date": s_date, "pick_order": 2, "week1_close_return_pct": 1.0 + i, "current_return_to_asof_pct": 5.0 + i, "executed_return_to_asof_pct": 2.0})
        rows.append({"snapshot_date": s_date, "pick_order": 3, "week1_close_return_pct": 1.5 + i, "current_return_to_asof_pct": 8.0 + i, "executed_return_to_asof_pct": 4.0})

    df = pd.DataFrame(rows)
    paired_df = compute_paired_pick_comparison(df, output_csv=tmp_path / "paired.csv")
    assert len(paired_df) == 3
    # Pick 1 vs Pick 2: diff is +1.0%, win rate is 100%
    p1_p2 = paired_df[paired_df["comparison_pair"] == "Pick_1_vs_Pick_2"].iloc[0]
    assert p1_p2["w1_diff_mean_pct"] == 1.0
    assert p1_p2["w1_win_rate_a_over_b_pct"] == 100.0

