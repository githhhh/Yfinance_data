import pandas as pd

from backtest.rd_agent_candidate_rule_audit.labels import (
    ExitPolicy,
    TradeLabelConfig,
    build_event_labels,
    classify_geometry,
    normalize_eps_pit,
)
from backtest.rd_agent_candidate_rule_audit.portfolio import PortfolioConfig, run_portfolio_backtest
from backtest.rd_agent_candidate_rule_audit.selectors import selector_configs, select_weekly
from backtest.rd_agent_candidate_rule_audit.stats import make_rolling_splits, week_block_bootstrap


def test_forward_return_is_censored_when_full_horizon_missing_and_asof_is_separate():
    event = _event("2026-01-02", "AAA", 100)
    prices = {"AAA": _bars([("2026-01-02", 99, 101, 98, 100), ("2026-01-05", 101, 103, 100, 102)])}

    labels = build_event_labels(pd.DataFrame([event]), prices, TradeLabelConfig())

    row = labels.iloc[0]
    assert pd.isna(row["forward_1w_return_pct"])
    assert bool(row["forward_1w_censored"]) is True
    assert row["as_of_return_pct"] == 0.990099


def test_mfe_mae_do_not_cross_registered_windows():
    event = _event("2026-01-02", "AAA", 100)
    prices = {
        "AAA": _bars(
            [
                ("2026-01-02", 99, 100, 99, 100),
                ("2026-01-05", 100, 101, 98, 100),
                ("2026-01-06", 100, 102, 97, 100),
                ("2026-01-07", 100, 150, 80, 100),
            ]
        )
    }

    labels = build_event_labels(pd.DataFrame([event]), prices, TradeLabelConfig(mfe_mae_days_3w=2))

    row = labels.iloc[0]
    assert row["mfe_3w_pct"] == 2.0
    assert row["mae_3w_pct"] == -3.0


def test_stop_profit_boundaries_gap_and_same_day_stop_first():
    prices = {
        "STOP": _bars([("2026-01-02", 99, 101, 99, 100), ("2026-01-05", 100, 110, 92, 100)]),
        "GAP": _bars([("2026-01-02", 99, 101, 99, 100), ("2026-01-05", 100, 101, 99, 100), ("2026-01-06", 91, 95, 90, 92)]),
        "PROFIT": _bars([("2026-01-02", 99, 101, 99, 100), ("2026-01-05", 100, 124, 99, 123)]),
        "BOTH": _bars([("2026-01-02", 99, 101, 99, 100), ("2026-01-05", 100, 124, 92, 110)]),
    }
    events = pd.DataFrame([_event("2026-01-02", code, 100) for code in prices])

    labels = build_event_labels(events, prices, TradeLabelConfig())

    by_code = labels.set_index("code")
    assert bool(by_code.loc["STOP", "stop_8_touched"]) is True
    assert by_code.loc["STOP", "realized_stop_loss_pct"] == -8.0
    assert bool(by_code.loc["GAP", "gap_stop_8"]) is True
    assert by_code.loc["GAP", "realized_stop_loss_pct"] == -9.0
    assert bool(by_code.loc["PROFIT", "profit_24_touched"]) is True
    assert by_code.loc["BOTH", "first_touch_8_24"] == "stop"
    assert bool(by_code.loc["BOTH", "same_day_path_ambiguous"]) is True


def test_power_trigger_uses_third_breakout_week_friday_not_fourth_week():
    events = pd.DataFrame(
        [
            _event("2026-01-09", "T3", 100, ibd_entry_date="2026-01-05"),
            _event("2026-01-09", "T4", 100, ibd_entry_date="2026-01-05"),
        ]
    )
    prices = {
        "T3": _bars([("2026-01-12", 100, 110, 99, 100), ("2026-01-23", 100, 120, 99, 119)]),
        "T4": _bars([("2026-01-12", 100, 110, 99, 100), ("2026-01-26", 100, 120, 99, 119)]),
    }

    labels = build_event_labels(events, prices, TradeLabelConfig())

    by_code = labels.set_index("code")
    assert bool(by_code.loc["T3", "power_trigger_3w_from_pivot"]) is True
    assert bool(by_code.loc["T4", "power_trigger_3w_from_pivot"]) is False


def test_pivot_power_and_entry_gain_are_separate_and_pre_entry_power_is_not_trade_power():
    event = _event("2026-01-09", "AAA", 100, ibd_entry_date="2026-01-05")
    prices = {
        "AAA": _bars(
            [
                ("2026-01-06", 100, 121, 99, 120),
                ("2026-01-12", 130, 131, 129, 130),
            ]
        )
    }

    labels = build_event_labels(pd.DataFrame([event]), prices, TradeLabelConfig())

    row = labels.iloc[0]
    assert bool(row["pattern_power_trigger"]) is True
    assert bool(row["trade_power_trigger"]) is False
    assert bool(row["gain_20_3w_from_entry"]) is False


def test_eight_week_lock_suspends_profit_but_keeps_stop_and_post_lock_resumes_profit():
    picks = pd.DataFrame([_event("2026-01-02", "AAA", 100, ibd_entry_date="2026-01-02")])
    prices = {
        "AAA": _bars(
            [
                ("2026-01-05", 100, 105, 99, 104),
                ("2026-01-16", 104, 121, 103, 120),
                ("2026-01-20", 120, 130, 119, 128),
                ("2026-02-23", 125, 126, 124, 125),
            ]
        )
    }

    trades, _, _ = run_portfolio_backtest(
        picks,
        prices,
        PortfolioConfig(capacity=3, initial_capital=10000, exit_policy=ExitPolicy(post_lock="resume_profit")),
    )

    trade = trades.iloc[0]
    assert trade["power_trigger_date"] == "2026-01-16"
    assert trade["exit_date"] == "2026-02-23"
    assert trade["exit_reason"] == "profit_target_post_lock_gap"


def test_post_lock_mark_to_market_does_not_resume_profit_exit():
    picks = pd.DataFrame([_event("2026-01-02", "AAA", 100, ibd_entry_date="2026-01-02")])
    prices = {
        "AAA": _bars(
            [
                ("2026-01-05", 100, 105, 99, 104),
                ("2026-01-16", 104, 121, 103, 120),
                ("2026-02-23", 125, 126, 124, 125),
            ]
        )
    }

    trades, _, _ = run_portfolio_backtest(
        picks,
        prices,
        PortfolioConfig(capacity=3, initial_capital=10000, exit_policy=ExitPolicy(post_lock="mark_to_market")),
    )

    trade = trades.iloc[0]
    assert bool(trade["censored"]) is True
    assert trade["exit_reason"] == "censored_mtm"


def test_pullback_dry_not_applicable_for_base_breakout_and_geometry_classes():
    assert classify_geometry(close_position=0.9, range_ratio=1.2) == "Full-range Breakout"
    assert classify_geometry(close_position=0.82, range_ratio=0.7) == "Strong Finish"
    assert classify_geometry(close_position=0.72, range_ratio=0.7) == "Constructive Breakout"
    assert classify_geometry(close_position=0.72, range_ratio=0.2) == "Marginal Breakout"
    assert classify_geometry(close_position=0.6, range_ratio=0.5) == "Squat / Upper Shadow"
    assert classify_geometry(close_position=0.8, range_ratio=-0.1) == "Defensive Failure"

    cfg = selector_configs()["B0_PIT_VERIFIED"]
    pick = select_weekly(pd.DataFrame([_event("2026-01-02", "AAA", 100, rule="ceiling_breakout")]), cfg)

    assert pick.iloc[0]["pullback_dry_state"] == "NOT_APPLICABLE"


def test_eps_unverified_becomes_unknown_before_scoring():
    eps = pd.DataFrame(
        [
            {
                "snapshot_date": "2026-01-02",
                "code": "AAA",
                "eps_yoy_growth": 99.0,
                "effective_date": "2025-12-31",
                "current_period": "2025-12-31",
                "status": "filled",
            }
        ]
    )

    normalized = normalize_eps_pit(eps)

    assert normalized.iloc[0]["pit_eps_state"] == "UNVERIFIED_AVAILABILITY"
    assert pd.isna(normalized.iloc[0]["pit_eps_yoy_growth"])


def test_b0_reproduction_and_atomic_ablation_only_changes_target_rule():
    events = pd.DataFrame(
        [
            _event("2026-01-02", "AAA", 100, eps=30, industry="A"),
            _event("2026-01-02", "BBB", 100, volume=1.2, eps=30, industry="B"),
            _event("2026-01-02", "CCC", 100, eps=30, industry="C"),
        ]
    )
    b0 = select_weekly(events, selector_configs()["B0_PIT_VERIFIED"])
    volume_soft = select_weekly(events, selector_configs()["B0 volume soft"])

    assert list(b0["code"]) == ["AAA", "CCC"]
    assert "BBB" in set(volume_soft["code"])
    assert set(b0.columns).issubset(set(volume_soft.columns))
    assert selector_configs()["B0 volume soft"].changed_rules == ("volume",)


def test_no_treatment_contrast_detected_when_selection_identical():
    events = pd.DataFrame([_event("2026-01-02", "AAA", 100, eps=30)])
    b0 = select_weekly(events, selector_configs()["B0_PIT_VERIFIED"])
    variant = select_weekly(events, selector_configs()["B0 close trigger soft"])

    assert set(b0["code"]) == set(variant["code"])


def test_rolling_split_has_embargo_and_week_block_bootstrap_keeps_week_records():
    weeks = pd.date_range("2026-01-02", periods=14, freq="W-FRI")
    frame = pd.DataFrame({"snapshot_date": weeks.astype(str), "value": range(len(weeks))})

    splits = make_rolling_splits(frame, test_weeks=3, embargo_weeks=8, min_train_weeks=3)
    assert splits[0].train_end < splits[0].test_start - pd.Timedelta(weeks=8)

    samples = week_block_bootstrap(frame.assign(effect=1.0), value_col="effect", seed=7, iterations=5)
    assert len(samples) == 5
    assert all(sample == 1.0 for sample in samples)


def test_portfolio_prevents_leverage_negative_cash_capacity_and_active_duplicates():
    picks = pd.DataFrame(
        [
            _event("2026-01-02", "AAA", 100),
            _event("2026-01-02", "BBB", 100),
            _event("2026-01-09", "AAA", 105),
        ]
    )
    prices = {
        "AAA": _bars([("2026-01-05", 100, 101, 99, 100), ("2026-01-12", 100, 101, 99, 100)]),
        "BBB": _bars([("2026-01-05", 100, 101, 99, 100), ("2026-01-12", 100, 101, 99, 100)]),
    }

    trades, equity, events = run_portfolio_backtest(
        picks,
        prices,
        PortfolioConfig(capacity=1, initial_capital=1000, cost_bps_per_side=25),
    )

    assert len(trades) == 1
    assert "capacity_skip" in set(events["event"])
    assert "repeat_signal_ignored" in set(events["event"])
    assert equity["cash"].min() >= 0
    assert equity["equity"].min() <= 1000


def _event(snapshot, code, pivot, **overrides):
    row = {
        "snapshot_date": snapshot,
        "code": code,
        "signal": True,
        "signal_source": "ceiling_breakout",
        "ibd_candidate_rule": "ceiling_breakout",
        "ibd_candidate_price": pivot,
        "ibd_entry_status": "ACTIONABLE",
        "ibd_entry_valid": True,
        "ibd_entry_date": snapshot,
        "ibd_entry_close_vs_trigger_pct": 1.0,
        "current_vs_ibd_candidate_pct": 1.0,
        "ibd_entry_volume_ratio": 1.6,
        "volume_ratio": 1.4,
        "ibd_entry_close_position": 0.9,
        "ibd_entry_breakout_range_ratio": 0.7,
        "pullback_v_is_dry": pd.NA,
        "eps_yoy_growth": overrides.pop("eps", 30.0),
        "pit_eps_yoy_growth": overrides.pop("pit_eps", 30.0),
        "pit_eps_state": overrides.pop("pit_eps_state", "VERIFIED"),
        "industry": overrides.pop("industry", "Software"),
        "sector": "Technology",
    }
    if "volume" in overrides:
        row["ibd_entry_volume_ratio"] = overrides.pop("volume")
    if "rule" in overrides:
        row["ibd_candidate_rule"] = overrides.pop("rule")
    row.update(overrides)
    return row


def _bars(rows):
    frame = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close"])
    frame["Volume"] = 100000
    return frame
