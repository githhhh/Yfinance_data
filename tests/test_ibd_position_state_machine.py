import pandas as pd

from backtest.rd_agent_research_bench.ibd_position_state_machine import (
    IBDTradeConfig,
    run_ibd_position_state_machine,
)


def test_enters_at_next_trading_day_open_and_marks_open_trade_censored():
    picks = _picks([("2026-01-02", "AAA", 100.0)])
    prices = {"AAA": _bars([("2026-01-02", 99, 101, 98, 100), ("2026-01-05", 103, 104, 99, 102)])}

    ledger, events = run_ibd_position_state_machine(picks, prices, IBDTradeConfig())

    trade = ledger.iloc[0]
    assert trade["entry_date"] == "2026-01-05"
    assert trade["entry_fill_price"] == 103.0
    assert trade["state"] == "OPEN"
    assert bool(trade["censored"]) is True
    assert events.iloc[0]["event"] == "entry"


def test_gap_through_stop_exits_at_open_when_open_is_below_stop():
    picks = _picks([("2026-01-02", "AAA", 100.0)])
    prices = {
        "AAA": _bars(
            [
                ("2026-01-02", 100, 101, 99, 100),
                ("2026-01-05", 100, 101, 99, 100),
                ("2026-01-06", 91, 93, 90, 92),
            ]
        )
    }

    ledger, _ = run_ibd_position_state_machine(picks, prices, IBDTradeConfig(stop_loss_pct=8.0))

    trade = ledger.iloc[0]
    assert trade["exit_date"] == "2026-01-06"
    assert trade["exit_fill_price"] == 91.0
    assert trade["exit_reason"] == "gap_stop"
    assert trade["state"] == "CLOSED"


def test_third_breakout_week_power_trigger_locks_and_suspends_profit_taking():
    picks = _picks([("2026-01-02", "AAA", 100.0)])
    prices = {
        "AAA": _bars(
            [
                ("2026-01-02", 100, 101, 99, 100),
                ("2026-01-05", 100, 105, 99, 104),
                ("2026-01-12", 104, 115, 103, 114),
                ("2026-01-16", 114, 121, 113, 120),
                ("2026-01-20", 120, 130, 119, 128),
                ("2026-02-27", 128, 129, 127, 128),
            ]
        )
    }

    ledger, events = run_ibd_position_state_machine(picks, prices, IBDTradeConfig(profit_take_pct=20.0))

    trade = ledger.iloc[0]
    assert trade["power_trigger_date"] == "2026-01-16"
    assert trade["minimum_hold_until"] == "2026-02-20"
    assert trade["state"] == "OPEN"
    assert bool(trade["censored"]) is True
    assert "profit_take" not in set(events["event"])


def test_fourth_breakout_week_profit_does_not_lock_and_can_take_profit():
    picks = _picks([("2026-01-02", "AAA", 100.0)])
    prices = {
        "AAA": _bars(
            [
                ("2026-01-02", 100, 101, 99, 100),
                ("2026-01-05", 100, 104, 99, 103),
                ("2026-01-12", 103, 110, 102, 109),
                ("2026-01-20", 109, 119, 108, 118),
                ("2026-01-26", 118, 122, 117, 121),
            ]
        )
    }

    ledger, _ = run_ibd_position_state_machine(picks, prices, IBDTradeConfig(profit_take_pct=20.0))

    trade = ledger.iloc[0]
    assert pd.isna(trade["power_trigger_date"])
    assert trade["exit_reason"] == "profit_take"
    assert trade["exit_fill_price"] == 120.0


def test_repeated_signal_does_not_reset_existing_position():
    picks = _picks([("2026-01-02", "AAA", 100.0), ("2026-01-09", "AAA", 108.0)])
    prices = {
        "AAA": _bars(
            [
                ("2026-01-02", 100, 101, 99, 100),
                ("2026-01-05", 100, 105, 99, 104),
                ("2026-01-09", 108, 110, 107, 109),
                ("2026-01-12", 109, 111, 108, 110),
            ]
        )
    }

    ledger, events = run_ibd_position_state_machine(picks, prices, IBDTradeConfig())

    assert len(ledger) == 1
    assert ledger.iloc[0]["signal_date"] == "2026-01-02"
    assert list(events["event"]).count("repeat_signal_ignored") == 1


def test_same_day_stop_and_power_trigger_uses_conservative_stop_first():
    picks = _picks([("2026-01-02", "AAA", 100.0)])
    prices = {
        "AAA": _bars(
            [
                ("2026-01-02", 100, 101, 99, 100),
                ("2026-01-05", 100, 101, 99, 100),
                ("2026-01-12", 100, 105, 99, 104),
                ("2026-01-16", 104, 121, 91, 118),
            ]
        )
    }

    ledger, _ = run_ibd_position_state_machine(picks, prices, IBDTradeConfig(stop_loss_pct=8.0))

    trade = ledger.iloc[0]
    assert trade["exit_reason"] == "stop_loss"
    assert pd.isna(trade["power_trigger_date"])


def test_power_trigger_uses_ibd_entry_date_as_breakout_week_anchor_when_available():
    picks = _picks([("2026-01-09", "AAA", 100.0)])
    picks.loc[0, "ibd_entry_date"] = "2025-12-26"
    prices = {
        "AAA": _bars(
            [
                ("2026-01-09", 100, 101, 99, 100),
                ("2026-01-12", 100, 105, 99, 104),
                ("2026-01-16", 104, 121, 103, 120),
            ]
        )
    }

    ledger, _ = run_ibd_position_state_machine(picks, prices, IBDTradeConfig(profit_take_pct=20.0))

    trade = ledger.iloc[0]
    assert trade["breakout_week"] == "2025-12-22"
    assert pd.isna(trade["power_trigger_date"])


def _picks(rows):
    return pd.DataFrame(
        [
            {
                "snapshot_date": snapshot,
                "code": code,
                "ibd_candidate_price": pivot,
                "signal_source": "ceiling_breakout",
                "ibd_candidate_rule": "ceiling",
                "pick_order": idx + 1,
            }
            for idx, (snapshot, code, pivot) in enumerate(rows)
        ]
    )


def _bars(rows):
    frame = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close"])
    frame["Volume"] = 100000
    return frame
