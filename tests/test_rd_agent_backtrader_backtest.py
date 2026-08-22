import pandas as pd

from backtest.rd_agent_research_bench.backtrader_backtest import run_backtrader_variant_backtest
from backtest.rd_agent_research_bench.research import backtrader_decision_matrix


def test_backtrader_variant_backtest_uses_initial_capital_and_backtrader_broker():
    picks = pd.DataFrame(
        [
            {"eps_mode": "with_eps", "variant": "sample", "snapshot_date": "2026-01-02", "code": "AAA", "pick_order": 1},
            {"eps_mode": "with_eps", "variant": "sample", "snapshot_date": "2026-01-02", "code": "BBB", "pick_order": 2},
        ]
    )
    prices = {
        "AAA": _bars(
            [
                ("2026-01-02", 100, 101, 99, 100),
                ("2026-01-05", 100, 112, 99, 110),
                ("2026-01-06", 110, 121, 109, 120),
            ]
        ),
        "BBB": _bars(
            [
                ("2026-01-02", 100, 101, 99, 100),
                ("2026-01-05", 100, 105, 99, 104),
                ("2026-01-06", 104, 109, 103, 108),
            ]
        ),
    }

    summary, trades, equity = run_backtrader_variant_backtest(
        picks,
        prices,
        eps_mode="with_eps",
        variant="sample",
        initial_capital=10000.0,
        stop_loss_pct=8.0,
    )

    assert summary["initial_capital"] == 10000.0
    assert summary["final_value"] > 10000.0
    assert summary["total_return_pct"] > 0
    assert summary["backtest_engine"] == "backtrader"
    assert summary["loaded_symbols"] == 2
    assert not trades.empty
    assert not equity.empty


def test_backtrader_variant_backtest_stop_order_limits_a_losing_position():
    picks = pd.DataFrame(
        [
            {"eps_mode": "with_eps", "variant": "sample", "snapshot_date": "2026-01-02", "code": "AAA", "pick_order": 1},
        ]
    )
    prices = {
        "AAA": _bars(
            [
                ("2026-01-02", 100, 101, 99, 100),
                ("2026-01-05", 100, 100, 99, 100),
                ("2026-01-06", 100, 101, 91, 95),
                ("2026-01-07", 95, 96, 80, 82),
            ]
        ),
    }

    summary, trades, _ = run_backtrader_variant_backtest(
        picks,
        prices,
        eps_mode="with_eps",
        variant="sample",
        initial_capital=10000.0,
        stop_loss_pct=8.0,
    )

    assert summary["stop_loss_pct"] == 8.0
    assert summary["final_value"] > 9000.0
    assert summary["total_return_pct"] > -10.0
    assert "stop" in set(trades["event"])


def test_backtrader_decision_matrix_blocks_direct_replacement_when_coverage_is_low():
    summary = pd.DataFrame(
        [
            {"eps_mode": "with_eps", "variant": "skill", "input_picks": 100, "rebalance_events": 40, "final_value": 15000.0, "total_return_pct": 50.0, "max_drawdown_pct": -15.0, "stop_events": 5},
            {"eps_mode": "with_eps", "variant": "guard", "input_picks": 70, "rebalance_events": 30, "final_value": 17000.0, "total_return_pct": 70.0, "max_drawdown_pct": -5.0, "stop_events": 1},
            {"eps_mode": "with_eps", "variant": "full", "input_picks": 95, "rebalance_events": 39, "final_value": 16000.0, "total_return_pct": 60.0, "max_drawdown_pct": -10.0, "stop_events": 3},
        ]
    )

    matrix = backtrader_decision_matrix(summary, eps_mode="with_eps", baseline_variant="skill")
    statuses = dict(zip(matrix["variant"], matrix["backtrader_status"]))

    assert statuses["guard"] == "high_confidence_candidate"
    assert statuses["full"] == "direct_replacement_candidate"


def _bars(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close"])
    frame["Volume"] = 100000
    return frame
