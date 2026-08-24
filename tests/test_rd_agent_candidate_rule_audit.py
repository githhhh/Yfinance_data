import json
import pickle
from pathlib import Path

import pandas as pd

from backtest.rd_agent_candidate_rule_audit.labels import (
    ExitPolicy,
    TradeLabelConfig,
    build_event_labels,
    classify_geometry,
    normalize_eps_pit,
)
from backtest.rd_agent_candidate_rule_audit.portfolio import PortfolioConfig, portfolio_metrics, run_portfolio_backtest
from backtest.rd_agent_candidate_rule_audit import evaluation as evaluation_module
from backtest.rd_agent_candidate_rule_audit import run as audit_run
from backtest.rd_agent_candidate_rule_audit import selectors as selector_module
from backtest.rd_agent_candidate_rule_audit.selectors import compose_selector_config, selector_configs, select_weekly
from backtest.rd_agent_candidate_rule_audit.stats import RollingSplit, make_rolling_splits, paired_week_route_bootstrap, week_block_bootstrap
from backtest.rd_agent_candidate_rule_audit.run import machine_rule_decisions, rule_treatment_contrast
from dashboard.skill_industry_eps_known import select_skill_industry_eps_known


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
    assert bool(by_code.loc["PROFIT", "profit_24_within_40d"]) is True
    assert by_code.loc["BOTH", "first_touch_8_24"] == "stop"
    assert by_code.loc["BOTH", "first_touch_40d_8_24"] == "stop"
    assert bool(by_code.loc["BOTH", "same_day_path_ambiguous"]) is True


def test_candidate_touch_labels_are_limited_to_40_trading_days():
    event = _event("2026-01-02", "AAA", 100)
    rows = [("2026-01-02", 99, 100, 99, 100)]
    for idx in range(1, 42):
        date = pd.Timestamp("2026-01-02") + pd.offsets.BDay(idx)
        high = 124 if idx == 41 else 101
        rows.append((str(date.date()), 100, high, 99, 100))
    prices = {"AAA": _bars(rows)}

    labels = build_event_labels(pd.DataFrame([event]), prices, TradeLabelConfig())

    row = labels.iloc[0]
    assert bool(row["profit_24_touched"]) is True
    assert bool(row["profit_24_within_40d"]) is False
    assert row["profit_24_within_40d_date"] == ""
    assert row["first_touch_40d_8_24"] == ""


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


def test_entry_gain_20_first_15_trading_days_is_not_truncated_by_pivot_power_window():
    event = _event("2026-01-09", "AAA", 100, ibd_entry_date="2026-01-05")
    rows = [
        ("2026-01-09", 100, 101, 99, 100),
        ("2026-01-12", 100, 101, 99, 100),
    ]
    for idx in range(2, 16):
        date = pd.Timestamp("2026-01-09") + pd.offsets.BDay(idx)
        rows.append((str(date.date()), 100, 120 if idx == 14 else 101, 99, 100))
    prices = {"AAA": _bars(rows)}

    labels = build_event_labels(pd.DataFrame([event]), prices, TradeLabelConfig())

    row = labels.iloc[0]
    assert bool(row["power_trigger_3w_from_pivot"]) is False
    assert bool(row["gain_20_within_first_15_trading_days"]) is True


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


def test_ordinary_stop_profit_policy_does_not_enable_eight_week_power_lock():
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
        PortfolioConfig(
            capacity=3,
            initial_capital=10000,
            exit_policy=ExitPolicy(enable_power_lock=False),
        ),
    )

    trade = trades.iloc[0]
    assert trade["power_trigger_date"] == ""
    assert trade["exit_date"] == "2026-01-20"
    assert trade["exit_reason"] == "profit_target"


def test_optimistic_same_day_sensitivity_exits_profit_when_daily_bar_hits_stop_and_target():
    pick = pd.DataFrame([_event("2026-01-02", "BOTH", 100, ibd_entry_date="")])
    prices = {
        "BOTH": _bars(
            [
                ("2026-01-05", 100, 124, 92, 110),
                ("2026-01-06", 110, 110, 110, 110),
            ]
        )
    }

    trades, _, _ = run_portfolio_backtest(
        pick,
        prices,
        PortfolioConfig(
            capacity=1,
            exit_policy=ExitPolicy(enable_power_lock=False, same_day_order="profit_first"),
        ),
    )

    assert trades.iloc[0]["exit_reason"] == "profit_target"
    assert trades.iloc[0]["exit_fill_price"] == 124.0


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

    production_codes = [item.code for item in select_skill_industry_eps_known(events)]
    assert list(b0["code"]) == production_codes
    assert list(b0["pick_order"]) == list(range(1, len(production_codes) + 1))
    assert list(b0["reason_codes"]) == [";".join(item.reason_codes) for item in select_skill_industry_eps_known(events)]
    assert list(b0["risk_codes"]) == [";".join(item.risk_codes) for item in select_skill_industry_eps_known(events)]
    assert set(b0["code"]) == {"AAA", "BBB", "CCC"}
    assert "BBB" in set(volume_soft["code"])
    assert set(b0.columns).issubset(set(volume_soft.columns))
    assert selector_configs()["B0 volume soft"].changed_rules == ("volume",)


def test_real_replay_pool_repo_b0_matches_production_codes_order_reasons_and_risks():
    audit = selector_module.audit_production_b0_replay_pools(Path("backtest/ibd_skill_replay_pools"))

    assert audit["snapshot_date"].nunique() == 42
    assert audit["production_selected_count"].sum() == 97
    assert audit["code_order_mismatches"].sum() == 0
    assert audit["reason_code_mismatches"].sum() == 0
    assert audit["risk_code_mismatches"].sum() == 0
    assert audit["parameterized_baseline_mismatches"].sum() == 0


def test_atomic_invariant_audits_real_rule_traces_and_identical_variants_are_blocked():
    events = pd.DataFrame(
        [
            _event("2026-01-02", "AAA", 100, eps=30, industry="A"),
            _event("2026-01-02", "BBB", 100, eps=30, industry="B"),
            _event(
                "2026-01-02",
                "DDD",
                100,
                eps=30,
                industry="D",
                ibd_entry_status="UNCONFIRMED",
            ),
        ]
    )

    no_entry_valid = selector_module.audit_atomic_variant(
        events,
        selector_configs()["B0_PIT_VERIFIED"],
        selector_configs()["B0 no entry_valid"],
    )
    status_soft = selector_module.audit_atomic_variant(
        events,
        selector_configs()["B0_PIT_VERIFIED"],
        selector_configs()["B0 status supplemental UNCONFIRMED"],
    )

    assert no_entry_valid["treatment_contrast"] == "NO_TREATMENT_CONTRAST"
    assert no_entry_valid["actual_target_trace_changes"] == 0
    assert no_entry_valid["non_target_trace_violations"] == 0
    assert status_soft["treatment_contrast"] == "OK"
    assert status_soft["actual_target_trace_changes"] > 0
    assert status_soft["non_target_trace_violations"] == 0


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

    paired = pd.DataFrame(
        {
            "snapshot_date": ["2026-01-02", "2026-01-02", "2026-01-09", "2026-01-09"],
            "signal_source": ["pivot", "ceiling", "pivot", "ceiling"],
            "treated": [2.0, 4.0, 6.0, 8.0],
            "control": [1.0, 1.0, 1.0, 1.0],
        }
    )
    pair_samples = paired_week_route_bootstrap(
        paired,
        treated_col="treated",
        control_col="control",
        seed=7,
        iterations=5,
        time_block_weeks=1,
    )
    assert len(pair_samples) == 5
    assert all(sample > 0 for sample in pair_samples)


def test_eight_week_outcomes_use_contiguous_time_blocks_and_do_not_claim_ci_before_one_full_block():
    short = pd.DataFrame(
        {
            "snapshot_date": pd.date_range("2026-01-02", periods=7, freq="W-FRI").astype(str),
            "effect": 1.0,
        }
    )
    mature = pd.DataFrame(
        {
            "snapshot_date": pd.date_range("2025-01-03", periods=16, freq="W-FRI").astype(str),
            "effect": 1.0,
        }
    )

    assert week_block_bootstrap(short, value_col="effect", seed=7, iterations=5, time_block_weeks=8) == []
    samples = week_block_bootstrap(mature, value_col="effect", seed=7, iterations=5, time_block_weeks=8)
    assert samples == [1.0] * 5


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


def test_portfolio_recycles_cash_after_closed_trade_and_equity_starts_at_first_entry():
    picks = pd.DataFrame(
        [
            _event("2026-01-02", "AAA", 100, ibd_entry_date=""),
            _event("2026-01-09", "BBB", 100, ibd_entry_date=""),
        ]
    )
    prices = {
        "AAA": _bars(
            [
                ("2025-01-02", 10, 10, 10, 10),
                ("2026-01-05", 100, 124, 99, 124),
                ("2026-01-06", 124, 124, 124, 124),
            ]
        ),
        "BBB": _bars(
            [
                ("2025-01-02", 10, 10, 10, 10),
                ("2026-01-12", 100, 124, 99, 124),
                ("2026-01-13", 124, 124, 124, 124),
            ]
        ),
    }

    trades, equity, events = run_portfolio_backtest(
        picks,
        prices,
        PortfolioConfig(capacity=1, initial_capital=1000),
    )

    assert len(trades) == 2
    assert "cash_skip" not in set(events["event"])
    assert equity.iloc[0]["date"] == "2026-01-05"
    assert equity["cash"].min() >= 0


def test_portfolio_with_closed_and_open_positions_values_through_explicit_common_asof():
    picks = pd.DataFrame(
        [
            _event("2026-01-02", "CLOSED", 100, ibd_entry_date=""),
            _event("2026-01-09", "OPEN", 100, ibd_entry_date=""),
        ]
    )
    prices = {
        "CLOSED": _bars(
            [
                ("2026-01-05", 100, 124, 99, 124),
                ("2026-01-06", 124, 124, 124, 124),
                ("2026-01-30", 124, 124, 124, 124),
            ]
        ),
        "OPEN": _bars(
            [
                ("2026-01-12", 100, 101, 99, 100),
                ("2026-01-30", 100, 101, 99, 101),
            ]
        ),
    }

    trades, equity, _ = run_portfolio_backtest(
        picks,
        prices,
        PortfolioConfig(capacity=2, initial_capital=2000, valuation_as_of="2026-01-30"),
    )

    assert set(trades["censored"]) == {False, True}
    assert equity.iloc[-1]["date"] == "2026-01-30"
    assert equity.iloc[-1]["open_positions"] == 1


def test_pit_b0_does_not_consume_production_eps_fallback(monkeypatch):
    import dashboard.skill_industry_eps_known as production_selector

    monkeypatch.setattr(production_selector, "row_eps", lambda row, code: 999.0)
    event = _event(
        "2026-01-02",
        "AAA",
        100,
        eps=pd.NA,
        pit_eps=pd.NA,
        pit_eps_state="UNKNOWN",
    )

    selected = select_weekly(pd.DataFrame([event]), selector_configs()["B0_PIT_VERIFIED"])

    assert selected.empty


def test_candidate_panel_ignores_forged_pool_pit_columns_and_uses_only_audited_eps_table():
    raw_event = _event(
        "2026-01-02",
        "AAA",
        100,
        pit_eps=999.0,
        pit_eps_state="VERIFIED",
    )
    pool = pd.DataFrame([raw_event])
    audited_eps = normalize_eps_pit(
        pd.DataFrame(
            [
                {
                    "snapshot_date": "2026-01-02",
                    "code": "AAA",
                    "eps_yoy_growth": 30.0,
                    "effective_date": "2025-12-31",
                    "current_period": "2025-12-31",
                    "status": "filled",
                    "source": "audited_test",
                }
            ]
        )
    )
    prices = {
        "AAA": _bars(
            [
                (str((pd.Timestamp("2026-01-02") + pd.offsets.BDay(idx)).date()), 100, 101, 99, 100)
                for idx in range(45)
            ]
        )
    }

    panel, _ = audit_run.build_candidate_event_panel(
        [("2026-01-02", pool, Path("test_pool.csv"))],
        audited_eps,
        prices,
    )
    selected = select_weekly(panel, selector_configs()["B0_PIT_VERIFIED"])

    assert panel.iloc[0]["pit_eps_state"] == "UNVERIFIED_AVAILABILITY"
    assert pd.isna(panel.iloc[0]["pit_eps_yoy_growth"])
    assert selected.empty


def test_fold_candidate_proposal_uses_train_only_and_freezes_rule_hash():
    weeks = pd.date_range("2025-01-03", periods=20, freq="W-FRI")
    rows = []
    for idx, week in enumerate(weeks):
        rows.append(_labeled_event(str(week.date()), f"A{idx}", "ceiling_breakout", "NOT_APPLICABLE", 1.0, 1.6, 2.0))
        rows.append(_labeled_event(str(week.date()), f"B{idx}", "pivot", "PASS", 1.0, 1.6, 1.0))
    panel = pd.DataFrame(rows)
    split = make_rolling_splits(panel, test_weeks=4, embargo_weeks=8, min_train_weeks=4)[-1]

    first = audit_run.propose_fold_candidates(panel, split, bootstrap_iterations=20)
    mutated = panel.copy()
    test_mask = pd.to_datetime(mutated["snapshot_date"]).between(split.test_start, split.test_end)
    mutated.loc[test_mask, "forward_8w_return_pct"] = 9999.0
    second = audit_run.propose_fold_candidates(mutated, split, bootstrap_iterations=20)

    assert first["train_end"] == str(split.train_end.date())
    assert pd.Timestamp(first["train_evidence_max_date"]) <= split.train_end
    assert first["frozen_rule_hash"] == second["frozen_rule_hash"]
    assert json.loads(first["frozen_rules_json"]) == json.loads(second["frozen_rules_json"])


def test_fold_candidate_proposal_does_not_filter_selector_universe_by_label_completion(monkeypatch):
    weeks = pd.date_range("2025-01-03", periods=20, freq="W-FRI")
    panel = pd.DataFrame(
        [
            _labeled_event(str(week.date()), f"A{idx}", "ceiling_breakout", "NOT_APPLICABLE", 1.0, 1.6, 2.0)
            for idx, week in enumerate(weeks)
        ]
    )
    split = make_rolling_splits(panel, test_weeks=4, embargo_weeks=8, min_train_weeks=4)[-1]
    train_mask = pd.to_datetime(panel["snapshot_date"]).between(split.train_start, split.train_end)
    censored_index = panel[train_mask].index[0]
    panel.loc[censored_index, "forward_8w_censored"] = True
    observed = {}

    def capture_training_frame(train, *, bootstrap_iterations, seed):
        observed["rows"] = len(train)
        observed["censored"] = int(train["forward_8w_censored"].eq(True).sum())
        return pd.DataFrame()

    monkeypatch.setattr(evaluation_module, "evaluate_atomic_training", capture_training_frame)

    audit_run.propose_fold_candidates(panel, split, bootstrap_iterations=5)

    assert observed == {"rows": int(train_mask.sum()), "censored": 1}


def test_fold_candidate_proposal_revalidates_composed_rule_set_before_freezing(monkeypatch):
    weeks = pd.date_range("2025-01-03", periods=20, freq="W-FRI")
    panel = pd.DataFrame(
        [
            _labeled_event(str(week.date()), f"A{idx}", "ceiling_breakout", "NOT_APPLICABLE", 1.0, 1.6, 2.0)
            for idx, week in enumerate(weeks)
        ]
    )
    split = make_rolling_splits(panel, test_weeks=4, embargo_weeks=8, min_train_weeks=4)[-1]
    atomic_evidence = pd.DataFrame(
        [
            {
                "variant": "B0 top1",
                "target_rule": "topk",
                "rule_family": "TopK",
                "train_decision": "VALIDATED",
                "mean_return_diff_pct": 2.0,
                "return_ci_low_pct": 0.5,
                "stop_40d_diff_pp": 0.0,
                "industry_top_share_diff": 0.0,
            }
        ]
    )
    no_composite_contrast = {
        "treatment_contrast": "NO_TREATMENT_CONTRAST",
        "selected_count": 8,
        "baseline_selected_count": 8,
        "added": 0,
        "removed": 0,
        "rank_changed": 0,
        "affected_weeks": 0,
        "mature_weeks": 8,
        "complete_outcomes": 8,
        "paired_weeks": 8,
        "mean_return_diff_pct": 2.0,
        "median_return_diff_pct": 2.0,
        "return_ci_low_pct": 0.5,
        "return_ci_high_pct": 3.5,
        "stop_40d_diff_pp": 0.0,
        "profit_24_40d_diff_pp": 0.0,
        "coverage_ratio": 1.0,
        "industry_top_share": 0.2,
        "baseline_industry_top_share": 0.2,
        "industry_top_share_diff": 0.0,
    }
    monkeypatch.setattr(evaluation_module, "evaluate_atomic_training", lambda *args, **kwargs: atomic_evidence)
    monkeypatch.setattr(evaluation_module, "compare_selected_outcomes", lambda *args, **kwargs: no_composite_contrast.copy())

    proposal = audit_run.propose_fold_candidates(panel, split, bootstrap_iterations=5)

    assert proposal["candidate_generation_status"] == "NO_STABLE_CANDIDATE"
    assert proposal["candidate_configs"] == []
    composite = proposal["train_evidence"]
    assert composite[composite["variant"].eq("R1_ATOMIC_IMPROVEMENTS")].iloc[0]["train_decision"] == "NOT_IDENTIFIABLE"


def test_each_frozen_candidate_result_hashes_its_own_config():
    test = pd.DataFrame(
        [_labeled_event("2026-01-02", "AAA", "ceiling_breakout", "NOT_APPLICABLE", 1.0, 1.6, 2.0)]
    )
    split = RollingSplit(
        train_start=pd.Timestamp("2025-01-03"),
        train_end=pd.Timestamp("2025-06-27"),
        test_start=pd.Timestamp("2026-01-02"),
        test_end=pd.Timestamp("2026-01-23"),
    )
    proposal = {
        "train_evidence_json": "[]",
        "frozen_rules_json": "all-candidates",
        "frozen_rule_hash": "shared-fold-hash",
    }
    first = evaluation_module.evaluate_frozen_config(
        test,
        compose_selector_config("R1_ATOMIC_IMPROVEMENTS", ["B0 top1"]),
        fold=1,
        split=split,
        proposal=proposal,
        bootstrap_iterations=5,
        seed=1,
    )
    second = evaluation_module.evaluate_frozen_config(
        test,
        compose_selector_config("R2_BALANCED_SOFT", ["B0 fresh continuous"]),
        fold=1,
        split=split,
        proposal=proposal,
        bootstrap_iterations=5,
        seed=1,
    )

    assert first["frozen_rule_hash"] != second["frozen_rule_hash"]
    assert first["frozen_rules_json"] != second["frozen_rules_json"]


def test_global_selection_build_contains_only_b0_and_true_atomic_variants():
    panel = pd.DataFrame(
        [
            _labeled_event("2026-01-02", "AAA", "ceiling_breakout", "NOT_APPLICABLE", 1.0, 1.6, 2.0),
            _labeled_event("2026-01-02", "BBB", "pivot", "PASS", 1.0, 1.6, 1.0),
        ]
    )

    selections = audit_run.build_all_selections(panel)

    variants = set(selections["variant"])
    assert "B0_REPO_EXACT" in variants
    assert "B0_PIT_VERIFIED" in variants
    assert not variants.intersection({"R1_ATOMIC_IMPROVEMENTS", "R2_BALANCED_SOFT", "R3_MINIMAL_TECHNICAL"})
    assert variants == set(selector_configs())


def test_coverage_regimes_are_detected_from_level_shifts_without_a_fixed_500_row_cutoff():
    counts = [20, 21, 19, 20, 22, 18, 55, 60, 65, 58, 62, 59, 115, 120, 118, 122, 119, 121]
    pools = [
        (
            str(pd.Timestamp("2025-01-03") + pd.Timedelta(weeks=index)),
            pd.DataFrame({"code": [f"C{index}_{row}" for row in range(count)]}),
            Path(f"pool_{index}.csv"),
        )
        for index, count in enumerate(counts)
    ]

    regimes = audit_run._coverage_regime_labels(pools)

    assert regimes[pools[0][0]] == "early_stable_low"
    assert regimes[pools[7][0]] == "coverage_transition"
    assert regimes[pools[-1][0]] == "late_stable_high"
    assert max(counts) < 500


def test_preregistered_numeric_bins_put_exact_boundaries_in_the_declared_groups():
    values = pd.Series([-0.1, 0.0, 1.0, 1.3, 1.5, 2.0, 3.0, 5.0, 10.0, pd.NA])

    assert list(audit_run._close_groups(values.iloc[[0, 1, 2, 6, 9]])) == ["<0", "0-1%", "0-1%", "1-3%", "UNKNOWN"]
    assert list(audit_run._fresh_groups(values.iloc[[0, 1, 5, 7, 8, 9]])) == ["<0", "0-2%", "0-2%", "2-5%", "5-10%", "UNKNOWN"]
    assert list(audit_run._volume_groups(values.iloc[[0, 2, 3, 4, 5, 6, 9]])) == [
        "<1.0",
        "1.0-1.3",
        "1.3-1.5",
        "1.5-2.0",
        "1.5-2.0",
        "2.0-3.0",
        "UNKNOWN",
    ]


def test_real_panel_all_atomic_variants_preserve_non_target_rule_traces():
    panel_path = Path("backtest/rd_agent_candidate_rule_audit/output/candidate_event_panel.parquet")
    try:
        panel = pd.read_parquet(panel_path)
    except Exception:
        panel = pd.read_pickle(panel_path)

    invariant = audit_run.b0_atomic_invariant_audit(panel)
    selections = audit_run.build_all_selections(panel)
    ablations = audit_run.b0_atomic_ablation(selections, panel)
    expected_contrast = ablations.set_index("variant")["treatment_contrast"]

    assert invariant["audited_weeks"].eq(42).all()
    assert invariant["non_target_trace_violations"].eq(0).all()
    assert invariant["atomicity_status"].eq("PASS").all()
    assert invariant["selection_contrast_weeks"].ge(0).all()
    assert all(
        row["treatment_contrast"] == expected_contrast.loc[row["variant"]]
        for _, row in invariant.iterrows()
    )
    noops = invariant[invariant["actual_target_trace_changes"].eq(0)]
    assert not noops.empty
    assert noops["treatment_contrast"].eq("NO_TREATMENT_CONTRAST").all()


def test_blocked_oos_returns_train_registry_and_never_labels_it_sealed_oos():
    weeks = pd.date_range("2025-01-03", periods=20, freq="W-FRI")
    rows = []
    for idx, week in enumerate(weeks):
        rows.append(_labeled_event(str(week.date()), f"A{idx}", "ceiling_breakout", "NOT_APPLICABLE", 1.0, 1.6, 2.0))
        rows.append(_labeled_event(str(week.date()), f"B{idx}", "pivot", "PASS", 1.0, 1.6, 1.0))
    panel = pd.DataFrame(rows)

    folds, summary, blocked_picks, registry = audit_run.oos_results(panel, bootstrap_iterations=10)

    assert not registry.empty
    assert registry["frozen_rule_hash"].astype(str).str.len().gt(0).all()
    assert registry["train_evidence_json"].astype(str).str.startswith("[").all()
    assert set(folds["evaluation_type"]) == {"blocked_retrospective_evaluation"}
    assert "sealed" not in " ".join(folds["evaluation_type"]).lower()
    assert set(blocked_picks["variant"]).issuperset({"B0_PIT_VERIFIED_BLOCKED"})


def test_blocked_candidate_portfolio_uses_b0_from_the_same_generated_folds(monkeypatch):
    weeks = pd.date_range("2025-01-03", periods=20, freq="W-FRI")
    panel = pd.DataFrame(
        [
            _labeled_event(str(week.date()), f"A{idx}", "ceiling_breakout", "NOT_APPLICABLE", 1.0, 1.6, 2.0)
            for idx, week in enumerate(weeks)
        ]
    )
    frozen = compose_selector_config("R1_ATOMIC_IMPROVEMENTS", ["B0 top1"])

    def fixed_proposal(panel, split, *, bootstrap_iterations):
        return {
            "train_start": str(split.train_start.date()),
            "train_end": str(split.train_end.date()),
            "train_evidence_max_date": str(split.train_end.date()),
            "train_evidence_json": "[]",
            "frozen_rules_json": json.dumps([{"variant": frozen.name}]),
            "frozen_rule_hash": "frozen-r1",
            "candidate_generation_status": "FROZEN_CANDIDATES",
            "candidate_configs": [frozen],
        }

    monkeypatch.setattr(audit_run, "propose_fold_candidates", fixed_proposal)

    _, summary, picks, _ = audit_run.oos_results(panel, bootstrap_iterations=5)

    matching_baseline = "B0_FOR_R1_ATOMIC_IMPROVEMENTS_BLOCKED"
    assert summary.iloc[0]["portfolio_baseline_variant"] == matching_baseline
    assert {"R1_ATOMIC_IMPROVEMENTS", matching_baseline}.issubset(set(picks["variant"]))


def test_blocked_oos_summary_uses_eight_week_time_blocks_for_overlapping_labels():
    fold_rows = pd.DataFrame(
        [
            {
                "variant": "R1_ATOMIC_IMPROVEMENTS",
                "fold": fold,
                "mean_return_diff_pct": float(fold),
                "paired_weeks": 4,
                "fold_direction": "better",
                "stop_40d_diff_pp": 0.0,
                "profit_24_40d_diff_pp": 0.0,
                "CVaR_20_diff_pct": 0.0,
                "coverage_ratio": 1.0,
                "industry_top_share_diff": 0.0,
                "frozen_rule_hash": f"hash-{fold}",
                "frozen_rule_families": "TopK",
                "source_atomic_variants": "B0 top1",
            }
            for fold in range(1, 5)
        ]
    )

    summary = evaluation_module.summarize_blocked_results(fold_rows, seed=7, bootstrap_iterations=20)

    assert summary.iloc[0]["ci_time_block_weeks"] == 8
    assert summary.iloc[0]["ci_block_folds"] == 2


def test_pareto_requires_all_registered_return_risk_coverage_and_concentration_bars():
    oos = pd.DataFrame(
        [
            {
                "variant": "MEAN_ONLY",
                "mean_oos_diff_pct": 5.0,
                "return_ci_low_pct": -2.0,
                "return_ci_high_pct": 12.0,
                "folds": 5,
                "better_folds": 4,
                "non_worse_folds": 4,
                "stop_40d_diff_pp": 0.0,
                "profit_24_40d_diff_pp": 1.0,
                "CVaR_20_diff_pct": 1.0,
                "coverage_ratio": 1.0,
                "industry_top_share_diff": 0.0,
            },
            {
                "variant": "BALANCED",
                "mean_oos_diff_pct": 2.0,
                "return_ci_low_pct": 0.5,
                "return_ci_high_pct": 3.5,
                "folds": 5,
                "better_folds": 4,
                "non_worse_folds": 5,
                "stop_40d_diff_pp": 0.0,
                "profit_24_40d_diff_pp": 1.0,
                "CVaR_20_diff_pct": 1.0,
                "coverage_ratio": 1.0,
                "industry_top_share_diff": 0.0,
            },
        ]
    )
    metrics = pd.DataFrame(
        [
            {
                "variant": variant,
                "capacity": capacity,
                "cost_bps_per_side": cost,
                "max_drawdown_pct": -10.0 if variant == "B0_PIT_VERIFIED" else -9.0,
                "trades": 20,
                "total_return_pct": 5.0 if variant == "B0_PIT_VERIFIED" else 6.0,
            }
            for variant in ("B0_PIT_VERIFIED", "MEAN_ONLY", "BALANCED")
            for capacity in (3, 6, 10)
            for cost in (0, 10, 25)
        ]
    )

    criteria, decisions = audit_run.evaluate_pareto_candidates(oos, metrics)

    assert set(criteria["criterion"]) == {
        "return_effect",
        "return_confidence_interval",
        "fold_direction_stability",
        "max_drawdown",
        "stop_8_within_40d",
        "profit_24_within_40d",
        "cvar_20",
        "coverage",
        "trade_count",
        "industry_concentration",
        "capacity_cost_robustness",
    }
    assert bool(decisions.set_index("variant").loc["MEAN_ONLY", "pareto_pass"]) is False
    assert bool(decisions.set_index("variant").loc["BALANCED", "pareto_pass"]) is True


def test_pareto_capacity_cost_robustness_fails_when_registered_scenarios_are_missing():
    oos = pd.DataFrame(
        [
            {
                "variant": "INCOMPLETE_SCENARIOS",
                "mean_oos_diff_pct": 2.0,
                "return_ci_low_pct": 0.5,
                "folds": 5,
                "fold_coverage_ratio": 1.0,
                "better_folds": 4,
                "non_worse_folds": 5,
                "stop_40d_diff_pp": 0.0,
                "profit_24_40d_diff_pp": 1.0,
                "CVaR_20_diff_pct": 1.0,
                "coverage_ratio": 1.0,
                "industry_top_share_diff": 0.0,
            }
        ]
    )
    metrics = pd.DataFrame(
        [
            {"variant": "B0_PIT_VERIFIED", "capacity": 3, "cost_bps_per_side": 10, "max_drawdown_pct": -10.0, "trades": 20, "total_return_pct": 5.0},
            {"variant": "INCOMPLETE_SCENARIOS", "capacity": 3, "cost_bps_per_side": 10, "max_drawdown_pct": -9.0, "trades": 20, "total_return_pct": 6.0},
        ]
    )

    criteria, decisions = audit_run.evaluate_pareto_candidates(oos, metrics)

    robustness = criteria[criteria["criterion"].eq("capacity_cost_robustness")].iloc[0]
    assert bool(robustness["passed"]) is False
    assert bool(decisions.iloc[0]["pareto_pass"]) is False


def test_best_balanced_candidate_is_selected_only_from_pareto_passes():
    oos = pd.DataFrame(
        [
            {
                "variant": "HIGH_RETURN_HIGH_RISK",
                "portfolio_baseline_variant": "B0_MATCHED",
                "return_ci_low_pct": 2.0,
                "mean_oos_diff_pct": 9.0,
                "worst_fold_diff_pct": 1.0,
                "CVaR_20_diff_pct": 0.1,
                "stop_40d_diff_pp": 1.0,
                "industry_top_share_diff": 0.05,
            },
            {
                "variant": "BALANCED_RISK",
                "portfolio_baseline_variant": "B0_MATCHED",
                "return_ci_low_pct": 0.5,
                "mean_oos_diff_pct": 2.0,
                "worst_fold_diff_pct": 0.1,
                "CVaR_20_diff_pct": 1.0,
                "stop_40d_diff_pp": 0.0,
                "industry_top_share_diff": 0.0,
            },
        ]
    )
    pareto = pd.DataFrame(
        [
            {"variant": "HIGH_RETURN_HIGH_RISK", "pareto_pass": True},
            {"variant": "BALANCED_RISK", "pareto_pass": True},
        ]
    )
    metrics = pd.DataFrame(
        [
            {"variant": "B0_MATCHED", "capacity": 3, "cost_bps_per_side": 10, "max_drawdown_pct": -10.0, "trades": 20, "total_return_pct": 5.0},
            {"variant": "HIGH_RETURN_HIGH_RISK", "capacity": 3, "cost_bps_per_side": 10, "max_drawdown_pct": -12.0, "trades": 18, "total_return_pct": 9.0},
            {"variant": "BALANCED_RISK", "capacity": 3, "cost_bps_per_side": 10, "max_drawdown_pct": -9.0, "trades": 20, "total_return_pct": 7.0},
        ]
    )

    assert audit_run.best_balanced_candidate(pareto, oos, metrics) == "BALANCED_RISK"


def test_production_change_support_is_computed_from_pareto_and_matching_prospective_rule_families():
    from backtest.rd_agent_candidate_rule_audit.decisions import production_change_supported

    rules = pd.DataFrame(
        [
            {"rule_family": "Fresh Zone", "production_change": True},
            {"rule_family": "Geometry", "production_change": False},
        ]
    )
    matching = pd.DataFrame(
        [{"variant": "R1", "pareto_pass": True, "required_rule_families": "Fresh Zone"}]
    )
    unrelated = pd.DataFrame(
        [{"variant": "R2", "pareto_pass": True, "required_rule_families": "Geometry"}]
    )

    assert production_change_supported(rules, matching) is True
    assert production_change_supported(rules, unrelated) is False


def test_portfolio_outputs_use_one_price_cache_asof_for_every_scenario():
    selections = pd.DataFrame(
        [
            {
                **_event("2026-01-02", "AAA", 100, ibd_entry_date=""),
                "variant": "B0_PIT_VERIFIED",
                "pick_order": 1,
            }
        ]
    )
    prices = {"AAA": _bars([("2026-01-05", 100, 101, 99, 100), ("2026-01-30", 100, 101, 99, 101)])}

    _, curves, metrics = audit_run.portfolio_outputs(selections, prices)

    assert set(metrics["valuation_as_of"]) == {"2026-01-30"}
    assert curves.groupby(["variant", "capacity", "cost_bps_per_side"])["date"].max().eq("2026-01-30").all()


def test_portfolio_outputs_use_one_strategy_start_for_every_variant():
    selections = pd.DataFrame(
        [
            {**_event("2026-01-02", "AAA", 100, ibd_entry_date=""), "variant": "EARLY", "pick_order": 1},
            {**_event("2026-01-09", "BBB", 100, ibd_entry_date=""), "variant": "LATE", "pick_order": 1},
        ]
    )
    prices = {
        "AAA": _bars([("2026-01-05", 100, 101, 99, 100), ("2026-01-30", 100, 101, 99, 101)]),
        "BBB": _bars([("2026-01-12", 100, 101, 99, 100), ("2026-01-30", 100, 101, 99, 101)]),
    }

    _, curves, metrics = audit_run.portfolio_outputs(selections, prices)

    assert curves.groupby(["variant", "capacity", "cost_bps_per_side"])["date"].min().eq("2026-01-05").all()
    assert set(metrics["valuation_start"]) == {"2026-01-05"}


def test_portfolio_metrics_include_closed_trade_payoff_expectancy_and_cash_utilization():
    trades = pd.DataFrame(
        [
            {"return_pct": 24.0, "holding_days": 5, "censored": False},
            {"return_pct": -8.0, "holding_days": 3, "censored": False},
            {"return_pct": 2.0, "holding_days": 10, "censored": True},
        ]
    )
    equity = pd.DataFrame(
        [
            {"date": "2026-01-05", "equity": 1000.0, "market_value": 500.0, "open_positions": 1},
            {"date": "2026-01-06", "equity": 1010.0, "market_value": 252.5, "open_positions": 1},
        ]
    )

    result = portfolio_metrics(equity, trades, initial_capital=1000.0)

    assert result["closed_trades"] == 2
    assert result["win_rate"] == 0.5
    assert result["payoff_ratio"] == 3.0
    assert result["expectancy_pct"] == 8.0
    assert result["cash_utilization"] == 0.375


def test_tiny_audit_orchestration_writes_invariant_freeze_pareto_and_manifest_artifacts(tmp_path):
    pool_root = tmp_path / "pools"
    week_dir = pool_root / "2026-01-02"
    week_dir.mkdir(parents=True)
    pool = pd.DataFrame([_event("2026-01-02", "AAA", 100, eps=30, industry="Software")])
    pool.to_csv(week_dir / "breakout_follow_pool.csv", index=False)
    pd.DataFrame(
        [
            {
                "snapshot_date": "2026-01-02",
                "code": "AAA",
                "eps_yoy_growth": 30.0,
                "effective_date": "2025-12-15",
                "current_period": "2025-09-30",
                "status": "filled",
                "source": "test",
            }
        ]
    ).to_csv(pool_root / "signal_eps_pit.csv", index=False)
    price_rows = []
    for idx in range(50):
        date = pd.Timestamp("2026-01-02") + pd.offsets.BDay(idx)
        price_rows.append((str(date.date()), 100, 101, 99, 100))
    price_cache = tmp_path / "prices.pkl"
    with price_cache.open("wb") as handle:
        pickle.dump({"AAA": _bars(price_rows)}, handle)
    output_dir = tmp_path / "output"

    outputs = audit_run.run_audit(
        pool_root=pool_root,
        price_cache=price_cache,
        output_dir=output_dir,
        bootstrap_iterations=2,
    )

    assert Path(outputs["b0_production_invariant_audit.csv"]).exists()
    assert Path(outputs["b0_atomic_invariant_audit.csv"]).exists()
    assert Path(outputs["fold_rule_freeze_registry.csv"]).exists()
    assert Path(outputs["pareto_criteria.csv"]).exists()
    assert Path(outputs["acceptance_summary.md"]).exists()
    manifest = Path(outputs["experiment_manifest.yaml"]).read_text(encoding="utf-8")
    report = Path(outputs["rd_agent_candidate_rule_report.md"]).read_text(encoding="utf-8")
    assert "uniform_valuation_as_of" in manifest
    assert "pareto_thresholds" in manifest
    assert all(f"{index}. " in report for index in range(1, 19))
    assert "This is not a sealed holdout" in report
    assert not (output_dir / "SKILL.proposed.md").exists()


def test_machine_decision_uses_effect_ci_risk_weeks_and_fold_stability_and_report_is_derived():
    evidence = pd.DataFrame([{"rule_family": "Fresh Zone", "complete_8w": 100, "mature_weeks": 20}])
    contrasts = pd.DataFrame(
        [
            {
                "rule_family": "Fresh Zone",
                "contrast": "Fresh_0_5_vs_other",
                "status": "OK",
                "treated_complete": 50,
                "control_complete": 50,
                "paired_week_routes": 20,
                "mean_return_diff_pct": -2.0,
                "stop_rate_diff": 0.03,
                "profit_24_rate_diff": -0.02,
                "ci_low": -3.0,
                "ci_high": -1.0,
                "blocker": "",
            }
        ]
    )
    folds = pd.DataFrame(
        [
            {"rule_family": "Fresh Zone", "fold_direction": "worse"},
            {"rule_family": "Fresh Zone", "fold_direction": "worse"},
            {"rule_family": "Fresh Zone", "fold_direction": "better"},
        ]
    )

    decisions = audit_run.machine_rule_decisions(evidence, contrasts, folds)
    answers = audit_run.rule_answer_lines(decisions)

    fresh = decisions.set_index("rule_family").loc["Fresh Zone"]
    assert fresh["evidence_status"] == "REJECTED"
    assert bool(fresh["production_change"]) is False
    assert "favorable" not in " ".join(answers).lower()
    assert "REJECTED" in " ".join(answers)


def test_machine_decision_blocks_noop_atomic_rule_and_ci_crossing_zero_is_not_validated():
    evidence = pd.DataFrame(
        [
            {"rule_family": "Fresh Zone", "complete_8w": 100, "mature_weeks": 20},
        ]
    )
    contrasts = pd.DataFrame(
        [
            {
                "rule_family": "Fresh Zone",
                "status": "OK",
                "paired_week_routes": 20,
                "mean_return_diff_pct": 2.0,
                "stop_rate_diff": 0.0,
                "profit_24_rate_diff": 0.0,
                "ci_low": -0.5,
                "ci_high": 4.5,
                "blocker": "",
            }
        ]
    )
    folds = pd.DataFrame(
        [
            {"rule_family": "Fresh Zone", "fold_direction": "better", "evaluation_type": "blocked_retrospective_evaluation"},
            {"rule_family": "Fresh Zone", "fold_direction": "better", "evaluation_type": "blocked_retrospective_evaluation"},
            {"rule_family": "Fresh Zone", "fold_direction": "worse", "evaluation_type": "blocked_retrospective_evaluation"},
            {
                "rule_family": "Entry Valid",
                "fold_direction": "no_contrast",
                "treatment_contrast": "NO_TREATMENT_CONTRAST",
                "evaluation_type": "blocked_retrospective_evaluation",
            },
        ]
    )

    decisions = audit_run.machine_rule_decisions(evidence, contrasts, folds).set_index("rule_family")

    assert decisions.loc["Fresh Zone", "evidence_status"] == "PROMISING_NEEDS_CONFIRMATION"
    assert decisions.loc["Fresh Zone", "evidence_status"] != "VALIDATED"
    assert decisions.loc["Entry Valid", "evidence_status"] == "NOT_IDENTIFIABLE"


def test_machine_decision_rejects_positive_return_when_registered_risk_bar_fails():
    evidence = pd.DataFrame([{"rule_family": "Geometry", "complete_8w": 100, "mature_weeks": 20}])
    contrasts = pd.DataFrame(
        [
            {
                "rule_family": "Geometry",
                "status": "OK",
                "paired_week_routes": 20,
                "mean_return_diff_pct": 2.0,
                "stop_rate_diff": 0.05,
                "profit_24_rate_diff": 0.0,
                "ci_low": 0.5,
                "ci_high": 3.5,
                "blocker": "",
            }
        ]
    )
    folds = pd.DataFrame(
        [
            {"rule_family": "Geometry", "fold_direction": "better", "evaluation_type": "blocked_retrospective_evaluation"},
            {"rule_family": "Geometry", "fold_direction": "better", "evaluation_type": "blocked_retrospective_evaluation"},
            {"rule_family": "Geometry", "fold_direction": "better", "evaluation_type": "blocked_retrospective_evaluation"},
        ]
    )

    decision = audit_run.machine_rule_decisions(evidence, contrasts, folds).set_index("rule_family").loc["Geometry"]

    assert decision["evidence_status"] == "REJECTED"
    assert "risk" in decision["blocker"].lower()


def test_machine_decision_requires_oos_folds_and_observed_risk_before_validation():
    evidence = pd.DataFrame([{"rule_family": "Fresh Zone", "complete_8w": 100, "mature_weeks": 20}])
    complete_contrast = pd.DataFrame(
        [
            {
                "rule_family": "Fresh Zone",
                "contrast": "Fresh_0_5_vs_other",
                "status": "OK",
                "paired_week_routes": 20,
                "mean_return_diff_pct": 2.0,
                "stop_rate_diff": 0.0,
                "profit_24_rate_diff": 0.0,
                "ci_low": 0.5,
                "ci_high": 3.5,
                "blocker": "",
            }
        ]
    )

    no_folds = audit_run.machine_rule_decisions(evidence, complete_contrast, pd.DataFrame()).set_index("rule_family")
    assert no_folds.loc["Fresh Zone", "evidence_status"] == "INSUFFICIENT_EVIDENCE"

    missing_risk = complete_contrast.copy()
    missing_risk[["stop_rate_diff", "profit_24_rate_diff"]] = pd.NA
    folds = pd.DataFrame(
        [
            {
                "variant": "B0 fresh continuous",
                "rule_family": "Fresh Zone",
                "fold_direction": "better",
                "treatment_contrast": "OK",
                "evaluation_type": "blocked_retrospective_evaluation",
            }
            for _ in range(3)
        ]
    )
    missing = audit_run.machine_rule_decisions(evidence, missing_risk, folds).set_index("rule_family")
    assert missing.loc["Fresh Zone", "evidence_status"] == "INSUFFICIENT_EVIDENCE"


def test_machine_decision_uses_one_registered_treatment_instead_of_mixing_pullback_contrasts():
    evidence = pd.DataFrame([{"rule_family": "Pullback", "complete_8w": 100, "mature_weeks": 20}])
    contrasts = pd.DataFrame(
        [
            {
                "rule_family": "Pullback",
                "contrast": "Pullback_dry_PASS_vs_FAIL",
                "status": "OK",
                "paired_week_routes": 20,
                "mean_return_diff_pct": -2.0,
                "stop_rate_diff": 0.0,
                "profit_24_rate_diff": 0.0,
                "ci_low": -3.0,
                "ci_high": -1.0,
                "blocker": "",
            },
            {
                "rule_family": "Pullback",
                "contrast": "Pullback_dry_PASS_vs_UNKNOWN",
                "status": "OK",
                "paired_week_routes": 20,
                "mean_return_diff_pct": 100.0,
                "stop_rate_diff": 0.0,
                "profit_24_rate_diff": 0.0,
                "ci_low": 90.0,
                "ci_high": 110.0,
                "blocker": "",
            },
        ]
    )
    folds = pd.DataFrame(
        [
            {
                "variant": "B0 pullback dry hard",
                "rule_family": "Pullback",
                "fold_direction": "worse",
                "treatment_contrast": "OK",
                "evaluation_type": "blocked_retrospective_evaluation",
            }
            for _ in range(3)
        ]
    )

    decision = audit_run.machine_rule_decisions(evidence, contrasts, folds).set_index("rule_family").loc["Pullback"]

    assert decision["decision_contrast"] == "Pullback_dry_PASS_vs_FAIL"
    assert decision["atomic_variant"] == "B0 pullback dry hard"
    assert decision["mean_effect_pct"] == -2.0
    assert decision["evidence_status"] == "REJECTED"


def test_rule_not_identifiable_when_observed_groups_missing_and_pullback_excludes_base():
    panel = pd.DataFrame(
        [
            _labeled_event("2026-01-02", "BASE", "ceiling_breakout", "NOT_APPLICABLE", 1.0, 1.6, 2.0),
            _labeled_event("2026-01-02", "P1", "pivot", "PASS", 1.0, 1.6, 4.0),
            _labeled_event("2026-01-02", "P2", "pivot", "FAIL", 1.0, 1.6, -2.0),
            _labeled_event("2026-01-09", "P3", "pivot", "UNKNOWN", 1.0, 1.6, 1.0),
        ]
    )

    contrasts = rule_treatment_contrast(panel, bootstrap_iterations=5, min_group_size=1, min_weeks=1)

    close = contrasts[contrasts["contrast"].eq("Close_nonnegative_vs_negative")].iloc[0]
    assert close["status"] == "RULE_NOT_IDENTIFIABLE"
    assert "negative group not observed" in close["blocker"]
    dry = contrasts[contrasts["contrast"].eq("Pullback_dry_PASS_vs_FAIL")].iloc[0]
    assert dry["treated_complete"] == 1
    assert dry["control_complete"] == 1
    assert dry["applicable_events"] == 2


def test_small_but_observed_treatment_is_insufficient_not_no_treatment_contrast():
    panel = pd.DataFrame(
        [
            _labeled_event("2026-01-02", "A", "pivot", "PASS", 1.0, 1.6, 2.0),
            {
                **_labeled_event("2026-01-02", "B", "pivot", "FAIL", 1.0, 1.6, 1.0),
                "ibd_entry_status": "UNCONFIRMED",
            },
        ]
    )

    contrasts = rule_treatment_contrast(panel, bootstrap_iterations=5)

    status = contrasts[contrasts["contrast"].eq("Status_ACTIONABLE_vs_other")].iloc[0]
    assert status["treated_complete"] == 1
    assert status["control_complete"] == 1
    assert status["status"] == "INSUFFICIENT_EVIDENCE"


def test_treatment_stop_and_profit_risk_differences_are_paired_by_week_and_route():
    rows = []
    rows.append({**_labeled_event("2026-01-02", "T1", "pivot", "PASS", 1.0, 1.6, 2.0), "stop_8_within_40d": True})
    rows.extend(
        {
            **_labeled_event("2026-01-02", f"C1{idx}", "pivot", "FAIL", 1.0, 1.6, 1.0),
            "ibd_entry_status": "UNCONFIRMED",
            "stop_8_within_40d": False,
        }
        for idx in range(9)
    )
    rows.extend(
        {
            **_labeled_event("2026-01-09", f"T2{idx}", "pivot", "PASS", 1.0, 1.6, 2.0),
            "stop_8_within_40d": False,
        }
        for idx in range(9)
    )
    rows.append(
        {
            **_labeled_event("2026-01-09", "C2", "pivot", "FAIL", 1.0, 1.6, 1.0),
            "ibd_entry_status": "UNCONFIRMED",
            "stop_8_within_40d": False,
        }
    )

    contrasts = rule_treatment_contrast(
        pd.DataFrame(rows),
        bootstrap_iterations=5,
        min_group_size=1,
        min_weeks=1,
    )

    status = contrasts[contrasts["contrast"].eq("Status_ACTIONABLE_vs_other")].iloc[0]
    assert status["status"] == "OK"
    assert status["stop_rate_diff"] == 0.5


def test_actionable_controlled_increment_requires_overlap_after_fresh_volume_geometry_controls():
    actionable = _labeled_event("2026-01-02", "A", "pivot", "PASS", 1.0, 1.6, 2.0)
    unconfirmed = {
        **_labeled_event("2026-01-02", "U", "pivot", "UNKNOWN", 1.0, 1.6, 1.0),
        "ibd_entry_status": "UNCONFIRMED",
        "current_vs_ibd_candidate_pct": pd.NA,
        "ibd_entry_volume_ratio": pd.NA,
        "geometry": "UNKNOWN",
    }

    contrasts = rule_treatment_contrast(
        pd.DataFrame([actionable, unconfirmed]),
        bootstrap_iterations=5,
        min_group_size=1,
        min_weeks=1,
    )

    controlled = contrasts[contrasts["contrast"].eq("Status_ACTIONABLE_increment_controlled")].iloc[0]
    assert controlled["status"] == "RULE_NOT_IDENTIFIABLE"
    assert "controlled overlap" in controlled["blocker"]


def test_machine_decision_uses_contrast_status_not_hardcoded_mapping():
    evidence = pd.DataFrame(
        [{"rule_family": "Entry Volume", "complete_8w": 100, "weeks": 10}]
    )
    contrasts = pd.DataFrame(
        [
            {
                "rule_family": "Entry Volume",
                "contrast": "Volume_1_5_vs_other",
                "status": "RULE_NOT_IDENTIFIABLE",
                "treated_complete": 100,
                "control_complete": 0,
                "mean_return_diff_pct": pd.NA,
                "stop_rate_diff": pd.NA,
                "profit_24_rate_diff": pd.NA,
                "power_rate_diff": pd.NA,
                "ci_low": pd.NA,
                "ci_high": pd.NA,
                "blocker": "below 1.5 group not observed",
            }
        ]
    )

    decisions = machine_rule_decisions(evidence, contrasts)

    row = decisions[decisions["rule_family"].eq("Entry Volume")].iloc[0]
    assert row["machine_role"] == "UNKNOWN"
    assert row["evidence_status"] == "NOT_IDENTIFIABLE"


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


def _labeled_event(snapshot, code, rule, dry_state, close_vs_trigger, volume, return_8w):
    row = _event(snapshot, code, 100, rule=rule)
    row.update(
        {
            "signal_source": rule,
            "pullback_dry_state": dry_state,
            "ibd_entry_close_vs_trigger_pct": close_vs_trigger,
            "ibd_entry_volume_ratio": volume,
            "geometry": "Strong Finish",
            "forward_8w_return_pct": return_8w,
            "forward_8w_censored": False,
            "stop_8_within_40d": False,
            "profit_24_within_40d": return_8w > 0,
            "pattern_power_trigger": False,
            "mfe_8w_pct": return_8w,
            "mae_8w_pct": min(return_8w, 0),
            "relative_8w_return_pct": return_8w,
            "industry": "Software",
        }
    )
    return row


def _bars(rows):
    frame = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close"])
    frame["Volume"] = 100000
    return frame
