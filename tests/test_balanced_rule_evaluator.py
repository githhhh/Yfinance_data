import pandas as pd

from backtest.rd_agent_research_bench.balanced_rule_evaluator import (
    SelectorConfig,
    _eps_pit_audit,
    _forward_row,
    _walk_forward,
    evaluate_candidate,
    select_by_config,
)


def test_base_signal_marks_pullback_dry_not_applicable_not_fail():
    row = pd.Series(_row("AAA", rule="ceiling", dry=False))

    evaluated = evaluate_candidate(row, row_index=0)

    dry_check = evaluated["checks"]["pullback_dry"]
    assert dry_check["state"] == "NOT_APPLICABLE"
    assert "pullback_not_dry" not in evaluated["risk_flags"]


def test_pullback_dry_modes_compare_hard_minor_and_drop():
    frame = pd.DataFrame(
        [
            _row("DRY", rule="ceiling_pullback", dry=True),
            _row("WET", rule="ceiling_pullback", dry=False),
        ]
    )

    hard = select_by_config(frame, SelectorConfig(name="hard", pullback_dry_mode="hard", top_n=3, industry_cover=False))
    minor = select_by_config(frame, SelectorConfig(name="minor", pullback_dry_mode="minor", top_n=3, industry_cover=False))
    drop = select_by_config(frame, SelectorConfig(name="drop", pullback_dry_mode="drop", top_n=3, industry_cover=False))

    assert list(hard["code"]) == ["DRY"]
    assert set(minor["code"]) == {"DRY", "WET"}
    assert set(drop["code"]) == {"DRY", "WET"}
    wet_minor = minor[minor["code"].eq("WET")].iloc[0]
    wet_drop = drop[drop["code"].eq("WET")].iloc[0]
    assert "pullback_not_dry" in wet_minor["risk_flags"]
    assert "pullback_not_dry" not in wet_drop["risk_flags"]


def test_forward_8w_is_unknown_when_less_than_40_trading_days_are_available():
    row = {"variant": "sample", "snapshot_date": "2026-01-02", "code": "AAA"}
    prices = {"AAA": _bars_for_forward([("2026-01-05", 100, 121, 99, 120), ("2026-01-06", 120, 122, 119, 121)])}

    result = _forward_row(row, prices)

    assert result["forward_1w_return_pct"] is None
    assert result["forward_8w_return_pct"] is None
    assert result["forward_8w_censored"] is True
    assert result["mfe_pct"] is None
    assert result["mae_pct"] is None
    assert result["first_touch"] == "CENSORED"


def test_forward_path_metrics_are_limited_to_8w_window():
    rows = []
    for idx in range(45):
        date = pd.Timestamp("2026-01-05") + pd.offsets.BDay(idx)
        if idx < 40:
            rows.append((date.date().isoformat(), 100, 105, 95, 100))
        else:
            rows.append((date.date().isoformat(), 100, 150, 50, 100))
    row = {"variant": "sample", "snapshot_date": "2026-01-02", "code": "AAA"}
    prices = {"AAA": _bars_for_forward(rows)}

    result = _forward_row(row, prices)

    assert result["mfe_pct"] == 5.0
    assert result["mae_pct"] == -5.0
    assert result["first_touch"] == "NONE"


def test_eps_pit_audit_flags_yahoo_period_date_as_unverified_availability(tmp_path):
    pit = tmp_path / "signal_eps_pit.csv"
    pd.DataFrame(
        [
            {
                "snapshot_date": "2026-07-10",
                "code": "EA",
                "eps_yoy_growth": 97.4,
                "source": "Yahoo",
                "effective_date": "2026-06-30",
                "current_period": "2026-06-30",
                "status": "filled",
            }
        ]
    ).to_csv(pit, index=False)

    audit = _eps_pit_audit(pit)

    assert audit["unverified_availability"] == 1
    assert audit["usable_rows_for_formal_eps_eval"] == 0


def test_walk_forward_freezes_one_variant_and_does_not_rank_all_holdout_variants():
    weeks = [pd.Timestamp("2026-01-02") + pd.offsets.Week(weekday=4) * i for i in range(14)]
    rows = []
    for week in weeks:
        for variant, ret in [("b0_repository_skill", 1.0), ("v_good", 5.0), ("v_bad", -5.0)]:
            rows.append(
                {
                    "snapshot_date": week.date().isoformat(),
                    "variant": variant,
                    "code": variant,
                    "forward_8w_return_pct": ret,
                    "forward_8w_censored": False,
                    "first_touch": "NONE",
                }
            )
    sealed = {weeks[-2].date().isoformat(), weeks[-1].date().isoformat()}

    wf = _walk_forward(pd.DataFrame(), pd.DataFrame(rows), sealed, min_train_weeks=4, embargo_weeks=2, test_window_weeks=2)

    holdout = wf[wf["segment"].eq("sealed_holdout")]
    assert set(holdout["selected_variant"]) == {"v_good"}
    assert "v_bad" not in set(holdout["selected_variant"])
    assert holdout.iloc[0]["paired_weeks"] == 2


def test_walk_forward_marks_holdout_unusable_when_8w_labels_are_censored():
    weeks = [pd.Timestamp("2026-01-02") + pd.offsets.Week(weekday=4) * i for i in range(12)]
    rows = []
    for week in weeks:
        for variant, ret, censored in [
            ("b0_repository_skill", 1.0, week in weeks[-2:]),
            ("v_good", 5.0, week in weeks[-2:]),
        ]:
            rows.append(
                {
                    "snapshot_date": week.date().isoformat(),
                    "variant": variant,
                    "code": variant,
                    "forward_8w_return_pct": None if censored else ret,
                    "forward_8w_censored": bool(censored),
                    "first_touch": "CENSORED" if censored else "NONE",
                }
            )
    sealed = {weeks[-2].date().isoformat(), weeks[-1].date().isoformat()}

    wf = _walk_forward(pd.DataFrame(), pd.DataFrame(rows), sealed, min_train_weeks=4, embargo_weeks=2, test_window_weeks=2)

    holdout = wf[wf["segment"].eq("sealed_holdout")].iloc[0]
    assert holdout["status"] == "no_complete_8w_labels"
    assert pd.isna(holdout["test_mean_delta_vs_b0"])


def _row(code, *, rule="ceiling", dry=True):
    return {
        "snapshot_date": "2026-01-02",
        "code": code,
        "signal": True,
        "ibd_entry_status": "ACTIONABLE",
        "ibd_entry_valid": 1,
        "ibd_candidate_rule": rule,
        "ibd_candidate_price": 100.0,
        "latest_close": 101.0,
        "current_vs_ibd_candidate_pct": 1.0,
        "ibd_entry_volume_ratio": 2.0,
        "ibd_entry_close_vs_trigger_pct": 0.01,
        "ibd_entry_close_position": 0.9,
        "ibd_entry_breakout_range_ratio": 0.7,
        "volume_ratio": 1.5,
        "dist_to_52w_high_pct": -2.0,
        "pullback_v_is_dry": dry,
        "eps_yoy_growth": 30.0,
        "industry": "Software",
        "signal_source": "ceiling_breakout",
    }


def _bars_for_forward(rows):
    frame = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close"])
    frame["Volume"] = 100000
    return frame
