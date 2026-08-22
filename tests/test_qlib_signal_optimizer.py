import pandas as pd

from backtest.ibd_weekly_signal_oracle_eval.price_cache import latest_daily_price_cache, resolve_price_cache
from backtest.ibd_weekly_signal_oracle_eval.qlib_optimizer import (
    PortfolioRule,
    build_qlib_panel,
    candidate_rules,
    effective_eps,
    evaluate_rule_on_weeks,
    filter_valid_return_panel,
    select_portfolio,
    walk_forward_optimize,
)


def _sample_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "snapshot_date": "2026-01-02",
                "code": "AAA",
                "industry": "Software",
                "entry_status": "ACTIONABLE",
                "eps_state": "pass_25",
                "current_vs_ibd_candidate_pct": 1.0,
                "ibd_entry_volume_ratio": 2.0,
                "volume_ratio": 1.6,
                "dist_to_52w_high_pct": -1.0,
                "pullback_v_is_dry": True,
                "latest_return_pct": 20.0,
                "max_gain_pct": 25.0,
                "hit_stop_8pct": False,
                "loss_rank": 3,
                "latest_rank": 1,
                "gain_rank": 1,
            },
            {
                "snapshot_date": "2026-01-02",
                "code": "BBB",
                "industry": "Software",
                "entry_status": "ACTIONABLE",
                "eps_state": "missing",
                "current_vs_ibd_candidate_pct": 1.5,
                "ibd_entry_volume_ratio": 1.7,
                "volume_ratio": 1.2,
                "dist_to_52w_high_pct": -2.0,
                "pullback_v_is_dry": False,
                "latest_return_pct": -10.0,
                "max_gain_pct": 1.0,
                "hit_stop_8pct": True,
                "loss_rank": 1,
                "latest_rank": 3,
                "gain_rank": 3,
            },
            {
                "snapshot_date": "2026-01-09",
                "code": "CCC",
                "industry": "Banks",
                "entry_status": "ACTIONABLE",
                "eps_state": "pass_25",
                "current_vs_ibd_candidate_pct": 0.8,
                "ibd_entry_volume_ratio": 2.2,
                "volume_ratio": 1.5,
                "dist_to_52w_high_pct": -0.5,
                "pullback_v_is_dry": True,
                "latest_return_pct": 12.0,
                "max_gain_pct": 18.0,
                "hit_stop_8pct": False,
                "loss_rank": 3,
                "latest_rank": 1,
                "gain_rank": 1,
            },
            {
                "snapshot_date": "2026-01-16",
                "code": "DDD",
                "industry": "Retail",
                "entry_status": "ACTIONABLE",
                "eps_state": "known_below_25",
                "current_vs_ibd_candidate_pct": 0.5,
                "ibd_entry_volume_ratio": 2.5,
                "volume_ratio": 1.4,
                "dist_to_52w_high_pct": -3.0,
                "pullback_v_is_dry": True,
                "latest_return_pct": 8.0,
                "max_gain_pct": 10.0,
                "hit_stop_8pct": False,
                "loss_rank": 2,
                "latest_rank": 1,
                "gain_rank": 1,
            },
        ]
    )


def test_build_qlib_panel_uses_datetime_instrument_index_and_numeric_features():
    panel = build_qlib_panel(_sample_rows())

    assert panel.index.names == ["datetime", "instrument"]
    assert ("2026-01-02", "AAA") in [(str(idx[0].date()), idx[1]) for idx in panel.index]
    assert "$eps_pass_25" in panel.columns
    assert "$entry_volume_ratio" in panel.columns
    assert "label_return" in panel.columns
    assert panel.loc[(pd.Timestamp("2026-01-02"), "AAA"), "$eps_pass_25"] == 1.0
    assert panel.loc[(pd.Timestamp("2026-01-02"), "BBB"), "$eps_known"] == 0.0


def test_select_portfolio_respects_industry_cap_and_actionable_filter():
    panel = build_qlib_panel(_sample_rows())
    rule = PortfolioRule(
        name="eps_quality",
        weights={"$eps_pass_25": 2.0, "$entry_volume_ratio": 1.0},
        require_actionable=True,
        industry_cap=True,
    )

    picks = select_portfolio(panel.xs(pd.Timestamp("2026-01-02"), level="datetime"), rule, top_k=2)

    assert picks["instrument"].tolist() == ["AAA"]
    assert picks["industry"].tolist() == ["Software"]


def test_scoring_rules_do_not_use_future_labels():
    panel = build_qlib_panel(_sample_rows())
    assert "label_stop_8pct" in panel.columns
    assert all(
        not column.startswith("label_")
        for rule in candidate_rules()
        for column in [*rule.weights.keys(), *rule.penalties.keys()]
    )

    leaking_rule = PortfolioRule(name="leak", weights={"label_return": 1.0})

    try:
        select_portfolio(panel.xs(pd.Timestamp("2026-01-02"), level="datetime"), leaking_rule, top_k=1)
    except ValueError as exc:
        assert "future label" in str(exc)
    else:
        raise AssertionError("future labels must not be accepted in portfolio scoring")


def test_no_eps_mode_ignores_raw_eps_values():
    assert effective_eps("2026-01-02", "AAA", 42.0, eps_enabled=False) is None
    assert effective_eps("2026-01-02", "AAA", 42.0, eps_enabled=True) == 42.0


def test_filter_valid_return_panel_tracks_missing_return_coverage():
    rows = _sample_rows()
    rows.loc[0, "latest_return_pct"] = None
    panel = build_qlib_panel(rows)

    filtered, coverage = filter_valid_return_panel(panel)

    assert coverage["signal_rows"] == 4
    assert coverage["valid_return_rows"] == 3
    assert coverage["missing_return_rows"] == 1
    assert "AAA" not in filtered.index.get_level_values("instrument")


def test_rule_evaluation_empty_pick_metrics_include_all_score_fields():
    rows = _sample_rows()
    rows["eps_state"] = "missing"
    panel = build_qlib_panel(rows)
    rule = PortfolioRule(name="requires_eps_pass", weights={"$eps_pass_25": 1.0}, require_eps_pass=True)

    metrics = evaluate_rule_on_weeks(panel, rule, [pd.Timestamp("2026-01-02")], top_k=1)

    assert metrics["weeks"] == 0
    assert metrics["top5_return_rate"] == 0.0
    assert metrics["top5_gain_rate"] == 0.0


def test_walk_forward_chooses_rules_from_prior_weeks_only():
    panel = build_qlib_panel(_sample_rows())
    eps_rule = PortfolioRule(
        name="eps_rule",
        weights={"$eps_pass_25": 3.0},
        require_actionable=True,
        industry_cap=True,
    )
    below_eps_rule = PortfolioRule(
        name="below_eps_rule",
        weights={"$eps_below_25": 3.0},
        require_actionable=True,
        industry_cap=True,
    )

    picks, summary, choices, rule_scores = walk_forward_optimize(
        panel,
        [eps_rule, below_eps_rule],
        min_train_weeks=1,
        top_k=1,
    )

    assert picks["snapshot_date"].tolist() == ["2026-01-09", "2026-01-16"]
    assert picks.loc[picks["snapshot_date"].eq("2026-01-09"), "selected_rule"].item() == "eps_rule"
    assert choices["snapshot_date"].tolist() == ["2026-01-09", "2026-01-16"]
    assert set(rule_scores["candidate_rule"]) == {"eps_rule", "below_eps_rule"}
    assert summary.loc[summary["strategy"].eq("walk_forward_best_rule"), "weeks"].item() == 2


def test_latest_daily_price_cache_uses_filename_date(tmp_path):
    older = tmp_path / "stock_data_150826_1d.pkl"
    latest = tmp_path / "stock_data_220826_1d.pkl"
    weekly = tmp_path / "stock_data_220826_1wk.pkl"
    older.write_bytes(b"old")
    latest.write_bytes(b"new")
    weekly.write_bytes(b"weekly")

    assert latest_daily_price_cache(tmp_path) == latest
    assert resolve_price_cache(older) == older
