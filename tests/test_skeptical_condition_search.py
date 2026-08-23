import pandas as pd

from backtest.rd_agent_research_bench.skeptical_condition_search import (
    abstain_quality_summary,
    build_skeptical_configs,
    choose_backtrader_finalists,
    render_skeptical_report,
)


def test_skeptical_configs_challenge_actionable_geometry_extended_and_entry_volume():
    configs = build_skeptical_configs()
    by_name = {config.name: config for config in configs}

    config = by_name["sk_sigext_epspass_coreloose_freshallow_bpallow_geomallow_noind_prox_top3"]

    assert config.cfg["allow_non_actionable"] is True
    assert config.cfg["allow_extended_from_buy_point"] is True
    assert config.cfg["allow_clear_geometry_failure"] is True
    assert config.cfg["allow_freshness_missing"] is True
    assert config.cfg["allow_below_candidate_buy_point"] is True
    assert config.cfg["allow_entry_volume_missing"] is True
    assert config.cfg["allow_entry_volume_below_standard"] is True
    assert config.cfg["allow_without_volume_confirm"] is True
    assert config.cfg["require_eps_pass"] is True
    assert config.cfg["max_picks"] == 3
    assert len(configs) == 2592
    forbidden = " ".join(f"{item.profile} {item.cfg}" for item in configs).lower()
    assert "imax" not in forbidden
    assert "2026-" not in forbidden


def test_abstain_quality_summary_shows_missed_baseline_weeks_and_selected_week_delta():
    weekly = pd.DataFrame(
        [
            {"eps_mode": "with_eps", "variant": "baseline", "snapshot_date": "W1", "avg_latest_return_pct": 10.0, "picks": 3, "stop_8pct_count": 0},
            {"eps_mode": "with_eps", "variant": "baseline", "snapshot_date": "W2", "avg_latest_return_pct": -8.0, "picks": 3, "stop_8pct_count": 2},
            {"eps_mode": "with_eps", "variant": "candidate", "snapshot_date": "W1", "avg_latest_return_pct": 12.0, "picks": 2, "stop_8pct_count": 0},
        ]
    )

    summary = abstain_quality_summary(weekly, eps_mode="with_eps", baseline_variant="baseline", variant="candidate")

    assert summary["candidate_weeks"] == 1
    assert summary["missed_baseline_weeks"] == 1
    assert summary["missed_baseline_avg_return_pct"] == -8.0
    assert summary["overlap_avg_return_delta_pct"] == 2.0
    assert summary["missed_baseline_stop_weeks"] == 1


def test_choose_backtrader_finalists_keeps_named_and_non_actionable_candidates():
    summary = pd.DataFrame(
        [
            {"variant": "skill_industry_eps_known", "final_equity_proxy": 10000, "median_week_avg_latest_return_pct": 5, "min_week_avg_latest_return_pct": -8, "pick_stop_rate": 0.2, "non_actionable_pick_rate": 0.0},
            {"variant": "non_actionable_alpha", "final_equity_proxy": 12000, "median_week_avg_latest_return_pct": 6, "min_week_avg_latest_return_pct": -6, "pick_stop_rate": 0.1, "non_actionable_pick_rate": 0.5},
            {"variant": "plain_alpha", "final_equity_proxy": 13000, "median_week_avg_latest_return_pct": 4, "min_week_avg_latest_return_pct": -10, "pick_stop_rate": 0.3, "non_actionable_pick_rate": 0.0},
        ]
    )

    finalists = choose_backtrader_finalists(summary, named_variants=["skill_industry_eps_known"], limit=2)

    assert "skill_industry_eps_known" in finalists
    assert "non_actionable_alpha" in finalists


def test_choose_backtrader_finalists_can_return_all_candidates_when_limit_covers_space():
    summary = pd.DataFrame(
        [
            {"variant": "skill_industry_eps_known", "final_equity_proxy": 10000, "median_week_avg_latest_return_pct": 5, "min_week_avg_latest_return_pct": -8, "pick_stop_rate": 0.2},
            {"variant": "candidate_a", "final_equity_proxy": 12000, "median_week_avg_latest_return_pct": 6, "min_week_avg_latest_return_pct": -6, "pick_stop_rate": 0.1},
            {"variant": "candidate_b", "final_equity_proxy": 13000, "median_week_avg_latest_return_pct": 4, "min_week_avg_latest_return_pct": -10, "pick_stop_rate": 0.3},
        ]
    )

    finalists = choose_backtrader_finalists(summary, named_variants=["skill_industry_eps_known"], limit=10)

    assert set(finalists) == {"skill_industry_eps_known", "candidate_a", "candidate_b"}


def test_report_marks_full_backtrader_and_uses_backtrader_challenge_evidence():
    summary = pd.DataFrame(
        [
            {"variant": "skill_industry_eps_known", "final_equity_proxy": 10000, "non_actionable_pick_rate": 0.0, "extended_pick_rate": 0.0, "clear_geometry_pick_rate": 0.0},
            {"variant": "non_actionable_alpha", "final_equity_proxy": 999999, "non_actionable_pick_rate": 0.4, "extended_pick_rate": 0.2, "clear_geometry_pick_rate": 0.1},
        ]
    )
    decisions = pd.DataFrame(
        [
            {
                "variant": "skill_industry_eps_known",
                "skeptical_status": "baseline",
                "final_value": 11000.0,
                "total_return_pct": 10.0,
                "max_drawdown_pct": -9.0,
                "stop_events": 1,
                "non_actionable_pick_rate": 0.0,
                "extended_pick_rate": 0.0,
                "clear_geometry_pick_rate": 0.0,
            },
            {
                "variant": "non_actionable_alpha",
                "skeptical_status": "portfolio_tradeoff",
                "final_value": 12000.0,
                "total_return_pct": 20.0,
                "max_drawdown_pct": -8.0,
                "stop_events": 1,
                "non_actionable_pick_rate": 0.4,
                "extended_pick_rate": 0.2,
                "clear_geometry_pick_rate": 0.1,
            },
        ]
    )

    report = render_skeptical_report(summary, decisions, pd.DataFrame(), ["skill_industry_eps_known", "non_actionable_alpha"])

    assert "Backtrader is run on all 2 variants" in report
    assert "best Backtrader non-ACTIONABLE scheme `non_actionable_alpha`" in report
    assert "best proxy non-actionable scheme" not in report
