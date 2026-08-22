import pandas as pd

from backtest.rd_agent_research_bench.signal_core_grid_search import (
    build_eps_pass_fallback_configs,
    build_grid_configs,
    select_best_candidates,
)


def test_grid_configs_are_finite_and_do_not_embed_ticker_or_date_rules():
    configs = build_grid_configs(eps_gates=("none", "known", "pass25"))

    assert len(configs) == 192
    assert len({config.name for config in configs}) == len(configs)
    assert "grid_sig_epspass_ind_def_clean_relaxed" in {config.name for config in configs}

    forbidden = " ".join(f"{config.profile} {config.cfg}" for config in configs).lower()
    assert "imax" not in forbidden
    assert "blfs" not in forbidden
    assert "2026-" not in forbidden


def test_grid_configs_keep_signal_wide_core_quality_candidate_available():
    configs = {config.name: config for config in build_grid_configs(eps_gates=("pass25",))}

    config = configs["grid_sig_epspass_ind_def_keep_strict"]

    assert config.cfg["allow_non_actionable"] is True
    assert config.cfg["require_core_quality"] is True
    assert config.cfg["require_eps_pass"] is True
    assert config.cfg["industry_cover"] is True
    assert "exclude_pullback_not_dry" not in config.cfg
    assert "exclude_geometry_caution" not in config.cfg


def test_eps_pass_fallback_configs_only_relax_eps_during_fill():
    configs = build_eps_pass_fallback_configs()

    assert len(configs) == 32
    config = {item.name: item for item in configs}["grid_sig_epspass2known_noind_prox_nogeom_relaxed"]
    assert config.cfg["require_eps_pass"] is True
    assert config.cfg["fill_relaxed"] is True
    assert config.cfg["fill_eps_fallback"] == "known"
    assert config.cfg["allow_non_actionable"] is True
    assert config.profile["eps_gate"] == "pass25_then_known_fill"


def test_select_best_candidates_marks_direct_replacement_and_pareto_rows():
    summary = pd.DataFrame(
        [
            {
                "eps_mode": "with_eps",
                "variant": "baseline",
                "final_value": 15000.0,
                "total_return_pct": 50.0,
                "max_drawdown_pct": -16.0,
                "stop_events": 5,
                "rebalance_events": 40,
                "input_picks": 100,
                "rule_count": 3,
            },
            {
                "eps_mode": "with_eps",
                "variant": "better",
                "final_value": 16000.0,
                "total_return_pct": 60.0,
                "max_drawdown_pct": -12.0,
                "stop_events": 4,
                "rebalance_events": 39,
                "input_picks": 92,
                "rule_count": 4,
            },
            {
                "eps_mode": "with_eps",
                "variant": "thin",
                "final_value": 17000.0,
                "total_return_pct": 70.0,
                "max_drawdown_pct": -10.0,
                "stop_events": 2,
                "rebalance_events": 20,
                "input_picks": 40,
                "rule_count": 4,
            },
        ]
    )

    selected = select_best_candidates(summary, eps_mode="with_eps", baseline_variant="baseline")
    statuses = dict(zip(selected["variant"], selected["grid_status"]))

    assert statuses["better"] == "direct_replacement_candidate"
    assert statuses["thin"] == "high_return_low_coverage"
    assert bool(selected.loc[selected["variant"].eq("better"), "pareto_frontier"].item())


def test_select_best_candidates_blocks_direct_replacement_when_weekly_floor_degrades():
    summary = pd.DataFrame(
        [
            {
                "eps_mode": "with_eps",
                "variant": "baseline",
                "final_value": 15000.0,
                "total_return_pct": 50.0,
                "max_drawdown_pct": -16.0,
                "stop_events": 5,
                "rebalance_events": 40,
                "input_picks": 100,
                "rule_count": 3,
                "median_week_avg_latest_return_pct": 16.0,
                "min_week_avg_latest_return_pct": -4.0,
                "pick_bottom5_precision_bad": 0.15,
                "pick_stop_rate": 0.15,
            },
            {
                "eps_mode": "with_eps",
                "variant": "higher_return_worse_floor",
                "final_value": 17000.0,
                "total_return_pct": 70.0,
                "max_drawdown_pct": -10.0,
                "stop_events": 2,
                "rebalance_events": 40,
                "input_picks": 100,
                "rule_count": 4,
                "median_week_avg_latest_return_pct": 14.0,
                "min_week_avg_latest_return_pct": -8.0,
                "pick_bottom5_precision_bad": 0.14,
                "pick_stop_rate": 0.14,
            },
        ]
    )

    selected = select_best_candidates(summary, eps_mode="with_eps", baseline_variant="baseline")
    status = selected.loc[selected["variant"].eq("higher_return_worse_floor"), "grid_status"].item()

    assert status == "portfolio_only_quality_tradeoff"
