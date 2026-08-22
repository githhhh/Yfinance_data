import pandas as pd

from backtest.rd_agent_research_bench.metrics import (
    coverage_rates,
    rank_weighted_week_return,
    robust_weekly_summary,
)
from backtest.rd_agent_research_bench.research import (
    absorption_candidate_matrix,
    intuitive_variant_summary,
    imax_rank_audit,
    pair_outcome_audit,
    render_markdown_report,
    rule_minimality_summary,
    rule_status_coverage,
    semiconductor_capture_audit,
    summarize_variant_quality,
    stop_loss_capital_backtest,
)
from backtest.rd_agent_research_bench.hypotheses import hypothesis_space


def test_rank_weighted_week_return_rewards_higher_imax_rank():
    base = pd.DataFrame(
        [
            {"pick_order": 1, "code": "BLFS", "latest_return_pct": 18.0},
            {"pick_order": 2, "code": "IMAX", "latest_return_pct": 22.0},
            {"pick_order": 3, "code": "NWFL", "latest_return_pct": 8.0},
        ]
    )
    imax_first = pd.DataFrame(
        [
            {"pick_order": 1, "code": "IMAX", "latest_return_pct": 22.0},
            {"pick_order": 2, "code": "BLFS", "latest_return_pct": 18.0},
            {"pick_order": 3, "code": "NWFL", "latest_return_pct": 8.0},
        ]
    )

    assert rank_weighted_week_return(imax_first) > rank_weighted_week_return(base)


def test_robust_weekly_summary_uses_median_tail_and_stop_metrics():
    weekly = pd.DataFrame(
        [
            {"snapshot_date": "2026-01-02", "avg_latest_return_pct": 10.0, "worst_latest_return_pct": 1.0, "stop_8pct_count": 0},
            {"snapshot_date": "2026-01-09", "avg_latest_return_pct": 30.0, "worst_latest_return_pct": 5.0, "stop_8pct_count": 1},
            {"snapshot_date": "2026-01-16", "avg_latest_return_pct": -12.0, "worst_latest_return_pct": -20.0, "stop_8pct_count": 2},
        ]
    )
    picks = pd.DataFrame(
        [
            {"hit_latest_top5": True, "hit_loss_bottom5": False, "hit_stop_8pct": False},
            {"hit_latest_top5": False, "hit_loss_bottom5": True, "hit_stop_8pct": True},
            {"hit_latest_top5": False, "hit_loss_bottom5": False, "hit_stop_8pct": True},
        ]
    )

    summary = robust_weekly_summary(weekly, picks)

    assert summary["median_week_avg_latest_return_pct"] == 10.0
    assert summary["min_week_avg_latest_return_pct"] == -12.0
    assert summary["median_worst_pick_return_pct"] == 1.0
    assert summary["pick_top5_precision"] == 1 / 3
    assert summary["pick_stop_rate"] == 2 / 3


def test_coverage_rates_distinguish_coverage_from_precision():
    universe = pd.DataFrame(
        [
            {"code": "A", "latest_rank": 1, "gain_rank": 2, "loss_rank": 10, "hit_stop_8pct": False, "valid_path": True},
            {"code": "B", "latest_rank": 5, "gain_rank": 6, "loss_rank": 1, "hit_stop_8pct": True, "valid_path": True},
            {"code": "C", "latest_rank": 8, "gain_rank": 1, "loss_rank": 2, "hit_stop_8pct": False, "valid_path": True},
        ]
    )
    picks = pd.DataFrame(
        [
            {"code": "A", "hit_latest_top5": True, "hit_gain_top5": True, "hit_loss_bottom5": False, "hit_stop_8pct": False},
            {"code": "B", "hit_latest_top5": True, "hit_gain_top5": False, "hit_loss_bottom5": True, "hit_stop_8pct": True},
        ]
    )

    rates = coverage_rates(universe, picks)

    assert rates["top5_coverage"] == 1.0
    assert rates["gain5_coverage"] == 0.5
    assert rates["bottom5_exposure_vs_slots"] == 0.5
    assert rates["pick_top5_precision"] == 1.0
    assert rates["pick_bottom5_precision_bad"] == 0.5


def test_hypothesis_space_keeps_rd_agent_outputs_out_of_official_skill():
    hypotheses = hypothesis_space()

    names = {item["name"] for item in hypotheses}
    assert "fresh_demand_proximity_first" in names
    assert "pullback_vcp_lane_interleave" in names
    assert all("label_" not in feature for item in hypotheses for feature in item["features"])
    assert all(item["official_skill_absorption"] in {"candidate_only", "audit_only"} for item in hypotheses)


def test_summarize_variant_quality_combines_robust_metrics_and_coverage():
    universe = pd.DataFrame(
        [
            {"snapshot_date": "2026-07-24", "code": "IMAX", "latest_rank": 1, "gain_rank": 2, "loss_rank": 4, "hit_stop_8pct": False, "valid_path": True},
            {"snapshot_date": "2026-07-24", "code": "BLFS", "latest_rank": 3, "gain_rank": 6, "loss_rank": 3, "hit_stop_8pct": True, "valid_path": True},
        ]
    )
    weekly = pd.DataFrame(
        [
            {
                "eps_mode": "with_eps",
                "variant": "sample",
                "snapshot_date": "2026-07-24",
                "picks": 2,
                "avg_latest_return_pct": 20.0,
                "worst_latest_return_pct": 18.0,
                "stop_8pct_count": 1,
            }
        ]
    )
    picks = pd.DataFrame(
        [
            {"eps_mode": "with_eps", "variant": "sample", "snapshot_date": "2026-07-24", "pick_order": 1, "code": "IMAX", "latest_return_pct": 22.0, "hit_latest_top3": True, "hit_latest_top5": True, "hit_gain_top5": True, "hit_loss_bottom3": False, "hit_loss_bottom5": True, "hit_stop_8pct": False},
            {"eps_mode": "with_eps", "variant": "sample", "snapshot_date": "2026-07-24", "pick_order": 2, "code": "BLFS", "latest_return_pct": 18.0, "hit_latest_top3": True, "hit_latest_top5": True, "hit_gain_top5": False, "hit_loss_bottom3": True, "hit_loss_bottom5": True, "hit_stop_8pct": True},
        ]
    )

    row = summarize_variant_quality("with_eps", "sample", universe, weekly, picks)

    assert row["eps_mode"] == "with_eps"
    assert row["variant"] == "sample"
    assert row["rank_weighted_week_return_median"] == 20.4
    assert row["top5_coverage"] == 1.0
    assert row["pick_stop_rate"] == 0.5


def test_imax_rank_audit_reports_selected_order_and_oracle_rank():
    universe = pd.DataFrame(
        [
            {"snapshot_date": "2026-07-24", "code": "IMAX", "latest_rank": 6, "gain_rank": 8, "loss_rank": 95, "latest_return_pct": 22.5, "max_gain_pct": 25.7, "valid_path": True}
        ]
    )
    picks = pd.DataFrame(
        [
            {"eps_mode": "with_eps", "snapshot_date": "2026-07-24", "variant": "skill", "pick_order": 2, "code": "IMAX"},
            {"eps_mode": "with_eps", "snapshot_date": "2026-07-24", "variant": "other", "pick_order": 1, "code": "BLFS"},
        ]
    )

    audit = imax_rank_audit(universe, picks)

    assert audit.loc[audit["variant"].eq("skill"), "selected"].item() is True
    assert audit.loc[audit["variant"].eq("skill"), "pick_order"].item() == 2
    assert audit.loc[audit["variant"].eq("skill"), "latest_rank"].item() == 6
    assert audit.loc[audit["variant"].eq("other"), "selected"].item() is False


def test_pair_outcome_audit_shows_imax_returned_more_than_blfs_despite_lower_baseline_rank():
    universe = pd.DataFrame(
        [
            {"snapshot_date": "2026-07-24", "code": "BLFS", "latest_return_pct": 18.4, "max_gain_pct": 19.3, "latest_rank": 8, "gain_rank": 14},
            {"snapshot_date": "2026-07-24", "code": "IMAX", "latest_return_pct": 22.5, "max_gain_pct": 25.7, "latest_rank": 6, "gain_rank": 8},
        ]
    )
    picks = pd.DataFrame(
        [
            {"snapshot_date": "2026-07-24", "variant": "skill_industry_eps_known", "pick_order": 1, "code": "BLFS", "reason_codes": "near_buy_point;volume_confirms_breakout", "risk_codes": ""},
            {"snapshot_date": "2026-07-24", "variant": "skill_industry_eps_known", "pick_order": 2, "code": "IMAX", "reason_codes": "geometry_caution_not_failure;near_buy_point", "risk_codes": ""},
            {"snapshot_date": "2026-07-24", "variant": "research_fresh_demand_proximity_first", "pick_order": 1, "code": "IMAX", "reason_codes": "geometry_caution_not_failure;near_buy_point", "risk_codes": ""},
        ]
    )

    audit = pair_outcome_audit(universe, picks, snapshot_date="2026-07-24", codes=("BLFS", "IMAX"))

    imax = audit[audit["code"].eq("IMAX")].iloc[0]
    blfs = audit[audit["code"].eq("BLFS")].iloc[0]
    assert imax["latest_return_pct"] > blfs["latest_return_pct"]
    assert imax["max_gain_pct"] > blfs["max_gain_pct"]
    assert imax["latest_return_delta_vs_peer"] == 4.1
    assert "research_fresh_demand_proximity_first:1" in imax["variant_orders"]
    assert "skill_industry_eps_known:1" in blfs["variant_orders"]


def test_intuitive_variant_summary_explains_positive_negative_week_distribution():
    weekly = pd.DataFrame(
        [
            {"eps_mode": "with_eps", "variant": "skill_industry_eps_known", "snapshot_date": "2026-01-02", "picks": 3, "avg_latest_return_pct": 10.0, "worst_latest_return_pct": 1.0},
            {"eps_mode": "with_eps", "variant": "skill_industry_eps_known", "snapshot_date": "2026-01-09", "picks": 3, "avg_latest_return_pct": -4.0, "worst_latest_return_pct": -8.0},
            {"eps_mode": "with_eps", "variant": "skill_industry_eps_known", "snapshot_date": "2026-01-16", "picks": 2, "avg_latest_return_pct": 0.0, "worst_latest_return_pct": 0.0},
        ]
    )

    summary = intuitive_variant_summary(weekly, eps_mode="with_eps", variant="skill_industry_eps_known")

    assert summary["positive_weeks"] == 1
    assert summary["negative_weeks"] == 1
    assert summary["flat_weeks"] == 1
    assert summary["weeks_with_less_than_3_picks"] == 1
    assert summary["positive_week_rate"] == 1 / 3


def test_absorption_candidate_matrix_separates_candidate_audit_and_reject():
    quality = pd.DataFrame(
        [
            {"eps_mode": "with_eps", "variant": "skill_industry_eps_known", "weeks": 40, "picks": 100, "median_week_avg_latest_return_pct": 16.0, "min_week_avg_latest_return_pct": -4.0, "pick_bottom5_precision_bad": 0.15, "pick_stop_rate": 0.15},
            {"eps_mode": "with_eps", "variant": "guarded", "weeks": 36, "picks": 82, "median_week_avg_latest_return_pct": 16.5, "min_week_avg_latest_return_pct": 1.0, "pick_bottom5_precision_bad": 0.10, "pick_stop_rate": 0.09},
            {"eps_mode": "with_eps", "variant": "shadow", "weeks": 42, "picks": 126, "median_week_avg_latest_return_pct": 22.0, "min_week_avg_latest_return_pct": -12.0, "pick_bottom5_precision_bad": 0.23, "pick_stop_rate": 0.21},
            {"eps_mode": "with_eps", "variant": "thin", "weeks": 10, "picks": 12, "median_week_avg_latest_return_pct": 19.0, "min_week_avg_latest_return_pct": 2.0, "pick_bottom5_precision_bad": 0.01, "pick_stop_rate": 0.01},
        ]
    )

    matrix = absorption_candidate_matrix(quality, eps_mode="with_eps", baseline_variant="skill_industry_eps_known")
    statuses = dict(zip(matrix["variant"], matrix["absorption_status"]))

    assert statuses["guarded"] == "candidate_absorbable"
    assert statuses["shadow"] == "audit_only"
    assert statuses["thin"] == "reject_low_coverage"
    assert "ticker" not in " ".join(matrix["decision_reason"].str.lower())


def test_rule_minimality_summary_compares_core_rules_and_imax_selection():
    quality = pd.DataFrame(
        [
            {"eps_mode": "with_eps", "variant": "skill_industry_eps_known", "picks": 100, "median_week_avg_latest_return_pct": 16.0, "min_week_avg_latest_return_pct": -4.0, "pick_bottom5_precision_bad": 0.15, "pick_stop_rate": 0.15},
            {"eps_mode": "with_eps", "variant": "signal_core_quality_eps_pass", "picks": 88, "median_week_avg_latest_return_pct": 15.5, "min_week_avg_latest_return_pct": -4.0, "pick_bottom5_precision_bad": 0.14, "pick_stop_rate": 0.13},
            {"eps_mode": "with_eps", "variant": "clean_eps_pass_no_dry_no_geom_caution", "picks": 88, "median_week_avg_latest_return_pct": 15.2, "min_week_avg_latest_return_pct": -4.0, "pick_bottom5_precision_bad": 0.14, "pick_stop_rate": 0.14},
        ]
    )
    backtrader = pd.DataFrame(
        [
            {"eps_mode": "with_eps", "variant": "skill_industry_eps_known", "final_value": 15000.0, "total_return_pct": 50.0, "max_drawdown_pct": -16.0, "stop_events": 5},
            {"eps_mode": "with_eps", "variant": "signal_core_quality_eps_pass", "final_value": 16000.0, "total_return_pct": 60.0, "max_drawdown_pct": -11.0, "stop_events": 3},
            {"eps_mode": "with_eps", "variant": "clean_eps_pass_no_dry_no_geom_caution", "final_value": 18000.0, "total_return_pct": 80.0, "max_drawdown_pct": -6.0, "stop_events": 2},
        ]
    )
    imax = pd.DataFrame(
        [
            {"eps_mode": "with_eps", "variant": "signal_core_quality_eps_pass", "selected": True, "pick_order": 2},
            {"eps_mode": "with_eps", "variant": "clean_eps_pass_no_dry_no_geom_caution", "selected": False, "pick_order": pd.NA},
        ]
    )

    summary = rule_minimality_summary(quality, backtrader, imax)
    by_variant = summary.set_index("variant")

    assert by_variant.loc["signal_core_quality_eps_pass", "rule_count"] < by_variant.loc["clean_eps_pass_no_dry_no_geom_caution", "rule_count"]
    assert by_variant.loc["signal_core_quality_eps_pass", "final_value"] > by_variant.loc["skill_industry_eps_known", "final_value"]
    assert bool(by_variant.loc["signal_core_quality_eps_pass", "imax_selected"])
    assert not bool(by_variant.loc["clean_eps_pass_no_dry_no_geom_caution", "imax_selected"])


def test_semiconductor_capture_audit_uses_pool_industry_labels(tmp_path):
    pool_dir = tmp_path / "2026-01-02"
    pool_dir.mkdir()
    pd.DataFrame(
        [
            {"code": "TSM", "signal": True, "sector": "Electronic Technology", "industry": "Semiconductors"},
            {"code": "SOFT", "signal": True, "sector": "Technology Services", "industry": "Packaged Software"},
        ]
    ).to_csv(pool_dir / "breakout_follow_pool.csv", index=False)
    trades = pd.DataFrame(
        [
            {
                "eps_mode": "with_eps",
                "variant": "sample",
                "event": "rebalance",
                "snapshot_date": "2026-01-02",
                "target_codes": "TSM,SOFT",
            }
        ]
    )

    audit = semiconductor_capture_audit(trades, pool_root=tmp_path)

    row = audit.iloc[0]
    assert row["semi_signal_weeks"] == 1
    assert row["semi_hit_weeks"] == 1
    assert row["semi_pick_slots"] == 1
    assert row["unique_semis"] == "TSM"


def test_stop_loss_capital_backtest_caps_stopped_picks_and_compounds_weekly():
    picks = pd.DataFrame(
        [
            {"eps_mode": "with_eps", "variant": "skill_industry_eps_known", "snapshot_date": "2026-01-02", "code": "A", "latest_return_pct": 20.0, "hit_stop_8pct": False},
            {"eps_mode": "with_eps", "variant": "skill_industry_eps_known", "snapshot_date": "2026-01-02", "code": "B", "latest_return_pct": 50.0, "hit_stop_8pct": True},
            {"eps_mode": "with_eps", "variant": "skill_industry_eps_known", "snapshot_date": "2026-01-09", "code": "C", "latest_return_pct": 10.0, "hit_stop_8pct": False},
        ]
    )

    result = stop_loss_capital_backtest(
        picks,
        eps_mode="with_eps",
        variant="skill_industry_eps_known",
        initial_capital=1000.0,
        stop_loss_pct=-8.0,
    )

    assert result["weeks"] == 2
    assert result["initial_capital"] == 1000.0
    assert result["first_week_return_pct"] == 6.0
    assert round(result["final_equity"], 2) == 1166.0
    assert round(result["total_return_pct"], 2) == 16.6


def test_rule_status_coverage_keeps_pullback_and_status_visible():
    universe = pd.DataFrame(
        [
            {"snapshot_date": "2026-07-24", "code": "PLSE", "rule": "ceiling_pullback", "entry_status": "ACTIONABLE", "latest_rank": 2, "valid_path": True},
            {"snapshot_date": "2026-07-24", "code": "HALO", "rule": "ceiling", "entry_status": "UNCONFIRMED", "latest_rank": 4, "valid_path": True},
        ]
    )
    picks = pd.DataFrame(
        [
            {"variant": "shadow", "snapshot_date": "2026-07-24", "code": "HALO", "entry_status": "UNCONFIRMED", "hit_latest_top5": True, "hit_gain_top5": False, "hit_loss_bottom5": False, "hit_stop_8pct": False}
        ]
    )

    coverage = rule_status_coverage(universe, picks)

    halo = coverage[coverage["rule"].eq("ceiling") & coverage["entry_status"].eq("UNCONFIRMED")].iloc[0]
    plse = coverage[coverage["rule"].eq("ceiling_pullback") & coverage["entry_status"].eq("ACTIONABLE")].iloc[0]
    assert halo["picks"] == 1
    assert halo["top5_picks"] == 1
    assert plse["universe_top5"] == 1
    assert plse["picks"] == 0


def test_render_markdown_report_documents_rd_agent_boundary():
    quality = pd.DataFrame(
        [
            {"eps_mode": "with_eps", "variant": "skill_industry_eps_known", "median_week_avg_latest_return_pct": 16.0, "min_week_avg_latest_return_pct": -4.0, "top5_coverage": 0.05, "bottom5_exposure_vs_slots": 0.07, "pick_stop_rate": 0.15},
        ]
    )
    imax = pd.DataFrame(
        [
            {"variant": "skill_industry_eps_known", "selected": True, "pick_order": 2, "latest_rank": 6, "gain_rank": 8},
        ]
    )
    coverage = pd.DataFrame(
        [
            {"rule": "ceiling_pullback", "entry_status": "ACTIONABLE", "universe_top5": 9, "picks": 1, "top5_picks": 1},
        ]
    )

    markdown = render_markdown_report(quality, imax, coverage)

    assert "RD-Agent 只生成候选假设" in markdown
    assert "不得直接改写正式 skill" in markdown
    assert "ceiling_pullback" in markdown
    assert "IMAX" in markdown
