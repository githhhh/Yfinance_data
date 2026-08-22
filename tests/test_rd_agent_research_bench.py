import pandas as pd

from backtest.rd_agent_research_bench.metrics import (
    coverage_rates,
    rank_weighted_week_return,
    robust_weekly_summary,
)
from backtest.rd_agent_research_bench.research import (
    imax_rank_audit,
    render_markdown_report,
    rule_status_coverage,
    summarize_variant_quality,
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
