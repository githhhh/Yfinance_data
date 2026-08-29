from pathlib import Path

from backtest.b0_top3_quality_audit.generate_b0_rank_topk_audit import (
    describe_statistical_support,
)
from backtest.b0_top3_quality_audit.run_eps_restatement import (
    _write_restatement_report,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]


def test_restatement_builder_has_quantitative_sections_and_corrected_verdicts(tmp_path):
    audit_root = ROOT / "backtest/b0_top3_quality_audit"
    report_path = _write_restatement_report(
        "593bd333181da4fe301b3f61397c7bc95ac86ced",
        {"e0_membership_changed_count": 22, "e0_affected_weeks": 10,
         "b0_selected_count_changed_weeks": 0, "b0_codes_changed_weeks": 6,
         "b0_order_only_changed_weeks": 2},
        {"price": sha256_file(audit_root / "data/signal_daily_prices.parquet"),
         "weekly": sha256_file(audit_root / "data/candidate_weekly_outcomes.parquet"),
         "train_weekly": sha256_file(audit_root / "data/frozen/train_candidate_weekly_outcomes.parquet")},
        output_path=tmp_path / "restatement.md",
    )
    report = report_path.read_text()
    for heading in (
        "EPS PIT Data Revision",
        "All-Historical Old -> New: Three-Tier",
        "B0 vs Matched-N Random",
        "Matched Random Percentile Old -> New",
        "Rank Diagnostics and Top3 vs Top2 / MC3",
        "Rank1 / Rank2 / Rank3 Median Return Old -> New",
        "EPS25 Tightening Probe",
    ):
        assert heading in report
    for verdict in (
        "| EPS Known | RETAINED |",
        "| R3 vs R2 | WEAKENED |",
        "| Top3 vs Top2 / MC3 | WEAKENED |",
        "| B0 W4 quality | STRENGTHENED |",
        "| Layer2 | RETAINED |",
    ):
        assert verdict in report
    assert "independent return alpha not demonstrated" in report
    assert "Old | New | Delta" in report


def test_validation_generator_uses_fixed_calendar_and_v2_manifest():
    source = (ROOT / "backtest/b0_top3_quality_audit/historical_validation_verifier.py").read_text()
    assert "len(validation_weeks)" in source
    assert "frozen_rules_manifest_eps_recalibrated_v2.json" not in source
    report = (ROOT / "backtest/b0_top3_quality_audit/output/CONTAMINATED_HISTORICAL_VALIDATION_REPORT.md").read_text()
    assert "2026-05-29" in report and "2026-08-07" in report and "11 个 snapshot weeks" in report
    assert "frozen_rules_manifest_eps_recalibrated_v2.json" in report
    assert "共 10 周" not in report


def test_rank_significance_formatter_marks_p2872_directional_only():
    assert describe_statistical_support(0.2872) == "directional and not statistically significant"
