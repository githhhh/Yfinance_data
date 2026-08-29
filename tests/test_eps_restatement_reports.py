from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_restatement_report_has_quantitative_sections_and_corrected_verdicts():
    report = (ROOT / "backtest/b0_top3_quality_audit/output/EPS_RECALIBRATION_RESEARCH_RESTATEMENT.md").read_text()
    for heading in (
        "EPS PIT Data Revision",
        "All-Historical Old -> New: Three-Tier",
        "B0 vs Matched-N Random",
        "Rank Diagnostics and Top3 vs Top2 / MC3",
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


def test_rank_report_never_calls_p2872_significant():
    report = (ROOT / "backtest/b0_top3_quality_audit/output/B0_RANK_POSITION_TOPK_AUDIT_REPORT.md").read_text()
    line = next(line for line in report.splitlines() if "p=0.2872" in line)
    assert "not statistically significant" in line
    assert "显著战胜" not in line
