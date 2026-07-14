import subprocess
import sys


def test_self_check_runs_as_direct_script(tmp_path):
    csv_path = tmp_path / "breakout_follow_pool.csv"
    csv_path.write_text(
        "\n".join(
            [
                "code,signal,signal_source,ibd_candidate_rule,ibd_entry_valid,ibd_entry_volume_ratio,ibd_entry_close_vs_trigger_pct,ibd_entry_close_position,ibd_entry_breakout_range_ratio,volume_ratio,is_bullish,pullback_v_is_dry,breakout_date,pct_above_ceiling,touched_ema10_count,rank_C_continuous,ibd_entry_price,sector,industry,latest_close,ibd_candidate_price,current_vs_ibd_candidate_pct,price_52_week_high,dist_to_52w_high_pct,ibd_entry_status",
                "AAA,True,ceiling_breakout,ceiling_pullback,1,2.5,0.04,0.80,1.20,1.4,True,True,2026-05-10,4.0,2,2,10.0,Technology Services,Software - Enterprise,10.4,10.0,4.0,12.0,-13.33,ACTIONABLE",
                "BBB,True,pivot,pivot,0,3.5,0.02,0.40,0.60,1.6,False,False,2026-04-15,8.0,5,1,12.0,Finance,Regional Banks,12.24,12.0,2.0,15.0,-18.40,UNCONFIRMED",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "dashboard/self_check.py", "--csv", str(csv_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "[PASS] load and normalize" in result.stdout


def test_self_check_reports_setup_failures_with_label(tmp_path):
    missing_csv = tmp_path / "missing.csv"

    result = subprocess.run(
        [sys.executable, "dashboard/self_check.py", "--csv", str(missing_csv)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "[FAIL] setup:" in result.stderr
