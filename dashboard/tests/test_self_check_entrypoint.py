import subprocess
import sys


def test_self_check_runs_as_direct_script(tmp_path):
    csv_path = tmp_path / "breakout_follow_pool.csv"
    csv_path.write_text(
        "\n".join(
            [
                "code,signal,signal_source,ibd_candidate_rule,ibd_entry_valid,ibd_entry_volume_ratio,ibd_entry_close_vs_trigger_pct,volume_ratio,pullback_v_is_dry,breakout_date,pct_above_ceiling,touched_ema10_count,rank_C_continuous,ibd_entry_price,sector,industry",
                "AAA,True,ceiling_breakout,ceiling_pullback,1,2.5,0.04,1.4,True,2026-05-10,4.0,2,2,10.0,Tech,Software",
                "BBB,True,pivot,pivot,0,3.5,0.02,1.6,False,2026-04-15,8.0,5,1,12.0,Tech,Software",
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
