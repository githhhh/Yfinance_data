from pathlib import Path
from dashboard.data_utils import get_latest_pool_csv_path


def test_get_latest_pool_csv_path(tmp_path: Path):
    complete_csv = tmp_path / "breakout_follow_pool.csv"
    midweek_csv = tmp_path / "breakout_follow_pool_midweek.csv"

    # 1. Both do not exist -> defaults to complete_path
    path = get_latest_pool_csv_path(complete_csv, midweek_csv)
    assert path == complete_csv

    # 2. Only complete_csv exists
    complete_csv.write_text("code,snapshot_date,signal\nAAPL,2026-07-31,True", encoding="utf-8")
    path = get_latest_pool_csv_path(complete_csv, midweek_csv)
    assert path == complete_csv

    # 3. Midweek has newer snapshot_date
    midweek_csv.write_text("code,snapshot_date,signal\nNVDA,2026-08-05,True", encoding="utf-8")
    path = get_latest_pool_csv_path(complete_csv, midweek_csv)
    assert path == midweek_csv

    # 4. Complete updated with newer snapshot_date
    complete_csv.write_text("code,snapshot_date,signal\nAAPL,2026-08-07,True", encoding="utf-8")
    path = get_latest_pool_csv_path(complete_csv, midweek_csv)
    assert path == complete_csv


