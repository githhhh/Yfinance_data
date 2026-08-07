from pathlib import Path
import time
import pytest
from dashboard.data_utils import get_latest_pool_csv_path


def test_get_latest_pool_csv_path(tmp_path: Path):
    complete_csv = tmp_path / "breakout_follow_pool.csv"
    midweek_csv = tmp_path / "breakout_follow_pool_midweek.csv"

    # 1. Both do not exist -> defaults to complete_path
    path = get_latest_pool_csv_path(complete_csv, midweek_csv)
    assert path == complete_csv

    # 2. Only complete_csv exists
    complete_csv.write_text("code,signal\nAAPL,True", encoding="utf-8")
    path = get_latest_pool_csv_path(complete_csv, midweek_csv)
    assert path == complete_csv

    # 3. Midweek added later -> midweek has larger st_mtime
    time.sleep(0.01)
    midweek_csv.write_text("code,signal\nNVDA,True", encoding="utf-8")
    path = get_latest_pool_csv_path(complete_csv, midweek_csv)
    assert path == midweek_csv

    # 4. Complete updated later -> complete has larger st_mtime
    time.sleep(0.01)
    complete_csv.write_text("code,signal\nAAPL,True\nMSFT,True", encoding="utf-8")
    path = get_latest_pool_csv_path(complete_csv, midweek_csv)
    assert path == complete_csv
