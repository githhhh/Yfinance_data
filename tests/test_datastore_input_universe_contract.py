from pathlib import Path

import pandas as pd

import DataStore


def _write_codes(root: Path, filename: str, codes: list[str]) -> None:
    pd.DataFrame({"code": codes}).to_csv(root / filename, index=False)


def test_read_stock_list_preserves_the_named_business_input_union(tmp_path):
    stock_list_dir = tmp_path / "us"
    stock_list_dir.mkdir()
    _write_codes(stock_list_dir, "52wk_new_high_results.csv", ["HIGH", "DOT.NAME"])
    _write_codes(stock_list_dir, "breakout_follow_pool.csv", ["POOL", "HIGH"])
    _write_codes(stock_list_dir, "breakout_follow_pool_midweek.csv", ["MID"])
    _write_codes(stock_list_dir, "eps_growth_screener_results.csv", ["EPS"])
    _write_codes(stock_list_dir, "weekly_vol_screener_results.csv", ["VOL"])

    assert set(DataStore.read_stock_list(str(stock_list_dir))) == {
        "DOT-NAME",
        "EPS",
        "HIGH",
        "MID",
        "POOL",
        "VOL",
    }
