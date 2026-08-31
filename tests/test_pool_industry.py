from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from yfinance_data import pool_industry
from yfinance_data.pool_industry import enrich_pool_with_industry, load_industry_lookup


def _write_industry_source(
    root: Path,
    relative_path: str,
    rows: list[dict[str, object]],
) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_pool_industry_uses_current_screener_priority_and_warns_without_blocking(tmp_path, caplog):
    _write_industry_source(
        tmp_path,
        "us/weekly_vol_screener_results.csv",
        [{"code": "DUP", "sector": "Old sector", "industry": "Old industry"}],
    )
    _write_industry_source(
        tmp_path,
        "us/52wk_new_high_results.csv",
        [{"code": "HIGH", "sector": "High sector", "industry": "High industry"}],
    )
    _write_industry_source(
        tmp_path,
        "us/eps_growth_screener_results.csv",
        [{"code": "DUP", "sector": "EPS sector", "industry": "EPS industry"}],
    )
    _write_industry_source(
        tmp_path,
        "us/stage2/stage2_whitelist.csv",
        [{"code": "DUP", "sector": "Stage2 sector", "industry": "Stage2 industry"}],
    )

    lookup = load_industry_lookup(data_root=tmp_path)
    assert lookup["DUP"] == ("Stage2 sector", "Stage2 industry")
    assert lookup["HIGH"] == ("High sector", "High industry")

    with caplog.at_level(logging.WARNING):
        enriched = enrich_pool_with_industry(
            pd.DataFrame(
                [
                    {"code": "DUP", "sector": "legacy", "industry": "legacy"},
                    {"code": "UNKNOWN"},
                ]
            ),
            lookup=lookup,
        )

    by_code = enriched.set_index("code")
    assert by_code.loc["DUP", "sector"] == "Stage2 sector"
    assert by_code.loc["DUP", "industry"] == "Stage2 industry"
    assert pd.isna(by_code.loc["UNKNOWN", "sector"])
    assert pd.isna(by_code.loc["UNKNOWN", "industry"])
    assert "BF Pool industry unresolved codes: UNKNOWN" in caplog.text


def test_pool_industry_reuses_the_lookup_for_the_same_data_root(tmp_path, monkeypatch):
    for relative_path in pool_industry.INDUSTRY_SOURCE_FILES:
        _write_industry_source(
            tmp_path,
            relative_path,
            [{"code": "AAA", "sector": "Technology", "industry": "Software"}],
        )

    read_csv = pool_industry.pd.read_csv
    calls = []

    def count_industry_source_reads(*args, **kwargs):
        calls.append(args[0])
        return read_csv(*args, **kwargs)

    monkeypatch.setattr(pool_industry.pd, "read_csv", count_industry_source_reads)

    first = load_industry_lookup(data_root=tmp_path)
    second = load_industry_lookup(data_root=tmp_path)

    assert first == second
    assert len(calls) == len(pool_industry.INDUSTRY_SOURCE_FILES)
