from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

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
