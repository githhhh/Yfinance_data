"""Industry enrichment for published BreakoutFollow pools.

Industry is a display attribute sourced from screener inputs.  It is kept
separate from EPS PIT because missing industry metadata must not block Pool
publication.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


INDUSTRY_SOURCE_FILES = (
    "us/weekly_vol_screener_results.csv",
    "us/52wk_new_high_results.csv",
    "us/eps_growth_screener_results.csv",
    "us/stage2/stage2_whitelist.csv",
)
INDUSTRY_COLUMNS = ("sector", "industry")


def _default_data_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _clean_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None


def load_industry_lookup(*, data_root: str | Path | None = None) -> dict[str, tuple[str, str]]:
    """Load screener industry values from low to high authority priority."""
    root = Path(data_root) if data_root is not None else _default_data_root()
    lookup: dict[str, tuple[str, str]] = {}

    for relative_path in INDUSTRY_SOURCE_FILES:
        source_path = root / relative_path
        if not source_path.exists():
            logging.warning("BF Pool industry source missing: %s", source_path)
            continue
        try:
            source = pd.read_csv(source_path, dtype={"code": str})
        except Exception as exc:
            logging.warning("BF Pool industry source unreadable: %s (%s)", source_path, exc)
            continue
        if not {"code", *INDUSTRY_COLUMNS}.issubset(source.columns):
            logging.warning("BF Pool industry source missing required columns: %s", source_path)
            continue

        resolved = 0
        for _, row in source.iterrows():
            code = _clean_text(row["code"])
            sector = _clean_text(row["sector"])
            industry = _clean_text(row["industry"])
            if code is None or sector is None or industry is None:
                continue
            lookup[code] = (sector, industry)
            resolved += 1
        logging.info("BF Pool industry source %s: %s mappings", relative_path, resolved)

    logging.info("BF Pool industry lookup loaded: %s mappings", len(lookup))
    return lookup


def enrich_pool_with_industry(
    pool: pd.DataFrame,
    *,
    lookup: dict[str, tuple[str, str]] | None = None,
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    """Attach source-owned industry fields without rejecting unresolved codes."""
    if "code" not in pool.columns:
        raise ValueError("BF Pool 缺少字段: ['code']")

    result = pool.copy()
    industry_lookup = lookup if lookup is not None else load_industry_lookup(data_root=data_root)
    result["sector"] = pd.NA
    result["industry"] = pd.NA
    unresolved: list[str] = []

    for index, raw_code in result["code"].items():
        code = _clean_text(raw_code)
        details = industry_lookup.get(code) if code is not None else None
        if details is None:
            if code is not None:
                unresolved.append(code)
            continue
        result.at[index, "sector"], result.at[index, "industry"] = details

    if unresolved:
        logging.warning("BF Pool industry unresolved codes: %s", ", ".join(sorted(set(unresolved))))
    return result
