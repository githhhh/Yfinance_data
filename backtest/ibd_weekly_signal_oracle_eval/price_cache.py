from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import re


DAILY_CACHE_PATTERN = "stock_data_*_1d.pkl"
_DAILY_CACHE_RE = re.compile(r"stock_data_(\d{6})_1d\.pkl$")


def _daily_cache_sort_key(path: Path) -> tuple[date, str]:
    match = _DAILY_CACHE_RE.fullmatch(path.name)
    if not match:
        return date.min, path.name
    try:
        return datetime.strptime(match.group(1), "%d%m%y").date(), path.name
    except ValueError:
        return date.min, path.name


def latest_daily_price_cache(root: Path = Path("results_pkl")) -> Path:
    candidates = sorted(root.glob(DAILY_CACHE_PATTERN), key=_daily_cache_sort_key)
    if not candidates:
        raise FileNotFoundError(f"No daily price cache found under {root}/{DAILY_CACHE_PATTERN}")
    return candidates[-1]


def resolve_price_cache(path: str | Path | None) -> Path:
    if path is None or str(path).strip() == "":
        return latest_daily_price_cache()
    return Path(path)
