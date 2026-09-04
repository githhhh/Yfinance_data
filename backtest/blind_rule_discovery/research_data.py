"""Build the long-history market-data bundle used by blind rule discovery.

This is intentionally separate from the production rolling cache.  Yahoo's
historical ``Close``/OHLC series with ``auto_adjust=False`` is split-adjusted
but not dividend-adjusted; that is the price basis wanted for executable
stop/target research.  The bundle records that provenance in DataFrame attrs
and a sidecar manifest instead of requiring dividend-adjusted ``Adj Close``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import pickle
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from market_universe import build_download_universe

RESEARCH_PRICE_MODE = "yahoo_split_adjusted_ohlc_no_dividend_adjustment"
DEFAULT_PRICE_START = "2017-01-01"
DEFAULT_REPLAY_ROOT = Path("backtest/ibd_skill_replay_pools")
DEFAULT_SEED_PKL = Path("results_pkl/stock_data_290826_1d.pkl")
DEFAULT_OUTPUT_DIR = Path("backtest/blind_rule_discovery/work/prices")
BENCHMARK_CODES = ("SPY", "^GSPC")
REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_symbol(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    symbol = str(value).strip().upper().replace(".", "-")
    return symbol or None


def _symbols_from_replay_root(root: Path) -> set[str]:
    symbols: set[str] = set()
    if not root.exists():
        return symbols
    for path in root.glob("*/breakout_follow_pool.csv"):
        try:
            frame = pd.read_csv(path, dtype={"code": str}, usecols=["code"], encoding="utf-8-sig")
        except Exception:
            continue
        symbols.update(
            symbol for symbol in (_normalize_symbol(v) for v in frame["code"]) if symbol
        )
    return symbols


def _symbols_from_pickle(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        with path.open("rb") as handle:
            raw = pickle.load(handle)
    except Exception:
        return set()
    if not isinstance(raw, dict):
        return set()
    return {
        symbol for symbol in (_normalize_symbol(v) for v in raw.keys()) if symbol and not symbol.startswith("__")
    }


def collect_research_universe(
    *,
    data_root: Path = Path("."),
    replay_roots: Iterable[Path] = (DEFAULT_REPLAY_ROOT,),
    seed_pkl: Path | None = DEFAULT_SEED_PKL,
) -> list[str]:
    """Union current strategy inputs with every symbol already known to replay/cache.

    This deliberately broadens the current strategy universe, but it is not a
    historical listings database.  The manifest calls that limitation out so
    downstream research cannot silently claim point-in-time universe purity.
    """
    symbols = {
        symbol
        for symbol in (_normalize_symbol(v) for v in build_download_universe(data_root=data_root))
        if symbol
    }
    for root in replay_roots:
        symbols.update(_symbols_from_replay_root(root))
    if seed_pkl is not None:
        symbols.update(_symbols_from_pickle(seed_pkl))
    symbols.update(BENCHMARK_CODES)
    return sorted(symbols)


def _download_one(symbol: str, *, start: str, end: str | None, timeout: float) -> tuple[str, pd.DataFrame | None, str]:
    try:
        data = yf.Ticker(symbol).history(
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            actions=True,
            repair=True,
            rounding=False,
            timeout=timeout,
        )
    except Exception as exc:
        return symbol, None, f"{type(exc).__name__}: {exc}"
    if data is None or data.empty:
        return symbol, None, "empty_history"
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        return symbol, None, "missing_columns:" + ",".join(missing)
    out = data.loc[:, list(REQUIRED_COLUMNS)].copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce"))
    out = out.loc[~out.index.isna()].sort_index()
    for column in REQUIRED_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    if out.empty:
        return symbol, None, "no_valid_ohlc"
    out.attrs["price_adjustment_mode"] = RESEARCH_PRICE_MODE
    out.attrs["source"] = "yfinance.Ticker.history"
    out.attrs["auto_adjust"] = False
    out.attrs["repair"] = True
    out.attrs["rounding"] = False
    return symbol, out, ""


def daily_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily bars by exchange week and label with the actual last session.

    Using the actual final trading date avoids hard-coded holiday calendars and
    preserves shortened weeks (e.g. a Thursday close before Good Friday).
    """
    if daily.empty:
        return daily.copy()
    work = daily.loc[:, list(REQUIRED_COLUMNS)].copy().sort_index()
    idx = pd.DatetimeIndex(pd.to_datetime(work.index, errors="coerce"))
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    work["_date"] = idx.normalize()
    work = work.dropna(subset=["_date"])
    work["_week"] = work["_date"].dt.to_period("W-FRI")
    rows: list[dict[str, object]] = []
    for _, group in work.groupby("_week", sort=True):
        group = group.sort_values("_date")
        rows.append(
            {
                "date": pd.Timestamp(group.iloc[-1]["_date"]),
                "Open": float(group.iloc[0]["Open"]),
                "High": float(pd.to_numeric(group["High"], errors="coerce").max()),
                "Low": float(pd.to_numeric(group["Low"], errors="coerce").min()),
                "Close": float(group.iloc[-1]["Close"]),
                "Volume": float(pd.to_numeric(group["Volume"], errors="coerce").fillna(0).sum()),
            }
        )
    weekly = pd.DataFrame(rows).set_index("date")
    weekly.index = pd.DatetimeIndex(weekly.index)
    weekly.attrs.update(daily.attrs)
    weekly.attrs["derived_from"] = "daily_actual_session_week"
    return weekly


def build_price_bundle(
    *,
    universe: list[str],
    start: str,
    end: str | None,
    output_dir: Path,
    max_workers: int = 8,
    timeout: float = 15.0,
    min_coverage: float = 0.90,
) -> dict[str, object]:
    if not universe:
        raise ValueError("research universe is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    daily: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = {
            executor.submit(_download_one, symbol, start=start, end=end, timeout=timeout): symbol
            for symbol in universe
        }
        for future in as_completed(futures):
            symbol, frame, reason = future.result()
            if frame is None:
                failures[symbol] = reason
            else:
                daily[symbol] = frame

    requested = len(universe)
    coverage = len(daily) / requested if requested else 0.0
    if "SPY" not in daily:
        raise RuntimeError("canonical research bundle requires SPY")
    if coverage < min_coverage:
        raise RuntimeError(
            f"research price coverage {coverage:.3f} below minimum {min_coverage:.3f}; "
            f"downloaded={len(daily)} requested={requested}"
        )

    weekly = {symbol: daily_to_weekly(frame) for symbol, frame in daily.items()}
    daily_path = output_dir / "research_daily.pkl"
    weekly_path = output_dir / "research_weekly.pkl"
    with daily_path.open("wb") as handle:
        pickle.dump(daily, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with weekly_path.open("wb") as handle:
        pickle.dump(weekly, handle, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Yahoo Finance via yfinance",
        "price_adjustment_mode": RESEARCH_PRICE_MODE,
        "price_semantics": "OHLC split-adjusted, not dividend-adjusted; suitable for price-only stop/target paths",
        "auto_adjust": False,
        "repair": True,
        "rounding": False,
        "start": start,
        "end_exclusive": end,
        "universe_mode": "current_strategy_plus_known_replay_and_seed_cache",
        "universe_limitation": "not a point-in-time historical listings database; survivorship remains a documented limitation",
        "requested_symbols": requested,
        "downloaded_symbols": len(daily),
        "coverage": coverage,
        "failed_symbols": failures,
        "benchmark_codes_requested": list(BENCHMARK_CODES),
        "benchmark_codes_downloaded": [code for code in BENCHMARK_CODES if code in daily],
        "daily_path": str(daily_path),
        "weekly_path": str(weekly_path),
        "daily_sha256": _sha256_file(daily_path),
        "weekly_sha256": _sha256_file(weekly_path),
    }
    (output_dir / "price_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_PRICE_START)
    parser.add_argument("--end", default=None, help="exclusive YYYY-MM-DD; omit for latest available")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed-pkl", type=Path, default=DEFAULT_SEED_PKL)
    parser.add_argument("--replay-root", type=Path, action="append", default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    replay_roots = args.replay_root or [DEFAULT_REPLAY_ROOT]
    universe = collect_research_universe(
        data_root=Path("."), replay_roots=replay_roots, seed_pkl=args.seed_pkl
    )
    manifest = build_price_bundle(
        universe=universe,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        timeout=args.timeout,
        min_coverage=args.min_coverage,
    )
    print(json.dumps({
        "status": "ok",
        "requested_symbols": manifest["requested_symbols"],
        "downloaded_symbols": manifest["downloaded_symbols"],
        "coverage": manifest["coverage"],
        "daily_path": manifest["daily_path"],
        "weekly_path": manifest["weekly_path"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
