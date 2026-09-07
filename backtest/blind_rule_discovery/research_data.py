"""Build the long-history market-data bundle used by blind rule discovery.

This is intentionally separate from the production rolling cache. Yahoo's
historical ``Close``/OHLC series with ``auto_adjust=False`` is split-adjusted
but not dividend-adjusted; that is the price basis wanted for executable
stop/target research. Daily and weekly bars are downloaded independently from
Yahoo so replay receives the same interval semantics as production.
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

from backtest.latest_quant_trade_replay.runner import (
    _history_commit_rows,
    _pkl_paths_at_commit,
    load_git_pickle_data,
)
from market_universe import build_download_universe
from .outcomes import RESEARCH_PRICE_MODE

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


def _symbols_from_git_historical_pkls(repo_path: Path) -> set[str]:
    """Best-effort union of symbols that appeared in committed historical 1d caches."""
    symbols: set[str] = set()
    seen_paths: set[str] = set()
    try:
        commits = _history_commit_rows(repo_path, "2000-01-01", "2100-01-01")
    except Exception:
        return symbols
    for commit, _ in commits:
        try:
            paths = _pkl_paths_at_commit(repo_path, commit)
        except Exception:
            continue
        for _, period, path in paths:
            if period != "1d" or path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                raw = load_git_pickle_data(repo_path, commit, path)
            except Exception:
                continue
            symbols.update(
                symbol
                for symbol in (_normalize_symbol(v) for v in raw.keys())
                if symbol and not symbol.startswith("__")
            )
    return symbols


def collect_research_universe(
    *,
    data_root: Path = Path("."),
    replay_roots: Iterable[Path] = (DEFAULT_REPLAY_ROOT,),
    seed_pkl: Path | None = DEFAULT_SEED_PKL,
    include_git_history: bool = True,
) -> list[str]:
    """Union current strategy inputs with every symbol already known to the repo.

    Git historical pkl symbols reduce current-cache survivorship, but this still
    is not a complete point-in-time historical listings database. The manifest
    retains that limitation explicitly.
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
    if include_git_history:
        symbols.update(_symbols_from_git_historical_pkls(data_root))
    symbols.update(BENCHMARK_CODES)
    return sorted(symbols)


def _download_one(
    symbol: str,
    *,
    start: str,
    end: str | None,
    interval: str,
    timeout: float,
) -> tuple[str, pd.DataFrame | None, str]:
    try:
        data = yf.Ticker(symbol).history(
            start=start,
            end=end,
            interval=interval,
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
    out.attrs["interval"] = interval
    out.attrs["auto_adjust"] = False
    out.attrs["repair"] = True
    out.attrs["rounding"] = False
    return symbol, out, ""


def _download_interval(
    universe: list[str],
    *,
    start: str,
    end: str | None,
    interval: str,
    max_workers: int,
    timeout: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    data: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = {
            executor.submit(
                _download_one,
                symbol,
                start=start,
                end=end,
                interval=interval,
                timeout=timeout,
            ): symbol
            for symbol in universe
        }
        for future in as_completed(futures):
            symbol, frame, reason = future.result()
            if frame is None:
                failures[symbol] = reason
            else:
                data[symbol] = frame
    return data, failures


def daily_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Audit/test utility: aggregate daily bars using actual final session labels.

    Canonical replay does not use this function; it downloads Yahoo 1wk bars
    directly to match production interval semantics.
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
    min_coverage: float = 0.98,
) -> dict[str, object]:
    if not universe:
        raise ValueError("research universe is empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[blind-data] downloading {len(universe)} symbols at 1d")
    daily, daily_failures = _download_interval(
        universe,
        start=start,
        end=end,
        interval="1d",
        max_workers=max_workers,
        timeout=timeout,
    )
    print(f"[blind-data] downloading {len(universe)} symbols at 1wk")
    weekly, weekly_failures = _download_interval(
        universe,
        start=start,
        end=end,
        interval="1wk",
        max_workers=max_workers,
        timeout=timeout,
    )

    requested = len(universe)
    daily_downloaded = len(daily)
    weekly_downloaded = len(weekly)
    daily_coverage = daily_downloaded / requested if requested else 0.0
    weekly_coverage = weekly_downloaded / requested if requested else 0.0
    joint_symbols = sorted(set(daily) & set(weekly))
    joint_coverage = len(joint_symbols) / requested if requested else 0.0
    if "SPY" not in daily or "SPY" not in weekly:
        raise RuntimeError("canonical research bundle requires SPY in both 1d and 1wk data")
    if joint_coverage < min_coverage:
        raise RuntimeError(
            f"joint research price coverage {joint_coverage:.3f} below minimum {min_coverage:.3f}; "
            f"joint={len(joint_symbols)} requested={requested}"
        )

    daily = {symbol: daily[symbol] for symbol in joint_symbols}
    weekly = {symbol: weekly[symbol] for symbol in joint_symbols}

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
        "yfinance_version": getattr(yf, "__version__", None),
        "price_adjustment_mode": RESEARCH_PRICE_MODE,
        "price_semantics": "Yahoo OHLC split-adjusted, not dividend-adjusted; suitable for price-only stop/target paths",
        "daily_interval": "1d",
        "weekly_interval": "1wk",
        "weekly_source": "direct_yahoo_1wk_not_daily_resample",
        "auto_adjust": False,
        "repair": True,
        "rounding": False,
        "start": start,
        "end_exclusive": end,
        "universe_mode": "current_strategy_plus_committed_replay_seed_cache_and_git_historical_pkls",
        "universe_limitation": "best-effort union of repo-known symbols, not a complete point-in-time historical listings database; survivorship remains",
        "requested_symbols": requested,
        "daily_downloaded_symbols": daily_downloaded,
        "weekly_downloaded_symbols": weekly_downloaded,
        "joint_downloaded_symbols": len(joint_symbols),
        "daily_coverage_before_intersection": daily_coverage,
        "weekly_coverage_before_intersection": weekly_coverage,
        "coverage": joint_coverage,
        "failed_symbols_daily": daily_failures,
        "failed_symbols_weekly": weekly_failures,
        "benchmark_codes_requested": list(BENCHMARK_CODES),
        "benchmark_codes_downloaded": [
            code for code in BENCHMARK_CODES if code in daily and code in weekly
        ],
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
    parser.add_argument("--min-coverage", type=float, default=0.98)
    parser.add_argument(
        "--no-git-history-universe",
        action="store_true",
        help="debug only; canonical build includes symbols found in committed historical 1d pkls",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    replay_roots = args.replay_root or [DEFAULT_REPLAY_ROOT]
    universe = collect_research_universe(
        data_root=Path("."),
        replay_roots=replay_roots,
        seed_pkl=args.seed_pkl,
        include_git_history=not args.no_git_history_universe,
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
    print(
        json.dumps(
            {
                "status": "ok",
                "requested_symbols": manifest["requested_symbols"],
                "joint_downloaded_symbols": manifest["joint_downloaded_symbols"],
                "coverage": manifest["coverage"],
                "daily_path": manifest["daily_path"],
                "weekly_path": manifest["weekly_path"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
