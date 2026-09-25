#!/usr/bin/env python3
"""Pine-compatible RS snapshot built from a validated 1d PKL.

Manual backfill after the daily download has finished (no download is run)::

    python rs_snapshot.py --pkl results_pkl/stock_data_DDMMYY_1d.pkl

Omit --pkl to use the newest daily PKL in results_pkl/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
import tempfile
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

RESULTS_DIR = Path("results_pkl")
CACHE_DIR = Path("output/rs_snapshot_cache")
CACHE_PATH = CACHE_DIR / "rsrating_cache.json"
RSRATING_URL = "https://raw.githubusercontent.com/Fred6725/rs-log/main/output/RSRATING.csv"
REFERENCE_TICKER = "^GSPC"
LOOKBACKS = (63, 126, 189, 252)
WEIGHTS = (0.4, 0.2, 0.2, 0.2)
MAX_ANCHOR_AGE = 5


class RSSnapshotError(RuntimeError):
    pass


class StaleAnchorError(RSSnapshotError):
    pass


def load_daily_pkl(path: Path) -> dict[str, pd.DataFrame]:
    with path.open("rb") as fh:
        data = pickle.load(fh)
    out = {}
    for ticker, value in data.items():
        if isinstance(value, pd.DataFrame):
            out[str(ticker)] = value.copy()
        elif isinstance(value, dict) and set(value) == {"index", "columns", "data"}:
            out[str(ticker)] = pd.DataFrame(**value)
        else:
            raise RSSnapshotError(f"Unsupported PKL payload for {ticker}")
    return out


def find_latest_daily_pkl(results_dir: Path = RESULTS_DIR) -> Path:
    found = []
    for path in results_dir.glob("stock_data_*_1d.pkl"):
        try:
            token = path.name[len("stock_data_") : -len("_1d.pkl")]
            found.append((datetime.strptime(token, "%d%m%y").date(), path))
        except ValueError:
            continue
    if not found:
        raise RSSnapshotError(f"No daily PKL found under {results_dir}")
    return max(found, key=lambda item: item[0])[1]


def close_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    if "Close" not in df.columns:
        raise RSSnapshotError(f"{ticker}: Close column missing")
    series = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if series.empty:
        raise RSSnapshotError(f"{ticker}: no valid Close data")
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def session_dates(reference: pd.Series) -> list[date]:
    return list(dict.fromkeys(pd.Timestamp(ts).date() for ts in reference.index))


def normalize_anchor_date(source_date: date, sessions: Sequence[date]) -> date:
    eligible = [d for d in sessions if d <= source_date]
    if not eligible:
        raise RSSnapshotError("RSRATING source date predates benchmark history")
    return eligible[-1]


def anchor_age(anchor: date, sessions: Sequence[date]) -> int:
    if not sessions:
        raise RSSnapshotError("Benchmark has no trading sessions")
    if anchor > sessions[-1]:
        raise RSSnapshotError(
            f"AnchorDate {anchor.isoformat()} is newer than benchmark {sessions[-1].isoformat()}"
        )
    return sum(1 for d in sessions if d > anchor)


def parse_rsrating_csv(text: str) -> tuple[list[float], date]:
    rows = []
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        if len(row) < 5:
            raise RSSnapshotError("Malformed RSRATING row")
        try:
            d = datetime.strptime(row[0].strip().rstrip("T"), "%Y%m%d").date()
            value = float(row[4])
        except (TypeError, ValueError) as exc:
            raise RSSnapshotError(f"Malformed RSRATING row: {row}") from exc
        if not math.isfinite(value) or value <= 0:
            raise RSSnapshotError(f"Invalid RSRATING anchor: {value}")
        rows.append((d, value))

    if len(rows) != 35:
        raise RSSnapshotError(f"Expected 35 RSRATING rows, got {len(rows)}")

    unique = []
    counts = {}
    for _, value in rows:
        counts[value] = counts.get(value, 0) + 1
        if value not in unique:
            unique.append(value)
    if len(unique) != 7:
        raise RSSnapshotError(f"Expected 7 RSRATING anchors, got {len(unique)}")
    if any(counts[value] != 5 for value in unique):
        raise RSSnapshotError("Each RSRATING anchor must be repeated exactly 5 times")

    anchors = sorted(unique, reverse=True)
    return anchors, max(d for d, _ in rows)


def read_cache() -> dict:
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def write_cache(payload: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="rsrating_", suffix=".json", dir=CACHE_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, CACHE_PATH)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def cached_anchors(cache: dict) -> tuple[list[float], date]:
    try:
        anchors = sorted([float(x) for x in cache["anchors"]], reverse=True)
        source_date = date.fromisoformat(cache["source_date"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RSSnapshotError("RSRATING cache is unusable") from exc
    if len(anchors) != 7 or any(not math.isfinite(x) or x <= 0 for x in anchors):
        raise RSSnapshotError("RSRATING cache anchors are invalid")
    return anchors, source_date


def get_rsrating_anchors(today: date | None = None) -> tuple[list[float], date, str]:
    today = today or date.today()
    cache = read_cache()
    if cache.get("last_check_date") == today.isoformat():
        anchors, source_date = cached_anchors(cache)
        return anchors, source_date, "cache"

    attempted = dict(cache)
    attempted["last_check_date"] = today.isoformat()
    try:
        req = urllib.request.Request(
            RSRATING_URL, headers={"User-Agent": "Yfinance_data-rs-snapshot/1.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            anchors, source_date = parse_rsrating_csv(response.read().decode("utf-8"))
        attempted.update({"anchors": anchors, "source_date": source_date.isoformat()})
        write_cache(attempted)
        return anchors, source_date, "remote"
    except Exception as exc:
        try:
            write_cache(attempted)
        except OSError:
            pass
        try:
            anchors, source_date = cached_anchors(cache)
        except RSSnapshotError:
            raise RSSnapshotError(
                f"RSRATING refresh failed and no usable cache exists: {exc}"
            ) from exc
        return anchors, source_date, "cache-fallback"


def raw_rs(stock: pd.Series, benchmark: pd.Series) -> float:
    stock = pd.to_numeric(stock, errors="coerce").dropna()
    benchmark = pd.to_numeric(benchmark, errors="coerce").dropna()
    if stock.empty or benchmark.empty:
        raise RSSnapshotError("Insufficient close history")
    stock_values = stock.to_numpy(dtype=float)
    benchmark_dates = pd.Index([pd.Timestamp(ts).date() for ts in benchmark.index])
    stock_dates = pd.Index([pd.Timestamp(ts).date() for ts in stock.index])
    if benchmark_dates.has_duplicates or stock_dates.has_duplicates:
        raise RSSnapshotError("Duplicate daily Close dates")
    # Pine's request.security() aligns the SPX close to each stock chart bar.
    # A stock can lack an older session even when its latest bar is validated.
    benchmark_on_stock_bars = pd.Series(benchmark.to_numpy(dtype=float), index=benchmark_dates)
    benchmark_values = benchmark_on_stock_bars.reindex(stock_dates, method="ffill").to_numpy()
    if not all(math.isfinite(value) for value in benchmark_values):
        raise RSSnapshotError("Benchmark history does not cover stock bars")

    bar_index = len(stock_values) - 1
    ticker_perf = []
    benchmark_perf = []
    for lookback in LOOKBACKS:
        n = min(bar_index, lookback)
        if n >= len(benchmark_values):
            raise RSSnapshotError("Benchmark history shorter than Pine lookback")
        stock_base = stock_values[-1 - n]
        benchmark_base = benchmark_values[-1 - n]
        if stock_base <= 0 or benchmark_base <= 0:
            raise RSSnapshotError("Non-positive Close encountered")
        ticker_perf.append(stock_values[-1] / stock_base)
        benchmark_perf.append(benchmark_values[-1] / benchmark_base)

    stock_score = sum(w * p for w, p in zip(WEIGHTS, ticker_perf))
    ref_score = sum(w * p for w, p in zip(WEIGHTS, benchmark_perf))
    if ref_score <= 0:
        raise RSSnapshotError("Invalid benchmark RS denominator")
    return stock_score / ref_score * 100.0


def attribute_percentile(
    score: float,
    taller: float,
    smaller: float,
    range_up: float,
    range_dn: float,
    weight: float,
) -> float:
    adjusted = score + (score - smaller) * weight
    adjusted = min(adjusted, taller - 1)
    k1 = smaller / range_dn
    k2 = (taller - 1) / range_up
    span = taller - 1 - smaller
    if span == 0:
        raise RSSnapshotError("Degenerate RSRATING anchors")
    k3 = (k1 - k2) / span
    denominator = k1 - k3 * (score - smaller)
    if denominator == 0:
        raise RSSnapshotError("Invalid percentile denominator")
    return min(range_up, max(range_dn, adjusted / denominator))


def rs_rating(score: float, anchors: Sequence[float]) -> float:
    if len(anchors) != 7:
        raise RSSnapshotError("Exactly 7 anchors are required")
    first, scnd, thrd, frth, ffth, sxth, svth = anchors
    if score >= first:
        return 99.0
    if score <= svth:
        return 1.0

    bands = (
        (first, scnd, 98, 90, 0.33),
        (scnd, thrd, 89, 70, 2.1),
        (thrd, frth, 69, 50, 0.0),
        (frth, ffth, 49, 30, 0.0),
        (ffth, sxth, 29, 10, 0.0),
        (sxth, svth, 9, 2, 0.0),
    )
    for taller, smaller, up, down, weight in bands:
        if smaller <= score < taller:
            return attribute_percentile(score, taller, smaller, up, down, weight)
    raise RSSnapshotError(f"Score {score} did not match an RS band")


def display_rating(value: float) -> int:
    return int(math.floor(value + 0.5))


def remove_old_snapshots(results_dir: Path, keep: Path | None = None) -> None:
    for path in results_dir.glob("rs_data_*.csv"):
        if keep is not None and path.resolve() == keep.resolve():
            continue
        path.unlink(missing_ok=True)


def cleanup_stale_published_snapshots(
    sessions: Sequence[date], results_dir: Path = RESULTS_DIR
) -> list[Path]:
    """Remove already-published RS snapshots whose AnchorDate is >=5 sessions old.

    This is used as a fallback when today's RSRATING refresh cannot produce a
    usable anchor. It prevents an old RS CSV from lingering indefinitely simply
    because the remote anchor source or local cache is unavailable.
    """
    removed: list[Path] = []
    for path in results_dir.glob("rs_data_*.csv"):
        try:
            with path.open("r", newline="", encoding="utf-8") as fh:
                row = next(csv.DictReader(fh))
            published_anchor = date.fromisoformat(row["AnchorDate"])
            age = anchor_age(published_anchor, sessions)
        except (OSError, StopIteration, KeyError, TypeError, ValueError, RSSnapshotError):
            path.unlink(missing_ok=True)
            removed.append(path)
            continue
        if age >= MAX_ANCHOR_AGE:
            path.unlink(missing_ok=True)
            removed.append(path)
    return removed


def write_snapshot(rows: list[dict], run_date: date, results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / f"rs_data_{run_date.strftime('%d%m%y')}.csv"
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=results_dir)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["Ticker", "RS", "AnchorDate"],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: -int(row["RS"])))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    # The new snapshot is published before old rs_data_*.csv files are removed.
    remove_old_snapshots(results_dir, keep=target)
    return target


def compute_rows(
    data: dict[str, pd.DataFrame],
    anchors: Sequence[float],
    anchor: date,
    requested: Sequence[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    if REFERENCE_TICKER not in data:
        raise RSSnapshotError(f"{REFERENCE_TICKER} missing from daily PKL")
    benchmark = close_series(data[REFERENCE_TICKER], REFERENCE_TICKER)
    age = anchor_age(anchor, session_dates(benchmark))
    if age >= MAX_ANCHOR_AGE:
        raise StaleAnchorError(
            f"AnchorDate {anchor.isoformat()} is {age} US trading days old"
        )

    tickers = (
        sorted(t for t in data if not t.startswith("^"))
        if requested is None
        else [t.strip().upper() for t in requested if t.strip()]
    )
    rows = []
    diagnostics = []
    for ticker in tickers:
        if ticker not in data:
            raise RSSnapshotError(f"{ticker}: missing from daily PKL")
        score = raw_rs(close_series(data[ticker], ticker), benchmark)
        rating = display_rating(rs_rating(score, anchors))
        rows.append({"Ticker": ticker, "RS": rating, "AnchorDate": anchor.isoformat()})
        diagnostics.append(
            {
                "Ticker": ticker,
                "AnchorDate": anchor.isoformat(),
                "AnchorAge": age,
                "RawRS": score,
                "RS": rating,
            }
        )
    return rows, diagnostics


def run(tickers: Sequence[str] | None = None, pkl_path: Path | None = None) -> Path | None:
    pkl_path = pkl_path or find_latest_daily_pkl()
    if not pkl_path.name.startswith("stock_data_") or not pkl_path.name.endswith("_1d.pkl"):
        raise RSSnapshotError(f"Expected a daily stock_data_*_1d.pkl file: {pkl_path}")
    if not pkl_path.is_file():
        raise RSSnapshotError(f"Daily PKL not found: {pkl_path}")
    data = load_daily_pkl(pkl_path)
    if REFERENCE_TICKER not in data:
        raise RSSnapshotError(f"{REFERENCE_TICKER} missing from {pkl_path}")

    benchmark = close_series(data[REFERENCE_TICKER], REFERENCE_TICKER)
    sessions = session_dates(benchmark)
    publishing = tickers is None
    try:
        anchors, source_date, source = get_rsrating_anchors()
        anchor = normalize_anchor_date(source_date, sessions)
        age = anchor_age(anchor, sessions)
    except Exception:
        if publishing:
            cleanup_stale_published_snapshots(sessions)
        raise

    if age >= MAX_ANCHOR_AGE:
        if publishing:
            remove_old_snapshots(RESULTS_DIR)
        raise StaleAnchorError(
            f"AnchorDate {anchor.isoformat()} is {age} US trading days old; "
            "old RS snapshots removed and no new snapshot published"
        )

    rows, diagnostics = compute_rows(data, anchors, anchor, tickers)
    if tickers is not None:
        print(f"PKL={pkl_path} Anchors={source}")
        print("Ticker\tAnchorDate\tAnchorAge\tRawRS\tRS")
        for item in diagnostics:
            print(
                f"{item['Ticker']}\t{item['AnchorDate']}\t{item['AnchorAge']}\t"
                f"{item['RawRS']:.4f}\t{item['RS']}"
            )
        return None

    target = write_snapshot(rows, date.today())
    print(
        f"[RS] Published {len(rows)} rows to {target} "
        f"(AnchorDate={anchor.isoformat()}, AnchorAge={age}, anchors={source})"
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish Pine-compatible RS from an existing daily PKL; no download is run",
        epilog="Use --pkl results_pkl/stock_data_DDMMYY_1d.pkl to select a completed download.",
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        help="Existing daily PKL; defaults to the newest stock_data_*_1d.pkl in results_pkl/",
    )
    parser.add_argument(
        "--tickers",
        help="Diagnostic only: FORM,NET,P; prints AnchorDate/AnchorAge/RawRS/RS",
    )
    args = parser.parse_args()
    tickers = None
    if args.tickers is not None:
        tickers = [x.strip() for x in args.tickers.split(",") if x.strip()]
        if not tickers:
            parser.error("--tickers requires at least one ticker")
    try:
        run(tickers, pkl_path=args.pkl)
        return 0
    except StaleAnchorError as exc:
        print(f"[RS] stale anchors: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[RS] update failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
