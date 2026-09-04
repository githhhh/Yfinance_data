"""Rebuild long-history weekly replay pools from the canonical research bundle.

Unlike ``latest_quant_trade_replay`` historical-git mode, this runner does not
need a point-in-time pickle commit for every week.  It loads one long-history
bundle, clips it causally at each snapshot, warms old_pool before the analysis
window, and only persists analysis weeks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import pandas as pd

from backtest.latest_quant_trade_replay import SnapshotWeek
from backtest.latest_quant_trade_replay.runner import (
    clean_replay_output_root,
    git_commit,
    load_pickle_data,
    load_replay_old_pool_from_metadata,
    run_one_week,
    sha256_file,
    write_data_source_audit_report,
    write_execution_log,
    write_manifest,
    write_report,
)

from .research_data import RESEARCH_PRICE_MODE

DEFAULT_WARMUP_START = "2022-07-01"
DEFAULT_ANALYSIS_START = "2022-10-01"
DEFAULT_ANALYSIS_END = "2026-03-27"
DEFAULT_DAILY_PKL = Path("backtest/blind_rule_discovery/work/prices/research_daily.pkl")
DEFAULT_WEEKLY_PKL = Path("backtest/blind_rule_discovery/work/prices/research_weekly.pkl")
DEFAULT_OUTPUT_ROOT = Path("backtest/blind_rule_discovery/work/replay_pools")
DEFAULT_BENCHMARK_CODE = "SPY"
MIN_ANALYSIS_QUARTERS = 14


def _normalized_dates(frame: pd.DataFrame) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"))
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    return idx.normalize()


def enumerate_snapshot_weeks_from_benchmark(
    benchmark: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
) -> list[SnapshotWeek]:
    """Use actual benchmark sessions, so holidays never need a hard-coded table."""
    if benchmark is None or benchmark.empty:
        raise ValueError("benchmark price history is empty")
    dates = pd.Series(_normalized_dates(benchmark)).dropna().drop_duplicates().sort_values()
    if dates.empty:
        raise ValueError("benchmark contains no valid session dates")
    frame = pd.DataFrame({"date": dates})
    frame["week"] = frame["date"].dt.to_period("W-FRI")
    last_sessions = frame.groupby("week", sort=True)["date"].max()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    selected = last_sessions.loc[(last_sessions >= start) & (last_sessions <= end)]
    return [
        SnapshotWeek(snapshot_date=pd.Timestamp(value).strftime("%Y-%m-%d"), expected_last_trading_day=pd.Timestamp(value).strftime("%Y-%m-%d"))
        for value in selected.tolist()
    ]


def _assert_research_bundle(data: dict[str, pd.DataFrame], *, benchmark_code: str) -> None:
    if benchmark_code not in data:
        raise KeyError(f"research bundle missing benchmark {benchmark_code!r}")
    bad = [
        code for code, frame in data.items()
        if frame is not None and not frame.empty and frame.attrs.get("price_adjustment_mode") != RESEARCH_PRICE_MODE
    ]
    if bad:
        raise ValueError(
            f"research bundle contains {len(bad)} frames without verified price mode {RESEARCH_PRICE_MODE}: "
            + ",".join(sorted(bad)[:10])
        )


def _quarter_count(rows: list[dict]) -> int:
    return len({pd.Timestamp(row["snapshot_date"]).to_period("Q") for row in rows if row.get("status") == "success"})


def run_research_replay(
    *,
    daily_pkl: Path,
    weekly_pkl: Path,
    output_root: Path,
    quant_trade_path: Path,
    quant_trade_env: Path | None,
    benchmark_code: str = DEFAULT_BENCHMARK_CODE,
    warmup_start: str = DEFAULT_WARMUP_START,
    analysis_start: str = DEFAULT_ANALYSIS_START,
    analysis_end: str = DEFAULT_ANALYSIS_END,
    min_analysis_quarters: int = MIN_ANALYSIS_QUARTERS,
    clean: bool = True,
) -> list[dict]:
    daily_data = load_pickle_data(daily_pkl)
    weekly_data = load_pickle_data(weekly_pkl)
    _assert_research_bundle(daily_data, benchmark_code=benchmark_code)
    _assert_research_bundle(weekly_data, benchmark_code=benchmark_code)
    if benchmark_code not in weekly_data:
        raise KeyError(f"weekly research bundle missing benchmark {benchmark_code!r}")

    all_weeks = enumerate_snapshot_weeks_from_benchmark(
        daily_data[benchmark_code], start_date=warmup_start, end_date=analysis_end
    )
    analysis_start_ts = pd.Timestamp(analysis_start).normalize()
    analysis_weeks = [week for week in all_weeks if pd.Timestamp(week.snapshot_date) >= analysis_start_ts]
    if not analysis_weeks:
        raise ValueError("analysis replay window contains no complete weeks")

    if clean:
        clean_replay_output_root(
            output_root,
            reason="canonical blind-discovery long-history replay rebuild",
        )
    else:
        output_root.mkdir(parents=True, exist_ok=True)

    quant_trade_commit = git_commit(quant_trade_path)
    yfinance_data_path = Path.cwd()
    daily_sha = sha256_file(daily_pkl)
    weekly_sha = sha256_file(weekly_pkl)
    rows: list[dict] = []
    replay_old_pool: set[str] = set()
    replay_old_pool_source = "cold_start_warmup"

    with tempfile.TemporaryDirectory(prefix="blind_replay_warmup_") as tmp:
        warmup_root = Path(tmp)
        for week in all_weeks:
            persist = pd.Timestamp(week.snapshot_date) >= analysis_start_ts
            target_root = output_root if persist else warmup_root
            row = run_one_week(
                snapshot_date=week.snapshot_date,
                expected_last_trading_day=week.expected_last_trading_day,
                daily_pkl=None,
                weekly_pkl=None,
                daily_data=daily_data,
                weekly_data=weekly_data,
                data_source_mode="research_full_history_bundle",
                historical_pkl_commit=None,
                historical_pkl_commit_date=None,
                historical_pkl_candidate_count=None,
                daily_pkl_file=str(daily_pkl),
                weekly_pkl_file=str(weekly_pkl),
                daily_pkl_sha256=daily_sha,
                weekly_pkl_sha256=weekly_sha,
                output_root=target_root,
                quant_trade_path=quant_trade_path,
                quant_trade_env=quant_trade_env,
                yfinance_data_path=yfinance_data_path,
                quant_trade_commit=quant_trade_commit,
                replay_old_pool=replay_old_pool,
                replay_old_pool_source=replay_old_pool_source,
            )
            replay_old_pool = load_replay_old_pool_from_metadata(row)
            replay_old_pool_source = f"previous_replay_week:{week.snapshot_date}"
            if persist:
                rows.append(row)

    write_manifest(output_root, rows)
    write_report(output_root, rows)
    write_data_source_audit_report(output_root, rows)
    write_execution_log(output_root, rows, quant_trade_path=quant_trade_path)

    failed = [row for row in rows if row.get("status") != "success"]
    quarter_count = _quarter_count(rows)
    preflight = {
        "analysis_start": analysis_start,
        "analysis_end": analysis_end,
        "warmup_start": warmup_start,
        "analysis_weeks": len(rows),
        "successful_weeks": len(rows) - len(failed),
        "failed_weeks": len(failed),
        "successful_quarters": quarter_count,
        "minimum_required_quarters": min_analysis_quarters,
        "benchmark_code": benchmark_code,
        "daily_pkl": str(daily_pkl),
        "weekly_pkl": str(weekly_pkl),
        "price_adjustment_mode": RESEARCH_PRICE_MODE,
    }
    (output_root / "research_replay_preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
    )
    if failed:
        sample = ",".join(str(row.get("snapshot_date")) for row in failed[:8])
        raise RuntimeError(f"canonical replay has {len(failed)} failed weeks: {sample}")
    if quarter_count < min_analysis_quarters:
        raise RuntimeError(
            f"canonical replay has only {quarter_count} successful quarters; "
            f"requires at least {min_analysis_quarters}"
        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-pkl", type=Path, default=DEFAULT_DAILY_PKL)
    parser.add_argument("--weekly-pkl", type=Path, default=DEFAULT_WEEKLY_PKL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--quant-trade-path", type=Path, required=True)
    parser.add_argument("--quant-trade-env", type=Path, default=None)
    parser.add_argument("--benchmark-code", default=DEFAULT_BENCHMARK_CODE)
    parser.add_argument("--warmup-start", default=DEFAULT_WARMUP_START)
    parser.add_argument("--analysis-start", default=DEFAULT_ANALYSIS_START)
    parser.add_argument("--analysis-end", default=DEFAULT_ANALYSIS_END)
    parser.add_argument("--min-analysis-quarters", type=int, default=MIN_ANALYSIS_QUARTERS)
    parser.add_argument("--no-clean", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    env_path = args.quant_trade_env
    if env_path is None:
        candidate = args.quant_trade_path / ".env"
        env_path = candidate if candidate.exists() else None
    rows = run_research_replay(
        daily_pkl=args.daily_pkl,
        weekly_pkl=args.weekly_pkl,
        output_root=args.output_root,
        quant_trade_path=args.quant_trade_path,
        quant_trade_env=env_path,
        benchmark_code=args.benchmark_code,
        warmup_start=args.warmup_start,
        analysis_start=args.analysis_start,
        analysis_end=args.analysis_end,
        min_analysis_quarters=args.min_analysis_quarters,
        clean=not args.no_clean,
    )
    print(json.dumps({"status": "ok", "analysis_weeks": len(rows), "output_root": str(args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
