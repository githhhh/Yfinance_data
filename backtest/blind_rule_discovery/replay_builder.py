"""Rebuild long-history weekly replay pools from the canonical research bundle.

Unlike ``latest_quant_trade_replay`` historical-git mode, this runner does not
need a point-in-time pickle commit for every week. It loads one long-history
bundle, validates its provenance, clips it causally at each snapshot, warms
old_pool before the analysis window, and only persists analysis weeks.
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
    write_manifest,
)
from eps_pit.lookup import SignalEPSLookup

from .outcomes import RESEARCH_PRICE_MODE

DEFAULT_WARMUP_START = "2022-07-01"
DEFAULT_ANALYSIS_START = "2022-10-01"
DEFAULT_ANALYSIS_END = "2026-03-27"
DEFAULT_DAILY_PKL = Path("backtest/blind_rule_discovery/work/prices/research_daily.pkl")
DEFAULT_WEEKLY_PKL = Path("backtest/blind_rule_discovery/work/prices/research_weekly.pkl")
DEFAULT_PRICE_MANIFEST = Path("backtest/blind_rule_discovery/work/prices/price_manifest.json")
DEFAULT_OUTPUT_ROOT = Path("backtest/blind_rule_discovery/work/replay_pools")
DEFAULT_EPS_PIT_CACHE = Path("backtest/blind_rule_discovery/work/eps_pit_replay.csv")
DEFAULT_BENCHMARK_CODE = "SPY"
MIN_ANALYSIS_QUARTERS = 14
MIN_WEEKLY_LOOKBACK_DAYS = 5 * 365
MIN_BUNDLE_COVERAGE = 0.98


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
        SnapshotWeek(
            snapshot_date=pd.Timestamp(value).strftime("%Y-%m-%d"),
            expected_last_trading_day=pd.Timestamp(value).strftime("%Y-%m-%d"),
        )
        for value in selected.tolist()
    ]


def validate_price_manifest(
    manifest_path: Path,
    *,
    daily_pkl: Path,
    weekly_pkl: Path,
) -> dict[str, object]:
    """Fail closed if the bundle differs from the manifest that created it."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"research price manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("research price manifest requires schema_version=1")
    if manifest.get("provider") != "Yahoo Finance via yfinance":
        raise ValueError("research price manifest has unexpected provider")
    if not str(manifest.get("yfinance_version") or "").strip():
        raise ValueError("research price manifest must record yfinance_version")
    if manifest.get("price_adjustment_mode") != RESEARCH_PRICE_MODE:
        raise ValueError("research price manifest has unexpected price_adjustment_mode")
    if manifest.get("daily_interval") != "1d":
        raise ValueError("research price manifest must use direct Yahoo 1d data")
    if manifest.get("weekly_interval") != "1wk":
        raise ValueError("research price manifest must use direct Yahoo 1wk data")
    if manifest.get("weekly_source") != "direct_yahoo_1wk_not_daily_resample":
        raise ValueError("research price manifest weekly source is not canonical")
    if manifest.get("auto_adjust") is not False:
        raise ValueError("research price manifest must record auto_adjust=false")
    if manifest.get("repair") is not True:
        raise ValueError("research price manifest must record repair=true")
    if manifest.get("rounding") is not False:
        raise ValueError("research price manifest must record rounding=false")
    try:
        coverage = float(manifest.get("coverage"))
    except (TypeError, ValueError):
        raise ValueError("research price manifest has invalid joint coverage") from None
    if coverage < MIN_BUNDLE_COVERAGE:
        raise ValueError(
            f"research price manifest coverage {coverage:.3f} is below canonical minimum {MIN_BUNDLE_COVERAGE:.3f}"
        )
    if "SPY" not in set(manifest.get("benchmark_codes_downloaded") or []):
        raise ValueError("research price manifest does not contain SPY")
    actual_daily = sha256_file(daily_pkl)
    actual_weekly = sha256_file(weekly_pkl)
    if str(manifest.get("daily_sha256") or "") != actual_daily:
        raise ValueError("research daily pkl SHA256 does not match price_manifest.json")
    if str(manifest.get("weekly_sha256") or "") != actual_weekly:
        raise ValueError("research weekly pkl SHA256 does not match price_manifest.json")
    return manifest


def _assert_research_bundle(
    data: dict[str, pd.DataFrame],
    *,
    benchmark_code: str,
    expected_interval: str | None = None,
) -> None:
    if benchmark_code not in data:
        raise KeyError(f"research bundle missing benchmark {benchmark_code!r}")
    bad_mode = [
        code
        for code, frame in data.items()
        if frame is not None
        and not frame.empty
        and frame.attrs.get("price_adjustment_mode") != RESEARCH_PRICE_MODE
    ]
    if bad_mode:
        raise ValueError(
            f"research bundle contains {len(bad_mode)} frames without verified price mode {RESEARCH_PRICE_MODE}: "
            + ",".join(sorted(bad_mode)[:10])
        )
    if expected_interval is not None:
        bad_interval = [
            code
            for code, frame in data.items()
            if frame is not None
            and not frame.empty
            and frame.attrs.get("interval") != expected_interval
        ]
        if bad_interval:
            raise ValueError(
                f"research bundle contains {len(bad_interval)} frames with wrong interval; "
                f"expected={expected_interval}: " + ",".join(sorted(bad_interval)[:10])
            )


def assert_history_coverage(
    benchmark: pd.DataFrame,
    *,
    warmup_start: str,
    analysis_end: str,
    min_lookback_days: int = MIN_WEEKLY_LOOKBACK_DAYS,
) -> dict[str, str]:
    """Require enough pre-warmup history for the strategy's 5Y weekly context."""
    dates = pd.Series(_normalized_dates(benchmark)).dropna().sort_values()
    if dates.empty:
        raise ValueError("benchmark has no usable history")
    first = pd.Timestamp(dates.iloc[0]).normalize()
    last = pd.Timestamp(dates.iloc[-1]).normalize()
    required_first = pd.Timestamp(warmup_start).normalize() - pd.Timedelta(days=min_lookback_days)
    required_last = pd.Timestamp(analysis_end).normalize()
    if first > required_first:
        raise ValueError(
            f"research bundle starts too late: first={first.date()} required<={required_first.date()}"
        )
    if last < required_last:
        raise ValueError(
            f"research bundle ends too early: last={last.date()} required>={required_last.date()}"
        )
    return {
        "first_session": str(first.date()),
        "last_session": str(last.date()),
        "required_first_session_on_or_before": str(required_first.date()),
        "required_last_session_on_or_after": str(required_last.date()),
    }


def assert_weekly_lookback(
    benchmark: pd.DataFrame,
    *,
    warmup_start: str,
    min_lookback_days: int = MIN_WEEKLY_LOOKBACK_DAYS,
) -> str:
    dates = pd.Series(_normalized_dates(benchmark)).dropna().sort_values()
    if dates.empty:
        raise ValueError("weekly benchmark has no usable history")
    first = pd.Timestamp(dates.iloc[0]).normalize()
    required_first = pd.Timestamp(warmup_start).normalize() - pd.Timedelta(days=min_lookback_days)
    if first > required_first:
        raise ValueError(
            f"weekly research bundle starts too late: first={first.date()} required<={required_first.date()}"
        )
    return str(first.date())


def _quarter_count(rows: list[dict]) -> int:
    return len(
        {
            pd.Timestamp(row["snapshot_date"]).to_period("Q")
            for row in rows
            if row.get("status") == "success"
        }
    )


def _write_research_replay_report(
    output_root: Path,
    *,
    rows: list[dict],
    warmup_weeks: int,
    warmup_failures: list[dict],
    manifest: dict[str, object],
    history_coverage: dict[str, str],
    benchmark_code: str,
    analysis_start: str,
    analysis_end: str,
    eps_pit_cache: Path,
) -> None:
    failures = [row for row in rows if row.get("status") != "success"]
    successful = [row for row in rows if row.get("status") == "success"]
    lines = [
        "# Blind Discovery Long-History Replay Report",
        "",
        "## Contract",
        "",
        "- Data source: one verified long-history Yahoo research bundle, clipped as-of each replay week.",
        f"- Price mode: `{RESEARCH_PRICE_MODE}`.",
        "- Daily/weekly inputs: direct Yahoo `1d` and `1wk`; weekly bars are not re-sampled locally.",
        "- Weekly snapshot calendar: actual SPY daily sessions; no hard-coded holiday table.",
        "- Warmup weeks establish chronological old_pool state and are not persisted into discovery.",
        f"- EPS PIT replay cache: `{eps_pit_cache}` (ignored work area, not tracked `us/signal_eps_pit_replay.csv`).",
        "- Existing B0 ranks/selections are not inputs to replay generation.",
        "",
        "## Coverage",
        "",
        f"- Analysis window: `{analysis_start}` through `{analysis_end}`.",
        f"- Benchmark: `{benchmark_code}`.",
        f"- Price history: `{history_coverage['first_session']}` through `{history_coverage['last_session']}`.",
        f"- Warmup weeks executed: {warmup_weeks}.",
        f"- Warmup failures: {len(warmup_failures)}.",
        f"- Persisted analysis weeks: {len(rows)}.",
        f"- Successful analysis weeks: {len(successful)}.",
        f"- Failed analysis weeks: {len(failures)}.",
        f"- Successful analysis quarters: {_quarter_count(rows)}.",
        f"- Price bundle joint coverage: {manifest.get('coverage')}.",
        f"- Universe limitation: {manifest.get('universe_limitation')}.",
        "",
        "## Failures",
        "",
    ]
    all_failures = [*warmup_failures, *failures]
    if not all_failures:
        lines.append("- None.")
    else:
        for row in all_failures:
            lines.append(
                f"- `{row.get('snapshot_date')}`: `{row.get('status')}` — {row.get('failure_reason') or '-'}"
            )
    (output_root / "research_replay_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_research_replay(
    *,
    daily_pkl: Path,
    weekly_pkl: Path,
    price_manifest: Path,
    output_root: Path,
    quant_trade_path: Path,
    quant_trade_env: Path | None,
    eps_pit_cache: Path = DEFAULT_EPS_PIT_CACHE,
    benchmark_code: str = DEFAULT_BENCHMARK_CODE,
    warmup_start: str = DEFAULT_WARMUP_START,
    analysis_start: str = DEFAULT_ANALYSIS_START,
    analysis_end: str = DEFAULT_ANALYSIS_END,
    min_analysis_quarters: int = MIN_ANALYSIS_QUARTERS,
    min_lookback_days: int = MIN_WEEKLY_LOOKBACK_DAYS,
    clean: bool = True,
) -> list[dict]:
    manifest = validate_price_manifest(
        price_manifest, daily_pkl=daily_pkl, weekly_pkl=weekly_pkl
    )
    daily_data = load_pickle_data(daily_pkl)
    weekly_data = load_pickle_data(weekly_pkl)
    if set(daily_data) != set(weekly_data):
        raise ValueError("research daily/weekly symbol sets differ after bundle materialization")
    _assert_research_bundle(
        daily_data, benchmark_code=benchmark_code, expected_interval="1d"
    )
    _assert_research_bundle(
        weekly_data, benchmark_code=benchmark_code, expected_interval="1wk"
    )
    history_coverage = assert_history_coverage(
        daily_data[benchmark_code],
        warmup_start=warmup_start,
        analysis_end=analysis_end,
        min_lookback_days=min_lookback_days,
    )
    weekly_first_session = assert_weekly_lookback(
        weekly_data[benchmark_code],
        warmup_start=warmup_start,
        min_lookback_days=min_lookback_days,
    )

    all_weeks = enumerate_snapshot_weeks_from_benchmark(
        daily_data[benchmark_code], start_date=warmup_start, end_date=analysis_end
    )
    analysis_start_ts = pd.Timestamp(analysis_start).normalize()
    analysis_weeks = [
        week for week in all_weeks if pd.Timestamp(week.snapshot_date) >= analysis_start_ts
    ]
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
    warmup_failures: list[dict] = []
    warmup_weeks = 0
    replay_old_pool: set[str] = set()
    replay_old_pool_source = "cold_start_warmup"

    eps_pit_cache = eps_pit_cache.resolve()
    eps_pit_cache.parent.mkdir(parents=True, exist_ok=True)
    eps_cache_sha_before = sha256_file(eps_pit_cache) if eps_pit_cache.exists() else None
    old_replay_eps_path = SignalEPSLookup.DEFAULT_REPLAY_CSV_PATH
    SignalEPSLookup.DEFAULT_REPLAY_CSV_PATH = str(eps_pit_cache)
    try:
        with tempfile.TemporaryDirectory(prefix="blind_replay_warmup_") as tmp:
            warmup_root = Path(tmp)
            for week in all_weeks:
                persist = pd.Timestamp(week.snapshot_date) >= analysis_start_ts
                target_root = output_root if persist else warmup_root
                if not persist:
                    warmup_weeks += 1
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
                if row.get("status") != "success":
                    if persist:
                        rows.append(row)
                    else:
                        warmup_failures.append(row)
                    # A failed week breaks chronological old_pool state. Never
                    # continue and pretend later weeks form a valid sequence.
                    break
                replay_old_pool = load_replay_old_pool_from_metadata(row)
                replay_old_pool_source = f"previous_replay_week:{week.snapshot_date}"
                if persist:
                    rows.append(row)
    finally:
        SignalEPSLookup.DEFAULT_REPLAY_CSV_PATH = old_replay_eps_path

    write_manifest(output_root, rows)
    if rows:
        write_data_source_audit_report(output_root, rows)

    eps_cache_sha_after = sha256_file(eps_pit_cache) if eps_pit_cache.exists() else None
    failed = [row for row in rows if row.get("status") != "success"]
    quarter_count = _quarter_count(rows)
    preflight = {
        "analysis_start": analysis_start,
        "analysis_end": analysis_end,
        "warmup_start": warmup_start,
        "warmup_weeks": warmup_weeks,
        "warmup_failed_weeks": len(warmup_failures),
        "analysis_weeks_expected": len(analysis_weeks),
        "analysis_weeks_persisted": len(rows),
        "successful_weeks": len(rows) - len(failed),
        "failed_weeks": len(failed),
        "successful_quarters": quarter_count,
        "minimum_required_quarters": min_analysis_quarters,
        "benchmark_code": benchmark_code,
        "daily_pkl": str(daily_pkl),
        "weekly_pkl": str(weekly_pkl),
        "price_manifest": str(price_manifest),
        "price_manifest_sha256": sha256_file(price_manifest),
        "price_adjustment_mode": RESEARCH_PRICE_MODE,
        "bundle_coverage": float(manifest["coverage"]),
        "history_coverage": history_coverage,
        "weekly_first_session": weekly_first_session,
        "eps_pit_cache": str(eps_pit_cache),
        "eps_pit_cache_sha256_before": eps_cache_sha_before,
        "eps_pit_cache_sha256_after": eps_cache_sha_after,
        "universe_mode": manifest.get("universe_mode"),
        "universe_limitation": manifest.get("universe_limitation"),
    }
    (output_root / "research_replay_preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
    )
    _write_research_replay_report(
        output_root,
        rows=rows,
        warmup_weeks=warmup_weeks,
        warmup_failures=warmup_failures,
        manifest=manifest,
        history_coverage=history_coverage,
        benchmark_code=benchmark_code,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        eps_pit_cache=eps_pit_cache,
    )

    if warmup_failures:
        row = warmup_failures[0]
        raise RuntimeError(
            f"canonical replay warmup failed at {row.get('snapshot_date')}: "
            f"{row.get('failure_reason') or row.get('status')}"
        )
    if failed:
        row = failed[0]
        raise RuntimeError(
            f"canonical replay failed at {row.get('snapshot_date')}: "
            f"{row.get('failure_reason') or row.get('status')}"
        )
    if len(rows) != len(analysis_weeks):
        raise RuntimeError(
            f"canonical replay persisted {len(rows)} of {len(analysis_weeks)} expected analysis weeks"
        )
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
    parser.add_argument("--price-manifest", type=Path, default=DEFAULT_PRICE_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--eps-pit-cache", type=Path, default=DEFAULT_EPS_PIT_CACHE)
    parser.add_argument("--quant-trade-path", type=Path, required=True)
    parser.add_argument("--quant-trade-env", type=Path, default=None)
    parser.add_argument("--benchmark-code", default=DEFAULT_BENCHMARK_CODE)
    parser.add_argument("--warmup-start", default=DEFAULT_WARMUP_START)
    parser.add_argument("--analysis-start", default=DEFAULT_ANALYSIS_START)
    parser.add_argument("--analysis-end", default=DEFAULT_ANALYSIS_END)
    parser.add_argument("--min-analysis-quarters", type=int, default=MIN_ANALYSIS_QUARTERS)
    parser.add_argument("--min-lookback-days", type=int, default=MIN_WEEKLY_LOOKBACK_DAYS)
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
        price_manifest=args.price_manifest,
        output_root=args.output_root,
        quant_trade_path=args.quant_trade_path,
        quant_trade_env=env_path,
        eps_pit_cache=args.eps_pit_cache,
        benchmark_code=args.benchmark_code,
        warmup_start=args.warmup_start,
        analysis_start=args.analysis_start,
        analysis_end=args.analysis_end,
        min_analysis_quarters=args.min_analysis_quarters,
        min_lookback_days=args.min_lookback_days,
        clean=not args.no_clean,
    )
    print(
        json.dumps(
            {"status": "ok", "analysis_weeks": len(rows), "output_root": str(args.output_root)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
