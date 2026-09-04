"""Verify that spawned-process replay is data-equivalent to serial golden pools.

This audit is intentionally separate from the long replay run. It treats
already-computed successful serial replay pools as golden data, recomputes a
small deterministic cross-section with the new ProcessPoolExecutor path, and
compares every pool field keyed by ticker. It never mutates the golden replay
root or the canonical EPS PIT seed cache.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing as mp
from pathlib import Path
import shutil
import zipfile
from typing import Any

import pandas as pd

from backtest.latest_quant_trade_replay import SnapshotWeek
from backtest.latest_quant_trade_replay.runner import git_commit, load_pickle_data, sha256_file

from .replay_builder import (
    DEFAULT_ANALYSIS_END,
    DEFAULT_ANALYSIS_START,
    DEFAULT_BENCHMARK_CODE,
    DEFAULT_DAILY_PKL,
    DEFAULT_EPS_PIT_CACHE,
    DEFAULT_PRICE_MANIFEST,
    DEFAULT_WEEKLY_PKL,
    DEFAULT_WORKERS,
    _assert_research_bundle,
    _completed_week_metadata,
    _init_replay_worker,
    _run_week_worker,
    assert_quant_trade_replay_stateless,
    enumerate_snapshot_weeks_from_benchmark,
    validate_price_manifest,
)

DEFAULT_BASELINE_ROOT = Path("backtest/blind_rule_discovery/work/replay_pools")
DEFAULT_AUDIT_ROOT = Path("backtest/blind_rule_discovery/work/replay_equivalence")
DEFAULT_SAMPLE_WEEKS = 12
DEFAULT_MIN_QUARTERS = 4
NUMERIC_RTOL = 1e-12
NUMERIC_ATOL = 1e-12
MAX_MISMATCH_EXAMPLES = 30


def _signal_count(pool_path: Path) -> int:
    try:
        frame = pd.read_csv(pool_path, usecols=["signal"])
    except Exception:
        return 0
    if frame.empty:
        return 0
    return int(
        frame["signal"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes"})
        .sum()
    )


def select_equivalence_weeks(
    weeks: list[SnapshotWeek],
    *,
    baseline_root: Path,
    sample_weeks: int,
) -> list[SnapshotWeek]:
    """Select temporal anchors plus high-signal weeks, deterministically."""
    if sample_weeks < 1:
        raise ValueError("sample_weeks must be >= 1")
    ordered = sorted(weeks, key=lambda week: week.snapshot_date)
    if len(ordered) < sample_weeks:
        raise RuntimeError(
            f"only {len(ordered)} compatible serial baseline weeks are available; "
            f"requested {sample_weeks}"
        )
    if len(ordered) == sample_weeks:
        return ordered

    spread_target = max(2, sample_weeks // 2) if sample_weeks > 1 else 1
    spread_target = min(spread_target, sample_weeks)
    if spread_target == 1:
        spread_indices = [len(ordered) // 2]
    else:
        spread_indices = [
            round(i * (len(ordered) - 1) / (spread_target - 1))
            for i in range(spread_target)
        ]
    selected: dict[str, SnapshotWeek] = {
        ordered[index].snapshot_date: ordered[index] for index in spread_indices
    }

    ranked = sorted(
        ordered,
        key=lambda week: (
            -_signal_count(
                baseline_root / week.snapshot_date / "breakout_follow_pool.csv"
            ),
            week.snapshot_date,
        ),
    )
    for week in ranked:
        if len(selected) >= sample_weeks:
            break
        selected.setdefault(week.snapshot_date, week)
    return sorted(selected.values(), key=lambda week: week.snapshot_date)


def _read_pool(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig")
    if "code" not in frame.columns:
        raise ValueError(f"replay pool missing code column: {path}")
    codes = frame["code"].fillna("").astype(str).str.strip()
    if codes.eq("").any():
        raise ValueError(f"replay pool contains empty code: {path}")
    if codes.duplicated().any():
        duplicates = sorted(codes.loc[codes.duplicated(keep=False)].unique())
        raise ValueError(
            f"replay pool contains duplicate code keys: {path}: {duplicates[:10]}"
        )
    frame = frame.copy()
    frame["code"] = codes
    return frame.sort_values("code", kind="stable").reset_index(drop=True)


def _nonempty_mask(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def _numeric_pair(left: pd.Series, right: pd.Series) -> tuple[bool, pd.Series, pd.Series]:
    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    left_present = _nonempty_mask(left)
    right_present = _nonempty_mask(right)
    any_present = bool(left_present.any() or right_present.any())
    numeric = (
        any_present
        and bool(left_numeric.loc[left_present].notna().all())
        and bool(right_numeric.loc[right_present].notna().all())
    )
    return numeric, left_numeric, right_numeric


def compare_pool_frames(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    numeric_rtol: float = NUMERIC_RTOL,
    numeric_atol: float = NUMERIC_ATOL,
) -> list[dict[str, Any]]:
    """Return strict field-level mismatches; an empty list means equivalent."""
    mismatches: list[dict[str, Any]] = []
    if list(baseline.columns) != list(candidate.columns):
        mismatches.append(
            {
                "kind": "columns",
                "baseline": list(baseline.columns),
                "candidate": list(candidate.columns),
            }
        )
        return mismatches
    if len(baseline) != len(candidate):
        mismatches.append(
            {"kind": "row_count", "baseline": len(baseline), "candidate": len(candidate)}
        )
        return mismatches
    if baseline["code"].tolist() != candidate["code"].tolist():
        mismatches.append(
            {
                "kind": "code_set",
                "baseline": baseline["code"].tolist(),
                "candidate": candidate["code"].tolist(),
            }
        )
        return mismatches

    for column in baseline.columns:
        left = baseline[column]
        right = candidate[column]
        if column == "code":
            continue
        numeric, left_numeric, right_numeric = _numeric_pair(left, right)
        for index, code in enumerate(baseline["code"]):
            left_missing = pd.isna(left.iloc[index]) or str(left.iloc[index]).strip() == ""
            right_missing = pd.isna(right.iloc[index]) or str(right.iloc[index]).strip() == ""
            if left_missing and right_missing:
                continue
            if left_missing != right_missing:
                equal = False
            elif numeric:
                equal = math.isclose(
                    float(left_numeric.iloc[index]),
                    float(right_numeric.iloc[index]),
                    rel_tol=numeric_rtol,
                    abs_tol=numeric_atol,
                )
            else:
                equal = str(left.iloc[index]) == str(right.iloc[index])
            if not equal:
                mismatches.append(
                    {
                        "kind": "value",
                        "code": str(code),
                        "column": column,
                        "baseline": None if left_missing else str(left.iloc[index]),
                        "candidate": None if right_missing else str(right.iloc[index]),
                    }
                )
                if len(mismatches) >= MAX_MISMATCH_EXAMPLES:
                    return mismatches
    return mismatches


def compare_pool_files(baseline_path: Path, candidate_path: Path) -> list[dict[str, Any]]:
    return compare_pool_frames(_read_pool(baseline_path), _read_pool(candidate_path))


def _safe_sha256(path: Path) -> str | None:
    try:
        return sha256_file(path) if path.exists() else None
    except Exception:
        return None


def _assert_local_eps_assets(repo_root: Path, eps_seed: Path) -> dict[str, Any]:
    """Fail closed when the claimed offline EPS prerequisites are not present."""
    companyfacts = repo_root / "output" / "eps_pit_cache" / "sec" / "companyfacts.zip"
    ticker_map = repo_root / "output" / "eps_pit_cache" / "sec" / "company_tickers.json"
    if not companyfacts.exists() or companyfacts.stat().st_size <= 0:
        raise FileNotFoundError(
            f"offline SEC companyfacts archive missing or empty: {companyfacts}"
        )
    if not zipfile.is_zipfile(companyfacts):
        raise RuntimeError(f"offline SEC companyfacts archive is not a valid ZIP: {companyfacts}")
    if not ticker_map.exists() or ticker_map.stat().st_size <= 0:
        raise FileNotFoundError(
            "offline SEC ticker map missing; replay could otherwise attempt network CIK lookup: "
            f"{ticker_map}"
        )
    try:
        ticker_payload = json.loads(ticker_map.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"offline SEC ticker map is unreadable: {ticker_map}") from exc
    if not isinstance(ticker_payload, dict) or not ticker_payload:
        raise RuntimeError(f"offline SEC ticker map is empty/invalid: {ticker_map}")
    if not eps_seed.exists():
        raise FileNotFoundError(
            f"canonical EPS PIT seed cache missing: {eps_seed}; equivalence audit requires the serial-run seed"
        )
    try:
        seed_header = pd.read_csv(eps_seed, nrows=5)
    except Exception as exc:
        raise RuntimeError(f"canonical EPS PIT seed cache is unreadable: {eps_seed}") from exc
    if not {"snapshot_date", "code"}.issubset(seed_header.columns):
        raise RuntimeError(
            f"canonical EPS PIT seed cache lacks snapshot_date/code keys: {eps_seed}"
        )
    return {
        "companyfacts_zip": str(companyfacts),
        "companyfacts_size": companyfacts.stat().st_size,
        "company_tickers_json": str(ticker_map),
        "company_ticker_entries": len(ticker_payload),
        "eps_seed": str(eps_seed),
        "eps_seed_sha256": sha256_file(eps_seed),
    }


def run_equivalence_audit(
    *,
    baseline_root: Path,
    audit_root: Path,
    daily_pkl: Path,
    weekly_pkl: Path,
    price_manifest: Path,
    eps_pit_cache: Path,
    quant_trade_path: Path,
    quant_trade_env: Path | None,
    benchmark_code: str = DEFAULT_BENCHMARK_CODE,
    analysis_start: str = DEFAULT_ANALYSIS_START,
    analysis_end: str = DEFAULT_ANALYSIS_END,
    sample_weeks: int = DEFAULT_SAMPLE_WEEKS,
    min_quarters: int = DEFAULT_MIN_QUARTERS,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    """Recompute selected serial golden weeks in spawned workers and compare pools."""
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if min_quarters < 1:
        raise ValueError("min_quarters must be >= 1")

    validate_price_manifest(price_manifest, daily_pkl=daily_pkl, weekly_pkl=weekly_pkl)
    daily_data = load_pickle_data(daily_pkl)
    weekly_data = load_pickle_data(weekly_pkl)
    if set(daily_data) != set(weekly_data):
        raise ValueError("research daily/weekly symbol sets differ")
    _assert_research_bundle(daily_data, benchmark_code=benchmark_code, expected_interval="1d")
    _assert_research_bundle(weekly_data, benchmark_code=benchmark_code, expected_interval="1wk")
    expected_weeks = enumerate_snapshot_weeks_from_benchmark(
        daily_data[benchmark_code], start_date=analysis_start, end_date=analysis_end
    )

    repo_root = Path.cwd().resolve()
    quant_trade_path = quant_trade_path.resolve()
    assert_quant_trade_replay_stateless(quant_trade_path)
    quant_trade_commit = git_commit(quant_trade_path)
    daily_pkl = daily_pkl.resolve()
    weekly_pkl = weekly_pkl.resolve()
    baseline_root = baseline_root.resolve()
    audit_root = audit_root.resolve()
    eps_pit_cache = eps_pit_cache.resolve()
    quant_trade_env = quant_trade_env.resolve() if quant_trade_env is not None else None
    daily_sha = sha256_file(daily_pkl)
    weekly_sha = sha256_file(weekly_pkl)
    eps_assets = _assert_local_eps_assets(repo_root, eps_pit_cache)

    compatible: list[SnapshotWeek] = []
    for week in expected_weeks:
        if _completed_week_metadata(
            baseline_root,
            week,
            daily_sha=daily_sha,
            weekly_sha=weekly_sha,
            quant_trade_commit=quant_trade_commit,
        ) is not None:
            compatible.append(week)
    selected = select_equivalence_weeks(
        compatible, baseline_root=baseline_root, sample_weeks=int(sample_weeks)
    )
    selected_quarters = sorted(
        {str(pd.Timestamp(week.snapshot_date).to_period("Q")) for week in selected}
    )
    if len(selected_quarters) < min_quarters:
        raise RuntimeError(
            f"equivalence sample spans only {len(selected_quarters)} quarters; minimum is {min_quarters}"
        )

    if audit_root.exists():
        shutil.rmtree(audit_root)
    parallel_root = audit_root / "parallel"
    worker_cache_dir = audit_root / "worker_eps_cache"
    parallel_root.mkdir(parents=True, exist_ok=True)
    worker_cache_dir.mkdir(parents=True, exist_ok=True)

    worker_config: dict[str, Any] = {
        "daily_pkl": str(daily_pkl),
        "weekly_pkl": str(weekly_pkl),
        "daily_sha": daily_sha,
        "weekly_sha": weekly_sha,
        "output_root": str(parallel_root),
        "quant_trade_path": str(quant_trade_path),
        "quant_trade_env": str(quant_trade_env) if quant_trade_env is not None else "",
        "yfinance_data_path": str(repo_root),
        "quant_trade_commit": quant_trade_commit,
    }

    # Parent no longer needs the large objects after selecting weeks.
    del daily_data, weekly_data

    rows: dict[str, dict[str, Any]] = {}
    pool_level_error: str | None = None
    try:
        spawn_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(workers, len(selected)),
            mp_context=spawn_context,
            initializer=_init_replay_worker,
            initargs=(
                str(daily_pkl),
                str(weekly_pkl),
                worker_config,
                str(eps_pit_cache),
                str(worker_cache_dir),
            ),
        ) as executor:
            future_to_week = {
                executor.submit(
                    _run_week_worker,
                    (week.snapshot_date, week.expected_last_trading_day),
                ): week
                for week in selected
            }
            for future in as_completed(future_to_week):
                week = future_to_week[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "snapshot_date": week.snapshot_date,
                        "status": "failed_worker_exception",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    }
                rows[week.snapshot_date] = row
                print(
                    f"[replay-equivalence] {week.snapshot_date} status={row.get('status')}"
                )
    except Exception as exc:
        pool_level_error = f"{type(exc).__name__}: {exc}"
        for week in selected:
            rows.setdefault(
                week.snapshot_date,
                {
                    "snapshot_date": week.snapshot_date,
                    "status": "failed_process_pool",
                    "failure_reason": pool_level_error,
                },
            )

    week_reports: list[dict[str, Any]] = []
    total_mismatches = 0
    for week in selected:
        row = rows.get(week.snapshot_date) or {}
        baseline_pool = baseline_root / week.snapshot_date / "breakout_follow_pool.csv"
        candidate_pool = parallel_root / week.snapshot_date / "breakout_follow_pool.csv"
        if row.get("status") != "success":
            mismatches = [
                {
                    "kind": "parallel_status",
                    "candidate_status": row.get("status"),
                    "failure_reason": row.get("failure_reason"),
                }
            ]
        else:
            try:
                mismatches = compare_pool_files(baseline_pool, candidate_pool)
            except Exception as exc:
                mismatches = [
                    {
                        "kind": "comparison_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ]
        total_mismatches += len(mismatches)
        week_reports.append(
            {
                "snapshot_date": week.snapshot_date,
                "quarter": str(pd.Timestamp(week.snapshot_date).to_period("Q")),
                "baseline_pool": str(baseline_pool),
                "candidate_pool": str(candidate_pool),
                "baseline_sha256": _safe_sha256(baseline_pool),
                "candidate_sha256": _safe_sha256(candidate_pool),
                "parallel_status": row.get("status"),
                "equivalent": not mismatches,
                "mismatch_count": len(mismatches),
                "mismatch_examples": mismatches[:MAX_MISMATCH_EXAMPLES],
            }
        )

    report = {
        "schema_version": 1,
        "verified": total_mismatches == 0 and pool_level_error is None,
        "comparison_policy": {
            "scope": "all breakout_follow_pool.csv columns",
            "row_key": "code",
            "row_order_ignored": True,
            "numeric_rtol": NUMERIC_RTOL,
            "numeric_atol": NUMERIC_ATOL,
            "non_numeric": "exact string equality with missing==missing",
            "excluded_pool_columns": [],
        },
        "execution_model": "ProcessPoolExecutor_spawn",
        "thread_pool_used": False,
        "workers": min(workers, len(selected)),
        "pool_level_error": pool_level_error,
        "baseline_root": str(baseline_root),
        "parallel_root": str(parallel_root),
        "daily_pkl_sha256": daily_sha,
        "weekly_pkl_sha256": weekly_sha,
        "quant_trade_commit": quant_trade_commit,
        "eps_offline_assets": eps_assets,
        "compatible_serial_weeks": len(compatible),
        "sample_weeks": len(selected),
        "sample_quarters": selected_quarters,
        "selection": "temporal anchors plus highest-signal compatible weeks",
        "total_mismatch_examples": total_mismatches,
        "weeks": week_reports,
    }
    audit_root.mkdir(parents=True, exist_ok=True)
    report_path = audit_root / "parallel_equivalence_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    shutil.rmtree(worker_cache_dir, ignore_errors=True)

    if not report["verified"]:
        failed_weeks = [item["snapshot_date"] for item in week_reports if not item["equivalent"]]
        raise RuntimeError(
            "parallel replay is NOT equivalent to the serial golden pools; "
            f"failed weeks={failed_weeks}; see {report_path}"
        )
    print(
        f"[replay-equivalence] PASS weeks={len(selected)} quarters={len(selected_quarters)} "
        f"report={report_path}"
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--daily-pkl", type=Path, default=DEFAULT_DAILY_PKL)
    parser.add_argument("--weekly-pkl", type=Path, default=DEFAULT_WEEKLY_PKL)
    parser.add_argument("--price-manifest", type=Path, default=DEFAULT_PRICE_MANIFEST)
    parser.add_argument("--eps-pit-cache", type=Path, default=DEFAULT_EPS_PIT_CACHE)
    parser.add_argument("--quant-trade-path", type=Path, required=True)
    parser.add_argument("--quant-trade-env", type=Path, default=None)
    parser.add_argument("--benchmark-code", default=DEFAULT_BENCHMARK_CODE)
    parser.add_argument("--analysis-start", default=DEFAULT_ANALYSIS_START)
    parser.add_argument("--analysis-end", default=DEFAULT_ANALYSIS_END)
    parser.add_argument("--sample-weeks", type=int, default=DEFAULT_SAMPLE_WEEKS)
    parser.add_argument("--min-quarters", type=int, default=DEFAULT_MIN_QUARTERS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    env_path = args.quant_trade_env
    if env_path is None:
        candidate = args.quant_trade_path / ".env"
        env_path = candidate if candidate.exists() else None
    run_equivalence_audit(
        baseline_root=args.baseline_root,
        audit_root=args.audit_root,
        daily_pkl=args.daily_pkl,
        weekly_pkl=args.weekly_pkl,
        price_manifest=args.price_manifest,
        eps_pit_cache=args.eps_pit_cache,
        quant_trade_path=args.quant_trade_path,
        quant_trade_env=env_path,
        benchmark_code=args.benchmark_code,
        analysis_start=args.analysis_start,
        analysis_end=args.analysis_end,
        sample_weeks=args.sample_weeks,
        min_quarters=args.min_quarters,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
