"""Rebuild long-history weekly replay pools from the canonical research bundle.

Each analysis week is independent under the current stateless quant_trade replay
contract. Weeks are executed with ``ProcessPoolExecutor`` so every worker has an
isolated Python module namespace; this is required because ``run_one_week``
temporarily monkey-patches ``weekly_job._daily_map``.

Large daily/weekly bundles are loaded once per worker by the pool initializer.
Tasks send only lightweight week metadata through IPC. Successful on-disk weeks
are reused with ``--no-clean`` when their price-bundle and quant_trade provenance
still matches the current run.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from backtest.latest_quant_trade_replay import SnapshotWeek
from backtest.latest_quant_trade_replay.runner import (
    clean_replay_output_root,
    git_commit,
    load_pickle_data,
    run_one_week,
    sha256_file,
    write_data_source_audit_report,
    write_manifest,
)
from eps_pit.lookup import SignalEPSLookup

from .outcomes import RESEARCH_PRICE_MODE
from .pipeline_contract import replay_dataset_digest

DEFAULT_WARMUP_START = "2022-07-01"
DEFAULT_ANALYSIS_START = "2022-10-01"
DEFAULT_ANALYSIS_END = "2026-03-27"
DEFAULT_DAILY_PKL = Path("backtest/blind_rule_discovery/work/prices/research_daily.pkl")
DEFAULT_WEEKLY_PKL = Path("backtest/blind_rule_discovery/work/prices/research_weekly.pkl")
DEFAULT_PRICE_MANIFEST = Path("backtest/blind_rule_discovery/work/prices/price_manifest.json")
DEFAULT_OUTPUT_ROOT = Path("backtest/blind_rule_discovery/work/replay_pools")
DEFAULT_EPS_PIT_CACHE = Path("backtest/blind_rule_discovery/work/eps_pit_replay.csv")
DEFAULT_BENCHMARK_CODE = "SPY"
DEFAULT_WORKERS = max(1, min(6, max(1, (os.cpu_count() or 2) - 1)))
MIN_ANALYSIS_QUARTERS = 14
MIN_WEEKLY_LOOKBACK_DAYS = 5 * 365
MIN_BUNDLE_COVERAGE = 0.98

_WORKER_DAILY_DATA: dict[str, pd.DataFrame] | None = None
_WORKER_WEEKLY_DATA: dict[str, pd.DataFrame] | None = None
_WORKER_CONFIG: dict[str, Any] | None = None


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
    """Require enough pre-analysis history for the strategy's 5Y weekly context."""
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


def assert_quant_trade_replay_stateless(quant_trade_path: Path) -> dict[str, object]:
    """Verify the checked-out RunContext has no cross-week old-pool state contract.

    Parallel week execution is valid only under this contract. The check is
    static and read-only; quant_trade is never modified.
    """
    run_context_path = quant_trade_path / "strategy" / "run_context.py"
    if not run_context_path.exists():
        raise FileNotFoundError(f"quant_trade RunContext not found: {run_context_path}")
    tree = ast.parse(run_context_path.read_text(encoding="utf-8"), filename=str(run_context_path))
    run_context = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "RunContext"
        ),
        None,
    )
    if run_context is None:
        raise RuntimeError("quant_trade strategy.run_context has no RunContext class")
    replay = next(
        (
            node
            for node in run_context.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "replay"
        ),
        None,
    )
    if replay is None:
        raise RuntimeError("quant_trade RunContext has no replay() constructor")

    arg_names = [
        arg.arg
        for arg in [*replay.args.posonlyargs, *replay.args.args, *replay.args.kwonlyargs]
    ]
    class_fields: set[str] = set()
    for node in run_context.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            class_fields.add(node.target.id)
        elif isinstance(node, ast.Assign):
            class_fields.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
    referenced_attrs = {
        node.attr for node in ast.walk(run_context) if isinstance(node, ast.Attribute)
    }
    forbidden = {"old_pool", "replay_old_pool"}
    if (
        forbidden.intersection(arg_names)
        or forbidden.intersection(class_fields)
        or forbidden.intersection(referenced_attrs)
    ):
        raise RuntimeError(
            "parallel replay requires stateless quant_trade RunContext; "
            f"found cross-week state in replay contract: args={arg_names}, fields={sorted(class_fields)}"
        )
    if replay.args.vararg is not None or replay.args.kwarg is not None:
        raise RuntimeError(
            "parallel replay refuses variadic RunContext.replay(); cross-week state contract is ambiguous"
        )
    return {
        "run_context_path": str(run_context_path),
        "replay_parameters": [name for name in arg_names if name != "cls"],
        "state_mode": "stateless_independent_weeks",
    }


def _quarter_count(rows: list[dict]) -> int:
    return len(
        {
            pd.Timestamp(row["snapshot_date"]).to_period("Q")
            for row in rows
            if row.get("status") == "success"
        }
    )


def _completed_week_metadata(
    output_root: Path,
    week: SnapshotWeek,
    *,
    daily_sha: str,
    weekly_sha: str,
    quant_trade_commit: str,
) -> dict[str, Any] | None:
    """Return a compatible successful checkpoint, otherwise force recomputation."""
    week_dir = output_root / week.snapshot_date
    metadata_path = week_dir / "metadata.json"
    pool_path = week_dir / "breakout_follow_pool.csv"
    if not metadata_path.exists() or not pool_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if metadata.get("status") != "success":
        return None
    if str(metadata.get("snapshot_date")) != week.snapshot_date:
        return None
    if str(metadata.get("expected_last_trading_day")) != week.expected_last_trading_day:
        return None
    if metadata.get("data_source_mode") != "research_full_history_bundle":
        return None
    if str(metadata.get("daily_pkl_sha256") or "") != daily_sha:
        return None
    if str(metadata.get("weekly_pkl_sha256") or "") != weekly_sha:
        return None
    if str(metadata.get("quant_trade_commit") or "") != quant_trade_commit:
        return None
    return metadata


def _worker_failure_row(
    week: SnapshotWeek,
    output_root: Path,
    exc: Exception,
) -> dict[str, Any]:
    week_dir = output_root / week.snapshot_date
    week_dir.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "snapshot_date": week.snapshot_date,
        "expected_last_trading_day": week.expected_last_trading_day,
        "status": "failed_worker_exception",
        "failure_reason": f"{type(exc).__name__}: {exc}",
        "output_pool_path": str(week_dir / "breakout_follow_pool.csv"),
        "output_row_count": 0,
        "data_source_mode": "research_full_history_bundle",
        "replay_used_clipped_data": False,
        "has_future_data_before_clip": False,
        "schema_audit": {"schema_validation_status": "failed_worker_exception"},
    }
    (week_dir / "metadata.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return row


def _init_replay_worker(
    daily_pkl: str,
    weekly_pkl: str,
    config: dict[str, Any],
    eps_seed_path: str,
    worker_cache_dir: str,
) -> None:
    """Load large immutable bundles once into each isolated worker process."""
    global _WORKER_DAILY_DATA, _WORKER_WEEKLY_DATA, _WORKER_CONFIG
    _WORKER_DAILY_DATA = load_pickle_data(Path(daily_pkl))
    _WORKER_WEEKLY_DATA = load_pickle_data(Path(weekly_pkl))
    _WORKER_CONFIG = dict(config)

    cache_dir = Path(worker_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    worker_cache = cache_dir / f"eps_pit_{os.getpid()}.csv"
    seed = Path(eps_seed_path)
    if seed.exists():
        shutil.copy2(seed, worker_cache)
    SignalEPSLookup.DEFAULT_REPLAY_CSV_PATH = str(worker_cache)


def _run_week_worker(task: tuple[str, str]) -> dict[str, Any]:
    """Execute one week using only process-local global state and lightweight IPC."""
    if _WORKER_DAILY_DATA is None or _WORKER_WEEKLY_DATA is None or _WORKER_CONFIG is None:
        raise RuntimeError("replay worker was not initialized")
    snapshot_date, expected_last_trading_day = task
    config = _WORKER_CONFIG
    return run_one_week(
        snapshot_date=snapshot_date,
        expected_last_trading_day=expected_last_trading_day,
        daily_pkl=None,
        weekly_pkl=None,
        daily_data=_WORKER_DAILY_DATA,
        weekly_data=_WORKER_WEEKLY_DATA,
        data_source_mode="research_full_history_bundle",
        historical_pkl_commit=None,
        historical_pkl_commit_date=None,
        historical_pkl_candidate_count=None,
        daily_pkl_file=config["daily_pkl"],
        weekly_pkl_file=config["weekly_pkl"],
        daily_pkl_sha256=config["daily_sha"],
        weekly_pkl_sha256=config["weekly_sha"],
        output_root=Path(config["output_root"]),
        quant_trade_path=Path(config["quant_trade_path"]),
        quant_trade_env=Path(config["quant_trade_env"]) if config["quant_trade_env"] else None,
        yfinance_data_path=Path(config["yfinance_data_path"]),
        quant_trade_commit=config["quant_trade_commit"],
        replay_old_pool=set(),
        replay_old_pool_source="stateless_parallel_independent_week",
    )


def _row_values_signature(frame: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    return frame[value_columns].fillna("<NA>").astype(str).drop_duplicates()


def _merge_worker_eps_caches(worker_cache_dir: Path, eps_pit_cache: Path) -> int:
    """Merge process-local EPS PIT stores after workers exit.

    Every worker begins with the same read-only seed snapshot, so duplicate seed
    rows are expected. For keys absent from the seed, disagreeing worker values
    are a real correctness error. For an existing seed key, a worker refresh is
    allowed and the newest ``retrieved_at`` wins deterministically.
    """
    worker_paths = sorted(worker_cache_dir.glob("*.csv"))
    if not worker_paths:
        return 0

    tagged_frames: list[pd.DataFrame] = []
    base_keys: set[tuple[str, str]] = set()
    if eps_pit_cache.exists():
        base = pd.read_csv(eps_pit_cache)
        if not base.empty:
            base["_origin_rank"] = 0
            base["_source_order"] = 0
            tagged_frames.append(base)
            if {"snapshot_date", "code"}.issubset(base.columns):
                base_keys = {
                    (str(row.snapshot_date)[:10], str(row.code).strip().upper())
                    for row in base[["snapshot_date", "code"]].itertuples(index=False)
                }
    for order, path in enumerate(worker_paths, start=1):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["_origin_rank"] = 1
        frame["_source_order"] = order
        tagged_frames.append(frame)
    if not tagged_frames:
        return len(worker_paths)

    combined = pd.concat(tagged_frames, ignore_index=True, sort=False)
    required = {"snapshot_date", "code"}
    if not required.issubset(combined.columns):
        raise RuntimeError("worker EPS PIT cache missing snapshot_date/code key columns")

    combined["snapshot_date"] = combined["snapshot_date"].astype(str).str[:10]
    combined["code"] = combined["code"].astype(str).str.strip().str.upper()
    helper_columns = {"_origin_rank", "_source_order", "_retrieved_ts"}
    value_columns = [
        column
        for column in combined.columns
        if column not in {"snapshot_date", "code", "retrieved_at", *helper_columns}
    ]

    duplicate = combined.duplicated(["snapshot_date", "code"], keep=False)
    for key, group in combined.loc[duplicate].groupby(
        ["snapshot_date", "code"], dropna=False
    ):
        normalized_key = (str(key[0])[:10], str(key[1]).strip().upper())
        workers_only = group.loc[group["_origin_rank"] == 1]
        if normalized_key not in base_keys and len(_row_values_signature(workers_only, value_columns)) > 1:
            raise RuntimeError(
                f"conflicting worker EPS PIT records for {normalized_key[0]} {normalized_key[1]}"
            )

    if "retrieved_at" in combined.columns:
        combined["_retrieved_ts"] = pd.to_datetime(
            combined["retrieved_at"], errors="coerce", utc=True
        )
    else:
        combined["_retrieved_ts"] = pd.NaT
    combined = combined.sort_values(
        ["snapshot_date", "code", "_retrieved_ts", "_origin_rank", "_source_order"],
        kind="stable",
        na_position="first",
    )
    merged = (
        combined.drop_duplicates(["snapshot_date", "code"], keep="last")
        .sort_values(["snapshot_date", "code"], kind="stable")
        .drop(columns=list(helper_columns), errors="ignore")
        .reset_index(drop=True)
    )
    eps_pit_cache.parent.mkdir(parents=True, exist_ok=True)
    temp_path = eps_pit_cache.with_name(f".{eps_pit_cache.name}.merge.tmp")
    if temp_path.exists():
        temp_path.unlink()
    merged.to_csv(temp_path, index=False)
    temp_path.replace(eps_pit_cache)
    return len(worker_paths)


def _write_research_replay_report(
    output_root: Path,
    *,
    rows: list[dict],
    manifest: dict[str, object],
    history_coverage: dict[str, str],
    benchmark_code: str,
    analysis_start: str,
    analysis_end: str,
    eps_pit_cache: Path,
    workers: int,
    skipped_weeks: int,
    computed_weeks: int,
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
        "- Execution model: `ProcessPoolExecutor` with spawn; no thread pool is used.",
        "- Every analysis week is independent under the verified stateless quant_trade replay contract.",
        "- Large daily/weekly bundles are loaded once per worker; week tasks carry only date metadata.",
        f"- EPS PIT replay cache: `{eps_pit_cache}`; workers write process-local caches which are merged after completion.",
        "- Existing B0 ranks/selections are not inputs to replay generation.",
        "",
        "## Coverage",
        "",
        f"- Analysis window: `{analysis_start}` through `{analysis_end}`.",
        f"- Benchmark: `{benchmark_code}`.",
        f"- Price history: `{history_coverage['first_session']}` through `{history_coverage['last_session']}`.",
        f"- Worker processes: {workers}.",
        f"- Resume checkpoints reused: {skipped_weeks}.",
        f"- Weeks submitted to workers: {computed_weeks}.",
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
    if not failures:
        lines.append("- None.")
    else:
        for row in failures:
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
    workers: int = DEFAULT_WORKERS,
    clean: bool = True,
) -> list[dict]:
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be >= 1")

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
    analysis_weeks = enumerate_snapshot_weeks_from_benchmark(
        daily_data[benchmark_code], start_date=analysis_start, end_date=analysis_end
    )
    if not analysis_weeks:
        raise ValueError("analysis replay window contains no complete weeks")

    quant_trade_path = quant_trade_path.resolve()
    stateless_contract = assert_quant_trade_replay_stateless(quant_trade_path)
    quant_trade_commit = git_commit(quant_trade_path)
    yfinance_data_path = Path.cwd().resolve()
    daily_pkl = daily_pkl.resolve()
    weekly_pkl = weekly_pkl.resolve()
    price_manifest = price_manifest.resolve()
    output_root = output_root.resolve()
    eps_pit_cache = eps_pit_cache.resolve()
    quant_trade_env = quant_trade_env.resolve() if quant_trade_env is not None else None

    if clean:
        clean_replay_output_root(
            output_root,
            reason="canonical blind-discovery stateless parallel replay rebuild",
        )
    else:
        output_root.mkdir(parents=True, exist_ok=True)

    daily_sha = sha256_file(daily_pkl)
    weekly_sha = sha256_file(weekly_pkl)
    rows: list[dict[str, Any]] = []
    pending_weeks: list[SnapshotWeek] = []

    for week in analysis_weeks:
        checkpoint = None
        if not clean:
            checkpoint = _completed_week_metadata(
                output_root,
                week,
                daily_sha=daily_sha,
                weekly_sha=weekly_sha,
                quant_trade_commit=quant_trade_commit,
            )
        if checkpoint is None:
            pending_weeks.append(week)
        else:
            rows.append(checkpoint)

    skipped_weeks = len(rows)
    computed_weeks = len(pending_weeks)
    print(
        f"[blind-replay] analysis_weeks={len(analysis_weeks)} "
        f"resume_skipped={skipped_weeks} submitted={computed_weeks} workers={workers}"
    )

    # Parent validation no longer needs the ~185 MB bundle copies. Drop them
    # before spawned workers each load their own process-local copy.
    del daily_data, weekly_data
    gc.collect()

    worker_cache_dir = output_root / "_worker_eps_cache"
    if worker_cache_dir.exists():
        shutil.rmtree(worker_cache_dir)
    worker_cache_dir.mkdir(parents=True, exist_ok=True)
    eps_pit_cache.parent.mkdir(parents=True, exist_ok=True)
    eps_cache_sha_before = sha256_file(eps_pit_cache) if eps_pit_cache.exists() else None

    if pending_weeks:
        worker_config: dict[str, Any] = {
            "daily_pkl": str(daily_pkl),
            "weekly_pkl": str(weekly_pkl),
            "daily_sha": daily_sha,
            "weekly_sha": weekly_sha,
            "output_root": str(output_root),
            "quant_trade_path": str(quant_trade_path),
            "quant_trade_env": str(quant_trade_env) if quant_trade_env is not None else "",
            "yfinance_data_path": str(yfinance_data_path),
            "quant_trade_commit": quant_trade_commit,
        }
        spawn_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
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
                for week in pending_weeks
            }
            completed = 0
            for future in as_completed(future_to_week):
                week = future_to_week[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = _worker_failure_row(week, output_root, exc)
                rows.append(row)
                completed += 1
                if completed == 1 or completed % 10 == 0 or completed == computed_weeks:
                    print(
                        f"[blind-replay] completed={completed}/{computed_weeks} "
                        f"latest={week.snapshot_date} status={row.get('status')}"
                    )

    worker_cache_count = _merge_worker_eps_caches(worker_cache_dir, eps_pit_cache)
    shutil.rmtree(worker_cache_dir, ignore_errors=True)

    rows.sort(key=lambda row: str(row.get("snapshot_date") or ""))
    write_manifest(output_root, rows)
    if rows:
        write_data_source_audit_report(output_root, rows)

    eps_cache_sha_after = sha256_file(eps_pit_cache) if eps_pit_cache.exists() else None
    replay_digest, replay_pool_files = replay_dataset_digest(output_root)
    failed = [row for row in rows if row.get("status") != "success"]
    quarter_count = _quarter_count(rows)
    preflight = {
        "analysis_start": analysis_start,
        "analysis_end": analysis_end,
        "warmup_start": warmup_start,
        "warmup_mode": "not_executed_stateless_quant_trade",
        "warmup_weeks": 0,
        "warmup_failed_weeks": 0,
        "analysis_weeks_expected": len(analysis_weeks),
        "analysis_weeks_persisted": len(rows),
        "resume_skipped_weeks": skipped_weeks,
        "worker_submitted_weeks": computed_weeks,
        "worker_processes": workers,
        "execution_model": "ProcessPoolExecutor_spawn",
        "thread_pool_used": False,
        "worker_bundle_loading": "initializer_once_per_process",
        "worker_eps_cache_files_merged": worker_cache_count,
        "replay_pool_files": replay_pool_files,
        "replay_dataset_sha256": replay_digest,
        "successful_weeks": len(rows) - len(failed),
        "failed_weeks": len(failed),
        "successful_quarters": quarter_count,
        "minimum_required_quarters": min_analysis_quarters,
        "benchmark_code": benchmark_code,
        "daily_pkl": str(daily_pkl),
        "weekly_pkl": str(weekly_pkl),
        "daily_pkl_sha256": daily_sha,
        "weekly_pkl_sha256": weekly_sha,
        "price_manifest": str(price_manifest),
        "price_manifest_sha256": sha256_file(price_manifest),
        "price_adjustment_mode": RESEARCH_PRICE_MODE,
        "bundle_coverage": float(manifest["coverage"]),
        "history_coverage": history_coverage,
        "weekly_first_session": weekly_first_session,
        "eps_pit_cache": str(eps_pit_cache),
        "eps_pit_cache_sha256_before": eps_cache_sha_before,
        "eps_pit_cache_sha256_after": eps_cache_sha_after,
        "quant_trade_commit": quant_trade_commit,
        "quant_trade_replay_contract": stateless_contract,
        "universe_mode": manifest.get("universe_mode"),
        "universe_limitation": manifest.get("universe_limitation"),
    }
    (output_root / "research_replay_preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
    )
    _write_research_replay_report(
        output_root,
        rows=rows,
        manifest=manifest,
        history_coverage=history_coverage,
        benchmark_code=benchmark_code,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        eps_pit_cache=eps_pit_cache,
        workers=workers,
        skipped_weeks=skipped_weeks,
        computed_weeks=computed_weeks,
    )

    if failed:
        row = failed[0]
        raise RuntimeError(
            f"canonical replay failed at {row.get('snapshot_date')}: "
            f"{row.get('failure_reason') or row.get('status')}"
        )
    if len(rows) != len(analysis_weeks) or replay_pool_files != len(analysis_weeks):
        raise RuntimeError(
            f"canonical replay is incomplete: rows={len(rows)} "
            f"pools={replay_pool_files} expected={len(analysis_weeks)}"
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
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
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
        workers=args.workers,
        clean=not args.no_clean,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "analysis_weeks": len(rows),
                "workers": args.workers,
                "output_root": str(args.output_root),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
