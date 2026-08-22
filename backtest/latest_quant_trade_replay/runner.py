"""CLI for clean complete-week pool replay using quant_trade dev logic."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from . import (
    EXPECTED_POOL_FIELDS,
    HistoricalPklCandidate,
    ReplayPoolSink,
    apply_replay_strategy_env,
    audit_pool_null_semantics,
    audit_pool_schema,
    clip_price_data_asof,
    clear_snapshot_contaminated_eps,
    enrich_pool_with_asof_52w_high,
    enumerate_complete_snapshot_weeks,
    max_price_date,
    normalize_empty_pool_schema,
    repair_research_fields,
    select_historical_pkl_pair,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_pickle_data(data: Any) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for code, value in data.items():
        if isinstance(value, dict) and {"index", "columns", "data"}.issubset(value.keys()):
            value = pd.DataFrame(index=value["index"], columns=value["columns"], data=value["data"])
        out[str(code)] = value
    return out


def load_pickle_data(path: Path) -> dict[str, pd.DataFrame]:
    with path.open("rb") as f:
        return normalize_pickle_data(pickle.load(f))


def git_blob_bytes(repo_path: Path, commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=repo_path)


def load_git_pickle_data(repo_path: Path, commit: str, path: str) -> dict[str, pd.DataFrame]:
    return normalize_pickle_data(pickle.loads(git_blob_bytes(repo_path, commit, path)))


def git_blob_sha256(repo_path: Path, commit: str, path: str) -> str:
    return hashlib.sha256(git_blob_bytes(repo_path, commit, path)).hexdigest()


def _normalized_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(df.index, errors="coerce"))
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    return idx.normalize()


def _daily_map_from_price_data(
    data: dict[str, pd.DataFrame],
    codes: set[str],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    required = {"Open", "High", "Low", "Close"}
    for code in codes:
        df = data.get(code)
        if df is None or df.empty or not required.issubset(df.columns):
            continue
        cur = df.loc[:, ["Open", "High", "Low", "Close"]].copy()
        cur["Volume"] = df["Volume"] if "Volume" in df.columns else pd.NA
        cur["_date"] = _normalized_index(df)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            cur[col] = pd.to_numeric(cur[col], errors="coerce")
        out[code] = cur.dropna(subset=["_date", "High", "Low"]).sort_values("_date").reset_index(drop=True)
    return out


def git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _relative_removed_paths(root: Path) -> list[str]:
    if not root.exists():
        return []
    paths: list[str] = []
    for path in sorted(root.rglob("*")):
        paths.append(path.relative_to(root).as_posix())
    return paths


def clean_replay_output_root(output_root: Path, *, reason: str) -> dict[str, Any]:
    resolved = output_root.resolve()
    cwd = Path.cwd().resolve()
    if resolved in {cwd, cwd.parent, Path("/").resolve()}:
        raise ValueError(f"Refusing to clean unsafe replay output root: {output_root}")
    removed_paths = _relative_removed_paths(output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log = {
        "reason": reason,
        "output_root": str(output_root),
        "removed_path_count": len(removed_paths),
        "removed_paths": removed_paths,
    }
    (output_root / "clean_rebuild_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Replay Pool Clean Rebuild Log",
        "",
        f"- Reason: {reason}",
        f"- Output root: `{output_root}`",
        f"- Removed path count: {len(removed_paths)}",
        "",
        "## Removed Paths",
        "",
    ]
    lines.extend(f"- `{path}`" for path in removed_paths)
    (output_root / "clean_rebuild_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log


def load_replay_old_pool_from_metadata(metadata: dict[str, Any]) -> set[str]:
    if metadata.get("status") != "success":
        return set()
    pool_path = Path(str(metadata.get("output_pool_path") or ""))
    if not pool_path.exists():
        return set()
    pool = pd.read_csv(pool_path, dtype={"code": str}, encoding="utf-8-sig")
    if "code" not in pool.columns:
        return set()
    return set(pool["code"].dropna().astype(str).str.strip())


PKL_NAME_RE = re.compile(r"stock_data_(\d{6})_(1d|1wk)\.pkl$")


def _history_commit_rows(repo_path: Path, start_date: str, end_date: str) -> list[tuple[str, str]]:
    output = subprocess.check_output(
        [
            "git",
            "log",
            "--all",
            "--reverse",
            "--date=iso-strict",
            f"--since={start_date}T00:00:00+00:00",
            f"--until={end_date}T23:59:59+00:00",
            "--pretty=format:%H%x00%cI",
            "--",
            "results_pkl",
        ],
        cwd=repo_path,
        text=True,
    )
    rows: list[tuple[str, str]] = []
    for line in output.splitlines():
        if "\x00" not in line:
            continue
        commit, commit_date = line.split("\x00", 1)
        rows.append((commit, commit_date))
    return rows


def _pkl_tag_date(tag: str) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(pd.to_datetime(tag, format="%d%m%y")).normalize()
    except Exception:
        return None


def _pkl_paths_at_commit(repo_path: Path, commit: str) -> list[tuple[str, str, str]]:
    output = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit, "results_pkl"],
        cwd=repo_path,
        text=True,
    )
    paths: list[tuple[str, str, str]] = []
    for path in output.splitlines():
        match = PKL_NAME_RE.search(path)
        if not match:
            continue
        tag, period = match.groups()
        paths.append((tag, period, path))
    return sorted(paths)


def _candidate_paths_for_window(
    paths: list[tuple[str, str, str]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[list[str], list[str]]:
    daily_paths: list[str] = []
    weekly_paths: list[str] = []
    for tag, period, path in paths:
        tag_date = _pkl_tag_date(tag)
        if tag_date is None or tag_date < start or tag_date > end:
            continue
        if period == "1d":
            daily_paths.append(path)
        elif period == "1wk":
            weekly_paths.append(path)
    return daily_paths, weekly_paths


def discover_historical_pkl_pair(
    *,
    repo_path: Path,
    snapshot_date: str,
    expected_last_trading_day: str,
    search_days: int = 7,
) -> tuple[HistoricalPklCandidate | None, list[HistoricalPklCandidate]]:
    start = pd.Timestamp(expected_last_trading_day).normalize()
    end = start + pd.Timedelta(days=search_days)
    candidates: list[HistoricalPklCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for commit, commit_date in _history_commit_rows(
        repo_path,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    ):
        daily_paths, weekly_paths = _candidate_paths_for_window(_pkl_paths_at_commit(repo_path, commit), start, end)
        daily_by_path: dict[str, str | None] = {}
        weekly_by_path: dict[str, str | None] = {}
        for daily_path in daily_paths:
            try:
                daily_data = load_git_pickle_data(repo_path, commit, daily_path)
            except Exception:
                continue
            daily_by_path[daily_path] = max_price_date(daily_data)
        for weekly_path in weekly_paths:
            try:
                weekly_data = load_git_pickle_data(repo_path, commit, weekly_path)
            except Exception:
                continue
            weekly_by_path[weekly_path] = max_price_date(weekly_data)
        for daily_path, daily_max_date in daily_by_path.items():
            for weekly_path, weekly_max_date in weekly_by_path.items():
                key = (commit, daily_path, weekly_path)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    HistoricalPklCandidate(
                        commit=commit,
                        commit_date=commit_date,
                        daily_path=daily_path,
                        weekly_path=weekly_path,
                        daily_max_date=daily_max_date,
                        weekly_max_date=weekly_max_date,
                    )
                )
    return (
        select_historical_pkl_pair(
            snapshot_date=snapshot_date,
            expected_last_trading_day=expected_last_trading_day,
            candidates=candidates,
        ),
        candidates,
    )


def run_one_week(
    *,
    snapshot_date: str,
    expected_last_trading_day: str,
    daily_pkl: Path | None,
    weekly_pkl: Path | None,
    output_root: Path,
    quant_trade_path: Path,
    quant_trade_env: Path | None,
    yfinance_data_path: Path,
    quant_trade_commit: str,
    daily_data: dict[str, pd.DataFrame] | None = None,
    weekly_data: dict[str, pd.DataFrame] | None = None,
    data_source_mode: str = "current_files",
    historical_pkl_commit: str | None = None,
    historical_pkl_commit_date: str | None = None,
    historical_pkl_candidate_count: int | None = None,
    daily_pkl_file: str | None = None,
    weekly_pkl_file: str | None = None,
    daily_pkl_sha256: str | None = None,
    weekly_pkl_sha256: str | None = None,
    replay_old_pool: set[str] | None = None,
    replay_old_pool_source: str = "cold_start",
) -> dict[str, Any]:
    if daily_data is None:
        if daily_pkl is None:
            raise ValueError("daily_pkl is required when daily_data is not provided")
        daily_raw = load_pickle_data(daily_pkl)
    else:
        daily_raw = daily_data
    if weekly_data is None:
        if weekly_pkl is None:
            raise ValueError("weekly_pkl is required when weekly_data is not provided")
        weekly_raw = load_pickle_data(weekly_pkl)
    else:
        weekly_raw = weekly_data
    daily_clip = clip_price_data_asof(daily_raw, expected_last_trading_day)
    weekly_clip = clip_price_data_asof(weekly_raw, expected_last_trading_day)

    week_dir = output_root / snapshot_date
    pool_path = week_dir / "breakout_follow_pool.csv"
    sink = ReplayPoolSink(pool_path)
    run_output_tmp = tempfile.TemporaryDirectory(prefix=f"quant_trade_replay_{snapshot_date}_")
    run_output_dir = Path(run_output_tmp.name)

    old_sys_path = list(sys.path)
    old_env = os.environ.copy()
    status = "success"
    failure_reason = ""
    error_counts: dict[str, Any] = {}
    output_fields: list[str] = []
    output_row_count = 0
    schema_audit: dict[str, Any] = {}

    try:
        sys.path.insert(0, str(quant_trade_path))
        applied_env = apply_replay_strategy_env(quant_trade_env)
        import yfinance_data as qtd_yfinance_data

        qtd_yfinance_data.DATA_ROOT = str(yfinance_data_path)
        qtd_yfinance_data.RESULTS_PKL_DIR = str(yfinance_data_path / "results_pkl")
        qtd_yfinance_data.EPS_SCREENER_PATH = str(yfinance_data_path / "us" / "eps_growth_screener_results.csv")
        qtd_yfinance_data.FIFTY_TWO_WK_HIGH_SCREENER_PATH = str(yfinance_data_path / "us" / "52wk_new_high_results.csv")
        qtd_yfinance_data.WEEKLY_VOL_SCREENER_PATH = str(yfinance_data_path / "us" / "weekly_vol_screener_results.csv")
        qtd_yfinance_data.BREAKOUT_FOLLOW_POOL_PATH = str(yfinance_data_path / "us" / "breakout_follow_pool.csv")
        qtd_yfinance_data.BREAKOUT_FOLLOW_POOL_MIDWEEK_PATH = str(
            yfinance_data_path / "us" / "breakout_follow_pool_midweek.csv"
        )
        qtd_yfinance_data.STAGE2_WHITELIST_PATH = str(
            yfinance_data_path / "us" / "stage2" / "stage2_whitelist.csv"
        )

        from data.stock_data import StockPeriod
        from strategy.executor import core_run
        from strategy.run_context import RunContext
        from strategy_analysis.breakout_follow import weekly_job

        ctx = RunContext.replay(snapshot_date, old_pool=set(replay_old_pool or set()))
        ctx.output_dir = str(run_output_dir)
        ctx.enable_futu = False
        ctx.enable_telegram = False
        ctx.enable_pool_save = False
        results = core_run(
            ctx,
            period_data={
                StockPeriod.DAILY: daily_clip.data,
                StockPeriod.WEEKLY: weekly_clip.data,
            }
        )
        error_counts = {str(k): v for k, v in results.get("error_counts", {}).items()}
        preview = run_output_dir / "breakout_follow_signal_weekly.csv"
        pool = pd.read_csv(preview, dtype={"code": str}, encoding="utf-8-sig") if preview.exists() else pd.DataFrame()
        sink.save_snapshot(pool)

        if preview.exists() and not pool.empty:
            original_daily_map = weekly_job._daily_map
            weekly_job._daily_map = lambda codes: _daily_map_from_price_data(daily_clip.data, set(codes))
            try:
                weekly_job._enrich_current_outputs_from_signal_csv(
                    str(preview),
                    pd.Timestamp(snapshot_date).normalize(),
                    str(pool_path),
                )
            finally:
                weekly_job._daily_map = original_daily_map
            pool = pd.read_csv(pool_path, dtype={"code": str}, encoding="utf-8-sig")

        pool = repair_research_fields(pool)
        pool = normalize_empty_pool_schema(pool)
        pool = clear_snapshot_contaminated_eps(pool)
        pool = enrich_pool_with_asof_52w_high(pool, daily_clip.data, expected_last_trading_day)
        pool.to_csv(pool_path, index=False, encoding="utf-8-sig")
        audit = audit_pool_schema(pool)
        schema_audit = audit.to_dict()
        output_fields = list(pool.columns)
        output_row_count = int(len(pool))
        if audit.schema_validation_status == "failed_critical_schema":
            status = "failed_critical_schema"
            failure_reason = ",".join(audit.missing_critical_fields)
    except Exception as exc:
        status = "failed"
        failure_reason = f"{type(exc).__name__}: {exc}"
    finally:
        sys.path[:] = old_sys_path
        os.environ.clear()
        os.environ.update(old_env)
        run_output_tmp.cleanup()

    metadata = {
        "snapshot_date": snapshot_date,
        "expected_last_trading_day": expected_last_trading_day,
        "quant_trade_commit": quant_trade_commit,
        "Yfinance_data_commit": git_commit(yfinance_data_path),
        "data_source_mode": data_source_mode,
        "historical_pkl_commit": historical_pkl_commit,
        "historical_pkl_commit_date": historical_pkl_commit_date,
        "historical_pkl_candidate_count": historical_pkl_candidate_count,
        "replay_old_pool_source": replay_old_pool_source,
        "replay_old_pool_count": len(replay_old_pool or set()),
        "replay_old_pool_codes_sample": sorted(replay_old_pool or set())[:25],
        "replay_new_pool_count": output_row_count if status == "success" else 0,
        "daily_pkl_file": daily_pkl_file or str(daily_pkl),
        "weekly_pkl_file": weekly_pkl_file or str(weekly_pkl),
        "daily_pkl_sha256": daily_pkl_sha256 or (sha256_file(daily_pkl) if daily_pkl is not None else ""),
        "weekly_pkl_sha256": weekly_pkl_sha256 or (sha256_file(weekly_pkl) if weekly_pkl is not None else ""),
        "daily_max_date_before_clip": daily_clip.max_date_before_clip,
        "weekly_max_date_before_clip": weekly_clip.max_date_before_clip,
        "daily_max_date_after_clip": daily_clip.max_date_after_clip,
        "weekly_max_date_after_clip": weekly_clip.max_date_after_clip,
        "has_future_data_before_clip": bool(
            daily_clip.has_future_data_before_clip or weekly_clip.has_future_data_before_clip
        ),
        "replay_used_clipped_data": bool(
            daily_clip.replay_used_clipped_data or weekly_clip.replay_used_clipped_data
        ),
        "output_pool_path": str(pool_path),
        "output_row_count": output_row_count,
        "output_fields": output_fields,
        "schema_validation_status": schema_audit.get("schema_validation_status"),
        "missing_critical_fields": schema_audit.get("missing_critical_fields", []),
        "missing_repairable_fields": schema_audit.get("missing_repairable_fields", []),
        "missing_optional_fields": schema_audit.get("missing_optional_fields", []),
        "schema_audit": schema_audit,
        "error_counts": error_counts,
        "status": status,
        "failure_reason": failure_reason,
        "side_effects_disabled": {
            "futu": True,
            "telegram": True,
            "database": True,
            "production_pool_write": True,
            "pool_publish": True,
            "pool_commit": True,
        },
        "replay_strategy_env_keys": sorted(applied_env.keys()) if "applied_env" in locals() else [],
    }
    week_dir.mkdir(parents=True, exist_ok=True)
    (week_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return metadata


def metadata_for_missing_historical_pkl(
    *,
    snapshot_date: str,
    expected_last_trading_day: str,
    output_root: Path,
    yfinance_data_path: Path,
    quant_trade_commit: str,
    candidate_count: int,
    replay_old_pool: set[str] | None = None,
    replay_old_pool_source: str = "cold_start",
) -> dict[str, Any]:
    week_dir = output_root / snapshot_date
    week_dir.mkdir(parents=True, exist_ok=True)
    pool_path = week_dir / "breakout_follow_pool.csv"
    metadata = {
        "snapshot_date": snapshot_date,
        "expected_last_trading_day": expected_last_trading_day,
        "quant_trade_commit": quant_trade_commit,
        "Yfinance_data_commit": git_commit(yfinance_data_path),
        "data_source_mode": "historical_git",
        "historical_pkl_commit": None,
        "historical_pkl_commit_date": None,
        "historical_pkl_candidate_count": candidate_count,
        "replay_old_pool_source": replay_old_pool_source,
        "replay_old_pool_count": len(replay_old_pool or set()),
        "replay_old_pool_codes_sample": sorted(replay_old_pool or set())[:25],
        "replay_new_pool_count": 0,
        "daily_pkl_file": "",
        "weekly_pkl_file": "",
        "daily_pkl_sha256": "",
        "weekly_pkl_sha256": "",
        "daily_max_date_before_clip": None,
        "weekly_max_date_before_clip": None,
        "daily_max_date_after_clip": None,
        "weekly_max_date_after_clip": None,
        "has_future_data_before_clip": False,
        "replay_used_clipped_data": False,
        "output_pool_path": str(pool_path),
        "output_row_count": 0,
        "output_fields": [],
        "schema_validation_status": "failed_missing_historical_pkl",
        "missing_critical_fields": [],
        "missing_repairable_fields": [],
        "missing_optional_fields": [],
        "schema_audit": {"schema_validation_status": "failed_missing_historical_pkl"},
        "error_counts": {},
        "status": "failed_missing_historical_pkl",
        "failure_reason": "No git-history daily/weekly pkl pair with matching internal as-of dates",
        "side_effects_disabled": {
            "futu": True,
            "telegram": True,
            "database": True,
            "production_pool_write": True,
            "pool_publish": True,
            "pool_commit": True,
        },
        "replay_strategy_env_keys": [],
    }
    (week_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return metadata


def write_manifest(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    flat_rows = []
    for row in rows:
        flat_rows.append(
            {
                "snapshot_date": row["snapshot_date"],
                "expected_last_trading_day": row["expected_last_trading_day"],
                "status": row["status"],
                "output_pool_path": row["output_pool_path"],
                "output_row_count": row["output_row_count"],
                "failure_reason": row["failure_reason"],
                "data_source_mode": row.get("data_source_mode"),
                "historical_pkl_commit": row.get("historical_pkl_commit"),
                "historical_pkl_commit_date": row.get("historical_pkl_commit_date"),
                "replay_old_pool_source": row.get("replay_old_pool_source"),
                "replay_old_pool_count": row.get("replay_old_pool_count"),
                "replay_new_pool_count": row.get("replay_new_pool_count"),
                "daily_pkl_file": row.get("daily_pkl_file"),
                "weekly_pkl_file": row.get("weekly_pkl_file"),
                "daily_max_date_before_clip": row.get("daily_max_date_before_clip"),
                "weekly_max_date_before_clip": row.get("weekly_max_date_before_clip"),
                "daily_max_date_after_clip": row.get("daily_max_date_after_clip"),
                "weekly_max_date_after_clip": row.get("weekly_max_date_after_clip"),
                "replay_used_clipped_data": row["replay_used_clipped_data"],
                "has_future_data_before_clip": row["has_future_data_before_clip"],
                "schema_validation_status": row.get("schema_audit", {}).get("schema_validation_status"),
            }
        )
    pd.DataFrame(flat_rows).to_csv(output_root / "manifest.csv", index=False)


def _total_counts(counts: dict[str, int]) -> int:
    return int(sum(int(v) for v in counts.values()))


def _format_counts(counts: dict[str, int], *, limit: int = 8) -> str:
    if not counts:
        return "-"
    items = sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))
    shown = [f"{key}={value}" for key, value in items[:limit]]
    if len(items) > limit:
        shown.append(f"...+{len(items) - limit}")
    return "; ".join(shown)


def _field_codes(pool: pd.DataFrame, field: str, mask: pd.Series) -> list[str]:
    if field not in pool.columns or "code" not in pool.columns:
        return []
    empty = pool[field].isna() | pool[field].astype(str).str.strip().eq("")
    return sorted(pool.loc[mask & empty, "code"].astype(str).unique().tolist())


def _load_local_eps_supplement_sources(base_path: Path) -> dict[str, dict[str, float]]:
    specs = [
        ("us/eps_growth_screener_results.csv", "eps_growth"),
        ("us/stage2/stage2_whitelist.csv", "eps_yoy_growth"),
        ("us/52wk_new_high_results.csv", "eps_growth"),
        ("us/weekly_vol_screener_results.csv", "eps_growth"),
    ]
    sources: dict[str, dict[str, float]] = {}
    for relative_path, eps_col in specs:
        path = base_path / relative_path
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype={"code": str})
        if "code" not in df.columns or eps_col not in df.columns:
            continue
        cur = df.loc[:, ["code", eps_col]].copy()
        cur[eps_col] = pd.to_numeric(cur[eps_col], errors="coerce")
        cur = cur.dropna(subset=[eps_col]).drop_duplicates(subset=["code"])
        sources[relative_path] = dict(zip(cur["code"].astype(str), cur[eps_col].astype(float)))
    return sources


def _lookup_eps_supplement(
    code: str,
    sources: dict[str, dict[str, float]],
) -> tuple[str, float | None]:
    for source_name, values in sources.items():
        if code in values:
            return source_name, values[code]
    return "", None


def write_data_source_audit_report(
    output_root: Path,
    rows: list[dict[str, Any]],
    *,
    expected_fields: list[str] | None = None,
    eps_supplement_sources: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    expected = EXPECTED_POOL_FIELDS if expected_fields is None else expected_fields
    eps_sources = eps_supplement_sources
    if eps_sources is None:
        eps_sources = _load_local_eps_supplement_sources(Path.cwd())
    audit_rows: list[dict[str, Any]] = []
    eps_gap_rows: list[dict[str, Any]] = []
    details: list[str] = []

    for row in rows:
        snapshot_date = str(row["snapshot_date"])
        pool_path = Path(row["output_pool_path"])
        if not pool_path.exists():
            audit = {
                "status": "failed",
                "row_count": 0,
                "column_count": 0,
                "missing_fields": ["breakout_follow_pool.csv"],
                "abnormal_empty_fields": {},
                "normal_empty_fields": {},
                "repairable_fallback_fields": {},
                "optional_gap_fields": {},
                "signal_rows": 0,
                "valid_ibd_entry_rows": 0,
                "invalid_ibd_entry_rows": 0,
            }
            pool = pd.DataFrame()
            signal_mask = pd.Series(dtype=bool)
        else:
            pool = pd.read_csv(pool_path, dtype={"code": str}, encoding="utf-8-sig")
            audit = audit_pool_null_semantics(pool, expected_fields=expected)
            signal_mask = (
                pool["signal"].fillna(False).astype(bool)
                if "signal" in pool.columns
                else pd.Series(False, index=pool.index)
            )

        non_eps_abnormal = {
            key: value
            for key, value in audit["abnormal_empty_fields"].items()
            if not key.startswith("eps_yoy_growth")
        }
        report_status = audit["status"]
        if audit["status"] == "failed" and not audit["missing_fields"] and not non_eps_abnormal:
            report_status = "passed_except_eps"

        signal_eps_missing_codes = _field_codes(pool, "eps_yoy_growth", signal_mask)
        signal_eps_supplement_available = 0
        signal_eps_unresolved = 0
        for code in signal_eps_missing_codes:
            source_name, source_value = _lookup_eps_supplement(code, eps_sources)
            if source_name:
                signal_eps_supplement_available += 1
            else:
                signal_eps_unresolved += 1
            eps_gap_rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "code": code,
                    "supplement_source": source_name,
                    "supplement_value": source_value,
                    "status": "current_snapshot_available_not_pit_safe" if source_name else "unresolved",
                }
            )
        audit_row = {
            "snapshot_date": snapshot_date,
            "status": report_status,
            "row_count": audit["row_count"],
            "column_count": audit["column_count"],
            "signal_rows": audit["signal_rows"],
            "valid_ibd_entry_rows": audit["valid_ibd_entry_rows"],
            "invalid_ibd_entry_rows": audit["invalid_ibd_entry_rows"],
            "missing_field_count": len(audit["missing_fields"]),
            "abnormal_empty_total": _total_counts(audit["abnormal_empty_fields"]),
            "non_eps_abnormal_empty_total": _total_counts(non_eps_abnormal),
            "signal_eps_missing": int(audit["abnormal_empty_fields"].get("eps_yoy_growth_signal", 0)),
            "signal_eps_supplement_available": signal_eps_supplement_available,
            "signal_eps_unresolved": signal_eps_unresolved,
            "normal_empty_total": _total_counts(audit["normal_empty_fields"]),
            "repairable_fallback_total": _total_counts(audit["repairable_fallback_fields"]),
            "optional_gap_total": _total_counts(audit["optional_gap_fields"]),
            "missing_fields": ";".join(audit["missing_fields"]),
            "abnormal_empty_fields": _format_counts(audit["abnormal_empty_fields"]),
            "non_eps_abnormal_empty_fields": _format_counts(non_eps_abnormal),
            "normal_empty_fields": _format_counts(audit["normal_empty_fields"]),
            "repairable_fallback_fields": _format_counts(audit["repairable_fallback_fields"]),
            "optional_gap_fields": _format_counts(audit["optional_gap_fields"]),
            "signal_eps_missing_codes": ";".join(signal_eps_missing_codes),
            "output_pool_path": str(pool_path),
        }
        audit_rows.append(audit_row)
        details.extend(
            [
                f"### {snapshot_date}",
                "",
                f"- 状态: `{audit_row['status']}`",
                f"- 需要补充/修复: {audit_row['abnormal_empty_fields']}",
                f"- 正常空值: {audit_row['normal_empty_fields']}",
                f"- repairable fallback: {audit_row['repairable_fallback_fields']}",
                f"- optional gap: {audit_row['optional_gap_fields']}",
                f"- signal EPS 缺失代码: {audit_row['signal_eps_missing_codes'] or '-'}",
                f"- signal EPS 本地补源覆盖: {signal_eps_supplement_available}; unresolved: {signal_eps_unresolved}",
                "",
            ]
        )

    (output_root / "data_source_audit_manifest.json").write_text(
        json.dumps(audit_rows, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(audit_rows).to_csv(output_root / "data_source_audit_manifest.csv", index=False)
    pd.DataFrame(eps_gap_rows).to_csv(output_root / "signal_eps_gap_supplement_plan.csv", index=False)

    total_abnormal = sum(int(row["non_eps_abnormal_empty_total"]) for row in audit_rows)
    total_eps_missing = sum(int(row["signal_eps_missing"]) for row in audit_rows)
    total_eps_supplement_available = sum(int(row["signal_eps_supplement_available"]) for row in audit_rows)
    total_eps_unresolved = sum(int(row["signal_eps_unresolved"]) for row in audit_rows)
    passed_weeks = sum(1 for row in audit_rows if row["status"] in {"passed", "passed_except_eps"})
    lines = [
        "# Replay Pool Data Source Audit",
        "",
        "## 判定规则",
        "",
        "- 字段列缺失: 不正常，必须修复。",
        "- 核心价格/结构字段空值: 不正常，必须修复。",
        "- signal 行的 `eps_yoy_growth` 空值: 单独隔离；只有 point-in-time EPS 源可安全补充，当前快照源不得回填。",
        "- signal 行的 IBD candidate / entry 判断字段必须完整；非 signal 行对应空值视为正常。",
        "- `industry` / `sector` 允许用 `Unknown` 作为 repairable fallback，但会单独计数。",
        "- `price_52_week_high` / `dist_to_52w_high_pct` 是价格 as-of 派生字段，必须由已裁剪 daily pkl 重算且不得为空。",
        "- pullback、dryness 等解释增强字段空值计为 optional gap，不阻断 pool 基准使用。",
        "",
        "## 总览",
        "",
        f"- Weeks audited: {len(audit_rows)}",
        f"- Passed weeks including EPS-isolated weeks: {passed_weeks}",
        f"- Weeks requiring supplement/repair: {len(audit_rows) - passed_weeks}",
        f"- Non-EPS abnormal empty values needing supplement/repair: {total_abnormal}",
        f"- Signal EPS gaps isolated pending point-in-time supplement: {total_eps_missing}",
        f"- Signal EPS gaps with current snapshot-only source: {total_eps_supplement_available}",
        f"- Signal EPS gaps unresolved: {total_eps_unresolved}",
        "- Current snapshot EPS supplement sources are reported separately and are not point-in-time safe.",
        "",
        "## 每周审计",
        "",
        "| snapshot_date | status | rows | cols | signal | missing_fields | non_eps_abnormal | signal_eps_missing | eps_supp_available | eps_unresolved | repairable_fallback | optional_gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit_rows:
        lines.append(
            f"| {row['snapshot_date']} | {row['status']} | {row['row_count']} | {row['column_count']} | "
            f"{row['signal_rows']} | {row['missing_field_count']} | {row['non_eps_abnormal_empty_total']} | "
            f"{row['signal_eps_missing']} | {row['signal_eps_supplement_available']} | "
            f"{row['signal_eps_unresolved']} | {row['repairable_fallback_total']} | {row['optional_gap_total']} |"
        )
    lines.extend(["", "## 每周明细", "", *details])
    (output_root / "data_source_audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    return audit_rows


def write_report(output_root: Path, rows: list[dict[str, Any]]) -> None:
    success = [r for r in rows if r["status"] == "success"]
    failed = [r for r in rows if r["status"] != "success"]
    clipped = [r for r in rows if r["replay_used_clipped_data"]]
    snapshot_dates = [str(row["snapshot_date"]) for row in rows]
    chronological_ok = snapshot_dates == sorted(snapshot_dates)
    excluded_ok = "2026-08-14" not in {r["snapshot_date"] for r in rows}
    successful_rows = [r for r in rows if r.get("status") == "success"]
    missing_pkl_rows = [r for r in rows if r.get("status") == "failed_missing_historical_pkl"]
    schema_ok = all(
        r.get("schema_audit", {}).get("schema_validation_status") in {"passed", "passed_with_repairs_or_optional_gaps"}
        for r in successful_rows
    )
    side_effects_ok = all(all(r.get("side_effects_disabled", {}).values()) for r in rows)
    historical_git_ok = all(r.get("data_source_mode") == "historical_git" for r in rows)
    carry_forward_ok = bool(rows) and rows[0].get("replay_old_pool_source") == "cold_start"
    for prev, cur in zip(rows, rows[1:]):
        expected_source = f"previous_replay_week:{prev['snapshot_date']}"
        if cur.get("replay_old_pool_source") not in {
            expected_source,
            f"reset_after_missing_pkl:{prev['snapshot_date']}",
        }:
            carry_forward_ok = False
    future_leak_ok = True
    historical_source_ok = True
    ibd_totals = {
        "signal_candidates": 0,
        "ibd_valid_nonempty": 0,
        "valid_entries": 0,
        "valid_entry_price_nonempty": 0,
        "invalid_entries": 0,
        "invalid_reject_nonempty": 0,
    }
    ibd_resolver_ok = True
    for row in rows:
        expected = pd.Timestamp(row["expected_last_trading_day"])
        if row["status"] == "success":
            if not row.get("historical_pkl_commit") or not row.get("daily_pkl_file") or not row.get("weekly_pkl_file"):
                historical_source_ok = False
            if row.get("daily_max_date_before_clip") != row.get("expected_last_trading_day"):
                historical_source_ok = False
            expected_week_start = (
                pd.Timestamp(row["expected_last_trading_day"]).normalize()
                - pd.Timedelta(days=pd.Timestamp(row["expected_last_trading_day"]).weekday())
            )
            weekly_max_date = row.get("weekly_max_date_before_clip")
            if (
                not weekly_max_date
                or pd.Timestamp(weekly_max_date).normalize() < expected_week_start
                or pd.Timestamp(weekly_max_date).normalize() > expected
            ):
                historical_source_ok = False
        for key in ("daily_max_date_after_clip", "weekly_max_date_after_clip"):
            value = row.get(key)
            if value and pd.Timestamp(value) > expected:
                future_leak_ok = False
        pool_path = Path(row["output_pool_path"])
        if pool_path.exists():
            pool = pd.read_csv(pool_path)
            signal_mask = pool["signal"].astype(str).str.lower().isin(["true", "1", "1.0"])
            candidate_mask = pool["ibd_candidate_rule"].notna()
            signal_candidates = pool[signal_mask & candidate_mask]
            valid_mask = signal_candidates["ibd_entry_valid"].astype(str).str.lower().isin(["true", "1", "1.0"])
            valid_entries = signal_candidates[valid_mask]
            invalid_entries = signal_candidates[~valid_mask]
            ibd_totals["signal_candidates"] += int(len(signal_candidates))
            ibd_totals["ibd_valid_nonempty"] += int(signal_candidates["ibd_entry_valid"].notna().sum())
            ibd_totals["valid_entries"] += int(len(valid_entries))
            ibd_totals["valid_entry_price_nonempty"] += int(valid_entries["ibd_entry_price"].notna().sum())
            ibd_totals["invalid_entries"] += int(len(invalid_entries))
            ibd_totals["invalid_reject_nonempty"] += int(invalid_entries["ibd_entry_reject_reason"].notna().sum())
            if signal_candidates["ibd_entry_valid"].isna().any():
                ibd_resolver_ok = False
            valid_required = [
                "ibd_entry_date",
                "ibd_entry_price",
                "ibd_trigger_price",
                "ibd_entry_volume_ratio",
                "ibd_entry_close_position",
                "ibd_entry_breakout_range_ratio",
            ]
            if not valid_entries.empty and valid_entries[valid_required].isna().any().any():
                ibd_resolver_ok = False
            if not invalid_entries.empty and invalid_entries["ibd_entry_reject_reason"].isna().any():
                ibd_resolver_ok = False
    lines = [
        "# Latest Quant Trade Replay Pool Audit",
        "",
        f"- Quant trade commit: `{rows[0].get('quant_trade_commit', '') if rows else ''}`",
        f"- Weeks processed: {len(rows)}",
        f"- Success weeks: {len(success)}",
        f"- Failed weeks: {len(failed)}",
        f"- Weeks using clipped data: {len(clipped)}",
        f"- Chronological boundary check: {'passed' if chronological_ok and excluded_ok else 'failed'} "
        f"(first={rows[0]['snapshot_date'] if rows else ''}, last={rows[-1]['snapshot_date'] if rows else ''}, "
        f"excluded_2026_08_14={excluded_ok})",
        f"- Replay old_pool carry-forward check: {'passed' if carry_forward_ok else 'failed'}",
        f"- Future-date leak check after clip: {'passed' if future_leak_ok else 'failed'}",
        f"- Schema check on successful pool weeks: {'passed' if schema_ok else 'failed'}",
        f"- Missing historical pkl weeks recorded as data gaps: {len(missing_pkl_rows)}",
        f"- Historical git pkl source check: {'passed' if historical_git_ok and historical_source_ok else 'failed'}",
        f"- IBD resolver field check: {'passed' if ibd_resolver_ok else 'failed'} "
        f"(signal_candidates={ibd_totals['signal_candidates']}, "
        f"ibd_entry_valid_nonempty={ibd_totals['ibd_valid_nonempty']}, "
        f"valid_entries={ibd_totals['valid_entries']}, "
        f"valid_entry_price_nonempty={ibd_totals['valid_entry_price_nonempty']}, "
        f"invalid_entries={ibd_totals['invalid_entries']}, "
        f"invalid_reject_nonempty={ibd_totals['invalid_reject_nonempty']})",
        f"- Side-effect isolation check: {'passed' if side_effects_ok else 'failed'}",
        "- Production pool write/publish/commit/Futu/Telegram/database side effects: disabled by replay wrapper.",
        "- Old `ibd_skill_replay_pools` contents are treated as untrusted and replaced by this clean replay baseline.",
        "",
        "## Week Status",
        "",
        "| snapshot_date | status | old_pool | new_pool | old_pool_source | rows | data_source | pkl_commit | daily_pkl | weekly_pkl | daily_max | weekly_max | clipped | schema | failure_reason |",
        "|---|---:|---:|---:|---|---:|---|---|---|---|---|---|---:|---|---|",
    ]
    for row in rows:
        schema_status = row.get("schema_audit", {}).get("schema_validation_status", "")
        commit = str(row.get("historical_pkl_commit") or "")
        lines.append(
            f"| {row['snapshot_date']} | {row['status']} | {row.get('replay_old_pool_count')} | "
            f"{row.get('replay_new_pool_count')} | {row.get('replay_old_pool_source')} | "
            f"{row.get('output_row_count', 0)} | "
            f"{row.get('data_source_mode', '')} | {commit[:8]} | "
            f"{Path(str(row.get('daily_pkl_file') or '')).name} | {Path(str(row.get('weekly_pkl_file') or '')).name} | "
            f"{row.get('daily_max_date_before_clip')} | {row.get('weekly_max_date_before_clip')} | "
            f"{row.get('replay_used_clipped_data', False)} | {schema_status} | {row.get('failure_reason', '')} |"
        )
    (output_root / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_execution_log(output_root: Path, rows: list[dict[str, Any]], *, quant_trade_path: Path) -> None:
    lines = [
        "# Latest Quant Trade Historical Pkl Replay Execution Log",
        "",
        "## Scope",
        "",
        f"- Rebuild complete-week breakout/follow pools from {rows[0]['snapshot_date'] if rows else ''} to {rows[-1]['snapshot_date'] if rows else ''}.",
        "- Use the latest checked-out quant_trade dev logic for pool generation.",
        "- Use git-history pkl blobs from this Yfinance_data repository as point-in-time market data.",
        "- Do not use existing historical pool CSV files as inputs.",
        "- Do not write production `us/` pool files, publish, commit from the strategy pipeline, send Telegram, connect Futu, or update databases.",
        "- Carry `old_pool` chronologically: first replay week cold-starts with an empty set; each successful week provides the next week's old_pool codes.",
        "",
        "## Procedure",
        "",
        "1. Enumerate complete NYSE weeks from the requested start date up to but excluding the configured production week.",
        "2. For each snapshot week, scan git history commits touching `results_pkl` from the expected close date through the configured search window.",
        "3. In each candidate commit tree, inspect available `stock_data_*_1d.pkl` and `stock_data_*_1wk.pkl` blobs by reading their internal price dates.",
        "4. Select the earliest commit whose daily pkl max date equals the snapshot close and whose weekly pkl max date stays inside the snapshot week without exceeding the close.",
        "5. Load selected pkl blobs directly with `git show <commit>:<path>`, run quant_trade `core_run` in replay mode with the carried old_pool set, then run the IBD entry enrichment helper against the replay output only.",
        "6. Clear current-snapshot EPS values, recompute 52-week-high fields from the selected as-of daily pkl, validate schema/null semantics, and write per-week metadata.",
        "",
        "## Commits",
        "",
        f"- quant_trade repo: `{quant_trade_path}`",
        f"- quant_trade commit: `{rows[0]['quant_trade_commit'] if rows else ''}`",
        f"- Yfinance_data commit at run start: `{rows[0]['Yfinance_data_commit'] if rows else ''}`",
        "",
        "## Weekly Pkl Mapping",
        "",
        "| snapshot_date | status | old_pool | new_pool | old_pool_source | pkl_commit | commit_date | daily_pkl | daily_max | weekly_pkl | weekly_max | future_before_clip | clipped | rows |",
        "|---|---|---:|---:|---|---|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        commit = str(row.get("historical_pkl_commit") or "")
        lines.append(
            f"| {row['snapshot_date']} | {row['status']} | "
            f"{row.get('replay_old_pool_count')} | {row.get('replay_new_pool_count')} | "
            f"{row.get('replay_old_pool_source')} | {commit[:12]} | "
            f"{row.get('historical_pkl_commit_date') or ''} | "
            f"{row.get('daily_pkl_file') or ''} | {row.get('daily_max_date_before_clip')} | "
            f"{row.get('weekly_pkl_file') or ''} | {row.get('weekly_max_date_before_clip')} | "
            f"{row.get('has_future_data_before_clip')} | {row.get('replay_used_clipped_data')} | "
            f"{row.get('output_row_count')} |"
        )
    (output_root / "execution_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--exclude-week-ending", default="2026-08-14")
    parser.add_argument("--pkl-source", choices=["historical-git", "current-files"], default="historical-git")
    parser.add_argument("--history-search-days", type=int, default=7)
    parser.add_argument("--daily-pkl", default="results_pkl/stock_data_150826_1d.pkl")
    parser.add_argument("--weekly-pkl", default="results_pkl/stock_data_150826_1wk.pkl")
    parser.add_argument("--output-root", default="backtest/ibd_skill_replay_pools")
    parser.add_argument("--quant-trade-path", default="/Users/tbin/Documents/quant_trade")
    parser.add_argument("--quant-trade-env", default="/Users/tbin/Documents/quant_trade/.env")
    parser.add_argument("--max-weeks", type=int, default=None)
    parser.add_argument("--clean-output-root", action="store_true")
    args = parser.parse_args(argv)

    yfinance_data_path = Path.cwd()
    output_root = Path(args.output_root)
    daily_pkl = Path(args.daily_pkl) if args.pkl_source == "current-files" else None
    weekly_pkl = Path(args.weekly_pkl) if args.pkl_source == "current-files" else None
    quant_trade_path = Path(args.quant_trade_path)
    quant_trade_env = Path(args.quant_trade_env) if args.quant_trade_env else None
    quant_trade_commit = git_commit(quant_trade_path)

    clean_log = None
    if args.clean_output_root:
        clean_log = clean_replay_output_root(
            output_root,
            reason="clean historical replay rebuild before regenerating pool/EPS audit data",
        )

    weeks = enumerate_complete_snapshot_weeks(
        start_date=args.start_date,
        exclude_week_ending=args.exclude_week_ending,
    )
    if args.max_weeks is not None:
        weeks = weeks[: args.max_weeks]

    rows = []
    replay_old_pool: set[str] = set()
    replay_old_pool_source = "cold_start"
    for week in weeks:
        if args.pkl_source == "historical-git":
            chosen, candidates = discover_historical_pkl_pair(
                repo_path=yfinance_data_path,
                snapshot_date=week.snapshot_date,
                expected_last_trading_day=week.expected_last_trading_day,
                search_days=args.history_search_days,
            )
            if chosen is None:
                rows.append(
                    metadata_for_missing_historical_pkl(
                        snapshot_date=week.snapshot_date,
                        expected_last_trading_day=week.expected_last_trading_day,
                        output_root=output_root,
                        yfinance_data_path=yfinance_data_path,
                        quant_trade_commit=quant_trade_commit,
                        candidate_count=len(candidates),
                        replay_old_pool=replay_old_pool,
                        replay_old_pool_source=replay_old_pool_source,
                    )
                )
                replay_old_pool = set()
                replay_old_pool_source = f"reset_after_missing_pkl:{week.snapshot_date}"
                continue
            daily_blob = git_blob_bytes(yfinance_data_path, chosen.commit, chosen.daily_path)
            weekly_blob = git_blob_bytes(yfinance_data_path, chosen.commit, chosen.weekly_path)
            row = run_one_week(
                snapshot_date=week.snapshot_date,
                expected_last_trading_day=week.expected_last_trading_day,
                daily_pkl=None,
                weekly_pkl=None,
                daily_data=normalize_pickle_data(pickle.loads(daily_blob)),
                weekly_data=normalize_pickle_data(pickle.loads(weekly_blob)),
                data_source_mode="historical_git",
                historical_pkl_commit=chosen.commit,
                historical_pkl_commit_date=chosen.commit_date,
                historical_pkl_candidate_count=len(candidates),
                daily_pkl_file=chosen.daily_path,
                weekly_pkl_file=chosen.weekly_path,
                daily_pkl_sha256=hashlib.sha256(daily_blob).hexdigest(),
                weekly_pkl_sha256=hashlib.sha256(weekly_blob).hexdigest(),
                output_root=output_root,
                quant_trade_path=quant_trade_path,
                quant_trade_env=quant_trade_env,
                yfinance_data_path=yfinance_data_path,
                quant_trade_commit=quant_trade_commit,
                replay_old_pool=replay_old_pool,
                replay_old_pool_source=replay_old_pool_source,
            )
            rows.append(row)
            replay_old_pool = load_replay_old_pool_from_metadata(row)
            replay_old_pool_source = f"previous_replay_week:{week.snapshot_date}"
            continue
        row = run_one_week(
            snapshot_date=week.snapshot_date,
            expected_last_trading_day=week.expected_last_trading_day,
            daily_pkl=daily_pkl,
            weekly_pkl=weekly_pkl,
            output_root=output_root,
            quant_trade_path=quant_trade_path,
            quant_trade_env=quant_trade_env,
            yfinance_data_path=yfinance_data_path,
            quant_trade_commit=quant_trade_commit,
            data_source_mode="current_files",
            replay_old_pool=replay_old_pool,
            replay_old_pool_source=replay_old_pool_source,
        )
        rows.append(row)
        replay_old_pool = load_replay_old_pool_from_metadata(row)
        replay_old_pool_source = f"previous_replay_week:{week.snapshot_date}"
    write_manifest(output_root, rows)
    write_report(output_root, rows)
    write_data_source_audit_report(output_root, rows)
    write_execution_log(output_root, rows, quant_trade_path=quant_trade_path)
    if clean_log is not None:
        (output_root / "clean_rebuild_log.json").write_text(
            json.dumps({**clean_log, "weeks_rebuilt": len(rows)}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
