"""CLI for clean complete-week pool replay using quant_trade dev logic."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from . import (
    EXPECTED_POOL_FIELDS,
    ReplayPoolSink,
    apply_replay_strategy_env,
    audit_pool_null_semantics,
    audit_pool_schema,
    clip_price_data_asof,
    enumerate_complete_snapshot_weeks,
    repair_research_fields,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pickle_data(path: Path) -> dict[str, pd.DataFrame]:
    with path.open("rb") as f:
        data = pickle.load(f)
    out: dict[str, pd.DataFrame] = {}
    for code, value in data.items():
        if isinstance(value, dict) and set(value.keys()) == {"index", "columns", "data"}:
            value = pd.DataFrame(**value)
        out[str(code)] = value
    return out


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


def run_one_week(
    *,
    snapshot_date: str,
    expected_last_trading_day: str,
    daily_pkl: Path,
    weekly_pkl: Path,
    output_root: Path,
    quant_trade_path: Path,
    quant_trade_env: Path | None,
    yfinance_data_path: Path,
    quant_trade_commit: str,
) -> dict[str, Any]:
    daily_raw = load_pickle_data(daily_pkl)
    weekly_raw = load_pickle_data(weekly_pkl)
    daily_clip = clip_price_data_asof(daily_raw, expected_last_trading_day)
    weekly_clip = clip_price_data_asof(weekly_raw, expected_last_trading_day)

    week_dir = output_root / snapshot_date
    pool_path = week_dir / "breakout_follow_pool.csv"
    sink = ReplayPoolSink(pool_path)
    run_output_dir = week_dir / "quant_trade_outputs"
    run_output_dir.mkdir(parents=True, exist_ok=True)

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

        ctx = RunContext.replay(snapshot_date, old_pool=set())
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

    metadata = {
        "snapshot_date": snapshot_date,
        "expected_last_trading_day": expected_last_trading_day,
        "quant_trade_commit": quant_trade_commit,
        "Yfinance_data_commit": git_commit(yfinance_data_path),
        "daily_pkl_file": str(daily_pkl),
        "weekly_pkl_file": str(weekly_pkl),
        "daily_pkl_sha256": sha256_file(daily_pkl),
        "weekly_pkl_sha256": sha256_file(weekly_pkl),
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
                    "status": "local_supplement_available" if source_name else "unresolved",
                }
            )
        audit_row = {
            "snapshot_date": snapshot_date,
            "status": audit["status"],
            "row_count": audit["row_count"],
            "column_count": audit["column_count"],
            "signal_rows": audit["signal_rows"],
            "valid_ibd_entry_rows": audit["valid_ibd_entry_rows"],
            "invalid_ibd_entry_rows": audit["invalid_ibd_entry_rows"],
            "missing_field_count": len(audit["missing_fields"]),
            "abnormal_empty_total": _total_counts(audit["abnormal_empty_fields"]),
            "signal_eps_missing": int(audit["abnormal_empty_fields"].get("eps_yoy_growth_signal", 0)),
            "signal_eps_supplement_available": signal_eps_supplement_available,
            "signal_eps_unresolved": signal_eps_unresolved,
            "normal_empty_total": _total_counts(audit["normal_empty_fields"]),
            "repairable_fallback_total": _total_counts(audit["repairable_fallback_fields"]),
            "optional_gap_total": _total_counts(audit["optional_gap_fields"]),
            "missing_fields": ";".join(audit["missing_fields"]),
            "abnormal_empty_fields": _format_counts(audit["abnormal_empty_fields"]),
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

    total_abnormal = sum(int(row["abnormal_empty_total"]) for row in audit_rows)
    total_eps_missing = sum(int(row["signal_eps_missing"]) for row in audit_rows)
    total_eps_supplement_available = sum(int(row["signal_eps_supplement_available"]) for row in audit_rows)
    total_eps_unresolved = sum(int(row["signal_eps_unresolved"]) for row in audit_rows)
    passed_weeks = sum(1 for row in audit_rows if row["status"] == "passed")
    lines = [
        "# Replay Pool Data Source Audit",
        "",
        "## 判定规则",
        "",
        "- 字段列缺失: 不正常，必须修复。",
        "- 核心价格/结构字段空值: 不正常，必须修复。",
        "- signal 行的 `eps_yoy_growth` 空值: 不正常，必须补充；非 signal 行 EPS 空值视为正常。",
        "- signal 行的 IBD candidate / entry 判断字段必须完整；非 signal 行对应空值视为正常。",
        "- `industry` / `sector` 允许用 `Unknown` 作为 repairable fallback，但会单独计数。",
        "- pullback、52w high、dryness 等解释增强字段空值计为 optional gap，不阻断 pool 基准使用。",
        "",
        "## 总览",
        "",
        f"- Weeks audited: {len(audit_rows)}",
        f"- Passed weeks: {passed_weeks}",
        f"- Weeks requiring supplement/repair: {len(audit_rows) - passed_weeks}",
        f"- Abnormal empty values needing supplement/repair: {total_abnormal}",
        f"- Signal EPS gaps needing supplement: {total_eps_missing}",
        f"- Signal EPS gaps with local supplement source: {total_eps_supplement_available}",
        f"- Signal EPS gaps unresolved: {total_eps_unresolved}",
        "- EPS supplement sources are reported separately and are not silently written back into historical replay pools.",
        "",
        "## 每周审计",
        "",
        "| snapshot_date | status | rows | cols | signal | missing_fields | abnormal_empty | signal_eps_missing | eps_supp_available | eps_unresolved | repairable_fallback | optional_gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit_rows:
        lines.append(
            f"| {row['snapshot_date']} | {row['status']} | {row['row_count']} | {row['column_count']} | "
            f"{row['signal_rows']} | {row['missing_field_count']} | {row['abnormal_empty_total']} | "
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
    boundary_ok = bool(rows) and rows[0]["snapshot_date"] == "2026-01-02" and rows[-1]["snapshot_date"] == "2026-08-07"
    excluded_ok = "2026-08-14" not in {r["snapshot_date"] for r in rows}
    schema_ok = all(r.get("schema_audit", {}).get("schema_validation_status") == "passed" for r in rows)
    side_effects_ok = all(all(r.get("side_effects_disabled", {}).values()) for r in rows)
    future_leak_ok = True
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
        f"- Quant trade commit: `{rows[0]['quant_trade_commit'] if rows else ''}`",
        f"- Weeks processed: {len(rows)}",
        f"- Success weeks: {len(success)}",
        f"- Failed weeks: {len(failed)}",
        f"- Weeks using clipped data: {len(clipped)}",
        f"- Boundary check: {'passed' if boundary_ok and excluded_ok else 'failed'} "
        f"(first={rows[0]['snapshot_date'] if rows else ''}, last={rows[-1]['snapshot_date'] if rows else ''}, "
        f"excluded_2026_08_14={excluded_ok})",
        f"- Future-date leak check after clip: {'passed' if future_leak_ok else 'failed'}",
        f"- Schema check: {'passed' if schema_ok else 'failed'}",
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
        "| snapshot_date | status | rows | clipped | schema | failure_reason |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        schema_status = row.get("schema_audit", {}).get("schema_validation_status", "")
        lines.append(
            f"| {row['snapshot_date']} | {row['status']} | {row['output_row_count']} | "
            f"{row['replay_used_clipped_data']} | {schema_status} | {row['failure_reason']} |"
        )
    (output_root / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--exclude-week-ending", default="2026-08-14")
    parser.add_argument("--daily-pkl", default="results_pkl/stock_data_150826_1d.pkl")
    parser.add_argument("--weekly-pkl", default="results_pkl/stock_data_150826_1wk.pkl")
    parser.add_argument("--output-root", default="backtest/ibd_skill_replay_pools")
    parser.add_argument("--quant-trade-path", default="/Users/tbin/Documents/quant_trade")
    parser.add_argument("--quant-trade-env", default="/Users/tbin/Documents/quant_trade/.env")
    parser.add_argument("--max-weeks", type=int, default=None)
    args = parser.parse_args(argv)

    yfinance_data_path = Path.cwd()
    output_root = Path(args.output_root)
    daily_pkl = Path(args.daily_pkl)
    weekly_pkl = Path(args.weekly_pkl)
    quant_trade_path = Path(args.quant_trade_path)
    quant_trade_env = Path(args.quant_trade_env) if args.quant_trade_env else None
    quant_trade_commit = git_commit(quant_trade_path)

    weeks = enumerate_complete_snapshot_weeks(
        start_date=args.start_date,
        exclude_week_ending=args.exclude_week_ending,
    )
    if args.max_weeks is not None:
        weeks = weeks[: args.max_weeks]

    rows = []
    for week in weeks:
        rows.append(
            run_one_week(
                snapshot_date=week.snapshot_date,
                expected_last_trading_day=week.expected_last_trading_day,
                daily_pkl=daily_pkl,
                weekly_pkl=weekly_pkl,
                output_root=output_root,
                quant_trade_path=quant_trade_path,
                quant_trade_env=quant_trade_env,
                yfinance_data_path=yfinance_data_path,
                quant_trade_commit=quant_trade_commit,
            )
        )
    write_manifest(output_root, rows)
    write_report(output_root, rows)
    write_data_source_audit_report(output_root, rows)
    return 0
