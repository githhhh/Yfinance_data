from __future__ import annotations

import argparse
import csv
import io
import json
import pickle
import subprocess
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

from backtest.ibd_skill_replay.core import (
    SnapshotMeta,
    choose_complete_week_snapshots,
    compute_path_metrics,
    repair_pool_fields,
    select_current_skill_top3,
    select_old_skill_proxy_top3,
    to_float,
)


REQUIRED_REPLAY_FIELDS = {
    "code",
    "snapshot_date",
    "signal",
    "ibd_candidate_rule",
    "ibd_entry_status",
    "ibd_candidate_price",
    "latest_close",
    "current_vs_ibd_candidate_pct",
    "ibd_entry_volume_ratio",
    "ibd_entry_close_position",
    "ibd_entry_breakout_range_ratio",
    "industry",
}


OLD_SELECTION_OVERRIDES = {
    # User-confirmed old/Gemini ordering for IMAX week.
    "2026-07-24": ["IMAX", "OFG", "OVV"],
    # Saved report in doc/ibd_prescreen_report.md.
    "2026-07-31": ["LH", "SHOO", "OBK"],
}

NEW_SELECTION_OVERRIDES = {
    # Preserve the already-audited IMAX-week current-skill comparison.
    "2026-07-24": ["PCAR", "PKG", "NWFL"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay old/current IBD skill picks over historical complete-week pools.")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--exclude-snapshot", default="2026-08-14")
    parser.add_argument("--price-cache", default="results_pkl/stock_data_150826_1d.pkl")
    parser.add_argument("--supplemental-price-csv", default="backtest/ibd_skill_replay_audit/supplemental_price_bars.csv")
    parser.add_argument("--fetch-missing-prices", action="store_true")
    parser.add_argument("--output-root", default="backtest")
    args = parser.parse_args()

    root = Path(args.output_root)
    pools_dir = root / "ibd_skill_replay_pools"
    reports_dir = root / "ibd_skill_replay_reports"
    audit_dir = root / "ibd_skill_replay_audit"
    for directory in (pools_dir, reports_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)

    prices = _load_price_cache(Path(args.price_cache))
    supplemental_path = Path(args.supplemental_price_csv)
    prices.update(_load_supplemental_prices(supplemental_path))
    metas, pools = _scan_git_history()
    chosen = choose_complete_week_snapshots(
        metas,
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        excluded_snapshot=args.exclude_snapshot,
    )
    chosen = [meta for meta in chosen if date.fromisoformat(meta.snapshot_date).weekday() == 4]

    audit_rows = []
    replay_rows = []
    for meta in metas:
        if not (args.start <= meta.snapshot_date <= args.end):
            continue
        pool = pools[(meta.snapshot_date, meta.commit)]
        repaired = repair_pool_fields(pool, prices, snapshot_date=meta.snapshot_date)
        repaired_path = pools_dir / f"{meta.snapshot_date}_{meta.commit}_pool.csv"
        repaired.to_csv(repaired_path, index=False, encoding="utf-8-sig")
        missing_fields = sorted(REQUIRED_REPLAY_FIELDS - set(pool.columns))
        actionables = _actionable_count(pool)
        audit_rows.append(
            {
                "snapshot_date": meta.snapshot_date,
                "commit": meta.commit,
                "commit_date": meta.commit_date,
                "row_count": len(pool),
                "actionable_count": actionables,
                "comparable_schema": meta.comparable_schema,
                "selected_for_replay": meta in chosen,
                "missing_required_replay_fields": ";".join(missing_fields),
                "repaired_pool_path": str(repaired_path),
            }
        )

    selected_codes = _selected_codes_for_snapshots(chosen, pools_dir)
    missing_selected_prices = sorted(code for code in selected_codes if code not in prices or prices[code].empty)
    if args.fetch_missing_prices and missing_selected_prices:
        fetched = _fetch_missing_prices(missing_selected_prices, start=args.start, end=args.end)
        _append_supplemental_prices(supplemental_path, fetched)
        prices.update(fetched)
        missing_selected_prices = sorted(code for code in selected_codes if code not in prices or prices[code].empty)

    for meta in chosen:
        pool = pd.read_csv(pools_dir / f"{meta.snapshot_date}_{meta.commit}_pool.csv", encoding="utf-8-sig")
        replay_rows.extend(_replay_snapshot(meta, pool, prices, end_date=args.end))

    audit_csv = audit_dir / "snapshot_field_audit.csv"
    pd.DataFrame(audit_rows).sort_values(["snapshot_date", "commit_date"]).to_csv(audit_csv, index=False)
    replay_csv = reports_dir / "replay_pick_metrics.csv"
    pd.DataFrame(replay_rows).to_csv(replay_csv, index=False)
    summary_md = reports_dir / "replay_summary.md"
    summary_md.write_text(_render_summary(audit_rows, replay_rows, args), encoding="utf-8")

    manifest = {
        "start": args.start,
        "end": args.end,
        "excluded_snapshot": args.exclude_snapshot,
        "price_cache": args.price_cache,
        "supplemental_price_csv": args.supplemental_price_csv,
        "fetch_missing_prices": args.fetch_missing_prices,
        "missing_selected_prices": missing_selected_prices,
        "complete_week_filter": "snapshot_date.weekday == Friday",
        "audit_csv": str(audit_csv),
        "replay_csv": str(replay_csv),
        "summary_md": str(summary_md),
        "comparable_snapshots": [asdict(meta) for meta in chosen],
    }
    (reports_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def _load_price_cache(path: Path) -> dict[str, pd.DataFrame]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    prices = {}
    for code, obj in raw.items():
        if not isinstance(obj, dict) or not {"index", "columns", "data"}.issubset(obj):
            continue
        prices[code] = pd.DataFrame(obj["data"], index=pd.to_datetime(obj["index"]), columns=obj["columns"])
        prices[code].attrs["source"] = "daily_cache"
    return prices


def _load_supplemental_prices(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists():
        return {}
    raw = pd.read_csv(path)
    if raw.empty:
        return {}
    prices = {}
    for code, group in raw.groupby("code"):
        prices[str(code)] = group.drop(columns=["code"]).copy()
        prices[str(code)].attrs["source"] = "supplemental_yfinance"
    return prices


def _fetch_missing_prices(codes: list[str], *, start: str, end: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    fetched = {}
    end_exclusive = (date.fromisoformat(end) + pd.Timedelta(days=1)).isoformat()
    for code in codes:
        data = yf.download(code, start=start, end=end_exclusive, progress=False, auto_adjust=False)
        if data.empty:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [column[0] for column in data.columns]
        fetched[code] = data.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
        fetched[code].attrs["source"] = "supplemental_yfinance"
    return fetched


def _append_supplemental_prices(path: Path, fetched: dict[str, pd.DataFrame]) -> None:
    if not fetched:
        return
    rows = []
    for code, data in fetched.items():
        frame = data.copy()
        frame.insert(0, "code", code)
        rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)
    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, combined], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"]).dt.date.astype(str)
    combined = combined.drop_duplicates(["code", "Date"], keep="last").sort_values(["code", "Date"])
    combined.to_csv(path, index=False)


def _scan_git_history() -> tuple[list[SnapshotMeta], dict[tuple[str, str], pd.DataFrame]]:
    log = subprocess.check_output(
        ["git", "log", "--pretty=format:%H%x09%h%x09%ad", "--date=short", "--", "us/breakout_follow_pool.csv"],
        text=True,
    )
    metas = []
    pools = {}
    for line in log.splitlines():
        full, short, commit_date = line.split("\t", 2)
        blob = subprocess.check_output(["git", "show", f"{full}:us/breakout_follow_pool.csv"])
        rows = list(csv.DictReader(io.StringIO(blob.decode("utf-8-sig"))))
        if not rows:
            continue
        pool = pd.DataFrame(rows)
        snapshots = sorted({str(value).strip() for value in pool.get("snapshot_date", pd.Series(dtype=str)).dropna() if str(value).strip()})
        if len(snapshots) != 1:
            continue
        snapshot = snapshots[0]
        comparable = REQUIRED_REPLAY_FIELDS.issubset(pool.columns) and _actionable_count(pool) > 0
        meta = SnapshotMeta(
            snapshot_date=snapshot,
            commit=short,
            commit_date=commit_date,
            row_count=len(pool),
            actionable_count=_actionable_count(pool),
            comparable_schema=comparable,
        )
        metas.append(meta)
        pools[(snapshot, short)] = pool
    return metas, pools


def _actionable_count(pool: pd.DataFrame) -> int:
    if "signal" not in pool.columns or "ibd_entry_status" not in pool.columns:
        return 0
    signal = pool["signal"].astype(str).str.lower().isin({"true", "1"})
    status = pool["ibd_entry_status"].astype(str).str.upper().eq("ACTIONABLE")
    return int((signal & status).sum())


def _selected_codes_for_snapshots(metas: list[SnapshotMeta], pools_dir: Path) -> set[str]:
    codes = set()
    for meta in metas:
        pool = pd.read_csv(pools_dir / f"{meta.snapshot_date}_{meta.commit}_pool.csv", encoding="utf-8-sig")
        old_codes = OLD_SELECTION_OVERRIDES.get(meta.snapshot_date)
        new_codes = NEW_SELECTION_OVERRIDES.get(meta.snapshot_date)
        if old_codes is None:
            old_codes = [item.code for item in select_old_skill_proxy_top3(pool).selected]
        if new_codes is None:
            new_codes = [item.code for item in select_current_skill_top3(pool).selected]
        codes.update(old_codes)
        codes.update(new_codes)
    return codes


def _replay_snapshot(
    meta: SnapshotMeta,
    pool: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    *,
    end_date: str,
) -> list[dict[str, object]]:
    rows = []
    selections = {
        "old": OLD_SELECTION_OVERRIDES.get(meta.snapshot_date),
        "new": NEW_SELECTION_OVERRIDES.get(meta.snapshot_date),
    }
    if selections["old"] is None:
        selections["old"] = [item.code for item in select_old_skill_proxy_top3(pool).selected]
    if selections["new"] is None:
        selections["new"] = [item.code for item in select_current_skill_top3(pool).selected]

    pool_by_code = {str(row["code"]): row for _, row in pool.iterrows()}
    for skill, codes in selections.items():
        for order, code in enumerate(codes, 1):
            row = pool_by_code.get(code)
            if row is None:
                continue
            metrics = compute_path_metrics(
                code=code,
                snapshot_date=meta.snapshot_date,
                buy_price=to_float(row.get("ibd_candidate_price")),
                snapshot_close=to_float(row.get("latest_close")),
                price_bars=prices.get(code),
                end_date=end_date,
            )
            rows.append(
                {
                    "snapshot_date": meta.snapshot_date,
                    "commit": meta.commit,
                    "skill": skill,
                    "pick_order": order,
                    "code": code,
                    "sector": row.get("sector"),
                    "industry": row.get("industry"),
                    "buy_price": metrics.buy_price,
                    "snapshot_close": metrics.snapshot_close,
                    "latest_close": metrics.latest_close,
                    "latest_close_return_pct": metrics.latest_close_return_pct,
                    "max_gain_pct": metrics.max_gain_pct,
                    "max_gain_date": metrics.max_gain_date,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "max_drawdown_date": metrics.max_drawdown_date,
                    "hit_stop_8pct": metrics.hit_stop_8pct,
                    "stop_8pct_date": metrics.stop_8pct_date,
                    "path_source": metrics.source,
                    "current_vs_ibd_candidate_pct": to_float(row.get("current_vs_ibd_candidate_pct")),
                    "ibd_entry_volume_ratio": to_float(row.get("ibd_entry_volume_ratio")),
                    "eps_yoy_growth": to_float(row.get("eps_yoy_growth")),
                }
            )
    return rows


def _render_summary(audit_rows: list[dict[str, object]], replay_rows: list[dict[str, object]], args: argparse.Namespace) -> str:
    replay = pd.DataFrame(replay_rows)
    audit = pd.DataFrame(audit_rows)
    lines = [
        "# IBD Skill Replay Summary",
        "",
        f"- Range requested: `{args.start}` to `{args.end}`",
        f"- Excluded performance snapshot: `{args.exclude_snapshot}`",
        f"- Comparable replay snapshots: `{int(audit['selected_for_replay'].sum()) if not audit.empty else 0}`",
        f"- Repaired pool files: `{len(audit_rows)}`",
        "",
        "## Scope Note",
        "",
        "Historical git pool rows define the candidate universe. Price-derived repair fields use snapshot-bounded bars only. "
        "Weeks without ACTIONABLE entry fields are kept in the audit and repaired where possible, but they are not used for same-schema skill replay.",
        "",
    ]
    if replay.empty:
        lines.append("No replay rows were generated.")
        return "\n".join(lines)

    lines.extend(["## Skill Summary", "", "| Skill | Picks | With Metrics | Avg Close Return | Avg Max Gain | -8% Stops | Positive |", "|---|---:|---:|---:|---:|---:|---:|"])
    for skill, group in replay.groupby("skill"):
        close_ret = pd.to_numeric(group["latest_close_return_pct"], errors="coerce")
        max_gain = pd.to_numeric(group["max_gain_pct"], errors="coerce")
        lines.append(
            f"| {skill} | {len(group)} | {close_ret.notna().sum()} | {_fmt(close_ret.mean())} | {_fmt(max_gain.mean())} | "
            f"{int(group['hit_stop_8pct'].sum())} | {int((close_ret > 0).sum())} |"
        )
    lines.extend(["", "## Pick Order Summary", "", "| Skill | Order | Picks | Avg Close Return | Avg Max Gain | -8% Stops |", "|---|---:|---:|---:|---:|---:|"])
    for (skill, order), group in replay.groupby(["skill", "pick_order"]):
        close_ret = pd.to_numeric(group["latest_close_return_pct"], errors="coerce")
        max_gain = pd.to_numeric(group["max_gain_pct"], errors="coerce")
        lines.append(
            f"| {skill} | {order} | {close_ret.notna().sum()} | {_fmt(close_ret.mean())} | {_fmt(max_gain.mean())} | "
            f"{int(group['hit_stop_8pct'].sum())} |"
        )
    lines.extend(
        [
            "",
            "## Detailed Picks",
            "",
            "| Week | Skill | Order | Code | Sector | Industry | Buy | Latest Close | Close Ret | Max Gain | Max Gain Date | Max Drawdown | -8% Stop | Stop Date | Source |",
            "|---|---|---:|---|---|---|---:|---:|---:|---:|---|---:|---|---|---|",
        ]
    )
    for _, row in replay.sort_values(["snapshot_date", "skill", "pick_order"]).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("snapshot_date", "")),
                    str(row.get("skill", "")),
                    str(row.get("pick_order", "")),
                    str(row.get("code", "")),
                    str(row.get("sector", "")),
                    str(row.get("industry", "")),
                    _fmt_number(row.get("buy_price")),
                    _fmt_number(row.get("latest_close")),
                    _fmt(row.get("latest_close_return_pct")),
                    _fmt(row.get("max_gain_pct")),
                    str(row.get("max_gain_date", "") or ""),
                    _fmt(row.get("max_drawdown_pct")),
                    "yes" if bool(row.get("hit_stop_8pct")) else "no",
                    str(row.get("stop_8pct_date", "") or ""),
                    str(row.get("path_source", "")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Outputs", "", "- `backtest/ibd_skill_replay_audit/snapshot_field_audit.csv`", "- `backtest/ibd_skill_replay_reports/replay_pick_metrics.csv`", "- `backtest/ibd_skill_replay_pools/*.csv`"])
    return "\n".join(lines)


def _fmt(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f}%"


def _fmt_number(value: object) -> str:
    numeric = to_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.2f}"


if __name__ == "__main__":
    main()
