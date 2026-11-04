"""BF Pool Exit Watch published CSV and protected download codes."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

WATCH_COMPLETE = "us/bf_pool_exit_watch.csv"
WATCH_MIDWEEK = "us/bf_pool_exit_watch_midweek.csv"
ACTIVE_PHASES = frozenset({"AWAITING_DATA", "OBSERVING"})
COLUMNS = (
    "code", "watch_cycle_id", "snapshot_date", "source_pool_date",
    "source_pool_codes_hash", "original_pool_date", "original_signal",
    "original_ceiling", "anchor_close", "last_verified_price_state",
    "breakout_date", "breakout_evidence",
    "absence_week", "verified_week", "phase", "current_in_pool",
    "last_exit_reason", "exit_count", "latest_close", "price_state",
    "vs_original_pct", "observed_weeks", "pending_status", "events",
)


def parse_watch_csv(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.exists():
        return []
    frame = pd.read_csv(source, dtype={"code": str}, encoding="utf-8-sig", keep_default_na=False)
    required = set(COLUMNS)
    if not required.issubset(frame.columns):
        raise ValueError(f"Watch CSV schema missing: {sorted(required - set(frame.columns))}")
    if frame.code.eq("").any() or frame.code.duplicated().any():
        raise ValueError("Watch CSV duplicate/blank code")
    if not frame.phase.isin(
        ("AWAITING_DATA", "OBSERVING", "RETURNED", "PROVISIONAL_OUT")
    ).all():
        raise ValueError("Watch CSV unknown phase")
    rows = frame.to_dict("records")
    for row in rows:
        value = row["events"]
        row["events"] = json.loads(value) if value else []
        if not isinstance(row["events"], list):
            raise ValueError("Watch CSV events must be a JSON array")
    return rows


def active_watch_codes(data_root: str | Path = ".") -> list[str]:
    """Fail closed on malformed existing data, not on a never-enabled file."""
    rows = parse_watch_csv(Path(data_root) / WATCH_COMPLETE)
    return sorted({str(r["code"]) for r in rows if r["phase"] in ACTIVE_PHASES})


def watch_frame(rows: list[dict], *, snapshot_date: str, pool_codes_hash: str) -> pd.DataFrame:
    """Canonical minimal public data; row count may legally be zero."""
    if not snapshot_date or not pool_codes_hash:
        raise ValueError("Watch requires source Pool identity")
    output = []
    for row in rows:
        rendered = {name: row.get(name, "") for name in COLUMNS}
        rendered["snapshot_date"] = snapshot_date
        rendered["source_pool_date"] = snapshot_date
        rendered["source_pool_codes_hash"] = pool_codes_hash
        rendered["events"] = json.dumps(row.get("events") or [], ensure_ascii=False, separators=(",", ":"))
        output.append(rendered)
    frame = pd.DataFrame(output, columns=COLUMNS)
    if not frame.empty and (frame.code.eq("").any() or frame.code.duplicated().any()):
        raise ValueError("invalid Watch rows")
    return frame


def pool_identity(pool: pd.DataFrame) -> tuple[str, str]:
    from hashlib import sha256
    if pool.empty or not {"code", "snapshot_date"}.issubset(pool.columns):
        raise ValueError("Pool needs code and snapshot_date")
    dates = pd.to_datetime(pool["snapshot_date"], errors="coerce")
    if dates.isna().any() or dates.dt.date.nunique() != 1:
        raise ValueError("invalid or mixed Pool snapshot_date")
    codes = sorted(pool.code.astype(str))
    if len(codes) != len(set(codes)):
        raise ValueError("duplicate Pool codes")
    return dates.iloc[0].date().isoformat(), sha256("\n".join(codes).encode()).hexdigest()


def published_complete_pools(data_root: str | Path = ".", *, limit: int = 3) -> list[dict]:
    """Read latest distinct complete Pool weeks from published Git history."""
    import subprocess
    import io
    from datetime import date
    from bf_snapshot import complete_snapshot_week, monday_of_week
    root = Path(data_root)
    log = subprocess.check_output(["git", "-C", str(root), "log", "origin/main",
                                   "--first-parent", "--format=%H", "--",
                                   "us/breakout_follow_pool.csv"], text=True)
    found = {}
    for ref in log.splitlines():
        try:
            data = subprocess.check_output(["git", "-C", str(root), "show",
                                            f"{ref}:us/breakout_follow_pool.csv"])
            frame = pd.read_csv(io.BytesIO(data), dtype={"code": str})
            snapshot, _ = pool_identity(frame)
            complete_snapshot_week(date.fromisoformat(snapshot))
            week = monday_of_week(date.fromisoformat(snapshot)).isoformat()
            found.setdefault(week, dict(ref=ref, week=week, snapshot_date=snapshot,
                                        rows=frame.to_dict("records")))
            if len(found) >= limit:
                break
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
            continue
    return sorted(found.values(), key=lambda x: x["week"], reverse=True)[:limit]


def watch_was_ever_published(data_root: str | Path = ".") -> bool:
    import subprocess
    return bool(subprocess.check_output(["git", "-C", str(data_root), "log",
        "origin/main", "--format=%H", "--", WATCH_COMPLETE], text=True).strip())


def official_watch_for_week(data_root: str | Path, week: str) -> list[dict] | None:
    """None=missing/corrupt baseline; []=valid but legally empty Watch."""
    import subprocess
    import io
    from datetime import date
    from bf_snapshot import monday_of_week
    root = Path(data_root)
    refs = subprocess.check_output(["git", "-C", str(root), "log", "origin/main",
        "--first-parent", "--format=%H", "--", WATCH_COMPLETE], text=True)
    for ref in refs.splitlines():
        try:
            blob = subprocess.check_output(["git", "-C", str(root), "show",
                                            f"{ref}:{WATCH_COMPLETE}"])
            frame = pd.read_csv(io.BytesIO(blob), dtype={"code": str}, keep_default_na=False)
            if not set(COLUMNS).issubset(frame.columns):
                continue
            if frame.empty:
                pool_blob = subprocess.check_output(["git", "-C", str(root), "show",
                                                     f"{ref}:us/breakout_follow_pool.csv"])
                source, _ = pool_identity(pd.read_csv(io.BytesIO(pool_blob), dtype={"code": str}))
            else:
                dates = set(frame.source_pool_date.astype(str))
                if len(dates) != 1:
                    continue
                source = next(iter(dates))
            if monday_of_week(date.fromisoformat(source)).isoformat() != week:
                continue
            rows = frame.to_dict("records")
            for row in rows:
                row["events"] = json.loads(row["events"]) if row["events"] else []
            return rows
        except (OSError, ValueError, subprocess.CalledProcessError):
            continue
    return None
