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
    "original_ceiling", "breakout_date", "breakout_evidence",
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
