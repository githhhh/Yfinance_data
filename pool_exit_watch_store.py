"""BF Pool Exit Watch published CSV and protected download codes."""
from __future__ import annotations
from pathlib import Path
import json
from datetime import date
import pandas as pd

WATCH_COMPLETE = "us/bf_pool_exit_watch.csv"
WATCH_MIDWEEK = "us/bf_pool_exit_watch_midweek.csv"
ACTIVE_PHASES = frozenset({"AWAITING_DATA", "OBSERVING"})
COLUMNS = (
    "code", "watch_cycle_id", "snapshot_date", "original_pool_date", "original_signal",
    "original_ceiling", "current_ceiling", "anchor_close", "last_verified_price_state",
    "last_price_observed_week",
    "breakout_date", "absence_week", "verified_week", "phase", "current_in_pool",
    "last_exit_reason", "latest_close", "price_state",
    "vs_original_pct", "observed_weeks", "pending_status", "pending_high",
    "events", "download_hold_reason",
)


def parse_watch_frame(frame: pd.DataFrame) -> list[dict]:
    """Validate one identical Watch contract for staged, local and Git blobs."""
    if list(frame.columns) != list(COLUMNS):
        raise ValueError("Watch CSV must use the exact 24-column published schema")
    codes = frame.code.astype(str)
    if (frame.code.isna().any() or codes.eq("").any() or
        codes.str.lower().eq("nan").any() or
        codes.str.strip().ne(codes).any() or codes.duplicated().any()):
        raise ValueError("Watch CSV duplicate/blank/noncanonical code")
    if not frame.phase.isin(
        ("AWAITING_DATA", "OBSERVING", "RETURNED", "PROVISIONAL_OUT")
    ).all():
        raise ValueError("Watch CSV unknown phase")
    rows = frame.to_dict("records")
    for row in rows:
        # Empty historical fields are legal during initial bootstrap; any
        # populated market date must be canonical before it can be persisted.
        reason = row["download_hold_reason"]
        if not isinstance(reason, str) or len(reason) > 160 or (
            reason and (reason != reason.strip() or
                        any(ord(char) < 32 for char in reason))
        ):
            raise ValueError("Watch CSV invalid download_hold_reason")
        if reason and (row["phase"] not in ACTIVE_PHASES or
                       not str(row.get("watch_cycle_id") or "")):
            raise ValueError("only an active official Watch Cycle can be held")
        for field in ("snapshot_date", "original_pool_date",
                      "absence_week", "verified_week", "last_price_observed_week"):
            value = row.get(field)
            if value is None or value == "":
                continue
            try:
                parsed = date.fromisoformat(str(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Watch CSV invalid {field}") from exc
            if (str(value) != parsed.isoformat() or
                (field in {"absence_week", "verified_week",
                           "last_price_observed_week"} and parsed.weekday() != 0)):
                raise ValueError(f"Watch CSV noncanonical {field}")
        value = row["events"]
        events = json.loads(value) if isinstance(value, str) and value else value or []
        if not isinstance(events, list):
            raise ValueError("Watch CSV events must be a JSON array")
        seen = set()
        for event in events:
            if not isinstance(event, dict) or not all(
                isinstance(event.get(k), str) and event[k]
                for k in ("id", "type", "week")
            ):
                raise ValueError("Watch CSV event is missing its identity")
            try:
                monday = date.fromisoformat(event["week"])
            except ValueError as exc:
                raise ValueError("Watch CSV invalid event market week") from exc
            if (monday.weekday() != 0 or event["week"] != monday.isoformat() or
                event["type"] not in {"POOL_EXIT", "POOL_RETURN", "CEILING_LOST",
                                     "CEILING_RECLAIMED", "ZONE_TOUCH",
                                     "PENDING_CONFIRMED", "PENDING_BLOCKED",
                                     "COVERAGE_GAP"}):
                raise ValueError("Watch CSV invalid event type or canonical week")
            if event["id"] != f'{event["type"]}:{event["week"]}' or event["id"] in seen:
                raise ValueError("Watch CSV duplicate or mismatched event identity")
            seen.add(event["id"])
            snapshot_date = row.get("snapshot_date")
            if snapshot_date:
                current_date = date.fromisoformat(str(snapshot_date))
                current_week = current_date.toordinal() - current_date.weekday()
                if monday.toordinal() > current_week:
                    raise ValueError("Watch CSV event references a future market week")
        if row.get("snapshot_date"):
            current_date = date.fromisoformat(str(row["snapshot_date"]))
            current_week = current_date.toordinal() - current_date.weekday()
            for field in ("absence_week", "verified_week", "last_price_observed_week"):
                value = row.get(field)
                if value and date.fromisoformat(str(value)).toordinal() > current_week:
                    raise ValueError(f"Watch CSV {field} references future market week")
            original = row.get("original_pool_date")
            if original and date.fromisoformat(str(original)) > current_date:
                raise ValueError("Watch CSV original_pool_date after snapshot")
        row["events"] = events
    return rows


def parse_watch_csv(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.exists():
        return []
    frame = pd.read_csv(source, dtype={"code": str, "download_hold_reason": str}, encoding="utf-8-sig", keep_default_na=False)
    return parse_watch_frame(frame)


def _validated_watch_rows(data_root: str | Path = ".", *, required: bool = False) -> list[dict]:
    """Read only the authoritative published complete Pool/Watch pair."""
    import subprocess
    root = Path(data_root)
    source = root / WATCH_COMPLETE
    is_repo = (root / ".git").exists()
    if not source.is_file():
        if is_repo and watch_was_ever_published(root):
            raise RuntimeError("published BF Watch missing; recovery required")
        if required:
            raise RuntimeError("BF Watch must be remotely bootstrapped before market-data download")
        return []
    if is_repo:
        try:
            committed = subprocess.check_output(
                ["git", "-C", str(root), "show", f"origin/main:{WATCH_COMPLETE}"],
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("BF Watch is not confirmed on origin/main") from exc
        if source.read_bytes() != committed:
            raise RuntimeError("local BF Watch differs from published origin/main")
        # A failed BF run may have written a local COMPLETE Pool without ever
        # committing its Watch companion. Never download against that dirty
        # Pool while authenticating the OLD remote Watch by itself.
        pool_path = root / "us/breakout_follow_pool.csv"
        try:
            official_pool = subprocess.check_output(
                ["git", "-C", str(root), "show",
                 "origin/main:us/breakout_follow_pool.csv"],
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("complete BF Pool is not on origin/main") from exc
        if not pool_path.is_file() or pool_path.read_bytes() != official_pool:
            raise RuntimeError("local complete BF Pool differs from published origin/main")
        # Current authority must be the actual HEAD Pool. Historical discovery
        # deliberately skips malformed/older vintages, and therefore cannot
        # decide whether the *current* Pool is safe to use for downloads.
        import io
        from datetime import date
        from bf_snapshot import complete_snapshot_week, monday_of_week
        try:
            head_source, _ = pool_identity(pd.read_csv(
                io.BytesIO(official_pool), dtype={"code": str}, keep_default_na=False))
            complete_snapshot_week(date.fromisoformat(head_source))
        except (ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
            raise RuntimeError("latest published complete BF Pool invalid") from exc
        head_week = monday_of_week(date.fromisoformat(head_source)).isoformat()
    rows = parse_watch_csv(source)
    if is_repo:
        # The most recent valid historical Pool is NOT necessarily the HEAD
        # Pool: a malformed new revision or a retroactive rollback may
        # otherwise make stale Watch evidence look authoritative.
        latest = published_complete_pools(root, limit=1)
        if (not latest or latest[0]["week"] != head_week or
            latest[0]["snapshot_date"] != head_source):
            raise RuntimeError("latest published complete BF Pool history differs from HEAD")
        official = official_watch_for_week(root, head_week)
        if official is None or rows != official:
            raise RuntimeError("BF Watch is not paired to latest published complete Pool")
    return rows


def active_watch_codes(data_root: str | Path = ".", *, required: bool = False) -> list[str]:
    """All active official Watch codes, including held ones."""
    from market_universe import _normalize_code
    return sorted({code for row in _validated_watch_rows(data_root, required=required)
                   if row["phase"] in ACTIVE_PHASES
                   if (code := _normalize_code(row["code"])) is not None})


def held_watch_codes(data_root: str | Path = ".", *, required: bool = False) -> list[str]:
    """Manual holds in official complete active cycles, normalized for downloads."""
    from market_universe import _normalize_code
    return sorted({code for row in _validated_watch_rows(data_root, required=required)
                   if row["phase"] in ACTIVE_PHASES and row["download_hold_reason"]
                   if (code := _normalize_code(row["code"])) is not None})


def published_watch_state(data_root: str | Path = ".") -> tuple[list[dict], str | None, str, str]:
    """Read a consistent published Pool/Watch pair, not the dirty working Pool."""
    import io
    import subprocess
    from bf_snapshot import complete_snapshot_week, monday_of_week
    root = Path(data_root)
    if not (root / ".git").exists():
        source = root / WATCH_COMPLETE
        return (parse_watch_csv(source) if source.exists() else []), None, "", ""
    def git_bytes(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(root), *args])
    try:
        pinned_ref = git_bytes("rev-parse", "origin/main").decode().strip()
        raw = git_bytes("show", f"{pinned_ref}:us/breakout_follow_pool.csv")
        pool = pd.read_csv(io.BytesIO(raw), dtype={"code": str}, keep_default_na=False)
        source, digest = pool_identity(pool)
        complete_snapshot_week(date.fromisoformat(source))
        week = monday_of_week(date.fromisoformat(source)).isoformat()
        rows = official_watch_for_week(root, week, ref=pinned_ref)
        if rows is None:
            raise RuntimeError("latest published Pool/Watch is not paired")
        version = git_bytes("log", "-1", "--format=%H", pinned_ref,
                            "--", WATCH_COMPLETE).decode().strip()
        if not version:
            raise RuntimeError("official Watch has no Git version")
        return rows, version, source, digest
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError) as exc:
        raise RuntimeError("cannot read paired official Watch from origin/main") from exc


def inherit_download_holds(frame: pd.DataFrame, published: list[dict]) -> pd.DataFrame:
    """Only identical active cycles inherit, including an empty manual restore."""
    previous = {(r["code"], str(r["watch_cycle_id"])): r
                for r in published if r["phase"] in ACTIVE_PHASES}
    output = frame.copy()
    output["download_hold_reason"] = [
        previous.get((str(r["code"]), str(r["watch_cycle_id"])), {})
                .get("download_hold_reason", "")
        if r["phase"] in ACTIVE_PHASES else ""
        for r in output.to_dict("records")
    ]
    return output


def watch_frame(rows: list[dict], *, snapshot_date: str, pool_codes_hash: str) -> pd.DataFrame:
    """Canonical minimal public data; row count may legally be zero."""
    if not snapshot_date or not pool_codes_hash:
        raise ValueError("Watch requires source Pool identity")
    output = []
    for row in rows:
        rendered = {name: row.get(name, "") for name in COLUMNS}
        rendered["snapshot_date"] = snapshot_date
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
    if (pool.code.isna().any() or
        any(not code or code.strip() != code or code.lower() == "nan"
            for code in codes) or len(codes) != len(set(codes))):
        raise ValueError("invalid/duplicate Pool codes")
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
            frame = pd.read_csv(io.BytesIO(data), dtype={"code": str}, keep_default_na=False)
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


def official_watch_for_week(data_root: str | Path, week: str, *,
                            ref: str = "origin/main") -> list[dict] | None:
    """Find the latest published Pool/Watch pair, including unchanged empty CSV."""
    import subprocess
    import io
    from datetime import date
    from bf_snapshot import complete_snapshot_week, monday_of_week

    root = Path(data_root)
    refs = subprocess.check_output([
        "git", "-C", str(root), "log", ref, "--first-parent",
        "--format=%H", "--", "us/breakout_follow_pool.csv", WATCH_COMPLETE
    ], text=True)
    for ref in refs.splitlines():
        try:
            raw = subprocess.check_output(["git", "-C", str(root), "show",
                                           f"{ref}:us/breakout_follow_pool.csv"])
            source, digest = pool_identity(pd.read_csv(io.BytesIO(raw), dtype={"code": str}, keep_default_na=False))
            complete_snapshot_week(date.fromisoformat(source))
            if monday_of_week(date.fromisoformat(source)).isoformat() != week:
                continue
            message = subprocess.check_output(["git", "-C", str(root), "show",
                                               "-s", "--format=%B", ref], text=True)
            if (f"Watch-Source-Date: {source}" not in message or
                f"Watch-Codes-SHA256: {digest}" not in message):
                return None
            blob = subprocess.check_output(["git", "-C", str(root), "show",
                                            f"{ref}:{WATCH_COMPLETE}"])
            frame = pd.read_csv(io.BytesIO(blob), dtype={"code": str, "download_hold_reason": str}, keep_default_na=False)
            rows = parse_watch_frame(frame)
            if any(row["phase"] == "PROVISIONAL_OUT" for row in rows):
                return None
            if any(row["snapshot_date"] != source for row in rows):
                return None
            return rows
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
            return None
    return None
