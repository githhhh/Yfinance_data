"""Read-only, buy-point-centric Midweek Review projection.

Trade eligibility stays in bf_transition._project; this module must not be
used to build Futu ACTIONABLE codes. No persisted event/price cache is needed.
"""
from __future__ import annotations

from datetime import date, datetime
from math import isfinite
from pathlib import Path
import pickle
import subprocess
from io import StringIO

import pandas as pd

from bf_snapshot import monday_of_week
from dashboard.services.bf_transition import (
    ENTRY_STATUSES, SOURCE_FACT_FIELDS, _calculate_status, _price_zone,
    _same_buy_point_price, to_bool,
)

_ZONES = {"BELOW_BUY_POINT": "BELOW", "BUY_ZONE": "ZONE", "EXTENDED": "EXT"}
_CHANGES = {"BUY_ZONE": "ENTERED_BUY_ZONE", "BELOW_BUY_POINT": "BELOW_BUY_POINT",
            "EXTENDED": "BECAME_EXTENDED"}


def _number(value):
    try:
        result = float(value)
        return result if pd.notna(result) and result > 0 else None
    except (ValueError, TypeError):
        return None


def _day(value):
    try:
        parsed = pd.Timestamp(value)
        return parsed.date() if pd.notna(parsed) else None
    except (ValueError, TypeError):
        return None


def _zone(price, close):
    return _price_zone(price, close) if _number(price) and _number(close) else None


def _arrow(before, after):
    if not before or not after or before == after:
        return ""
    return f"{_ZONES[before]} → {_ZONES[after]}"


def load_daily_snapshot(data_root: str | Path, *, window_date: date) -> dict:
    """Read the newest available 1D PKL not newer than the requested build day.

    Missing data is normal: observations remain in the review as NO DATA.
    Only trusted project-local PKL files should ever be used here.
    """
    folder = Path(data_root) / "results_pkl"
    choices = []
    for path in folder.glob("stock_data_*_1d.pkl"):
        try:
            file_date = datetime.strptime(path.stem.removeprefix("stock_data_").removesuffix("_1d"), "%d%m%y").date()
        except ValueError:
            continue
        if file_date <= window_date:
            choices.append((file_date, path))
    if not choices:
        return {}
    try:
        with max(choices)[1].open("rb") as handle:
            raw = pickle.load(handle)  # trusted repository-owned asset
    except (OSError, ValueError, EOFError, pickle.UnpicklingError):
        return {}
    if not isinstance(raw, dict):
        return {}
    result = {}
    for code, frame in raw.items():
        if isinstance(frame, dict) and {"index", "columns", "data"} == set(frame):
            try:
                frame = pd.DataFrame(**frame)
            except (TypeError, ValueError):
                continue
        if isinstance(frame, pd.DataFrame) and "Close" in frame.columns and not frame.empty:
            result[str(code)] = frame
    return result


def load_previous_midweek(data_root: str | Path, current_date: date) -> pd.DataFrame | None:
    """Use an earlier committed version of the midweek CSV, not a new DB.

    A shallow/non-git checkout simply disables yesterday-diff labels.
    """
    path = "us/breakout_follow_pool_midweek.csv"
    try:
        history = subprocess.run(
            ["git", "log", f"--since={monday_of_week(current_date).isoformat()}T00:00:00",
             "--format=%H", "--", path],
            cwd=data_root, capture_output=True, text=True, check=True, timeout=10,
        ).stdout.splitlines()
        for sha in history:
            raw = subprocess.run(
                ["git", "show", f"{sha}:{path}"],
                cwd=data_root, capture_output=True, text=True, check=True, timeout=10,
            ).stdout
            frame = pd.read_csv(StringIO(raw), dtype={"code": str})
            required = {"snapshot_date", "code", "signal"}
            if frame.empty or not required.issubset(frame.columns):
                return None
            parsed = pd.to_datetime(frame["snapshot_date"], errors="coerce")
            signals = frame["signal"].map(to_bool)
            if (parsed.isna().any() or frame["code"].isna().any()
                    or frame["code"].astype(str).str.strip().eq("").any()
                    or frame["code"].duplicated().any()
                    or signals.isna().any()):
                return None
            # Corrupted prior selected prices/statuses are not evidence of a
            # buy-point revision or a confirmation transition. Do not silently
            # skip one bad commit and compare against an older stale version.
            active = frame.loc[signals.eq(True)]
            if not active.empty:
                signal_fields = {
                    "ibd_candidate_price", "latest_close",
                    "ibd_entry_status", "ibd_entry_valid",
                }
                if not signal_fields.issubset(frame.columns):
                    return None
                if (active["ibd_entry_valid"].map(to_bool).isna().any()
                        or not active["ibd_entry_status"].isin(ENTRY_STATUSES).all()):
                    return None
                for field in ("ibd_candidate_price", "latest_close"):
                    prices = pd.to_numeric(active[field], errors="coerce")
                    if not prices.map(
                        lambda value: pd.notna(value) and
                        isfinite(float(value)) and float(value) > 0
                    ).all():
                        return None
            dates = parsed.dt.date.unique()
            if len(dates) != 1:
                return None
            previous_date = dates[0]
            if previous_date < current_date and monday_of_week(previous_date) == monday_of_week(current_date):
                return frame
    except (OSError, ValueError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired, pd.errors.ParserError):
        return None
    return None


def _price_path(weekend, current, daily, *, as_of: date, previous=None,
                previous_date: date | None = None):
    """Compare the latest two *observed* prices within one review week.

    Missing trading days do not prevent a transition. The complete-week
    baseline is used only for Vs Weekend, never as the previous midweek
    observation. Pool snapshots can fill holes in the 1D PKL, but neither
    source may manufacture prices for dates on which none was observed.
    """
    trigger = _number(weekend.get("ibd_candidate_price"))
    baseline_date = _day(weekend.get("snapshot_date"))
    baseline_close = _number(weekend.get("latest_close"))
    if trigger is None or baseline_date is None:
        return None

    week_start = monday_of_week(as_of)
    prices = {}
    if baseline_close is not None:
        prices[baseline_date] = baseline_close

    daily_dates = set()
    if isinstance(daily, pd.DataFrame) and "Close" in daily.columns:
        # Preserve the current fail-closed protection for split-rebased bars.
        baseline_bars = [
            _number(close) for ts, close in daily["Close"].items()
            if _day(ts) == baseline_date
        ]
        if baseline_close and baseline_bars and baseline_bars[-1] and abs(
            baseline_bars[-1] / baseline_close - 1
        ) > 0.02:
            return {"as_of": None, "close": None, "zone": None,
                    "baseline_zone": _zone(trigger, baseline_close),
                    "latest_event": "", "week_event": "", "path_changed": False,
                    "change": "UNCHANGED", "pct": None}
        for timestamp, close in daily["Close"].items():
            day = _day(timestamp)
            value = _number(close)
            if (day is not None and baseline_date < day <= as_of
                    and monday_of_week(day) == week_start and value is not None):
                prices[day] = value
                daily_dates.add(day)

    for snapshot in (previous, current):
        if snapshot is None:
            continue
        day = _day(snapshot.get("snapshot_date"))
        value = _number(snapshot.get("latest_close"))
        if (day is not None and baseline_date < day <= as_of
                and monday_of_week(day) == week_start and value is not None):
            # The actual 1D candle wins if both sources observe the same day.
            if day not in prices:
                prices[day] = value

    observations = sorted(
        (day, _zone(trigger, value), value)
        for day, value in prices.items()
        if day > baseline_date and monday_of_week(day) == week_start
    )
    if not observations:
        return {"as_of": None, "close": None, "zone": None,
                "baseline_zone": _zone(trigger, baseline_close),
                "latest_event": "", "week_event": "", "path_changed": False,
                "change": "UNCHANGED", "pct": None}

    latest_day, latest_zone, latest_close = observations[-1]
    base_zone = _zone(trigger, baseline_close)
    previous_zone = None
    if (previous_date is not None
            and monday_of_week(previous_date) == week_start):
        # The last successful midweek snapshot is the comparison boundary.
        # Intermediate PKL bars may exist, but must not change the baseline
        # of the user's last-versus-current increment.
        if previous_date < latest_day:
            previous_close = (
                _number(previous.get("latest_close"))
                if previous is not None
                and _day(previous.get("snapshot_date")) == previous_date
                else None
            )
            if previous_close is not None:
                previous_zone = _zone(trigger, previous_close)
            else:
                earlier = [z for z in observations if z[0] <= previous_date]
                if earlier:
                    previous_zone = earlier[-1][1]
    elif len(observations) >= 2:
        # No published prior snapshot: use the last genuine in-week bar.
        previous_zone = observations[-2][1]
    # No consecutive-session requirement or backfill of missing prices.
    latest_event = _arrow(previous_zone, latest_zone)
    week_event = _arrow(base_zone, latest_zone)
    observed_path = []
    path_zones = (
        [(baseline_date, base_zone, baseline_close)] + observations
        if daily_dates else observations
    )
    for _, zone, _ in path_zones:
        if zone and (not observed_path or zone != observed_path[-1]):
            observed_path.append(zone)
    short_path = " → ".join(_ZONES[zone] for zone in observed_path)
    focus_from = previous_zone if latest_event else base_zone
    focus_change = (_CHANGES.get(latest_zone, "UNCHANGED")
                    if focus_from != latest_zone else "UNCHANGED")
    history_changed = len(observed_path) > 1
    return {"as_of": latest_day.isoformat(), "close": latest_close,
            "zone": latest_zone, "baseline_zone": base_zone,
            "latest_event": latest_event, "week_event": week_event,
            "path_changed": history_changed, "path": short_path,
            "change": focus_change,
            "pct": round((latest_close / trigger - 1) * 100, 2)}


def _same_identity(a, b):
    """Compatibility name: compare the selected buy-point prices only."""
    if a is None or b is None:
        return None
    return _same_buy_point_price(a.get("ibd_candidate_price"), b.get("ibd_candidate_price"))


def _b_event(weekend, today, previous, *, comparison_available):
    if today is None or to_bool(today.get("signal")) is not True:
        return None
    if weekend is not None and _same_identity(weekend, today) is True:
        return None
    if not comparison_available:
        return "SNAPSHOT"
    if previous is None or to_bool(previous.get("signal")) is not True:
        return "NEW BP"
    # The preceding snapshot may have matched A, so no separate B existed.
    if weekend is not None and _same_identity(weekend, previous) is True:
        return "NEW BP"
    same = _same_identity(previous, today)
    if same is False:
        return "BP CHANGED"
    old_status = str(previous.get("ibd_entry_status") or "")
    new_status = str(today.get("ibd_entry_status") or "")
    if old_status != new_status:
        return f"{old_status} → {new_status}"
    return _arrow(
        _zone(previous.get("ibd_candidate_price"), previous.get("latest_close")),
        _zone(today.get("ibd_candidate_price"), today.get("latest_close")),
    ) or "CURRENT"


def build_observation_view(
    complete: pd.DataFrame,
    current: pd.DataFrame,
    trade_review: pd.DataFrame,
    *, daily: dict | None = None,
    previous: pd.DataFrame | None = None,
    as_of: date,
) -> pd.DataFrame:
    """One ticker row; A primary, B secondary; no changes to trade_review."""
    complete_by_code = {str(r["code"]): r for _, r in complete.iterrows()}
    current_by_code = {str(r["code"]): r for _, r in current.iterrows()}
    prev_date = None
    if previous is not None and not previous.empty:
        days = pd.to_datetime(previous["snapshot_date"], errors="coerce").dropna().dt.date.unique()
        if len(days) == 1:
            prev_date = days[0]
    current_dates = pd.to_datetime(current["snapshot_date"], errors="coerce").dropna().dt.date.unique() if not current.empty else []
    current_date = current_dates[0] if len(current_dates) == 1 else None
    prev_valid = (prev_date is not None and current_date is not None and
                  prev_date < current_date and monday_of_week(prev_date) == monday_of_week(current_date))
    prev_by_code = {str(r["code"]): r for _, r in previous.iterrows()} if prev_valid else {}
    trade_rows = {str(r["code"]): r.to_dict() for _, r in trade_review.iterrows()}
    all_codes = set(trade_rows)
    all_codes.update(code for code, r in complete_by_code.items() if to_bool(r.get("signal")) is True)
    output = []
    for code in sorted(all_codes):
        weekend = complete_by_code.get(code)
        if weekend is not None and to_bool(weekend.get("signal")) is not True:
            weekend = None
        midweek = current_by_code.get(code)
        row = dict(trade_rows.get(code, weekend.to_dict() if weekend is not None else {}))
        row["code"] = code
        membership = "IN" if midweek is not None else "OUT"
        prev_membership = "IN" if code in prev_by_code else "OUT"
        membership_event = (f"{prev_membership} → {membership}" if prev_valid and
                            membership != prev_membership else "")
        row["review_membership"] = membership
        row["review_membership_event"] = membership_event
        row["review_a_active"] = weekend is not None
        row["review_b_active"] = False
        row["review_b_event"] = ""
        row["review_b_price"] = None
        row["review_b_status"] = None
        row["review_b_entry_valid"] = None
        row["review_b_rule"] = None
        row["review_b_date"] = None
        row["review_b_zone"] = None
        row["review_b_close"] = None
        row["review_b_quality"] = None
        row["review_b_entry_volume"] = None
        row["review_b_entry_reason"] = None
        row["review_b_details"] = None
        row["review_b_pct"] = None
        row["review_a_has_value_change"] = False
        row["review_b_has_value_change"] = False
        row["review_a_as_of"] = None
        row["review_a_zone"] = None
        row["review_a_event"] = ""
        row["review_a_week_event"] = ""
        row["review_a_path"] = ""
        row["review_a_price"] = None
        row["review_a_pct"] = None
        row["review_data_available"] = True
        a_change = "UNCHANGED"
        a_changed = False
        if weekend is not None:
            row["review_watch_active"] = True
            row["signal"] = True
            for field in SOURCE_FACT_FIELDS:
                row[field] = weekend.get(field)
            # A's structural review facts must not silently inherit B's
            # different setup when current midweek has a new trigger.
            for field in (
                "base_depth_pct", "base_duration_weeks", "ceiling",
                "ceiling_date", "breakout_date", "pullback_pct",
                "pullback_duration_weeks", "pullback_peak_date",
                "pullback_peak_price", "pullback_v_is_dry",
            ):
                row[field] = weekend.get(field)
            row["ibd_breakout_quality"] = weekend.get("ibd_breakout_quality")
            if to_bool(weekend.get("ibd_entry_valid")) is not True:
                row["ibd_entry_vol_or_reject"] = str(weekend.get("ibd_entry_reject_reason") or "Volume not confirmed")
            else:
                volume = _number(weekend.get("ibd_entry_volume_ratio"))
                row["ibd_entry_vol_or_reject"] = f"{volume:.2f}x" if volume else "n/a"
            trigger = _number(weekend.get("ibd_candidate_price"))
            row["review_a_price"] = trigger
            path = _price_path(
                weekend, midweek, (daily or {}).get(code), as_of=as_of,
                previous=prev_by_code.get(code),
                previous_date=prev_date if prev_valid else None,
            )
            if path is None or path["close"] is None:
                row["review_data_available"] = False
                row["ibd_entry_status"] = None
                row["latest_close"] = None
                row["current_vs_ibd_candidate_pct"] = None
                row["review_change_label"] = "NO DATA"
            else:
                close = path["close"]
                row["latest_close"] = close
                row["current_vs_ibd_candidate_pct"] = path["pct"]
                row["ibd_entry_status"] = _calculate_status(weekend.get("ibd_entry_valid"), trigger, close)
                row["review_a_pct"] = path["pct"]
                row["review_a_as_of"] = path["as_of"]
                row["review_a_zone"] = _ZONES.get(path["zone"])
                row["review_a_event"] = path["latest_event"]
                row["review_a_week_event"] = path["week_event"]
                row["review_a_path"] = path["path"]
                a_change = path["change"]
                a_changed = bool(path["week_event"] or path["latest_event"])
                focus = path["latest_event"] or path["week_event"] or ""
                membership_label = membership_event or ("OUT" if membership == "OUT" else "")
                row["review_change_label"] = " · ".join(
                    part for part in (focus, membership_label, row["review_a_zone"] if membership_event and not focus else "") if part
                )
            row["review_weekend_buy_point_change"] = a_change
            row["review_futu_actionable"] = trade_rows.get(code, {}).get("review_futu_actionable", False)
        event = _b_event(weekend, midweek, prev_by_code.get(code),
                         comparison_available=prev_valid or not complete.empty)
        # A's anchor may coincide with B; that is not a new buy point.
        if event:
            bp = _number(midweek.get("ibd_candidate_price"))
            zone = _zone(bp, midweek.get("latest_close"))
            row["review_b_active"] = True
            row["review_b_event"] = event
            row["review_b_price"] = bp
            row["review_b_status"] = midweek.get("ibd_entry_status")
            row["review_b_entry_valid"] = to_bool(midweek.get("ibd_entry_valid"))
            row["review_b_rule"] = midweek.get("ibd_candidate_rule")
            row["review_b_date"] = midweek.get("snapshot_date")
            row["review_b_close"] = _number(midweek.get("latest_close"))
            row["review_b_quality"] = midweek.get("ibd_breakout_quality")
            row["review_b_entry_volume"] = _number(midweek.get("ibd_entry_volume_ratio"))
            row["review_b_entry_reason"] = midweek.get("ibd_entry_reject_reason")
            row["review_b_details"] = {field: midweek.get(field) for field in (
                "ibd_entry_date", "ibd_entry_close_position",
                "ibd_entry_breakout_range_ratio", "base_depth_pct",
                "base_duration_weeks", "ceiling", "ceiling_date",
                "breakout_date", "pullback_pct", "pullback_duration_weeks",
                "pullback_peak_date", "pullback_peak_price", "pullback_v_is_dry",
                "eps_yoy_growth", "dist_to_52w_high_pct", "volume_ratio",
            )}
            row["review_b_zone"] = _ZONES.get(zone)
            row["review_b_pct"] = (round((_number(midweek.get("latest_close")) / bp - 1) * 100, 2)
                                   if bp and _number(midweek.get("latest_close")) else None)
            row["review_new_buy_point"] = True
            row["review_new_buy_point_zone"] = zone
            if weekend is None:
                row["review_change_label"] = event
        else:
            row["review_new_buy_point"] = False
            row["review_new_buy_point_zone"] = None
        row["review_a_has_value_change"] = bool(a_changed or (membership_event and weekend is not None))
        row["review_b_has_value_change"] = bool(event and event not in {"CURRENT", "SNAPSHOT"})
        row["review_has_value_change"] = bool(
            row["review_a_has_value_change"] or row["review_b_has_value_change"]
        )
        row["review_buy_point_change_label"] = row.get("review_change_label") or ""
        row["review_buy_point_priority"] = (
            0 if a_change == "ENTERED_BUY_ZONE" else
            1 if a_change == "BELOW_BUY_POINT" else
            2 if a_change == "BECAME_EXTENDED" else
            3 if event and event != "CURRENT" else 9
        )
        # Existing weekend rows may no longer be in the current pool: they
        # remain visible for observation, never automatically enter Futu.
        output.append(row)
    return pd.DataFrame(output)


def build_observation_attention(
    current: pd.DataFrame,
    complete: pd.DataFrame,
    trade_review: pd.DataFrame,
    *, previous: pd.DataFrame | None,
    daily: dict | None,
    as_of: date,
) -> list[dict]:
    """Small, date-grounded events for Telegram; no persistent state.

    Pool membership transitions require the previous actual pool snapshot;
    price transitions are replayed independently from daily K lines.
    """
    view = build_observation_view(
        complete, current, trade_review, previous=previous, daily=daily, as_of=as_of,
    )
    previous_date = None
    if previous is not None and not previous.empty and "snapshot_date" in previous:
        dates = pd.to_datetime(previous["snapshot_date"], errors="coerce").dropna().dt.date.unique()
        if len(dates) == 1 and dates[0] < as_of and monday_of_week(dates[0]) == monday_of_week(as_of):
            previous_date = dates[0]
    events = []
    actionable_statuses = {"ACTIONABLE", "UNCONFIRMED"}
    for _, row in view.iterrows():
        code = str(row["code"])
        if bool(row.get("review_a_active")) and bool(row.get("review_data_available")):
            # Telegram is an entry-opportunity alert, not an A lifecycle log.
            # Only a *real* return to the fixed weekend buy zone is useful.
            event = str(row.get("review_a_event") or "")
            status = str(row.get("ibd_entry_status") or "")
            zone = str(row.get("review_a_zone") or "")
            observed = _day(row.get("review_a_as_of"))
            if (
                event in {"BELOW → ZONE", "EXT → ZONE"}
                and zone == "ZONE"
                and status in actionable_statuses
                and (previous_date is None or (observed is not None and observed > previous_date))
            ):
                events.append({
                    "source": "WEEKEND", "code": code, "event": event,
                    "zone": zone, "status": status,
                    "membership": row.get("review_membership"),
                    "vs_candidate_pct": row.get("review_a_pct"),
                    "as_of": row.get("review_a_as_of"),
                })
        if bool(row.get("review_b_active")):
            # A new independent buy-point anchor is a NEW opportunity,
            # but a changed price for the same anchor (BP UPDATED) is not.
            event = str(row.get("review_b_event") or "")
            status = str(row.get("review_b_status") or "")
            zone = str(row.get("review_b_zone") or "")
            # ACT is a confirmed *entry*, never an alias for price merely
            # being in ZONE. Fail closed if status and resolver validity disagree.
            if status == "ACTIONABLE" and to_bool(row.get("review_b_entry_valid")) is not True:
                status = "UNCONFIRMED"
            if (
                event in {"NEW BP", "BP CHANGED"}
                and zone == "ZONE"
                and status in actionable_statuses
            ):
                events.append({
                    "source": "MIDWEEK", "code": code, "event": "NEW BP",
                    "zone": zone, "status": status,
                    "vs_candidate_pct": row.get("review_b_pct"),
                    "as_of": row.get("review_b_date"),
                })
    return events


# V2 intentionally supplements the existing A/B projection. Neither
# build_observation_view nor the v1 A admission contract is rewritten.
def load_observation_history(data_root, *, as_of, previous=None, run_cutoff=None,
                             require_previous=True):
    """Read only previously committed, same-week MIDWEEK Pool snapshots.

    No fetch, no write and no assumption that the user's live checkout has
    complete Git history. A captured previous snapshot is trusted as a
    publication boundary only when it matches a committed snapshot.
    """
    from datetime import datetime, timedelta, timezone
    from bf_snapshot import _is_market_trading_day
    run_cutoff = run_cutoff or datetime.now(timezone.utc)
    if run_cutoff.tzinfo is None:
        raise ValueError("run_cutoff must be timezone-aware")
    root = Path(data_root)
    path = "us/breakout_follow_pool_midweek.csv"

    def unavailable(reason):
        return [], {"usable": False, "history_coverage": "UNAVAILABLE",
                    "reason": reason, "snapshot_dates": []}

    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True,
                              text=True, check=True, timeout=20).stdout.strip()

    try:
        if git("rev-parse", "--is-inside-work-tree") != "true":
            return unavailable("DATA_ROOT is not a git checkout")
        if git("rev-parse", "--is-shallow-repository") != "false":
            return unavailable("shallow git checkout")
        # Commit timestamps follow *publication* time, not EOD market_date.
        # Scan only the relevant week, without an arbitrary last-N truncation.
        earliest = monday_of_week(as_of) - timedelta(days=3)
        commits = git("log", f"--since={earliest.isoformat()}T00:00:00",
                      "--format=%H%x09%cI", "--", path).splitlines()
        by_day = {}
        for commit in commits:
            sha, published_at = commit.split("\t", 1)
            when = datetime.fromisoformat(published_at)
            if when > run_cutoff:
                continue
            raw = git("show", f"{sha}:{path}")
            frame = pd.read_csv(StringIO(raw), dtype={"code": str})
            required = {"snapshot_date", "code", "signal", "latest_close"}
            if not required.issubset(frame.columns):
                return unavailable("history file lacks required columns")
            parsed_dates = pd.to_datetime(frame["snapshot_date"], errors="coerce")
            if parsed_dates.isna().any() or frame["code"].isna().any() or (
                frame["code"].astype(str).str.strip().eq("")
            ).any() or frame["code"].duplicated().any():
                return unavailable("history contains invalid dates, codes or duplicates")
            days = parsed_dates.dt.date.unique()
            if len(days) != 1:
                return unavailable("mixed/invalid historical market date")
            # Invalid published truth values are not equivalent to signal=False.
            # Otherwise the replay could invent FIRST_OBSERVED_BP when a
            # previous historical B signal is corrupt or unparseable.
            if frame["signal"].map(to_bool).isna().any():
                return unavailable("historical signal contains invalid booleans")
            day = days[0]
            if not (monday_of_week(as_of) <= day < as_of):
                continue
            # Multiple edited versions of a date cannot silently be treated
            # as one published fact; don't choose a revision arbitrarily.
            signature = frame.sort_values("code").fillna("").astype(str).to_json(orient="records")
            if day in by_day:
                if by_day[day]["signature"] != signature:
                    return unavailable(f"ambiguous conflicting snapshot for {day}")
                continue
            by_day[day] = {"date": day.isoformat(), "frame": frame,
                           "source_ref": sha, "signature": signature}
        rows = [by_day[d] for d in sorted(by_day)]
        week_start = monday_of_week(as_of)
        # On the first trading day the fixed MIDWEEK file normally still
        # contains last week's old snapshot. It is not a same-week prior.
        previous_same_week = False
        if previous is not None and not previous.empty:
            if "snapshot_date" not in previous or "code" not in previous:
                return unavailable("captured previous snapshot missing required columns")
            dates = pd.to_datetime(previous["snapshot_date"], errors="coerce")
            if dates.isna().any() or len(set(dates.dt.date)) != 1:
                return unavailable("captured previous has mixed or invalid dates")
            previous_day = dates.dt.date.iloc[0]
            if previous_day >= as_of:
                return unavailable("captured previous is not older than current")
            if monday_of_week(previous_day) == week_start:
                previous_same_week = True
            elif previous_day >= week_start:
                return unavailable("captured previous is not a valid earlier snapshot")
        # Missing history after the first US trading session cannot
        # masquerade as an authoritative first B observation.
        prior_sessions = [week_start + timedelta(days=i)
                          for i in range((as_of - week_start).days)
                          if _is_market_trading_day(week_start + timedelta(days=i))]
        first_session = _is_market_trading_day(as_of) and not prior_sessions
        if not rows:
            if first_session and not previous_same_week:
                return [], {"usable": True, "history_coverage": "AVAILABLE",
                            "reason": "first market session, complete Pool baseline required",
                            "snapshot_dates": []}
            return unavailable("missing prior published midweek history")
        if require_previous and not previous_same_week:
            return unavailable("captured same-week Pool missing despite committed prior history")

        if require_previous and previous_same_week:
            prev_days = pd.to_datetime(previous["snapshot_date"], errors="coerce").dropna().dt.date.unique()
            if len(prev_days) != 1 or prev_days[0] >= as_of or monday_of_week(prev_days[0]) != monday_of_week(as_of):
                return unavailable("captured previous date is not in the review week")
            git_previous = by_day.get(prev_days[0])
            if git_previous is None:
                return unavailable("captured previous not in committed history")
            signature = previous.sort_values("code").fillna("").astype(str).to_json(orient="records")
            if signature != git_previous["signature"]:
                return unavailable("captured previous does not match committed version")
            # Previous must be the most recent published snapshot, not merely
            # any valid older snapshot.
            if prev_days[0].isoformat() != rows[-1]["date"]:
                return unavailable("captured previous is not latest committed midweek")
        result = [{"date": x["date"], "frame": x["frame"], "source_ref": x["source_ref"],
                   "verified": True, "origin": "MIDWEEK"} for x in rows]
        return result, {"usable": True, "history_coverage": "AVAILABLE",
                        "previous_boundary_verified": previous_same_week,
                        "reason": None, "snapshot_dates": [x["date"] for x in rows]}
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired, pd.errors.ParserError) as exc:
        return unavailable(f"history preflight failed: {exc}")


def build_observation_events_v2(current, complete, *, as_of, history_snapshots, diagnostic, daily=None):
    """Current B only; A continues through the unchanged v1 event path."""
    from dashboard.services.bf_observation_replay import observe_midweek_buy_point, zone as b_zone
    from bf_snapshot import is_valid_complete_baseline
    if not diagnostic.get("usable") or current.empty or complete.empty:
        return []
    required = {"code", "snapshot_date", "signal"}
    if not required.issubset(current.columns) or not required.issubset(complete.columns):
        return []
    current_dates = pd.to_datetime(current["snapshot_date"], errors="coerce")
    complete_dates = pd.to_datetime(complete["snapshot_date"], errors="coerce")
    if (current_dates.isna().any() or complete_dates.isna().any()
            or set(current_dates.dt.date) != {as_of}
            or len(set(complete_dates.dt.date)) != 1
            or current["code"].isna().any() or complete["code"].isna().any()
            or current["code"].duplicated().any() or complete["code"].duplicated().any()):
        return []
    if not is_valid_complete_baseline(complete_dates.dt.date.iloc[0], as_of):
        return []
    current_by_code = {str(r["code"]): r for _, r in current.iterrows()}
    weekend_by_code = {str(r["code"]): r for _, r in complete.iterrows()}
    history_maps = [{
        "date": snap["date"], "source_ref": snap["source_ref"],
        "origin": snap["origin"], "verified": snap["verified"],
        "rows": {str(r["code"]): r for _, r in snap["frame"].iterrows()}
    } for snap in history_snapshots]
    # Only current signal rows are B candidates; historical B is not a
    # persistent tracking list, and never influences Futu eligibility.
    results = []
    week = monday_of_week(as_of).isoformat()
    for code, current_row in current_by_code.items():
        if to_bool(current_row.get("signal")) is not True:
            continue
        history = [{**{k: v for k, v in item.items() if k != "rows"},
                    "row": item["rows"].get(code)} for item in history_maps]
        weekend = weekend_by_code.get(code)
        # The verified COMPLETE snapshot is also a historical price/watch
        # observation. A missing ticker remains row=None (not a signal).
        weekend_date = complete_dates.dt.date.iloc[0]
        history.insert(0, {"date": weekend_date.isoformat(),
                           "row": weekend, "verified": True,
                           "origin": "WEEKEND", "source_ref": "complete-baseline"})
        daily_closes = []
        frame = (daily or {}).get(code)
        if isinstance(frame, pd.DataFrame) and "Close" in frame:
            for timestamp, raw_close in frame["Close"].items():
                trade_day = _day(timestamp)
                observed_close = _number(raw_close)
                if (trade_day is not None and observed_close is not None
                        and trade_day <= as_of and
                        trade_day >= monday_of_week(as_of) - pd.Timedelta(days=3)):
                    daily_closes.append((trade_day.isoformat(), observed_close))
        # Missing row in a *full* published Pool is not a signal, but may
        # have an observable EOD price in the daily PKL. Keep provenance
        # separate: populate price_close without inventing a Pool row.
        # A 1D artifact must have one unambiguous close per market date.
        # Collapsing contradictory duplicate bars through dict() would
        # silently pick an arbitrary price and possibly reverse direction.
        daily_by_day = {}
        conflicting_daily_bar = False
        for trading_day, observed_close in daily_closes:
            prior_close = daily_by_day.get(trading_day)
            if prior_close is not None and abs(observed_close / prior_close - 1) > 1e-9:
                conflicting_daily_bar = True
                break
            daily_by_day[trading_day] = observed_close
        if conflicting_daily_bar:
            continue
        coherent = True
        active_trigger = current_row.get("ibd_candidate_price")
        for observation in history:
            row = observation.get("row")
            daily_close = daily_by_day.get(observation["date"])
            if row is None:
                observation["price_close"] = daily_close
            elif daily_close is not None:
                pool_close = _number(row.get("latest_close"))
                if pool_close is None:
                    # Retain a genuinely observed PKL close even if the
                    # published Pool row exists without a valid latest_close.
                    # This fills price evidence only, never signal/Watch.
                    observation["price_close"] = daily_close
                elif (
                    abs(daily_close / pool_close - 1) > 0.02
                    or b_zone(active_trigger, daily_close) != b_zone(active_trigger, pool_close)
                ):
                    coherent = False
                    break
        current_close = _number(current_row.get("latest_close"))
        current_daily_close = daily_by_day.get(as_of.isoformat())
        if current_close is not None and current_daily_close is not None:
            coherent = coherent and (
                abs(current_daily_close / current_close - 1) <= 0.02
                and b_zone(active_trigger, current_daily_close) ==
                b_zone(active_trigger, current_close)
            )
        if not coherent:
            continue
        event = observe_midweek_buy_point(
            code=code, current=current_row.to_dict(),
            history=history, weekend=weekend.to_dict() if weekend is not None else None,
            as_of=as_of.isoformat(), review_week=week,
            baseline_verified=True,
            history_coverage=diagnostic["history_coverage"],
            daily_closes=daily_closes,
        )
        if event is not None:
            results.append(event)
    return results
