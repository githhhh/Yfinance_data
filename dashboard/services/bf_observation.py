"""Read-only, buy-point-centric Midweek Review projection.

Trade eligibility stays in bf_transition._project; this module must not be
used to build Futu ACTIONABLE codes. No persisted event/price cache is needed.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import pickle
import subprocess
from io import StringIO

import pandas as pd

from bf_snapshot import monday_of_week
from dashboard.services.bf_transition import (
    SOURCE_FACT_FIELDS, _buy_point_anchor, _calculate_status, _price_zone, to_bool,
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
            ["git", "log", "-20", "--format=%H", "--", path],
            cwd=data_root, capture_output=True, text=True, check=True, timeout=10,
        ).stdout.splitlines()
        for sha in history:
            raw = subprocess.run(
                ["git", "show", f"{sha}:{path}"],
                cwd=data_root, capture_output=True, text=True, check=True, timeout=10,
            ).stdout
            frame = pd.read_csv(StringIO(raw), dtype={"code": str})
            dates = pd.to_datetime(frame.get("snapshot_date"), errors="coerce").dropna().dt.date.unique()
            if len(dates) != 1:
                continue
            previous_date = dates[0]
            if previous_date < current_date and monday_of_week(previous_date) == monday_of_week(current_date):
                return frame
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
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
    left = _buy_point_anchor(a) if a is not None else None
    right = _buy_point_anchor(b) if b is not None else None
    if left and right:
        return left["identity"] == right["identity"]
    # Missing source identity must not be invented from an approximate price.
    return None


def _b_event(weekend, today, previous, *, comparison_available):
    if today is None or to_bool(today.get("signal")) is not True:
        return None
    if weekend is not None and _same_identity(weekend, today) is True:
        return None
    if weekend is not None and _same_identity(weekend, today) is None:
        if (str(weekend.get("ibd_candidate_rule") or "") ==
                str(today.get("ibd_candidate_rule") or "")):
            return None  # fail closed on ambiguous same-rule anchors
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
    if same is None:
        return "SNAPSHOT"
    if _number(previous.get("ibd_candidate_price")) != _number(today.get("ibd_candidate_price")):
        return "BP UPDATED"
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
