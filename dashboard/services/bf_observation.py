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


def _price_path(weekend, current, daily, *, as_of: date):
    """Reconstruct A's zone path from its fixed complete-week trigger."""
    trigger = _number(weekend.get("ibd_candidate_price"))
    baseline_date = _day(weekend.get("snapshot_date"))
    baseline_close = _number(weekend.get("latest_close"))
    if trigger is None or baseline_date is None:
        return None
    points = []
    if baseline_close is not None:
        points.append((baseline_date, baseline_close))
    if isinstance(daily, pd.DataFrame) and "Close" in daily.columns:
        # Yahoo raw OHLC is split-adjusted. A later split may rebase the
        # complete-week bar but not the immutable Pool trigger. In that case,
        # refuse a false replay instead of reporting a fabricated breakout.
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
            val = _number(close)
            if day is not None and baseline_date < day <= as_of and val is not None:
                points.append((day, val))
    if current is not None and not any(day > baseline_date for day, _ in points):
        # Fallback only gives a present zone, never a fabricated daily path.
        fallback_date = _day(current.get("snapshot_date"))
        fallback_close = _number(current.get("latest_close"))
        if fallback_date and baseline_date < fallback_date <= as_of and fallback_close:
            points.append((fallback_date, fallback_close))
    # The complete-week close is the baseline, not evidence of live freshness.
    has_new_price = any(day > baseline_date for day, _ in points)
    if not has_new_price:
        return {"as_of": None, "close": None, "zone": None, "baseline_zone": _zone(trigger, baseline_close),
                "latest_event": "", "week_event": "", "path_changed": False,
                "change": "UNCHANGED", "pct": None}
    ordered = sorted(dict(points).items())
    zones = [(day, _zone(trigger, close), close) for day, close in ordered]
    latest_day, latest_zone, latest_close = zones[-1]
    base_zone = _zone(trigger, baseline_close)
    previous_zone = zones[-2][1] if len(zones) >= 2 else None
    latest_event = _arrow(previous_zone, latest_zone)
    week_event = _arrow(base_zone, latest_zone)
    history_changed = any(left[1] != right[1] for left, right in zip(zones, zones[1:]))
    focus_from = previous_zone if latest_event else base_zone
    focus_change = _CHANGES.get(latest_zone, "UNCHANGED") if focus_from != latest_zone else "UNCHANGED"
    return {"as_of": latest_day.isoformat(), "close": latest_close, "zone": latest_zone,
            "baseline_zone": base_zone, "latest_event": latest_event, "week_event": week_event,
            "path_changed": history_changed, "change": focus_change,
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
    return "CURRENT"


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
        row["review_b_rule"] = None
        row["review_b_date"] = None
        row["review_b_zone"] = None
        row["review_b_close"] = None
        row["review_b_quality"] = None
        row["review_b_entry_volume"] = None
        row["review_b_entry_reason"] = None
        row["review_b_pct"] = None
        row["review_a_as_of"] = None
        row["review_a_zone"] = None
        row["review_a_event"] = ""
        row["review_a_week_event"] = ""
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
            row["ibd_breakout_quality"] = weekend.get("ibd_breakout_quality")
            if to_bool(weekend.get("ibd_entry_valid")) is not True:
                row["ibd_entry_vol_or_reject"] = str(weekend.get("ibd_entry_reject_reason") or "Volume not confirmed")
            else:
                volume = _number(weekend.get("ibd_entry_volume_ratio"))
                row["ibd_entry_vol_or_reject"] = f"{volume:.2f}x" if volume else "n/a"
            trigger = _number(weekend.get("ibd_candidate_price"))
            row["review_a_price"] = trigger
            path = _price_path(weekend, midweek, (daily or {}).get(code), as_of=as_of)
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
            row["review_b_rule"] = midweek.get("ibd_candidate_rule")
            row["review_b_date"] = midweek.get("snapshot_date")
            row["review_b_close"] = _number(midweek.get("latest_close"))
            row["review_b_quality"] = midweek.get("ibd_breakout_quality")
            row["review_b_entry_volume"] = _number(midweek.get("ibd_entry_volume_ratio"))
            row["review_b_entry_reason"] = midweek.get("ibd_entry_reject_reason")
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
        row["review_has_value_change"] = bool(
            a_changed or membership_event and weekend is not None or
            event and event not in {"CURRENT", "SNAPSHOT"}
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
    for _, row in view.iterrows():
        code = str(row["code"])
        membership = str(row.get("review_membership_event") or "")
        if bool(row.get("review_a_active")) and bool(row.get("review_data_available")):
            price_event = str(row.get("review_a_event") or "")
            # A's cached last daily bar must not be notified again if no
            # new dated daily observation followed the previous snapshot.
            observed = _day(row.get("review_a_as_of"))
            if previous_date is not None and (observed is None or observed <= previous_date):
                price_event = ""
            zone = str(row.get("review_a_zone") or "")
            if price_event or membership:
                labels = [part for part in (membership, price_event or (zone if membership else "")) if part]
                events.append({
                    "source": "WEEKEND", "code": code, "event": " · ".join(labels),
                    "vs_candidate_pct": row.get("review_a_pct"),
                    "as_of": row.get("review_a_as_of"),
                })
        if bool(row.get("review_b_active")):
            event = str(row.get("review_b_event") or "")
            if event not in ("", "CURRENT", "SNAPSHOT"):
                labels = [event]
                if row.get("review_b_status"):
                    labels.append(str(row["review_b_status"]))
                events.append({
                    "source": "MIDWEEK", "code": code, "event": " · ".join(labels),
                    "vs_candidate_pct": row.get("review_b_pct"),
                    "as_of": row.get("review_b_date"),
                })
    return events
