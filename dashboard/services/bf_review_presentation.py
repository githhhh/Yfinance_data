"""Read-only presentation of buy-point observation facts.

The original A (Weekend Buy Point) observation/alert functions in
bf_observation.py must remain AST-identical to main. This adapter is the only
place the public Dashboard adds source-specific labels and V2 B identity.
Nothing here is used by Futu or the Telegram v1/v2 event producers.
"""
from __future__ import annotations

import math
import pandas as pd

from dashboard.services.bf_observation_replay import identity_v2, price_replay, zone as classify_zone
from dashboard.services.bf_transition import to_bool


def _observed_date(frame: pd.DataFrame):
    if frame is None or frame.empty or "snapshot_date" not in frame.columns:
        return None
    parsed = pd.to_datetime(frame["snapshot_date"], errors="coerce").dropna().dt.date.unique()
    return parsed[0] if len(parsed) == 1 else None


def _event_for_verified_b(
    current, previous, now_identity, previous_identity, *,
    has_previous_snapshot: bool, first_midweek_day: bool,
    weekend_identity: dict | None = None,
):
    # No prior published snapshot on Tuesday-Thursday is not evidence of a
    # NEW event. The B still belongs to this week's snapshot collection.
    if not has_previous_snapshot:
        return "NEW BP" if first_midweek_day else "SNAPSHOT"
    if previous is None or to_bool(previous.get("signal")) is not True:
        return "NEW BP"
    if previous_identity["resolution"] != "EXACT":
        return "UNVERIFIED"
    if (weekend_identity and weekend_identity.get("resolution") == "EXACT"
            and previous_identity["key"] == weekend_identity["key"]
            and now_identity["key"] != weekend_identity["key"]):
        return "NEW BP"
    if previous_identity["key"] != now_identity["key"]:
        return "BP CHANGED"

    previous_trigger, current_trigger = previous_identity["trigger"], now_identity["trigger"]
    if (previous_trigger is None or current_trigger is None or not
            math.isclose(previous_trigger, current_trigger, rel_tol=1e-9, abs_tol=1e-4)):
        return "BP UPDATED"
    old_status = str(previous.get("ibd_entry_status") or "")
    new_status = str(current.get("ibd_entry_status") or "")
    if old_status != new_status:
        return f"{old_status} → {new_status}"
    p = price_replay(
        trigger=current.get("ibd_candidate_price"),
        previous_close=previous.get("latest_close"),
        current_close=current.get("latest_close"),
        previous_date=str(previous.get("snapshot_date") or "")[:10],
        current_date=str(current.get("snapshot_date") or "")[:10],
        basis="TRACKED_ANCHOR",
        previous_trigger=previous.get("ibd_candidate_price"),
    )
    translation = {"BELOW_BUY_POINT":"BELOW", "BUY_ZONE":"ZONE", "EXTENDED":"EXT"}
    if p["transition"]:
        before, after = p["transition"].split("_TO_")
        return f"{translation.get(before, before)} → {translation.get(after, after)}"
    return "CURRENT"


def decorate_observation_view(
    observation: pd.DataFrame, *,
    complete: pd.DataFrame,
    current: pd.DataFrame,
    previous: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add UI-only source identity, event and comparison labels.

    One Ticker remains one row. The independent Weekend and Midweek Buy Point
    facts are preserved in different fields; publication and Futu are untouched.
    """
    if observation.empty:
        return observation
    result = observation.copy()
    weekend = {str(r["code"]): r for _, r in complete.iterrows()} if not complete.empty else {}
    today = {str(r["code"]): r for _, r in current.iterrows()} if not current.empty else {}
    prior = {str(r["code"]): r for _, r in previous.iterrows()} if previous is not None and not previous.empty else {}
    current_day, prior_day = _observed_date(current), _observed_date(previous)
    prior_valid = bool(current_day and prior_day and prior_day < current_day
                       and current_day.isocalendar()[:2] == prior_day.isocalendar()[:2])
    for index, row in result.iterrows():
        code = str(row["code"])
        has_a = bool(row.get("review_a_active"))
        a_event = str(row.get("review_a_event") or "")
        a_week = str(row.get("review_a_week_event") or "")
        result.at[index, "review_a_focus_basis"] = (
            "PREVIOUS_SNAPSHOT" if a_event and prior_valid
            else "PRIOR_SESSION" if a_event else "WEEKEND" if a_week else None
        )
        result.at[index, "review_b_verified"] = False
        result.at[index, "review_b_identity_state"] = "NONE"
        result.at[index, "review_b_display_event"] = ""
        result.at[index, "review_b_display_basis"] = None

        fresh = today.get(code)
        if fresh is None or to_bool(fresh.get("signal")) is not True:
            continue
        new_id = identity_v2(fresh)
        old_a = weekend.get(code) if has_a else None
        a_id = identity_v2(old_a) if old_a is not None else {"resolution":"NONE","key":None}
        # Same buy point as the fixed weekend anchor: it is A, not a second B.
        if new_id["resolution"] == "EXACT" and a_id.get("key") == new_id.get("key"):
            result.at[index, "review_b_active"] = False
            result.at[index, "review_new_buy_point"] = False
            result.at[index, "review_new_buy_point_zone"] = None
            result.at[index, "review_b_has_value_change"] = False
            result.at[index, "review_has_value_change"] = bool(row.get("review_a_has_value_change"))
            continue

        # A v1 may collapse a pivot S_BOX and M_BOX with the same date.
        # Restore the independent B observation using current, not A, facts.
        if not bool(row.get("review_b_active")) and (
            not has_a or new_id["resolution"] == "EXACT" and a_id.get("key")
            and new_id.get("key") != a_id.get("key")
        ):
            result.at[index, "review_b_active"] = True
            result.at[index, "review_b_price"] = fresh.get("ibd_candidate_price")
            result.at[index, "review_b_close"] = fresh.get("latest_close")
            result.at[index, "review_b_status"] = fresh.get("ibd_entry_status")
            result.at[index, "review_b_rule"] = fresh.get("ibd_candidate_rule")
            result.at[index, "review_b_date"] = fresh.get("snapshot_date")
            result.at[index, "review_b_entry_valid"] = to_bool(fresh.get("ibd_entry_valid"))
            result.at[index, "review_b_entry_volume"] = fresh.get("ibd_entry_volume_ratio")
            result.at[index, "review_b_entry_reason"] = fresh.get("ibd_entry_reject_reason")
            result.at[index, "review_b_quality"] = fresh.get("ibd_breakout_quality")
            result.at[index, "review_b_details"] = {key: fresh.get(key) for key in (
                "ibd_entry_date","ibd_entry_close_position","ibd_entry_breakout_range_ratio",
                "base_depth_pct","base_duration_weeks","ceiling","ceiling_date",
                "breakout_date","pullback_pct","pullback_duration_weeks",
                "pullback_peak_date","pullback_peak_price","pullback_v_is_dry",
                "eps_yoy_growth","dist_to_52w_high_pct","volume_ratio"
            )}
            trigger, close = pd.to_numeric(fresh.get("ibd_candidate_price"), errors="coerce"), pd.to_numeric(fresh.get("latest_close"),errors="coerce")
            if pd.notna(trigger) and pd.notna(close) and trigger > 0 and close > 0:
                pct = round((close / trigger - 1) * 100, 2)
                result.at[index, "review_b_pct"] = pct
                zone = "BELOW_BUY_POINT" if pct < 0 else "BUY_ZONE" if pct <= 5 else "EXTENDED"
                result.at[index, "review_b_zone"] = {"BELOW_BUY_POINT":"BELOW", "BUY_ZONE":"ZONE", "EXTENDED":"EXT"}[zone]
        if not bool(result.at[index,"review_b_active"]):
            continue
        identity_exact = new_id["resolution"] == "EXACT"
        weekend_distinct = (not has_a or
                            a_id["resolution"] == "EXACT" and
                            a_id.get("key") != new_id.get("key"))
        verified = identity_exact and weekend_distinct and not complete.empty
        result.at[index, "review_b_identity_state"] = (
            new_id["resolution"] if new_id["resolution"] != "EXACT"
            else "UNVERIFIED_WEEKEND" if has_a and a_id["resolution"] != "EXACT"
            else "NO_WEEKEND_BASELINE" if complete.empty
            else new_id["resolution"]
        )
        result.at[index, "review_b_display_basis"] = (
            "PREVIOUS_SNAPSHOT" if prior_valid else
            "WEEKEND" if current_day and current_day.weekday() == 0 and not complete.empty
            else "UNAVAILABLE"
        )
        event = "UNVERIFIED"
        if verified:
            previous_row = prior.get(code) if prior_valid else None
            previous_identity = (
                identity_v2(previous_row)
                if previous_row is not None and to_bool(previous_row.get("signal")) is True
                else {"resolution": "NONE", "key": None}
            )
            event = _event_for_verified_b(
                fresh, previous_row, new_id, previous_identity,
                has_previous_snapshot=prior_valid,
                first_midweek_day=bool(current_day and current_day.weekday() == 0),
                weekend_identity=a_id if has_a else None,
            )
        verified = verified and event != "UNVERIFIED"
        result.at[index, "review_b_verified"] = bool(verified)
        result.at[index, "review_b_display_event"] = event
        # Always derive the B zone from B's own authoritative trigger / close,
        # including a B that V2 recovered after V1 suppressed it.
        observed_zone = classify_zone(fresh.get("ibd_candidate_price"), fresh.get("latest_close"))
        enum_zone = {"BELOW": "BELOW_BUY_POINT", "ZONE": "BUY_ZONE", "EXT": "EXTENDED"}
        result.at[index, "review_b_zone"] = observed_zone
        result.at[index, "review_new_buy_point"] = bool(verified and observed_zone)
        result.at[index, "review_new_buy_point_zone"] = (
            enum_zone.get(observed_zone) if verified else None
        )
        changed = verified and event not in {"CURRENT", "SNAPSHOT", ""}
        result.at[index, "review_b_has_value_change"] = bool(changed)
        result.at[index, "review_has_value_change"] = bool(
            result.at[index, "review_a_has_value_change"] or changed
        )
    return result
