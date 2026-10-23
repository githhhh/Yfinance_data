"""Read-only presentation of buy-point observation facts.

The original A (Weekend Buy Point) observation/alert functions in
bf_observation.py must remain AST-identical to main. This adapter is the only
place the public Dashboard adds source-specific labels and V2 B identity.
Nothing here is used by Futu or the Telegram v1/v2 event producers.
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from bf_snapshot import monday_of_week, _is_market_trading_day

from dashboard.services.bf_observation_replay import zone as classify_zone
from dashboard.services.bf_transition import _same_buy_point_price
from dashboard.services.bf_transition import to_bool


def _observed_date(frame: pd.DataFrame):
    if frame is None or frame.empty or "snapshot_date" not in frame.columns:
        return None
    parsed = pd.to_datetime(frame["snapshot_date"], errors="coerce").dropna().dt.date.unique()
    return parsed[0] if len(parsed) == 1 else None


def _first_trading_session(day):
    if not day or not _is_market_trading_day(day):
        return False
    start = monday_of_week(day)
    return not any(
        _is_market_trading_day(start + timedelta(days=i))
        for i in range((day - start).days)
    )


def _midweek_price_event(
    current, previous, *, has_previous_snapshot: bool,
    first_midweek_day: bool, weekend_price=None,
):
    # No prior published snapshot on Tuesday-Thursday is not evidence of a
    # NEW event. The B still belongs to this week's snapshot collection.
    if not has_previous_snapshot:
        return "NEW BP" if first_midweek_day else "SNAPSHOT"
    if previous is None or to_bool(previous.get("signal")) is not True:
        return "NEW BP"
    previous_price = previous.get("ibd_candidate_price")
    current_price = current.get("ibd_candidate_price")
    if weekend_price is not None and _same_buy_point_price(previous_price, weekend_price):
        return "NEW BP"
    if not _same_buy_point_price(previous_price, current_price):
        return "BP CHANGED"
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
        membership_event = str(row.get("review_membership_event") or "")
        # Show persistent Vs Weekend state in What Changed / All Review, but
        # do not repeatedly inject an old difference into Review Now.
        # On the first US trading session, comparison against Weekend is new.
        first_session = _first_trading_session(current_day)
        a_attention = bool(
            a_event or membership_event or (first_session and a_week)
        ) if has_a else False
        result.at[index, "review_a_has_value_change"] = a_attention
        result.at[index, "review_has_value_change"] = bool(
            a_attention or row.get("review_b_has_value_change")
        )
        result.at[index, "review_a_focus_basis"] = (
            "PREVIOUS_SNAPSHOT" if a_event and prior_valid and code in prior
            and pd.notna(pd.to_numeric(prior[code].get("latest_close"), errors="coerce"))
            else "PRIOR_SESSION" if a_event else "WEEKEND" if a_week else None
        )
        result.at[index, "review_b_display_event"] = ""
        result.at[index, "review_b_display_basis"] = None

        fresh = today.get(code)
        if fresh is None or to_bool(fresh.get("signal")) is not True:
            continue
        current_price = fresh.get("ibd_candidate_price")
        weekend_price = weekend[code].get("ibd_candidate_price") if has_a else None
        # An identical selected price is still Weekend A, regardless of setup.
        if has_a and _same_buy_point_price(current_price, weekend_price):
            result.at[index, "review_b_active"] = False
            result.at[index, "review_new_buy_point"] = False
            result.at[index, "review_new_buy_point_zone"] = None
            result.at[index, "review_b_has_value_change"] = False
            result.at[index, "review_has_value_change"] = bool(row.get("review_a_has_value_change"))
            continue

        # A distinct current price is B even when its structural metadata
        # does not uniquely identify a Pivot box or resistance date.
        if not bool(row.get("review_b_active")):
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
        result.at[index, "review_b_display_basis"] = (
            "PREVIOUS_SNAPSHOT" if prior_valid else
            "WEEKEND" if _first_trading_session(current_day) and not complete.empty
            else "UNAVAILABLE"
        )
        previous_row = prior.get(code) if prior_valid else None
        event = _midweek_price_event(
            fresh, previous_row, has_previous_snapshot=prior_valid,
            first_midweek_day=_first_trading_session(current_day),
            weekend_price=weekend_price,
        )
        result.at[index, "review_b_display_event"] = event
        # Always derive the B zone from B's own authoritative trigger / close,
        # including a B that V2 recovered after V1 suppressed it.
        observed_zone = classify_zone(fresh.get("ibd_candidate_price"), fresh.get("latest_close"))
        enum_zone = {"BELOW": "BELOW_BUY_POINT", "ZONE": "BUY_ZONE", "EXT": "EXTENDED"}
        result.at[index, "review_b_zone"] = observed_zone
        result.at[index, "review_new_buy_point"] = bool(observed_zone)
        result.at[index, "review_new_buy_point_zone"] = (
            enum_zone.get(observed_zone)
        )
        changed = event not in {"CURRENT", "SNAPSHOT", ""}
        result.at[index, "review_b_has_value_change"] = bool(changed)
        result.at[index, "review_has_value_change"] = bool(
            result.at[index, "review_a_has_value_change"] or changed
        )
    return result
