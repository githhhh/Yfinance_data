"""Read-only presentation of buy-point observation facts.

The original A (Weekend Buy Point) observation/alert functions in
bf_observation.py must remain AST-identical to main. This adapter is the only
place the public Dashboard adds source-specific labels and V2 B identity.
Nothing here is used by Futu or the Telegram v1/v2 event producers.
"""
from __future__ import annotations

import pandas as pd

from bf_snapshot import is_valid_complete_baseline

from dashboard.services.bf_observation_replay import price_replay, zone as classify_zone
from dashboard.services.bf_transition import _same_buy_point_price, to_bool


def _observed_date(frame: pd.DataFrame):
    if frame is None or frame.empty or "snapshot_date" not in frame.columns:
        return None
    parsed = pd.to_datetime(frame["snapshot_date"], errors="coerce").dropna().dt.date.unique()
    return parsed[0] if len(parsed) == 1 else None


def _midweek_price_event(
    current, previous, *, has_previous_snapshot: bool,
    first_midweek_snapshot: bool, weekend_price=None,
):
    # A verified first publication compares against the complete Weekend Pool,
    # irrespective of how many trading sessions were skipped before publishing.
    if not has_previous_snapshot:
        return "NEW BP" if first_midweek_snapshot else "SNAPSHOT"
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
    first_midweek_snapshot: bool = False,
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
    weekend_day = _observed_date(complete)
    first_weekend_comparison = bool(
        first_midweek_snapshot and not prior_valid and current_day and weekend_day
        and is_valid_complete_baseline(weekend_day, current_day)
    )
    for index, row in result.iterrows():
        code = str(row["code"])
        has_a = bool(row.get("review_a_active"))
        fresh = today.get(code)
        status_event = ""
        if has_a and fresh is not None and to_bool(fresh.get("signal")) is True:
            # Identity belongs to the immutable Weekend price; confirmation
            # belongs to the current snapshot whenever it selects that price.
            weekend_price = weekend[code].get("ibd_candidate_price")
            if _same_buy_point_price(fresh.get("ibd_candidate_price"), weekend_price):
                old = weekend[code] if first_weekend_comparison else prior.get(code) if prior_valid else None
                new_status = str(fresh.get("ibd_entry_status") or "")
                new_valid = to_bool(fresh.get("ibd_entry_valid"))
                if new_status in {"ACTIONABLE", "UNCONFIRMED", "BELOW_TRIGGER", "EXTENDED"}:
                    if new_status == "ACTIONABLE" and new_valid is not True:
                        new_status = "UNCONFIRMED"
                    result.at[index, "ibd_entry_status"] = new_status
                    result.at[index, "ibd_entry_valid"] = new_valid
                    ratio = pd.to_numeric(fresh.get("ibd_entry_volume_ratio"), errors="coerce")
                    result.at[index, "ibd_entry_vol_or_reject"] = (
                        f"{float(ratio):.2f}x" if new_valid is True and pd.notna(ratio)
                        else "n/a" if new_valid is True
                        else str(fresh.get("ibd_entry_reject_reason") or "Volume not confirmed")
                    )
                    if (old is not None and to_bool(old.get("signal")) is True
                            and _same_buy_point_price(old.get("ibd_candidate_price"), weekend_price)):
                        old_status = str(old.get("ibd_entry_status") or "")
                        if old_status in {"ACTIONABLE", "UNCONFIRMED", "BELOW_TRIGGER", "EXTENDED"} and old_status != new_status:
                            status_event = f"{old_status} → {new_status}"

        if (first_weekend_comparison or prior_valid) and has_a:
            # Only published Pool snapshot closes define Review Now deltas;
            # intervening daily bars remain observation evidence, not baselines.
            baseline_row = weekend[code] if first_weekend_comparison else prior.get(code)
            price = weekend[code].get("ibd_candidate_price")
            before = classify_zone(price, baseline_row.get("latest_close")) if baseline_row is not None else None
            after = classify_zone(price, fresh.get("latest_close")) if fresh is not None else None
            zone_event = f"{before} → {after}" if before and after and before != after else ""
            if after:
                result.at[index, "review_a_zone"] = after
                trigger = pd.to_numeric(price, errors="coerce")
                close = pd.to_numeric(fresh.get("latest_close"), errors="coerce")
                if pd.notna(trigger) and trigger > 0 and pd.notna(close) and close > 0:
                    pct = round((close / trigger - 1) * 100, 2)
                    result.at[index, "review_a_pct"] = pct
                    result.at[index, "current_vs_ibd_candidate_pct"] = pct
                    result.at[index, "latest_close"] = float(close)
                    result.at[index, "review_a_as_of"] = current_day.isoformat()
                    result.at[index, "review_data_available"] = True
            result.at[index, "review_a_event"] = zone_event or status_event
            if first_weekend_comparison:
                result.at[index, "review_a_week_event"] = zone_event
            previous_member = baseline_row is not None
            result.at[index, "review_membership_event"] = (
                "IN → OUT" if previous_member and fresh is None else
                "OUT → IN" if not previous_member and fresh is not None else ""
            )
        elif status_event and not row.get("review_a_event"):
            # Same-price confirmation changes remain Review Now events.
            result.at[index, "review_a_event"] = status_event

        a_event = str(result.at[index, "review_a_event"] or "")
        a_week = str(result.at[index, "review_a_week_event"] or "")
        membership_event = str(result.at[index, "review_membership_event"] or "")
        # The Weekend quick groups are cumulative relative to the immutable A
        # baseline. Legacy path["change"] chooses yesterday as its basis when
        # today's price crosses a zone, and therefore cannot drive the
        # Weekend quick-group membership (e.g. ZONE → BELOW → ZONE).
        # Keep the legacy path untouched for its own price replay consumers.
        if has_a:
            weekend_zone_group = {
                "ZONE": "ENTERED_BUY_ZONE",
                "BELOW": "BELOW_BUY_POINT",
                "EXT": "BECAME_EXTENDED",
            }
            result.at[index, "review_weekend_buy_point_change"] = (
                weekend_zone_group.get(row.get("review_a_zone"), "UNCHANGED")
                if a_week else "UNCHANGED"
            )
        # Only published snapshot differences belong in Review Now. Weekend
        # quick groups remain cumulative and independently queryable.
        a_attention = bool(a_event or membership_event) if has_a and (prior_valid or first_weekend_comparison) else False
        result.at[index, "review_a_has_value_change"] = a_attention
        result.at[index, "review_has_value_change"] = bool(
            a_attention or row.get("review_b_has_value_change")
        )
        result.at[index, "review_a_focus_basis"] = (
            "WEEKEND" if a_event and first_weekend_comparison else
            "PREVIOUS_SNAPSHOT" if a_event and prior_valid else
            "WEEKEND" if a_week else None
        )
        result.at[index, "review_b_display_event"] = ""
        result.at[index, "review_b_display_basis"] = None
        result.at[index, "review_selected_price_event"] = ""

        fresh = today.get(code)
        if fresh is None or to_bool(fresh.get("signal")) is not True:
            continue
        current_price = fresh.get("ibd_candidate_price")
        weekend_price = weekend[code].get("ibd_candidate_price") if has_a else None
        # An identical selected price is still Weekend A, regardless of setup.
        if has_a and _same_buy_point_price(current_price, weekend_price):
            # The previously selected B can return to fixed Weekend A.
            # This is a price-identity transition, not an A-zone movement.
            previous_row = prior.get(code) if prior_valid else None
            if (previous_row is not None
                    and to_bool(previous_row.get("signal")) is True
                    and not _same_buy_point_price(previous_row.get("ibd_candidate_price"), current_price)):
                result.at[index, "review_selected_price_event"] = "BP CHANGED"
            result.at[index, "review_b_active"] = False
            result.at[index, "review_new_buy_point"] = False
            result.at[index, "review_new_buy_point_zone"] = None
            result.at[index, "review_b_has_value_change"] = False
            result.at[index, "review_has_value_change"] = bool(
                result.at[index, "review_a_has_value_change"]
                or result.at[index, "review_selected_price_event"]
            )
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
            "WEEKEND" if first_weekend_comparison
            else "UNAVAILABLE"
        )
        previous_row = prior.get(code) if prior_valid else None
        event = _midweek_price_event(
            fresh, previous_row, has_previous_snapshot=prior_valid,
            first_midweek_snapshot=first_weekend_comparison,
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
