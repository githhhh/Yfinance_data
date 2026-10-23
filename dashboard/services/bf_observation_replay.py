"""Pure, read-only BF buy-point observation facts (schema v2).

No git, filesystem, trading eligibility, Futu, Pool mutation or Telegram I/O.
A's published v1 path/identity semantics remain in bf_observation/bf_transition.
"""
from __future__ import annotations

from datetime import date
import math
from typing import Any

from dashboard.services.bf_transition import _price_zone, to_bool


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _day(value: Any) -> str | None:
    if value is None:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(value).split("T")[0]).date().isoformat()
    except (TypeError, ValueError):
        return None


def zone(trigger: Any, close: Any) -> str | None:
    p, c = _num(trigger), _num(close)
    if p is None or c is None:
        return None
    # A and B must use one authoritative rounding/zone classifier.
    return {"BELOW_BUY_POINT": "BELOW", "BUY_ZONE": "ZONE",
            "EXTENDED": "EXT"}.get(_price_zone(p, c))


def identity_v2(row: Any) -> dict:
    """Compatibility facade: price is the only buy-point identity.

    Setup, box type, resistance date and selected-pivot metadata may explain
    how a price was found; none may create or invalidate its identity.
    """
    trigger = _num(row.get("ibd_candidate_price"))
    if trigger is None:
        return {"resolution": "UNKNOWN", "key": None, "trigger": None}
    return {"resolution": "EXACT", "key": f"price|{repr(trigger).removesuffix('.0')}",
            "trigger": trigger, "anchor_type": None}


def price_replay(*, trigger: Any, previous_close: Any, current_close: Any,
                 previous_date: str | None, current_date: str | None,
                 basis: str, previous_trigger: Any = None,
                 daily_closes: list[tuple[str, float]] | None = None) -> dict:
    """Compare *published* boundary prices. Daily bars are separate path evidence."""
    old_close, new_close, now_trigger = _num(previous_close), _num(current_close), _num(trigger)
    before = zone(trigger, old_close)
    after = zone(trigger, new_close)
    out = {"previous_zone": before, "current_zone": after, "transition": None,
           "previous_date": previous_date, "current_date": current_date,
           "basis": basis, "trigger": now_trigger, "pct": None,
           "trigger_revision": "UNKNOWN", "daily_path": []}
    if now_trigger is None or new_close is None:
        return out
    if daily_closes:
        out["daily_path"] = [
            {"date": day, "zone": zone(now_trigger, close)}
            for day, close in sorted(daily_closes)
            if day <= current_date and _num(close) is not None
        ]
    out["pct"] = round((new_close / now_trigger - 1) * 100, 2)
    old_trigger = _num(previous_trigger)
    out["trigger_revision"] = (
        "UNKNOWN" if old_trigger is None else
        "REVISED" if not math.isclose(old_trigger, now_trigger, rel_tol=1e-9, abs_tol=1e-4)
        else "UNCHANGED"
    )
    if before is None or after is None or previous_date is None or current_date is None:
        return out
    if previous_date >= current_date:
        return out
    # A changed trigger that independently changes either classification
    # cannot be presented as a pure price crossing.
    if out["trigger_revision"] == "REVISED" and (
        zone(old_trigger, old_close) != before or zone(old_trigger, new_close) != after
    ):
        return out
    if before != after:
        out["transition"] = f"{before}_TO_{after}"
    return out


def watch_provenance(history: list[dict], *, as_of: str,
                     price_boundary_date: str | None = None) -> dict:
    watches = []
    for observation in history:
        when = observation.get("date")
        row = observation.get("row")
        if not when or when >= as_of or row is None:
            continue
        if to_bool(row.get("bf_watch_active")) is not True:
            continue
        watches.append({"date": when, "type": row.get("bf_watch_type"),
                        "trigger": _num(row.get("bf_watch_trigger_price")),
                        "origin": observation.get("origin", "MIDWEEK"),
                        "relation": "TICKER_CONTEXT",
                        "source_ref": observation.get("source_ref")})
    watches.sort(key=lambda x: x["date"])
    return {"active_prior": bool(watches),
            "active_at_price_boundary": bool(price_boundary_date and any(
                watch["date"] == price_boundary_date for watch in watches)),
            "latest": watches[-1] if watches else None,
            "observations": watches}


def entry_confirmation(current: Any, *, current_zone: str | None,
                       previous_same_anchor: Any = None) -> dict:
    valid = to_bool(current.get("ibd_entry_valid")) is True
    status = str(current.get("ibd_entry_status") or "")
    act = valid and status == "ACTIONABLE" and current_zone == "ZONE"
    previous_status = (str(previous_same_anchor.get("ibd_entry_status") or "")
                       if previous_same_anchor is not None else "")
    return {"valid": valid, "status": status, "act": act,
            "upgrade": bool(act and previous_status == "UNCONFIRMED")}


def resolve_midweek_facts(*, code: str, current: dict,
                          history: list[dict], weekend: dict | None,
                          as_of: str, review_week: str,
                          baseline_verified: bool,
                          history_coverage: str = "AVAILABLE",
                          daily_closes: list[tuple[str, float]] | None = None) -> dict | None:
    """Resolve the four orthogonal facts; never decide Telegram eligibility.

    Historical identity is independent of the current price and ACT. This
    function also returns facts for EXT / BELOW B signals when evidence exists.
    """
    if to_bool(current.get("signal")) is not True:
        return None
    ident = identity_v2(current)
    if ident["resolution"] != "EXACT":
        return None
    if weekend is not None and to_bool(weekend.get("signal")) is True:
        wk = identity_v2(weekend)
        if wk["resolution"] != "EXACT" or wk["key"] == ident["key"]:
            return None

    if not baseline_verified or history_coverage != "AVAILABLE":
        return None
    ordered = sorted([r for r in history if r.get("date") and r["date"] < as_of],
                     key=lambda record: record["date"])
    if not ordered or any(not record.get("verified") for record in ordered):
        return None

    same = []
    for record in ordered:
        row = record.get("row")
        if row is None or to_bool(row.get("signal")) is not True:
            continue
        old = identity_v2(row)
        if old["resolution"] != "EXACT":
            # Never infer first-observed from an ambiguous earlier signal.
            return None
        if old["key"] == ident["key"]:
            same.append((record, old))

    last = ordered[-1]
    last_row = last.get("row")
    last_anchor = (identity_v2(last_row)
                   if last_row is not None and to_bool(last_row.get("signal")) is True
                   else None)
    same_on_last = bool(last_anchor and last_anchor["resolution"] == "EXACT"
                        and last_anchor["key"] == ident["key"])
    lifecycle = ("CONTINUING" if same_on_last else
                 "REAPPEARED" if same else "FIRST_OBSERVED")
    basis = "TRACKED_ANCHOR" if same else "RETROSPECTIVE"
    prior_trigger = same[-1][1]["trigger"] if same else None
    prior_record = same[-1][0] if same else None
    last_close = (_num(last_row.get("latest_close"))
                  if last_row is not None else None)
    if last_close is None:
        last_close = last.get("price_close")
    price = price_replay(
        trigger=ident["trigger"], previous_close=last_close,
        current_close=current.get("latest_close"), previous_date=last["date"],
        current_date=as_of, basis=basis, previous_trigger=prior_trigger,
        daily_closes=daily_closes,
    )
    watches = watch_provenance(ordered, as_of=as_of,
                               price_boundary_date=price["previous_date"])
    entry = entry_confirmation(
        current, current_zone=price["current_zone"],
        previous_same_anchor=prior_record["row"] if prior_record else None,
    )
    return {
        "schema_version": 2, "source": "MIDWEEK", "code": code,
        "market_date": as_of, "review_week": review_week,
        "anchor_key_v2": ident["key"],
        "identity": {
            "resolution": ident["resolution"], "lifecycle": lifecycle,
            "trigger_revision": price["trigger_revision"],
            "first_observed_date": same[0][0]["date"] if same else as_of,
        },
        "price": price, "watch": watches, "entry": entry,
        "evidence": {
            "history_coverage": history_coverage,
            "prior_snapshot_verified": True,
            "price_observation_verified": bool(price["previous_zone"]),
        },
    }


def midweek_opportunity_policy(facts: dict | None) -> dict | None:
    """Midweek B is a current-snapshot opportunity, not another A watch list."""
    if not facts:
        return None
    price, identity = facts["price"], facts["identity"]
    entry, watch = facts["entry"], facts["watch"]
    if price["current_zone"] != "ZONE":
        return None
    # Entry state is an independent resolver fact. Reject contradictions;
    # neither ZONE nor a historical WATCH implies ACT.
    state = entry["status"]
    if (state not in {"ACTIONABLE", "UNCONFIRMED"}
            or (state == "ACTIONABLE") != entry["valid"]):
        return None
    existing = identity["lifecycle"] in {"REAPPEARED", "CONTINUING"}
    if existing:
        if price["transition"] not in {"BELOW_TO_ZONE", "EXT_TO_ZONE"}:
            return None
        primary = "ZONE_REENTRY"
    else:
        if identity["lifecycle"] != "FIRST_OBSERVED":
            return None
        primary = "FIRST_OBSERVED_BP"

    modifiers = []
    if watch["active_prior"]:
        modifiers.append("PRIOR_WATCH")
    if identity["lifecycle"] == "REAPPEARED":
        modifiers.append("REAPPEARED")
    if entry["act"]:
        modifiers.append("ACT")
    import hashlib
    import json
    event_identity = [2, facts["review_week"], facts["market_date"],
                      facts["code"], facts["anchor_key_v2"], primary]
    event_key = hashlib.sha256(
        json.dumps(event_identity, separators=(",", ":")).encode()
    ).hexdigest()
    return {**facts, "primary": primary, "modifiers": modifiers,
            "event_key": event_key}


def observe_midweek_buy_point(*, code: str, current: dict,
                              history: list[dict], weekend: dict | None,
                              as_of: str, review_week: str,
                              baseline_verified: bool,
                              history_coverage: str = "AVAILABLE",
                              daily_closes: list[tuple[str, float]] | None = None) -> dict | None:
    """Compatibility facade: resolve facts first, then apply B policy."""
    return midweek_opportunity_policy(resolve_midweek_facts(
        code=code, current=current, history=history, weekend=weekend,
        as_of=as_of, review_week=review_week,
        baseline_verified=baseline_verified, history_coverage=history_coverage,
        daily_closes=daily_closes,
    ))
