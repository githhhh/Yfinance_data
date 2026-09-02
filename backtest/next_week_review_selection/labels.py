from __future__ import annotations

from typing import Any

import pandas as pd

from backtest.rd_agent_candidate_rule_audit.utils import fmt_date, normalize_bars, parse_date, pct, to_float


def add_next_week_labels(
    events: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    *,
    sessions: int = 5,
) -> pd.DataFrame:
    """Attach forward-only labels to frozen weekend candidate rows.

    A row is evaluable only when all requested forward sessions are available.
    Partial windows remain explicitly censored and are excluded by the evaluator.
    """
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        row = event.to_dict()
        code = str(event.get("code", "") or "").strip()
        snapshot = parse_date(event.get("snapshot_date"))
        pivot = to_float(event.get("ibd_candidate_price"))
        status = str(event.get("ibd_entry_status", "") or "").strip().upper()
        labels = _empty_labels()

        bars = normalize_bars(prices.get(code))
        if snapshot is None or pivot is None or pivot <= 0 or bars.empty:
            rows.append({**row, **labels})
            continue

        forward = bars[bars.index > snapshot].head(sessions)
        if forward.empty:
            rows.append({**row, **labels})
            continue

        complete = len(forward) >= sessions
        labels["forward_sessions"] = int(len(forward))
        labels["forward_5d_censored"] = not complete
        labels["label_available"] = complete

        first_open = to_float(forward.iloc[0].get("Open"))
        if first_open is not None and first_open > 0:
            labels["observation_start_date"] = fmt_date(pd.Timestamp(forward.index[0]))
            labels["observation_start_open"] = first_open
            labels["mfe_5d_pct"] = pct(to_float(forward["High"].max()), first_open)
            labels["mae_5d_pct"] = pct(to_float(forward["Low"].min()), first_open)
            labels["stop_8_within_5d"] = bool((forward["Low"] <= first_open * 0.92).any())
            if complete:
                labels["forward_5d_return_pct"] = pct(
                    to_float(forward.iloc[sessions - 1].get("Close")),
                    first_open,
                )

        lower = pivot
        upper = pivot * 1.05
        zone_mask = forward["Close"].between(lower, upper, inclusive="both")
        if bool(zone_mask.any()):
            first_zone_idx = zone_mask[zone_mask].index[0]
            labels["first_zone_date"] = fmt_date(pd.Timestamp(first_zone_idx))
            labels["first_zone_close"] = to_float(forward.loc[first_zone_idx, "Close"])

        if complete:
            entered_zone = bool(zone_mask.any())
            labels["review_opportunity_5d"] = status == "ACTIONABLE" or entered_zone
            labels["opportunity_type"] = _opportunity_type(status, entered_zone)
        rows.append({**row, **labels})
    return pd.DataFrame(rows)


def _opportunity_type(status: str, entered_zone: bool) -> str:
    if status == "ACTIONABLE":
        return "CURRENT_ACTIONABLE"
    if not entered_zone:
        return ""
    if status == "UNCONFIRMED":
        return "UNCONFIRMED_TO_ZONE"
    if status == "BELOW_TRIGGER":
        return "BELOW_TO_ZONE"
    if status == "EXTENDED":
        return "EXTENDED_RETEST_TO_ZONE"
    return "OTHER_TO_ZONE"


def _empty_labels() -> dict[str, Any]:
    return {
        "label_available": False,
        "forward_sessions": 0,
        "forward_5d_censored": True,
        "observation_start_date": "",
        "observation_start_open": pd.NA,
        "forward_5d_return_pct": pd.NA,
        "mfe_5d_pct": pd.NA,
        "mae_5d_pct": pd.NA,
        "stop_8_within_5d": False,
        "review_opportunity_5d": False,
        "opportunity_type": "",
        "first_zone_date": "",
        "first_zone_close": pd.NA,
    }
