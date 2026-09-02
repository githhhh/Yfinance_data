from __future__ import annotations

from typing import Any

import pandas as pd

from .utils import fmt_date, normalize_bars, parse_date, pct, to_float


HORIZONS = {
    "1w": 5,
    "2w": 10,
    "3w": 15,
    "4w": 20,
}


def add_forward_labels(
    events: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Attach forward-only 1W/2W/3W/4W labels to frozen weekend rows."""
    rows: list[dict[str, Any]] = []
    max_sessions = max(HORIZONS.values())

    for _, event in events.iterrows():
        row = event.to_dict()
        code = str(event.get("code", "") or "").strip()
        snapshot = parse_date(event.get("snapshot_date"))
        pivot = to_float(event.get("ibd_candidate_price"))
        status = str(event.get("ibd_entry_status", "") or "").strip().upper()
        labels = _empty_labels()

        bars = normalize_bars(prices.get(code))
        if snapshot is None or bars.empty:
            rows.append({**row, **labels})
            continue

        forward = bars[bars.index > snapshot].head(max_sessions)
        if forward.empty:
            rows.append({**row, **labels})
            continue

        first_open = to_float(forward.iloc[0].get("Open"))
        if first_open is not None and first_open > 0:
            labels["observation_start_date"] = fmt_date(pd.Timestamp(forward.index[0]))
            labels["observation_start_open"] = first_open

        for horizon, sessions in HORIZONS.items():
            _add_horizon_labels(labels, forward, first_open, horizon, sessions)

        if pivot is not None and pivot > 0:
            first_week = forward.head(HORIZONS["1w"])
            if len(first_week) >= HORIZONS["1w"]:
                zone_mask = first_week["Close"].between(pivot, pivot * 1.05, inclusive="both")
                entered_zone = bool(zone_mask.any())
                labels["review_opportunity_1w"] = status == "ACTIONABLE" or entered_zone
                labels["opportunity_type_1w"] = _opportunity_type(status, entered_zone)
                if entered_zone:
                    first_idx = zone_mask[zone_mask].index[0]
                    labels["first_zone_date_1w"] = fmt_date(pd.Timestamp(first_idx))
                    labels["first_zone_close_1w"] = to_float(first_week.loc[first_idx, "Close"])

        rows.append({**row, **labels})

    return pd.DataFrame(rows)


def _add_horizon_labels(
    labels: dict[str, Any],
    forward: pd.DataFrame,
    first_open: float | None,
    horizon: str,
    sessions: int,
) -> None:
    complete = len(forward) >= sessions
    labels[f"forward_{horizon}_censored"] = not complete
    labels[f"forward_{horizon}_sessions"] = min(len(forward), sessions)
    if not complete or first_open is None or first_open <= 0:
        return

    window = forward.head(sessions)
    labels[f"forward_{horizon}_return_pct"] = pct(
        to_float(window.iloc[-1].get("Close")),
        first_open,
    )
    labels[f"mfe_{horizon}_pct"] = pct(to_float(window["High"].max()), first_open)
    labels[f"mae_{horizon}_pct"] = pct(to_float(window["Low"].min()), first_open)
    labels[f"stop_8_within_{horizon}"] = bool(
        (window["Low"] <= first_open * 0.92).any()
    )


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
    labels: dict[str, Any] = {
        "observation_start_date": "",
        "observation_start_open": pd.NA,
        "review_opportunity_1w": False,
        "opportunity_type_1w": "",
        "first_zone_date_1w": "",
        "first_zone_close_1w": pd.NA,
    }
    for horizon in HORIZONS:
        labels[f"forward_{horizon}_censored"] = True
        labels[f"forward_{horizon}_sessions"] = 0
        labels[f"forward_{horizon}_return_pct"] = pd.NA
        labels[f"mfe_{horizon}_pct"] = pd.NA
        labels[f"mae_{horizon}_pct"] = pd.NA
        labels[f"stop_8_within_{horizon}"] = False
    return labels
