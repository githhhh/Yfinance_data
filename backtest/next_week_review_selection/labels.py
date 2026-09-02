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
MAX_FORWARD_SESSIONS = HORIZONS["1w"] + HORIZONS["4w"]


def add_forward_labels(
    events: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Attach two forward clocks.

    Snapshot clock starts at the first session after the weekend snapshot.
    Opportunity clock starts after a real review opportunity is known:
    - ACTIONABLE: weekend latest_close is the anchor.
    - non-ACTIONABLE: first 1W close inside frozen Pivot..Pivot+5% is the anchor.

    The selector never reads these ex-post columns.
    """
    rows: list[dict[str, Any]] = []

    for _, event in events.iterrows():
        row = event.to_dict()
        code = str(event.get("code", "") or "").strip()
        snapshot = parse_date(event.get("snapshot_date"))
        pivot = to_float(event.get("ibd_candidate_price"))
        latest_close = to_float(event.get("latest_close"))
        status = str(event.get("ibd_entry_status", "") or "").strip().upper()
        labels = _empty_labels()

        bars = normalize_bars(prices.get(code))
        if snapshot is None or bars.empty:
            rows.append({**row, **labels})
            continue

        forward = bars[bars.index > snapshot].head(MAX_FORWARD_SESSIONS)
        if forward.empty:
            rows.append({**row, **labels})
            continue

        first_open = to_float(forward.iloc[0].get("Open"))
        if first_open is not None and first_open > 0:
            labels["observation_start_date"] = fmt_date(pd.Timestamp(forward.index[0]))
            labels["observation_start_open"] = first_open

        for horizon, sessions in HORIZONS.items():
            _add_snapshot_horizon(labels, forward, first_open, horizon, sessions)

        first_week = forward.head(HORIZONS["1w"])
        first_week_complete = len(first_week) >= HORIZONS["1w"]
        anchor_date: pd.Timestamp | None = None
        anchor_price: float | None = None

        if first_week_complete and status == "ACTIONABLE":
            labels["review_opportunity_1w"] = True
            labels["opportunity_type_1w"] = "CURRENT_ACTIONABLE"
            labels["opportunity_delay_sessions"] = 0
            anchor_date = snapshot
            anchor_price = latest_close if latest_close is not None else pivot
        elif first_week_complete and pivot is not None and pivot > 0:
            zone_mask = first_week["Close"].between(
                pivot, pivot * 1.05, inclusive="both"
            )
            if bool(zone_mask.any()):
                first_idx = pd.Timestamp(zone_mask[zone_mask].index[0])
                anchor_date = first_idx
                anchor_price = to_float(first_week.loc[first_idx, "Close"])
                labels["review_opportunity_1w"] = True
                labels["opportunity_type_1w"] = _opportunity_type(status)
                labels["first_zone_date_1w"] = fmt_date(first_idx)
                labels["first_zone_close_1w"] = anchor_price
                labels["opportunity_delay_sessions"] = (
                    list(first_week.index).index(first_idx) + 1
                )

        if (
            labels["review_opportunity_1w"]
            and anchor_date is not None
            and anchor_price is not None
            and anchor_price > 0
        ):
            labels["opportunity_anchor_date"] = fmt_date(anchor_date)
            labels["opportunity_anchor_price"] = anchor_price
            post_opportunity = bars[bars.index > anchor_date]
            for horizon, sessions in HORIZONS.items():
                _add_opportunity_horizon(
                    labels,
                    post_opportunity,
                    anchor_price,
                    horizon,
                    sessions,
                )

        rows.append({**row, **labels})

    return pd.DataFrame(rows)


def _add_snapshot_horizon(
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
        to_float(window.iloc[-1].get("Close")), first_open
    )
    labels[f"mfe_{horizon}_pct"] = pct(to_float(window["High"].max()), first_open)
    labels[f"mae_{horizon}_pct"] = pct(to_float(window["Low"].min()), first_open)
    labels[f"stop_8_within_{horizon}"] = bool(
        (window["Low"] <= first_open * 0.92).any()
    )


def _add_opportunity_horizon(
    labels: dict[str, Any],
    post_opportunity: pd.DataFrame,
    anchor_price: float,
    horizon: str,
    sessions: int,
) -> None:
    complete = len(post_opportunity) >= sessions
    labels[f"opp_forward_{horizon}_censored"] = not complete
    labels[f"opp_forward_{horizon}_sessions"] = min(len(post_opportunity), sessions)
    if not complete:
        return

    window = post_opportunity.head(sessions)
    labels[f"opp_forward_{horizon}_return_pct"] = pct(
        to_float(window.iloc[-1].get("Close")), anchor_price
    )
    labels[f"opp_mfe_{horizon}_pct"] = pct(
        to_float(window["High"].max()), anchor_price
    )
    labels[f"opp_mae_{horizon}_pct"] = pct(
        to_float(window["Low"].min()), anchor_price
    )
    labels[f"opp_stop_8_within_{horizon}"] = bool(
        (window["Low"] <= anchor_price * 0.92).any()
    )


def _opportunity_type(status: str) -> str:
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
        "opportunity_delay_sessions": pd.NA,
        "opportunity_anchor_date": "",
        "opportunity_anchor_price": pd.NA,
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

        labels[f"opp_forward_{horizon}_censored"] = True
        labels[f"opp_forward_{horizon}_sessions"] = 0
        labels[f"opp_forward_{horizon}_return_pct"] = pd.NA
        labels[f"opp_mfe_{horizon}_pct"] = pd.NA
        labels[f"opp_mae_{horizon}_pct"] = pd.NA
        labels[f"opp_stop_8_within_{horizon}"] = False
    return labels
