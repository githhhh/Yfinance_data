from __future__ import annotations

import pandas as pd

from .labels import HORIZONS
from .oracle import add_weekly_oracle_flags


def panel_asof_cutoff(panel: pd.DataFrame, cutoff: str | pd.Timestamp) -> pd.DataFrame:
    """Return a leakage-safe training view as known at cutoff close.

    Future labels whose end date is after cutoff are censored and cleared before
    weekly Oracle flags are recomputed. This lets 1W/2W/3W/4W use different
    amounts of history without dropping whole snapshot weeks.
    """
    out = panel.copy()
    cutoff_ts = pd.Timestamp(cutoff).tz_localize(None)

    for horizon in HORIZONS:
        _mask_clock(
            out,
            cutoff_ts,
            end_col=f"forward_{horizon}_end_date",
            censored_col=f"forward_{horizon}_censored",
            sessions_col=f"forward_{horizon}_sessions",
            value_cols=[
                f"forward_{horizon}_return_pct",
                f"mfe_{horizon}_pct",
                f"mae_{horizon}_pct",
                f"stop_8_within_{horizon}",
            ],
        )
        _mask_clock(
            out,
            cutoff_ts,
            end_col=f"opp_forward_{horizon}_end_date",
            censored_col=f"opp_forward_{horizon}_censored",
            sessions_col=f"opp_forward_{horizon}_sessions",
            value_cols=[
                f"opp_forward_{horizon}_return_pct",
                f"opp_mfe_{horizon}_pct",
                f"opp_mae_{horizon}_pct",
                f"opp_stop_8_within_{horizon}",
            ],
        )

    first_week_end = pd.to_datetime(
        out["forward_1w_end_date"], errors="coerce"
    )
    unresolved_opportunity = first_week_end.isna() | first_week_end.gt(cutoff_ts)
    out.loc[unresolved_opportunity, "review_opportunity_1w"] = False
    for col in (
        "opportunity_type_1w",
        "opportunity_delay_sessions",
        "opportunity_anchor_date",
        "opportunity_anchor_price",
        "first_zone_date_1w",
        "first_zone_close_1w",
    ):
        if col in out.columns:
            out.loc[unresolved_opportunity, col] = pd.NA

    return add_weekly_oracle_flags(out)


def resolved_week_counts(panel: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for horizon in HORIZONS:
        complete = panel[panel[f"forward_{horizon}_censored"].eq(False)]
        counts[horizon] = int(
            complete["snapshot_date"].astype(str).nunique()
        )
    return counts


def _mask_clock(
    frame: pd.DataFrame,
    cutoff: pd.Timestamp,
    *,
    end_col: str,
    censored_col: str,
    sessions_col: str,
    value_cols: list[str],
) -> None:
    end = pd.to_datetime(frame[end_col], errors="coerce")
    unresolved = end.isna() | end.gt(cutoff)
    frame.loc[unresolved, censored_col] = True
    frame.loc[unresolved, sessions_col] = 0
    frame.loc[unresolved, end_col] = ""
    for col in value_cols:
        if col.endswith("stop_8_within_1w") or "stop_8_within" in col:
            frame.loc[unresolved, col] = False
        else:
            frame.loc[unresolved, col] = pd.NA
