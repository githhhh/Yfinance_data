from __future__ import annotations

import math

import pandas as pd

from .labels import HORIZONS


TOP_K = 5
TOP_SHARE = 0.10


def add_weekly_oracle_flags(panel: pd.DataFrame) -> pd.DataFrame:
    """Create snapshot-clock and opportunity-clock weekly winner/loser oracles."""
    out = panel.copy()
    for horizon in HORIZONS:
        for prefix in ("", "opp_"):
            for name in _oracle_columns(horizon, prefix):
                out[name] = False

    for _, group in out.groupby("snapshot_date", sort=True):
        for horizon in HORIZONS:
            snapshot_eligible = group.loc[
                group[f"forward_{horizon}_censored"].eq(False)
            ]
            _mark_oracle(
                out,
                snapshot_eligible,
                horizon=horizon,
                value_prefix="",
                flag_prefix="",
            )

            opportunity_eligible = group.loc[
                group["review_opportunity_1w"].eq(True)
                & group[f"opp_forward_{horizon}_censored"].eq(False)
            ]
            _mark_oracle(
                out,
                opportunity_eligible,
                horizon=horizon,
                value_prefix="opp_",
                flag_prefix="opp_",
            )

    return out


def oracle_projection(panel: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "snapshot_date",
        "code",
        "ibd_entry_status",
        "ibd_candidate_rule",
        "ibd_candidate_price",
        "current_vs_ibd_candidate_pct",
        "review_opportunity_1w",
        "opportunity_type_1w",
        "opportunity_delay_sessions",
        "opportunity_anchor_date",
        "opportunity_anchor_price",
    ]
    for horizon in HORIZONS:
        columns.extend(
            [
                f"forward_{horizon}_return_pct",
                f"mfe_{horizon}_pct",
                f"mae_{horizon}_pct",
                f"big_winner_any_{horizon}",
                f"big_loser_any_{horizon}",
                f"opp_forward_{horizon}_return_pct",
                f"opp_mfe_{horizon}_pct",
                f"opp_mae_{horizon}_pct",
                f"opp_big_winner_any_{horizon}",
                f"opp_big_loser_any_{horizon}",
            ]
        )
    return panel[[column for column in columns if column in panel.columns]].copy()


def _mark_oracle(
    out: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    horizon: str,
    value_prefix: str,
    flag_prefix: str,
) -> None:
    if eligible.empty:
        return

    k = min(TOP_K, len(eligible))
    share_k = max(1, int(math.ceil(len(eligible) * TOP_SHARE)))
    return_col = f"{value_prefix}forward_{horizon}_return_pct"
    mfe_col = f"{value_prefix}mfe_{horizon}_pct"
    mae_col = f"{value_prefix}mae_{horizon}_pct"

    _mark_top(
        out,
        eligible,
        value_col=return_col,
        flag_col=f"{flag_prefix}winner_return_top5_{horizon}",
        k=k,
        ascending=False,
    )
    _mark_top(
        out,
        eligible,
        value_col=mfe_col,
        flag_col=f"{flag_prefix}winner_mfe_top5_{horizon}",
        k=k,
        ascending=False,
    )
    _mark_top(
        out,
        eligible,
        value_col=return_col,
        flag_col=f"{flag_prefix}loser_return_bottom5_{horizon}",
        k=k,
        ascending=True,
    )
    _mark_top(
        out,
        eligible,
        value_col=return_col,
        flag_col=f"{flag_prefix}winner_return_top10pct_{horizon}",
        k=share_k,
        ascending=False,
    )
    _mark_top(
        out,
        eligible,
        value_col=return_col,
        flag_col=f"{flag_prefix}loser_return_bottom10pct_{horizon}",
        k=share_k,
        ascending=True,
    )

    severe = pd.to_numeric(eligible[mae_col], errors="coerce").le(-8.0)
    out.loc[
        eligible.index, f"{flag_prefix}severe_loser_{horizon}"
    ] = severe.fillna(False)

    out.loc[eligible.index, f"{flag_prefix}big_winner_any_{horizon}"] = (
        out.loc[eligible.index, f"{flag_prefix}winner_return_top5_{horizon}"]
        | out.loc[eligible.index, f"{flag_prefix}winner_mfe_top5_{horizon}"]
    )
    out.loc[eligible.index, f"{flag_prefix}big_loser_any_{horizon}"] = (
        out.loc[eligible.index, f"{flag_prefix}loser_return_bottom5_{horizon}"]
        | out.loc[eligible.index, f"{flag_prefix}severe_loser_{horizon}"]
    )


def _oracle_columns(horizon: str, prefix: str) -> tuple[str, ...]:
    return (
        f"{prefix}winner_return_top5_{horizon}",
        f"{prefix}winner_mfe_top5_{horizon}",
        f"{prefix}big_winner_any_{horizon}",
        f"{prefix}loser_return_bottom5_{horizon}",
        f"{prefix}severe_loser_{horizon}",
        f"{prefix}big_loser_any_{horizon}",
        f"{prefix}winner_return_top10pct_{horizon}",
        f"{prefix}loser_return_bottom10pct_{horizon}",
    )


def _mark_top(
    out: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    value_col: str,
    flag_col: str,
    k: int,
    ascending: bool,
) -> None:
    ranked = eligible.assign(
        _oracle_value=pd.to_numeric(eligible[value_col], errors="coerce")
    ).dropna(subset=["_oracle_value"])
    if ranked.empty:
        return
    ranked["_oracle_code"] = ranked["code"].fillna("").astype(str).str.upper()
    ranked = ranked.sort_values(
        ["_oracle_value", "_oracle_code"],
        ascending=[ascending, True],
        kind="mergesort",
    )
    out.loc[ranked.head(min(k, len(ranked))).index, flag_col] = True
