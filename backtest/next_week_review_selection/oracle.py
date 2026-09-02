from __future__ import annotations

import math

import pandas as pd

from .labels import HORIZONS


TOP_K = 5
TOP_SHARE = 0.10


def add_weekly_oracle_flags(panel: pd.DataFrame) -> pd.DataFrame:
    """Create ex-post winner/loser labels within each snapshot week."""
    out = panel.copy()
    for horizon in HORIZONS:
        for name in _oracle_columns(horizon):
            out[name] = False

    for _, group in out.groupby("snapshot_date", sort=True):
        for horizon in HORIZONS:
            eligible = group.loc[group[f"forward_{horizon}_censored"].eq(False)]
            if eligible.empty:
                continue

            k = min(TOP_K, len(eligible))
            share_k = max(1, int(math.ceil(len(eligible) * TOP_SHARE)))

            _mark_top(
                out,
                eligible,
                value_col=f"forward_{horizon}_return_pct",
                flag_col=f"winner_return_top5_{horizon}",
                k=k,
                ascending=False,
            )
            _mark_top(
                out,
                eligible,
                value_col=f"mfe_{horizon}_pct",
                flag_col=f"winner_mfe_top5_{horizon}",
                k=k,
                ascending=False,
            )
            _mark_top(
                out,
                eligible,
                value_col=f"forward_{horizon}_return_pct",
                flag_col=f"loser_return_bottom5_{horizon}",
                k=k,
                ascending=True,
            )
            _mark_top(
                out,
                eligible,
                value_col=f"forward_{horizon}_return_pct",
                flag_col=f"winner_return_top10pct_{horizon}",
                k=share_k,
                ascending=False,
            )
            _mark_top(
                out,
                eligible,
                value_col=f"forward_{horizon}_return_pct",
                flag_col=f"loser_return_bottom10pct_{horizon}",
                k=share_k,
                ascending=True,
            )

            severe = pd.to_numeric(
                eligible[f"mae_{horizon}_pct"], errors="coerce"
            ).le(-8.0)
            out.loc[eligible.index, f"severe_loser_{horizon}"] = severe.fillna(False)

            out.loc[eligible.index, f"big_winner_any_{horizon}"] = (
                out.loc[eligible.index, f"winner_return_top5_{horizon}"]
                | out.loc[eligible.index, f"winner_mfe_top5_{horizon}"]
            )
            out.loc[eligible.index, f"big_loser_any_{horizon}"] = (
                out.loc[eligible.index, f"loser_return_bottom5_{horizon}"]
                | out.loc[eligible.index, f"severe_loser_{horizon}"]
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
    ]
    for horizon in HORIZONS:
        columns.extend(
            [
                f"forward_{horizon}_return_pct",
                f"mfe_{horizon}_pct",
                f"mae_{horizon}_pct",
                f"winner_return_top5_{horizon}",
                f"winner_mfe_top5_{horizon}",
                f"big_winner_any_{horizon}",
                f"loser_return_bottom5_{horizon}",
                f"severe_loser_{horizon}",
                f"big_loser_any_{horizon}",
                f"winner_return_top10pct_{horizon}",
                f"loser_return_bottom10pct_{horizon}",
            ]
        )
    return panel[[column for column in columns if column in panel.columns]].copy()


def _oracle_columns(horizon: str) -> tuple[str, ...]:
    return (
        f"winner_return_top5_{horizon}",
        f"winner_mfe_top5_{horizon}",
        f"big_winner_any_{horizon}",
        f"loser_return_bottom5_{horizon}",
        f"severe_loser_{horizon}",
        f"big_loser_any_{horizon}",
        f"winner_return_top10pct_{horizon}",
        f"loser_return_bottom10pct_{horizon}",
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
