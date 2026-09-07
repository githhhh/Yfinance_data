from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .config import TOP_N
from .metrics import moving_block_bootstrap_ci


def _panel_lookup(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.copy()
    work["snapshot_date"] = work["snapshot_date"].astype(str)
    work["code"] = work["code"].astype(str)
    return work.set_index(["snapshot_date", "code"], drop=False)


def idealized_stop8_capital_return(
    lookup: pd.DataFrame,
    snapshot: str,
    codes: list[str],
) -> float | None:
    """Idealized no-slippage Stop8 execution.

    A name that ever touches the -8% stop is booked at exactly -8%; otherwise
    its next-open W4 terminal return is used. Empty slots remain cash because
    capital is always divided by TOP_N.

    This is intentionally optimistic for gap-through-stop events and is a
    scenario diagnostic, not a claim about realized execution.
    """
    total = 0.0
    for code in codes:
        key = (str(snapshot), str(code))
        if key not in lookup.index:
            return None
        row = lookup.loc[key]
        if isinstance(row, pd.DataFrame):
            raise RuntimeError(f"Duplicate stop-execution row: {key}")
        if not bool(row.get("next_open_price_valid", False)):
            return None
        terminal = pd.to_numeric(
            pd.Series([row.get("next_open_w4_return_pct")]),
            errors="coerce",
        ).iloc[0]
        if pd.isna(terminal):
            return None
        stop = bool(row.get("next_open_w4_stop8", False))
        total += -8.0 if stop else float(terminal)
    return float(total / TOP_N)


def _paired_summary(
    valid: pd.DataFrame,
    *,
    value_col: str,
    benchmark_col: str,
) -> dict[str, Any]:
    if valid.empty:
        return {
            "support_weeks": 0,
            "mean_return": None,
            "median_return": None,
            "mean_spread_vs_b0": None,
            "median_spread_vs_b0": None,
            "beat_b0_rate": None,
            "spread_ci_low": None,
            "spread_ci_high": None,
            "worst_return": None,
            "p10_return": None,
        }

    value = pd.to_numeric(valid[value_col], errors="coerce")
    bench = pd.to_numeric(valid[benchmark_col], errors="coerce")
    spread = value - bench
    ci = moving_block_bootstrap_ci(spread.to_numpy())
    return {
        "support_weeks": int(len(valid)),
        "mean_return": float(value.mean()),
        "median_return": float(value.median()),
        "mean_spread_vs_b0": float(spread.mean()),
        "median_spread_vs_b0": float(spread.median()),
        "beat_b0_rate": float((spread > 0).mean()),
        "spread_ci_low": ci["mean_ci_low"],
        "spread_ci_high": ci["mean_ci_high"],
        "worst_return": float(value.min()),
        "p10_return": float(np.percentile(value.to_numpy(), 10)),
    }


def simple_stop8_execution(
    panel: pd.DataFrame,
    simple_weekly: pd.DataFrame,
    b0_weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if panel.empty or simple_weekly.empty or b0_weekly.empty:
        return pd.DataFrame(), pd.DataFrame()

    lookup = _panel_lookup(panel)
    b0_codes = {
        str(row["snapshot_date"]): json.loads(
            str(row.get("selected_codes", "[]") or "[]")
        )
        for _, row in b0_weekly.iterrows()
    }

    rows: list[dict[str, Any]] = []
    for _, week in simple_weekly.iterrows():
        if not bool(week.get("primary_valid", False)):
            continue
        snapshot = str(week["snapshot_date"])
        baseline = str(week["baseline"])
        codes = json.loads(str(week.get("codes", "[]") or "[]"))
        bcodes = b0_codes.get(snapshot, [])

        value = idealized_stop8_capital_return(lookup, snapshot, codes)
        bench = idealized_stop8_capital_return(lookup, snapshot, bcodes)
        if value is None or bench is None:
            continue
        rows.append({
            "snapshot_date": snapshot,
            "baseline": baseline,
            "codes": json.dumps(codes),
            "b0_codes": json.dumps(bcodes),
            "idealized_stop8_return": value,
            "b0_idealized_stop8_return": bench,
            "spread_vs_b0": value - bench,
        })

    weekly = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    if not weekly.empty:
        for baseline, group in weekly.groupby("baseline", sort=False):
            summary_rows.append({
                "baseline": baseline,
                **_paired_summary(
                    group,
                    value_col="idealized_stop8_return",
                    benchmark_col="b0_idealized_stop8_return",
                ),
            })
    return weekly, pd.DataFrame(summary_rows)


def capacity_stop8_execution(
    panel: pd.DataFrame,
    capacity_weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if panel.empty or capacity_weekly.empty:
        return pd.DataFrame(), pd.DataFrame()

    lookup = _panel_lookup(panel)
    rows: list[dict[str, Any]] = []

    for _, week in capacity_weekly.iterrows():
        if not bool(week.get("mature", False)):
            continue
        snapshot = str(week["snapshot_date"])
        policy = str(week["policy_id"])
        codes = json.loads(str(week.get("codes", "[]") or "[]"))
        original = json.loads(str(week.get("original_codes", "[]") or "[]"))

        value = idealized_stop8_capital_return(lookup, snapshot, codes)
        bench = idealized_stop8_capital_return(lookup, snapshot, original)
        if value is None or bench is None:
            continue
        rows.append({
            "snapshot_date": snapshot,
            "policy_id": policy,
            "original_pick_count": int(week["original_pick_count"]),
            "underfill_cause": str(week["underfill_cause"]),
            "codes": json.dumps(codes),
            "original_codes": json.dumps(original),
            "idealized_stop8_return": value,
            "b0_idealized_stop8_return": bench,
            "spread_vs_b0": value - bench,
        })

    weekly = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    if not weekly.empty:
        for policy, group in weekly.groupby("policy_id", sort=False):
            for scope, scoped in [
                ("all_mature", group),
                (
                    "underfilled_only",
                    group[
                        pd.to_numeric(
                            group["original_pick_count"], errors="coerce"
                        ) < TOP_N
                    ],
                ),
            ]:
                summary_rows.append({
                    "policy_id": policy,
                    "scope": scope,
                    **_paired_summary(
                        scoped,
                        value_col="idealized_stop8_return",
                        benchmark_col="b0_idealized_stop8_return",
                    ),
                })

    return weekly, pd.DataFrame(summary_rows)
