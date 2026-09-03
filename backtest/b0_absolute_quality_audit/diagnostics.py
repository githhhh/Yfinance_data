from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from .metrics import four_offset_nonoverlap


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def _pick_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "picks": 0,
            "mean_w4": None,
            "median_w4": None,
            "p10_w4": None,
            "cvar10_w4": None,
            "positive_rate": None,
            "stop8_rate": None,
            "terminal_le_minus8_rate": None,
        }

    rets = np.asarray([float(r["ret"]) for r in rows], dtype=float)
    p10 = float(np.percentile(rets, 10))
    tail = rets[rets <= p10]
    return {
        "picks": int(len(rows)),
        "mean_w4": float(np.mean(rets)),
        "median_w4": float(np.median(rets)),
        "p10_w4": p10,
        "cvar10_w4": float(np.mean(tail)) if len(tail) else None,
        "positive_rate": float(np.mean(rets > 0)),
        "stop8_rate": float(np.mean([bool(r["stop"]) for r in rows])),
        "terminal_le_minus8_rate": float(np.mean(rets <= -8.0)),
    }


def capacity_pick_quality(
    panel: pd.DataFrame,
    capacity_weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate per-pick quality from portfolio-size effects.

    Only mature underfilled weeks are included. Original B0 picks and fill picks
    are summarized independently so an increase from 1->3 positions cannot by
    itself masquerade as worse per-pick risk.
    """
    if panel.empty or capacity_weekly.empty:
        return pd.DataFrame(), pd.DataFrame()

    panel = panel.copy()
    panel["snapshot_date"] = panel["snapshot_date"].astype(str)
    panel["code"] = panel["code"].astype(str)
    lookup = panel.set_index(["snapshot_date", "code"], drop=False)
    summary_rows: list[dict[str, Any]] = []
    reason_rows: list[dict[str, Any]] = []

    for policy_id, group in capacity_weekly.groupby("policy_id", sort=False):
        if policy_id == "B0_ORIGINAL":
            continue

        group = group[
            (pd.to_numeric(group["original_pick_count"], errors="coerce") < 3)
            & (group["mature"] == True)
        ].copy()

        original_pick_rows: list[dict[str, Any]] = []
        added_pick_rows: list[dict[str, Any]] = []
        added_by_reason: dict[str, list[dict[str, Any]]] = {}

        for _, week in group.iterrows():
            snapshot = str(week["snapshot_date"])
            original_codes = json.loads(str(week.get("original_codes", "[]") or "[]"))
            added_codes = json.loads(str(week.get("added_codes", "[]") or "[]"))

            for cohort, codes in [
                ("original_b0", original_codes),
                ("added_fill", added_codes),
            ]:
                for code in codes:
                    key = (snapshot, str(code))
                    if key not in lookup.index:
                        raise RuntimeError(f"Missing capacity pick row: {key}")
                    row = lookup.loc[key]
                    if isinstance(row, pd.DataFrame):
                        raise RuntimeError(f"Duplicate capacity pick row: {key}")
                    if not bool(row.get("next_open_price_valid", False)):
                        continue
                    item = {
                        "ret": float(row["next_open_w4_return_pct"]),
                        "stop": bool(row["next_open_w4_stop8"]),
                        "reason": str(row.get("current_b0_reject_reasons", "") or ""),
                    }
                    if cohort == "original_b0":
                        original_pick_rows.append(item)
                    else:
                        added_pick_rows.append(item)
                        reason = item["reason"] or "(none)"
                        added_by_reason.setdefault(reason, []).append(item)

        for cohort, rows in [
            ("original_b0", original_pick_rows),
            ("added_fill", added_pick_rows),
        ]:
            summary_rows.append({
                "policy_id": policy_id,
                "scope": "underfilled_only",
                "cohort": cohort,
                **_pick_stats(rows),
            })

        for reason, rows in sorted(added_by_reason.items()):
            reason_rows.append({
                "policy_id": policy_id,
                "scope": "underfilled_only",
                "reject_reason": reason,
                **_pick_stats(rows),
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(reason_rows)


def support_calendar_summary(
    snapshots: list[str],
    raw_weekly: pd.DataFrame,
    simple_weekly: pd.DataFrame,
) -> pd.DataFrame:
    """Expose temporal concentration of strict-support weeks."""
    base = pd.DataFrame({"snapshot_date": [str(x) for x in snapshots]})
    base["quarter"] = pd.PeriodIndex(
        pd.to_datetime(base["snapshot_date"]), freq="Q"
    ).astype(str)
    total = base.groupby("quarter").size().to_dict()

    rows: list[dict[str, Any]] = []

    raw = raw_weekly.copy()
    if not raw.empty:
        raw["quarter"] = pd.PeriodIndex(
            pd.to_datetime(raw["snapshot_date"]), freq="Q"
        ).astype(str)
        valid = raw[raw["primary_valid"] == True]
        counts = valid.groupby("quarter").size().to_dict()
        for quarter in sorted(total):
            rows.append({
                "comparison": "raw_fixed3_primary",
                "quarter": quarter,
                "support_weeks": int(counts.get(quarter, 0)),
                "total_snapshots": int(total[quarter]),
                "support_rate": float(counts.get(quarter, 0) / total[quarter]),
            })

    if not simple_weekly.empty:
        for baseline in sorted(simple_weekly["baseline"].astype(str).unique()):
            work = simple_weekly[simple_weekly["baseline"].astype(str) == baseline].copy()
            work["quarter"] = pd.PeriodIndex(
                pd.to_datetime(work["snapshot_date"]), freq="Q"
            ).astype(str)
            valid = work[work["primary_valid"] == True]
            counts = valid.groupby("quarter").size().to_dict()
            for quarter in sorted(total):
                rows.append({
                    "comparison": f"simple_{baseline}",
                    "quarter": quarter,
                    "support_weeks": int(counts.get(quarter, 0)),
                    "total_snapshots": int(total[quarter]),
                    "support_rate": float(counts.get(quarter, 0) / total[quarter]),
                })

    return pd.DataFrame(rows)


def momentum_gate_diagnostics(
    panel: pd.DataFrame,
    simple_weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Locate where the raw momentum baseline's picks sit relative to B0 gates."""
    if panel.empty or simple_weekly.empty:
        return pd.DataFrame(), pd.DataFrame()

    weeks = simple_weekly[
        (simple_weekly["baseline"].astype(str) == "momentum_20")
        & (simple_weekly["primary_valid"] == True)
    ].copy()
    if weeks.empty:
        return pd.DataFrame(), pd.DataFrame()

    panel = panel.copy()
    panel["snapshot_date"] = panel["snapshot_date"].astype(str)
    panel["code"] = panel["code"].astype(str)
    lookup = panel.set_index(["snapshot_date", "code"], drop=False)
    picks: list[dict[str, Any]] = []

    for _, week in weeks.iterrows():
        snapshot = str(week["snapshot_date"])
        for code in json.loads(str(week.get("codes", "[]") or "[]")):
            key = (snapshot, str(code))
            if key not in lookup.index:
                raise RuntimeError(f"Missing momentum pick row: {key}")
            row = lookup.loc[key]
            if isinstance(row, pd.DataFrame):
                raise RuntimeError(f"Duplicate momentum pick row: {key}")
            if not bool(row.get("next_open_price_valid", False)):
                continue
            eligible = bool(row.get("current_b0_eligible", False))
            selected = bool(row.get("current_b0_selected", False))
            picks.append({
                "snapshot_date": snapshot,
                "code": str(code),
                "cohort": "eligible" if eligible else "gate_outside",
                "eligible": eligible,
                "selected": selected,
                "reason": str(row.get("current_b0_reject_reasons", "") or ""),
                "ret": float(row["next_open_w4_return_pct"]),
                "stop": bool(row["next_open_w4_stop8"]),
            })

    summary_rows: list[dict[str, Any]] = []
    for cohort in ["all", "eligible", "gate_outside"]:
        rows = picks if cohort == "all" else [r for r in picks if r["cohort"] == cohort]
        stats = _pick_stats(rows)
        summary_rows.append({
            "cohort": cohort,
            **stats,
            "share_of_momentum_picks": (
                None if not picks else float(len(rows) / len(picks))
            ),
            "selected_by_b0_rate": (
                None if not rows else float(np.mean([r["selected"] for r in rows]))
            ),
        })

    reason_rows: list[dict[str, Any]] = []
    outside = [r for r in picks if r["cohort"] == "gate_outside"]
    reasons = sorted({r["reason"] or "(none)" for r in outside})
    for reason in reasons:
        rows = [r for r in outside if (r["reason"] or "(none)") == reason]
        reason_rows.append({
            "reject_reason": reason,
            **_pick_stats(rows),
            "share_of_gate_outside_momentum": (
                None if not outside else float(len(rows) / len(outside))
            ),
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(reason_rows)


def momentum_nonoverlap(
    simple_weekly: pd.DataFrame,
    b0_weekly: pd.DataFrame,
) -> pd.DataFrame:
    if simple_weekly.empty or b0_weekly.empty:
        return pd.DataFrame()

    mom = simple_weekly[
        (simple_weekly["baseline"].astype(str) == "momentum_20")
        & (simple_weekly["primary_valid"] == True)
    ][["snapshot_date", "return"]].copy()
    b0 = b0_weekly[["snapshot_date", "next_open_capital_adjusted"]].copy()
    merged = mom.merge(b0, on="snapshot_date", how="inner")
    merged = merged.dropna(subset=["return", "next_open_capital_adjusted"])
    if merged.empty:
        return pd.DataFrame()

    out = four_offset_nonoverlap(
        merged,
        value_col="return",
        benchmark_col="next_open_capital_adjusted",
    )
    out["comparison"] = "momentum20_vs_b0_next_open"
    return out
