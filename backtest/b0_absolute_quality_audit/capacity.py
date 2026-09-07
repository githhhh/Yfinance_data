from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .config import CAPACITY_POLICY_IDS, TOP_N
from .portfolio import industry_key, portfolio_from_codes


def _original_codes(s_df: pd.DataFrame) -> list[str]:
    selected = s_df[s_df["current_b0_selected"]].copy()
    if selected.empty:
        return []
    return (
        selected.sort_values("current_b0_pick_order")["code"]
        .astype(str)
        .tolist()
    )


def _reject_reasons(row: pd.Series) -> list[str]:
    return [
        token
        for token in str(row.get("current_b0_reject_reasons", "") or "").split("|")
        if token
    ]


def underfill_cause(s_df: pd.DataFrame) -> str:
    original = _original_codes(s_df)
    if len(original) >= TOP_N:
        return "FULL"

    eligible = s_df[s_df["current_b0_eligible"]].copy()
    eligible_count = int(len(eligible))
    distinct_industries = {
        industry_key(row)
        for _, row in eligible.iterrows()
        if str(row.get("industry", "") or "").strip()
    }

    if eligible_count < TOP_N:
        return "ELIGIBILITY_SHORTAGE"
    if len(distinct_industries) < TOP_N:
        return "INDUSTRY_CONSTRAINT"
    return "UNEXPECTED_SELECTOR_UNDERFILL"


def fill_capacity_codes(s_df: pd.DataFrame, policy_id: str) -> list[str]:
    """Preserve Production picks and only fill empty slots.

    None of these policies may displace an original B0 pick. They are diagnostic
    counterfactuals, not Production recommendations.
    """
    if policy_id not in CAPACITY_POLICY_IDS:
        raise RuntimeError(f"Unknown capacity policy: {policy_id}")

    selected = _original_codes(s_df)
    if policy_id == "B0_ORIGINAL" or len(selected) >= TOP_N:
        return selected

    used = {
        industry_key(row)
        for _, row in s_df[s_df["code"].isin(selected)].iterrows()
    }

    work = s_df[~s_df["code"].isin(selected)].copy()
    work = work[
        pd.to_numeric(work["current_b0_raw_rank"], errors="coerce").notna()
    ].copy()
    work["_rank"] = pd.to_numeric(work["current_b0_raw_rank"], errors="coerce")
    work = work.sort_values(["_rank", "code"], kind="stable")

    for _, row in work.iterrows():
        if len(selected) >= TOP_N:
            break

        is_eligible = bool(row.get("current_b0_eligible", False))
        reasons = _reject_reasons(row)
        known_industry = bool(str(row.get("industry", "") or "").strip())
        ind = industry_key(row)

        if policy_id == "B0_FILL3_RELAX_INDUSTRY":
            if not is_eligible:
                continue
            # This policy isolates distinct_1 only.
            selected.append(str(row["code"]))
            continue

        if not known_industry:
            # The remaining soft-gate diagnostics keep industry metadata and
            # distinct_1 intact. industry_missing is therefore never silently
            # treated as a valid diversified fill.
            continue
        if ind in used:
            continue

        if policy_id == "B0_FILL3_EPS_ONLY":
            if reasons != ["eps_unknown"]:
                continue
        elif policy_id == "B0_FILL3_SINGLE_REJECT":
            if len(reasons) != 1:
                continue
        elif policy_id == "B0_FILL3_ANY_REJECT":
            if len(reasons) < 1:
                continue

        selected.append(str(row["code"]))
        used.add(ind)

    return selected


def capacity_policy_weekly(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for snapshot in sorted(panel["snapshot_date"].astype(str).unique().tolist()):
        s_df = panel[panel["snapshot_date"].astype(str) == snapshot].copy()
        original_codes = _original_codes(s_df)
        cause = underfill_cause(s_df)

        for policy_id in CAPACITY_POLICY_IDS:
            codes = fill_capacity_codes(s_df, policy_id)
            port = portfolio_from_codes(
                s_df,
                codes,
                return_col="next_open_w4_return_pct",
                stop_col="next_open_w4_stop8",
            )

            added = [code for code in codes if code not in original_codes]
            added_port = portfolio_from_codes(
                s_df,
                added,
                return_col="next_open_w4_return_pct",
                stop_col="next_open_w4_stop8",
            )

            added_reason_map: dict[str, list[str]] = {}
            if added:
                lookup = s_df.set_index("code", drop=False)
                for code in added:
                    row = lookup.loc[code]
                    if isinstance(row, pd.DataFrame):
                        raise RuntimeError(f"Duplicate code in capacity audit: {code}")
                    added_reason_map[code] = _reject_reasons(row)

            rows.append({
                "snapshot_date": snapshot,
                "policy_id": policy_id,
                "underfill_cause": cause,
                "original_pick_count": len(original_codes),
                "pick_count": len(codes),
                "full3": len(codes) == TOP_N,
                "original_codes": json.dumps(original_codes),
                "codes": json.dumps(codes),
                "added_codes": json.dumps(added),
                "added_reasons": json.dumps(added_reason_map, sort_keys=True),
                "mature": bool(port["mature"]),
                "capital_adjusted_return": (
                    port["capital_adjusted_return"] if port["mature"] else None
                ),
                "selection_quality_return": (
                    port["selection_quality_return"] if port["mature"] else None
                ),
                "capital_stop8_pct": (
                    port["capital_adjusted_stop8"] if port["mature"] else None
                ),
                "one_pick_ruined": (
                    port["one_pick_ruined"] if port["mature"] else None
                ),
                "added_pick_count": len(added),
                "added_selection_quality_return": (
                    added_port["selection_quality_return"]
                    if added and added_port["mature"]
                    else None
                ),
                "added_stop8_pct": (
                    added_port["capital_adjusted_stop8"] * TOP_N / max(1, len(added))
                    if added and added_port["mature"]
                    else None
                ),
            })

    return pd.DataFrame(rows)
