from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backtest.rd_agent_candidate_rule_audit.utils import to_bool, to_float


PULLBACK_RULES = {"ceiling_pullback", "pivot", "ma10_touch_confirm", "three_weeks_tight"}


@dataclass(frozen=True)
class ReviewRule:
    name: str
    near_below_pct: float = 5.0
    extended_above_pct: float = 10.0
    min_support_count: int = 0


def review_rules() -> dict[str, ReviewRule]:
    """Pre-registered primary variants from the research protocol."""
    return {
        "R1_PATH": ReviewRule("R1_PATH", min_support_count=0),
        "R2_BALANCED": ReviewRule("R2_BALANCED", min_support_count=1),
        "R3_STRICT": ReviewRule("R3_STRICT", min_support_count=2),
    }


def is_review_universe(row: pd.Series) -> bool:
    return (
        to_bool(row.get("signal")) is True
        and bool(str(row.get("ibd_candidate_rule", "") or "").strip())
    )


def clear_geometry_failure(row: pd.Series) -> bool:
    """Only explicit observed geometry failure is a hard structural reject.

    Missing geometry remains UNKNOWN and is not converted to failure.
    """
    rr = to_float(row.get("ibd_entry_breakout_range_ratio"))
    pos = to_float(row.get("ibd_entry_close_position"))
    if rr is not None and rr <= 0:
        return True
    if pos is not None and pos < 0.65:
        return True
    return False


def support_flags(row: pd.Series) -> dict[str, bool]:
    """Positive-only evidence cluster.

    False or missing evidence is neutral. In particular, pullback_v_is_dry=False
    is not a negative score and PIT EPS missing is not a failure.
    """
    entry_vol = to_float(row.get("ibd_entry_volume_ratio"))
    weekly_vol = to_float(row.get("volume_ratio"))
    eps_state = str(row.get("pit_eps_state", "") or "").strip().upper()
    pit_eps = to_float(row.get("pit_eps_yoy_growth")) if eps_state == "VERIFIED" else None
    dist = to_float(row.get("dist_to_52w_high_pct"))
    rule = str(row.get("ibd_candidate_rule", "") or "").strip()
    dry = to_bool(row.get("pullback_v_is_dry"))

    return {
        "entry_volume_confirmed": entry_vol is not None and entry_vol >= 1.5,
        "weekly_volume_follow_through": weekly_vol is not None and weekly_vol >= 1.3,
        "eps_support": pit_eps is not None and pit_eps >= 25.0,
        "near_52w_high": dist is not None and dist > -5.0,
        "dry_pullback": rule in PULLBACK_RULES and dry is True,
    }


def support_count(row: pd.Series) -> int:
    return sum(support_flags(row).values())


def path_eligible(row: pd.Series, rule: ReviewRule) -> bool:
    status = str(row.get("ibd_entry_status", "") or "").strip().upper()
    cur = to_float(row.get("current_vs_ibd_candidate_pct"))

    if status == "ACTIONABLE":
        return True
    if cur is None:
        return False
    if status == "UNCONFIRMED":
        return -rule.near_below_pct <= cur <= 5.0
    if status == "BELOW_TRIGGER":
        return -rule.near_below_pct <= cur < 0.0
    if status == "EXTENDED":
        return 5.0 < cur <= rule.extended_above_pct
    return False


def _enrich(pool: pd.DataFrame) -> pd.DataFrame:
    frame = pool.copy()
    frame["_source_row_order"] = np.arange(len(frame))
    mask = frame.apply(is_review_universe, axis=1)
    frame = frame.loc[mask].copy()
    frame["_geometry_failure"] = frame.apply(clear_geometry_failure, axis=1)
    frame["_support_count"] = frame.apply(support_count, axis=1)
    frame["_vs_buy_point"] = frame["current_vs_ibd_candidate_pct"].map(to_float)
    frame["_abs_vs_buy_point"] = frame["_vs_buy_point"].map(
        lambda value: abs(value) if value is not None else float("inf")
    )
    frame["_status"] = (
        frame["ibd_entry_status"].fillna("").astype(str).str.strip().str.upper()
    )
    frame["_code_sort"] = frame["code"].fillna("").astype(str).str.strip().str.upper()
    return frame


def select_b0_actionable(pool: pd.DataFrame) -> pd.DataFrame:
    """Actual weekend sync baseline: all active-signal ACTIONABLE rows."""
    frame = _enrich(pool)
    selected = frame.loc[frame["_status"].eq("ACTIONABLE")].copy()
    selected["variant"] = "B0_ACTIONABLE_ONLY"
    selected["review_reason"] = "weekend_actionable"
    return _stable_output(selected)


def select_review_variant(
    pool: pd.DataFrame,
    rule: ReviewRule,
    *,
    cap: int | None = None,
) -> pd.DataFrame:
    frame = _enrich(pool)
    eligible = frame.apply(lambda row: path_eligible(row, rule), axis=1)
    selected = frame.loc[
        eligible
        & ~frame["_geometry_failure"]
        & frame["_support_count"].ge(rule.min_support_count)
    ].copy()
    selected["variant"] = rule.name
    selected["review_reason"] = selected.apply(_review_reason, axis=1)
    selected = _priority_sort(selected)
    if cap is not None:
        selected = selected.head(max(int(cap), 0)).copy()
    return _stable_output(selected)


def select_attention_matched(
    pool: pd.DataFrame,
    rule: ReviewRule,
) -> pd.DataFrame:
    """Select the same N as the B0 ACTIONABLE count for a fair attention control."""
    n = len(select_b0_actionable(pool))
    selected = select_review_variant(pool, rule, cap=n)
    selected["variant"] = f"{rule.name}_ATTENTION_MATCHED"
    return selected


def _review_reason(row: pd.Series) -> str:
    status = str(row.get("_status", ""))
    cur = to_float(row.get("_vs_buy_point"))
    if status == "ACTIONABLE":
        return "current_actionable"
    if status == "EXTENDED":
        return "extended_retest_candidate"
    if status == "BELOW_TRIGGER":
        return "below_near_buy_point"
    if status == "UNCONFIRMED" and cur is not None and cur < 0:
        return "unconfirmed_pre_breakout"
    if status == "UNCONFIRMED":
        return "unconfirmed_in_buy_zone"
    return "review_candidate"


def _priority_sort(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministic review priority without C Rank and without status-first sorting."""
    if frame.empty:
        return frame
    return frame.sort_values(
        ["_support_count", "_abs_vs_buy_point", "_code_sort", "_source_row_order"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )


def _stable_output(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    return frame.reset_index(drop=True)


def variant_diagnostics(selected: pd.DataFrame) -> dict[str, Any]:
    if selected.empty:
        return {
            "selected": 0,
            "mean_support_count": np.nan,
            "median_abs_vs_buy_point": np.nan,
        }
    return {
        "selected": int(len(selected)),
        "mean_support_count": float(selected["_support_count"].mean()),
        "median_abs_vs_buy_point": float(selected["_abs_vs_buy_point"].median()),
    }
