"""Point-in-Time (PIT) Production Eligibility Predicate for B0 Top3 Quality Audit.

This module provides a pure, read-only mirror of the eligibility criteria defined in
`dashboard/skill_industry_eps_known.py`. It is strictly used by the backtesting and
ranking alpha decoupling pipeline to construct the Level 1 (Production Eligible) universe
without modifying production code.
"""

from __future__ import annotations

import math
from typing import Any
import pandas as pd

from backtest.replay_eps import get_replay_signal_eps


def to_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "<na>", "nat"}:
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def to_bool(value: object) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0"}:
        return True
    if text in {"false", "0", "0.0"}:
        return False
    return None


def get_effective_eps_pit(row: pd.Series | dict[str, Any], code: str) -> float | None:
    """Retrieve EPS YoY growth with Point-in-Time fallback if available."""
    eps = to_float(row.get("eps_yoy_growth"))
    if eps is not None:
        return eps
    snapshot = str(row.get("snapshot_date", "") or "").strip()
    if not snapshot or not code:
        return None
    try:
        return get_replay_signal_eps(snapshot, code, allow_network=False)
    except Exception:
        return None


def is_production_eligible_pit(row: pd.Series | dict[str, Any]) -> bool:
    """Check if a candidate row meets all Point-in-Time eligibility conditions for B0.
    
    Conditions mirrored exactly from `dashboard/skill_industry_eps_known.py`:
    1. Signal Universe: `signal == True` and `ibd_candidate_rule` is non-empty.
    2. Status: `ibd_entry_status == "ACTIONABLE"`.
    3. Geometry: No clear geometry failure (close_pos >= 0.65, breakout_range_ratio > 0).
    4. Buy-point proximity: `current_vs_ibd_candidate_pct >= 0` and is not None.
    5. EPS: Effective EPS (including PIT fallback) is known (not None).
    6. Industry: Valid non-empty industry string.
    """
    # 1. Review universe check
    sig = to_bool(row.get("signal"))
    rule = str(row.get("ibd_candidate_rule", "") or "").strip()
    if sig is not True or not rule:
        return False

    # 2. Actionable status check
    status = str(row.get("ibd_entry_status", "") or "").strip().upper()
    if status != "ACTIONABLE":
        return False

    # 3. Geometry check (clear_geometry_failure)
    pos = to_float(row.get("ibd_entry_close_position"))
    rr = to_float(row.get("ibd_entry_breakout_range_ratio"))
    if rr is not None and rr <= 0:
        return False
    if pos is not None and pos < 0.65:
        return False

    # 4. Buy-point proximity check (below_candidate_buy_point / freshness_missing)
    cur = to_float(row.get("current_vs_ibd_candidate_pct"))
    if cur is None or cur < 0:
        return False

    # 5. Effective EPS check
    code = str(row.get("code", "") or "").strip()
    eps = get_effective_eps_pit(row, code)
    if eps is None:
        return False

    # 6. Industry check
    industry = str(row.get("industry", "") or "").strip()
    if not industry or industry.lower() in {"nan", "none", "<na>"}:
        return False

    return True


def extract_pit_eligible_universe(events_df: pd.DataFrame) -> pd.DataFrame:
    """Filter an events dataframe down to strictly production-eligible candidates."""
    mask = events_df.apply(is_production_eligible_pit, axis=1)
    return events_df[mask].copy()
