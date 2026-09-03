from __future__ import annotations

import hashlib
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dashboard.skill_industry_eps_known import (
    effective_eps,
    is_review_universe,
    rank_skill_industry_eps_known,
    reasoned_item,
    select_skill_industry_eps_known,
)

from .config import PANEL_SOURCE, PRICE_CACHE, PRODUCTION_B0_PATH, SNAPSHOT_FORWARD_DAYS


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_panel() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL_SOURCE).copy()
    required = {
        "snapshot_date",
        "code",
        "signal",
        "ibd_candidate_rule",
        "industry",
        "w4_return_pct",
        "w4_stop8",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise RuntimeError(f"B0 absolute audit panel missing columns: {missing}")

    panel["snapshot_date"] = panel["snapshot_date"].astype(str)
    panel["code"] = panel["code"].astype(str).str.upper().str.strip()

    # The panel is expected to already be the raw Review Universe. Verify rather
    # than silently trusting the old b0_eligible/is_b0 helper columns.
    bad = []
    for _, row in panel.iterrows():
        if not is_review_universe(row):
            bad.append((str(row["snapshot_date"]), str(row["code"])))
            if len(bad) >= 10:
                break
    if bad:
        raise RuntimeError(
            "Candidate panel contains rows outside raw Review Universe; "
            f"first examples={bad}"
        )
    return panel


def current_b0_eligibility(item) -> tuple[bool, list[str]]:
    """Exact current Production selection gates, excluding industry de-duplication."""
    reasons: list[str] = []
    if item.entry_status != "ACTIONABLE":
        reasons.append("non_actionable")
    if "clear_geometry_failure" in item.risk_codes:
        reasons.append("clear_geometry_failure")
    if "below_candidate_buy_point" in item.risk_codes:
        reasons.append("below_candidate_buy_point")
    if effective_eps(item) is None:
        reasons.append("eps_unknown")
    if not str(item.industry or "").strip():
        reasons.append("industry_missing")
    return (len(reasons) == 0), reasons


def current_b0_eligible(row: pd.Series, row_idx: int) -> bool:
    """Recompute current Production eligibility without reading panel.b0_eligible."""
    item = reasoned_item(row, row_idx)
    eligible, _ = current_b0_eligibility(item)
    return eligible


def add_current_b0_state(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach current Production rank/eligibility/selection to each frozen PIT row."""
    parts: list[pd.DataFrame] = []

    for snapshot in sorted(panel["snapshot_date"].unique().tolist()):
        s_df = panel[panel["snapshot_date"] == snapshot].copy()
        ranked = rank_skill_industry_eps_known(s_df)
        selected = select_skill_industry_eps_known(s_df, limit=3)

        rank_map = {str(item.code).upper(): int(item.raw_rank) for item in ranked}
        lane_map = {str(item.code).upper(): str(item.lane) for item in ranked}
        selected_order = {
            str(item.code).upper(): idx + 1
            for idx, item in enumerate(selected)
        }

        elig: dict[str, bool] = {}
        reject_map: dict[str, str] = {}
        for local_idx, (_, row) in enumerate(s_df.iterrows()):
            code = str(row["code"]).upper()
            item = reasoned_item(row, local_idx)
            is_eligible, reasons = current_b0_eligibility(item)
            elig[code] = is_eligible
            reject_map[code] = "|".join(reasons)

        s_df["current_b0_raw_rank"] = s_df["code"].map(rank_map)
        s_df["current_b0_lane"] = s_df["code"].map(lane_map)
        s_df["current_b0_eligible"] = s_df["code"].map(elig).fillna(False).astype(bool)
        s_df["current_b0_reject_reasons"] = s_df["code"].map(reject_map).fillna("")
        s_df["current_b0_selected"] = s_df["code"].isin(selected_order)
        s_df["current_b0_pick_order"] = s_df["code"].map(selected_order)

        bad_selected = s_df[
            s_df["current_b0_selected"]
            & ~s_df["current_b0_eligible"]
        ]
        if not bad_selected.empty:
            raise RuntimeError(
                f"Production selected non-eligible names on {snapshot}: "
                f"{bad_selected['code'].tolist()}"
            )

        parts.append(s_df)

    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["snapshot_date", "code"]).reset_index(drop=True)


def load_price_cache() -> pd.DataFrame:
    prices = pd.read_parquet(PRICE_CACHE).copy()
    required = {"date", "code", "close", "low"}
    missing = sorted(required - set(prices.columns))
    if missing:
        raise RuntimeError(f"Price cache missing columns: {missing}")
    prices["date"] = pd.to_datetime(prices["date"])
    prices["code"] = prices["code"].astype(str).str.upper().str.strip()
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices["low"] = pd.to_numeric(prices["low"], errors="coerce")
    return prices.sort_values(["code", "date"]).reset_index(drop=True)


def _asof_index(dates: np.ndarray, target: np.datetime64) -> int:
    idx = int(np.searchsorted(dates, target, side="right") - 1)
    return idx


def build_snapshot_forward_returns(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    extra_codes: tuple[str, ...] = ("SPY", "QQQ"),
) -> pd.DataFrame:
    """Compute common snapshot-close to +28 calendar-day returns from frozen prices.

    Start price is the last close on/before snapshot date, with max 4 calendar days
    staleness. End price is the last close on/before snapshot+28d, also max 4 days
    stale relative to target. Stop8 is measured from start close using subsequent
    daily lows through end date; same-day pre-close lows are intentionally excluded.
    """
    events = panel[["snapshot_date", "code"]].drop_duplicates().copy()
    snapshots = sorted(panel["snapshot_date"].unique().tolist())
    extras = pd.DataFrame(
        [(snap, code) for snap in snapshots for code in extra_codes],
        columns=["snapshot_date", "code"],
    )
    events = pd.concat([events, extras], ignore_index=True).drop_duplicates()
    events["snapshot_date"] = events["snapshot_date"].astype(str)
    events["code"] = events["code"].astype(str).str.upper().str.strip()

    rows: list[dict[str, Any]] = []
    grouped_prices = {code: g.copy() for code, g in prices.groupby("code", sort=False)}

    for code, ev in events.groupby("code", sort=False):
        g = grouped_prices.get(code)
        if g is None or g.empty:
            for snap in ev["snapshot_date"]:
                rows.append({
                    "snapshot_date": snap,
                    "code": code,
                    "snapshot_w4_return_pct": np.nan,
                    "snapshot_w4_stop8": np.nan,
                    "snapshot_price_start_date": None,
                    "snapshot_price_end_date": None,
                    "snapshot_price_valid": False,
                })
            continue

        gd = g["date"].to_numpy(dtype="datetime64[ns]")
        gc = g["close"].to_numpy(dtype=float)
        gl = g["low"].to_numpy(dtype=float)

        for snap in ev["snapshot_date"]:
            snap_ts = pd.Timestamp(snap)
            target_ts = snap_ts + pd.Timedelta(days=SNAPSHOT_FORWARD_DAYS)
            snap64 = np.datetime64(snap_ts.to_datetime64())
            target64 = np.datetime64(target_ts.to_datetime64())
            si = _asof_index(gd, snap64)
            ei = _asof_index(gd, target64)

            valid = True
            # A +28d outcome is mature only if the frozen cache actually reaches
            # the target date. Weekend/holiday staleness is allowed only when the
            # target is inside the observed cache range; never use the cache's
            # final pre-target bar to pretend a still-future horizon is mature.
            if target_ts > pd.Timestamp(gd[-1]):
                valid = False
            if si < 0 or ei <= si:
                valid = False
            if valid:
                start_date = pd.Timestamp(gd[si])
                end_date = pd.Timestamp(gd[ei])
                if (snap_ts - start_date).days > 4:
                    valid = False
                if (target_ts - end_date).days > 4:
                    valid = False
                start_close = float(gc[si])
                end_close = float(gc[ei])
                if (
                    not math.isfinite(start_close)
                    or not math.isfinite(end_close)
                    or start_close <= 0
                ):
                    valid = False

            if not valid:
                rows.append({
                    "snapshot_date": snap,
                    "code": code,
                    "snapshot_w4_return_pct": np.nan,
                    "snapshot_w4_stop8": np.nan,
                    "snapshot_price_start_date": None if si < 0 else str(pd.Timestamp(gd[si]).date()),
                    "snapshot_price_end_date": None if ei < 0 else str(pd.Timestamp(gd[ei]).date()),
                    "snapshot_price_valid": False,
                })
                continue

            future_lows = gl[si + 1 : ei + 1]
            finite_lows = future_lows[np.isfinite(future_lows)]
            stop8 = bool(
                len(finite_lows) > 0
                and float(np.min(finite_lows)) <= start_close * 0.92
            )
            ret = (end_close / start_close - 1.0) * 100.0

            rows.append({
                "snapshot_date": snap,
                "code": code,
                "snapshot_w4_return_pct": round(float(ret), 6),
                "snapshot_w4_stop8": stop8,
                "snapshot_price_start_date": str(start_date.date()),
                "snapshot_price_end_date": str(end_date.date()),
                "snapshot_price_valid": True,
            })

    return pd.DataFrame(rows)


def build_audit_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = add_current_b0_state(load_panel())
    prices = load_price_cache()
    forward = build_snapshot_forward_returns(panel, prices)

    merged = panel.merge(
        forward[
            [
                "snapshot_date",
                "code",
                "snapshot_w4_return_pct",
                "snapshot_w4_stop8",
                "snapshot_price_valid",
            ]
        ],
        on=["snapshot_date", "code"],
        how="left",
        validate="one_to_one",
    )
    benchmarks = forward[forward["code"].isin(["SPY", "QQQ"])].copy()
    return merged, benchmarks, prices


def source_manifest(panel: pd.DataFrame) -> dict[str, Any]:
    return {
        "source_git_sha": git_sha(),
        "panel_hash": sha256_file(PANEL_SOURCE),
        "price_cache_hash": sha256_file(PRICE_CACHE),
        "production_b0_hash": sha256_file(PRODUCTION_B0_PATH),
        "snapshot_count": int(panel["snapshot_date"].nunique()),
        "panel_min_snapshot": str(panel["snapshot_date"].min()),
        "panel_max_snapshot": str(panel["snapshot_date"].max()),
        "review_rows": int(len(panel)),
        "current_eligible_rows": int(panel["current_b0_eligible"].sum()),
        "current_selected_rows": int(panel["current_b0_selected"].sum()),
    }
