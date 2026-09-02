from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.track_c_ranking_discovery.evaluate_econometrics import evaluate_paired_challenger
from backtest.track_d_mechanism_discovery.mechanism_lab import run_policy
from backtest.track_d_mechanism_discovery.walk_forward import build_locked_forward_split

from .config import (
    BASELINE_POLICY_ID,
    CHALLENGER_POLICY_ID,
    PANEL_SOURCE,
    PRIMARY_HORIZON,
    PRODUCTION_B0_PATH,
    TRACK_D_HISTORICAL_END,
)
from .policy import SoftActiveLaneChallenger, baseline_policy


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(PANEL_SOURCE)
    if "snapshot_date" not in df.columns or "code" not in df.columns:
        raise RuntimeError("Track E panel missing snapshot_date/code")
    return df.copy()


def build_segments(panel: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, Any]]:
    historical = panel[
        panel["snapshot_date"].astype(str) <= TRACK_D_HISTORICAL_END
    ].copy()
    snaps = sorted(historical["snapshot_date"].astype(str).unique().tolist())
    split = build_locked_forward_split(snaps)

    screening = [
        s for b in split["forward_blocks"]
        if b["stage"] == "screening"
        for s in b["snapshots"]
    ]
    confirmation = [
        s for b in split["forward_blocks"]
        if b["stage"] == "confirmation"
        for s in b["snapshots"]
    ]
    all_panel_snaps = sorted(panel["snapshot_date"].astype(str).unique().tolist())
    post_shadow = [s for s in all_panel_snaps if s > TRACK_D_HISTORICAL_END]

    segments = {
        "discovery_train_18": list(split["discovery_train"]),
        "purge_4": list(split["purge"]),
        "screening_6": screening,
        "confirmation_12": confirmation,
        "locked_forward_18": screening + confirmation,
        "retrospective_all_40": list(split["all_used_snapshots"]),
        "post_track_d_shadow": post_shadow,
    }
    return segments, split


def _summary_row(segment: str, challenger_outcomes, baseline_outcomes) -> dict[str, Any]:
    summary = evaluate_paired_challenger(
        challenger_outcomes,
        baseline_outcomes,
        CHALLENGER_POLICY_ID,
        "track_e_soft_lane",
        segment,
        PRIMARY_HORIZON,
    )
    raw = dataclasses.asdict(summary)
    row = {
        "segment": segment,
        "support_weeks": raw["support_weeks"],
        "challenger_mean": raw["challenger_mean"],
        "b0_mean": raw["b0_mean"],
        "mean_spread": raw["mean_spread"],
        "challenger_median": raw["challenger_median"],
        "b0_median": raw["b0_median"],
        "median_spread": raw["median_spread"],
        "challenger_cvar10": raw["challenger_cvar10"],
        "b0_cvar10": raw["b0_cvar10"],
        "cvar_delta": raw["cvar_delta"],
        "stop_delta_pct": raw["stop_delta_pct"],
        "one_pick_ruins_delta_pct": raw["one_pick_ruins_delta_pct"],
        "slot_coverage_pct": raw["slot_coverage_pct"],
        "full_top3_rate_pct": raw["full_top3_rate_pct"],
        "jaccard_vs_b0": raw["top3_membership_jaccard_vs_b0"],
        "ci_low": (raw["bootstrap"] or {}).get("mean_spread_ci_low", 0.0),
        "ci_high": (raw["bootstrap"] or {}).get("mean_spread_ci_high", 0.0),
        "lowo_sign_stability": (raw["lowo"] or {}).get("sign_stability", 0.0),
        "lowo_edge_concentration": (raw["lowo"] or {}).get("positive_edge_concentration", 0.0),
        "lowo_fragile": (raw["lowo"] or {}).get("is_fragile_overfit", False),
    }
    return row


def evaluate_segments(panel: pd.DataFrame, segments: dict[str, list[str]]) -> pd.DataFrame:
    baseline = baseline_policy()
    challenger = SoftActiveLaneChallenger()
    rows = []
    for name, snaps in segments.items():
        b0 = run_policy(baseline, panel, snaps, BASELINE_POLICY_ID, PRIMARY_HORIZON)
        b1 = run_policy(challenger, panel, snaps, CHALLENGER_POLICY_ID, PRIMARY_HORIZON)
        rows.append(_summary_row(name, b1, b0))
    return pd.DataFrame(rows)


def _pick(policy, snapshot_df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    scored = policy.score_candidates(snapshot_df)
    quotas = policy.allocate_industries(scored)
    picks = policy.pick_stocks(scored, quotas)
    return picks, scored


def _lane_map(scored: pd.DataFrame) -> dict[str, str]:
    if scored.empty:
        return {}
    return dict(zip(scored["code"].astype(str), scored["lane"].astype(str)))


def _rank_map(scored: pd.DataFrame) -> dict[str, int]:
    if scored.empty:
        return {}
    return {
        str(row["code"]): int(row["raw_rank"])
        for _, row in scored.iterrows()
    }


def _returns(panel_snap: pd.DataFrame) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for _, row in panel_snap.iterrows():
        code = str(row["code"])
        value = pd.to_numeric(pd.Series([row.get("w4_return_pct")]), errors="coerce").iloc[0]
        out[code] = None if pd.isna(value) else float(value)
    return out


def _capital_return(codes: list[str], returns: dict[str, float | None]) -> float | None:
    vals = [returns.get(code) for code in codes]
    if any(v is None for v in vals):
        return None
    return round(float(sum(v for v in vals if v is not None) / 3.0), 4)


def _mean_return(codes: list[str], returns: dict[str, float | None]) -> float | None:
    if not codes:
        return None
    vals = [returns.get(code) for code in codes]
    if any(v is None for v in vals):
        return None
    return round(float(np.mean([v for v in vals if v is not None])), 4)


def _segment_label(snapshot: str, split: dict[str, Any]) -> str:
    if snapshot in split["discovery_train"]:
        return "discovery_train"
    if snapshot in split["purge"]:
        return "purge"
    for block in split["forward_blocks"]:
        if snapshot in block["snapshots"]:
            return str(block["stage"])
    return "post_track_d_shadow"


def collect_selection_events(panel: pd.DataFrame, split: dict[str, Any]) -> pd.DataFrame:
    baseline = baseline_policy()
    challenger = SoftActiveLaneChallenger()
    rows: list[dict[str, Any]] = []

    for snap in sorted(panel["snapshot_date"].astype(str).unique().tolist()):
        s_df = panel[panel["snapshot_date"].astype(str) == snap].copy()
        b0_codes, b0_scored = _pick(baseline, s_df)
        b1_codes, b1_scored = _pick(challenger, s_df)

        b0_set = set(b0_codes)
        b1_set = set(b1_codes)
        incoming = [c for c in b1_codes if c not in b0_set]
        outgoing = [c for c in b0_codes if c not in b1_set]

        b0_lanes = _lane_map(b0_scored)
        b1_lanes = _lane_map(b1_scored)
        b0_ranks = _rank_map(b0_scored)
        b1_ranks = _rank_map(b1_scored)
        returns = _returns(s_df)

        incoming_standard = [c for c in incoming if b1_lanes.get(c) == "standard_breakout"]
        outgoing_fresh = [c for c in outgoing if b0_lanes.get(c) == "fresh_demand_alpha"]
        targeted = bool(incoming_standard and outgoing_fresh)

        b0_ret = _capital_return(b0_codes, returns)
        b1_ret = _capital_return(b1_codes, returns)
        target_in_ret = _mean_return(incoming_standard, returns)
        target_out_ret = _mean_return(outgoing_fresh, returns)
        target_delta = (
            None
            if target_in_ret is None or target_out_ret is None
            else round(target_in_ret - target_out_ret, 4)
        )

        rows.append({
            "snapshot_date": snap,
            "segment": _segment_label(snap, split),
            "selection_changed": b0_codes != b1_codes,
            "target_standard_over_fresh": targeted,
            "baseline_codes": json.dumps(b0_codes),
            "challenger_codes": json.dumps(b1_codes),
            "incoming_codes": json.dumps(incoming),
            "outgoing_codes": json.dumps(outgoing),
            "incoming_lanes": json.dumps({c: b1_lanes.get(c, "") for c in incoming}, sort_keys=True),
            "outgoing_lanes": json.dumps({c: b0_lanes.get(c, "") for c in outgoing}, sort_keys=True),
            "baseline_ranks": json.dumps({c: b0_ranks.get(c) for c in b0_codes}, sort_keys=True),
            "challenger_ranks": json.dumps({c: b1_ranks.get(c) for c in b1_codes}, sort_keys=True),
            "incoming_standard_codes": json.dumps(incoming_standard),
            "outgoing_fresh_codes": json.dumps(outgoing_fresh),
            "baseline_w4_capital_return": b0_ret,
            "challenger_w4_capital_return": b1_ret,
            "portfolio_spread_w4": (
                None if b0_ret is None or b1_ret is None else round(b1_ret - b0_ret, 4)
            ),
            "incoming_standard_mean_w4": target_in_ret,
            "outgoing_fresh_mean_w4": target_out_ret,
            "target_pair_delta_w4": target_delta,
            "w4_mature": b0_ret is not None and b1_ret is not None,
        })

    return pd.DataFrame(rows)


def build_event_summary(events: pd.DataFrame) -> dict[str, Any]:
    changed = events[events["selection_changed"] == True].copy()
    target = events[events["target_standard_over_fresh"] == True].copy()
    mature_target = target[target["target_pair_delta_w4"].notna()].copy()
    post = events[
        (events["segment"] == "post_track_d_shadow")
        & (events["w4_mature"] == True)
    ].copy()

    return {
        "total_snapshots": int(len(events)),
        "selection_changed_weeks": int(len(changed)),
        "target_standard_over_fresh_weeks": int(len(target)),
        "target_mature_weeks": int(len(mature_target)),
        "target_mean_pair_delta_w4": (
            None if mature_target.empty else round(float(mature_target["target_pair_delta_w4"].mean()), 4)
        ),
        "target_median_pair_delta_w4": (
            None if mature_target.empty else round(float(mature_target["target_pair_delta_w4"].median()), 4)
        ),
        "target_positive_pair_ratio": (
            None if mature_target.empty else round(float((mature_target["target_pair_delta_w4"] > 0).mean()), 4)
        ),
        "changed_week_mean_portfolio_spread_w4": (
            None
            if changed["portfolio_spread_w4"].dropna().empty
            else round(float(changed["portfolio_spread_w4"].dropna().mean()), 4)
        ),
        "post_track_d_mature_weeks": int(len(post)),
        "post_track_d_changed_weeks": int((post["selection_changed"] == True).sum()),
    }


def build_manifest(panel: pd.DataFrame, split: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": "track_e_soft_lane_audit",
        "source_git_sha": git_sha(),
        "panel_hash": sha256_file(PANEL_SOURCE),
        "production_b0_hash": sha256_file(PRODUCTION_B0_PATH),
        "panel_snapshot_count": int(panel["snapshot_date"].astype(str).nunique()),
        "track_d_historical_end": TRACK_D_HISTORICAL_END,
        "historical_snapshots": len(split["all_used_snapshots"]),
        "hypothesis_status": (
            "post-Track-D hypothesis; <=2026-07-24 evidence is retrospective and "
            "must not be presented as untouched OOS"
        ),
        "baseline": {
            "lane": "hard B0_LANE",
            "dry": "symmetric True reward / False penalty",
            "selector": "distinct_1",
        },
        "challenger": {
            "dry": "True reward / False neutral",
            "active_lane_order": (
                "active guard -> status -> evidence/risk -> lane prior -> remaining B0 tie-breaks"
            ),
            "active_lanes": [
                "fresh_demand_alpha",
                "constructive_pullback",
                "standard_breakout",
            ],
            "downgraded_lanes": ["incomplete_evidence", "tail_risk"],
            "selector": "distinct_1",
        },
    }
