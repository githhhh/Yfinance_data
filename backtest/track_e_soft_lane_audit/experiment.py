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
    PROTOCOL_VERSION,
    TARGET_SOFT_LANES,
    TRACK_D_HISTORICAL_END,
)
from .policy import PairwiseSoftLaneChallenger, baseline_policy


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


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
    historical = panel[panel["snapshot_date"].astype(str) <= TRACK_D_HISTORICAL_END].copy()
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

    return {
        "discovery_train_18": list(split["discovery_train"]),
        "purge_4": list(split["purge"]),
        "screening_6": screening,
        "confirmation_12": confirmation,
        "locked_forward_18": screening + confirmation,
        "retrospective_all_40": list(split["all_used_snapshots"]),
        "post_track_d_shadow": post_shadow,
    }, split


def _summary_row(segment: str, challenger_outcomes, baseline_outcomes) -> dict[str, Any]:
    summary = evaluate_paired_challenger(
        challenger_outcomes,
        baseline_outcomes,
        CHALLENGER_POLICY_ID,
        "track_e_pairwise_soft_lane",
        segment,
        PRIMARY_HORIZON,
    )
    raw = dataclasses.asdict(summary)
    return {
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


def evaluate_segments(panel: pd.DataFrame, segments: dict[str, list[str]]) -> pd.DataFrame:
    baseline = baseline_policy()
    challenger = PairwiseSoftLaneChallenger()
    rows = []
    for name, snaps in segments.items():
        b0 = run_policy(baseline, panel, snaps, BASELINE_POLICY_ID, PRIMARY_HORIZON)
        b1 = run_policy(challenger, panel, snaps, CHALLENGER_POLICY_ID, PRIMARY_HORIZON)
        rows.append(_summary_row(name, b1, b0))
    return pd.DataFrame(rows)


def _pick(policy, snapshot_df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    scored = policy.score_candidates(snapshot_df)
    quotas = policy.allocate_industries(scored)
    return policy.pick_stocks(scored, quotas), scored


def _map(scored: pd.DataFrame, column: str) -> dict[str, Any]:
    if scored.empty or column not in scored.columns:
        return {}
    return dict(zip(scored["code"].astype(str), scored[column].tolist()))


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


def _rank_crossover_codes(scored: pd.DataFrame) -> tuple[list[str], list[str]]:
    if scored.empty:
        return [], []

    lanes = _map(scored, "lane")
    skeleton = {k: int(v) for k, v in _map(scored, "skeleton_rank").items()}
    final = {k: int(v) for k, v in _map(scored, "raw_rank").items()}
    standards = [c for c, lane in lanes.items() if lane == "standard_breakout"]
    fresh = [c for c, lane in lanes.items() if lane == "fresh_demand_alpha"]

    crossed_standard: set[str] = set()
    crossed_fresh: set[str] = set()
    for standard in standards:
        for fresh_code in fresh:
            if (
                skeleton.get(standard, 10**9) > skeleton.get(fresh_code, 10**9)
                and final.get(standard, 10**9) < final.get(fresh_code, 10**9)
            ):
                crossed_standard.add(standard)
                crossed_fresh.add(fresh_code)
    return sorted(crossed_standard), sorted(crossed_fresh)


def collect_selection_events(panel: pd.DataFrame, split: dict[str, Any]) -> pd.DataFrame:
    baseline = baseline_policy()
    challenger = PairwiseSoftLaneChallenger()
    rows: list[dict[str, Any]] = []

    for snap in sorted(panel["snapshot_date"].astype(str).unique().tolist()):
        s_df = panel[panel["snapshot_date"].astype(str) == snap].copy()
        b0_codes, b0_scored = _pick(baseline, s_df)
        b1_codes, b1_scored = _pick(challenger, s_df)

        lanes = _map(b1_scored, "lane")
        skeleton_ranks = {k: int(v) for k, v in _map(b1_scored, "skeleton_rank").items()}
        final_ranks = {k: int(v) for k, v in _map(b1_scored, "raw_rank").items()}

        non_target_drift = [
            code for code, lane in lanes.items()
            if lane not in TARGET_SOFT_LANES
            and skeleton_ranks.get(code) != final_ranks.get(code)
        ]
        if non_target_drift:
            raise RuntimeError(
                f"Track E isolation breach on {snap}; non-target ranks moved: {non_target_drift}"
            )

        b0_set = set(b0_codes)
        b1_set = set(b1_codes)
        incoming = [c for c in b1_codes if c not in b0_set]
        outgoing = [c for c in b0_codes if c not in b1_set]

        b0_lanes = _map(b0_scored, "lane")
        incoming_standard = [c for c in incoming if lanes.get(c) == "standard_breakout"]
        outgoing_fresh = [c for c in outgoing if b0_lanes.get(c) == "fresh_demand_alpha"]

        crossed_standard, crossed_fresh = _rank_crossover_codes(b1_scored)
        returns = _returns(s_df)

        b0_ret = _capital_return(b0_codes, returns)
        b1_ret = _capital_return(b1_codes, returns)
        target_in_ret = _mean_return(incoming_standard, returns)
        target_out_ret = _mean_return(outgoing_fresh, returns)
        rank_in_ret = _mean_return(crossed_standard, returns)
        rank_out_ret = _mean_return(crossed_fresh, returns)

        rows.append({
            "snapshot_date": snap,
            "segment": _segment_label(snap, split),
            "order_changed": b0_codes != b1_codes,
            "membership_changed": b0_set != b1_set,
            "target_selection_swap": bool(incoming_standard and outgoing_fresh),
            "target_rank_crossover": bool(crossed_standard and crossed_fresh),
            "baseline_codes": json.dumps(b0_codes),
            "challenger_codes": json.dumps(b1_codes),
            "incoming_codes": json.dumps(incoming),
            "outgoing_codes": json.dumps(outgoing),
            "incoming_lanes": json.dumps({c: lanes.get(c, "") for c in incoming}, sort_keys=True),
            "outgoing_lanes": json.dumps({c: b0_lanes.get(c, "") for c in outgoing}, sort_keys=True),
            "challenger_skeleton_ranks": json.dumps(skeleton_ranks, sort_keys=True),
            "challenger_final_ranks": json.dumps(final_ranks, sort_keys=True),
            "rank_crossed_standard_codes": json.dumps(crossed_standard),
            "rank_crossed_fresh_codes": json.dumps(crossed_fresh),
            "incoming_standard_codes": json.dumps(incoming_standard),
            "outgoing_fresh_codes": json.dumps(outgoing_fresh),
            "baseline_w4_capital_return": b0_ret,
            "challenger_w4_capital_return": b1_ret,
            "portfolio_spread_w4": (
                None if b0_ret is None or b1_ret is None else round(b1_ret - b0_ret, 4)
            ),
            "selection_standard_mean_w4": target_in_ret,
            "selection_fresh_mean_w4": target_out_ret,
            "selection_pair_delta_w4": (
                None
                if target_in_ret is None or target_out_ret is None
                else round(target_in_ret - target_out_ret, 4)
            ),
            "rank_standard_mean_w4": rank_in_ret,
            "rank_fresh_mean_w4": rank_out_ret,
            "rank_pair_delta_w4": (
                None
                if rank_in_ret is None or rank_out_ret is None
                else round(rank_in_ret - rank_out_ret, 4)
            ),
            "w4_mature": b0_ret is not None and b1_ret is not None,
        })

    return pd.DataFrame(rows)


def build_event_summary(events: pd.DataFrame) -> dict[str, Any]:
    membership = events[events["membership_changed"] == True].copy()
    order = events[events["order_changed"] == True].copy()
    selection_target = events[events["target_selection_swap"] == True].copy()
    rank_target = events[events["target_rank_crossover"] == True].copy()

    mature_selection = selection_target[selection_target["selection_pair_delta_w4"].notna()].copy()
    mature_rank = rank_target[rank_target["rank_pair_delta_w4"].notna()].copy()
    mature_membership = membership[membership["portfolio_spread_w4"].notna()].copy()
    post = events[
        (events["segment"] == "post_track_d_shadow")
        & (events["w4_mature"] == True)
    ].copy()

    def stats(frame: pd.DataFrame, column: str) -> tuple[float | None, float | None, float | None]:
        values = frame[column].dropna()
        if values.empty:
            return None, None, None
        return (
            round(float(values.mean()), 4),
            round(float(values.median()), 4),
            round(float((values > 0).mean()), 4),
        )

    selection_mean, selection_median, selection_positive = stats(
        mature_selection, "selection_pair_delta_w4"
    )
    rank_mean, rank_median, rank_positive = stats(mature_rank, "rank_pair_delta_w4")

    return {
        "total_snapshots": int(len(events)),
        "order_changed_weeks": int(len(order)),
        "membership_changed_weeks": int(len(membership)),
        "mature_membership_changed_weeks": int(len(mature_membership)),
        "target_rank_crossover_weeks": int(len(rank_target)),
        "target_rank_crossover_mature_weeks": int(len(mature_rank)),
        "target_rank_mean_pair_delta_w4": rank_mean,
        "target_rank_median_pair_delta_w4": rank_median,
        "target_rank_positive_pair_ratio": rank_positive,
        "target_selection_swap_weeks": int(len(selection_target)),
        "target_selection_swap_mature_weeks": int(len(mature_selection)),
        "target_selection_mean_pair_delta_w4": selection_mean,
        "target_selection_median_pair_delta_w4": selection_median,
        "target_selection_positive_pair_ratio": selection_positive,
        "membership_changed_mean_portfolio_spread_w4": (
            None
            if mature_membership.empty
            else round(float(mature_membership["portfolio_spread_w4"].mean()), 4)
        ),
        "post_track_d_mature_weeks": int(len(post)),
        "post_track_d_membership_changed_weeks": int((post["membership_changed"] == True).sum()),
        "post_track_d_target_rank_crossover_weeks": int((post["target_rank_crossover"] == True).sum()),
        "post_track_d_target_selection_swap_weeks": int((post["target_selection_swap"] == True).sum()),
    }


def build_manifest(panel: pd.DataFrame, split: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": "track_e_soft_lane_audit",
        "protocol_version": PROTOCOL_VERSION,
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
            "ranking_intervention": (
                "build reward-only B0 rank skeleton; keep every non-target absolute "
                "rank position fixed; reorder only fresh_demand_alpha and "
                "standard_breakout candidates among their original target slots by "
                "status -> evidence/risk -> original lane -> remaining B0 tie-breaks"
            ),
            "target_soft_lanes": list(TARGET_SOFT_LANES),
            "fixed_skeleton_lanes": [
                "constructive_pullback",
                "incomplete_evidence",
                "tail_risk",
            ],
            "selector": "distinct_1",
        },
    }
