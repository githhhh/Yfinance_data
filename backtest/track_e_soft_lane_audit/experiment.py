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
    PRODUCTION_REFERENCE_ID,
    PROTOCOL_VERSION,
    TARGET_FRESH_LANE,
    TARGET_STANDARD_LANE,
    TRACK_D_HISTORICAL_END,
)
from .policy import (
    baseline_policy,
    challenger_policy,
    production_reference_policy,
)


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
    required = {"snapshot_date", "code", "w4_return_pct"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Track E panel missing required columns: {missing}")
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


def _summary_row(
    segment: str,
    comparator_name: str,
    challenger_outcomes,
    comparator_outcomes,
) -> dict[str, Any]:
    summary = evaluate_paired_challenger(
        challenger_outcomes,
        comparator_outcomes,
        CHALLENGER_POLICY_ID,
        "track_e_pairwise_replacement",
        segment,
        PRIMARY_HORIZON,
    )
    raw = dataclasses.asdict(summary)
    return {
        "segment": segment,
        "comparator": comparator_name,
        "support_weeks": raw["support_weeks"],
        "challenger_mean": raw["challenger_mean"],
        "comparator_mean": raw["b0_mean"],
        "mean_spread": raw["mean_spread"],
        "challenger_median": raw["challenger_median"],
        "comparator_median": raw["b0_median"],
        "median_spread": raw["median_spread"],
        "challenger_cvar10": raw["challenger_cvar10"],
        "comparator_cvar10": raw["b0_cvar10"],
        "cvar_delta": raw["cvar_delta"],
        "stop_delta_pct": raw["stop_delta_pct"],
        "one_pick_ruins_delta_pct": raw["one_pick_ruins_delta_pct"],
        "slot_coverage_pct": raw["slot_coverage_pct"],
        "full_top3_rate_pct": raw["full_top3_rate_pct"],
        "jaccard_vs_comparator": raw["top3_membership_jaccard_vs_b0"],
        "ci_low": (raw["bootstrap"] or {}).get("mean_spread_ci_low", 0.0),
        "ci_high": (raw["bootstrap"] or {}).get("mean_spread_ci_high", 0.0),
        "lowo_sign_stability": (raw["lowo"] or {}).get("sign_stability", 0.0),
        "lowo_edge_concentration": (raw["lowo"] or {}).get("positive_edge_concentration", 0.0),
        "lowo_fragile": (raw["lowo"] or {}).get("is_fragile_overfit", False),
    }


def evaluate_segments(panel: pd.DataFrame, segments: dict[str, list[str]]) -> pd.DataFrame:
    control = baseline_policy()
    challenger = challenger_policy()
    production = production_reference_policy()

    rows: list[dict[str, Any]] = []
    for name, snaps in segments.items():
        control_outcomes = run_policy(
            control, panel, snaps, BASELINE_POLICY_ID, PRIMARY_HORIZON
        )
        challenger_outcomes = run_policy(
            challenger, panel, snaps, CHALLENGER_POLICY_ID, PRIMARY_HORIZON
        )
        production_outcomes = run_policy(
            production, panel, snaps, PRODUCTION_REFERENCE_ID, PRIMARY_HORIZON
        )
        rows.append(
            _summary_row(
                name,
                "dry_neutral_hard_lane_control",
                challenger_outcomes,
                control_outcomes,
            )
        )
        rows.append(
            _summary_row(
                name,
                "production_b0_reference",
                challenger_outcomes,
                production_outcomes,
            )
        )
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
        value = pd.to_numeric(
            pd.Series([row.get("w4_return_pct")]),
            errors="coerce",
        ).iloc[0]
        out[code] = None if pd.isna(value) else float(value)
    return out


def _capital_return(codes: list[str], returns: dict[str, float | None]) -> float | None:
    vals = [returns.get(code) for code in codes]
    if any(v is None for v in vals):
        return None
    return round(float(sum(v for v in vals if v is not None) / 3.0), 4)


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
    control = baseline_policy()
    challenger = challenger_policy()
    production = production_reference_policy()

    rows: list[dict[str, Any]] = []

    for snap in sorted(panel["snapshot_date"].astype(str).unique().tolist()):
        s_df = panel[panel["snapshot_date"].astype(str) == snap].copy()

        control_codes, control_scored = _pick(control, s_df)
        challenger_codes, challenger_scored = _pick(challenger, s_df)
        production_codes, _ = _pick(production, s_df)

        if len(control_codes) != len(challenger_codes):
            raise RuntimeError(
                f"Track E pairwise replacement changed capacity on {snap}: "
                f"{control_codes} -> {challenger_codes}"
            )

        control_lanes = _map(control_scored, "lane")
        challenger_lanes = _map(challenger_scored, "lane")
        returns = _returns(s_df)

        swap_pairs: list[dict[str, Any]] = []
        for idx, (before, after) in enumerate(zip(control_codes, challenger_codes)):
            if before == after:
                continue
            before_lane = str(control_lanes.get(before, ""))
            after_lane = str(challenger_lanes.get(after, ""))
            if before_lane != TARGET_FRESH_LANE or after_lane != TARGET_STANDARD_LANE:
                raise RuntimeError(
                    f"Track E isolation breach on {snap} slot {idx}: "
                    f"{before}({before_lane}) -> {after}({after_lane})"
                )
            before_ret = returns.get(before)
            after_ret = returns.get(after)
            swap_pairs.append({
                "slot": idx + 1,
                "fresh_code": before,
                "standard_code": after,
                "fresh_w4": before_ret,
                "standard_w4": after_ret,
                "pair_delta_w4": (
                    None
                    if before_ret is None or after_ret is None
                    else round(after_ret - before_ret, 4)
                ),
            })

        # Every non-fresh control slot must remain untouched.
        for idx, before in enumerate(control_codes):
            if str(control_lanes.get(before, "")) != TARGET_FRESH_LANE:
                if challenger_codes[idx] != before:
                    raise RuntimeError(
                        f"Track E non-target selected slot moved on {snap}: "
                        f"{before} -> {challenger_codes[idx]}"
                    )

        opportunities = challenger.pairwise_opportunities(
            challenger_scored,
            control_codes,
        )

        control_ret = _capital_return(control_codes, returns)
        challenger_ret = _capital_return(challenger_codes, returns)
        production_ret = _capital_return(production_codes, returns)

        mature_pair_deltas = [
            float(x["pair_delta_w4"])
            for x in swap_pairs
            if x["pair_delta_w4"] is not None
        ]

        rows.append({
            "snapshot_date": snap,
            "segment": _segment_label(snap, split),
            "control_codes": json.dumps(control_codes),
            "challenger_codes": json.dumps(challenger_codes),
            "production_codes": json.dumps(production_codes),
            "control_vs_production_order_changed": control_codes != production_codes,
            "control_vs_production_membership_changed": set(control_codes) != set(production_codes),
            "pairwise_opportunity_count": len(opportunities),
            "pairwise_opportunities_json": json.dumps(opportunities, sort_keys=True, default=str),
            "target_selection_swap": bool(swap_pairs),
            "swap_count": len(swap_pairs),
            "swap_pairs_json": json.dumps(swap_pairs, sort_keys=True),
            "membership_changed_vs_control": set(control_codes) != set(challenger_codes),
            "control_w4_capital_return": control_ret,
            "challenger_w4_capital_return": challenger_ret,
            "production_w4_capital_return": production_ret,
            "portfolio_spread_vs_control_w4": (
                None
                if control_ret is None or challenger_ret is None
                else round(challenger_ret - control_ret, 4)
            ),
            "portfolio_spread_vs_production_w4": (
                None
                if production_ret is None or challenger_ret is None
                else round(challenger_ret - production_ret, 4)
            ),
            "mean_swap_pair_delta_w4": (
                None
                if not mature_pair_deltas
                else round(float(np.mean(mature_pair_deltas)), 4)
            ),
            "w4_mature_vs_control": control_ret is not None and challenger_ret is not None,
            "w4_mature_vs_production": production_ret is not None and challenger_ret is not None,
        })

    return pd.DataFrame(rows)


def build_event_summary(events: pd.DataFrame) -> dict[str, Any]:
    opportunity = events[events["pairwise_opportunity_count"] > 0].copy()
    swaps = events[events["target_selection_swap"] == True].copy()
    mature_swaps = swaps[swaps["mean_swap_pair_delta_w4"].notna()].copy()
    mature_changed = events[
        (events["membership_changed_vs_control"] == True)
        & (events["portfolio_spread_vs_control_w4"].notna())
    ].copy()
    post = events[
        (events["segment"] == "post_track_d_shadow")
        & (events["w4_mature_vs_control"] == True)
    ].copy()

    pair_deltas: list[float] = []
    for raw in mature_swaps["swap_pairs_json"].tolist():
        for pair in json.loads(raw):
            if pair.get("pair_delta_w4") is not None:
                pair_deltas.append(float(pair["pair_delta_w4"]))

    return {
        "total_snapshots": int(len(events)),
        "opportunity_weeks": int(len(opportunity)),
        "opportunity_pairs": int(opportunity["pairwise_opportunity_count"].sum()),
        "actual_swap_weeks": int(len(swaps)),
        "actual_swap_pairs": int(swaps["swap_count"].sum()),
        "mature_swap_weeks": int(len(mature_swaps)),
        "mature_swap_pairs": int(len(pair_deltas)),
        "swap_pair_mean_w4_delta": (
            None if not pair_deltas else round(float(np.mean(pair_deltas)), 4)
        ),
        "swap_pair_median_w4_delta": (
            None if not pair_deltas else round(float(np.median(pair_deltas)), 4)
        ),
        "swap_pair_positive_ratio": (
            None
            if not pair_deltas
            else round(float(np.mean(np.array(pair_deltas) > 0.0)), 4)
        ),
        "membership_changed_weeks_vs_control": int(
            (events["membership_changed_vs_control"] == True).sum()
        ),
        "mature_membership_changed_weeks_vs_control": int(len(mature_changed)),
        "changed_week_mean_portfolio_spread_vs_control_w4": (
            None
            if mature_changed.empty
            else round(float(mature_changed["portfolio_spread_vs_control_w4"].mean()), 4)
        ),
        "dry_control_vs_production_order_changed_weeks": int(
            (events["control_vs_production_order_changed"] == True).sum()
        ),
        "dry_control_vs_production_membership_changed_weeks": int(
            (events["control_vs_production_membership_changed"] == True).sum()
        ),
        "post_track_d_mature_weeks": int(len(post)),
        "post_track_d_opportunity_weeks": int((post["pairwise_opportunity_count"] > 0).sum()),
        "post_track_d_swap_weeks": int((post["target_selection_swap"] == True).sum()),
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
        "primary_control": {
            "policy_id": BASELINE_POLICY_ID,
            "dry": "True reward / False neutral",
            "lane": "hard B0_LANE",
            "selector": "distinct_1",
            "reason": (
                "dry=False penalty was independently null in Track D; using the "
                "dry-neutral hard-Lane control isolates the Lane intervention"
            ),
        },
        "production_reference": {
            "policy_id": PRODUCTION_REFERENCE_ID,
            "dry": "symmetric True reward / False penalty",
            "lane": "hard B0_LANE",
            "selector": "distinct_1",
        },
        "challenger": {
            "policy_id": CHALLENGER_POLICY_ID,
            "intervention": (
                "start from dry-neutral hard-Lane Top3; preserve every non-fresh "
                "selected slot; allow only an unselected standard_breakout to replace "
                "a selected fresh_demand_alpha when the standard Pareto-dominates on "
                "risk_count, exact buy-point distance, and entry volume; preserve distinct_1"
            ),
            "lane_defining_followthrough_excluded_from_dominance": [
                "eps_acceleration_support",
                "weekly_volume_follow_through",
            ],
            "dominance_axes": [
                "risk_count (lower/equal)",
                "abs(current_vs_ibd_candidate_pct) (lower/equal)",
                "ibd_entry_volume_ratio (higher/equal)",
                "at least one strict improvement",
            ],
        },
    }
