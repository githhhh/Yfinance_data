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
    HIST_MAX_RUIN_DELTA_PCT,
    HIST_MAX_STOP_DELTA_PCT,
    HIST_MIN_CI_LOW,
    HIST_MIN_CVAR_DELTA,
    HIST_MIN_FULL_SUPPORT,
    HIST_MIN_LOCKED_SUPPORT,
    HIST_MIN_MEAN_SPREAD,
    HIST_MIN_MEDIAN_SPREAD,
    PANEL_SOURCE,
    PRIMARY_HORIZON,
    PRODUCTION_B0_PATH,
    PROTOCOL_VERSION,
    TRACK_D_HISTORICAL_END,
)
from .policy import (
    LaneCompositionPolicy,
    all_policies,
    composition_specs,
    legacy_pullback_parity_anchor,
    production_baseline,
)
from .taxonomy import expected_actionable_mapping_ok, lane_fact_rows


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_panel() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL_SOURCE)
    required = {
        "snapshot_date",
        "code",
        "industry",
        "b0_eligible",
        "w4_return_pct",
        "w4_stop8",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise RuntimeError(f"Track F panel missing required columns: {missing}")
    panel = panel.copy()
    panel["snapshot_date"] = panel["snapshot_date"].astype(str)
    panel["code"] = panel["code"].astype(str)
    return panel


def build_segments(panel: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, Any]]:
    historical = panel[
        panel["snapshot_date"].astype(str) <= TRACK_D_HISTORICAL_END
    ].copy()
    historical_snaps = sorted(historical["snapshot_date"].astype(str).unique().tolist())
    split = build_locked_forward_split(historical_snaps)

    screening = [
        s
        for block in split["forward_blocks"]
        if block["stage"] == "screening"
        for s in block["snapshots"]
    ]
    confirmation = [
        s
        for block in split["forward_blocks"]
        if block["stage"] == "confirmation"
        for s in block["snapshots"]
    ]
    all_snaps = sorted(panel["snapshot_date"].astype(str).unique().tolist())
    post_track_d = [s for s in all_snaps if s > TRACK_D_HISTORICAL_END]

    segments = {
        "discovery_train_18": list(split["discovery_train"]),
        "purge_4": list(split["purge"]),
        "screening_6": screening,
        "confirmation_12": confirmation,
        "locked_forward_18": screening + confirmation,
        "retrospective_track_d_40": list(split["all_used_snapshots"]),
        "post_track_d_current_panel": post_track_d,
        "retrospective_current_panel": all_snaps,
    }
    return segments, split


def _summary_row(
    policy: LaneCompositionPolicy,
    segment: str,
    challenger_outcomes,
    baseline_outcomes,
) -> dict[str, Any]:
    summary = evaluate_paired_challenger(
        challenger_outcomes,
        baseline_outcomes,
        policy.policy_id,
        policy.family,
        segment,
        PRIMARY_HORIZON,
    )
    raw = dataclasses.asdict(summary)
    return {
        "policy_id": policy.policy_id,
        "role": policy.spec.role,
        "mode": policy.spec.mode,
        "industry_policy": policy.spec.industry_policy,
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


def evaluate_policies(panel: pd.DataFrame, segments: dict[str, list[str]]) -> pd.DataFrame:
    baseline = production_baseline()
    policies = all_policies()
    rows: list[dict[str, Any]] = []

    for segment, snapshots in segments.items():
        b0_outcomes = run_policy(
            baseline,
            panel,
            snapshots,
            selector_id=BASELINE_POLICY_ID,
            horizon=PRIMARY_HORIZON,
        )
        for policy in policies:
            outcomes = run_policy(
                policy,
                panel,
                snapshots,
                selector_id=policy.policy_id,
                horizon=PRIMARY_HORIZON,
            )
            rows.append(_summary_row(policy, segment, outcomes, b0_outcomes))

    return pd.DataFrame(rows)


def build_taxonomy_rows(panel: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for snapshot in sorted(panel["snapshot_date"].astype(str).unique().tolist()):
        s_df = panel[panel["snapshot_date"].astype(str) == snapshot].copy()
        part = lane_fact_rows(s_df)
        if not part.empty:
            parts.append(part)

    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if rows.empty:
        raise RuntimeError("Track F taxonomy audit produced zero rows")

    invalid = rows[
        rows.apply(lambda row: not expected_actionable_mapping_ok(row), axis=1)
    ]
    if not invalid.empty:
        sample = invalid[
            [
                "snapshot_date",
                "code",
                "current_lane",
                "setup_route",
                "quality_state",
                "composition_group",
                "b0_eligible",
            ]
        ].head(10)
        raise RuntimeError(
            "Track F actionable taxonomy invariant failed:\n"
            + sample.to_string(index=False)
        )
    return rows


def taxonomy_matrix(taxonomy_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        taxonomy_rows.groupby(
            [
                "b0_eligible",
                "current_lane",
                "entry_status",
                "setup_route",
                "quality_state",
                "composition_group",
                "actionable_pullback_context_branch",
                "non_actionable_pullback_context_branch",
            ],
            dropna=False,
        )
        .agg(
            row_count=("code", "size"),
            snapshot_count=("snapshot_date", "nunique"),
            unique_codes=("code", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["b0_eligible", "current_lane", "entry_status", "setup_route", "quality_state"],
            ascending=[False, True, True, True, True],
        )
    )
    return grouped


def taxonomy_summary(taxonomy_rows: pd.DataFrame) -> dict[str, Any]:
    constructive = taxonomy_rows[
        taxonomy_rows["current_lane"] == "constructive_pullback"
    ].copy()
    standard = taxonomy_rows[
        taxonomy_rows["current_lane"] == "standard_breakout"
    ].copy()
    eligible = taxonomy_rows[taxonomy_rows["b0_eligible"] == True].copy()

    eligible_groups = (
        eligible["composition_group"].value_counts(dropna=False).to_dict()
        if not eligible.empty
        else {}
    )
    standard_routes = (
        standard["setup_route"].value_counts(dropna=False).to_dict()
        if not standard.empty
        else {}
    )

    return {
        "total_review_rows": int(len(taxonomy_rows)),
        "b0_eligible_rows": int(len(eligible)),
        "eligible_composition_groups": {
            str(k): int(v) for k, v in eligible_groups.items()
        },
        "constructive_pullback_rows": int(len(constructive)),
        "constructive_actionable_confirmed_branch_rows": int(
            constructive["actionable_pullback_context_branch"].sum()
        ),
        "constructive_non_actionable_context_branch_rows": int(
            constructive["non_actionable_pullback_context_branch"].sum()
        ),
        "constructive_other_rows": int(
            len(constructive)
            - constructive["actionable_pullback_context_branch"].sum()
            - constructive["non_actionable_pullback_context_branch"].sum()
        ),
        "standard_breakout_routes": {
            str(k): int(v) for k, v in standard_routes.items()
        },
        "eligible_standard_pullback_rows": int(
            len(
                eligible[
                    (eligible["current_lane"] == "standard_breakout")
                    & (eligible["setup_route"] == "pullback")
                ]
            )
        ),
        "eligible_standard_non_pullback_rows": int(
            len(
                eligible[
                    (eligible["current_lane"] == "standard_breakout")
                    & (eligible["setup_route"] == "non_pullback")
                ]
            )
        ),
    }


def _pick(policy, snapshot_df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    scored = policy.score_candidates(snapshot_df)
    quotas = policy.allocate_industries(scored)
    picks = policy.pick_stocks(scored, quotas)
    return picks, scored


def _facts_for_codes(snapshot_df: pd.DataFrame, codes: list[str]) -> dict[str, dict[str, Any]]:
    by_code = {
        str(row["code"]): row
        for _, row in snapshot_df.iterrows()
    }
    out: dict[str, dict[str, Any]] = {}
    for code in codes:
        row = by_code.get(str(code))
        if row is None:
            raise RuntimeError(f"Track F selected code outside snapshot panel: {code}")
        facts_df = lane_fact_rows(snapshot_df.loc[[row.name]])
        if facts_df.empty:
            raise RuntimeError(f"Track F cannot derive Lane facts for selected code {code}")
        fact = facts_df.iloc[0]
        out[code] = {
            "current_lane": str(fact["current_lane"]),
            "setup_route": str(fact["setup_route"]),
            "quality_state": str(fact["quality_state"]),
            "composition_group": str(fact["composition_group"]),
            "industry": str(row.get("industry", "") or "").strip(),
        }
    return out


def selection_composition(panel: pd.DataFrame) -> pd.DataFrame:
    baseline = production_baseline()
    policies: list[Any] = [baseline] + all_policies()
    rows: list[dict[str, Any]] = []

    for snapshot in sorted(panel["snapshot_date"].astype(str).unique().tolist()):
        s_df = panel[panel["snapshot_date"].astype(str) == snapshot].copy()
        for policy in policies:
            picks, _ = _pick(policy, s_df)
            facts = _facts_for_codes(s_df, picks)
            groups = [facts[code]["composition_group"] for code in picks]
            routes = [facts[code]["setup_route"] for code in picks]
            industries = [facts[code]["industry"] for code in picks]

            rows.append({
                "snapshot_date": snapshot,
                "policy_id": (
                    BASELINE_POLICY_ID
                    if policy is baseline
                    else str(policy.policy_id)
                ),
                "pick_count": len(picks),
                "selected_codes": json.dumps(picks),
                "composition_groups": json.dumps(groups),
                "setup_routes": json.dumps(routes),
                "industries": json.dumps(industries),
                "confirmed_non_pullback_count": groups.count("confirmed_non_pullback"),
                "confirmed_pullback_count": groups.count("confirmed_pullback"),
                "standard_count": groups.count("standard"),
                "incomplete_count": groups.count("incomplete"),
                "distinct_industry_count": len(set(industries)),
            })

    return pd.DataFrame(rows)


def parity_anchor_audit(panel: pd.DataFrame) -> dict[str, Any]:
    """Track F normalized parity should reproduce Track C PULLBACK_PARITY picks."""
    track_f = next(
        p
        for p in all_policies()
        if p.spec.policy_id == "CONFIRMED_PARITY_FALLBACK"
    )
    legacy = legacy_pullback_parity_anchor()

    mismatches: list[dict[str, Any]] = []
    for snapshot in sorted(panel["snapshot_date"].astype(str).unique().tolist()):
        s_df = panel[panel["snapshot_date"].astype(str) == snapshot].copy()
        f_codes, _ = _pick(track_f, s_df)
        legacy_codes, _ = _pick(legacy, s_df)
        if f_codes != legacy_codes:
            mismatches.append({
                "snapshot_date": snapshot,
                "track_f": f_codes,
                "legacy_pullback_parity": legacy_codes,
            })

    return {
        "snapshot_count": int(panel["snapshot_date"].astype(str).nunique()),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def route_bucket_weekly(panel: pd.DataFrame, taxonomy_rows: pd.DataFrame) -> pd.DataFrame:
    merged = panel.merge(
        taxonomy_rows[
            [
                "snapshot_date",
                "code",
                "b0_eligible",
                "composition_group",
            ]
        ],
        on=["snapshot_date", "code"],
        how="inner",
        suffixes=("", "_tax"),
    )
    merged = merged[merged["b0_eligible_tax"] == True].copy()

    rows: list[dict[str, Any]] = []
    groups = ["confirmed_non_pullback", "confirmed_pullback", "standard"]
    for snapshot in sorted(merged["snapshot_date"].astype(str).unique().tolist()):
        snap = merged[merged["snapshot_date"].astype(str) == snapshot]
        for group in groups:
            g = snap[snap["composition_group"] == group].copy()
            if g.empty:
                rows.append({
                    "snapshot_date": snapshot,
                    "composition_group": group,
                    "candidate_count": 0,
                    "mature": True,
                    "mean_w4": np.nan,
                    "median_w4": np.nan,
                    "stop8_pct": np.nan,
                })
                continue

            rets = pd.to_numeric(g["w4_return_pct"], errors="coerce")
            mature = bool(rets.notna().all())
            stops = g["w4_stop8"]
            stop_mature = bool(stops.notna().all())
            mature = bool(mature and stop_mature)
            stops = stops.astype(bool) if mature else pd.Series(dtype=bool)
            rows.append({
                "snapshot_date": snapshot,
                "composition_group": group,
                "candidate_count": int(len(g)),
                "mature": mature,
                "mean_w4": float(rets.mean()) if mature else np.nan,
                "median_w4": float(rets.median()) if mature else np.nan,
                "stop8_pct": float(stops.mean() * 100.0) if mature else np.nan,
            })

    return pd.DataFrame(rows)


def route_pair_summary(bucket_weekly: pd.DataFrame) -> dict[str, Any]:
    pivot_mean = bucket_weekly.pivot(
        index="snapshot_date",
        columns="composition_group",
        values="mean_w4",
    )
    pivot_stop = bucket_weekly.pivot(
        index="snapshot_date",
        columns="composition_group",
        values="stop8_pct",
    )
    counts = bucket_weekly.pivot(
        index="snapshot_date",
        columns="composition_group",
        values="candidate_count",
    )

    needed = ["confirmed_non_pullback", "confirmed_pullback"]
    for col in needed:
        if col not in pivot_mean.columns:
            return {
                "support_weeks": 0,
                "mean_pullback_minus_non_pullback_w4": None,
                "median_pullback_minus_non_pullback_w4": None,
                "positive_week_ratio": None,
                "mean_stop_delta_pct": None,
            }

    mask = (
        counts["confirmed_non_pullback"].fillna(0).gt(0)
        & counts["confirmed_pullback"].fillna(0).gt(0)
        & pivot_mean["confirmed_non_pullback"].notna()
        & pivot_mean["confirmed_pullback"].notna()
    )
    spreads = (
        pivot_mean.loc[mask, "confirmed_pullback"]
        - pivot_mean.loc[mask, "confirmed_non_pullback"]
    )
    stop_spreads = (
        pivot_stop.loc[mask, "confirmed_pullback"]
        - pivot_stop.loc[mask, "confirmed_non_pullback"]
    )

    if spreads.empty:
        return {
            "support_weeks": 0,
            "mean_pullback_minus_non_pullback_w4": None,
            "median_pullback_minus_non_pullback_w4": None,
            "positive_week_ratio": None,
            "mean_stop_delta_pct": None,
        }

    return {
        "support_weeks": int(len(spreads)),
        "mean_pullback_minus_non_pullback_w4": round(float(spreads.mean()), 4),
        "median_pullback_minus_non_pullback_w4": round(float(spreads.median()), 4),
        "positive_week_ratio": round(float((spreads > 0).mean()), 4),
        "mean_stop_delta_pct": round(float(stop_spreads.mean()), 4),
    }


def historical_support_decision(evaluation: pd.DataFrame) -> dict[str, Any]:
    """Pre-registered retrospective support gate for primary policies only.

    Passing this gate never promotes Production; it only qualifies a policy for
    future unseen shadow observation.
    """
    primary = evaluation[evaluation["role"] == "primary"].copy()
    verdicts: list[dict[str, Any]] = []

    for policy_id, group in primary.groupby("policy_id"):
        full = group[group["segment"] == "retrospective_track_d_40"]
        locked = group[group["segment"] == "locked_forward_18"]
        if full.empty or locked.empty:
            verdicts.append({
                "policy_id": str(policy_id),
                "verdict": "INSUFFICIENT_HISTORICAL_SUPPORT",
                "reasons": ["missing required retrospective segment"],
            })
            continue

        f = full.iloc[0]
        l = locked.iloc[0]
        checks = {
            "full_support": int(f["support_weeks"]) >= HIST_MIN_FULL_SUPPORT,
            "locked_support": int(l["support_weeks"]) >= HIST_MIN_LOCKED_SUPPORT,
            "full_mean": float(f["mean_spread"]) >= HIST_MIN_MEAN_SPREAD,
            "locked_mean": float(l["mean_spread"]) >= HIST_MIN_MEAN_SPREAD,
            "full_median": float(f["median_spread"]) >= HIST_MIN_MEDIAN_SPREAD,
            "locked_median": float(l["median_spread"]) >= HIST_MIN_MEDIAN_SPREAD,
            "full_cvar": float(f["cvar_delta"]) >= HIST_MIN_CVAR_DELTA,
            "locked_cvar": float(l["cvar_delta"]) >= HIST_MIN_CVAR_DELTA,
            "full_stop": float(f["stop_delta_pct"]) <= HIST_MAX_STOP_DELTA_PCT,
            "locked_stop": float(l["stop_delta_pct"]) <= HIST_MAX_STOP_DELTA_PCT,
            "full_ruin": float(f["one_pick_ruins_delta_pct"]) <= HIST_MAX_RUIN_DELTA_PCT,
            "locked_ruin": float(l["one_pick_ruins_delta_pct"]) <= HIST_MAX_RUIN_DELTA_PCT,
            "full_ci_low": float(f["ci_low"]) >= HIST_MIN_CI_LOW,
            "locked_ci_low": float(l["ci_low"]) >= HIST_MIN_CI_LOW,
        }
        failed = [name for name, passed in checks.items() if not passed]
        verdicts.append({
            "policy_id": str(policy_id),
            "verdict": (
                "HISTORICAL_SHADOW_CANDIDATE"
                if not failed
                else "NO_HISTORICAL_SUPPORT_TO_REPLACE_B0"
            ),
            "failed_checks": failed,
            "checks": checks,
        })

    candidates = [
        row["policy_id"]
        for row in verdicts
        if row["verdict"] == "HISTORICAL_SHADOW_CANDIDATE"
    ]
    return {
        "overall": (
            "HISTORICAL_SHADOW_CANDIDATE_EXISTS"
            if candidates
            else "RETAIN_B0_WITHIN_TRACK_F_TESTED_COMPOSITIONS"
        ),
        "shadow_candidates": candidates,
        "policy_verdicts": verdicts,
        "note": (
            "Retrospective gate only. No Track F policy can be promoted without "
            "genuinely unseen future observations."
        ),
    }


def build_manifest(panel: pd.DataFrame) -> dict[str, Any]:
    max_snapshot = max(panel["snapshot_date"].astype(str).unique().tolist())
    return {
        "experiment": "track_f_lane_composition_audit",
        "protocol_version": PROTOCOL_VERSION,
        "source_git_sha": git_sha(),
        "panel_hash": sha256_file(PANEL_SOURCE),
        "production_b0_hash": sha256_file(PRODUCTION_B0_PATH),
        "panel_snapshot_count": int(panel["snapshot_date"].astype(str).nunique()),
        "panel_max_snapshot": max_snapshot,
        "evidence_status": (
            "All currently materialized panel outcomes are retrospective for Track F because "
            "the Lane-composition hypotheses were formulated after observing prior Track C/D/E results. "
            "Track-D segment labels are reused only for regime consistency, not as untouched OOS."
        ),
        "orthogonal_taxonomy": {
            "setup_route": ["non_pullback", "pullback"],
            "quality_state": ["confirmed", "standard", "incomplete", "failure"],
            "confirmed_definition": "near_buy_point AND entry_volume>=1.5 AND (EPS>=25 OR weekly_volume>=1.3)",
            "standard_definition": "near_buy_point AND entry_volume>=1.5 AND NOT follow_through",
            "failure_definition": "Production clear_geometry_failure",
        },
        "baseline": {
            "policy_id": BASELINE_POLICY_ID,
            "semantics": "Exact Production B0: hard current Lane + symmetric dry + distinct_1",
        },
        "policies": [dataclasses.asdict(spec) for spec in composition_specs()],
        "historical_support_gate": {
            "min_full_support": HIST_MIN_FULL_SUPPORT,
            "min_locked_support": HIST_MIN_LOCKED_SUPPORT,
            "min_mean_spread": HIST_MIN_MEAN_SPREAD,
            "min_median_spread": HIST_MIN_MEDIAN_SPREAD,
            "min_cvar_delta": HIST_MIN_CVAR_DELTA,
            "max_stop_delta_pct": HIST_MAX_STOP_DELTA_PCT,
            "max_ruin_delta_pct": HIST_MAX_RUIN_DELTA_PCT,
            "min_ci_low": HIST_MIN_CI_LOW,
        },
        "primary_question": (
            "Does non-pullback route deserve unconditional priority over equally confirmed pullback route, "
            "and does explicit Lane/route composition improve Top3 portfolio quality?"
        ),
    }
