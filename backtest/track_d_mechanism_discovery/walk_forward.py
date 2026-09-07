from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from backtest.track_c_ranking_discovery.b0_ablation_grid import StructuralGridChallenger
from backtest.track_c_ranking_discovery.evaluate_econometrics import evaluate_paired_challenger

from .config import (
    AGENT_SCREENING_SHORTLIST,
    BOOTSTRAP_CI_LOW_MIN,
    COMPRESS_MEAN_MIN_SPREAD,
    COMPRESS_MEDIAN_MIN_SPREAD,
    COMPRESS_MIN_POSITIVE_BLOCK_RATIO,
    CONFIRMATION_BLOCKS,
    CVAR_MIN_DELTA,
    DISCOVERY_TRAIN_SNAPSHOTS,
    MIN_CONFIRM_SUPPORT_WEEKS,
    MIN_FORWARD_SUPPORT_WEEKS,
    MIN_POSITIVE_BLOCK_RATIO,
    MINIMAL_SCREENING_PER_REMOVAL_COUNT,
    OUTER_BLOCKS,
    OUTER_TEST_SNAPSHOTS,
    PRIMARY_HORIZON,
    PURGE_SNAPSHOTS,
    RETURN_MEAN_MIN_SPREAD,
    RETURN_MEDIAN_MIN_SPREAD,
    RUIN_MAX_DELTA_PCT,
    SCREENING_BLOCKS,
    STOP_MAX_DELTA_PCT,
)
from .mechanism_lab import (
    MechanismPolicy,
    generate_mechanism_specs,
    generate_minimal_b0_specs,
    run_policy,
)
from .policy_dsl import DSLPolicy


def build_locked_forward_split(snapshots: list[str]) -> dict[str, Any]:
    if SCREENING_BLOCKS + CONFIRMATION_BLOCKS != OUTER_BLOCKS:
        raise RuntimeError("Track D screening + confirmation block counts must equal OUTER_BLOCKS")

    snaps = sorted(str(x) for x in snapshots)
    required = (
        DISCOVERY_TRAIN_SNAPSHOTS
        + PURGE_SNAPSHOTS
        + OUTER_BLOCKS * OUTER_TEST_SNAPSHOTS
    )
    if len(snaps) < required:
        raise RuntimeError(
            f"Track D needs at least {required} frozen historical snapshots, got {len(snaps)}."
        )

    snaps = snaps[:required]
    discovery = snaps[:DISCOVERY_TRAIN_SNAPSHOTS]
    purge = snaps[
        DISCOVERY_TRAIN_SNAPSHOTS:
        DISCOVERY_TRAIN_SNAPSHOTS + PURGE_SNAPSHOTS
    ]
    start = DISCOVERY_TRAIN_SNAPSHOTS + PURGE_SNAPSHOTS
    blocks = []
    for i in range(OUTER_BLOCKS):
        a = start + i * OUTER_TEST_SNAPSHOTS
        b = a + OUTER_TEST_SNAPSHOTS
        blocks.append({
            "block_id": f"FWD_{i+1}",
            "stage": "screening" if i < SCREENING_BLOCKS else "confirmation",
            "snapshots": snaps[a:b],
        })

    return {
        "discovery_train": discovery,
        "purge": purge,
        "forward_blocks": blocks,
        "screening_blocks": [x["block_id"] for x in blocks if x["stage"] == "screening"],
        "confirmation_blocks": [x["block_id"] for x in blocks if x["stage"] == "confirmation"],
        "all_used_snapshots": snaps,
    }


def _baseline_outcomes(panel_df: pd.DataFrame, snapshots: list[str]):
    b0 = StructuralGridChallenger("B0_LANE", "symmetric", "distinct_1")
    return run_policy(b0, panel_df, snapshots, selector_id="B0_ORIGINAL", horizon=PRIMARY_HORIZON)


def _ensure_b0_mature(outcomes, label: str) -> None:
    immature = [o.snapshot_date for o in outcomes if not o.is_mature]
    if immature:
        raise RuntimeError(
            f"Track D {label} contains technically immature B0 W4 outcomes: "
            + ", ".join(immature)
        )


def _summary_metrics(policy_id: str, family: str, outcomes, b0_outcomes, segment: str) -> dict[str, Any]:
    s = evaluate_paired_challenger(
        outcomes,
        b0_outcomes,
        policy_id,
        family,
        segment,
        PRIMARY_HORIZON,
    )
    return {
        "support_weeks": s.support_weeks,
        "mean_spread": s.mean_spread,
        "median_spread": s.median_spread,
        "cvar_delta": s.cvar_delta,
        "stop_delta_pct": s.stop_delta_pct,
        "one_pick_ruins_delta_pct": s.one_pick_ruins_delta_pct,
        "slot_coverage_pct": s.slot_coverage_pct,
        "full_top3_rate_pct": s.full_top3_rate_pct,
        "jaccard_vs_b0": s.top3_membership_jaccard_vs_b0,
        "ci_low": s.bootstrap.mean_spread_ci_low if s.bootstrap else 0.0,
        "ci_high": s.bootstrap.mean_spread_ci_high if s.bootstrap else 0.0,
    }


def _block_metrics(policy: Any, family: str, panel_df: pd.DataFrame, block_defs: list[dict[str, Any]], b0_by_block: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    rows = []
    for b in block_defs:
        outcomes = run_policy(
            policy,
            panel_df,
            b["snapshots"],
            selector_id=policy.policy_id,
            horizon=PRIMARY_HORIZON,
        )
        m = _summary_metrics(
            policy.policy_id,
            family,
            outcomes,
            b0_by_block[b["block_id"]],
            b["block_id"],
        )
        rows.append({
            "block_id": b["block_id"],
            "support_weeks": m["support_weeks"],
            "mean_spread": m["mean_spread"],
            "median_spread": m["median_spread"],
            "cvar_delta": m["cvar_delta"],
            "stop_delta_pct": m["stop_delta_pct"],
        })
    positive = sum(float(x["mean_spread"]) > 0.0 for x in rows)
    ratio = positive / float(len(rows)) if rows else 0.0
    return rows, round(ratio, 4)


def _attach(prefix: str, row: dict[str, Any], metrics: dict[str, Any], blocks: list[dict[str, Any]], block_ratio: float) -> None:
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value
    row[f"{prefix}_positive_block_ratio"] = block_ratio
    row[f"{prefix}_block_summaries_json"] = json.dumps(blocks, sort_keys=True)


def _risk_pass(row: dict[str, Any] | pd.Series, prefix: str) -> bool:
    return bool(
        float(row[f"{prefix}_cvar_delta"]) >= CVAR_MIN_DELTA
        and float(row[f"{prefix}_stop_delta_pct"]) <= STOP_MAX_DELTA_PCT
        and float(row[f"{prefix}_one_pick_ruins_delta_pct"]) <= RUIN_MAX_DELTA_PCT
    )


def _b1_pass(row: dict[str, Any] | pd.Series) -> bool:
    return bool(
        bool(row.get("confirmation_evaluated", False))
        and int(row["confirmation_support_weeks"]) >= MIN_CONFIRM_SUPPORT_WEEKS
        and int(row["pooled_support_weeks"]) >= MIN_FORWARD_SUPPORT_WEEKS
        and _risk_pass(row, "confirmation")
        and _risk_pass(row, "pooled")
        and float(row["confirmation_mean_spread"]) >= RETURN_MEAN_MIN_SPREAD
        and float(row["confirmation_median_spread"]) >= RETURN_MEDIAN_MIN_SPREAD
        and float(row["confirmation_ci_low"]) >= BOOTSTRAP_CI_LOW_MIN
        and float(row["confirmation_positive_block_ratio"]) >= MIN_POSITIVE_BLOCK_RATIO
    )


def _compress_pass(row: dict[str, Any] | pd.Series) -> bool:
    return bool(
        bool(row.get("confirmation_evaluated", False))
        and int(row["confirmation_support_weeks"]) >= MIN_CONFIRM_SUPPORT_WEEKS
        and int(row["pooled_support_weeks"]) >= MIN_FORWARD_SUPPORT_WEEKS
        and _risk_pass(row, "confirmation")
        and _risk_pass(row, "pooled")
        and float(row["confirmation_mean_spread"]) >= COMPRESS_MEAN_MIN_SPREAD
        and float(row["confirmation_median_spread"]) >= COMPRESS_MEDIAN_MIN_SPREAD
        and float(row["confirmation_positive_block_ratio"]) >= COMPRESS_MIN_POSITIVE_BLOCK_RATIO
    )


def _mechanism_metadata(policy: Any) -> tuple[str, str]:
    if not isinstance(policy, MechanismPolicy):
        return "", ""
    removed = []
    for x in (
        "lane", "dry_false_penalty", "evidence_risk", "freshness",
        "eps_preference", "weekly_volume", "entry_volume", "distinct1",
    ):
        removed_now = (
            (x == "distinct1" and policy.spec.selector_mode != "distinct_1")
            or (x == "dry_false_penalty" and not policy.spec.dry_false_penalty)
            or (x not in {"distinct1", "dry_false_penalty"} and x not in policy.spec.components)
        )
        if removed_now:
            removed.append(x)
    return "|".join(policy.spec.components), "|".join(removed)


def _screen_sort(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["screen_catastrophic"] = (
        (work["screen_support_weeks"] < 5)
        | (work["screen_cvar_delta"] < -3.0)
        | (work["screen_stop_delta_pct"] > 10.0)
    )
    return work.sort_values(
        [
            "screen_catastrophic",
            "screen_support_weeks",
            "screen_mean_spread",
            "screen_median_spread",
            "screen_cvar_delta",
        ],
        ascending=[True, False, False, False, False],
    )


def _choose_confirmation_ids(screen_df: pd.DataFrame) -> set[str]:
    chosen: set[str] = set()

    agent = screen_df[screen_df["policy_kind"] == "agent_b1"].copy()
    if not agent.empty:
        chosen.update(_screen_sort(agent).head(AGENT_SCREENING_SHORTLIST)["policy_id"].astype(str))

    minimal = screen_df[screen_df["policy_kind"] == "minimal_b0"].copy()
    if not minimal.empty:
        for removed_count, group in minimal.groupby("removed_count"):
            chosen.update(
                _screen_sort(group)
                .head(MINIMAL_SCREENING_PER_REMOVAL_COUNT)["policy_id"]
                .astype(str)
            )

    mechanism = screen_df[screen_df["policy_kind"].isin(["knockout", "capacity"])].copy()
    chosen.update(mechanism["policy_id"].astype(str))
    return chosen


def evaluate_locked_forward(
    panel_df: pd.DataFrame,
    split: dict[str, Any],
    agent_specs: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Two-stage locked evaluation.

    All policies are frozen before any outer outcomes are used. Blocks 1-2 may only
    shortlist frozen policies. Production-shadow gates are computed on untouched
    confirmation blocks 3-6. Pooled 18-week metrics are supporting evidence only.
    """
    screen_blocks = [b for b in split["forward_blocks"] if b["stage"] == "screening"]
    confirm_blocks = [b for b in split["forward_blocks"] if b["stage"] == "confirmation"]
    screen_snaps = [s for b in screen_blocks for s in b["snapshots"]]
    confirm_snaps = [s for b in confirm_blocks for s in b["snapshots"]]
    all_test_snaps = screen_snaps + confirm_snaps

    b0_screen = _baseline_outcomes(panel_df, screen_snaps)
    b0_confirm = _baseline_outcomes(panel_df, confirm_snaps)
    b0_pooled = _baseline_outcomes(panel_df, all_test_snaps)
    _ensure_b0_mature(b0_screen, "screening window")
    _ensure_b0_mature(b0_confirm, "confirmation window")

    b0_by_block = {
        b["block_id"]: _baseline_outcomes(panel_df, b["snapshots"])
        for b in split["forward_blocks"]
    }

    registry: list[tuple[Any, str, str]] = []
    for spec in agent_specs:
        registry.append((DSLPolicy(spec), "agent_b1", "agent"))

    for spec in generate_mechanism_specs():
        if spec.experiment_kind in {"baseline", "neutral", "rescue", "interaction"}:
            continue
        registry.append((MechanismPolicy(spec), spec.experiment_kind, "mechanism"))

    for spec in generate_minimal_b0_specs():
        registry.append((MechanismPolicy(spec), "minimal_b0", "minimal_b0"))

    row_map: dict[str, dict[str, Any]] = {}
    policy_map: dict[str, tuple[Any, str]] = {}

    # Stage 1: screening only. Confirmation data has not been consulted.
    for policy, kind, family in registry:
        outcomes = run_policy(policy, panel_df, screen_snaps, selector_id=policy.policy_id, horizon=PRIMARY_HORIZON)
        metrics = _summary_metrics(policy.policy_id, family, outcomes, b0_screen, "screening")
        block_rows, block_ratio = _block_metrics(policy, family, panel_df, screen_blocks, b0_by_block)

        kept, removed = _mechanism_metadata(policy)
        removed_count = 0 if not removed else len(removed.split("|"))
        row = {
            "policy_id": policy.policy_id,
            "family": family,
            "policy_kind": kind,
            "components_kept": kept,
            "components_removed": removed,
            "removed_count": removed_count,
            "screen_shortlisted": False,
            "confirmation_evaluated": False,
        }
        _attach("screen", row, metrics, block_rows, block_ratio)
        row_map[policy.policy_id] = row
        policy_map[policy.policy_id] = (policy, family)

    screen_df = pd.DataFrame(row_map.values())
    confirm_ids = _choose_confirmation_ids(screen_df)
    for pid in confirm_ids:
        row_map[pid]["screen_shortlisted"] = True

    # Stage 2: untouched confirmation. Only pre-selected IDs can affect a final decision.
    for pid in sorted(confirm_ids):
        policy, family = policy_map[pid]
        confirm_outcomes = run_policy(
            policy, panel_df, confirm_snaps, selector_id=policy.policy_id, horizon=PRIMARY_HORIZON
        )
        confirm_metrics = _summary_metrics(
            policy.policy_id, family, confirm_outcomes, b0_confirm, "confirmation"
        )
        confirm_block_rows, confirm_block_ratio = _block_metrics(
            policy, family, panel_df, confirm_blocks, b0_by_block
        )

        pooled_outcomes = run_policy(
            policy, panel_df, all_test_snaps, selector_id=policy.policy_id, horizon=PRIMARY_HORIZON
        )
        pooled_metrics = _summary_metrics(
            policy.policy_id, family, pooled_outcomes, b0_pooled, "pooled"
        )
        pooled_block_rows, pooled_block_ratio = _block_metrics(
            policy, family, panel_df, split["forward_blocks"], b0_by_block
        )

        row = row_map[pid]
        row["confirmation_evaluated"] = True
        _attach("confirmation", row, confirm_metrics, confirm_block_rows, confirm_block_ratio)
        _attach("pooled", row, pooled_metrics, pooled_block_rows, pooled_block_ratio)
        row["confirmation_risk_noninferior"] = _risk_pass(row, "confirmation")
        row["pooled_risk_noninferior"] = _risk_pass(row, "pooled")
        row["b1_gate_pass"] = _b1_pass(row)
        row["compression_gate_pass"] = _compress_pass(row)

    result_df = pd.DataFrame(row_map.values())
    for col in [
        "confirmation_risk_noninferior", "pooled_risk_noninferior",
        "b1_gate_pass", "compression_gate_pass",
    ]:
        if col not in result_df:
            result_df[col] = False
        result_df[col] = result_df[col].fillna(False).astype(bool)

    decision = decide_track_d_exit(result_df, agent_specs)
    return result_df, decision


def _component_verdicts(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    ko = df[
        (df["policy_kind"] == "knockout")
        & (df["confirmation_evaluated"] == True)
    ].copy()
    for _, row in ko.iterrows():
        component = str(row["policy_id"]).split("KNOCKOUT__", 1)[-1]
        mean = float(row["confirmation_mean_spread"])
        med = float(row["confirmation_median_spread"])
        risk = _risk_pass(row, "confirmation") and _risk_pass(row, "pooled")
        blocks = float(row["confirmation_positive_block_ratio"])

        if risk and mean >= RETURN_MEAN_MIN_SPREAD and blocks >= MIN_POSITIVE_BLOCK_RATIO:
            verdict = "HARMFUL"
        elif (
            risk
            and mean >= COMPRESS_MEAN_MIN_SPREAD
            and med >= COMPRESS_MEDIAN_MIN_SPREAD
            and blocks >= COMPRESS_MIN_POSITIVE_BLOCK_RATIO
        ):
            verdict = "REDUNDANT"
        elif mean <= -1.0 and blocks <= 0.25:
            verdict = "ESSENTIAL"
        elif mean < COMPRESS_MEAN_MIN_SPREAD or not risk:
            verdict = "HELPFUL"
        else:
            verdict = "UNCERTAIN"

        out.append({
            "component": component,
            "confirmation_support_weeks": int(row["confirmation_support_weeks"]),
            "knockout_mean_spread": mean,
            "knockout_median_spread": med,
            "knockout_cvar_delta": float(row["confirmation_cvar_delta"]),
            "knockout_stop_delta_pct": float(row["confirmation_stop_delta_pct"]),
            "positive_block_ratio": blocks,
            "verdict": verdict,
        })
    return out


def decide_track_d_exit(
    result_df: pd.DataFrame,
    agent_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    component_verdicts = _component_verdicts(result_df)
    spec_map = {f"TRACK_D_DSL__{x['policy_id']}": x for x in agent_specs}

    agent = result_df[
        (result_df["policy_kind"] == "agent_b1")
        & (result_df["screen_shortlisted"] == True)
        & (result_df["b1_gate_pass"] == True)
    ].copy()
    if not agent.empty:
        winner = agent.sort_values(
            [
                "confirmation_positive_block_ratio",
                "confirmation_ci_low",
                "confirmation_mean_spread",
            ],
            ascending=[False, False, False],
        ).iloc[0].to_dict()
        return {
            "state": "STATE_A_PROMOTE_B1_TO_FORWARD_SHADOW",
            "winner": winner,
            "winner_spec": spec_map.get(str(winner["policy_id"])),
            "component_verdicts": component_verdicts,
            "decision_basis": (
                "A frozen Agent B1 survived screening and then passed all pre-registered "
                "return/risk/bootstrap gates on untouched confirmation blocks."
            ),
        }

    repair = result_df[
        (result_df["policy_kind"] == "knockout")
        & (result_df["b1_gate_pass"] == True)
    ].copy()
    if not repair.empty:
        winner = repair.sort_values(
            [
                "confirmation_positive_block_ratio",
                "confirmation_ci_low",
                "confirmation_mean_spread",
            ],
            ascending=[False, False, False],
        ).iloc[0].to_dict()
        return {
            "state": "STATE_B_REPAIR_B0",
            "winner": winner,
            "component_verdicts": component_verdicts,
            "decision_basis": (
                "A pre-registered single-component knockout improved untouched confirmation "
                "performance without violating risk non-inferiority."
            ),
        }

    compress = result_df[
        (result_df["policy_kind"] == "minimal_b0")
        & (result_df["screen_shortlisted"] == True)
        & (result_df["compression_gate_pass"] == True)
    ].copy()
    if not compress.empty:
        winner = compress.sort_values(
            ["removed_count", "confirmation_mean_spread", "confirmation_ci_low"],
            ascending=[False, False, False],
        ).iloc[0].to_dict()
        return {
            "state": "STATE_C_COMPRESS_B0",
            "winner": winner,
            "component_verdicts": component_verdicts,
            "decision_basis": (
                "A pre-screened simpler B0 subset remained within pre-registered return/risk "
                "tolerances on untouched confirmation blocks."
            ),
        }

    return {
        "state": "STATE_D_MECHANISM_MAP_WITH_B0_RETAINED",
        "winner": None,
        "component_verdicts": component_verdicts,
        "decision_basis": (
            "No frozen B1, single-rule repair, or pre-screened Minimal-B0 passed untouched "
            "confirmation gates. The actionable result is the component mechanism map, "
            "not a claim that B0 is unbeatable."
        ),
    }
