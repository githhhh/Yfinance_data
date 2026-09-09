"""Compact, optional RD-Agent interaction layer for R8.

This module is deliberately separate from the deterministic Winner/Stop atlas.
Agent failure must never invalidate or block an already completed atlas.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .r6_stability import LEAF_OPS, canonical, digest, numeric, purged_before
from .r8_atlas import (
    AtlasConfig,
    apply_interaction_rule,
    assess_interaction,
    cliffs_delta,
    fit_interaction_rule,
    inner_interaction_evidence,
    interaction_evidence_key,
    interaction_id,
    interaction_qualifies,
    validate_interaction,
)

PROFILE_SCHEMA = (
    "feature",
    "winner_known_n",
    "stop_known_n",
    "cliffs_delta",
    "quarter_support",
    "winner_high_quarter_fraction",
    "winner_minus_stop_missing",
)


def _rounded(value: float | None, digits: int = 4):
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def compact_past_profile(past: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    """Token-efficient fixed facts for every feature; no feature prefilter."""
    quarters = past.snapshot_date.dt.to_period("Q").astype(str)
    winner = numeric(past.fast_winner_3w) == 1
    stop = numeric(past.stop_first_3w) == 1
    rows: list[list[Any]] = []
    for feature in features:
        values = numeric(past[feature])
        known = np.isfinite(values)
        quarter_delta: list[float] = []
        for quarter in sorted(quarters.unique()):
            q = quarters == quarter
            w = q & winner & known
            s = q & stop & known
            if w.sum() >= 5 and s.sum() >= 5:
                delta = cliffs_delta(values[w], values[s])
                if delta is not None:
                    quarter_delta.append(delta)
        winner_known = int((winner & known).sum())
        stop_known = int((stop & known).sum())
        winner_missing = 1 - winner_known / winner.sum() if winner.sum() else np.nan
        stop_missing = 1 - stop_known / stop.sum() if stop.sum() else np.nan
        rows.append([
            feature,
            winner_known,
            stop_known,
            _rounded(cliffs_delta(values[winner], values[stop])),
            len(quarter_delta),
            _rounded(float(np.mean(np.asarray(quarter_delta) > 0)) if quarter_delta else None, 3),
            _rounded(winner_missing - stop_missing, 3),
        ])
    return {"schema": list(PROFILE_SCHEMA), "rows": rows}


def compact_feedback(proposal: dict, evidence: dict) -> dict[str, Any]:
    """Keep only aggregate inner evidence in the next model prompt.

    Full per-quarter evidence stays in the local frozen audit object and is never
    discarded; it is simply not echoed back into later prompts.
    """
    return {
        "proposal": proposal,
        "evidence": {
            "supported_quarters": evidence["supported_quarters"],
            "evaluated_quarters": evidence["evaluated_quarters"],
            "positive_fraction_all_quarters": _rounded(evidence["positive_fraction_all_quarters"], 3),
            "median_matched_target_lift": _rounded(evidence["median_matched_target_lift"]),
            "median_capture_minus_other_loss": _rounded(evidence["median_capture_minus_other_loss"]),
        },
    }


def _expression_leaves(node: dict) -> set[str]:
    if node["op"] in LEAF_OPS:
        return {node["feature"]}
    return _expression_leaves(node["left"]) | _expression_leaves(node["right"])


def discover_interactions_compact(
    frame: pd.DataFrame,
    features: list[str],
    calendar: list[str],
    propose: Callable[[dict], dict],
    cfg: AtlasConfig,
    output: Path,
):
    """Past-only compact Agent discovery; freeze every fold before outer use."""
    frozen: list[dict[str, Any]] = []
    fold_status: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    quarters = frame.snapshot_date.dt.to_period("Q").astype(str)
    trace_path = output / "discovery_trace.jsonl"
    trace_path.touch()

    for quarter in calendar[cfg.min_train_quarters:]:
        test = frame.loc[quarters == quarter]
        if test.empty:
            fold_status.append({"quarter": quarter, "status": "EMPTY_TEST_QUARTER", "frozen_n": 0})
            continue
        start = pd.Period(quarter, freq="Q").start_time
        past = purged_before(frame, start)
        prior_calendar = [q for q in calendar if q < quarter]
        candidates: dict[str, tuple[dict, dict]] = {}
        feedback: list[dict[str, Any]] = []
        stagnant = 0
        best_key = (-np.inf, -np.inf, -np.inf)
        base_profile = compact_past_profile(past, features)

        for round_index in range(cfg.agent_rounds):
            payload = {
                "contract": "R8_COMPACT_INTERACTION_V1",
                "fold": quarter,
                "round": round_index,
                "n": len(past),
                "max_label": str(pd.to_datetime(past.exit_date_w3).max().date()) if len(past) else None,
                "profile": base_profile,
                "feedback": feedback,
            }
            response = propose(payload)
            if not isinstance(response, dict) or not isinstance(response.get("proposals"), list):
                raise ValueError("R8 Agent response requires proposals list")
            if len(response["proposals"]) > cfg.proposals_per_round:
                raise ValueError("R8 Agent exceeded proposals-per-round contract")

            record = {
                "fold": quarter,
                "round": round_index,
                "prompt_digest": digest(payload),
                "prompt_chars": len(canonical(payload)),
                "response": response,
                "accepted": [],
                "rejected": [],
            }
            for proposal in response["proposals"]:
                try:
                    validate_interaction(proposal, features)
                    pid = interaction_id(proposal)
                    if pid in candidates:
                        raise ValueError("duplicate interaction expression/target/tail/quantile")
                    evidence = inner_interaction_evidence(proposal, past, prior_calendar, cfg)
                    candidates[pid] = (proposal, evidence)
                    record["accepted"].append(pid)
                    feedback.append(compact_feedback(proposal, evidence))
                except (ValueError, TypeError, KeyError) as exc:
                    record["rejected"].append({"proposal": proposal, "reason": str(exc)})
            traces.append(record)
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(canonical(record) + "\n")

            current_key = max((interaction_evidence_key(e) for _, e in candidates.values()), default=best_key)
            stagnant = 0 if current_key > best_key else stagnant + 1
            best_key = max(best_key, current_key)
            if not response["proposals"] or stagnant >= cfg.patience:
                break

        eligible = [(pid, p, e) for pid, (p, e) in candidates.items() if interaction_qualifies(e, cfg)]
        eligible.sort(key=lambda x: x[0])
        eligible.sort(key=lambda x: interaction_evidence_key(x[2]), reverse=True)
        chosen = eligible[:cfg.max_frozen_per_fold]
        for pid, proposal, evidence in chosen:
            rule = fit_interaction_rule(proposal, past)
            frozen.append({
                "fold": quarter,
                "interaction_id": pid,
                "signature": interaction_id(proposal),
                "features": sorted(_expression_leaves(proposal["expression"])),
                "rule": rule,
                "inner_evidence": evidence,
                "training_rows": len(past),
                "status": "FROZEN",
            })
        fold_status.append({
            "quarter": quarter,
            "status": "FROZEN" if chosen else "NO_SUPPORTED_INTERACTION",
            "candidate_n": len(candidates),
            "qualified_n": len(eligible),
            "frozen_n": len(chosen),
        })

    lock = {
        "config": asdict(cfg),
        "prompt_contract": "R8_COMPACT_INTERACTION_V1",
        "test_feedback_used": False,
        "fold_status": fold_status,
        "rules": frozen,
    }
    (output / "interaction_frozen.json").write_text(canonical(lock) + "\n", encoding="utf-8")

    # Outer evaluation begins only after every fold rule has been frozen above.
    outer_rows: list[dict[str, Any]] = []
    for item in frozen:
        quarter = item["fold"]
        start = pd.Period(quarter, freq="Q").start_time
        past = purged_before(frame, start)
        test = frame.loc[quarters == quarter]
        metrics = assess_interaction(
            test,
            apply_interaction_rule(item["rule"], past, test),
            item["rule"]["target"],
            cfg,
        )
        outer_rows.append({
            "quarter": quarter,
            "interaction_id": item["interaction_id"],
            "signature": item["signature"],
            "name": item["rule"]["name"],
            "target": item["rule"]["target"],
            "tail": item["rule"]["tail"],
            "quantile": item["rule"]["quantile"],
            "features": "+".join(item["features"]),
            **metrics,
        })

    outer_columns = [
        "quarter", "interaction_id", "signature", "name", "target", "tail", "quantile", "features",
        "test_n", "selected_n_total", "selected_primary_n", "complement_primary_n", "coverage_all",
        "matched_snapshots", "matched_target_lift", "matched_target_positive_fraction", "target_capture",
        "other_class_loss", "capture_minus_other_loss", "selected_unresolved_n", "selected_ambiguous_n", "supported",
    ]
    outer = pd.DataFrame(outer_rows, columns=outer_columns)
    stability_rows: list[dict[str, Any]] = []
    recurrence_rows: list[dict[str, Any]] = []
    if not outer.empty:
        for signature, group in outer.groupby("signature", sort=True):
            supported = group.loc[group.supported]
            lifts = supported.matched_target_lift.dropna().to_numpy(float)
            net = supported.capture_minus_other_loss.dropna().to_numpy(float)
            positive = float((lifts > 0).mean()) if len(lifts) else None
            stability_rows.append({
                "signature": signature,
                "name": group["name"].iloc[0],
                "target": group.target.iloc[0],
                "features": group.features.iloc[0],
                "selected_folds": len(group),
                "supported_outer_folds": len(supported),
                "positive_outer_fraction": positive,
                "median_outer_target_lift": float(np.median(lifts)) if len(lifts) else None,
                "median_outer_capture_minus_other_loss": float(np.median(net)) if len(net) else None,
                "descriptive_verdict": (
                    "RECURRING_DIRECTION"
                    if len(group) >= 2 and len(supported) >= 2
                    and positive is not None and positive >= .75
                    and len(lifts) and np.median(lifts) > 0
                    and len(net) and np.median(net) > 0
                    else "SPARSE_OR_MIXED"
                ),
                "independent_confirmation": False,
            })
        for (target, features_key), group in outer.groupby(["target", "features"], sort=True):
            supported = group.loc[group.supported]
            recurrence_rows.append({
                "target": target,
                "features": features_key,
                "frozen_occurrences": len(group),
                "supported_occurrences": len(supported),
                "positive_target_lift_occurrences": int((supported.matched_target_lift > 0).sum()),
                "median_target_lift": float(supported.matched_target_lift.median()) if len(supported) else None,
                "adaptive_descriptive_only": True,
            })

    stability_columns = [
        "signature", "name", "target", "features", "selected_folds", "supported_outer_folds",
        "positive_outer_fraction", "median_outer_target_lift", "median_outer_capture_minus_other_loss",
        "descriptive_verdict", "independent_confirmation",
    ]
    recurrence_columns = [
        "target", "features", "frozen_occurrences", "supported_occurrences",
        "positive_target_lift_occurrences", "median_target_lift", "adaptive_descriptive_only",
    ]
    return (
        lock,
        outer,
        pd.DataFrame(stability_rows, columns=stability_columns),
        pd.DataFrame(recurrence_rows, columns=recurrence_columns),
        traces,
    )
