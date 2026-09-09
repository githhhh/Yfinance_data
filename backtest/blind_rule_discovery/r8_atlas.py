"""Winner/Stop stable feature atlas and bounded interaction evaluation.

Primary discrimination is Fast Winner vs Stop First. Unresolved and ambiguous
rows remain visible in descriptive outputs and are never relabeled.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .dataset import DISCOVERY_FEATURE_ALLOWLIST
from .r6_stability import (
    BINARY_OPS,
    LABELS,
    LEAF_OPS,
    canonical,
    digest,
    evaluate_expression,
    numeric,
    purged_before,
)


@dataclass(frozen=True)
class AtlasConfig:
    min_train_quarters: int = 6
    min_class_n: int = 8
    min_matched_snapshots: int = 3
    min_stable_quarters: int = 6
    stable_direction_fraction: float = 0.75
    min_abs_cliffs_delta: float = 0.10
    inner_min_quarters: int = 3
    agent_rounds: int = 3
    proposals_per_round: int = 2
    patience: int = 2
    max_frozen_per_fold: int = 3
    min_interaction_selected: int = 10
    min_interaction_complement: int = 10


PRIMARY = ("fast_winner_3w", "stop_first_3w")
TARGETS = {"winner": "fast_winner_3w", "stop": "stop_first_3w"}
PROFILE_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
QUINTILE_CUTS = (0.20, 0.40, 0.60, 0.80)


def finite(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def cliffs_delta(x, y) -> float | None:
    """P(X>Y)-P(X<Y), with ties contributing zero."""
    x, y = finite(x), np.sort(finite(y))
    if not len(x) or not len(y):
        return None
    greater = np.searchsorted(y, x, side="left").sum()
    less = (len(y) - np.searchsorted(y, x, side="right")).sum()
    return float((greater - less) / (len(x) * len(y)))


def train_percentile(reference, values) -> np.ndarray:
    ref = np.sort(finite(reference))
    values = np.asarray(values, dtype=float)
    if not len(ref):
        return np.full(len(values), np.nan)
    out = (np.searchsorted(ref, values, side="left") +
           np.searchsorted(ref, values, side="right")) / (2 * len(ref))
    out[~np.isfinite(values)] = np.nan
    return out


def quantile_or_none(values, q: float) -> float | None:
    values = finite(values)
    return float(np.quantile(values, q)) if len(values) else None


def class_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {label: numeric(frame[label]) == 1 for label in LABELS}


def class_profiles(frame: pd.DataFrame, features: list[str], calendar: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    quarters = frame.snapshot_date.dt.to_period("Q").astype(str)
    for quarter in calendar:
        qframe = frame.loc[quarters == quarter]
        masks = class_masks(qframe)
        for feature in features:
            values = numeric(qframe[feature]) if feature in qframe else np.full(len(qframe), np.nan)
            for label in LABELS:
                mask = masks[label]
                raw = values[mask]
                known = finite(raw)
                row = {
                    "quarter": quarter,
                    "feature": feature,
                    "path_class": label,
                    "class_n": int(mask.sum()),
                    "known_n": int(len(known)),
                    "missing_fraction": float(1 - len(known) / mask.sum()) if mask.sum() else None,
                    "mean": float(known.mean()) if len(known) else None,
                }
                for q in PROFILE_QUANTILES:
                    row[f"q{int(q*100):02d}"] = float(np.quantile(known, q)) if len(known) else None
                rows.append(row)
    return pd.DataFrame(rows)


def _tail_log_odds(winner: np.ndarray, stop: np.ndarray, selected: np.ndarray) -> float | None:
    primary = winner | stop
    if not primary.any() or not (selected & primary).any():
        return None
    w, s = int((winner & selected).sum()), int((stop & selected).sum())
    all_w, all_s = int(winner.sum()), int(stop.sum())
    if not all_w or not all_s:
        return None
    tail_odds = (w + 0.5) / (s + 0.5)
    base_odds = (all_w + 0.5) / (all_s + 0.5)
    return float(np.log(tail_odds / base_odds))


def _matched_percentile_gaps(frame: pd.DataFrame, pct: np.ndarray) -> list[float]:
    winner = numeric(frame.fast_winner_3w) == 1
    stop = numeric(frame.stop_first_3w) == 1
    dates = pd.to_datetime(frame.snapshot_date).to_numpy()
    gaps: list[float] = []
    for date in np.unique(dates):
        wm = winner & (dates == date) & np.isfinite(pct)
        sm = stop & (dates == date) & np.isfinite(pct)
        if wm.any() and sm.any():
            gaps.append(float(np.mean(pct[wm]) - np.mean(pct[sm])))
    return gaps


def quarter_feature_contrasts(frame: pd.DataFrame, features: list[str], calendar: list[str],
                              cfg: AtlasConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    quarters = frame.snapshot_date.dt.to_period("Q").astype(str)
    for quarter in calendar[cfg.min_train_quarters:]:
        start = pd.Period(quarter, freq="Q").start_time
        fit = purged_before(frame, start)
        test = frame.loc[quarters == quarter]
        for feature in features:
            if test.empty:
                rows.append({"quarter": quarter, "feature": feature, "status": "EMPTY_TEST_QUARTER"})
                continue
            reference = numeric(fit[feature])
            ref = finite(reference)
            if not len(ref):
                rows.append({"quarter": quarter, "feature": feature, "status": "NO_TRAIN_FEATURE_SUPPORT"})
                continue
            values = numeric(test[feature])
            pct = train_percentile(reference, values)
            winner = numeric(test.fast_winner_3w) == 1
            stop = numeric(test.stop_first_3w) == 1
            w_known = winner & np.isfinite(values)
            s_known = stop & np.isfinite(values)
            gaps = _matched_percentile_gaps(test, pct)
            supported = (w_known.sum() >= cfg.min_class_n and s_known.sum() >= cfg.min_class_n
                         and len(gaps) >= cfg.min_matched_snapshots)
            q20, q80 = float(np.quantile(ref, .2)), float(np.quantile(ref, .8))
            low = np.isfinite(values) & (values <= q20)
            high = np.isfinite(values) & (values >= q80)
            raw_delta = cliffs_delta(values[w_known], values[s_known])
            rows.append({
                "quarter": quarter,
                "feature": feature,
                "status": "SUPPORTED" if supported else "INSUFFICIENT_SUPPORT",
                "test_n": len(test),
                "winner_n": int(winner.sum()),
                "stop_n": int(stop.sum()),
                "known_winner_n": int(w_known.sum()),
                "known_stop_n": int(s_known.sum()),
                "winner_missing_fraction": float(1-w_known.sum()/winner.sum()) if winner.sum() else None,
                "stop_missing_fraction": float(1-s_known.sum()/stop.sum()) if stop.sum() else None,
                "raw_winner_median": quantile_or_none(values[w_known], .5),
                "raw_stop_median": quantile_or_none(values[s_known], .5),
                "cliffs_delta_winner_minus_stop": raw_delta,
                "winner_percentile_mean": float(np.nanmean(pct[winner])) if np.isfinite(pct[winner]).any() else None,
                "stop_percentile_mean": float(np.nanmean(pct[stop])) if np.isfinite(pct[stop]).any() else None,
                "percentile_mean_gap": (float(np.nanmean(pct[winner])-np.nanmean(pct[stop]))
                                        if np.isfinite(pct[winner]).any() and np.isfinite(pct[stop]).any() else None),
                "matched_snapshots": len(gaps),
                "matched_percentile_gap_median": float(np.median(gaps)) if gaps else None,
                "matched_winner_high_fraction": float(np.mean(np.asarray(gaps) > 0)) if gaps else None,
                "train_q20": q20,
                "train_q80": q80,
                "low_tail_log_odds_winner_vs_stop": _tail_log_odds(winner, stop, low),
                "high_tail_log_odds_winner_vs_stop": _tail_log_odds(winner, stop, high),
            })
    return pd.DataFrame(rows)


def _sign(value: float | None) -> int:
    if value is None or not np.isfinite(value) or value == 0:
        return 0
    return 1 if value > 0 else -1


def aggregate_feature_stability(contrasts: pd.DataFrame, cfg: AtlasConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature, group in contrasts.groupby("feature", sort=True):
        supported = group.loc[group.status == "SUPPORTED"].copy()
        gaps = supported.matched_percentile_gap_median.dropna().to_numpy(float)
        deltas = supported.cliffs_delta_winner_minus_stop.dropna().to_numpy(float)
        pos, neg = int((gaps > 0).sum()), int((gaps < 0).sum())
        directional_n = pos + neg
        consistency = max(pos, neg) / directional_n if directional_n else None
        median_gap = float(np.median(gaps)) if len(gaps) else None
        median_delta = float(np.median(deltas)) if len(deltas) else None
        full_sign = _sign(median_gap)
        loo_sign_stable = False
        if len(gaps) >= 2 and full_sign:
            loo_sign_stable = all(_sign(float(np.median(np.delete(gaps, i)))) == full_sign for i in range(len(gaps)))
        enough = len(supported) >= cfg.min_stable_quarters
        aligned = full_sign and _sign(median_delta) == full_sign
        practical = median_delta is not None and abs(median_delta) >= cfg.min_abs_cliffs_delta
        stable = enough and consistency is not None and consistency >= cfg.stable_direction_fraction and aligned and practical and loo_sign_stable
        label = ("CONSISTENT_WINNER_HIGH" if stable and full_sign > 0 else
                 "CONSISTENT_STOP_HIGH" if stable and full_sign < 0 else
                 "INSUFFICIENT_EVIDENCE" if not enough else "MIXED_OR_WEAK")
        strength = ("MODERATE_PLUS" if median_delta is not None and abs(median_delta) >= .20 else
                    "SMALL" if median_delta is not None and abs(median_delta) >= .10 else "WEAK")
        rows.append({
            "feature": feature,
            "stability_label": label,
            "effect_strength": strength,
            "supported_quarters": len(supported),
            "winner_high_quarters": pos,
            "stop_high_quarters": neg,
            "direction_consistency": consistency,
            "median_matched_percentile_gap": median_gap,
            "median_cliffs_delta": median_delta,
            "loo_median_sign_stable": loo_sign_stable,
            "median_low_tail_log_odds": (float(supported.low_tail_log_odds_winner_vs_stop.median())
                                         if supported.low_tail_log_odds_winner_vs_stop.notna().any() else None),
            "median_high_tail_log_odds": (float(supported.high_tail_log_odds_winner_vs_stop.median())
                                          if supported.high_tail_log_odds_winner_vs_stop.notna().any() else None),
            "median_winner_minus_stop_missing": (float((supported.winner_missing_fraction-supported.stop_missing_fraction).median())
                                                  if len(supported) else None),
            "production_change": False,
        })
    return pd.DataFrame(rows)


def quintile_surfaces(frame: pd.DataFrame, features: list[str], calendar: list[str],
                      cfg: AtlasConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    quarters = frame.snapshot_date.dt.to_period("Q").astype(str)
    for quarter in calendar[cfg.min_train_quarters:]:
        start = pd.Period(quarter, freq="Q").start_time
        fit = purged_before(frame, start)
        test = frame.loc[quarters == quarter]
        for feature in features:
            ref = finite(numeric(fit[feature]))
            if test.empty or not len(ref):
                for bin_no in range(1, 6):
                    rows.append({"quarter": quarter, "feature": feature, "bin": bin_no,
                                 "status": "EMPTY_TEST_QUARTER" if test.empty else "NO_TRAIN_FEATURE_SUPPORT",
                                 "n": 0})
                continue
            cuts = np.quantile(ref, QUINTILE_CUTS)
            values = numeric(test[feature])
            known = np.isfinite(values)
            bins = np.full(len(test), 0, dtype=int)
            bins[known] = np.digitize(values[known], cuts, right=True) + 1
            masks = class_masks(test)
            for bin_no in range(1, 6):
                selected = bins == bin_no
                n = int(selected.sum())
                winner_n = int((selected & masks["fast_winner_3w"]).sum())
                stop_n = int((selected & masks["stop_first_3w"]).sum())
                primary_n = winner_n + stop_n
                rows.append({
                    "quarter": quarter,
                    "feature": feature,
                    "bin": bin_no,
                    "status": "OBSERVED" if n else "NO_ROWS",
                    "n": n,
                    "winner_n": winner_n,
                    "stop_n": stop_n,
                    "unresolved_n": int((selected & masks["unresolved_3w"]).sum()),
                    "ambiguous_n": int((selected & masks["ambiguous_3w"]).sum()),
                    "winner_rate": winner_n/n if n else None,
                    "stop_rate": stop_n/n if n else None,
                    "winner_share_within_primary": winner_n/primary_n if primary_n else None,
                    "train_q20": float(cuts[0]), "train_q40": float(cuts[1]),
                    "train_q60": float(cuts[2]), "train_q80": float(cuts[3]),
                })
    return pd.DataFrame(rows)


def _visit_expression(node: dict, allowed: set[str], depth: int = 0) -> set[str]:
    if not isinstance(node, dict) or depth > 2:
        raise ValueError("expression exceeds maximum depth 2")
    if node.get("op") in LEAF_OPS:
        if set(node) != {"op", "feature"} or node["feature"] not in allowed:
            raise ValueError("non-PIT or unavailable feature")
        return {node["feature"]}
    if node.get("op") in BINARY_OPS:
        if set(node) != {"op", "left", "right"}:
            raise ValueError("binary expression requires left/right")
        return _visit_expression(node["left"], allowed, depth+1) | _visit_expression(node["right"], allowed, depth+1)
    raise ValueError("unknown expression operation")


def validate_interaction(proposal: dict, features: list[str]) -> set[str]:
    required = {"name", "hypothesis", "expression", "target", "tail", "quantile"}
    if set(proposal) != required:
        raise ValueError("interaction requires name/hypothesis/expression/target/tail/quantile only")
    if not isinstance(proposal["name"], str) or not 1 <= len(proposal["name"]) <= 200:
        raise ValueError("invalid name")
    if not isinstance(proposal["hypothesis"], str) or not 1 <= len(proposal["hypothesis"]) <= 1000:
        raise ValueError("invalid hypothesis")
    if proposal["target"] not in TARGETS or proposal["tail"] not in {"high", "low"} or proposal["quantile"] not in {0.2, 0.8}:
        raise ValueError("target/tail/quantile outside frozen R8 contract")
    allowed = set(features) & set(DISCOVERY_FEATURE_ALLOWLIST)
    leaves = _visit_expression(proposal["expression"], allowed)
    if len(leaves) < 2:
        raise ValueError("R8 Agent proposals must combine at least two distinct PIT features")
    return leaves


def interaction_id(proposal: dict) -> str:
    return digest({k: proposal[k] for k in ("expression", "target", "tail", "quantile")})


def fit_interaction_rule(proposal: dict, train: pd.DataFrame) -> dict:
    values = evaluate_expression(proposal["expression"], train, train)
    values = finite(values)
    return {**proposal, "threshold": float(np.quantile(values, proposal["quantile"])) if len(values) else None}


def apply_interaction_rule(rule: dict, train: pd.DataFrame, data: pd.DataFrame) -> np.ndarray:
    if rule.get("threshold") is None:
        return np.zeros(len(data), dtype=bool)
    values = evaluate_expression(rule["expression"], train, data)
    selected = values >= rule["threshold"] if rule["tail"] == "high" else values <= rule["threshold"]
    return np.isfinite(values) & selected


def assess_interaction(frame: pd.DataFrame, mask: np.ndarray, target: str, cfg: AtlasConfig) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != len(frame) or target not in TARGETS:
        raise ValueError("invalid interaction assessment")
    winner = numeric(frame.fast_winner_3w) == 1
    stop = numeric(frame.stop_first_3w) == 1
    primary = winner | stop
    target_mask = winner if target == "winner" else stop
    other_mask = stop if target == "winner" else winner
    selected, complement = mask & primary, ~mask & primary
    dates = pd.to_datetime(frame.snapshot_date).to_numpy()
    lifts: list[float] = []
    for date in np.unique(dates):
        a, b = selected & (dates == date), complement & (dates == date)
        if a.any() and b.any():
            lifts.append(float(target_mask[a].mean() - target_mask[b].mean()))
    capture = float((target_mask & selected).sum() / target_mask.sum()) if target_mask.sum() else None
    other_loss = float((other_mask & selected).sum() / other_mask.sum()) if other_mask.sum() else None
    supported = (selected.sum() >= cfg.min_interaction_selected
                 and complement.sum() >= cfg.min_interaction_complement
                 and len(lifts) >= cfg.min_matched_snapshots
                 and capture is not None and other_loss is not None)
    return {
        "test_n": len(frame),
        "selected_n_total": int(mask.sum()),
        "selected_primary_n": int(selected.sum()),
        "complement_primary_n": int(complement.sum()),
        "coverage_all": float(mask.mean()) if len(mask) else None,
        "matched_snapshots": len(lifts),
        "matched_target_lift": float(np.mean(lifts)) if lifts else None,
        "matched_target_positive_fraction": float(np.mean(np.asarray(lifts) > 0)) if lifts else None,
        "target_capture": capture,
        "other_class_loss": other_loss,
        "capture_minus_other_loss": None if capture is None or other_loss is None else capture-other_loss,
        "selected_unresolved_n": int((mask & (numeric(frame.unresolved_3w) == 1)).sum()),
        "selected_ambiguous_n": int((mask & (numeric(frame.ambiguous_3w) == 1)).sum()),
        "supported": bool(supported),
    }


def inner_interaction_evidence(proposal: dict, past: pd.DataFrame, prior_calendar: list[str],
                               cfg: AtlasConfig) -> dict[str, Any]:
    rows = []
    for quarter in prior_calendar[cfg.inner_min_quarters:]:
        period = pd.Period(quarter, freq="Q")
        fit = purged_before(past, period.start_time)
        test = past.loc[past.snapshot_date.dt.to_period("Q").astype(str) == quarter]
        if fit.empty or test.empty:
            continue
        rule = fit_interaction_rule(proposal, fit)
        metrics = assess_interaction(test, apply_interaction_rule(rule, fit, test), proposal["target"], cfg)
        rows.append({"quarter": quarter, **metrics})
    supported = [r for r in rows if r["supported"]]
    lifts = [r["matched_target_lift"] for r in supported if r["matched_target_lift"] is not None]
    net = [r["capture_minus_other_loss"] for r in supported if r["capture_minus_other_loss"] is not None]
    return {
        "folds": rows,
        "supported_quarters": len(supported),
        "evaluated_quarters": len(rows),
        "positive_fraction_all_quarters": (sum(x > 0 for x in lifts)/len(rows) if rows else 0.0),
        "median_matched_target_lift": float(np.median(lifts)) if lifts else None,
        "median_capture_minus_other_loss": float(np.median(net)) if net else None,
    }


def interaction_evidence_key(evidence: dict) -> tuple:
    return (
        evidence["positive_fraction_all_quarters"],
        evidence["median_capture_minus_other_loss"] if evidence["median_capture_minus_other_loss"] is not None else -np.inf,
        evidence["median_matched_target_lift"] if evidence["median_matched_target_lift"] is not None else -np.inf,
    )


def interaction_qualifies(evidence: dict, cfg: AtlasConfig) -> bool:
    return (evidence["supported_quarters"] >= cfg.inner_min_quarters
            and evidence["positive_fraction_all_quarters"] >= 2/3
            and evidence["median_matched_target_lift"] is not None
            and evidence["median_matched_target_lift"] > 0
            and evidence["median_capture_minus_other_loss"] is not None
            and evidence["median_capture_minus_other_loss"] > 0)


def brief_interaction_evidence(evidence: dict) -> dict:
    return {
        "supported_quarters": evidence["supported_quarters"],
        "evaluated_quarters": evidence["evaluated_quarters"],
        "positive_fraction_all_quarters": evidence["positive_fraction_all_quarters"],
        "median_matched_target_lift": evidence["median_matched_target_lift"],
        "median_capture_minus_other_loss": evidence["median_capture_minus_other_loss"],
        "folds": [{k: row[k] for k in ("quarter", "supported", "selected_primary_n", "matched_snapshots",
                                         "matched_target_lift", "capture_minus_other_loss")}
                  for row in evidence["folds"]],
    }


def past_univariate_summary(past: pd.DataFrame, features: list[str]) -> list[dict[str, Any]]:
    quarters = past.snapshot_date.dt.to_period("Q").astype(str)
    result = []
    for feature in features:
        values = numeric(past[feature])
        winner = numeric(past.fast_winner_3w) == 1
        stop = numeric(past.stop_first_3w) == 1
        delta = cliffs_delta(values[winner], values[stop])
        quarter_delta = []
        for quarter in sorted(quarters.unique()):
            q = quarters == quarter
            w, s = q & winner & np.isfinite(values), q & stop & np.isfinite(values)
            if w.sum() >= 5 and s.sum() >= 5:
                d = cliffs_delta(values[w], values[s])
                if d is not None:
                    quarter_delta.append(d)
        result.append({
            "feature": feature,
            "winner_known_n": int((winner & np.isfinite(values)).sum()),
            "stop_known_n": int((stop & np.isfinite(values)).sum()),
            "cliffs_delta_winner_minus_stop": delta,
            "quarter_support": len(quarter_delta),
            "winner_high_quarter_fraction": (float(np.mean(np.asarray(quarter_delta) > 0)) if quarter_delta else None),
            "winner_missing_fraction": float(1-(winner & np.isfinite(values)).sum()/winner.sum()) if winner.sum() else None,
            "stop_missing_fraction": float(1-(stop & np.isfinite(values)).sum()/stop.sum()) if stop.sum() else None,
        })
    return result


def _expression_leaves(node: dict) -> set[str]:
    if node["op"] in LEAF_OPS:
        return {node["feature"]}
    return _expression_leaves(node["left"]) | _expression_leaves(node["right"])


def discover_interactions(frame: pd.DataFrame, features: list[str], calendar: list[str],
                          propose: Callable[[dict], dict], cfg: AtlasConfig, output: Path):
    """Past-only Agent discovery; all fold rules freeze before outer evaluation."""
    frozen: list[dict[str, Any]] = []
    fold_status: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    quarters = frame.snapshot_date.dt.to_period("Q").astype(str)
    trace_path = output / "discovery_trace.jsonl"
    for quarter in calendar[cfg.min_train_quarters:]:
        test = frame.loc[quarters == quarter]
        if test.empty:
            fold_status.append({"quarter": quarter, "status": "EMPTY_TEST_QUARTER", "frozen_n": 0})
            continue
        start = pd.Period(quarter, freq="Q").start_time
        past = purged_before(frame, start)
        prior_calendar = [q for q in calendar if q < quarter]
        candidates: dict[str, tuple[dict, dict]] = {}
        feedback = []
        stagnant, best_key = 0, (-np.inf, -np.inf, -np.inf)
        base_profile = past_univariate_summary(past, features)
        for round_index in range(cfg.agent_rounds):
            payload = {
                "fold": quarter,
                "round": round_index,
                "training_rows": len(past),
                "training_max_label_date": str(pd.to_datetime(past.exit_date_w3).max().date()) if len(past) else None,
                "univariate_profile": base_profile,
                "feedback": feedback,
            }
            response = propose(payload)
            if not isinstance(response, dict) or not isinstance(response.get("proposals"), list):
                raise ValueError("R8 Agent response requires proposals list")
            if len(response["proposals"]) > cfg.proposals_per_round:
                raise ValueError("R8 Agent exceeded proposals-per-round contract")
            record = {"fold": quarter, "round": round_index, "prompt_digest": digest(payload),
                      "response": response, "accepted": [], "rejected": []}
            for proposal in response["proposals"]:
                try:
                    validate_interaction(proposal, features)
                    pid = interaction_id(proposal)
                    if pid in candidates:
                        raise ValueError("duplicate interaction expression/target/tail/quantile")
                    evidence = inner_interaction_evidence(proposal, past, prior_calendar, cfg)
                    candidates[pid] = (proposal, evidence)
                    record["accepted"].append(pid)
                    feedback.append({"proposal": proposal, "evidence": brief_interaction_evidence(evidence)})
                except (ValueError, TypeError, KeyError) as exc:
                    record["rejected"].append({"proposal": proposal, "reason": str(exc)})
            traces.append(record)
            with trace_path.open("a") as handle:
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
        fold_status.append({"quarter": quarter,
                            "status": "FROZEN" if chosen else "NO_SUPPORTED_INTERACTION",
                            "candidate_n": len(candidates), "qualified_n": len(eligible), "frozen_n": len(chosen)})
    lock = {"config": asdict(cfg), "test_feedback_used": False, "fold_status": fold_status, "rules": frozen}
    (output / "interaction_frozen.json").write_text(canonical(lock) + "\n")

    outer_rows: list[dict[str, Any]] = []
    for item in frozen:
        quarter = item["fold"]
        start = pd.Period(quarter, freq="Q").start_time
        past = purged_before(frame, start)
        test = frame.loc[quarters == quarter]
        metrics = assess_interaction(test, apply_interaction_rule(item["rule"], past, test),
                                     item["rule"]["target"], cfg)
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
    outer = pd.DataFrame(outer_rows)
    stability_rows = []
    recurrence_rows = []
    if not outer.empty:
        for signature, group in outer.groupby("signature", sort=True):
            supported = group.loc[group.supported]
            lifts = supported.matched_target_lift.dropna().to_numpy(float)
            net = supported.capture_minus_other_loss.dropna().to_numpy(float)
            positive = float((lifts > 0).mean()) if len(lifts) else None
            stability_rows.append({
                "signature": signature,
                "name": group.iloc[0].name if "name" in group else None,
                "target": group.target.iloc[0],
                "features": group.features.iloc[0],
                "selected_folds": len(group),
                "supported_outer_folds": len(supported),
                "positive_outer_fraction": positive,
                "median_outer_target_lift": float(np.median(lifts)) if len(lifts) else None,
                "median_outer_capture_minus_other_loss": float(np.median(net)) if len(net) else None,
                "descriptive_verdict": ("RECURRING_DIRECTION" if len(group) >= 2 and len(supported) >= 2
                                        and positive is not None and positive >= .75
                                        and len(lifts) and np.median(lifts) > 0
                                        and len(net) and np.median(net) > 0 else "SPARSE_OR_MIXED"),
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
                "median_target_lift": (float(supported.matched_target_lift.median()) if len(supported) else None),
                "adaptive_descriptive_only": True,
            })
    return lock, outer, pd.DataFrame(stability_rows), pd.DataFrame(recurrence_rows), traces


def render_report(stability: pd.DataFrame, interactions: pd.DataFrame,
                  interaction_stability: pd.DataFrame, manifest: dict) -> str:
    lines = [
        "# R8 Winner / Stop Stable Feature Atlas", "",
        "Known-history retrospective research. NOT AN UNTOUCHED HOLDOUT.",
        "Primary contrast: fast_winner_3w vs stop_first_3w. Unresolved and ambiguous remain separate.",
        "No B0 ranking, production mutation, price refresh or EPS refresh occurs.", "",
        "## Bound Population", "",
        f"- Samples: {manifest['sample_rows']}; snapshot weeks: {manifest['snapshot_weeks']}; tickers: {manifest['unique_tickers']}.",
        f"- PIT features: {len(manifest['features'])}.",
        f"- Input SHA256: `{manifest['samples_sha256']}`.",
        "- Population is usable executable entries only; this is not all signals/listings.", "",
        "## Fixed Univariate Stability", "",
        "Every allowlisted feature is reported; no top-N prefilter was used.",
    ]
    stable = stability.loc[stability.stability_label.isin(["CONSISTENT_WINNER_HIGH", "CONSISTENT_STOP_HIGH"])] if not stability.empty else stability
    if stable.empty:
        lines.append("No feature met the predeclared descriptive stability label.")
    else:
        lines += ["| feature | label | supported_q | consistency | median matched pct gap | median Cliff delta |",
                  "|---|---|---:|---:|---:|---:|"]
        for r in stable.sort_values(["stability_label", "feature"]).to_dict("records"):
            lines.append(f"| {r['feature']} | {r['stability_label']} | {r['supported_quarters']} | "
                         f"{r['direction_consistency']:.3f} | {r['median_matched_percentile_gap']:.4f} | "
                         f"{r['median_cliffs_delta']:.4f} |")
    lines += ["", "These labels require >=6 supported outer quarters, >=75% same direction, |median Cliff delta|>=0.10, aligned median effects and leave-one-quarter sign stability. They are descriptive, not alpha certification.",
              "", "## RD-Agent Interaction Layer", "",
              f"- Provider attempts are metered separately; request accounting: {manifest.get('requests', {})}.",
              "- Agent sees purged-past aggregates only; all fold interactions freeze before outer evaluation.",
              "- Each expression combines at least two PIT features; no generated Python executes."]
    if interaction_stability.empty:
        lines.append("No recurring interaction stability result was produced.")
    else:
        recurring = interaction_stability.loc[interaction_stability.descriptive_verdict == "RECURRING_DIRECTION"]
        if recurring.empty:
            lines.append("No exact interaction signature met the recurring descriptive direction rule.")
        else:
            lines += ["| target | features | selected folds | supported folds | positive outer | median lift |",
                      "|---|---|---:|---:|---:|---:|"]
            for r in recurring.to_dict("records"):
                lines.append(f"| {r['target']} | {r['features']} | {r['selected_folds']} | {r['supported_outer_folds']} | "
                             f"{r['positive_outer_fraction']:.3f} | {r['median_outer_target_lift']:.4f} |")
    lines += ["", "## Boundaries", "",
              "- Raw four-class profiles and fixed quintile surfaces remain the primary atlas; Agent interactions are adaptive descriptive research.",
              "- Missingness is reported and never treated as safety.",
              "- No p-value, multiplicity correction, causal claim or production weight is implied.",
              "- Any future production hypothesis must be frozen before genuinely future observations.", "",
              "KEEP PRODUCTION FROZEN", ""]
    return "\n".join(lines)
