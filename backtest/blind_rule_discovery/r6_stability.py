"""R6 retrospective risk-feature discovery with an external RD-Agent proposer.

Only W3 labels enter discovery. All folds are frozen before any outer evaluation.
Expressions are interpreted by a bounded DSL, never by eval or generated Python.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .dataset import DISCOVERY_FEATURE_ALLOWLIST

LABELS = ("ambiguous_3w", "stop_first_3w", "fast_winner_3w", "unresolved_3w")
LEAF_OPS = {"raw", "train_percentile"}
BINARY_OPS = {"difference", "product", "minimum", "maximum"}


@dataclass(frozen=True)
class Config:
    min_train_quarters: int = 6
    min_inner_quarters: int = 3
    rounds: int = 6
    proposals_per_round: int = 3
    patience: int = 3
    min_selected: int = 10
    min_complement: int = 10
    min_snapshots: int = 3


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(float)


def validate_proposal(proposal: dict, features: list[str]) -> None:
    allowed = set(features) & set(DISCOVERY_FEATURE_ALLOWLIST)
    if set(proposal) != {"name", "hypothesis", "expression", "tail", "quantile"}:
        raise ValueError("proposal requires name/hypothesis/expression/tail/quantile only")
    for field in ("name", "hypothesis"):
        if not isinstance(proposal[field], str) or not 1 <= len(proposal[field]) <= 1000:
            raise ValueError(f"invalid {field}")
    if proposal["tail"] not in {"high", "low"} or proposal["quantile"] not in {0.2, 0.4, 0.6, 0.8}:
        raise ValueError("tail or quantile outside frozen search contract")

    def visit(node: dict, depth: int = 0) -> None:
        if not isinstance(node, dict) or depth > 2:
            raise ValueError("expression exceeds maximum depth 2")
        if node.get("op") in LEAF_OPS:
            if set(node) != {"op", "feature"} or node["feature"] not in allowed:
                raise ValueError("non-PIT or unavailable feature")
        elif node.get("op") in BINARY_OPS:
            if set(node) != {"op", "left", "right"}:
                raise ValueError("binary expression requires left/right")
            visit(node["left"], depth + 1)
            visit(node["right"], depth + 1)
        else:
            raise ValueError("unknown expression operation")
    visit(proposal["expression"])


def evaluate_expression(node: dict, fit: pd.DataFrame, data: pd.DataFrame) -> np.ndarray:
    op = node["op"]
    if op in LEAF_OPS:
        values = numeric(data[node["feature"]])
        if op == "raw":
            return values
        reference = numeric(fit[node["feature"]])
        reference = np.sort(reference[np.isfinite(reference)])
        if not len(reference):
            return np.full(len(data), np.nan)
        ranks = (np.searchsorted(reference, values, side="left")
                 + np.searchsorted(reference, values, side="right")) / (2 * len(reference))
        ranks[~np.isfinite(values)] = np.nan
        return ranks
    left = evaluate_expression(node["left"], fit, data)
    right = evaluate_expression(node["right"], fit, data)
    with np.errstate(over="ignore", invalid="ignore"):
        return {"difference": np.subtract, "product": np.multiply,
                "minimum": np.minimum, "maximum": np.maximum}[op](left, right)


def purged_before(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    dates = pd.to_datetime(frame.snapshot_date, errors="coerce")
    exits = pd.to_datetime(frame.exit_date_w3, errors="coerce")
    return frame.loc[(dates < cutoff) & (exits < cutoff)].reset_index(drop=True)


def fit_rule(proposal: dict, train: pd.DataFrame) -> dict:
    values = evaluate_expression(proposal["expression"], train, train)
    finite = values[np.isfinite(values)]
    return {**proposal, "threshold": float(np.quantile(finite, proposal["quantile"])) if len(finite) else None}


def apply_rule(rule: dict, train: pd.DataFrame, data: pd.DataFrame) -> np.ndarray:
    if rule["threshold"] is None:
        return np.zeros(len(data), dtype=bool)
    values = evaluate_expression(rule["expression"], train, data)
    comparison = values >= rule["threshold"] if rule["tail"] == "high" else values <= rule["threshold"]
    return np.isfinite(values) & comparison


def ratio(a: float, b: float) -> float | None:
    return float(a / b) if b > 0 else None


def assess(frame: pd.DataFrame, mask: np.ndarray) -> dict:
    """Risk association, removal cost, and equal-snapshot complementary contrast."""
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != len(frame):
        raise ValueError("mask length differs from frame")
    valid = numeric(frame.ambiguous_3w) == 0
    selected, complement = mask & valid, ~mask & valid
    stop = numeric(frame.stop_first_3w) == 1
    fast = numeric(frame.fast_winner_3w) == 1
    unresolved = numeric(frame.unresolved_3w) == 1
    stop_lifts, winner_lifts = [], []
    dates = pd.to_datetime(frame.snapshot_date).to_numpy()
    for date in np.unique(dates):
        a, b = selected & (dates == date), complement & (dates == date)
        if a.any() and b.any():
            stop_lifts.append(float(stop[a].mean() - stop[b].mean()))
            winner_lifts.append(float(fast[a].mean() - fast[b].mean()))
    capture = ratio((stop & selected).sum(), (stop & valid).sum())
    loss = ratio((fast & selected).sum(), (fast & valid).sum())
    return {
        "test_n": len(frame), "selected_n": int(mask.sum()),
        "evaluable_n": int(selected.sum()), "complement_n": int(complement.sum()),
        "ambiguous_n": int((mask & ~valid).sum()),
        "unresolved_n": int((selected & unresolved).sum()),
        "coverage": ratio(mask.sum(), len(frame)),
        "stop_rate": ratio((stop & selected).sum(), selected.sum()),
        "baseline_stop_rate": ratio((stop & valid).sum(), valid.sum()),
        "retained_stop_rate": ratio((stop & complement).sum(), complement.sum()),
        "stop_capture": capture, "winner_loss": loss,
        "capture_minus_winner_loss": None if capture is None or loss is None else capture-loss,
        "matched_snapshots": len(stop_lifts),
        "matched_stop_lift": float(np.mean(stop_lifts)) if stop_lifts else None,
        "matched_winner_lift": float(np.mean(winner_lifts)) if winner_lifts else None,
        "matched_stop_positive_fraction": float(np.mean(np.array(stop_lifts) > 0)) if stop_lifts else None,
    }


def supported(metrics: dict, cfg: Config) -> bool:
    return (metrics["evaluable_n"] >= cfg.min_selected
            and metrics["complement_n"] >= cfg.min_complement
            and metrics["matched_snapshots"] >= cfg.min_snapshots
            and metrics["capture_minus_winner_loss"] is not None)


def inner_evidence(proposal: dict, past: pd.DataFrame, quarters: list[str], cfg: Config) -> dict:
    rows = []
    for quarter in quarters[cfg.min_inner_quarters:]:
        period = pd.Period(quarter, freq="Q")
        fit = purged_before(past, period.start_time)
        data = past.loc[past.snapshot_date.dt.to_period("Q").astype(str) == quarter]
        if fit.empty:
            continue
        metrics = assess(data, apply_rule(fit_rule(proposal, fit), fit, data))
        rows.append({"quarter": quarter, "supported": supported(metrics, cfg), **metrics})
    usable = [row for row in rows if row["supported"]]
    lifts = [row["matched_stop_lift"] for row in usable]
    costs = [row["capture_minus_winner_loss"] for row in usable]
    return {
        "folds": rows, "supported_quarters": len(usable), "total_quarters": len(rows),
        "positive_fraction_all_quarters": sum(x > 0 for x in lifts) / len(rows) if rows else 0.,
        "median_matched_stop_lift": float(np.median(lifts)) if lifts else None,
        "worst_matched_stop_lift": min(lifts) if lifts else None,
        "median_capture_minus_winner_loss": float(np.median(costs)) if costs else None,
    }


def evidence_key(evidence: dict) -> tuple:
    return (evidence["positive_fraction_all_quarters"],
            evidence["median_capture_minus_winner_loss"] if evidence["median_capture_minus_winner_loss"] is not None else -np.inf,
            evidence["median_matched_stop_lift"] if evidence["median_matched_stop_lift"] is not None else -np.inf)


def qualifies(evidence: dict) -> bool:
    return (evidence["supported_quarters"] >= 3
            and evidence["positive_fraction_all_quarters"] >= 2/3
            and evidence["median_matched_stop_lift"] is not None
            and evidence["median_matched_stop_lift"] > 0
            and evidence["median_capture_minus_winner_loss"] is not None
            and evidence["median_capture_minus_winner_loss"] > 0)


def proposal_id(p: dict) -> str:
    return digest({k: p[k] for k in ("expression", "tail", "quantile")})


def brief_evidence(evidence: dict) -> dict:
    """Compact model feedback; full selected-rule metrics remain in local audits."""
    return {**{k: v for k, v in evidence.items() if k != "folds"},
            "folds": [{k: row[k] for k in ("quarter", "supported", "evaluable_n",
                         "matched_snapshots", "matched_stop_lift", "capture_minus_winner_loss")}
                      for row in evidence["folds"]]}


def discover(frame: pd.DataFrame, features: list[str], calendar: list[str],
             propose: Callable[[dict], dict], cfg: Config, output: Path) -> dict:
    """Freeze every fold before test evaluation; proposer sees aggregate past only."""
    if not features or set(features) - set(DISCOVERY_FEATURE_ALLOWLIST):
        raise ValueError("R6 requires an explicit snapshot-PIT allowlist")
    if calendar != [str(q) for q in pd.period_range(calendar[0], calendar[-1], freq="Q")]:
        raise ValueError("calendar must include every consecutive quarter")
    if min(cfg.rounds, cfg.patience, cfg.proposals_per_round, cfg.min_inner_quarters,
           cfg.min_selected, cfg.min_complement, cfg.min_snapshots) < 1:
        raise ValueError("positive research limits required")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("R6 requires a fresh output directory")
    frame = frame.copy()
    frame["snapshot_date"] = pd.to_datetime(frame.snapshot_date)
    frozen, traces = [], []
    for quarter in calendar[cfg.min_train_quarters:]:
        start = pd.Period(quarter, freq="Q").start_time
        past = purged_before(frame, start)
        prior_calendar = [q for q in calendar if q < quarter]
        simple, agent_candidates = {}, {}
        for feature in features:
            for tail, q in (("high", 0.8), ("low", 0.2)):
                p = dict(name=feature+"_"+tail, hypothesis="Single-feature comparator",
                         expression={"op": "raw", "feature": feature}, tail=tail, quantile=q)
                simple[proposal_id(p)] = (p, inner_evidence(p, past, prior_calendar, cfg))
        feature_profile = {}
        for feature in features:
            values = numeric(past[feature])
            finite = values[np.isfinite(values)]
            feature_profile[feature] = {
                "missing_fraction": float((~np.isfinite(values)).mean()) if len(values) else None,
                "q20_q50_q80": np.quantile(finite, [.2, .5, .8]).tolist() if len(finite) else [],
            }
        stagnant, best_key = 0, (-np.inf, -np.inf, -np.inf)
        feedback = []
        for round_index in range(cfg.rounds):
            if past.empty:
                break
            payload = {
                "fold": quarter, "round": round_index, "training_rows": len(past),
                "training_max_label_date": str(pd.to_datetime(past.exit_date_w3).max().date()),
                "feature_profile": feature_profile,
                "simple_comparators": [{"proposal": p, "evidence": brief_evidence(e)} for p, e in simple.values()],
                "feedback": feedback,
            }
            response = propose(payload)
            if not isinstance(response, dict) or not isinstance(response.get("proposals"), list):
                raise ValueError("Agent response requires a proposals list (empty means abstain)")
            if len(response["proposals"]) > cfg.proposals_per_round:
                raise ValueError("Agent exceeded proposals-per-round contract")
            record = {"fold": quarter, "round": round_index, "prompt_digest": digest(payload),
                      "response": response, "accepted": [], "rejected": []}
            for p in response["proposals"]:
                try:
                    validate_proposal(p, features)
                    pid = proposal_id(p)
                    if pid in agent_candidates:
                        raise ValueError("duplicate expression/tail/quantile")
                    evidence = inner_evidence(p, past, prior_calendar, cfg)
                    agent_candidates[pid] = (p, evidence)
                    record["accepted"].append(pid)
                    feedback.append({"proposal": p, "evidence": brief_evidence(evidence)})
                except (ValueError, TypeError, KeyError) as exc:
                    record["rejected"].append({"proposal": p, "reason": str(exc)})
            traces.append(record)
            with (output / "discovery_trace.jsonl").open("a") as handle:
                handle.write(canonical(record) + "\n")
            current_key = max((evidence_key(e) for _, e in agent_candidates.values()), default=best_key)
            stagnant = 0 if current_key > best_key else stagnant + 1
            best_key = max(best_key, current_key)
            if not response["proposals"] or stagnant >= cfg.patience:
                break
        for source, candidates in (("rdagent", agent_candidates), ("simple", simple)):
            eligible = [(pid, p, e) for pid, (p, e) in candidates.items() if qualifies(e)]
            eligible.sort(key=lambda item: item[0])
            eligible.sort(key=lambda item: evidence_key(item[2]), reverse=True)
            winner = eligible[0] if eligible else None
            # Only snapshot features and W3 availability identify the fitted surface.
            fitted_surface = past[["snapshot_date", "code", "exit_date_w3", *features]].to_csv(index=False)
            frozen.append({
                "fold": quarter, "source": source,
                "training_surface_sha256": hashlib.sha256(fitted_surface.encode()).hexdigest(),
                "training_rows": len(past), "candidate_count": len(candidates),
                "rule": fit_rule(winner[1], past) if winner else None,
                "inner_evidence": winner[2] if winner else None,
                "status": "FROZEN" if winner else "NO_SUPPORTED_RULE",
            })
    lock = {"config": asdict(cfg), "test_feedback_used": False, "rules": frozen}
    (output / "frozen_rules.json").write_text(canonical(lock) + "\n")
    rows = []
    for item in frozen:
        start = pd.Period(item["fold"], freq="Q").start_time
        past = purged_before(frame, start)
        test = frame.loc[frame.snapshot_date.dt.to_period("Q").astype(str) == item["fold"]]
        mask = apply_rule(item["rule"], past, test) if item["rule"] else np.zeros(len(test), dtype=bool)
        metrics = assess(test, mask)
        rows.append({"quarter": item["fold"], "source": item["source"],
                     "rule_status": item["status"], "supported": supported(metrics, cfg), **metrics})
    result = {"frozen": frozen, "folds": rows, "config": asdict(cfg),
              "frozen_sha256": digest(lock), "research_mode": "known_history_adaptive_retrospective",
              "request_rounds": len(traces), "production_change": False}
    pd.DataFrame(rows).to_csv(output / "outer_quarters.csv", index=False)
    (output / "summary.json").write_text(canonical(result) + "\n")
    return result


def write_report(result: dict, output: Path) -> Path:
    lines = ["# R6 Risk Feature Stability", "",
             "Known-history retrospective research. NOT AN UNTOUCHED HOLDOUT.",
             "RD-Agent backend proposes bounded expressions; the evaluator and ranking are local and fixed.",
             "This is a custom research loop, not the canonical fin_factor / CoSTEER experiment.",
             "Population: reconstructed signal candidates with usable executable entries; not all listings or all signals.",
             "Features are snapshot-PIT only. Risk flags are hypothetical removals, not portfolio P&L.",
             "Stop First is a W3 path event, not a long-term loser label. No 12-week labels enter selection.",
             "All outer rules were frozen before evaluation; each fold's prompts use purged past only.",
             "Inner feedback is adaptively reused and is not independent validation.",
             "No p-value/significance or causal claim. Repeat tickers and overlapping paths remain dependent.",
             "", "## Quarterly Evidence", "",
             "| Quarter | Source | Status | N flagged/evaluable | Matched snapshots | Stop lift | stop_capture | winner_loss | Supported |",
             "|---|---|---|---:|---:|---:|---:|---:|---|"]
    def fmt(value):
        return "N/A" if value is None else f"{value:.4f}"
    for r in result["folds"]:
        lines.append(f"| {r['quarter']} | {r['source']} | {r['rule_status']} | {r['selected_n']}/{r['evaluable_n']} | "
                     f"{r['matched_snapshots']} | {fmt(r['matched_stop_lift'])} | {fmt(r['stop_capture'])} | "
                     f"{fmt(r['winner_loss'])} | {r['supported']} |")
    lines += ["", "## Evidence Boundary", ""]
    for source in ("rdagent", "simple"):
        rows = [r for r in result["folds"] if r["source"] == source]
        good = [r for r in rows if r["supported"]]
        positive = sum(r["matched_stop_lift"] > 0 and r["capture_minus_winner_loss"] > 0 for r in good)
        lines.append(f"- {source}: favorable risk/cost direction {positive}/{len(rows)} total quarters; "
                     f"{len(good)} supported, {len(rows)-len(good)} unsupported/abstaining. "
                     "These counts are descriptive; no automatic promotion.")
    indexed = {(r["quarter"], r["source"]): r for r in result["folds"]}
    paired = []
    for quarter in sorted({r["quarter"] for r in result["folds"]}):
        agent, simple = indexed[(quarter, "rdagent")], indexed[(quarter, "simple")]
        if agent["supported"] and simple["supported"]:
            paired.append({"quarter": quarter,
                           "matched_stop_lift_delta": agent["matched_stop_lift"]-simple["matched_stop_lift"],
                           "winner_loss_delta": agent["winner_loss"]-simple["winner_loss"],
                           "coverage_delta": agent["coverage"]-simple["coverage"]})
    pd.DataFrame(paired, columns=["quarter", "matched_stop_lift_delta", "winner_loss_delta", "coverage_delta"]).to_csv(
        output / "agent_vs_simple.csv", index=False)
    lines += ["", "## RD-Agent Incremental Evidence", "",
              "Paired on commonly supported quarters only; different coverage remains visible.",
              "Positive stop-lift delta favors risk detection; positive winner-loss delta is a cost.",
              "| Quarter | Stop-lift delta | Winner-loss delta | Coverage delta |",
              "|---|---:|---:|---:|"]
    for row in paired:
        lines.append(f"| {row['quarter']} | {fmt(row['matched_stop_lift_delta'])} | "
                     f"{fmt(row['winner_loss_delta'])} | {fmt(row['coverage_delta'])} |")
    if not paired:
        lines.append("No common supported quarters: incremental advantage NOT DEMONSTRATED.")
    feature_rows = []
    def leaves(node):
        if node["op"] in LEAF_OPS:
            return {node["feature"]}
        return leaves(node["left"]) | leaves(node["right"])
    for item in result["frozen"]:
        if item["rule"] is None:
            continue
        outcome = indexed[(item["fold"], item["source"])]
        for feature in sorted(leaves(item["rule"]["expression"])):
            feature_rows.append({"feature": feature, "source": item["source"], "quarter": item["fold"],
                                 "supported": outcome["supported"],
                                 "positive_risk_cost": outcome["supported"] and outcome["matched_stop_lift"] > 0
                                 and outcome["capture_minus_winner_loss"] > 0})
    pd.DataFrame(feature_rows, columns=["feature", "source", "quarter", "supported", "positive_risk_cost"]).to_csv(
        output / "feature_stability.csv", index=False)
    lines += ["", "## Feature Recurrence", "",
              "Occurrence in a frozen expression is not independent feature attribution or a causal effect.",
              "| Feature | Source | Selected folds | Supported folds | Positive risk/cost folds |",
              "|---|---|---:|---:|---:|"]
    for feature, source in sorted({(r["feature"], r["source"]) for r in feature_rows}):
        group = [r for r in feature_rows if r["feature"] == feature and r["source"] == source]
        lines.append(f"| {feature} | {source} | {len(group)} | {sum(r['supported'] for r in group)} | "
                     f"{sum(r['positive_risk_cost'] for r in group)} |")
    lines += ["", "Compare RD-Agent and simple rules on the same quarters using outer_quarters.csv; "
              "coverage differs and this is not a Matched-N portfolio benchmark.",
              "A high stop_capture with high winner_loss may only describe high dispersion.",
              "Freeze any prospective candidate before collecting genuinely future evidence.",
              "", "KEEP PRODUCTION FROZEN", ""]
    path = output / "R6_REPORT.md"
    path.write_text("\n".join(lines))
    return path
