"""Read-only R7 economic triage of committed local R4/R6 artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .pipeline_contract import sha256_file
from .r6_runner import load_inputs
from .r6_stability import (apply_rule, assess, canonical, digest, fit_rule, numeric,
                           purged_before, validate_proposal)
from .r7_economics import (BLOCK_WEEKS, BOOTSTRAP_DRAWS, FAMILIES, FEATURES, HORIZONS,
                           MIN_TRAIN_QUARTERS, SEED, evaluate, freeze_families)


def leaves(node):
    if node["op"] in {"raw", "train_percentile"}:
        return {node["feature"]}
    return leaves(node["left"]) | leaves(node["right"])


def load_bound_inputs(samples: Path, metadata: Path, r6_dir: Path):
    manifest = json.loads((r6_dir / "input_manifest.json").read_text())
    if sha256_file(metadata) != manifest["metadata_sha256"]:
        raise ValueError("R6 metadata SHA256 mismatch")
    projected, observed = load_inputs(samples, metadata, manifest["samples_sha256"])
    for key in ("features", "calendar", "sample_rows", "snapshot_weeks", "unique_tickers", "source_replay_dataset_sha256"):
        if manifest[key] != observed[key]:
            raise ValueError(f"R6 input binding mismatch: {key}")
    frame = pd.read_csv(samples, dtype={"code": str})
    required = {*FEATURES, *(f"return_{h}" for h in HORIZONS), *(f"exit_date_{h}" for h in HORIZONS)}
    if required - set(frame):
        raise ValueError(f"missing terminal economic fields: {sorted(required-set(frame))}")
    for col in ("snapshot_date", "entry_date", *(f"exit_date_{h}" for h in HORIZONS)):
        frame[col] = pd.to_datetime(frame[col], errors="raise")
        if frame[col].isna().any() or (frame[col] != frame[col].dt.normalize()).any():
            raise ValueError(f"missing/non-daily date: {col}")
    previous = frame.entry_date
    for horizon in HORIZONS:
        dates = frame[f"exit_date_{horizon}"]
        if not (dates > previous).all():
            raise ValueError("terminal horizon temporal order is invalid")
        previous = dates
        values = numeric(frame[f"return_{horizon}"])
        if not np.isfinite(values).all() or (values < -1).any():
            raise ValueError(f"nonfinite/impossible terminal return: {horizon}")
        frame[f"return_{horizon}"] = values
    snapshot_dates = frame.snapshot_date.drop_duplicates()
    if snapshot_dates.dt.to_period("W-FRI").duplicated().any():
        raise ValueError("multiple snapshots in one weekly calendar period")
    for col in FEATURES:
        converted = pd.to_numeric(frame[col], errors="coerce")
        if (frame[col].notna() & converted.isna()).any() or np.isinf(converted).any():
            raise ValueError(f"malformed feature: {col}")
        frame[col] = converted
    if (frame.pullback_pct.dropna() > 0).any():
        raise ValueError("pullback_pct must follow the non-positive depth schema")
    lock = json.loads((r6_dir / "frozen_rules.json").read_text())
    summary = json.loads((r6_dir / "summary.json").read_text())
    if digest(lock) != summary["frozen_sha256"] or lock.get("test_feedback_used") is not False:
        raise ValueError("R6 frozen rule provenance mismatch")
    if lock["config"]["min_train_quarters"] != MIN_TRAIN_QUARTERS:
        raise ValueError("R6 training calendar differs from frozen R7 protocol")
    return frame, projected, manifest, lock, summary


def frozen_r6_flags(projected: pd.DataFrame, manifest: dict, lock: dict, summary: dict):
    expected = {(q, s) for q in manifest["calendar"][MIN_TRAIN_QUARTERS:] for s in ("rdagent", "simple")}
    identities = [(r["fold"], r["source"]) for r in lock["rules"]]
    if set(identities) != expected or len(identities) != len(expected):
        raise ValueError("incomplete/duplicate R6 frozen rule calendar")
    result, flags = [], {}
    for item in lock["rules"]:
        quarter, source = item["fold"], item["source"]
        past = purged_before(projected, pd.Period(quarter, freq="Q").start_time)
        test = projected.loc[projected.snapshot_date.dt.to_period("Q").astype(str) == quarter]
        surface = past[["snapshot_date", "code", "exit_date_w3", *manifest["features"]]].to_csv(index=False)
        if hashlib.sha256(surface.encode()).hexdigest() != item["training_surface_sha256"]:
            raise ValueError("R6 frozen training surface mismatch")
        rule = item["rule"]
        if rule is None:
            if item["status"] != "NO_SUPPORTED_RULE":
                raise ValueError("invalid R6 empty-rule status")
            mask, known = np.zeros(len(test), bool), np.zeros(len(test), bool)
        else:
            proposal = {k: v for k, v in rule.items() if k != "threshold"}
            validate_proposal(proposal, manifest["features"])
            if fit_rule(proposal, past) != rule:
                raise ValueError("R6 threshold does not match frozen past-only fit")
            mask = apply_rule(rule, past, test)
            known = np.logical_and.reduce([np.isfinite(numeric(test[f])) for f in leaves(rule["expression"])])
        original = [r for r in summary["folds"] if r["quarter"] == quarter and r["source"] == source]
        actual = assess(test, mask)
        if len(original) != 1 or any(actual[k] != original[0][k] for k in actual):
            raise ValueError("R6 frozen selection does not reproduce original aggregate facts")
        policy = "r6_" + source
        result.append(dict(quarter=quarter, policy=policy, rule=rule, status=item["status"],
            training_surface_sha256=item["training_surface_sha256"], training_rows=len(past)))
        flags[(quarter, policy)] = (mask, known)
    return result, flags


def table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    def fmt(value):
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return "N/A"
        return f"{value:.6f}" if isinstance(value, float) else str(value)
    return ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"]*len(columns)) + " |",
            *("| " + " | ".join(fmt(row[c]) for c in columns) + " |" for row in frame.to_dict("records"))]


def render_report(tables: dict, manifest: dict) -> str:
    summary = tables["economic_summary"]
    primary = summary.loc[summary.role == "PRIMARY"]
    lines = ["# R7 Frozen Risk Economic Triage", "",
        "Known-history retrospective research. NOT AN UNTOUCHED HOLDOUT.",
        "RD-Agent calls: 0. No discovery, threshold search, champion selection or production change.",
        "Terminal close-return attribution, not portfolio P&L and not realized stop-execution P&L.",
        "W1/W2/W4 primary; W3 diagnostic only. No best-horizon selection.", "",
        "## Input and Protocol", "",
        f"- Samples: {manifest['sample_rows']}; usable-entry snapshot weeks: {manifest['snapshot_weeks']}; tickers: {manifest['unique_tickers']}.",
        f"- Input SHA256: `{manifest['samples_sha256']}`.",
        f"- Protocol SHA256: `{manifest['protocol_sha256']}`.",
        f"- Round-trip cost: {manifest['round_trip_cost_bps']} bps per admitted candidate, cash return zero.",
        "- Return fields and report numbers are decimal returns (0.01 = 1 percentage point). Gross and net are both retained.",
        "- Four existing PIT families retain q20/q80 past-only calibration with W3 purge after six calendar quarters.",
        "- R6 Agent/simple arms replay their frozen decisions, not their names or narratives; source thresholds and aggregate facts must reproduce.",
        "- All rules freeze before economic evaluation. All input files are read-only and hash checked.", "",
        "## What This Run Can Decide", "",
        "1. Whether flagged candidates have worse terminal distributions and higher W3 stop risk.",
        "2. Whether avoided terminal losses exceed foregone terminal gains, including unresolved and ambiguous path classes.",
        "3. Whether cash veto adds stock-specific value beyond exact same-week Matched-N random removal.",
        "4. Whether direction survives ticker overlap control, fees, quarterly splits and removal of the best week.",
        "5. Whether evidence is insufficient, mixed, or historically directionally positive. None is production approval.", "",
        "## Decision Matrix", "", *table(tables["decision_matrix"],
            ["policy", "risk_direction", "verdict", "primary_horizons_positive", "primary_horizons_cash_positive"]), "",
        "Verdicts are descriptive triage, not hypothesis-test passes. HISTORICAL_ECONOMIC_DIRECTION requires all three primary horizons to have positive mean cash delta and positive random-relative increment, at least three supported quarters each, and positive leave-one-quarter-out, leave-one-ticker-out and best-week-removed increments.",
        "DIRECTIONALLY_ELEVATED risk requires at least three supported quarters, positive equal-week mean stop lift and positive stop lift in at least two thirds of supported quarters. The risk-only flag in decision_matrix.csv distinguishes this from consistent economic direction.",
        "INSUFFICIENT_EVIDENCE includes inadequate exposure support. MIXED is not a reason to select a winning horizon.", "",
        "## Complete Primary Economic Matrix", "",
        *table(primary, ["policy", "panel", "horizon", "supported_quarters", "total_quarters", "available_weeks", "wholly_unknown_weeks", "coverage",
                         "baseline_net", "veto_cash_net", "random_cash_net", "incremental_mean", "cash_delta"]), "",
        "## Loss, Gain and Cost Attribution", "",
        *table(primary.loc[primary.panel == "nonoverlap_w4"], ["policy", "horizon", "avoided_gross_loss",
            "foregone_gross_gain", "saved_cost", "cash_delta_gross", "break_even_cost_bps"]), "",
        "cash_delta = avoided_gross_loss - foregone_gross_gain + saved_cost. Components have the SAME initial candidate-slot denominator. Break-even cost is algebraic, not a suggested or fitted fee; negative means gross cash delta is already positive.",
        "Costs cancel from incremental_vs_random because both policies remove exactly the same number each week. A cost-only improvement over baseline is not selection alpha.", "",
        "## Time Stability and Concentration", "",
        *table(primary.loc[primary.panel == "nonoverlap_w4"], ["policy", "horizon", "positive_supported_quarters",
            "block_ci_low", "block_ci_high", "leave_one_quarter_out_min", "leave_one_ticker_out_min", "without_best_week", "without_worst_week"]), "",
        "Intervals resample contiguous eight-calendar-week blocks, 2000 draws, seed 42. Missing weeks are not zero returns. Intervals are unavailable below sixteen calendar weeks. These are descriptive uncertainty ranges, not multiplicity-adjusted significance or independent validation; repeat issuers and adaptive research history remain relevant.", "",
        "## Quarterly Support and Direction", "",
        *table(tables["quarterly_summary"].loc[(tables["quarterly_summary"].panel == "nonoverlap_w4") &
            (tables["quarterly_summary"].horizon == "w4")], ["quarter", "policy", "status", "n", "flagged_n",
                "unknown_n", "matched_weeks", "stop_lift", "incremental_vs_random", "cash_delta"]), "",
        "The quarterly table above is W4 for compactness; quarterly_summary.csv includes ALL horizons and both panels, including empty quarters. SUPPORTED means at least 10 flagged, 10 known retained and three matched snapshots; it is not a significance claim.", "",
        "## Interpretation Boundaries", "",
        "- Every week's baseline gets one equal initial slot per usable entry. Veto leaves removed slots in cash; retained slots never receive extra weight. Random is the exact expectation of uniform removal of the identical N. There is no shrinking or survivor averaging.",
        "- This is conditional on executable entries and upstream maturity filtering, not the full replay universe. Missing/non-executable candidates cannot be recovered from this CSV. The inherited calendar retains empty edge quarters; they are no evidence, not failures.",
        "- The nonoverlap_w4 panel admits each ticker's earliest entry and reserves it through W4 close, identically for ALL policies and horizons. Veto does not free a later admission. This controls duplicate exposure without claiming a live policy simulation.",
        "- Initial slots across weeks are not one funded portfolio. No CAGR, Sharpe, portfolio drawdown, leverage or reinvestment claim is made.",
        "- Unknown features remain visible and unflagged, not declared safe; known-only return and stop contrasts are exported separately. No imputation or EPS refresh occurs.",
        "- Fully unknown-feature weeks remain in operational weekly accounting, but are absent (not zero-alpha) in evidence means, intervals and concentration diagnostics. TEST_FEATURE_UNAVAILABLE is distinct from observed NO_FLAGS; evidence_weeks and observed weeks remain explicit.",
        "- W3 ambiguous labels remain in terminal-return accounting because their closing return is observed. They are excluded ONLY from path-order risk rates. unresolved samples remain throughout.",
        "- Avoided losses are terminal close losses, not assumed -8% fills. Actual gap-aware stop P&L, stopped-path opportunity costs, and policy-dependent capital reuse are NOT_IDENTIFIABLE_FROM_TERMINAL_RETURN_CSV.",
        "- A favorable W4 mark-to-market result can coexist with early stop risk. This run must not justify ignoring stops or changing exits.",
        "- Group outcomes are candidate-weighted descriptive distributions; economic summaries are equal-week means. Neither is an annualized portfolio return.",
        "- No winning family is selected. Even a positive matrix remains a prospective hypothesis; inspected quarters never become untouched OOS again.", "",
        "## Deliverables", "",
        "- economic_summary.csv: all 48 policy/panel/horizon cells, fee and stability accounting.",
        "- quarterly_summary.csv / weekly_economics.csv: all denominators, known coverage, risk capture and paired contrasts.",
        "- label_contributions.csv: four exhaustive path classes, including unresolved and ambiguous, with additive loss/gain/fee attribution.",
        "- group_outcomes.csv: per-quarter baseline/flagged/retained/unknown distributions, p10/p90, positive rates and available MAE/MFE facts.",
        "- event_flags.csv / frozen_rules.json: ticker-week membership, shared nonoverlap admission and exact frozen thresholds.",
        "- ticker_concentration.csv: exact leave-one-ticker-out random-relative results without refitting rules, for every panel/horizon. This is a concentration diagnostic, not a ticker blacklist.",
        "- decision_matrix.csv / input_manifest.json / COMPLETE.json: descriptive decisions, provenance and completion hashes.", "",
        "KEEP PRODUCTION FROZEN", ""]
    return "\n".join(lines)


def run(samples: Path, metadata: Path, r6_dir: Path, output: Path, *, cost_bps: float, preflight=False):
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError("round-trip cost must be finite bps in [0, 10000)")
    protected = [samples, metadata, *(r6_dir / f for f in ("input_manifest.json", "frozen_rules.json", "summary.json"))]
    for path in (samples, metadata, r6_dir):
        a, b = output.resolve(), path.resolve()
        if a == b or a in b.parents or b in a.parents:
            raise ValueError("output must be disjoint from protected inputs")
    if output.exists() and any(output.iterdir()):
        raise ValueError("R7 requires a fresh output directory")
    before = {str(p.resolve()): sha256_file(p) for p in protected}
    frame, projected, anchor, r6_lock, r6_summary = load_bound_inputs(samples, metadata, r6_dir)
    frozen, flags = freeze_families(frame, anchor["calendar"])
    inherited, inherited_flags = frozen_r6_flags(projected, anchor, r6_lock, r6_summary)
    frozen.extend(inherited)
    flags.update(inherited_flags)
    protocol = dict(revision="R7_FROZEN_RISK_ECONOMIC_TRIAGE_V1", families=list(FAMILIES),
        inherited_arms=["r6_rdagent", "r6_simple"], horizons=list(HORIZONS),
        primary=["w1", "w2", "w4"], diagnostic=["w3"], min_train_quarters=MIN_TRAIN_QUARTERS,
        round_trip_cost_bps=cost_bps, cash_return=0, reinvestment=False,
        random_control="exact_uniform_same_snapshot_same_removed_n_expectation",
        block_weeks=BLOCK_WEEKS, bootstrap_draws=BOOTSTRAP_DRAWS, seed=SEED,
        production_change=False, rdagent_calls=0, research_mode="known_history_retrospective")
    manifest = {**{k: anchor[k] for k in ("samples_sha256", "metadata_sha256", "source_replay_dataset_sha256",
        "sample_rows", "snapshot_weeks", "unique_tickers", "calendar")},
        "protocol": protocol, "protocol_sha256": digest(protocol), "round_trip_cost_bps": cost_bps,
        "r6_frozen_sha256": digest(r6_lock), "frozen_rules_sha256": digest(frozen),
        "source_replay_sha_independently_recomputed": False,
        "code_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in
            ("r7_runner.py", "r7_economics.py", "r6_runner.py", "r6_stability.py", "stop_risk_validation_r5.py")}}
    if preflight:
        return manifest
    tables = evaluate(frame, anchor["calendar"], frozen, flags, cost_bps)
    for path in protected:
        if sha256_file(path) != before[str(path.resolve())]:
            raise RuntimeError("protected input changed during R7 evaluation")
    output.mkdir(parents=True, exist_ok=True)
    for name, table_frame in tables.items():
        table_frame.to_csv(output / f"{name}.csv", index=False)
    (output / "frozen_rules.json").write_text(canonical(frozen) + "\n")
    (output / "input_manifest.json").write_text(canonical(manifest) + "\n")
    (output / "R7_REPORT.md").write_text(render_report(tables, manifest))
    outputs = {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}
    if any(sha256_file(p) != before[str(p.resolve())] for p in protected):
        raise RuntimeError("protected input changed before R7 publication")
    (output / "COMPLETE.json").write_text(canonical(dict(status="COMPLETE",
        completed_at_utc=datetime.now(timezone.utc).isoformat(), output_sha256=outputs,
        inputs_unchanged=True, production_change=False, rdagent_calls=0)) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--r6-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--round-trip-cost-bps", type=float, required=True,
                        help="predeclared fee+slippage assumption; one value, not a search grid")
    parser.add_argument("--preflight", action="store_true", help="validate inputs and frozen arms without writing output")
    args = parser.parse_args()
    manifest = run(args.samples, args.metadata, args.r6_dir, args.output_root,
        cost_bps=args.round_trip_cost_bps, preflight=args.preflight)
    print(canonical(dict(status="PREFLIGHT_PASS" if args.preflight else "COMPLETE",
        protocol_sha256=manifest["protocol_sha256"], rdagent_calls=0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
