"""Run R8 as an offline atlas plus an independently auditable optional Agent layer."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .pipeline_contract import sha256_file
from .r6_runner import load_inputs
from .r6_stability import canonical, digest
from .r8_atlas import (
    AtlasConfig,
    aggregate_feature_stability,
    class_profiles,
    quarter_feature_contrasts,
    quintile_surfaces,
)
from .r8_interactions import discover_interactions_compact

TOTAL_PROVIDER_BUDGET = 1000
ATLAS_STAGE = "atlas"
AGENT_STAGE = "agent"


def load_bound_inputs(samples: Path, metadata: Path, r6_dir: Path):
    required = {name: r6_dir / name for name in ("input_manifest.json", "frozen_rules.json", "summary.json", "R6_REPORT.md")}
    if (r6_dir / "FAILED.json").exists() or any(not path.exists() for path in required.values()):
        raise ValueError("R8 requires a completed, non-failed R6 directory")
    manifest_path = required["input_manifest.json"]
    anchor = json.loads(manifest_path.read_text())
    lock = json.loads(required["frozen_rules.json"].read_text())
    summary = json.loads(required["summary.json"].read_text())
    if digest(lock) != summary.get("frozen_sha256") or summary.get("research_mode") != "known_history_adaptive_retrospective":
        raise ValueError("R8 R6 frozen-rule provenance mismatch")
    if sha256_file(metadata) != anchor.get("metadata_sha256"):
        raise ValueError("R8 metadata does not match completed R6")
    frame, observed = load_inputs(samples, metadata, anchor["samples_sha256"])
    for key in ("features", "calendar", "sample_rows", "snapshot_weeks", "unique_tickers",
                "source_replay_dataset_sha256"):
        if observed[key] != anchor.get(key):
            raise ValueError(f"R8/R6 input binding mismatch: {key}")
    requests = anchor.get("requests", {})
    accounted = requests.get("accounted_total")
    if not isinstance(accounted, int) or accounted < 0:
        raise ValueError("completed R6 manifest lacks accounted_total request floor")
    bound_files = [manifest_path, required["frozen_rules.json"], required["summary.json"]]
    return frame, observed, anchor, bound_files, accounted, summary["frozen_sha256"]


def nonoverlap_w3_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Shared outcome-independent issuer schedule: one ticker reserved through W3 exit."""
    admitted = np.zeros(len(frame), dtype=bool)
    busy: dict[str, pd.Timestamp] = {}
    ordered = frame.assign(_position=np.arange(len(frame))).sort_values(
        ["entry_date", "snapshot_date", "code"], kind="stable")
    for row in ordered.to_dict("records"):
        code, entry, exit_w3 = row["code"], row["entry_date"], row["exit_date_w3"]
        if code not in busy or entry > busy[code]:  # same close date cannot re-enter.
            admitted[row["_position"]] = True
            busy[code] = exit_w3
    return frame.loc[admitted].reset_index(drop=True)


def _execution_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _immutable_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256_file(path.resolve()) for path in paths}


def _assert_unchanged(before: dict[str, str]) -> None:
    for name, expected in before.items():
        path = Path(name)
        if sha256_file(path) != expected:
            raise RuntimeError(f"R8 protected input changed during execution: {path}")


def _require_fresh_output(output: Path, protected_roots: list[Path]) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError("R8 stage requires a fresh output directory")
    out = output.resolve()
    for path in protected_roots:
        path = path.resolve()
        if out == path or out in path.parents or path in out.parents:
            raise ValueError("R8 output must be disjoint from protected inputs/state")


def _base_manifest(observed: dict, r6_bound_files: list[Path], r6_frozen_sha: str,
                   r6_accounted: int) -> dict:
    return {
        **{k: observed[k] for k in ("samples_sha256", "metadata_sha256", "source_replay_dataset_sha256",
                                    "sample_rows", "snapshot_weeks", "unique_tickers", "calendar", "features")},
        "r6_anchor_manifest_sha256": sha256_file(r6_bound_files[0]),
        "r6_frozen_sha256": r6_frozen_sha,
        "r6_accounted_total_floor": r6_accounted,
        "execution_head": _execution_head(),
        "source_replay_sha_independently_recomputed": False,
        "population": observed["population"],
        "execution_features_excluded": True,
    }


def _write_complete(output: Path, status: str, extra: dict) -> dict:
    hashes = {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "COMPLETE.json"}
    complete = {
        "status": status,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_sha256": hashes,
        "inputs_unchanged": True,
        "production_change": False,
        **extra,
    }
    (output / "COMPLETE.json").write_text(canonical(complete) + "\n", encoding="utf-8")
    return complete


def _atlas_report(stability: pd.DataFrame, manifest: dict) -> str:
    primary = stability.loc[stability.panel == "nonoverlap_w3"]
    stable = primary.loc[primary.stability_label.isin(["CONSISTENT_WINNER_HIGH", "CONSISTENT_STOP_HIGH"])]
    lines = [
        "# R8A Winner / Stop Deterministic Feature Atlas",
        "",
        "Known-history retrospective research. NOT AN UNTOUCHED HOLDOUT.",
        "This stage is fully deterministic and makes zero RD-Agent/provider calls.",
        "Primary panel: nonoverlap_w3; all_entries is retained as a sensitivity panel.",
        "Winner, Stop, Unresolved and Ambiguous remain distinct path classes.",
        "",
        "## Bound Population",
        "",
        f"- Samples: {manifest['sample_rows']}; nonoverlap_w3 rows: {manifest['nonoverlap_w3_rows']}.",
        f"- Snapshot-PIT features: {len(manifest['features'])}.",
        f"- Input SHA256: `{manifest['samples_sha256']}`.",
        "",
        "## Stable Winner / Stop Features — nonoverlap_w3",
        "",
    ]
    if stable.empty:
        lines.append("No feature met the predeclared descriptive stability label.")
    else:
        lines += [
            "| feature | label | supported_q | consistency | median matched pct gap | median Cliff delta |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in stable.sort_values(["stability_label", "feature"]).to_dict("records"):
            lines.append(
                f"| {row['feature']} | {row['stability_label']} | {row['supported_quarters']} | "
                f"{row['direction_consistency']:.3f} | {row['median_matched_percentile_gap']:.4f} | "
                f"{row['median_cliffs_delta']:.4f} |"
            )
    lines += [
        "",
        "Labels require >=6 supported outer quarters, >=75% same matched-week direction, "
        "|median Cliff delta|>=0.10, aligned median effects and leave-one-quarter sign stability.",
        "They are descriptive facts, not alpha certification or production weights.",
        "",
        "## Outputs",
        "",
        "class_profiles.csv and quintile_surfaces.csv contain both all_entries and nonoverlap_w3 panels.",
        "quarter_feature_contrasts.csv and feature_stability.csv likewise retain both panels.",
        "No feature was prefiltered or omitted because of weak results.",
        "",
        "KEEP PRODUCTION FROZEN",
        "",
    ]
    return "\n".join(lines)


def run_atlas(samples: Path, metadata: Path, r6_dir: Path, output: Path, *, preflight: bool = False):
    frame, observed, _, r6_bound_files, r6_accounted, r6_frozen_sha = load_bound_inputs(samples, metadata, r6_dir)
    cfg = AtlasConfig()
    nonoverlap = nonoverlap_w3_frame(frame)
    protocol = {
        "revision": "R8A_DETERMINISTIC_WINNER_STOP_ATLAS_V2",
        "stage": ATLAS_STAGE,
        "config": asdict(cfg),
        "primary_contrast": ["fast_winner_3w", "stop_first_3w"],
        "context_classes": ["unresolved_3w", "ambiguous_3w"],
        "atlas_panels": ["all_entries", "nonoverlap_w3"],
        "report_primary_panel": "nonoverlap_w3",
        "nonoverlap_rule": "earliest entry per ticker; reserve through exit_date_w3; same-date reentry forbidden",
        "rdagent_calls": 0,
        "production_change": False,
        "research_mode": "known_history_retrospective_deterministic_atlas",
    }
    manifest = {
        **_base_manifest(observed, r6_bound_files, r6_frozen_sha, r6_accounted),
        "nonoverlap_w3_rows": len(nonoverlap),
        "protocol": protocol,
        "protocol_sha256": digest(protocol),
        "source_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("r8_runner.py", "r8_atlas.py", "r6_runner.py", "r6_stability.py", "dataset.py")
        },
    }
    if preflight:
        return {**manifest, "status": "PREFLIGHT_PASS", "rdagent_calls": 0}

    _require_fresh_output(output, [samples, metadata, r6_dir])
    immutable_files = [samples.resolve(), metadata.resolve(), *(p.resolve() for p in r6_bound_files)]
    before = _immutable_hashes(immutable_files)
    output.mkdir(parents=True, exist_ok=True)
    try:
        profile_parts = []
        surface_parts = []
        contrast_parts = []
        stability_parts = []
        for panel, panel_frame in (("all_entries", frame), ("nonoverlap_w3", nonoverlap)):
            profile_parts.append(class_profiles(panel_frame, manifest["features"], manifest["calendar"]).assign(panel=panel))
            surface_parts.append(quintile_surfaces(panel_frame, manifest["features"], manifest["calendar"], cfg).assign(panel=panel))
            contrasts = quarter_feature_contrasts(panel_frame, manifest["features"], manifest["calendar"], cfg).assign(panel=panel)
            contrast_parts.append(contrasts)
            stability_parts.append(aggregate_feature_stability(contrasts, cfg).assign(panel=panel))

        profiles = pd.concat(profile_parts, ignore_index=True)
        surfaces = pd.concat(surface_parts, ignore_index=True)
        contrasts = pd.concat(contrast_parts, ignore_index=True)
        stability = pd.concat(stability_parts, ignore_index=True)
        _assert_unchanged(before)

        profiles.to_csv(output / "class_profiles.csv", index=False)
        contrasts.to_csv(output / "quarter_feature_contrasts.csv", index=False)
        stability.to_csv(output / "feature_stability.csv", index=False)
        surfaces.to_csv(output / "quintile_surfaces.csv", index=False)
        (output / "input_manifest.json").write_text(canonical(manifest) + "\n", encoding="utf-8")
        (output / "R8_ATLAS_REPORT.md").write_text(_atlas_report(stability, manifest), encoding="utf-8")
        _assert_unchanged(before)
        _write_complete(output, "ATLAS_COMPLETE", {"rdagent_calls": 0})
        return manifest
    except Exception as exc:
        (output / "FAILED.json").write_text(canonical({"status": "ATLAS_FAILED", "error_type": type(exc).__name__}) + "\n")
        raise


def _model_from_env() -> str:
    return os.environ.get("RD_AGENT_MODEL") or os.environ.get("CHAT_MODEL") or ""


def _backend_version() -> str:
    try:
        version = importlib.metadata.version("rdagent")
        from rdagent.oai.backend.litellm import LiteLLMAPIBackend
        if not callable(LiteLLMAPIBackend.build_messages_and_create_chat_completion):
            raise AttributeError("missing RD-Agent chat backend")
        return version
    except (ImportError, importlib.metadata.PackageNotFoundError, AttributeError) as exc:
        raise RuntimeError("Compatible official RD-Agent backend is required in quant_env") from exc


def _resolve_prior_floor(ledger: Path, r6_accounted: int, requested: int | None) -> int:
    requested_floor = max(r6_accounted, int(requested or 0))
    if not ledger.exists():
        return requested_floor
    data = json.loads(ledger.read_text())
    hard_limit = data.get("hard_limit")
    if not isinstance(hard_limit, int) or not 0 < hard_limit <= TOTAL_PROVIDER_BUDGET:
        raise ValueError("existing R8 ledger has invalid hard_limit")
    ledger_floor = TOTAL_PROVIDER_BUDGET - hard_limit
    if ledger_floor < r6_accounted:
        raise ValueError("existing R8 ledger predates the completed R6 accounting floor")
    if requested is not None and requested_floor != ledger_floor:
        raise ValueError("requested prior-used floor conflicts with the existing R8 ledger")
    return ledger_floor


def _load_atlas_anchor(atlas_dir: Path, observed: dict, r6_frozen_sha: str) -> tuple[dict, Path]:
    if (atlas_dir / "FAILED.json").exists():
        raise ValueError("Agent stage cannot bind to a failed atlas")
    complete_path = atlas_dir / "COMPLETE.json"
    manifest_path = atlas_dir / "input_manifest.json"
    if not complete_path.exists() or not manifest_path.exists():
        raise ValueError("Agent stage requires a completed R8A atlas")
    complete = json.loads(complete_path.read_text())
    if complete.get("status") != "ATLAS_COMPLETE" or complete.get("rdagent_calls") != 0:
        raise ValueError("invalid R8A completion contract")
    for name, expected in complete.get("output_sha256", {}).items():
        path = atlas_dir / name
        if not path.exists() or sha256_file(path) != expected:
            raise ValueError(f"R8A output hash mismatch: {name}")
    manifest = json.loads(manifest_path.read_text())
    for key in ("samples_sha256", "metadata_sha256", "source_replay_dataset_sha256", "sample_rows",
                "snapshot_weeks", "unique_tickers", "calendar", "features"):
        if manifest.get(key) != observed.get(key):
            raise ValueError(f"R8B/R8A binding mismatch: {key}")
    if manifest.get("r6_frozen_sha256") != r6_frozen_sha:
        raise ValueError("R8B/R8A R6 frozen-rule binding mismatch")
    return manifest, complete_path


def _agent_report(stability: pd.DataFrame, manifest: dict) -> str:
    recurring = stability.loc[stability.descriptive_verdict == "RECURRING_DIRECTION"] if not stability.empty else stability
    lines = [
        "# R8B Optional RD-Agent Winner / Stop Interactions",
        "",
        "Known-history adaptive retrospective research. NOT AN UNTOUCHED HOLDOUT.",
        "This stage is independent of the already completed deterministic R8A atlas.",
        "Interaction discovery uses the nonoverlap_w3 issuer panel only.",
        "Prompts use a compact fixed schema containing every PIT feature; no top-N feature prefilter is used.",
        "All fold interactions freeze before outer evaluation, and outer results never enter proposal feedback.",
        "",
        f"Request accounting: {manifest.get('requests', {})}",
        "",
        "## Recurring interaction signatures",
        "",
    ]
    if recurring.empty:
        lines.append("No exact interaction signature met the recurring descriptive direction rule.")
    else:
        lines += [
            "| target | features | selected folds | supported folds | positive outer | median lift |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in recurring.to_dict("records"):
            lines.append(
                f"| {row['target']} | {row['features']} | {row['selected_folds']} | "
                f"{row['supported_outer_folds']} | {row['positive_outer_fraction']:.3f} | "
                f"{row['median_outer_target_lift']:.4f} |"
            )
    lines += [
        "",
        "Agent recurrence is adaptive descriptive evidence, not independent validation or production alpha.",
        "Failure of this optional stage does not invalidate R8A.",
        "",
        "KEEP PRODUCTION FROZEN",
        "",
    ]
    return "\n".join(lines)


def run_agent(samples: Path, metadata: Path, r6_dir: Path, atlas_dir: Path, output: Path,
              ledger: Path, cache_dir: Path, *, prior_used_floor: int | None = None,
              run_cap: int = 80, agent_rounds: int = 3, preflight: bool = False):
    frame, observed, _, r6_bound_files, r6_accounted, r6_frozen_sha = load_bound_inputs(samples, metadata, r6_dir)
    atlas_manifest, atlas_complete_path = _load_atlas_anchor(atlas_dir, observed, r6_frozen_sha)
    model = _model_from_env()
    if not model:
        raise RuntimeError("R8B requires RD_AGENT_MODEL or CHAT_MODEL")
    version = _backend_version()
    if agent_rounds != 3:
        raise ValueError("R8B formal protocol freezes agent_rounds at 3")
    floor = _resolve_prior_floor(ledger, r6_accounted, prior_used_floor)
    if not 1 <= run_cap <= TOTAL_PROVIDER_BUDGET-floor:
        raise ValueError("R8B run cap exceeds remaining lower-bound provider budget")

    cfg = replace(AtlasConfig(), agent_rounds=agent_rounds)
    nonoverlap = nonoverlap_w3_frame(frame)
    protocol = {
        "revision": "R8B_COMPACT_OPTIONAL_INTERACTIONS_V2",
        "stage": AGENT_STAGE,
        "config": asdict(cfg),
        "interaction_panel": "nonoverlap_w3",
        "prompt_contract": "R8_COMPACT_INTERACTION_V1",
        "all_features_in_prompt": True,
        "feature_prefilter": False,
        "model": model,
        "prior_used_floor": floor,
        "provider_usage_verified": False,
        "run_attempt_cap": run_cap,
        "cache_identity_includes_model": True,
        "production_change": False,
        "research_mode": "known_history_adaptive_retrospective_optional_agent",
    }
    manifest = {
        **_base_manifest(observed, r6_bound_files, r6_frozen_sha, r6_accounted),
        "nonoverlap_w3_rows": len(nonoverlap),
        "atlas_manifest_sha256": sha256_file(atlas_dir / "input_manifest.json"),
        "atlas_complete_sha256": sha256_file(atlas_complete_path),
        "atlas_protocol_sha256": atlas_manifest["protocol_sha256"],
        "rdagent_version": version,
        "protocol": protocol,
        "protocol_sha256": digest(protocol),
        "source_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("r8_runner.py", "r8_interactions.py", "r8_atlas.py", "r8_agent.py",
                         "r6_runner.py", "r6_stability.py", "dataset.py")
        },
    }
    if preflight:
        attempts = None
        if ledger.exists():
            attempts = int(json.loads(ledger.read_text()).get("attempts_used", 0))
        return {**manifest, "status": "PREFLIGHT_PASS", "existing_r8_attempts": attempts, "rdagent_calls": 0}

    _require_fresh_output(output, [samples, metadata, r6_dir, atlas_dir, ledger, cache_dir])
    immutable_files = [samples.resolve(), metadata.resolve(), *(p.resolve() for p in r6_bound_files),
                       atlas_dir.resolve() / "input_manifest.json", atlas_complete_path.resolve()]
    before = _immutable_hashes(immutable_files)
    output.mkdir(parents=True, exist_ok=True)

    from .r8_agent import R8AgentProposer
    proposer = R8AgentProposer(
        ledger_path=ledger,
        cache_dir=cache_dir,
        model=model,
        prior_used_floor=floor,
        run_cap=run_cap,
    )
    try:
        lock, outer, interaction_stability, recurrence, _ = discover_interactions_compact(
            nonoverlap, manifest["features"], manifest["calendar"], proposer, cfg, output)
        _assert_unchanged(before)
        outer.to_csv(output / "interaction_outer.csv", index=False)
        interaction_stability.to_csv(output / "interaction_stability.csv", index=False)
        recurrence.to_csv(output / "interaction_feature_recurrence.csv", index=False)
        manifest["requests"] = proposer.snapshot()
        manifest["interaction_frozen_sha256"] = digest(lock)
        (output / "input_manifest.json").write_text(canonical(manifest) + "\n", encoding="utf-8")
        (output / "R8_INTERACTION_REPORT.md").write_text(_agent_report(interaction_stability, manifest), encoding="utf-8")
        _assert_unchanged(before)
        _write_complete(output, "AGENT_COMPLETE", {"requests": proposer.snapshot()})
        return manifest
    except Exception:
        (output / "FAILED.json").write_text(
            canonical({"status": "AGENT_FAILED", "requests": proposer.snapshot(), "atlas_invalidated": False}) + "\n",
            encoding="utf-8",
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(ATLAS_STAGE, AGENT_STAGE), required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--r6-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--atlas-dir", type=Path, help="required only for --stage agent")
    parser.add_argument("--ledger", type=Path, help="required only for --stage agent")
    parser.add_argument("--cache-dir", type=Path, help="required only for --stage agent")
    parser.add_argument("--prior-used-floor", type=int, default=None)
    parser.add_argument("--run-call-cap", type=int, default=80)
    parser.add_argument("--agent-rounds", type=int, default=3)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    if args.stage == ATLAS_STAGE:
        result = run_atlas(args.samples, args.metadata, args.r6_dir, args.output_root, preflight=args.preflight)
        status = "PREFLIGHT_PASS" if args.preflight else "ATLAS_COMPLETE"
        print(canonical({"status": status, "rdagent_calls": 0,
                         "report": None if args.preflight else str(args.output_root / "R8_ATLAS_REPORT.md")}))
        return 0

    if args.atlas_dir is None or args.ledger is None or args.cache_dir is None:
        parser.error("--stage agent requires --atlas-dir, --ledger and --cache-dir")
    result = run_agent(
        args.samples,
        args.metadata,
        args.r6_dir,
        args.atlas_dir,
        args.output_root,
        args.ledger,
        args.cache_dir,
        prior_used_floor=args.prior_used_floor,
        run_cap=args.run_call_cap,
        agent_rounds=args.agent_rounds,
        preflight=args.preflight,
    )
    status = "PREFLIGHT_PASS" if args.preflight else "AGENT_COMPLETE"
    print(canonical({"status": status,
                     "requests": None if args.preflight else result["requests"],
                     "report": None if args.preflight else str(args.output_root / "R8_INTERACTION_REPORT.md")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
