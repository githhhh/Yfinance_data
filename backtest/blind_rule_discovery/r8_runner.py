"""Run R8 Winner/Stop atlas on the exact R4 sample population bound by completed R6."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

from .pipeline_contract import sha256_file
from .r6_runner import load_inputs
from .r6_stability import canonical, digest
from .r8_agent import R8AgentProposer
from .r8_atlas import (
    AtlasConfig,
    aggregate_feature_stability,
    class_profiles,
    discover_interactions,
    quarter_feature_contrasts,
    quintile_surfaces,
    render_report,
)


def load_bound_inputs(samples: Path, metadata: Path, r6_dir: Path):
    manifest_path = r6_dir / "input_manifest.json"
    if not manifest_path.exists():
        raise ValueError("completed R6 input_manifest.json is required")
    anchor = json.loads(manifest_path.read_text())
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
    return frame, observed, anchor, manifest_path, accounted


def _model_from_env() -> str:
    return os.environ.get("RD_AGENT_MODEL") or os.environ.get("CHAT_MODEL") or ""


def run(samples: Path, metadata: Path, r6_dir: Path, output: Path,
        ledger: Path, cache_dir: Path, *, prior_used_floor: int | None = None,
        run_cap: int = 80, agent_rounds: int = 3, preflight: bool = False):
    frame, observed, r6_anchor, r6_manifest_path, r6_accounted = load_bound_inputs(samples, metadata, r6_dir)
    model = _model_from_env()
    if not model:
        raise RuntimeError("R8 requires RD_AGENT_MODEL or CHAT_MODEL in the existing environment")
    try:
        version = importlib.metadata.version("rdagent")
        from rdagent.oai.backend.litellm import LiteLLMAPIBackend
        if not callable(LiteLLMAPIBackend.build_messages_and_create_chat_completion):
            raise AttributeError("missing RD-Agent chat backend")
    except (ImportError, importlib.metadata.PackageNotFoundError, AttributeError) as exc:
        raise RuntimeError("Compatible official RD-Agent backend is required in quant_env") from exc

    floor = max(r6_accounted, int(prior_used_floor or 0))
    cfg = replace(AtlasConfig(), agent_rounds=agent_rounds)
    if not 1 <= run_cap <= 1000-floor:
        raise ValueError("R8 run cap exceeds remaining lower-bound provider budget")
    protocol = {
        "revision": "R8_WINNER_STOP_STABLE_ATLAS_V1",
        "config": asdict(cfg),
        "primary_contrast": ["fast_winner_3w", "stop_first_3w"],
        "context_classes": ["unresolved_3w", "ambiguous_3w"],
        "model": model,
        "prior_used_floor": floor,
        "provider_usage_verified": False,
        "run_attempt_cap": run_cap,
        "cache_identity_includes_model": True,
        "production_change": False,
        "research_mode": "known_history_retrospective_with_past_only_outer_agent",
    }
    manifest = {
        **{k: observed[k] for k in ("samples_sha256", "metadata_sha256", "source_replay_dataset_sha256",
                                    "sample_rows", "snapshot_weeks", "unique_tickers", "calendar", "features")},
        "r6_anchor_manifest_sha256": sha256_file(r6_manifest_path),
        "r6_accounted_total_floor": r6_accounted,
        "rdagent_version": version,
        "protocol": protocol,
        "protocol_sha256": digest(protocol),
        "execution_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_replay_sha_independently_recomputed": False,
        "population": observed["population"],
        "execution_features_excluded": True,
        "source_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("r8_runner.py", "r8_atlas.py", "r8_agent.py", "r6_runner.py", "r6_stability.py", "dataset.py")
        },
    }
    if preflight:
        return {**manifest, "status": "PREFLIGHT_PASS", "rdagent_calls": 0}

    if output.exists() and any(output.iterdir()):
        raise ValueError("R8 requires a fresh output directory")
    out = output.resolve()
    protected = [samples.resolve(), metadata.resolve(), r6_manifest_path.resolve(), ledger.resolve(), cache_dir.resolve()]
    for path in protected:
        if out == path or out in path.parents or path in out.parents:
            raise ValueError("R8 output must be disjoint from inputs, ledger and cache")
    before = {str(p): sha256_file(p) for p in (samples.resolve(), metadata.resolve(), r6_manifest_path.resolve())}

    contrasts = quarter_feature_contrasts(frame, manifest["features"], manifest["calendar"], cfg)
    stability = aggregate_feature_stability(contrasts, cfg)
    profiles = class_profiles(frame, manifest["features"], manifest["calendar"])
    surfaces = quintile_surfaces(frame, manifest["features"], manifest["calendar"], cfg)

    output.mkdir(parents=True, exist_ok=True)
    proposer = R8AgentProposer(ledger_path=ledger, cache_dir=cache_dir, model=model,
                               prior_used_floor=floor, run_cap=run_cap)
    try:
        lock, outer, interaction_stability, recurrence, _ = discover_interactions(
            frame, manifest["features"], manifest["calendar"], proposer, cfg, output)
        for path in (samples.resolve(), metadata.resolve(), r6_manifest_path.resolve()):
            if sha256_file(path) != before[str(path)]:
                raise RuntimeError("R8 protected input changed during execution")

        profiles.to_csv(output / "class_profiles.csv", index=False)
        contrasts.to_csv(output / "quarter_feature_contrasts.csv", index=False)
        stability.to_csv(output / "feature_stability.csv", index=False)
        surfaces.to_csv(output / "quintile_surfaces.csv", index=False)
        outer.to_csv(output / "interaction_outer.csv", index=False)
        interaction_stability.to_csv(output / "interaction_stability.csv", index=False)
        recurrence.to_csv(output / "interaction_feature_recurrence.csv", index=False)
        manifest["requests"] = proposer.snapshot()
        manifest["interaction_frozen_sha256"] = digest(lock)
        (output / "input_manifest.json").write_text(canonical(manifest) + "\n")
        (output / "R8_REPORT.md").write_text(render_report(stability, outer, interaction_stability, manifest))

        for path in (samples.resolve(), metadata.resolve(), r6_manifest_path.resolve()):
            if sha256_file(path) != before[str(path)]:
                raise RuntimeError("R8 protected input changed before publication")
        outputs = {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}
        (output / "COMPLETE.json").write_text(canonical({
            "status": "COMPLETE",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_sha256": outputs,
            "inputs_unchanged": True,
            "production_change": False,
            "requests": proposer.snapshot(),
        }) + "\n")
        return manifest
    except Exception:
        (output / "FAILED.json").write_text(canonical({"status": "FAILED", "requests": proposer.snapshot()}) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--r6-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True, help="R8-only persistent request ledger; never reset after a failed run")
    parser.add_argument("--cache-dir", type=Path, required=True, help="R8-only model-aware response cache")
    parser.add_argument("--prior-used-floor", type=int, default=None,
                        help="optional provider usage floor; completed R6 accounted_total is always the minimum")
    parser.add_argument("--run-call-cap", type=int, default=80)
    parser.add_argument("--agent-rounds", type=int, default=3)
    parser.add_argument("--preflight", action="store_true", help="validate only; no output/ledger/cache write and no model call")
    args = parser.parse_args()
    result = run(args.samples, args.metadata, args.r6_dir, args.output_root, args.ledger, args.cache_dir,
                 prior_used_floor=args.prior_used_floor, run_cap=args.run_call_cap,
                 agent_rounds=args.agent_rounds, preflight=args.preflight)
    if args.preflight:
        print(canonical(result))
    else:
        print(canonical({"status": "COMPLETE", "report": str(args.output_root / "R8_REPORT.md"),
                         "requests": result["requests"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
