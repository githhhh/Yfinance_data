from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .config import (
    ROOT,
    TRACK_C_ROOT,
    PRODUCTION_SKILL_PATH,
    PANEL_SOURCE,
    FEATURE_MANIFEST_PATH,
    TRAIN_END,
    CONTAM_VAL_START,
    CONTAM_VAL_END,
    RANDOM_SEED,
    BOOTSTRAP_ROUNDS,
    OUT,
)


def compute_hash_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _source_dirty_paths() -> list[str]:
    """Return dirty Track C/production source paths, excluding generated output artifacts."""
    try:
        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                "backtest/track_c_ranking_discovery",
                "dashboard/skill_industry_eps_known.py",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:
        raise RuntimeError("Unable to inspect git source cleanliness for Track C") from exc

    dirty: list[str] = []
    for line in result.stdout.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].split(" -> ")[-1].strip()
        if path.startswith("backtest/track_c_ranking_discovery/output/"):
            continue
        if path.startswith("backtest/track_c_ranking_discovery/discovery_sandbox/proposals/"):
            continue
        if path.endswith(".py") or path.endswith(".json") or path == "dashboard/skill_industry_eps_known.py":
            dirty.append(path)
    return dirty


def assert_track_c_source_clean() -> None:
    dirty = _source_dirty_paths()
    if dirty:
        raise RuntimeError(
            "TRACK C SOURCE FREEZE VIOLATION: research source changed after code handoff. "
            "Do not patch locally during materialization. Dirty paths: " + ", ".join(dirty)
        )


def compute_track_c_dependency_hashes() -> dict[str, str]:
    """Hash every source dependency that is allowed to affect a sealed Track C run."""
    pkg_files = sorted(
        list(TRACK_C_ROOT.glob("*.py"))
        + list(TRACK_C_ROOT.glob("*.json"))
        + list((TRACK_C_ROOT / "discovery_sandbox").glob("*.py"))
    )
    h_pkg = hashlib.sha256()
    for p in pkg_files:
        h_pkg.update(str(p.relative_to(TRACK_C_ROOT)).encode("utf-8"))
        h_pkg.update(p.read_bytes())
    pkg_hash = h_pkg.hexdigest()

    explicit = {
        "feature_manifest_hash": FEATURE_MANIFEST_PATH,
        "protocol_hash": TRACK_C_ROOT / "protocol.py",
        "b0_ablation_grid_hash": TRACK_C_ROOT / "b0_ablation_grid.py",
        "counterfactual_engine_hash": TRACK_C_ROOT / "counterfactual_engine.py",
        "evaluate_econometrics_hash": TRACK_C_ROOT / "evaluate_econometrics.py",
        "discovery_runner_hash": TRACK_C_ROOT / "discovery_sandbox" / "discovery_runner.py",
        "rdagent_policy_bridge_hash": TRACK_C_ROOT / "discovery_sandbox" / "rdagent_policy_bridge.py",
        "blind_prompt_hash": TRACK_C_ROOT / "discovery_sandbox" / "blind_prompt.py",
        "behavioral_dedup_hash": TRACK_C_ROOT / "discovery_sandbox" / "behavioral_dedup.py",
        "production_b0_hash": PRODUCTION_SKILL_PATH,
        "panel_hash": PANEL_SOURCE,
    }

    result = {
        "challenge_package_hash": pkg_hash,
        **{name: compute_hash_of_file(path) for name, path in explicit.items()},
    }
    h_all = hashlib.sha256()
    for key in sorted(result):
        h_all.update(key.encode("utf-8"))
        h_all.update(result[key].encode("utf-8"))
    result["codebase_hash"] = h_all.hexdigest()
    return result


def verify_phase0_integrity(manifest_path: Path) -> dict[str, Any]:
    assert_track_c_source_clean()
    if not manifest_path.exists():
        raise RuntimeError(f"Phase 0 manifest missing at {manifest_path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored = data.get("dependency_hashes")
    if not isinstance(stored, dict) or not stored:
        raise RuntimeError("Phase 0 manifest has no sealed dependency hashes")

    current = compute_track_c_dependency_hashes()
    if stored != current:
        mismatches = [
            k for k in sorted(set(stored) | set(current))
            if stored.get(k) != current.get(k)
        ]
        raise RuntimeError("Phase 0 dependency mismatch: " + ", ".join(mismatches))

    source_sha = str(data.get("source_git_sha") or "")
    if not source_sha or source_sha != get_git_sha():
        raise RuntimeError(
            f"Phase 0 source git SHA mismatch: sealed={source_sha}, current={get_git_sha()}"
        )
    if not data.get("run_id"):
        raise RuntimeError("Phase 0 manifest is missing run_id")
    return data


def verify_proposal_freeze_integrity(
    freeze_manifest_path: Path,
    current_proposals: list[Any],
    ledger_path: Path | None = None,
    phase0_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Verify the exact frozen specs and their model-response provenance."""
    if not freeze_manifest_path.exists():
        raise RuntimeError(f"Proposal freeze manifest missing at {freeze_manifest_path}")
    data = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))

    if phase0_manifest_path is not None:
        phase0 = verify_phase0_integrity(phase0_manifest_path)
        expected = str(data.get("phase0_manifest_hash") or "")
        actual = compute_hash_of_file(phase0_manifest_path)
        if expected != actual:
            raise RuntimeError("Proposal freeze was not created from the current Phase 0 manifest")
        if data.get("run_id") != phase0.get("run_id"):
            raise RuntimeError("Proposal freeze run_id does not match Phase 0 run_id")

    if ledger_path is not None:
        if not ledger_path.exists():
            raise RuntimeError("Proposal ledger missing")
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if canonical_json_hash(ledger) != data.get("ledger_hash"):
            raise RuntimeError("Proposal ledger canonical hash differs from freeze manifest")
        if compute_hash_of_file(ledger_path) != data.get("ledger_file_hash"):
            raise RuntimeError("Proposal ledger file hash differs from freeze manifest")

    frozen = data.get("proposals")
    if not isinstance(frozen, list) or not frozen:
        raise RuntimeError("Proposal freeze contains no proposals")

    frozen_spec_map = {p["policy_id"]: p["spec_hash"] for p in frozen}
    current_spec_map = {p.policy_id: p.spec_hash for p in current_proposals}
    if frozen_spec_map != current_spec_map:
        raise RuntimeError("Frozen proposal set/spec hashes do not match executable frozen specs")

    for record in frozen:
        rel = str(record.get("source_response_path") or "")
        expected_hash = str(record.get("source_response_hash") or "")
        if not rel or not expected_hash:
            raise RuntimeError(f"Frozen proposal {record['policy_id']} lacks RD-Agent raw-response provenance")
        raw_path = OUT / rel
        if not raw_path.exists() or compute_hash_of_file(raw_path) != expected_hash:
            raise RuntimeError(f"RD-Agent raw response hash mismatch for {record['policy_id']}")

    provenance_rel = str(data.get("rdagent_provenance_path") or "")
    provenance_hash = str(data.get("rdagent_provenance_hash") or "")
    if not provenance_rel or not provenance_hash:
        raise RuntimeError("Proposal freeze lacks RD-Agent provenance hash")
    provenance_path = OUT / provenance_rel
    if not provenance_path.exists() or compute_hash_of_file(provenance_path) != provenance_hash:
        raise RuntimeError("RD-Agent policy-discovery provenance hash mismatch")
    return data


def _locked_artifact_paths() -> dict[str, Path]:
    return {
        "phase0_manifest": OUT / "phase0_prepared_manifest.json",
        "proposal_freeze_manifest": OUT / "proposal_freeze_manifest.json",
        "proposals_ledger": OUT / "proposals_ledger.json",
        "counterfactual_decomposition": OUT / "counterfactual_2x2_decomposition.csv",
        "train_evaluations": OUT / "train_evaluations.parquet",
        "shortlist_summary": OUT / "shortlist_summary.json",
    }


def seal_track_c_lock_manifest(
    locked_challengers_summary: list[dict[str, Any]],
    manifest_path: Path,
) -> dict[str, Any]:
    """Seal source, proposals, Train artifacts and shortlist before observed re-validation."""
    assert_track_c_source_clean()
    phase0_path = OUT / "phase0_prepared_manifest.json"
    freeze_path = OUT / "proposal_freeze_manifest.json"
    ledger_path = OUT / "proposals_ledger.json"

    phase0 = verify_phase0_integrity(phase0_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))

    from .discovery_sandbox.discovery_runner import instantiate_discovery_proposals

    frozen_policies = instantiate_discovery_proposals(freeze["proposals"])
    verify_proposal_freeze_integrity(
        freeze_path,
        frozen_policies,
        ledger_path=ledger_path,
        phase0_manifest_path=phase0_path,
    )

    if get_git_sha() != phase0["source_git_sha"]:
        raise RuntimeError("Cannot lock: source HEAD changed after Phase 0")

    artifact_hashes: dict[str, str] = {}
    for name, path in _locked_artifact_paths().items():
        if not path.exists():
            raise RuntimeError(f"Cannot lock: required artifact missing: {path}")
        artifact_hashes[name] = compute_hash_of_file(path)

    manifest = {
        "protocol_version": "track_c_v1",
        "run_id": phase0["run_id"],
        "source_git_sha": phase0["source_git_sha"],
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT), text=True).strip()
        ),
        "code_dirty": False,
        "dependency_hashes": compute_track_c_dependency_hashes(),
        "artifact_hashes": artifact_hashes,
        "panel_hash": compute_hash_of_file(PANEL_SOURCE),
        "train_end": TRAIN_END,
        "validation_window": f"{CONTAM_VAL_START} .. {CONTAM_VAL_END}",
        "random_seed": RANDOM_SEED,
        "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        "proposal_ledger_hash": freeze["ledger_hash"],
        "rdagent_provenance_hash": freeze["rdagent_provenance_hash"],
        "locked_challengers": locked_challengers_summary,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_lock_integrity(manifest_path: Path) -> dict[str, Any]:
    """Fail closed before observed validation if any source or locked Train artifact changed."""
    assert_track_c_source_clean()
    if not manifest_path.exists():
        raise RuntimeError(f"Research lock manifest missing at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_git_sha") != get_git_sha():
        raise RuntimeError(
            f"Lock source SHA mismatch: sealed={manifest.get('source_git_sha')}, current={get_git_sha()}"
        )

    current_deps = compute_track_c_dependency_hashes()
    if manifest.get("dependency_hashes") != current_deps:
        raise RuntimeError("Lock dependency hashes differ from current source/data dependencies")

    for name, path in _locked_artifact_paths().items():
        expected = manifest.get("artifact_hashes", {}).get(name)
        if not expected or not path.exists() or compute_hash_of_file(path) != expected:
            raise RuntimeError(f"Locked artifact changed after seal: {name}")

    freeze_path = OUT / "proposal_freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    from .discovery_sandbox.discovery_runner import instantiate_discovery_proposals

    policies = instantiate_discovery_proposals(freeze["proposals"])
    verify_proposal_freeze_integrity(
        freeze_path,
        policies,
        ledger_path=OUT / "proposals_ledger.json",
        phase0_manifest_path=OUT / "phase0_prepared_manifest.json",
    )
    return manifest
