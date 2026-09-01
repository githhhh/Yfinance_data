from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any
from .config import (
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
    """Compute SHA256 of a single file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_track_c_dependency_hashes() -> dict[str, str]:
    """Compute combined SHA256 hash for Track C package and production dependencies."""
    # 1. Package source files
    pkg_files = sorted(
        list(TRACK_C_ROOT.glob("*.py")) +
        list(TRACK_C_ROOT.glob("*.json")) +
        list((TRACK_C_ROOT / "discovery_sandbox").glob("*.py"))
    )
    h_pkg = hashlib.sha256()
    for p in pkg_files:
        h_pkg.update(p.name.encode())
        with open(p, "rb") as f:
            h_pkg.update(f.read())
    pkg_hash = h_pkg.hexdigest()

    # 2. Key individual module hashes
    feat_man_hash = compute_hash_of_file(FEATURE_MANIFEST_PATH)
    protocol_hash = compute_hash_of_file(TRACK_C_ROOT / "protocol.py")
    b0_ablation_hash = compute_hash_of_file(TRACK_C_ROOT / "b0_ablation_grid.py")
    cf_engine_hash = compute_hash_of_file(TRACK_C_ROOT / "counterfactual_engine.py")
    eval_econ_hash = compute_hash_of_file(TRACK_C_ROOT / "evaluate_econometrics.py")
    disc_runner_hash = compute_hash_of_file(TRACK_C_ROOT / "discovery_sandbox" / "discovery_runner.py")
    b0_hash = compute_hash_of_file(PRODUCTION_SKILL_PATH)
    panel_hash = compute_hash_of_file(PANEL_SOURCE)

    # 3. Combined codebase hash
    h_all = hashlib.sha256()
    h_all.update(pkg_hash.encode())
    h_all.update(b0_hash.encode())

    return {
        "codebase_hash": h_all.hexdigest(),
        "challenge_package_hash": pkg_hash,
        "feature_manifest_hash": feat_man_hash,
        "protocol_hash": protocol_hash,
        "b0_ablation_grid_hash": b0_ablation_hash,
        "counterfactual_engine_hash": cf_engine_hash,
        "evaluate_econometrics_hash": eval_econ_hash,
        "discovery_runner_hash": disc_runner_hash,
        "production_b0_hash": b0_hash,
        "panel_hash": panel_hash,
    }


def verify_phase0_integrity(manifest_path: Path) -> None:
    """Verify Phase 0 manifest hashes match current environment before evaluation."""
    if not manifest_path.exists():
        raise RuntimeError(f"Phase 0 manifest missing at {manifest_path}!")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cur_hashes = compute_track_c_dependency_hashes()
    stored_hashes = data.get("dependency_hashes", {})
    for k, v in stored_hashes.items():
        if cur_hashes.get(k) != v:
            raise RuntimeError(f"Phase 0 integrity mismatch for {k}: stored {v[:16]} != current {cur_hashes.get(k, '')[:16]}")


def verify_proposal_freeze_integrity(
    freeze_manifest_path: Path,
    current_proposals: list[Any],
) -> None:
    """Verify sealed proposal freeze manifest matches currently instantiated proposals."""
    if not freeze_manifest_path.exists():
        raise RuntimeError(f"Proposal freeze manifest missing at {freeze_manifest_path}!")
    with open(freeze_manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frozen_spec_map = {p["policy_id"]: p["spec_hash"] for p in data["proposals"]}
    curr_spec_map = {p.policy_id: p.spec_hash for p in current_proposals}

    for p_id, spec_h in frozen_spec_map.items():
        if p_id not in curr_spec_map:
            raise RuntimeError(f"Frozen proposal {p_id} missing from current proposal generator!")
        if curr_spec_map[p_id] != spec_h:
            raise RuntimeError(f"Proposal {p_id} spec hash altered after freeze! Frozen: {spec_h[:16]}, Current: {curr_spec_map[p_id][:16]}")


def verify_lock_integrity(manifest_path: Path) -> None:
    """Verify Phase 4 lock manifest hashes match current environment before validation."""
    if not manifest_path.exists():
        raise RuntimeError(f"Research lock manifest missing at {manifest_path}!")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    cur_hashes = compute_track_c_dependency_hashes()
    stored_hashes = manifest.get("dependency_hashes", {})

    for k in ("codebase_hash", "challenge_package_hash", "production_b0_hash"):
        if stored_hashes.get(k) != cur_hashes.get(k):
            raise RuntimeError(
                f"Lock integrity mismatch for {k}! Sealed: {stored_hashes.get(k, '')[:16]}..., Current: {cur_hashes.get(k, '')[:16]}..."
            )


def seal_track_c_lock_manifest(
    locked_challengers_summary: list[dict[str, Any]],
    manifest_path: Path,
) -> dict[str, Any]:
    """Create sealed research lock manifest before validation."""
    dep_hashes = compute_track_c_dependency_hashes()
    panel_hash = compute_hash_of_file(PANEL_SOURCE)

    # Git metadata
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        code_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--", "*.py", ":!backtest/track_c_ranking_discovery/output/*"],
            text=True
        ).strip())
    except Exception:
        git_sha = "unknown"
        git_dirty = True
        code_dirty = False

    if code_dirty:
        raise RuntimeError("Cannot seal Track C manifest with uncommitted Python code changes!")

    manifest = {
        "protocol_version": "track_c_v1",
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "code_dirty": code_dirty,
        "dependency_hashes": dep_hashes,
        "panel_hash": panel_hash,
        "train_end": TRAIN_END,
        "validation_window": f"{CONTAM_VAL_START} .. {CONTAM_VAL_END}",
        "random_seed": RANDOM_SEED,
        "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        "locked_challengers": locked_challengers_summary,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
