from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .config import (
    ROOT,
    TRACK_D_ROOT,
    OUT,
    PANEL_SOURCE,
    PRODUCTION_SKILL_PATH,
    FEATURE_MANIFEST_PATH,
)


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()


def source_hashes() -> dict[str, str]:
    h = hashlib.sha256()
    for p in sorted(TRACK_D_ROOT.rglob("*")):
        if not p.is_file() or OUT in p.parents:
            continue
        if p.suffix not in {".py", ".json", ".md"}:
            continue
        rel = str(p.relative_to(TRACK_D_ROOT))
        h.update(rel.encode("utf-8"))
        h.update(p.read_bytes())

    dependencies = {
        "track_d_package_hash": h.hexdigest(),
        "production_b0_hash": hash_file(PRODUCTION_SKILL_PATH),
        "panel_hash": hash_file(PANEL_SOURCE),
        "feature_manifest_hash": hash_file(FEATURE_MANIFEST_PATH),
        "track_c_protocol_hash": hash_file(ROOT / "backtest" / "track_c_ranking_discovery" / "protocol.py"),
        "track_c_evaluator_hash": hash_file(ROOT / "backtest" / "track_c_ranking_discovery" / "evaluate_econometrics.py"),
        "track_c_b0_grid_hash": hash_file(ROOT / "backtest" / "track_c_ranking_discovery" / "b0_ablation_grid.py"),
        "track_c_discovery_runner_hash": hash_file(
            ROOT / "backtest" / "track_c_ranking_discovery" / "discovery_sandbox" / "discovery_runner.py"
        ),
        "track_d_tests_hash": hash_file(ROOT / "tests" / "test_track_d_mechanism_discovery.py"),
    }
    all_h = hashlib.sha256()
    for k in sorted(dependencies):
        all_h.update(k.encode("utf-8"))
        all_h.update(dependencies[k].encode("utf-8"))
    dependencies["codebase_hash"] = all_h.hexdigest()
    return dependencies


def assert_source_clean() -> None:
    result = subprocess.run(
        [
            "git", "status", "--porcelain", "--",
            "backtest/track_d_mechanism_discovery",
            "backtest/track_c_ranking_discovery",
            "dashboard/skill_industry_eps_known.py",
            "tests/test_track_d_mechanism_discovery.py",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    dirty = []
    for line in result.stdout.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].split(" -> ")[-1].strip()
        if path.startswith("backtest/track_d_mechanism_discovery/output/"):
            continue
        if path.startswith("backtest/track_c_ranking_discovery/output/"):
            continue
        dirty.append(path)
    if dirty:
        raise RuntimeError(
            "TRACK D SOURCE FREEZE VIOLATION: local research source is dirty. "
            "Gemini must not patch source during materialization. Dirty: " + ", ".join(dirty)
        )


def write_or_verify_phase0(manifest_path: Path, snapshots: list[str]) -> dict[str, Any]:
    assert_source_clean()
    deps = source_hashes()
    current_sha = git_sha()

    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("source_git_sha") == current_sha and old.get("dependency_hashes") == deps:
            return old

    import datetime as dt

    run_id = (
        "track_d_"
        + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + deps["codebase_hash"][:8]
    )
    manifest = {
        "protocol_version": "track_d_v1_focused",
        "run_id": run_id,
        "source_git_sha": current_sha,
        "snapshot_count": len(snapshots),
        "first_snapshot": snapshots[0] if snapshots else None,
        "last_snapshot": snapshots[-1] if snapshots else None,
        "dependency_hashes": deps,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_phase0(manifest_path: Path) -> dict[str, Any]:
    assert_source_clean()
    if not manifest_path.exists():
        raise RuntimeError("Track D Phase 0 manifest missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_git_sha") != git_sha():
        raise RuntimeError("Track D source git SHA changed after Phase 0.")
    if manifest.get("dependency_hashes") != source_hashes():
        raise RuntimeError("Track D source/data dependency hash changed after Phase 0.")
    return manifest
