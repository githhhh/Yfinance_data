from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import OUT, RAW_LLM_DIR
from .integrity import canonical_hash, hash_file, verify_phase0


def hash_tree(root: Path) -> str:
    h=hashlib.sha256()
    if not root.exists():
        return h.hexdigest()
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        h.update(str(p.relative_to(root)).encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def seal_policy_freeze(
    phase0_path:Path,
    policies:list[dict[str,Any]],
    research_ledger_path:Path,
    synthesis_path:Path,
    request_ledger_path:Path,
    mechanism_path:Path,
    failure_summary_path:Path,
    split_path:Path,
    manifest_path:Path,
) -> dict[str,Any]:
    phase0=verify_phase0(phase0_path)
    required=[
        research_ledger_path,synthesis_path,request_ledger_path,mechanism_path,failure_summary_path,split_path
    ]
    for p in required:
        if not p.exists():
            raise RuntimeError(f"Track D freeze artifact missing: {p}")

    manifest={
        "protocol_version":"track_d_v1_focused",
        "run_id":phase0["run_id"],
        "source_git_sha":phase0["source_git_sha"],
        "phase0_hash":hash_file(phase0_path),
        "policy_count":len(policies),
        "policy_set_hash":canonical_hash(policies),
        "policies":policies,
        "research_ledger_hash":hash_file(research_ledger_path),
        "policy_synthesis_hash":hash_file(synthesis_path),
        "request_ledger_hash":hash_file(request_ledger_path),
        "mechanism_results_hash":hash_file(mechanism_path),
        "failure_summary_hash":hash_file(failure_summary_path),
        "locked_split_hash":hash_file(split_path),
        "raw_llm_tree_hash":hash_tree(RAW_LLM_DIR),
    }
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    return manifest


def verify_policy_freeze(
    phase0_path:Path,
    manifest_path:Path,
    research_ledger_path:Path,
    synthesis_path:Path,
    request_ledger_path:Path,
    mechanism_path:Path,
    failure_summary_path:Path,
    split_path:Path,
) -> dict[str,Any]:
    phase0=verify_phase0(phase0_path)
    if not manifest_path.exists():
        raise RuntimeError("Track D policy freeze manifest missing")
    m=json.loads(manifest_path.read_text(encoding="utf-8"))
    if m.get("run_id")!=phase0.get("run_id") or m.get("source_git_sha")!=phase0.get("source_git_sha"):
        raise RuntimeError("Track D policy freeze does not belong to current Phase 0 source")
    # Only artifacts that can change executable policy are re-verified here.
    # request_ledger/raw_llm hashes are preserved as pre-freeze provenance, but
    # post-evaluation interpretation calls are allowed to append new request/raw
    # records without invalidating the already frozen policy set.
    checks={
        "phase0_hash":hash_file(phase0_path),
        "research_ledger_hash":hash_file(research_ledger_path),
        "policy_synthesis_hash":hash_file(synthesis_path),
        "mechanism_results_hash":hash_file(mechanism_path),
        "failure_summary_hash":hash_file(failure_summary_path),
        "locked_split_hash":hash_file(split_path),
    }
    bad=[k for k,v in checks.items() if m.get(k)!=v]
    if bad:
        raise RuntimeError("Track D policy freeze integrity mismatch: "+", ".join(bad))
    if m.get("policy_set_hash")!=canonical_hash(m.get("policies",[])):
        raise RuntimeError("Track D frozen policy set hash mismatch")
    return m


def seal_final_lock(
    phase0_path:Path,
    policy_freeze_path:Path,
    result_paths:dict[str,Path],
    final_path:Path,
) -> dict[str,Any]:
    phase0=verify_phase0(phase0_path)
    artifacts={}
    for name,path in result_paths.items():
        if not path.exists():
            raise RuntimeError(f"Track D final artifact missing: {name} -> {path}")
        artifacts[name]=hash_file(path)
    manifest={
        "protocol_version":"track_d_v1_focused",
        "run_id":phase0["run_id"],
        "source_git_sha":phase0["source_git_sha"],
        "phase0_hash":hash_file(phase0_path),
        "policy_freeze_hash":hash_file(policy_freeze_path),
        "artifacts":artifacts,
        "final_request_ledger_hash": hash_file(OUT / "request_budget_ledger.json") if (OUT / "request_budget_ledger.json").exists() else None,
        "final_raw_llm_tree_hash": hash_tree(RAW_LLM_DIR),
    }
    final_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest
