"""Agent isolation, rule validation/freeze, and one-shot holdout evaluation."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .dataset import assert_agent_surface_is_blind

MAX_RESEARCH_SECONDS = 3600
MAX_RULE_CLAUSES = 3
MAX_RULE_CONDITIONS = 6
RULE_OPERATORS = {">", ">=", "<", "<=", "==", "!="}

def write_agent_workspace(
    agent_df: pd.DataFrame,
    output_root: Path,
    *,
    period_summary: pd.DataFrame | None = None,
    feature_profile: pd.DataFrame | None = None,
) -> Path:
    """Write only blind discovery artifacts; no private map/holdout is present."""
    workspace = output_root / "agent_workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    assert_agent_surface_is_blind(agent_df.columns)
    agent_df.to_csv(workspace / "samples.csv", index=False)
    if period_summary is not None and not period_summary.empty:
        period_summary.to_csv(workspace / "period_summary.csv", index=False)
    if feature_profile is not None and not feature_profile.empty:
        feature_profile.to_csv(workspace / "feature_profile.csv", index=False)
    prompt = """You are given discovery-only time-indexed samples.
X### are anonymous stock features. M_* are broad-market features known at the signal date. Y_* are outcomes and MUST NEVER appear in the executable rule.
Primary objective: distinguish Y_primary=winner from Y_primary=loser. A stop_out_then_winner is a loser because the real trade stopped out first. Keep unresolved and ambiguous samples separate; do not coerce them into losers.
Use monthly/quarterly distributions and repeated-period evidence. Do not optimize a global average and do not infer X feature semantics.
The executable rule must be compact DNF: top-level clauses are OR; conditions inside each clause are AND. Use at most 3 clauses and 6 total conditions.
Only X### and M_* may be referenced by executable conditions. Allowed operators: >, >=, <, <=, ==, !=. Thresholds must be numeric.
Write rule.json with version=1 and clauses=[{\"all\":[{\"feature\":\"X001\",\"op\":\">=\",\"threshold\":1.0}]}]. Optional rationale/evidence fields may be added, but no executable code or free-form expression is allowed.
"""
    (workspace / "prompt.md").write_text(prompt, encoding="utf-8")
    return workspace

def run_research_command(
    command: list[str],
    workspace: Path,
    *,
    sandbox_prefix: list[str] | None,
    timeout_seconds: int = MAX_RESEARCH_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run only through a caller-provided OS/container sandbox; fail closed otherwise.

    The sandbox prefix may contain {workspace}, expanded to the isolated copy.
    The wrapper must restrict filesystem visibility to that workspace plus the
    runtime required by the agent. Canonical runs never execute an unsandboxed
    research command.
    """
    if not sandbox_prefix:
        raise RuntimeError("physical sandbox wrapper is required for blind discovery")
    known_wrappers = {"sandbox-exec", "bwrap", "docker", "podman", "firejail", "nsjail"}
    wrapper_name = Path(sandbox_prefix[0]).name
    if wrapper_name not in known_wrappers:
        raise RuntimeError(f"unsupported sandbox wrapper: {wrapper_name}")
    if not any("{workspace}" in part for part in sandbox_prefix):
        raise RuntimeError("sandbox prefix must explicitly reference {workspace}")
    effective_timeout = min(max(1, int(timeout_seconds)), MAX_RESEARCH_SECONDS)
    allowed_workspace_files = {"samples.csv", "period_summary.csv", "feature_profile.csv", "prompt.md"}
    extras = {p.name for p in workspace.iterdir() if p.is_file()} - allowed_workspace_files
    if extras:
        raise RuntimeError(f"agent workspace contains unexpected files: {sorted(extras)}")
    with tempfile.TemporaryDirectory(prefix="blind_rule_agent_") as tmp:
        isolated = Path(tmp) / "workspace"
        shutil.copytree(workspace, isolated)
        home = Path(tmp) / "home"
        home.mkdir()
        env = os.environ.copy()
        env.update({"HOME": str(home), "PYTHONNOUSERSITE": "1"})
        for name in ("PWD", "OLDPWD", "PYTHONPATH"):
            env.pop(name, None)
        wrapper = [part.replace("{workspace}", str(isolated)) for part in sandbox_prefix]

        # Empirically verify the same wrapper cannot read outside the agent workspace.
        # This catches a syntactically valid but ineffective sandbox policy before
        # the research process gets a chance to inspect B0/repository files.
        sentinel = Path(tmp) / "forbidden_sentinel.txt"
        sentinel.write_text("blind-discovery-private", encoding="utf-8")
        repo_root = Path(__file__).resolve().parents[2]
        forbidden = [str(sentinel)]
        if repo_root.exists():
            forbidden.append(str(repo_root))
        probe_script = 'for p in "$@"; do if [ -r "$p" ]; then exit 97; fi; done; exit 0'
        probe = subprocess.run(
            [*wrapper, "/bin/sh", "-c", probe_script, "sandbox-probe", *forbidden],
            cwd=isolated,
            text=True,
            capture_output=True,
            timeout=min(effective_timeout, 30),
            check=False,
            env=env,
        )
        if probe.returncode != 0:
            raise RuntimeError(
                "sandbox isolation preflight failed; wrapper can read outside workspace "
                f"or cannot execute the isolation probe (returncode={probe.returncode})"
            )

        completed = subprocess.run(
            [*wrapper, *command],
            cwd=isolated,
            text=True,
            capture_output=True,
            timeout=effective_timeout,
            check=False,
            env=env,
        )
        for name in ("rule.json", "research_notes.md"):
            produced = isolated / name
            if produced.exists():
                shutil.copy2(produced, workspace / name)
        return completed

def _condition_mask(df: pd.DataFrame, condition: Mapping[str, Any]) -> pd.Series:
    feature = str(condition["feature"])
    op = str(condition["op"])
    threshold = float(condition["threshold"])
    values = pd.to_numeric(df[feature], errors="coerce")
    if op == ">":
        return values > threshold
    if op == ">=":
        return values >= threshold
    if op == "<":
        return values < threshold
    if op == "<=":
        return values <= threshold
    if op == "==":
        return values == threshold
    if op == "!=":
        return values != threshold
    raise ValueError(f"unsupported operator: {op}")

def validate_rule_artifact(rule_path: Path, agent_columns: Iterable[str]) -> dict[str, Any]:
    """Validate the machine-executable subset before freezing or holdout access."""
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    if not isinstance(rule, dict) or rule.get("version") != 1:
        raise ValueError("rule.json requires version=1")
    clauses = rule.get("clauses")
    if not isinstance(clauses, list) or not 1 <= len(clauses) <= MAX_RULE_CLAUSES:
        raise ValueError(f"rule clauses must contain 1..{MAX_RULE_CLAUSES} items")
    allowed = {c for c in agent_columns if c.startswith("X") or c.startswith("M_")}
    total_conditions = 0
    for clause in clauses:
        if not isinstance(clause, dict) or set(clause) != {"all"} or not isinstance(clause["all"], list):
            raise ValueError("each clause must be exactly {'all': [...]} ")
        if not clause["all"]:
            raise ValueError("empty rule clause")
        for condition in clause["all"]:
            total_conditions += 1
            if total_conditions > MAX_RULE_CONDITIONS:
                raise ValueError(f"rule exceeds {MAX_RULE_CONDITIONS} total conditions")
            if not isinstance(condition, dict) or set(condition) != {"feature", "op", "threshold"}:
                raise ValueError("condition must contain only feature/op/threshold")
            feature = str(condition["feature"])
            if feature not in allowed:
                raise ValueError(f"rule references forbidden or unknown feature: {feature}")
            if str(condition["op"]) not in RULE_OPERATORS:
                raise ValueError(f"unsupported rule operator: {condition['op']}")
            threshold = condition["threshold"]
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
                raise ValueError("rule threshold must be a finite number")
    return rule

def apply_rule(rule: Mapping[str, Any], df: pd.DataFrame) -> pd.Series:
    selected = pd.Series(False, index=df.index)
    for clause in rule["clauses"]:
        clause_mask = pd.Series(True, index=df.index)
        for condition in clause["all"]:
            clause_mask &= _condition_mask(df, condition).fillna(False)
        selected |= clause_mask
    return selected

def validate_rule_support(
    rule: Mapping[str, Any],
    discovery_df: pd.DataFrame,
    *,
    min_selected: int = 20,
    min_active_quarters: int = 3,
) -> dict[str, int]:
    """Reject tiny, period-specific rules before freeze; this is support-only, not a performance gate."""
    selected = discovery_df.loc[apply_rule(rule, discovery_df)]
    active_quarters = int(selected["period_quarter"].nunique()) if not selected.empty else 0
    resolved_selected = int(selected["Y_primary"].isin(["winner", "loser"]).sum()) if not selected.empty else 0
    required_quarters = min(min_active_quarters, int(discovery_df["period_quarter"].nunique()))
    if len(selected) < min_selected:
        raise ValueError(f"rule selects only {len(selected)} discovery samples; minimum is {min_selected}")
    if active_quarters < required_quarters:
        raise ValueError(f"rule is active in only {active_quarters} quarters; minimum is {required_quarters}")
    if resolved_selected < min(10, min_selected):
        raise ValueError("rule has too few resolved winner/loser samples")
    return {
        "selected_samples": int(len(selected)),
        "active_quarters": active_quarters,
        "resolved_selected": resolved_selected,
    }

def freeze_rule_artifact(rule_path: Path, output_root: Path, *, agent_columns: Iterable[str]) -> dict[str, Any]:
    """Validate then hash/freeze the rule before any holdout/private artifact exists."""
    rule = validate_rule_artifact(rule_path, agent_columns)
    raw = rule_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    frozen_dir = output_root / "frozen"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen_rule = frozen_dir / "rule.json"
    frozen_rule.write_bytes(raw)
    manifest = {
        "sha256": digest,
        "source": str(rule_path),
        "frozen_rule": str(frozen_rule),
        "validated_condition_count": sum(len(c["all"]) for c in rule["clauses"]),
        "comparison_allowed_only_after_freeze": True,
    }
    (frozen_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest

def evaluate_frozen_rule(
    rule: Mapping[str, Any],
    holdout_df: pd.DataFrame,
    *,
    market_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One-shot holdout report, segmented by quarter and using distributional metrics."""
    selected = apply_rule(rule, holdout_df)
    work = holdout_df.copy()
    work["selected"] = selected
    rows: list[dict[str, Any]] = []
    groups = [("ALL", work), *[(str(q), g) for q, g in work.groupby("period_quarter")]]
    for period, group in groups:
        chosen = group.loc[group["selected"]]
        primary = chosen["Y_primary"].value_counts().to_dict() if not chosen.empty else {}
        universe_primary = group["Y_primary"].value_counts().to_dict()
        resolved = int(primary.get("winner", 0)) + int(primary.get("loser", 0))
        universe_resolved = int(universe_primary.get("winner", 0)) + int(universe_primary.get("loser", 0))
        row: dict[str, Any] = {
            "period": period,
            "holdout_n": int(len(group)),
            "selected_n": int(len(chosen)),
            "selection_coverage": (int(len(chosen)) / int(len(group))) if len(group) else None,
            "winner_n": int(primary.get("winner", 0)),
            "loser_n": int(primary.get("loser", 0)),
            "unresolved_n": int(primary.get("unresolved", 0)),
            "ambiguous_n": int(primary.get("ambiguous", 0)),
            "resolved_winner_rate": (int(primary.get("winner", 0)) / resolved) if resolved else None,
            "universe_resolved_winner_rate": (int(universe_primary.get("winner", 0)) / universe_resolved) if universe_resolved else None,
        }
        if period != "ALL" and market_context is not None and not market_context.empty:
            ctx = market_context.loc[
                (market_context["granularity"].astype(str) == "quarter")
                & (market_context["period"].astype(str) == period)
            ]
            if not ctx.empty:
                row["spy_period_return"] = ctx.iloc[0].get("spy_period_return")
                row["spy_period_max_drawdown"] = ctx.iloc[0].get("spy_period_max_drawdown")
        for target in ("Y_12w_excess", "Y_mae_12w", "Y_mfe_12w"):
            values = pd.to_numeric(chosen[target], errors="coerce").dropna() if not chosen.empty else pd.Series(dtype=float)
            row[f"{target}_p25"] = values.quantile(0.25) if not values.empty else None
            row[f"{target}_p50"] = values.quantile(0.50) if not values.empty else None
            row[f"{target}_p75"] = values.quantile(0.75) if not values.empty else None
        rows.append(row)
    return pd.DataFrame(rows)
