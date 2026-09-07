"""Cross-stage provenance checks for canonical blind discovery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .outcomes import RESEARCH_PRICE_MODE


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def replay_dataset_digest(replay_root: Path) -> tuple[str, int]:
    """Hash exactly the candidate CSV files consumed by blind discovery."""
    paths = sorted(replay_root.glob("*/breakout_follow_pool.csv"))
    h = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(replay_root).as_posix().encode("utf-8")
        h.update(len(relative).to_bytes(4, "big"))
        h.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest(), len(paths)


def validate_replay_preflight(
    replay_root: Path,
    *,
    daily_pkl: Path,
    required_quarters: int,
) -> dict[str, object]:
    """Verify replay completeness and that Stage 1 uses the exact replay inputs."""
    preflight_path = replay_root / "research_replay_preflight.json"
    if not preflight_path.exists():
        raise FileNotFoundError(
            f"canonical replay preflight missing: {preflight_path}; use replay_builder output"
        )
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("price_adjustment_mode") != RESEARCH_PRICE_MODE:
        raise ValueError("replay preflight has unexpected price_adjustment_mode")
    if preflight.get("benchmark_code") != "SPY":
        raise ValueError("canonical replay preflight must use SPY")
    if int(preflight.get("warmup_failed_weeks", -1)) != 0:
        raise ValueError("canonical replay preflight records warmup failures")
    if int(preflight.get("failed_weeks", -1)) != 0:
        raise ValueError("canonical replay preflight records analysis failures")
    expected = int(preflight.get("analysis_weeks_expected", -1))
    persisted = int(preflight.get("analysis_weeks_persisted", -2))
    if expected <= 0 or persisted != expected:
        raise ValueError(
            f"canonical replay is incomplete: persisted={persisted} expected={expected}"
        )
    declared_floor = int(preflight.get("minimum_required_quarters", 0))
    effective_required = max(int(required_quarters), declared_floor)
    quarters = int(preflight.get("successful_quarters", 0))
    if quarters < effective_required:
        raise ValueError(
            f"canonical replay has only {quarters} successful quarters; need {effective_required}"
        )
    actual_daily_sha = sha256_file(daily_pkl)
    if str(preflight.get("daily_pkl_sha256") or "") != actual_daily_sha:
        raise ValueError("Stage 1 daily pkl does not match the pkl used for replay")
    digest, pool_count = replay_dataset_digest(replay_root)
    if pool_count != persisted:
        raise ValueError(
            f"replay pool file count {pool_count} does not match persisted week count {persisted}"
        )
    if str(preflight.get("replay_dataset_sha256") or "") != digest:
        raise ValueError("replay candidate dataset digest does not match replay preflight")
    return preflight
