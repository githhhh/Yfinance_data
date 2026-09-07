"""Run R6 on an existing local trigger_path_samples.csv; no price/EPS refresh."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import subprocess

import pandas as pd

from .dataset import DISCOVERY_FEATURE_ALLOWLIST
from .pipeline_contract import sha256_file
from .r6_agent import RDAgentProposer
from .r6_stability import Config, LABELS, canonical, discover, write_report


def load_inputs(samples: Path, metadata: Path, expected_sha: str):
    actual_sha = sha256_file(samples)
    if actual_sha != expected_sha:
        raise ValueError("samples SHA256 differs from explicit execution input")
    meta = json.loads(metadata.read_text())
    replay_sha = str(meta.get("replay_dataset_sha256", ""))
    if len(replay_sha) != 64 or any(c not in "0123456789abcdef" for c in replay_sha):
        raise ValueError("source metadata requires a replay dataset SHA256")
    frame = pd.read_csv(samples, dtype={"code": str})
    required = {"code", "snapshot_date", "entry_date", "exit_date_w3", *LABELS}
    if required - set(frame):
        raise ValueError(f"missing sample columns: {sorted(required-set(frame))}")
    if len(frame) != meta.get("usable_trigger_entries"):
        raise ValueError("sample count differs from metadata usable_trigger_entries")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("empty samples or missing identifiers/labels")
    for field in ("snapshot_date", "entry_date", "exit_date_w3"):
        frame[field] = pd.to_datetime(frame[field], errors="raise")
    if frame.duplicated(["snapshot_date", "code"]).any():
        raise ValueError("duplicate snapshot/code sample keys")
    if not ((frame.snapshot_date < frame.entry_date) & (frame.entry_date <= frame.exit_date_w3)).all():
        raise ValueError("invalid signal/entry/label temporal order")
    labels = frame[list(LABELS)].apply(pd.to_numeric, errors="raise")
    if not labels.isin([0, 1]).all().all() or not (labels.sum(axis=1) == 1).all():
        raise ValueError("W3 labels must be binary, exhaustive and exclusive")
    frame[list(LABELS)] = labels
    features = [f for f in DISCOVERY_FEATURE_ALLOWLIST if f in frame]
    if not features:
        raise ValueError("no allowlisted PIT features")
    declared = meta.get("entry_quarters")
    actual_quarters = sorted(frame.entry_date.dt.to_period("Q").astype(str).unique())
    if declared != actual_quarters:
        raise ValueError("metadata entry quarters differ from samples")
    first = min(frame.snapshot_date.min().to_period("Q"), pd.Period(declared[0], freq="Q"))
    last = max(frame.snapshot_date.max().to_period("Q"), pd.Period(declared[-1], freq="Q"))
    calendar = [str(q) for q in pd.period_range(first, last, freq="Q")]
    manifest = {
        "samples_sha256": actual_sha, "metadata_sha256": sha256_file(metadata),
        "source_replay_dataset_sha256": replay_sha,
        "source_replay_sha_independently_recomputed": False,
        "sample_rows": len(frame), "snapshot_weeks": frame.snapshot_date.nunique(),
        "unique_tickers": frame.code.nunique(), "calendar": calendar,
        "features": features, "execution_features_excluded": True,
        "population": "usable executable entries only, conditional on reconstructed universe",
    }
    # No W4 or 12w columns can accidentally enter proposer or selection feedback.
    return frame[["code", "snapshot_date", "entry_date", "exit_date_w3", *LABELS, *features]], manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--samples-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True, help="reuse across R6 runs; never reset")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--prior-used", type=int, default=42)
    parser.add_argument("--run-call-cap", type=int, default=120)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--preflight", action="store_true", help="read-only input/backend checks; no model calls")
    args = parser.parse_args()
    frame, manifest = load_inputs(args.samples, args.metadata, args.samples_sha256)
    cfg = Config(rounds=args.rounds)
    if len(manifest["calendar"]) <= cfg.min_train_quarters:
        raise ValueError("not enough calendar quarters for chronological evaluation")
    try:
        version = importlib.metadata.version("rdagent")
        from rdagent.oai.backend.litellm import LiteLLMAPIBackend
        assert callable(LiteLLMAPIBackend.build_messages_and_create_chat_completion)
    except (ImportError, importlib.metadata.PackageNotFoundError, AttributeError) as exc:
        raise RuntimeError("Compatible official RD-Agent backend is required in quant_env; no substitute agent is used") from exc
    manifest["rdagent_version"] = version
    manifest["config"] = asdict(cfg)
    manifest["execution_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest["source_sha256"] = {
        name: sha256_file(Path(__file__).with_name(name))
        for name in ("r6_runner.py", "r6_stability.py", "r6_agent.py", "dataset.py", "pipeline_contract.py")
    }
    if args.preflight:
        print(canonical(manifest))
        return 0
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError("use a fresh output root; preserve prior runs")
    output = args.output_root.resolve()
    for protected in (args.samples.resolve(), args.metadata.resolve(), args.ledger.resolve(), args.cache_dir.resolve()):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("output, source inputs, ledger and cache must be separate")
    proposer = RDAgentProposer(ledger_path=args.ledger, cache_dir=args.cache_dir,
                              prior_used=args.prior_used, run_cap=args.run_call_cap)
    try:
        result = discover(frame, manifest["features"], manifest["calendar"], proposer, cfg, output)
        if sha256_file(args.samples) != manifest["samples_sha256"] or sha256_file(args.metadata) != manifest["metadata_sha256"]:
            raise RuntimeError("source inputs changed during execution; result invalid")
        manifest["requests"] = proposer.snapshot()
        (output / "input_manifest.json").write_text(canonical(manifest) + "\n")
        write_report(result, output)
    except Exception:
        output.mkdir(parents=True, exist_ok=True)
        (output / "FAILED.json").write_text(canonical({"status": "FAILED", "requests": proposer.snapshot()}) + "\n")
        raise
    print(canonical({"status": "COMPLETE", "requests": proposer.snapshot(),
                     "report": str(output / "R6_REPORT.md")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
