"""CLI for blind discovery, purge/holdout validation, and post-freeze materialization."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import shlex
import shutil
from pathlib import Path

from .experiment import (
    MAX_RESEARCH_SECONDS,
    OutcomeConfig,
    build_blind_dataset,
    build_feature_dossier,
    build_reviewer_market_context,
    evaluate_frozen_rule,
    freeze_rule_artifact,
    load_price_pickle,
    load_replay_candidates,
    purged_chronological_holdout,
    restrict_to_mature_outcome_quarters,
    run_research_command,
    validate_rule_artifact,
    validate_rule_support,
    write_agent_workspace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--daily-pkl", type=Path, required=True, help="full-history daily OHLC pickle")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spy-code", default="SPY")
    parser.add_argument("--benchmark-pkl", type=Path, default=None)
    parser.add_argument("--holdout-quarters", type=int, default=4)
    parser.add_argument("--stop-loss", type=float, default=-0.08)
    parser.add_argument("--winner-gain", type=float, default=0.20)
    parser.add_argument("--entry-window-sessions", type=int, default=5)
    parser.add_argument("--max-entry-extension", type=float, default=0.05)
    parser.add_argument(
        "--allow-unadjusted-outcomes",
        action="store_true",
        help="unsafe override when the pickle lacks Adj Close; canonical runs should not use it",
    )
    parser.add_argument("--agent-command", default="")
    parser.add_argument(
        "--sandbox-prefix",
        default="",
        help="required with --agent-command; OS/container wrapper, may use {workspace}",
    )
    parser.add_argument("--research-seconds", type=int, default=MAX_RESEARCH_SECONDS)
    return parser.parse_args()


def _write_public_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _prepare_output_root(output_root: Path) -> None:
    """Remove stale generated artifacts, but never erase evidence of consumed holdout."""
    output_root.mkdir(parents=True, exist_ok=True)
    consumed = output_root / "holdout_consumed.json"
    if consumed.exists():
        raise RuntimeError(
            f"sealed holdout for this output root was already consumed: {consumed}; "
            "do not tune and re-evaluate it"
        )
    generated_files = {
        "experiment_metadata.json", "agent_stdout.txt", "agent_stderr.txt",
        "reviewer_outcomes.csv", "embargo_samples.csv", "sealed_holdout.csv",
        "private_feature_map.json", "reviewer_market_context.csv", "holdout_report.csv",
    }
    for name in generated_files:
        path = output_root / name
        if path.exists():
            path.unlink()
    for dirname in ("agent_workspace", "frozen"):
        path = output_root / dirname
        if path.exists():
            shutil.rmtree(path)


def main() -> int:
    args = parse_args()
    _prepare_output_root(args.output_root)
    candidates_all = load_replay_candidates(args.replay_root)
    require_adjusted = not args.allow_unadjusted_outcomes
    prices = load_price_pickle(args.daily_pkl)
    benchmark_prices = load_price_pickle(args.benchmark_pkl) if args.benchmark_pkl else prices
    if args.spy_code not in benchmark_prices:
        raise KeyError(f"benchmark {args.spy_code!r} missing from benchmark price source")

    config = OutcomeConfig(
        stop_loss=args.stop_loss,
        winner_gain=args.winner_gain,
        entry_window_sessions=args.entry_window_sessions,
        max_entry_extension=args.max_entry_extension,
    )
    future_sessions_required = config.minimum_sessions + config.entry_window_sessions
    candidates, immature_quarter_rows, maturity_cutoff = restrict_to_mature_outcome_quarters(
        candidates_all, benchmark_prices[args.spy_code], minimum_sessions=future_sessions_required
    )
    if require_adjusted:
        benchmark_mode = benchmark_prices[args.spy_code].attrs.get("price_adjustment_mode")
        if benchmark_mode != "adj_close_factor":
            raise ValueError(f"benchmark {args.spy_code} lacks Adj Close; split-unsafe benchmark path")
        candidate_codes = set(candidates["code"].astype(str))
        unsafe_codes = sorted(
            code for code in candidate_codes
            if code in prices and prices[code].attrs.get("price_adjustment_mode") != "adj_close_factor"
        )
        if unsafe_codes:
            raise ValueError(
                f"{len(unsafe_codes)} candidate symbols lack Adj Close; split-unsafe outcome paths: "
                + ",".join(unsafe_codes[:10])
            )
    agent_df, feature_map, reviewer = build_blind_dataset(
        candidates,
        prices,
        benchmark_prices[args.spy_code],
        config=config,
    )
    discovery_df, embargo_df, holdout_df, sealed_quarters, holdout_start = purged_chronological_holdout(
        agent_df,
        reviewer,
        holdout_quarters=args.holdout_quarters,
    )
    period_summary, feature_profile = build_feature_dossier(discovery_df)
    workspace = write_agent_workspace(
        discovery_df,
        args.output_root,
        period_summary=period_summary,
        feature_profile=feature_profile,
    )

    metadata = {
        "daily_price_source_sha256": _sha256_file(args.daily_pkl),
        "benchmark_price_source_sha256": _sha256_file(args.benchmark_pkl) if args.benchmark_pkl else _sha256_file(args.daily_pkl),
        "price_adjustment_required": require_adjusted,
        "candidate_rows_before_maturity_filter": int(len(candidates_all)),
        "candidate_rows": int(len(candidates)),
        "excluded_immature_quarter_rows": int(len(immature_quarter_rows)),
        "outcome_maturity_cutoff": str(maturity_cutoff.date()),
        "usable_rows": int(len(agent_df)),
        "censored_rows": int((~reviewer["usable"].fillna(False)).sum()) if not reviewer.empty else 0,
        "censor_reasons": reviewer.loc[~reviewer["usable"].fillna(False), "reason"].fillna("unknown").value_counts().to_dict() if not reviewer.empty else {},
        "discovery_rows": int(len(discovery_df)),
        "embargo_rows": int(len(embargo_df)),
        "sealed_holdout_rows": int(len(holdout_df)),
        "sealed_holdout_quarters": sealed_quarters,
        "holdout_start": str(holdout_start.date()),
        "entry_semantics": "first_trigger_cross_or_open_within_buy_zone_after_snapshot",
        "entry_window_sessions": config.entry_window_sessions,
        "max_entry_extension": config.max_entry_extension,
        "winner_semantics": "plus20_before_minus8",
        "loser_semantics": "minus8_before_plus20_including_later_recovery",
        "unresolved_semantics": "neither_boundary_within_60_sessions",
        "purge_semantics": "drop_discovery_rows_whose_12w_outcome_window_reaches_holdout_start",
        "research_seconds_cap": MAX_RESEARCH_SECONDS,
        "private_artifacts_materialized": False,
        "agent_workspace": str(workspace),
    }
    metadata_path = args.output_root / "experiment_metadata.json"
    _write_public_metadata(metadata_path, metadata)

    if not args.agent_command:
        return 0
    if not args.sandbox_prefix:
        raise RuntimeError("--sandbox-prefix is required with --agent-command")

    completed = run_research_command(
        shlex.split(args.agent_command),
        workspace,
        sandbox_prefix=shlex.split(args.sandbox_prefix),
        timeout_seconds=args.research_seconds,
    )
    (args.output_root / "agent_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (args.output_root / "agent_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        return completed.returncode
    rule_path = workspace / "rule.json"
    if not rule_path.exists():
        raise FileNotFoundError("research command completed without agent_workspace/rule.json")

    # P0 barrier: validate + freeze before any private mapping or holdout is written.
    rule = validate_rule_artifact(rule_path, discovery_df.columns)
    support = validate_rule_support(rule, discovery_df)
    freeze_manifest = freeze_rule_artifact(rule_path, args.output_root, agent_columns=discovery_df.columns)

    # Only now materialize reviewer/private data and evaluate the sealed holdout once.
    reviewer.to_csv(args.output_root / "reviewer_outcomes.csv", index=False)
    embargo_df.to_csv(args.output_root / "embargo_samples.csv", index=False)
    holdout_df.to_csv(args.output_root / "sealed_holdout.csv", index=False)
    (args.output_root / "private_feature_map.json").write_text(
        json.dumps(feature_map, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    market = build_reviewer_market_context(reviewer, benchmark_prices[args.spy_code])
    market.to_csv(args.output_root / "reviewer_market_context.csv", index=False)
    holdout_report = evaluate_frozen_rule(rule, holdout_df, market_context=market)
    holdout_report.to_csv(args.output_root / "holdout_report.csv", index=False)

    metadata["private_artifacts_materialized"] = True
    metadata["holdout_evaluated_after_freeze"] = True
    metadata["frozen_rule_sha256"] = freeze_manifest["sha256"]
    metadata["rule_discovery_support"] = support
    _write_public_metadata(metadata_path, metadata)
    consumed_manifest = {
        "consumed_at_utc": datetime.now(timezone.utc).isoformat(),
        "sealed_holdout_quarters": sealed_quarters,
        "frozen_rule_sha256": freeze_manifest["sha256"],
        "holdout_report": str(args.output_root / "holdout_report.csv"),
    }
    (args.output_root / "holdout_consumed.json").write_text(
        json.dumps(consumed_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
