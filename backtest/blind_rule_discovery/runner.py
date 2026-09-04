"""CLI for producing a blind discovery workspace and time-bucketed data dossier."""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from .experiment import (
    MAX_RESEARCH_SECONDS,
    OutcomeConfig,
    build_blind_dataset,
    build_feature_dossier,
    build_market_context,
    chronological_holdout,
    freeze_rule_artifact,
    load_price_pickle,
    load_replay_candidates,
    run_research_command,
    write_agent_workspace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--daily-pkl", type=Path, required=True, help="full-history daily OHLC pickle")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spy-code", default="SPY")
    parser.add_argument(
        "--benchmark-pkl",
        type=Path,
        default=None,
        help="optional separate pickle containing benchmark OHLC",
    )
    parser.add_argument("--holdout-quarters", type=int, default=4)
    parser.add_argument("--stop-loss", type=float, default=-0.08)
    parser.add_argument("--winner-gain", type=float, default=0.20)
    parser.add_argument("--agent-command", default="", help="optional external command, run inside blind workspace")
    parser.add_argument("--research-seconds", type=int, default=MAX_RESEARCH_SECONDS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    candidates = load_replay_candidates(args.replay_root)
    prices = load_price_pickle(args.daily_pkl)
    benchmark_prices = load_price_pickle(args.benchmark_pkl) if args.benchmark_pkl else prices
    if args.spy_code not in benchmark_prices:
        raise KeyError(f"benchmark {args.spy_code!r} missing from benchmark price source")

    config = OutcomeConfig(stop_loss=args.stop_loss, winner_gain=args.winner_gain)
    agent_df, feature_map, reviewer = build_blind_dataset(
        candidates,
        prices,
        benchmark_prices[args.spy_code],
        config=config,
    )
    discovery_df, holdout_df, sealed_quarters = chronological_holdout(
        agent_df,
        holdout_quarters=args.holdout_quarters,
    )
    market = build_market_context(reviewer, benchmark_prices[args.spy_code])
    period_summary, feature_profile = build_feature_dossier(agent_df, market)

    # Reviewer-only artifacts live outside agent_workspace.
    reviewer.to_csv(args.output_root / "reviewer_outcomes.csv", index=False)
    holdout_df.to_csv(args.output_root / "sealed_holdout.csv", index=False)
    market.to_csv(args.output_root / "market_context.csv", index=False)
    period_summary.to_csv(args.output_root / "period_summary.csv", index=False)
    feature_profile.to_csv(args.output_root / "feature_profile.csv", index=False)
    (args.output_root / "private_feature_map.json").write_text(
        json.dumps(feature_map, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    discovery_periods = set(discovery_df["period_month"].astype(str)) | set(
        discovery_df["period_quarter"].astype(str)
    )
    discovery_market = market.loc[
        market["period"].astype(str).isin(discovery_periods)
    ].reset_index(drop=True)
    workspace = write_agent_workspace(discovery_df, args.output_root, discovery_market)

    metadata = {
        "candidate_rows": int(len(candidates)),
        "usable_rows": int(len(agent_df)),
        "discovery_rows": int(len(discovery_df)),
        "sealed_holdout_rows": int(len(holdout_df)),
        "sealed_holdout_quarters": sealed_quarters,
        "entry_semantics": "next_trading_session_open_after_snapshot",
        "winner_semantics": "target_before_stop",
        "stop_out_then_winner_is_training_winner": False,
        "horizons_sessions": {"4w": 20, "8w": 40, "12w": 60},
        "research_seconds_cap": MAX_RESEARCH_SECONDS,
        "agent_workspace": str(workspace),
    }
    (args.output_root / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.agent_command:
        completed = run_research_command(
            shlex.split(args.agent_command),
            workspace,
            timeout_seconds=args.research_seconds,
        )
        (args.output_root / "agent_stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (args.output_root / "agent_stderr.txt").write_text(completed.stderr, encoding="utf-8")
        rule_path = workspace / "rule.json"
        if completed.returncode != 0:
            return completed.returncode
        if not rule_path.exists():
            raise FileNotFoundError("research command completed without agent_workspace/rule.json")
        freeze_rule_artifact(rule_path, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
