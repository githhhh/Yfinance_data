"""Blind, causal rule-discovery utilities for replay candidates."""

from .experiment import (
    MAX_RESEARCH_SECONDS,
    OutcomeConfig,
    build_blind_dataset,
    build_feature_dossier,
    build_market_context,
    chronological_holdout,
    evaluate_candidate_path,
    freeze_rule_artifact,
    load_replay_candidates,
    run_research_command,
)

__all__ = [
    "MAX_RESEARCH_SECONDS",
    "OutcomeConfig",
    "build_blind_dataset",
    "build_feature_dossier",
    "build_market_context",
    "chronological_holdout",
    "evaluate_candidate_path",
    "freeze_rule_artifact",
    "load_replay_candidates",
    "run_research_command",
]
