"""Causal, strategy-blind discovery protocol facade."""
from __future__ import annotations

from .outcomes import (
    TRADING_HORIZONS, OutcomeConfig, evaluate_candidate_path, load_price_pickle,
    point_in_time_market_features, restrict_to_mature_outcome_quarters,
)
from .dataset import (
    DISCOVERY_FEATURE_ALLOWLIST, assert_agent_surface_is_blind, build_blind_dataset,
    build_feature_dossier, build_market_context, build_reviewer_market_context,
    load_replay_candidates, purged_chronological_holdout,
)
from .rules import (
    MAX_RESEARCH_SECONDS, apply_rule, evaluate_frozen_rule, freeze_rule_artifact,
    run_research_command, validate_rule_artifact, validate_rule_support,
    write_agent_workspace,
)

__all__ = [
    "DISCOVERY_FEATURE_ALLOWLIST", "MAX_RESEARCH_SECONDS", "TRADING_HORIZONS",
    "OutcomeConfig", "apply_rule", "assert_agent_surface_is_blind",
    "build_blind_dataset", "build_feature_dossier", "build_market_context",
    "build_reviewer_market_context", "evaluate_candidate_path",
    "evaluate_frozen_rule", "freeze_rule_artifact", "load_price_pickle",
    "load_replay_candidates", "point_in_time_market_features",
    "purged_chronological_holdout", "restrict_to_mature_outcome_quarters",
    "run_research_command", "validate_rule_artifact", "validate_rule_support",
    "write_agent_workspace",
]
