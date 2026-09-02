from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACK_D_ROOT = Path(__file__).resolve().parent
OUT = TRACK_D_ROOT / "output"
RAW_LLM_DIR = OUT / "rdagent_raw"

PANEL_SOURCE = ROOT / "backtest" / "b0_multifactor_challenge" / "data" / "candidate_factor_panel.parquet"
PRODUCTION_SKILL_PATH = ROOT / "dashboard" / "skill_industry_eps_known.py"
TRACK_C_ROOT = ROOT / "backtest" / "track_c_ranking_discovery"
FEATURE_MANIFEST_PATH = TRACK_C_ROOT / "feature_manifest.json"

PRIMARY_HORIZON = "W4"
HISTORICAL_END = "2026-07-24"  # Last fully W4-mature historical snapshot; later snapshots are future shadow.
TOP_N = 3
RANDOM_SEED = 20260901

# Locked chronological design. With the current 42 historical snapshots this gives:
# 18 discovery-train + 4 purge + 6 non-overlapping forward blocks x 3 mature weeks.
DISCOVERY_TRAIN_SNAPSHOTS = 18
PURGE_SNAPSHOTS = 4
OUTER_TEST_SNAPSHOTS = 3
OUTER_BLOCKS = 6
SCREENING_BLOCKS = 2
CONFIRMATION_BLOCKS = 4
AGENT_SCREENING_SHORTLIST = 6
MINIMAL_SCREENING_PER_REMOVAL_COUNT = 1

# Deep profile: use the model as a research team, not a random policy generator.
# 202/1000 requests were already used when this protocol was authored.
# The hard 650-call cap includes retries and leaves meaningful daily headroom.
REQUEST_HARD_LIMIT = 650
MAX_TOKENS_PER_CALL = 5000
# One-time migration source: retain already-paid DeepSeek cache/results when
# moving from the original 126-question protocol to the focused 78-question plan.
CACHE_MIGRATION_ALLOWED_FROM = (
    "00770fb35b75248bfd197598887a83ee85cd042f",
,)
RESEARCH_ROLE_SEQUENCE = ("researcher", "skeptic", "experimental_designer", "synthesizer")
# Focused deep protocol: preserve completed foundational work and spend remaining
# calls only on directions that can still change B1/Minimal-B0 design.
DIRECTION_QUESTION_COUNTS = {
    "mechanism_falsification": 20,
    "failure_archaeology": 22,
    "capacity_abstention": 10,
    "lane_mechanism": 8,
    "nonlinear_b1": 12,
    "adversarial_review": 6,
}
POLICY_SYNTHESIS_CALLS = 13
POLICY_REVIEW_CALLS = 6
FINAL_INTERPRETATION_CALLS = 4
MAX_FROZEN_AGENT_POLICIES = 120

# Minimal-B0 search is deterministic and pre-registered, not LLM-driven.
B0_COMPONENTS = (
    "lane",
    "dry_false_penalty",
    "evidence_risk",
    "freshness",
    "eps_preference",
    "weekly_volume",
    "entry_volume",
    "distinct1",
)
MAX_MINIMAL_B0_REMOVALS = 4
INTERACTION_PAIRS = (
    ("lane", "dry_false_penalty"),
    ("lane", "freshness"),
    ("lane", "distinct1"),
    ("dry_false_penalty", "freshness"),
    ("evidence_risk", "distinct1"),
    ("entry_volume", "distinct1"),
    ("distinct1", "capacity"),
)

# Pre-registered production decision gates on locked forward blocks.
RETURN_MEAN_MIN_SPREAD = 0.25
RETURN_MEDIAN_MIN_SPREAD = 0.0
BOOTSTRAP_CI_LOW_MIN = -0.50
CVAR_MIN_DELTA = -1.00
STOP_MAX_DELTA_PCT = 3.00
RUIN_MAX_DELTA_PCT = 5.00
MIN_POSITIVE_BLOCK_RATIO = 0.75
MIN_FORWARD_SUPPORT_WEEKS = 15  # At least 15/18 mature paired weeks across screening+confirmation.
MIN_CONFIRM_SUPPORT_WEEKS = 10  # At least 10/12 mature paired confirmation weeks.

# Compression allows small return tolerance only if risk is non-inferior.
COMPRESS_MEAN_MIN_SPREAD = -0.25
COMPRESS_MEDIAN_MIN_SPREAD = -0.25
COMPRESS_MIN_POSITIVE_BLOCK_RATIO = 0.50
