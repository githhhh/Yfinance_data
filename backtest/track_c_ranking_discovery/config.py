from __future__ import annotations
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[2]
TRACK_C_ROOT = Path(__file__).resolve().parent
DATA = TRACK_C_ROOT / "data"
OUT = TRACK_C_ROOT / "output"
PROPOSALS_DIR = TRACK_C_ROOT / "discovery_sandbox" / "proposals"
FEATURE_MANIFEST_PATH = TRACK_C_ROOT / "feature_manifest.json"

PANEL_SOURCE = ROOT / "backtest" / "b0_multifactor_challenge" / "data" / "candidate_factor_panel.parquet"
PRODUCTION_SKILL_PATH = ROOT / "dashboard" / "skill_industry_eps_known.py"

# Time Horizons
TRAIN_END = "2026-05-22"
CONTAM_VAL_START = "2026-05-29"
CONTAM_VAL_END = "2026-08-07"
PURGE_WEEKS = 4

# Horizions
PRIMARY_HORIZON = "W4"
EVAL_HORIZONS = ("W1", "W2", "W4")

# Core Parameters
TOP_N = 3  # Maximum capacity of portfolio slots
RANDOM_SEED = 20260901
MC_PATHS = 5000
BOOTSTRAP_ROUNDS = 2000
BLOCK_BOOTSTRAP_BLOCK_LEN = 4

# Research Search Budgets
MANDATORY_GRID_BUDGET = 36
DISCOVERY_BUDGET = 54
TOTAL_BUDGET = 90

FAMILY_BUDGETS = {
    "structural": 36,
    "industry_breadth": 11,
    "continuous": 11,
    "linear_ranking": 11,
    "portfolio": 10,
    "novel_heuristic": 11,
}

# LOWO Overfit Pre-Registered Fragility Rule
LOWO_MAX_POSITIVE_EDGE_CONCENTRATION = 0.50
LOWO_MIN_SIGN_STABILITY = 0.70

# Structural Grid Dimensions
LANE_POLICIES = ("B0_LANE", "PULLBACK_PARITY", "LANE_NEUTRAL", "SCORE_BEFORE_LANE")
DRY_POLICIES = ("symmetric", "reward_only", "ignored")
SELECTOR_POLICIES = ("distinct_1", "pure_top3", "max_2_per_ind")
