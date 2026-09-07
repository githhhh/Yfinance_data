from __future__ import annotations

from pathlib import Path

from backtest.track_c_ranking_discovery.config import PANEL_SOURCE

ROOT = Path(__file__).resolve().parents[2]
TRACK_E_ROOT = Path(__file__).resolve().parent
OUT = TRACK_E_ROOT / "output"

PRODUCTION_B0_PATH = ROOT / "dashboard" / "skill_industry_eps_known.py"
PRIMARY_HORIZON = "W4"
TRACK_D_HISTORICAL_END = "2026-07-24"
TOP_N = 3

BASELINE_POLICY_ID = "B0_DRY_NEUTRAL_HARD_LANE"
PRODUCTION_REFERENCE_ID = "B0_ORIGINAL"
CHALLENGER_POLICY_ID = "B0_1_PAIRWISE_STANDARD_CHALLENGES_FRESH"
PROTOCOL_VERSION = "track_e_v3_pairwise_top3_replacement"

TARGET_FRESH_LANE = "fresh_demand_alpha"
TARGET_STANDARD_LANE = "standard_breakout"
