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

ACTIVE_SOFT_LANES = (
    "fresh_demand_alpha",
    "constructive_pullback",
    "standard_breakout",
)
DOWNGRADED_LANES = (
    "incomplete_evidence",
    "tail_risk",
)

BASELINE_POLICY_ID = "B0_ORIGINAL"
CHALLENGER_POLICY_ID = "B0_1_DRY_REWARD_ONLY_SOFT_ACTIVE_LANE"
