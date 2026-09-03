from __future__ import annotations

from pathlib import Path

from backtest.track_c_ranking_discovery.config import PANEL_SOURCE

ROOT = Path(__file__).resolve().parents[2]
TRACK_F_ROOT = Path(__file__).resolve().parent
OUT = TRACK_F_ROOT / "output"

PRODUCTION_B0_PATH = ROOT / "dashboard" / "skill_industry_eps_known.py"
PRIMARY_HORIZON = "W4"
TOP_N = 3
PROTOCOL_VERSION = "track_f_v1_lane_taxonomy_composition"
TRACK_D_HISTORICAL_END = "2026-07-24"

BASELINE_POLICY_ID = "B0_ORIGINAL"

QUALITY_ORDER = {
    "confirmed": 0,
    "standard": 1,
    "incomplete": 2,
    "failure": 3,
}

PRIMARY_INDUSTRY_POLICY = "distinct_1"
SECONDARY_INDUSTRY_POLICY = "pure_top3"

POLICY_SPECS = (
    # Primary Lane-composition tests. All retain Production B0 industry dispersion.
    ("CONFIRMED_PARITY_FALLBACK", "parity_fallback", PRIMARY_INDUSTRY_POLICY, "primary"),
    ("CONFIRMED_ONLY_TOP3", "confirmed_only", PRIMARY_INDUSTRY_POLICY, "primary"),
    ("FCS_MAX1", "fcs_max1", PRIMARY_INDUSTRY_POLICY, "primary"),
    # Secondary industry diagnostic: identical Lane logic without distinct_1.
    ("CONFIRMED_PARITY_FALLBACK_NO_IND", "parity_fallback", SECONDARY_INDUSTRY_POLICY, "secondary"),
    ("CONFIRMED_ONLY_TOP3_NO_IND", "confirmed_only", SECONDARY_INDUSTRY_POLICY, "secondary"),
    ("FCS_MAX1_NO_IND", "fcs_max1", SECONDARY_INDUSTRY_POLICY, "secondary"),
)

# Pre-registered historical-support gate. Passing this gate can only justify a
# future shadow candidate because Track F hypotheses were formed after the
# current panel was observed.
HIST_MIN_FULL_SUPPORT = 30
HIST_MIN_LOCKED_SUPPORT = 15
HIST_MIN_MEAN_SPREAD = 0.0
HIST_MIN_MEDIAN_SPREAD = 0.0
HIST_MIN_CVAR_DELTA = -0.5
HIST_MAX_STOP_DELTA_PCT = 3.0
HIST_MAX_RUIN_DELTA_PCT = 5.0
HIST_MIN_CI_LOW = -1.0
