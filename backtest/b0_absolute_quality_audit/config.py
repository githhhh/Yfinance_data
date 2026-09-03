from __future__ import annotations

from pathlib import Path

from backtest.track_c_ranking_discovery.config import PANEL_SOURCE

ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = Path(__file__).resolve().parent
OUT = AUDIT_ROOT / "output"

PRICE_CACHE = ROOT / "backtest" / "b0_top3_quality_audit" / "data" / "signal_daily_prices.parquet"
PRODUCTION_B0_PATH = ROOT / "dashboard" / "skill_industry_eps_known.py"

PROTOCOL_VERSION = "b0_absolute_quality_v1_3"
PRIMARY_HORIZON = "W4"
TOP_N = 3
SNAPSHOT_FORWARD_DAYS = 28

# Freeze the Yahoo/live-data boundary for this retrospective audit. Materialization
# run after this date must still ignore bars after this date.
AUDIT_AS_OF_DATE = "2026-09-02"
BENCHMARK_CODES = ("SPY", "QQQ")

RANDOM_SEED = 20260903
RAW_MC_DRAWS = 200_000
MAX_EXACT_COMBINATIONS = 100_000
BLOCK_BOOTSTRAP_ROUNDS = 5000
BLOCK_BOOTSTRAP_LEN = 4

SIMPLE_BASELINES = (
    ("closest_to_trigger", "current_vs_ibd_candidate_pct", "abs_asc"),
    ("entry_volume", "ibd_entry_volume_ratio", "desc"),
    ("eps", "eps_yoy_growth", "desc"),
    ("momentum_20", "mom_20", "desc"),
)

# Primary raw inference remains fail-closed after Yahoo supplementation.
RAW_PRICE_COVERAGE_MIN_FOR_PRIMARY = 1.00
ELIGIBLE_ENTRY_COVERAGE_MIN_FOR_PRIMARY = 1.00

WINNER_TOP_FRAC = 0.20
BIG_WINNER_THRESHOLD_PCT = 20.0
LOSER_BOTTOM_FRAC = 0.20

YAHOO_SUPPLEMENT_PARQUET = OUT / "yahoo_price_supplement.parquet"
YAHOO_DOWNLOAD_AUDIT_CSV = OUT / "yahoo_download_audit.csv"

CAPACITY_POLICY_IDS = (
    "B0_ORIGINAL",
    "B0_FILL3_RELAX_INDUSTRY",
    "B0_FILL3_EPS_ONLY",
    "B0_FILL3_SINGLE_REJECT",
    "B0_FILL3_ANY_REJECT",
)
