from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACK_ROOT = Path(__file__).resolve().parent
OUT = TRACK_ROOT / "output"

PANEL_SOURCE = (
    ROOT / "backtest" / "b0_multifactor_challenge" / "data" / "candidate_factor_panel.parquet"
)
FEATURE_MANIFEST_SOURCE = (
    ROOT / "backtest" / "track_c_ranking_discovery" / "feature_manifest.json"
)
B0_STATE_SOURCE = (
    ROOT / "backtest" / "b0_absolute_quality_audit" / "output" / "current_b0_state.csv"
)
B0_AUDIT_MANIFEST_SOURCE = (
    ROOT / "backtest" / "b0_absolute_quality_audit" / "output" / "run_manifest.json"
)
BASE_PRICE_CACHE = (
    ROOT / "backtest" / "b0_top3_quality_audit" / "data" / "signal_daily_prices.parquet"
)
YAHOO_SUPPLEMENT = (
    ROOT / "backtest" / "b0_absolute_quality_audit" / "output" / "yahoo_price_supplement.parquet"
)

PROTOCOL_VERSION = "b0_error_atlas_v1"
AUDIT_AS_OF_DATE = "2026-09-02"
RANDOM_SEED = 20260903
TOP_N = 3

# Labels are deliberately coarse / tail-oriented. Middle outcomes are excluded
# from the binary recovery/veto training tasks rather than forcing a noisy label.
CLEAN_WINNER_THRESHOLD_PCT = 20.0
CLEAN_SELECTED_THRESHOLD_PCT = 8.0
TERMINAL_LOSER_THRESHOLD_PCT = -8.0
STOP_THRESHOLD_PCT = -8.0
PROFIT_THRESHOLD_PCT = 20.0
HORIZON_CALENDAR_DAYS = 28

# Search / exploration budgets are frozen before materialization.
TOP_NUMERIC_FOR_PAIRS = 10
MAX_PAIR_CANDIDATES = 45
PERMUTATION_REPEATS = 30

# Feature families. B0-derived columns are forbidden from RAW_ONLY.
B0_AUGMENTED_NUMERIC = (
    "current_b0_raw_rank",
    "current_b0_pick_order",
)
B0_AUGMENTED_CATEGORICAL = (
    "current_b0_lane",
    "current_b0_reject_reasons",
)

# Derived PIT features from pre-snapshot prices.
DERIVED_PRICE_FEATURES = (
    "pit_downside_vol_20",
    "pit_downside_vol_60",
    "pit_max_down_day_20",
    "pit_max_down_day_60",
    "pit_gap_down_freq_20",
    "pit_gap_down_freq_60",
    "pit_max_gap_down_20",
    "pit_max_gap_down_60",
    "pit_return_skew_20",
    "pit_return_skew_60",
    "pit_max_drawdown_20",
    "pit_max_drawdown_60",
    "pit_close_position_mean_20",
    "pit_low_close_frac_20",
    "pit_down_volume_share_20",
)

DERIVED_CONTEXT_FEATURES = (
    "xs_mom20_pct",
    "xs_rv20_pct",
    "xs_entry_volume_pct",
    "xs_dist52_pct",
    "sector_candidate_count",
    "industry_candidate_count",
    "sector_mom20_median",
    "mom20_minus_sector_median",
    "sector_actionable_share",
)


DERIVED_MARKET_FEATURES = (
    "pit_spy_mom20",
    "pit_spy_mom60",
    "pit_spy_rv20",
    "pit_spy_drawdown60",
)
