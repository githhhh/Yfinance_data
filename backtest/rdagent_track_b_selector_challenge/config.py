from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHALLENGE_ROOT = Path(__file__).resolve().parent
POOL_ROOT = ROOT / 'backtest/ibd_skill_replay_pools'
EVENT_OUTCOMES = ROOT / 'backtest/b0_top3_quality_audit/data/candidate_event_outcomes.parquet'
WEEKLY_OUTCOMES = ROOT / 'backtest/b0_top3_quality_audit/data/candidate_weekly_outcomes.parquet'
PRICE_CACHE = ROOT / 'backtest/b0_top3_quality_audit/data/signal_daily_prices.parquet'
B0_SELECTIONS = ROOT / 'backtest/b0_top3_quality_audit/output/b0_selection_events.csv'
PANEL_SOURCE = ROOT / 'backtest/b0_multifactor_challenge/data/candidate_factor_panel.parquet'

DATA = CHALLENGE_ROOT / 'data'
OUT = CHALLENGE_ROOT / 'output'
RAW_RDAGENT = CHALLENGE_ROOT / 'raw_rdagent'
AGENT_WORKSPACE = CHALLENGE_ROOT / 'agent_workspace'

TRAIN_END = '2026-05-22'
CONTAM_VAL_START = '2026-05-29'
CONTAM_VAL_END = '2026-08-07'
FORWARD_START = '2026-08-28'

PRIMARY_HORIZONS = (1, 2, 4)
DIAGNOSTIC_HORIZONS = (3,)
PURGE_WEEKS = 4
TOP_N = 3
RANDOM_SEED = 20260901
BOOTSTRAP_ROUNDS = 2000

UNIVERSES = ('signal', 'actionable')

DRY_POLICIES = ('symmetric', 'reward_only', 'ignored')
B0_SELECTOR_VARIANTS = ('distinct_1', 'pure_top3', 'max_2_per_ind')

BASE_FEATURES = [
    'current_vs_ibd_candidate_pct', 'ibd_entry_volume_ratio', 'volume_ratio',
    'ibd_entry_close_position', 'ibd_entry_breakout_range_ratio', 'ibd_entry_close_vs_trigger_pct',
    'dist_to_52w_high_pct', 'eps_yoy_growth', 'base_depth_pct', 'base_duration_weeks',
    'base_mbox_count', 'pullback_pct', 'pullback_duration_weeks', 'C_continuous',
    'pullback_v_is_dry',
]
SIGNAL_TYPES = ['ceiling', 'pivot', 'ma10_touch_confirm', 'ceiling_pullback', 'three_weeks_tight']
TECH_FEATURES = [
    'px_vs_ma10', 'px_vs_ma20', 'px_vs_ma50', 'ma10_slope_5', 'ma20_slope_5', 'ma50_slope_10',
    'mom_5', 'mom_10', 'mom_20', 'mom_60', 'rv_20', 'atr_14_pct', 'vol_ratio_5_20', 'up_day_ratio_20',
    'drawdown_20', 'rel_spy_20', 'rel_spy_60',
]

@dataclass(frozen=True)
class Fold:
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
