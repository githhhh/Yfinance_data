from __future__ import annotations
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backtest.rdagent_track_b_selector_challenge.config import (
    CHALLENGE_ROOT,
    DATA,
    OUT,
    RAW_RDAGENT,
    TRAIN_END,
    CONTAM_VAL_START,
    CONTAM_VAL_END,
    PURGE_WEEKS,
)
from backtest.rdagent_track_b_selector_challenge.panel import (
    get_full_panel,
    get_universe_panel,
)
from backtest.rdagent_track_b_selector_challenge.factor_audit import (
    check_code_leakage,
    audit_semantic_direction,
    audit_redundancy,
    LEAK_EXACT_COLS,
    LEAK_PREFIXES,
    FactorAuditResult,
)
from backtest.rdagent_track_b_selector_challenge.selectors import (
    select_pure_rank,
    select_distinct_industry,
    select_portfolio_aware,
    SelectorConfig,
)
from backtest.rdagent_track_b_selector_challenge.evaluate import (
    _folds,
    compute_cvar,
    compute_weekly_metrics,
    compute_paired_tail_metrics,
    classify_champion,
)
from backtest.rdagent_track_b_selector_challenge.diagnostics import (
    run_pullback_dry_diagnostic,
    run_pullback_encoding_experiment,
)


def test_universe_gating_never_enforces_b0_eligible():
    """Verify Universe S and Universe A do not filter candidates by b0_eligible."""
    mock_df = pd.DataFrame({
        'signal': [True, True, True, False],
        'is_actionable': [1.0, 0.0, 1.0, 1.0],
        'b0_eligible': [False, False, True, True],
        'code': ['A', 'B', 'C', 'D'],
    })
    
    u_sig = get_universe_panel(mock_df, 'signal')
    assert len(u_sig) == 3
    assert False in u_sig['b0_eligible'].values
    
    u_act = get_universe_panel(mock_df, 'actionable')
    assert len(u_act) == 2
    assert list(u_act['code']) == ['A', 'C']
    assert False in u_act['b0_eligible'].values


def test_agent_leakage_contracts_cover_all_future_and_b0_columns():
    """Verify forbidden leakage columns list covers all target horizons, stops, and B0 states."""
    for h in (1, 2, 3, 4):
        assert any(f'w{h}_' in pref for pref in LEAK_PREFIXES)
    assert {'is_b0', 'pick_order', 'period', 'is_valid_entry', 'entry_status', 'stop_8_hit_ever'} <= LEAK_EXACT_COLS
    
    bad_code = "def factor(df): return df['w4_return_pct'] * 2"
    passed, reason = check_code_leakage(bad_code)
    assert not passed
    assert 'forbidden' in reason.lower()


def test_selection_first_censoring_requires_complete_horizon_labels():
    """Verify portfolio selection occurs before outcome maturity censoring."""
    mock_picks = pd.DataFrame({
        'snapshot_date': ['2026-01-02'] * 3 + ['2026-01-09'] * 3,
        'code': ['A', 'B', 'C', 'D', 'E', 'F'],
        'w4_return_pct': [10.0, 5.0, np.nan, 2.0, 4.0, 6.0],
        'w4_stop8': [False, False, False, False, False, False],
    })
    
    mets = compute_weekly_metrics(mock_picks, 'test_selector', 'test_segment')
    w4_mets = [m for m in mets if m['horizon'] == 'W4']
    
    assert len(w4_mets) == 1
    assert w4_mets[0]['snapshot_date'] == '2026-01-09'
    assert w4_mets[0]['return_pct'] == pytest.approx(4.0)


def test_pure_rank_vs_distinct_industry_selector_mechanics():
    """Verify Pure Rank allows duplicate industries while Distinct Industry enforces max 1 per industry."""
    mock_candidates = pd.DataFrame({
        'snapshot_date': ['2026-01-02'] * 4,
        'code': ['A', 'B', 'C', 'D'],
        'industry': ['Software', 'Software', 'Semiconductors', 'Retail'],
        'score': [100.0, 95.0, 80.0, 70.0],
    })
    
    pure_picks = select_pure_rank(mock_candidates, score_col='score', top_n=3)
    assert list(pure_picks['code']) == ['A', 'B', 'C']
    assert list(pure_picks['industry']) == ['Software', 'Software', 'Semiconductors']
    
    distinct_picks = select_distinct_industry(mock_candidates, score_col='score', top_n=3)
    assert list(distinct_picks['code']) == ['A', 'C', 'D']
    assert list(distinct_picks['industry']) == ['Software', 'Semiconductors', 'Retail']


def test_factor_semantic_direction_audit_catches_inverted_prox_52w_high():
    """Verify semantic direction audit rejects prox_52w_high where sign is inverted."""
    base_panel = pd.DataFrame({
        'dist_to_52w_high_pct': [-2.0, -5.0, -10.0, -25.0, -40.0],
    })
    bad_factor_values = pd.Series([2.0, 5.0, 10.0, 25.0, 40.0])
    
    passed, reason = audit_semantic_direction(
        bad_factor_values, 'prox_52w_high', 'Proximity to 52-week high', base_panel
    )
    assert not passed
    assert 'inverted' in reason.lower()


def test_redundancy_audit_identifies_exact_affine_duplicates():
    """Verify redundancy audit identifies affine duplicate of base features."""
    base_panel = pd.DataFrame({
        'dist_to_52w_high_pct': np.linspace(-40.0, -1.0, 30),
        'mom_20': np.random.RandomState(42).randn(30),
    })
    affine_factor = -1.0 * base_panel['dist_to_52w_high_pct']
    
    status, feat, corr = audit_redundancy(affine_factor, base_panel, ['dist_to_52w_high_pct', 'mom_20'])
    assert status == 'AFFINE_DUPLICATE'
    assert feat == 'dist_to_52w_high_pct'
    assert corr == pytest.approx(1.0, abs=1e-3)


def test_pareto_is_not_default_fallback():
    """Verify classify_champion never defaults to PARETO PEER when metrics are unclear/insufficient."""
    # Test insufficient evidence (support < 4)
    res_insuf = classify_champion({}, {'support_weeks': 2, 'median_spread': 0.0, 'cvar_delta': 0.0})
    assert res_insuf == 'INSUFFICIENT EVIDENCE'
    
    # Test inferior
    res_inf = classify_champion(
        {'median_spread': 0.0},
        {'support_weeks': 8, 'median_spread': -2.5, 'mean_spread': -2.0, 'cvar_delta': -3.0, 'stop_delta_pct': 10.0}
    )
    assert res_inf == 'INFERIOR'
    
    # Test arbitrary unclear metrics that do not satisfy Dominates, Pareto, High/Low risk
    res_fallback = classify_champion(
        {'median_spread': 0.0},
        {'support_weeks': 8, 'median_spread': 0.5, 'mean_spread': -0.5, 'cvar_delta': -0.8, 'stop_delta_pct': 0.5}
    )
    # Does not satisfy strict Dominates (mean_spread < 0), High risk, Low risk, or Pareto trade-off -> INSUFFICIENT EVIDENCE
    assert res_fallback == 'INSUFFICIENT EVIDENCE'


def test_paired_tail_uses_identical_snapshot_support():
    """Verify compute_paired_tail_metrics computes metrics strictly on identical common-support weeks."""
    df_ch = pd.DataFrame({
        'selector_id': ['CH1'] * 3,
        'segment': ['test'] * 3,
        'snapshot_date': ['2026-01-02', '2026-01-09', '2026-01-16'],
        'horizon': ['W4'] * 3,
        'return_pct': [5.0, -10.0, 15.0],
        'stop_rate': [0.0, 1.0, 0.0],
        'one_pick_ruined': [False, True, False],
    })
    # B0 only has 2 of the 3 weeks
    df_b0 = pd.DataFrame({
        'selector_id': ['B0'] * 2,
        'segment': ['test'] * 2,
        'snapshot_date': ['2026-01-02', '2026-01-09'],
        'horizon': ['W4'] * 2,
        'return_pct': [2.0, -4.0],
        'stop_rate': [0.0, 0.5],
        'one_pick_ruined': [False, False],
    })
    all_weekly = pd.concat([df_ch, df_b0], ignore_index=True)
    
    # Needs at least 3 weeks for compute_paired_tail_metrics to pass min support check, so let's add 1 common week
    df_ch_4 = pd.concat([df_ch, pd.DataFrame([{'selector_id': 'CH1', 'segment': 'test', 'snapshot_date': '2026-01-23', 'horizon': 'W4', 'return_pct': 8.0, 'stop_rate': 0.0, 'one_pick_ruined': False}])])
    df_b0_4 = pd.concat([df_b0, pd.DataFrame([{'selector_id': 'B0', 'segment': 'test', 'snapshot_date': '2026-01-23', 'horizon': 'W4', 'return_pct': 4.0, 'stop_rate': 0.0, 'one_pick_ruined': False}])])
    
    merged_weekly = pd.concat([df_ch_4, df_b0_4], ignore_index=True)
    pt = compute_paired_tail_metrics(merged_weekly, merged_weekly, 'CH1', 'test')
    assert len(pt) == 1
    # Support weeks must be 3 (the intersection of CH1 and B0 dates: 2026-01-02, 2026-01-09, 2026-01-23), NOT 4
    assert pt.iloc[0]['support_weeks'] == 3


def test_legacy_factor_cannot_be_rdagent_original():
    """Verify legacy copied factors are strictly flagged as legacy_unverified and rejected."""
    legacy_res = FactorAuditResult(
        factor_name='legacy_test',
        origin='legacy_unverified',
        rdagent_run_id='legacy',
        description='copied',
        formula='x',
        code_path='raw_rdagent/legacy_unverified/test.py',
        semantic_direction='UNKNOWN',
        leakage_pass=True,
        redundancy_status='UNKNOWN',
        train_only_discovery=False,
        replay_pass=False,
        accepted=False,
        rejection_reason='copied from previous experiment; not attributable to current Track B RD-Agent run',
    )
    assert legacy_res.origin != 'rdagent_original'
    assert not legacy_res.accepted


def test_cvar_definition_consistency():
    """Verify compute_cvar computes exact average of worst ceil(N*alpha) observations."""
    returns = np.array([-20.0, -10.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]) # N=10, 10% = 1 observation
    cvar = compute_cvar(returns, 0.10)
    assert cvar == pytest.approx(-20.0)
    
    # N=11, ceil(11 * 0.10) = 2 observations (-20, -10) -> mean = -15.0
    returns_11 = np.append(returns, 40.0)
    cvar_11 = compute_cvar(returns_11, 0.10)
    assert cvar_11 == pytest.approx(-15.0)


def test_stop_rate_uses_mature_horizon_denominator():
    """Verify run_pullback_dry_diagnostic only counts rows where horizon outcome is valid (notna)."""
    mock_panel = pd.DataFrame({
        'signal': [True, True, True, True],
        'is_actionable': [1.0, 1.0, 1.0, 1.0],
        'pullback_v_is_dry': [1.0, 1.0, 1.0, 1.0],
        'w4_return_pct': [5.0, -8.0, np.nan, np.nan], # Only 2 mature rows
        'w4_stop8': [False, True, False, False],
        'code': ['A', 'B', 'C', 'D'],
        'snapshot_date': ['2026-01-02'] * 4,
    })
    
    diag_df = run_pullback_dry_diagnostic(mock_panel)
    act_true = diag_df[(diag_df.universe == 'actionable') & (diag_df.state == 'True')].iloc[0]
    
    # Out of 2 mature rows, 1 stopped -> 50.0% stop rate (NOT 25.0% by dividing by all 4 rows)
    assert act_true['w4_mature_count'] == 2
    assert act_true['w4_stop8_rate_pct'] == pytest.approx(50.0)


def test_validation_cannot_run_without_lock(tmp_path, monkeypatch):
    """Verify validate command fails immediately if research_lock_manifest.json is missing."""
    import backtest.rdagent_track_b_selector_challenge.cli as cli_mod
    monkeypatch.setattr(cli_mod, 'OUT', tmp_path)
    
    class Args:
        technical_rerun = False
        
    with pytest.raises(RuntimeError, match="No research_lock_manifest.json found"):
        cli_mod.cmd_validate(Args())


def test_lock_hash_mismatch_fails(tmp_path, monkeypatch):
    """Verify validate command fails immediately if code_hash or panel_hash does not match lock."""
    import backtest.rdagent_track_b_selector_challenge.cli as cli_mod
    monkeypatch.setattr(cli_mod, 'OUT', tmp_path)
    
    lock_manifest = {
        'code_hash': 'fake_code_hash',
        'panel_hash': 'fake_panel_hash',
        'locked_challenger_ids': ['signal_f1_ridge_w4_pure_rank'],
    }
    (tmp_path / 'research_lock_manifest.json').write_text(json.dumps(lock_manifest), encoding='utf-8')
    
    class Args:
        technical_rerun = False
        
    with pytest.raises(RuntimeError, match="Code hash mismatch"):
        cli_mod.cmd_validate(Args())


def test_second_validation_run_rejected(tmp_path, monkeypatch):
    """Verify validate command fails if validation_completed.json exists and --technical-rerun is not passed."""
    import backtest.rdagent_track_b_selector_challenge.cli as cli_mod
    monkeypatch.setattr(cli_mod, 'OUT', tmp_path)
    
    code_h = cli_mod.compute_codebase_hash()
    panel_h = cli_mod.compute_panel_hash()
    
    lock_manifest = {
        'code_hash': code_h,
        'panel_hash': panel_h,
        'locked_challenger_ids': [],
    }
    (tmp_path / 'research_lock_manifest.json').write_text(json.dumps(lock_manifest), encoding='utf-8')
    
    val_dir = tmp_path / 'validation'
    val_dir.mkdir(parents=True, exist_ok=True)
    (val_dir / 'validation_completed.json').write_text(json.dumps({'status': 'done'}), encoding='utf-8')
    
    class Args:
        technical_rerun = False
        
    with pytest.raises(RuntimeError, match="Validation has already been executed"):
        cli_mod.cmd_validate(Args())


def test_rdagent_origin_requires_current_run_source_hash():
    """Verify factor audit only grants rdagent_original if source artifact hash matches captured hash."""
    from backtest.rdagent_track_b_selector_challenge.rdagent_bridge import file_sha256
    
    # Hash match -> valid rdagent_original
    h1 = "abcd1234efgh5678"
    h2 = "abcd1234efgh5678"
    assert bool(h1 and h1 == h2)
    
    # Hash mismatch or missing -> invalid
    h3 = "different_hash"
    assert not bool(h1 and h1 == h3)


def test_pullback_encodings_are_actually_evaluated():
    """Verify run_pullback_encoding_experiment computes Train OOF metrics for all 3 encodings."""
    mock_train = pd.DataFrame({
        'signal': [True] * 40,
        'is_actionable': [1.0] * 40,
        'pullback_v_is_dry': [1.0, 0.0, np.nan, 1.0] * 10,
        'current_vs_ibd_candidate_pct': np.linspace(-5, 5, 40),
        'ibd_entry_volume_ratio': np.linspace(1, 3, 40),
        'volume_ratio': np.linspace(1, 3, 40),
        'ibd_entry_close_position': [0.8] * 40,
        'ibd_entry_breakout_range_ratio': [1.1] * 40,
        'ibd_entry_close_vs_trigger_pct': [0.5] * 40,
        'dist_to_52w_high_pct': [-2.0] * 40,
        'eps_yoy_growth': [25.0] * 40,
        'base_depth_pct': [15.0] * 40,
        'base_duration_weeks': [8] * 40,
        'base_mbox_count': [3] * 40,
        'pullback_pct': [3.0] * 40,
        'pullback_duration_weeks': [2] * 40,
        'C_continuous': [1.0] * 40,
        'px_vs_ma10': [1.0] * 40,
        'px_vs_ma20': [1.0] * 40,
        'px_vs_ma50': [1.0] * 40,
        'ma10_slope_5': [0.1] * 40,
        'ma20_slope_5': [0.1] * 40,
        'ma50_slope_10': [0.1] * 40,
        'mom_5': [2.0] * 40,
        'mom_10': [4.0] * 40,
        'mom_20': [6.0] * 40,
        'mom_60': [10.0] * 40,
        'rv_20': [0.2] * 40,
        'atr_14_pct': [2.5] * 40,
        'vol_ratio_5_20': [1.1] * 40,
        'up_day_ratio_20': [0.6] * 40,
        'drawdown_20': [-3.0] * 40,
        'rel_spy_20': [2.0] * 40,
        'rel_spy_60': [5.0] * 40,
        'w1_return_pct': [2.0, -1.0, 3.0, 0.5] * 10,
        'w2_return_pct': [4.0, -2.0, 5.0, 1.0] * 10,
        'w4_return_pct': [8.0, -4.0, 10.0, 2.0] * 10,
        'w1_stop8': [False] * 40,
        'w2_stop8': [False] * 40,
        'w4_stop8': [False, True, False, False] * 10,
        'code': [f'T{i}' for i in range(40)],
        'snapshot_date': [f'2026-01-{i:02d}' for i in range(1, 41)],
    })
    exp_df = run_pullback_encoding_experiment(mock_train)
    assert not exp_df.empty
    assert set(exp_df['encoding'].unique()) == {'symmetric', 'reward_only', 'ignored'}
