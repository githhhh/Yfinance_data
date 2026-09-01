from __future__ import annotations
import json
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
)
from backtest.rdagent_track_b_selector_challenge.selectors import (
    select_pure_rank,
    select_distinct_industry,
    select_portfolio_aware,
)
from backtest.rdagent_track_b_selector_challenge.evaluate import (
    _folds,
    compute_weekly_metrics,
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
    assert False in u_sig['b0_eligible'].values  # b0_eligible=False rows ARE included in Signal
    
    u_act = get_universe_panel(mock_df, 'actionable')
    assert len(u_act) == 2
    assert list(u_act['code']) == ['A', 'C']
    assert False in u_act['b0_eligible'].values  # b0_eligible=False rows ARE included in Actionable


def test_agent_leakage_contracts_cover_all_future_and_b0_columns():
    """Verify forbidden leakage columns list covers all target horizons, stops, and B0 states."""
    for h in (1, 2, 3, 4):
        assert any(f'w{h}_' in pref for pref in LEAK_PREFIXES)
    assert {'is_b0', 'pick_order', 'period', 'is_valid_entry', 'entry_status', 'stop_8_hit_ever'} <= LEAK_EXACT_COLS
    
    # Test code scanner
    bad_code = "def factor(df): return df['w4_return_pct'] * 2"
    passed, reason = check_code_leakage(bad_code)
    assert not passed
    assert 'forbidden' in reason.lower()


def test_selection_first_censoring_requires_complete_horizon_labels():
    """Verify portfolio selection occurs before outcome maturity censoring (no survivor reweighting)."""
    mock_picks = pd.DataFrame({
        'snapshot_date': ['2026-01-02'] * 3 + ['2026-01-09'] * 3,
        'code': ['A', 'B', 'C', 'D', 'E', 'F'],
        'w4_return_pct': [10.0, 5.0, np.nan, 2.0, 4.0, 6.0],  # Week 1 has a missing label, Week 2 is complete
        'w4_stop8': [False, False, False, False, False, False],
    })
    
    mets = compute_weekly_metrics(mock_picks, 'test_selector', 'test_segment')
    w4_mets = [m for m in mets if m['horizon'] == 'W4']
    
    # Week 1 (2026-01-02) MUST be completely censored because pick C has NaN return
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
    # dist_to_52w_high_pct is negative (e.g. -2% close, -25% far)
    base_panel = pd.DataFrame({
        'dist_to_52w_high_pct': [-2.0, -5.0, -10.0, -25.0, -40.0],
    })
    # Incorrect inverted factor (-dist)
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
    # Exact affine transform: y = -1.0 * dist_to_52w_high_pct + 0
    affine_factor = -1.0 * base_panel['dist_to_52w_high_pct']
    
    status, feat, corr = audit_redundancy(affine_factor, base_panel, ['dist_to_52w_high_pct', 'mom_20'])
    assert status == 'AFFINE_DUPLICATE'
    assert feat == 'dist_to_52w_high_pct'
    assert corr == pytest.approx(1.0, abs=1e-3)


def test_rdagent_provenance_and_raw_artifacts_integrity():
    """Verify RD-Agent provenance manifest and raw factor files are present and uncorrupted."""
    prov_file = OUT / 'rdagent_provenance.json'
    assert prov_file.exists(), "rdagent_provenance.json must exist"
    
    prov = json.loads(prov_file.read_text(encoding='utf-8'))
    assert prov['rdagent_version'] == '0.8.0'
    assert prov['command']
    assert prov['train_data_boundary'] == f'<= {TRAIN_END}'
    assert 'raw_artifacts' in str(prov) or len(prov.get('factor_paths', [])) >= 3
    
    for rel_path in prov.get('factor_paths', []):
        fp = CHALLENGE_ROOT / rel_path
        assert fp.exists(), f"Raw factor file must exist at {fp}"


def test_research_lock_manifest_sealing():
    """Verify research_lock_manifest.json exists and defines sealed train and validation parameters."""
    lock_file = OUT / 'research_lock_manifest.json'
    assert lock_file.exists(), "research_lock_manifest.json must exist"
    
    lock_data = json.loads(lock_file.read_text(encoding='utf-8'))
    assert lock_data['train_boundary'] == f'<= {TRAIN_END}'
    assert lock_data['contaminated_validation_boundary'] == f'{CONTAM_VAL_START} .. {CONTAM_VAL_END}'
    assert lock_data['purge_weeks'] == PURGE_WEEKS
    assert 'git_sha' in lock_data
