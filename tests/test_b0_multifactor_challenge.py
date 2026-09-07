import pandas as pd
from backtest.b0_multifactor_challenge.evaluate import _folds,_select_top3,_metric_rows
from backtest.b0_multifactor_challenge.agent import LEAK_EXACT,LEAK_PREFIX


def test_walk_forward_has_four_week_purge():
    weeks=[f'2026-{m:02d}-{d:02d}' for m,d in [(1,2),(1,9),(1,16),(1,23),(1,30),(2,6),(2,13),(2,20),(2,27),(3,6),(3,13),(3,20),(3,27),(4,3),(4,10),(4,17),(4,24),(5,1),(5,8),(5,15),(5,22)]]
    fs=_folds(weeks); assert fs
    tr,va=fs[0]; assert set(tr).isdisjoint(va); assert max(tr)<va[0]
    idx=weeks.index(va[0]); assert not set(weeks[idx-4:idx]) & set(tr)


def test_top3_respects_industry_diversity_and_does_not_pre_filter_missing_label():
    df=pd.DataFrame({'snapshot_date':['2026-01-01']*4,'code':['A','B','C','D'],'industry':['x','x','y','z'],'score':[4,3,2,1],'w4_return_pct':[None,9,3,2]})
    out=_select_top3(df,'score')
    assert list(out.code)==['A','C','D']
    assert _metric_rows(out,'m','train_oof')==[]


def test_agent_leakage_contract_names_cover_labels_and_future_entry_state():
    assert {'is_b0','pick_order','is_valid_entry','entry_status'} <= LEAK_EXACT
    assert 'w1_' in LEAK_PREFIX and 'w4_' in LEAK_PREFIX


def test_agent_feature_mode_includes_agent_factors():
    from backtest.b0_multifactor_challenge.evaluate import _features
    panel = pd.DataFrame({
        'current_vs_ibd_candidate_pct': [1.0],
        'mom_20': [0.5],
        'agent_factor_risk_adj_mom_20': [0.1],
        'agent_factor_vol_confirmed_mom_20': [0.2],
    })
    f_f0 = _features(panel, 'f0')
    f_f1 = _features(panel, 'f1')
    f_agent = _features(panel, 'agent')
    assert 'agent_factor_risk_adj_mom_20' not in f_f0
    assert 'agent_factor_risk_adj_mom_20' not in f_f1
    assert 'agent_factor_risk_adj_mom_20' in f_agent
    assert 'agent_factor_vol_confirmed_mom_20' in f_agent


def test_rdagent_factor_manifest_and_artifacts():
    import json
    from pathlib import Path
    base = Path(__file__).resolve().parents[1] / 'backtest/b0_multifactor_challenge'
    manifest_p = base / 'output/rdagent/factor_manifest.json'
    assert manifest_p.exists()
    m = json.loads(manifest_p.read_text(encoding='utf-8'))
    assert m['agent_framework']
    assert len(m['factors']) >= 3
    for f in m['factors']:
        assert f['audit_passed'] is True
        assert f['leakage'] is False
        factor_file = base / 'output/rdagent' / f['code_path']
        assert factor_file.exists()


def test_agent_b0_paired_comparison_schema():
    from pathlib import Path
    base = Path(__file__).resolve().parents[1] / 'backtest/b0_multifactor_challenge'
    p = base / 'output/agent_b0_paired_comparison.csv'
    assert p.exists()
    df = pd.read_csv(p)
    required_cols = {'model', 'segment', 'horizon', 'weeks', 'median_spread_pct', 'mean_spread_pct', 'beat_b0_pct', 'stop_delta_pct'}
    assert required_cols <= set(df.columns)
    assert len(df) > 0

