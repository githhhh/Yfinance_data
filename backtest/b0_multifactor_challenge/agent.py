from __future__ import annotations
import json, os, shutil, subprocess
from pathlib import Path
import pandas as pd
from .config import *

README='''# B0 Candidate Factor Research Dataset

This is a PIT-safe weekly US-stock candidate panel. Each row is (snapshot_date, code).
The RD-Agent coding workspace contains TRAIN rows only. Use only columns in daily_pv.h5 / candidate_panel.h5 key=data.
Never use returns, stops, future/as-of outcomes, B0 membership/order, dates/tickers as fitted rules, or any hard-coded threshold learned from outcomes.
Implement an economically interpretable continuous factor. The factor program must write result.h5 with one numeric column and the same MultiIndex(datetime,instrument). Higher value should mean stronger expected future selection quality.
'''
LEAK_PREFIX=('w1_','w2_','w3_','w4_')
LEAK_EXACT={'is_b0','pick_order','period','is_valid_entry','entry_status','entry_date','entry_open','stop_8_hit_ever','gap_stop','profit20_hit','max_drawdown_to_asof_pct'}


def _safe_panel(panel):
    safe=[c for c in panel.columns if c not in LEAK_EXACT and not c.startswith(LEAK_PREFIX) and not c.endswith('_return_pct')]
    out=panel[safe].copy(); out['datetime']=pd.to_datetime(out.snapshot_date); out['instrument']=out.code
    return out.set_index(['datetime','instrument']).sort_index()


def _write_source(folder: Path, frame: pd.DataFrame):
    folder.mkdir(parents=True,exist_ok=True)
    frame.to_hdf(folder/'candidate_panel.h5',key='data',mode='w')
    frame.to_hdf(folder/'daily_pv.h5',key='data',mode='w')
    (folder/'README.md').write_text(README,encoding='utf-8')


def prepare_agent_data(panel_path: Path | None=None):
    panel=pd.read_parquet(panel_path or DATA/'candidate_factor_panel.parquet').copy()
    safe=_safe_panel(panel)
    train=safe[safe.index.get_level_values('datetime')<=pd.Timestamp(TRAIN_END)]
    AGENT.mkdir(parents=True,exist_ok=True)
    train_dir=AGENT/'source_data_train'; debug_dir=AGENT/'source_data_debug'; full_dir=AGENT/'source_data_full'
    _write_source(train_dir,train)
    _write_source(debug_dir,train.groupby(level=0).head(30))
    _write_source(full_dir,safe)
    return train_dir,debug_dir,full_dir


def write_task_spec():
    spec={'objective':'Discover continuous, PIT-safe factors that improve weekly Top3 ranking relative to frozen B0.','agent_data_boundary':f'train only through {TRAIN_END}','primary_label':'W4 return, evaluated externally; agent never receives labels','constraints':['no future/outcome columns','no ticker/date-specific rules','continuous preferred','must execute on candidate_panel.h5','output result.h5'],'candidate_dimensions':['breakout geometry','buy-point/high proximity','volume/demand','momentum/MA trend','volatility/risk','signal-type interactions','EPS/quality interactions']}
    AGENT.mkdir(parents=True,exist_ok=True); p=AGENT/'research_task.json'; p.write_text(json.dumps(spec,indent=2),encoding='utf-8'); return p


def run_official_rdagent(step_n: int=2):
    train_dir,debug_dir,_=prepare_agent_data(); write_task_spec()
    if shutil.which('rdagent') is None:
        raise RuntimeError('rdagent CLI not found; pip install rdagent in the local research env')
    env=os.environ.copy()
    env['FACTOR_CoSTEER_DATA_FOLDER']=str(train_dir)
    env['FACTOR_CoSTEER_DATA_FOLDER_DEBUG']=str(debug_dir)
    return subprocess.run(['rdagent','fin_factor',f'--step_n={step_n}'],env=env,check=False,text=True).returncode


def replay_factor_code(name: str, factor_py: Path, panel_path: Path | None=None):
    _,_,full_dir=prepare_agent_data(panel_path)
    run_dir=AGENT/'replay'/''.join(ch if ch.isalnum() or ch in '_-' else '_' for ch in name.lower())
    if run_dir.exists(): shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    for src in full_dir.iterdir():
        shutil.copy2(src,run_dir/src.name)
    shutil.copy2(factor_py,run_dir/'factor.py')
    proc=subprocess.run([os.environ.get('PYTHON','python'),'factor.py'],cwd=run_dir,text=True,capture_output=True)
    if proc.returncode!=0:
        raise RuntimeError(f'factor replay failed: {proc.stdout}\n{proc.stderr}')
    result=run_dir/'result.h5'
    if not result.exists():
        raise RuntimeError('factor.py completed but result.h5 was not produced')
    return import_factor_result(name,result,panel_path)


def import_factor_result(name: str, result_h5: Path, panel_path: Path | None=None):
    target=panel_path or DATA/'candidate_factor_panel.parquet'
    panel=pd.read_parquet(target).copy(); f=pd.read_hdf(result_h5)
    if isinstance(f,pd.Series): f=f.to_frame('value')
    if f.shape[1]!=1: raise ValueError('agent factor result must have exactly one column')
    if not isinstance(f.index,pd.MultiIndex): raise ValueError('agent factor index must be MultiIndex(datetime,instrument)')
    tmp=f.reset_index(); tmp.columns=['datetime','code','value']; tmp['snapshot_date']=pd.to_datetime(tmp.datetime).dt.strftime('%Y-%m-%d'); tmp['code']=tmp.code.astype(str).str.upper()
    if tmp.duplicated(['snapshot_date','code']).any(): raise ValueError('duplicate factor keys')
    col='agent_factor_'+''.join(ch if ch.isalnum() or ch=='_' else '_' for ch in name.lower())
    out=panel.merge(tmp[['snapshot_date','code','value']],on=['snapshot_date','code'],how='left',validate='one_to_one')
    out[col]=pd.to_numeric(out.pop('value'),errors='coerce')
    out.to_parquet(target,index=False)
    return col
