from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import *
from dashboard.skill_industry_eps_known import reasoned_item, effective_eps, is_review_universe


def _bool(v):
    if pd.isna(v): return np.nan
    if isinstance(v, bool): return float(v)
    s=str(v).strip().lower()
    if s in {'true','1','yes'}: return 1.0
    if s in {'false','0','no'}: return 0.0
    return np.nan


def load_pools(pool_root: Path = POOL_ROOT) -> pd.DataFrame:
    frames=[]
    for p in sorted(pool_root.glob('*/breakout_follow_pool.csv')):
        df=pd.read_csv(p)
        if df.empty: continue
        snap=p.parent.name
        if 'snapshot_date' not in df: df['snapshot_date']=snap
        df['snapshot_date']=df['snapshot_date'].astype(str)
        sig=df['signal'].astype(str).str.lower().eq('true') if 'signal' in df else pd.Series(False,index=df.index)
        df=df[sig & df['ibd_candidate_rule'].fillna('').astype(str).str.strip().ne('')]
        eligible=[]
        for row_idx,row in df.iterrows():
            if not is_review_universe(row):
                eligible.append(False); continue
            item=reasoned_item(row,row_idx)
            ok=(item.entry_status=='ACTIONABLE' and 'clear_geometry_failure' not in item.risk_codes and 'below_candidate_buy_point' not in item.risk_codes and effective_eps(item) is not None and bool(item.industry.strip()))
            eligible.append(bool(ok))
        df=df.copy(); df['b0_eligible']=eligible
        frames.append(df)
    if not frames: raise RuntimeError(f'No replay pools under {pool_root}')
    out=pd.concat(frames,ignore_index=True)
    out['code']=out['code'].astype(str).str.upper().str.strip()
    return out


def _technical_from_prices(prices: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    p=prices.copy(); p['date']=pd.to_datetime(p['date']); p['code']=p['code'].astype(str).str.upper()
    spy=p[p.code=='SPY'].sort_values('date').copy()
    for n in (20,60): spy[f'spy_mom_{n}']=spy.close/spy.close.shift(n)-1
    spy=spy[['date','spy_mom_20','spy_mom_60']]
    rows=[]
    for code,g in p.groupby('code',sort=False):
        g=g.sort_values('date').copy()
        c=g['close'].astype(float); h=g['high'].astype(float); l=g['low'].astype(float); v=g.get('volume',pd.Series(index=g.index,dtype=float)).astype(float)
        for n in (10,20,50): g[f'ma{n}']=c.rolling(n,min_periods=n).mean()
        g['px_vs_ma10']=c/g.ma10-1; g['px_vs_ma20']=c/g.ma20-1; g['px_vs_ma50']=c/g.ma50-1
        g['ma10_slope_5']=g.ma10/g.ma10.shift(5)-1; g['ma20_slope_5']=g.ma20/g.ma20.shift(5)-1; g['ma50_slope_10']=g.ma50/g.ma50.shift(10)-1
        for n in (5,10,20,60): g[f'mom_{n}']=c/c.shift(n)-1
        ret=c.pct_change(); g['rv_20']=ret.rolling(20,min_periods=20).std()
        prev=c.shift(1); tr=pd.concat([(h-l).abs(),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
        g['atr_14_pct']=tr.rolling(14,min_periods=14).mean()/c
        g['vol_ratio_5_20']=v.rolling(5,min_periods=5).mean()/v.rolling(20,min_periods=20).mean()
        g['up_day_ratio_20']=(ret>0).rolling(20,min_periods=20).mean()
        g['drawdown_20']=c/c.rolling(20,min_periods=20).max()-1
        g=g.merge(spy,on='date',how='left')
        g['rel_spy_20']=g.mom_20-g.spy_mom_20; g['rel_spy_60']=g.mom_60-g.spy_mom_60
        ev=events[events.code==code][['snapshot_date','code']].drop_duplicates().copy(); ev['dt']=pd.to_datetime(ev.snapshot_date)
        if ev.empty: continue
        m=pd.merge_asof(ev.sort_values('dt'),g[['date',*TECH_FEATURES]].sort_values('date'),left_on='dt',right_on='date',direction='backward',allow_exact_matches=True)
        if (m['date']>m['dt']).any(): raise AssertionError(f'future price leakage for {code}')
        rows.append(m[['snapshot_date','code',*TECH_FEATURES]])
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=['snapshot_date','code',*TECH_FEATURES])


def build_panel(output: Path | None = None) -> pd.DataFrame:
    pools=load_pools()
    events=pd.read_parquet(EVENT_OUTCOMES)
    weekly=pd.read_parquet(WEEKLY_OUTCOMES)
    prices=pd.read_parquet(PRICE_CACHE)
    for df in (events,weekly):
        df['code']=df['code'].astype(str).str.upper().str.strip(); df['snapshot_date']=df['snapshot_date'].astype(str)
    evcols=['snapshot_date','code','is_valid_entry','entry_status','entry_date','entry_open','stop_8_hit_ever','gap_stop','profit20_hit','max_drawdown_to_asof_pct']
    panel=pools.merge(events[[c for c in evcols if c in events]],on=['snapshot_date','code'],how='left',validate='one_to_one')
    for h in (*PRIMARY_HORIZONS,*DIAGNOSTIC_HORIZONS):
        w=weekly[(weekly['holding_week_index']==h) & (weekly['is_complete_week']==True)][['snapshot_date','code','week_close_return_from_entry_pct','stop_8_hit_by_week_end']].copy()
        w=w.rename(columns={'week_close_return_from_entry_pct':f'w{h}_return_pct','stop_8_hit_by_week_end':f'w{h}_stop8'})
        panel=panel.merge(w,on=['snapshot_date','code'],how='left',validate='one_to_one')
    tech=_technical_from_prices(prices,panel[['snapshot_date','code']])
    panel=panel.merge(tech,on=['snapshot_date','code'],how='left',validate='one_to_one')
    panel['trigger_pos']=pd.to_numeric(panel.get('ibd_entry_close_position'),errors='coerce')-pd.to_numeric(panel.get('ibd_entry_breakout_range_ratio'),errors='coerce')
    panel['eps_known']=pd.to_numeric(panel.get('eps_yoy_growth'),errors='coerce').notna().astype(float)
    panel['pullback_v_is_dry']=panel.get('pullback_v_is_dry',np.nan).map(_bool)
    panel['is_actionable']=(panel.get('ibd_entry_status','').astype(str).str.upper()=='ACTIONABLE').astype(float)
    for rule in SIGNAL_TYPES: panel[f'rule_{rule}']=(panel.ibd_candidate_rule.astype(str)==rule).astype(float)
    b0=pd.read_csv(B0_SELECTIONS,usecols=['snapshot_date','code','pick_order']); b0['code']=b0.code.astype(str).str.upper()
    panel=panel.merge(b0.assign(is_b0=1),on=['snapshot_date','code'],how='left'); panel['is_b0']=panel.is_b0.fillna(0).astype(int)
    panel['period']=np.select([panel.snapshot_date<=TRAIN_END,(panel.snapshot_date>=CONTAM_VAL_START)&(panel.snapshot_date<=CONTAM_VAL_END),panel.snapshot_date>=FORWARD_START],['train','contaminated_validation','forward'],'other')
    panel=panel.sort_values(['snapshot_date','code']).reset_index(drop=True)
    output=output or DATA/'candidate_factor_panel.parquet'; output.parent.mkdir(parents=True,exist_ok=True); panel.to_parquet(output,index=False)
    manifest={'rows':len(panel),'weeks':int(panel.snapshot_date.nunique()),'train_end':TRAIN_END,'contaminated_validation':[CONTAM_VAL_START,CONTAM_VAL_END],'forward_start':FORWARD_START,'price_max_le_snapshot_checked':True,'source':'EPS_RECALIBRATED_V2 frozen pools/outcomes/prices'}
    (output.parent/'panel_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return panel
