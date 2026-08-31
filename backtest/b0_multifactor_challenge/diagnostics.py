from __future__ import annotations
import numpy as np
import pandas as pd
from .config import *
from .evaluate import _features


def _spearman(g, f, y):
    x=pd.to_numeric(g[f],errors='coerce'); z=pd.to_numeric(g[y],errors='coerce'); m=x.notna()&z.notna()
    if m.sum()<5 or x[m].nunique()<2 or z[m].nunique()<2:
        return np.nan
    return float(x[m].rank().corr(z[m].rank()))


def run_diagnostics(panel_path=None, feature_mode='f1'):
    p=pd.read_parquet(panel_path or DATA/'candidate_factor_panel.parquet')
    p=p[p.b0_eligible==True].copy(); feats=_features(p,feature_mode); rows=[]
    segments={
        'train': p.snapshot_date<=TRAIN_END,
        'contaminated_validation': (p.snapshot_date>=CONTAM_VAL_START)&(p.snapshot_date<=CONTAM_VAL_END),
        'all_historical': p.snapshot_date<=CONTAM_VAL_END,
    }
    for segment,mask in segments.items():
        z0=p[mask]
        for h in (*PRIMARY_HORIZONS,*DIAGNOSTIC_HORIZONS):
            y=f'w{h}_return_pct'
            for f in feats:
                ics=[]; qsp=[]
                for _,g in z0.groupby('snapshot_date'):
                    ic=_spearman(g,f,y)
                    if np.isfinite(ic): ics.append(ic)
                    x=pd.to_numeric(g[f],errors='coerce'); z=pd.to_numeric(g[y],errors='coerce'); m=x.notna()&z.notna()
                    if m.sum()>=10 and x[m].nunique()>=5:
                        try:
                            q=pd.qcut(x[m].rank(method='first'),5,labels=False)
                            qsp.append(float(z[m][q==4].mean()-z[m][q==0].mean()))
                        except ValueError:
                            pass
                rows.append({'segment':segment,'factor':f,'horizon':f'W{h}','weeks_ic':len(ics),'mean_ic':np.nanmean(ics) if ics else np.nan,'median_ic':np.nanmedian(ics) if ics else np.nan,'ic_positive_pct':100*np.mean(np.array(ics)>0) if ics else np.nan,'weeks_quintile':len(qsp),'mean_q5_minus_q1_pct':np.nanmean(qsp) if qsp else np.nan,'median_q5_minus_q1_pct':np.nanmedian(qsp) if qsp else np.nan})
    out=pd.DataFrame(rows); OUT.mkdir(parents=True,exist_ok=True); out.to_csv(OUT/'factor_diagnostics.csv',index=False); return out
