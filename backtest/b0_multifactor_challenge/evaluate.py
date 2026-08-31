from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .config import *


def _features(panel, mode):
    f=[x for x in BASE_FEATURES if x in panel.columns]+['trigger_pos','eps_known']+[f'rule_{r}' for r in SIGNAL_TYPES if f'rule_{r}' in panel.columns]
    if mode in {'f1','agent'}: f += [x for x in TECH_FEATURES if x in panel.columns]
    if mode=='agent': f += [x for x in panel.columns if x.startswith('agent_factor_')]
    return list(dict.fromkeys(f))


def _folds(weeks):
    weeks=sorted([str(w) for w in weeks if str(w)<=TRAIN_END])
    folds=[]
    for cut in range(16,len(weeks)-3,4):
        va=weeks[cut:cut+4]
        tr=weeks[:max(0,cut-PURGE_WEEKS)]
        if len(tr)>=12 and va:
            folds.append((tr,va))
    return folds


def _prep_xy(train, score_frame, features, label):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    imp=SimpleImputer(strategy='median',add_indicator=True)
    sc=StandardScaler()
    xt=imp.fit_transform(train[features].apply(pd.to_numeric,errors='coerce'))
    xs=imp.transform(score_frame[features].apply(pd.to_numeric,errors='coerce'))
    xt=sc.fit_transform(xt); xs=sc.transform(xs)
    return xt,train[label].astype(float).to_numpy(),xs


def _model(name, seed=RANDOM_SEED):
    if name=='ridge':
        from sklearn.linear_model import Ridge
        return Ridge(alpha=10.0)
    if name=='elastic':
        from sklearn.linear_model import ElasticNet
        return ElasticNet(alpha=.03,l1_ratio=.2,max_iter=20000,random_state=seed)
    if name=='lgbm':
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            return None
        return LGBMRegressor(n_estimators=80,num_leaves=7,max_depth=3,learning_rate=.03,subsample=.8,colsample_bytree=.8,reg_lambda=5,random_state=seed,verbosity=-1)
    raise ValueError(name)


def _select_top3(df, score_col):
    picks=[]
    for snap,g in df.groupby('snapshot_date',sort=True):
        g=g.sort_values([score_col,'code'],ascending=[False,True])
        used=set(); rows=[]
        for _,r in g.iterrows():
            ind=str(r.get('industry','') or '').strip().lower()
            if ind and ind in used:
                continue
            rows.append(r)
            if ind:
                used.add(ind)
            if len(rows)==TOP_N:
                break
        for i,r in enumerate(rows,1):
            d=r.to_dict(); d['model_pick_order']=i; picks.append(d)
    return pd.DataFrame(picks)


def _metric_rows(picks, model, segment):
    out=[]
    if picks.empty:
        return out
    for h in (*PRIMARY_HORIZONS,*DIAGNOSTIC_HORIZONS):
        col=f'w{h}_return_pct'
        if col not in picks.columns:
            continue
        for snap,g in picks.groupby('snapshot_date'):
            vals=pd.to_numeric(g[col],errors='coerce')
            if len(g)==TOP_N and vals.notna().all():
                stop_col=f'w{h}_stop8'
                stop=pd.Series(g[stop_col] if stop_col in g else False,index=g.index).fillna(False).astype(bool)
                out.append({'model':model,'segment':segment,'snapshot_date':snap,'horizon':f'W{h}','return_pct':float(vals.mean()),'stop_rate':float(stop.mean())})
    return out


def _score_frame(model_name, model, train, score_frame, features, label, model_tag, segment, fold):
    xt,y,xs=_prep_xy(train,score_frame,features,label)
    model.fit(xt,y)
    keep=['snapshot_date','code','industry',*[f'w{x}_return_pct' for x in (*PRIMARY_HORIZONS,*DIAGNOSTIC_HORIZONS) if f'w{x}_return_pct' in score_frame],*[f'w{x}_stop8' for x in (*PRIMARY_HORIZONS,*DIAGNOSTIC_HORIZONS) if f'w{x}_stop8' in score_frame]]
    z=score_frame[keep].copy()
    z['score']=model.predict(xs); z['model']=model_tag; z['segment']=segment; z['fold']=fold
    return z


def run(panel_path: Path | None=None, feature_mode='f1'):
    panel=pd.read_parquet(panel_path or DATA/'candidate_factor_panel.parquet')
    panel=panel[panel['b0_eligible']==True].copy()
    features=_features(panel,feature_mode)
    folds=_folds(panel.snapshot_date.unique())
    predictions=[]
    train_weeks=sorted(panel.loc[panel.snapshot_date<=TRAIN_END,'snapshot_date'].unique())
    sealed_train_weeks=train_weeks[:-PURGE_WEEKS] if len(train_weeks)>PURGE_WEEKS else []

    for h in PRIMARY_HORIZONS:
        label=f'w{h}_return_pct'
        for model_name in ('ridge','elastic','lgbm'):
            for fi,(trw,vaw) in enumerate(folds):
                tr=panel[panel.snapshot_date.isin(trw)&panel[label].notna()]
                va=panel[panel.snapshot_date.isin(vaw)].copy()
                model=_model(model_name)
                if tr.empty or va.empty or model is None:
                    continue
                predictions.append(_score_frame(model_name,model,tr,va,features,label,f'{feature_mode}_{model_name}_w{h}','train_oof',fi))

            tr=panel[panel.snapshot_date.isin(sealed_train_weeks)&panel[label].notna()]
            va=panel[(panel.snapshot_date>=CONTAM_VAL_START)&(panel.snapshot_date<=CONTAM_VAL_END)].copy()
            model=_model(model_name)
            if not tr.empty and not va.empty and model is not None:
                predictions.append(_score_frame(model_name,model,tr,va,features,label,f'{feature_mode}_{model_name}_w{h}','contaminated_validation',-1))

    pred_df=pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame()
    OUT.mkdir(parents=True,exist_ok=True)
    pred_df.to_parquet(OUT/'cv_predictions.parquet',index=False)

    all_metrics=[]; all_picks=[]
    if not pred_df.empty:
        for (model,segment),g in pred_df.groupby(['model','segment']):
            p=_select_top3(g,'score')
            if not p.empty:
                p['segment']=segment; all_picks.append(p)
            all_metrics += _metric_rows(p,model,segment)

    b0=panel[panel.is_b0==1].copy(); b0['model']='B0'; b0['model_pick_order']=b0['pick_order']
    for segment, mask in {
        'train_oof': b0.snapshot_date<=TRAIN_END,
        'contaminated_validation': (b0.snapshot_date>=CONTAM_VAL_START)&(b0.snapshot_date<=CONTAM_VAL_END),
    }.items():
        all_metrics += _metric_rows(b0[mask], 'B0', segment)

    picks=pd.concat(all_picks+[b0],ignore_index=True,sort=False) if all_picks else b0
    metrics=pd.DataFrame(all_metrics)
    picks.to_csv(OUT/'top3_picks.csv',index=False); metrics.to_csv(OUT/'weekly_metrics.csv',index=False)

    paired=[]
    if not metrics.empty:
        b=metrics[metrics.model=='B0'][['segment','snapshot_date','horizon','return_pct','stop_rate']].rename(columns={'return_pct':'b0_return','stop_rate':'b0_stop'})
        for (model,segment),m in metrics[metrics.model!='B0'].groupby(['model','segment']):
            q=m.merge(b,on=['segment','snapshot_date','horizon'],how='inner')
            for horizon,x in q.groupby('horizon'):
                d=x.return_pct-x.b0_return
                paired.append({'model':model,'segment':segment,'horizon':horizon,'weeks':len(x),'median_spread_pct':float(d.median()),'mean_spread_pct':float(d.mean()),'beat_b0_pct':float((d>0).mean()*100),'stop_delta_pct':float((x.stop_rate-x.b0_stop).mean()*100)})
    pd.DataFrame(paired).to_csv(OUT/'b0_paired_comparison.csv',index=False)
    (OUT/'run_manifest.json').write_text(json.dumps({'feature_mode':feature_mode,'features':features,'train_folds':len(folds),'purge_weeks':PURGE_WEEKS,'contaminated_validation':[CONTAM_VAL_START,CONTAM_VAL_END],'models':['ridge','elastic','lgbm(optional)'],'primary_horizons':PRIMARY_HORIZONS,'selection':'same b0_eligible universe; Top3; distinct industry','censoring':'select first, then require complete Top3 horizon labels'},indent=2),encoding='utf-8')
    return metrics
