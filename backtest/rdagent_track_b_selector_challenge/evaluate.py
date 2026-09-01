from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from .config import *
from .panel import get_full_panel, get_universe_panel
from .selectors import SelectorConfig, apply_selector


def _get_features(panel: pd.DataFrame, mode: str) -> list[str]:
    """Retrieve features based on mode ('f0', 'f1', 'agent')."""
    base = [x for x in BASE_FEATURES if x in panel.columns] + ['trigger_pos', 'eps_known'] + [f'rule_{r}' for r in SIGNAL_TYPES if f'rule_{r}' in panel.columns]
    if mode in {'f1', 'agent'}:
        base += [x for x in TECH_FEATURES if x in panel.columns]
    if mode == 'agent':
        base += [x for x in panel.columns if x.startswith('agent_factor_')]
    return list(dict.fromkeys(base))


def _folds(weeks: Sequence[str]) -> list[tuple[list[str], list[str]]]:
    """Generate purged walk-forward folds on Train weeks."""
    w_sorted = sorted([str(w) for w in weeks if str(w) <= TRAIN_END])
    folds = []
    for cut in range(16, len(w_sorted) - 3, 4):
        va = w_sorted[cut:cut + 4]
        tr = w_sorted[:max(0, cut - PURGE_WEEKS)]
        if len(tr) >= 12 and va:
            folds.append((tr, va))
    return folds


def _prep_xy(train: pd.DataFrame, test: pd.DataFrame, features: list[str], label: str):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    import warnings
    
    # Filter features that have at least some non-null values in train
    tr_feats = train[features].apply(pd.to_numeric, errors='coerce')
    valid_feats = [col for col in features if tr_feats[col].notna().any()]
    if not valid_feats:
        valid_feats = features
        
    xt_df = tr_feats[valid_feats].fillna(0.0)
    xs_df = test[valid_feats].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    
    sc = StandardScaler()
    xt = sc.fit_transform(xt_df)
    xs = sc.transform(xs_df)
    y = train[label].astype(float).to_numpy()
    return xt, y, xs


def _create_model(name: str, seed: int = RANDOM_SEED):
    if name == 'ridge':
        from sklearn.linear_model import Ridge
        return Ridge(alpha=10.0)
    elif name == 'elastic':
        from sklearn.linear_model import ElasticNet
        return ElasticNet(alpha=0.03, l1_ratio=0.2, max_iter=20000, random_state=seed)
    elif name == 'lgbm':
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(
                n_estimators=80, num_leaves=7, max_depth=3, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=5, random_state=seed, verbosity=-1
            )
        except ImportError:
            return None
    else:
        raise ValueError(f"Unknown model name: {name}")


def _cvar(series: pd.Series, q: float = 0.10) -> float:
    vals = series.dropna().to_numpy(dtype=float)
    if len(vals) < 3:
        return np.nan
    cutoff = np.quantile(vals, q)
    tail = vals[vals <= cutoff]
    return float(np.mean(tail)) if len(tail) > 0 else np.nan


def compute_weekly_metrics(picks: pd.DataFrame, selector_id: str, segment: str) -> list[dict]:
    """Compute weekly portfolio returns and metrics ensuring Selection-First censoring."""
    rows = []
    if picks.empty:
        return rows
        
    for h in (*PRIMARY_HORIZONS, *DIAGNOSTIC_HORIZONS):
        ret_col = f'w{h}_return_pct'
        stop_col = f'w{h}_stop8'
        if ret_col not in picks.columns:
            continue
            
        for snap, g in picks.groupby('snapshot_date'):
            rets = pd.to_numeric(g[ret_col], errors='coerce')
            # Require full maturity: exactly TOP_N picks, all valid returns
            if len(g) == TOP_N and rets.notna().all():
                stops = np.where(g[stop_col].isna(), False, g[stop_col].values).astype(bool) if stop_col in g.columns else np.zeros(len(g), dtype=bool)
                ret_vals = rets.values
                
                # Internal concentration metrics
                # Worst pick contribution to total return
                worst_idx = np.argmin(ret_vals)
                best_idx = np.argmax(ret_vals)
                mean_ret = float(np.mean(ret_vals))
                
                # One-pick-ruins: portfolio return < 0, but exactly 1 pick was negative and other 2 were >= 0
                is_neg_week = mean_ret < 0
                one_pick_ruined = is_neg_week and (np.sum(ret_vals < 0) == 1)
                
                # Loss concentration: fraction of negative sum caused by worst pick
                neg_picks = ret_vals[ret_vals < 0]
                worst_loss_conc = (min(ret_vals) / np.sum(neg_picks)) if len(neg_picks) > 0 and np.sum(neg_picks) < 0 else np.nan
                
                # Gain concentration: fraction of positive sum caused by best pick
                pos_picks = ret_vals[ret_vals > 0]
                best_gain_conc = (max(ret_vals) / np.sum(pos_picks)) if len(pos_picks) > 0 and np.sum(pos_picks) > 0 else np.nan

                rows.append({
                    'selector_id': selector_id,
                    'segment': segment,
                    'snapshot_date': str(snap),
                    'horizon': f'W{h}',
                    'picks_count': len(g),
                    'return_pct': mean_ret,
                    'stop_rate': float(stops.mean()),
                    'worst_pick_ret': float(np.min(ret_vals)),
                    'best_pick_ret': float(np.max(ret_vals)),
                    'one_pick_ruined': one_pick_ruined,
                    'worst_loss_conc': worst_loss_conc,
                    'best_gain_conc': best_gain_conc,
                })
    return rows


def compute_tail_metrics(weekly_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate return, downside, upside, and tail ratio metrics."""
    out = []
    for (sel, seg, h), g in weekly_metrics_df.groupby(['selector_id', 'segment', 'horizon']):
        rets = g['return_pct'].dropna()
        if len(rets) < 3:
            continue
        vals = rets.to_numpy()
        stops = g['stop_rate'].dropna().to_numpy()
        
        p10 = float(np.quantile(vals, 0.10))
        p90 = float(np.quantile(vals, 0.90))
        cvar10 = _cvar(rets, 0.10)
        cvar20 = _cvar(rets, 0.20)
        
        # Top 10% / Top 20% mean
        top10_cutoff = np.quantile(vals, 0.90)
        top10_mean = float(np.mean(vals[vals >= top10_cutoff]))
        top20_cutoff = np.quantile(vals, 0.80)
        top20_mean = float(np.mean(vals[vals >= top20_cutoff]))
        
        tail_ratio10 = top10_mean / abs(cvar10) if abs(cvar10) > 1e-6 else np.nan
        
        neg_weeks = np.sum(vals < 0)
        ruined_count = g['one_pick_ruined'].sum()
        one_pick_ruins_rate = (ruined_count / neg_weeks * 100.0) if neg_weeks > 0 else 0.0
        
        out.append({
            'selector_id': sel,
            'segment': seg,
            'horizon': h,
            'weeks': len(rets),
            'mean_return_pct': float(np.mean(vals)),
            'median_return_pct': float(np.median(vals)),
            'p10_pct': p10,
            'p90_pct': p90,
            'cvar10_pct': cvar10,
            'cvar20_pct': cvar20,
            'top10_mean_pct': top10_mean,
            'top20_mean_pct': top20_mean,
            'tail_ratio10': tail_ratio10,
            'negative_week_rate_pct': float(100.0 * neg_weeks / len(vals)),
            'worst_week_pct': float(np.min(vals)),
            'avg_stop_rate_pct': float(100.0 * np.mean(stops)) if len(stops) > 0 else np.nan,
            'one_pick_ruins_rate_pct': float(one_pick_ruins_rate),
            'avg_worst_loss_conc_pct': float(100.0 * g['worst_loss_conc'].dropna().mean()) if g['worst_loss_conc'].notna().any() else np.nan,
            'avg_best_gain_conc_pct': float(100.0 * g['best_gain_conc'].dropna().mean()) if g['best_gain_conc'].notna().any() else np.nan,
        })
    return pd.DataFrame(out)


def bootstrap_comparison(
    challenger_weekly: pd.DataFrame,
    b0_weekly: pd.DataFrame,
    n_rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = RANDOM_SEED,
) -> dict:
    """Run deterministic paired bootstrap for mean/median spread and CVaR differences."""
    merged = pd.merge(
        challenger_weekly[['snapshot_date', 'return_pct', 'stop_rate']].rename(columns={'return_pct': 'ch_ret', 'stop_rate': 'ch_stop'}),
        b0_weekly[['snapshot_date', 'return_pct', 'stop_rate']].rename(columns={'return_pct': 'b0_ret', 'stop_rate': 'b0_stop'}),
        on='snapshot_date',
        how='inner',
    )
    if len(merged) < 3:
        return {'support_weeks': len(merged), 'mean_spread_ci': [np.nan, np.nan], 'median_spread_ci': [np.nan, np.nan], 'cvar_diff_ci': [np.nan, np.nan]}
        
    rng = np.random.RandomState(seed)
    n = len(merged)
    mean_spreads = []
    median_spreads = []
    cvar_diffs = []
    
    ch_rets = merged['ch_ret'].to_numpy()
    b0_rets = merged['b0_ret'].to_numpy()
    
    for _ in range(n_rounds):
        idx = rng.choice(n, size=n, replace=True)
        sample_ch = ch_rets[idx]
        sample_b0 = b0_rets[idx]
        
        diff = sample_ch - sample_b0
        mean_spreads.append(np.mean(diff))
        median_spreads.append(np.median(diff))
        
        ch_cvar = np.mean(sample_ch[sample_ch <= np.quantile(sample_ch, 0.10)]) if len(sample_ch) >= 5 else np.nan
        b0_cvar = np.mean(sample_b0[sample_b0 <= np.quantile(sample_b0, 0.10)]) if len(sample_b0) >= 5 else np.nan
        cvar_diffs.append(ch_cvar - b0_cvar)
        
    return {
        'support_weeks': n,
        'mean_spread_pct': float(np.mean(ch_rets - b0_rets)),
        'mean_spread_ci_95': [float(np.quantile(mean_spreads, 0.025)), float(np.quantile(mean_spreads, 0.975))],
        'median_spread_pct': float(np.median(ch_rets - b0_rets)),
        'median_spread_ci_95': [float(np.quantile(median_spreads, 0.025)), float(np.quantile(median_spreads, 0.975))],
        'cvar_diff_ci_95': [float(np.nanquantile(cvar_diffs, 0.025)), float(np.nanquantile(cvar_diffs, 0.975))],
    }


def classify_champion(
    train_oof_metrics: dict,
    val_metrics: dict,
    bootstrap_res: dict,
) -> str:
    """Classify challenger under Champion Hierarchy."""
    val_med_spread = val_metrics.get('median_spread_pct', -999)
    val_mean_spread = val_metrics.get('mean_spread_pct', -999)
    val_cvar_delta = val_metrics.get('cvar_delta', -999)  # positive means less negative (better)
    val_stop_delta = val_metrics.get('stop_delta_pct', 999) # negative means fewer stops (better)
    val_support = val_metrics.get('weeks', 0)
    
    oof_med_spread = train_oof_metrics.get('median_spread_pct', -999)
    
    if val_support < 4:
        return 'INSUFFICIENT EVIDENCE'
        
    # Check direction reversal
    if (oof_med_spread > 0 and val_med_spread < -2.0) or (oof_med_spread < -1.0 and val_med_spread > 3.0):
        return 'UNSTABLE'
        
    # Dominates B0
    if val_med_spread >= 0.0 and val_mean_spread >= 0.0 and val_cvar_delta >= -1.5 and val_stop_delta <= 5.0 and oof_med_spread >= -0.5:
        return 'DOMINATES B0'
        
    # High Return / High Risk
    if (val_med_spread > 1.0 or val_mean_spread > 1.0) and (val_cvar_delta < -2.5 or val_stop_delta > 8.0):
        return 'HIGH RETURN / HIGH RISK'
        
    # Lower Return / Lower Risk
    if (val_med_spread < 0.0 or val_mean_spread < 0.0) and (val_cvar_delta > 2.0 and val_stop_delta < -5.0):
        return 'LOWER RETURN / LOWER RISK'
        
    # Pareto Peer
    if abs(val_med_spread) <= 1.5 and abs(val_cvar_delta) <= 2.0:
        return 'PARETO PEER'
        
    # Inferior
    if val_med_spread < 0.0 and val_mean_spread < 0.0 and (val_cvar_delta <= 0.0 or val_stop_delta >= 0.0):
        return 'INFERIOR'
        
    return 'PARETO PEER'
