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


def compute_cvar(returns: np.ndarray | pd.Series, alpha: float = 0.10) -> float:
    """Unified Conditional Value at Risk (CVaR10 / Expected Shortfall)."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    k = max(1, int(np.ceil(len(arr) * alpha)))
    sorted_arr = np.sort(arr)
    return float(np.mean(sorted_arr[:k]))


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
        cvar10 = compute_cvar(rets, 0.10)
        cvar20 = compute_cvar(rets, 0.20)
        
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


def compute_paired_tail_metrics(
    challenger_weekly: pd.DataFrame,
    b0_weekly: pd.DataFrame,
    selector_id: str,
    segment: str,
) -> pd.DataFrame:
    """Compute paired tail metrics strictly on identical common-support snapshot dates."""
    records = []
    for h in (*PRIMARY_HORIZONS, *DIAGNOSTIC_HORIZONS):
        h_str = f'W{h}'
        ch_sub = challenger_weekly[(challenger_weekly.selector_id == selector_id) & (challenger_weekly.segment == segment) & (challenger_weekly.horizon == h_str)]
        b0_sub = b0_weekly[(b0_weekly.selector_id == 'B0') & (b0_weekly.segment == segment) & (b0_weekly.horizon == h_str)]
        
        merged = pd.merge(
            ch_sub[['snapshot_date', 'return_pct', 'stop_rate', 'one_pick_ruined']],
            b0_sub[['snapshot_date', 'return_pct', 'stop_rate', 'one_pick_ruined']],
            on='snapshot_date',
            suffixes=('_ch', '_b0'),
            how='inner',
        )
        if len(merged) < 3:
            continue
            
        ch_rets = merged['return_pct_ch'].to_numpy(dtype=float)
        b0_rets = merged['return_pct_b0'].to_numpy(dtype=float)
        ch_stops = merged['stop_rate_ch'].to_numpy(dtype=float)
        b0_stops = merged['stop_rate_b0'].to_numpy(dtype=float)
        
        ch_cvar10 = compute_cvar(ch_rets, 0.10)
        b0_cvar10 = compute_cvar(b0_rets, 0.10)
        
        ch_p10 = float(np.quantile(ch_rets, 0.10))
        b0_p10 = float(np.quantile(b0_rets, 0.10))
        
        ch_top10 = float(np.mean(ch_rets[ch_rets >= np.quantile(ch_rets, 0.90)]))
        b0_top10 = float(np.mean(b0_rets[b0_rets >= np.quantile(b0_rets, 0.90)]))
        
        ch_tr10 = ch_top10 / abs(ch_cvar10) if abs(ch_cvar10) > 1e-6 else np.nan
        b0_tr10 = b0_top10 / abs(b0_cvar10) if abs(b0_cvar10) > 1e-6 else np.nan
        
        ch_neg = np.sum(ch_rets < 0)
        b0_neg = np.sum(b0_rets < 0)
        
        records.append({
            'selector_id': selector_id,
            'segment': segment,
            'horizon': h_str,
            'support_weeks': len(merged),
            'challenger_mean': float(np.mean(ch_rets)),
            'b0_mean': float(np.mean(b0_rets)),
            'mean_spread': float(np.mean(ch_rets - b0_rets)),
            'challenger_median': float(np.median(ch_rets)),
            'b0_median': float(np.median(b0_rets)),
            'median_spread': float(np.median(ch_rets - b0_rets)),
            'challenger_cvar10': ch_cvar10,
            'b0_cvar10': b0_cvar10,
            'cvar_delta': float(ch_cvar10 - b0_cvar10),
            'challenger_p10': ch_p10,
            'b0_p10': b0_p10,
            'challenger_top10_mean': ch_top10,
            'b0_top10_mean': b0_top10,
            'challenger_tail_ratio10': ch_tr10,
            'b0_tail_ratio10': b0_tr10,
            'challenger_stop_rate_pct': float(100.0 * np.mean(ch_stops)),
            'b0_stop_rate_pct': float(100.0 * np.mean(b0_stops)),
            'stop_delta_pct': float(100.0 * np.mean(ch_stops - b0_stops)),
            'challenger_one_pick_ruins_pct': float(100.0 * merged['one_pick_ruined_ch'].sum() / max(1, ch_neg)),
            'b0_one_pick_ruins_pct': float(100.0 * merged['one_pick_ruined_b0'].sum() / max(1, b0_neg)),
            'one_pick_ruins_delta_pct': float(100.0 * (merged['one_pick_ruined_ch'].sum() / max(1, ch_neg) - merged['one_pick_ruined_b0'].sum() / max(1, b0_neg))),
        })
    return pd.DataFrame(records)


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
        return {'support_weeks': len(merged), 'mean_spread_ci_95': [np.nan, np.nan], 'median_spread_ci_95': [np.nan, np.nan], 'cvar_diff_ci_95': [np.nan, np.nan]}
        
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
        mean_spreads.append(float(np.mean(diff)))
        median_spreads.append(float(np.median(diff)))
        
        ch_cvar = compute_cvar(sample_ch, 0.10)
        b0_cvar = compute_cvar(sample_b0, 0.10)
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
    bootstrap_res: dict | None = None,
) -> str:
    """Classify challenger under Champion Hierarchy without any default PARETO catch-all."""
    val_med_spread = val_metrics.get('median_spread', val_metrics.get('median_spread_pct', np.nan))
    val_mean_spread = val_metrics.get('mean_spread', val_metrics.get('mean_spread_pct', np.nan))
    val_cvar_delta = val_metrics.get('cvar_delta', np.nan)  # positive means less negative (better downside)
    val_stop_delta = val_metrics.get('stop_delta_pct', np.nan) # negative means fewer stops (better)
    val_support = val_metrics.get('support_weeks', val_metrics.get('weeks', 0))
    
    oof_med_spread = train_oof_metrics.get('median_spread', train_oof_metrics.get('median_spread_pct', np.nan))
    oof_mean_spread = train_oof_metrics.get('mean_spread', train_oof_metrics.get('mean_spread_pct', np.nan))
    
    # 1. Check support sufficiency
    if val_support < 4 or np.isnan(val_med_spread) or np.isnan(val_cvar_delta):
        return 'INSUFFICIENT EVIDENCE'
        
    # 2. Check direction reversal / instability
    if (oof_med_spread > 2.0 and val_med_spread < -1.0) or (oof_med_spread < -1.0 and val_med_spread > 2.0) or (oof_med_spread > 0 and val_mean_spread < -3.0):
        return 'UNSTABLE'
        
    # 3. Dominates B0
    if val_med_spread >= -0.05 and val_mean_spread >= 0.0 and val_cvar_delta >= -0.5 and val_stop_delta <= 0.5 and oof_med_spread >= -0.5:
        return 'DOMINATES B0'
        
    # 4. High Return / High Risk
    if (val_med_spread >= 1.0 or val_mean_spread >= 1.0) and (val_cvar_delta < -2.0 or val_stop_delta > 2.0):
        return 'HIGH RETURN / HIGH RISK'
        
    # 5. Lower Return / Lower Risk
    if (val_med_spread < -1.0 or val_mean_spread < -1.0) and (val_cvar_delta >= 2.0 or val_stop_delta <= -2.0):
        return 'LOWER RETURN / LOWER RISK'
        
    # 6. Pareto Peer (True trade-off, strictly defined)
    # Trade-off Case A: Return approx equal, Downside materially better
    pareto_a = (abs(val_med_spread) <= 1.0 and abs(val_mean_spread) <= 1.0) and (val_cvar_delta >= 2.0 or val_stop_delta <= -2.0)
    # Trade-off Case B: Return materially better, Downside approx equal
    pareto_b = (val_med_spread >= 2.0 or val_mean_spread >= 2.0) and (val_cvar_delta >= -1.0 and val_stop_delta <= 1.0)
    if pareto_a or pareto_b:
        return 'PARETO PEER'
        
    # 7. Inferior (Return worse and Risk worse or equal)
    if val_med_spread < 0.0 and val_mean_spread < 0.0 and (val_cvar_delta <= 0.0 or val_stop_delta >= 0.0):
        return 'INFERIOR'
        
    # 8. Strict fallback: Never catch-all PARETO
    return 'INSUFFICIENT EVIDENCE'

