from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .config import *
from .panel import get_full_panel, get_universe_panel


def compute_cvar(returns: np.ndarray | pd.Series, alpha: float = 0.10) -> float:
    """Unified Conditional Value at Risk (CVaR10 / Expected Shortfall)."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    k = max(1, int(np.ceil(len(arr) * alpha)))
    sorted_arr = np.sort(arr)
    return float(np.mean(sorted_arr[:k]))


def _spearman(x: pd.Series, y: pd.Series) -> float:
    s = pd.concat([pd.to_numeric(x, errors='coerce'), pd.to_numeric(y, errors='coerce')], axis=1).dropna()
    if len(s) < 5 or s.iloc[:, 0].nunique() < 2 or s.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(s.iloc[:, 0].rank().corr(s.iloc[:, 1].rank()))


def run_pullback_dry_diagnostic(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Detailed empirical outcome analysis for pullback_v_is_dry states (True, False, Missing)."""
    if panel is None:
        panel = get_full_panel()

    records = []
    universes = {
        'signal': get_universe_panel(panel, 'signal'),
        'actionable': get_universe_panel(panel, 'actionable'),
        'b0_eligible': get_universe_panel(panel, 'b0_eligible'),
    }

    for u_name, u_df in universes.items():
        dry_raw = u_df['pullback_v_is_dry']
        state = pd.Series('Missing', index=u_df.index)
        state[dry_raw == 1.0] = 'True'
        state[dry_raw == 0.0] = 'False'

        for st in ('True', 'False', 'Missing', 'All'):
            sub = u_df if st == 'All' else u_df[state == st]
            n = len(sub)
            if n == 0:
                continue

            rec = {
                'universe': u_name,
                'state': st,
                'sample_size': n,
                'pct_of_universe': round(100.0 * n / len(u_df), 2),
            }

            for h in (1, 2, 4):
                ret_col = f'w{h}_return_pct'
                stop_col = f'w{h}_stop8'
                
                # Filter to mature rows where return is valid (notna)
                if ret_col in sub.columns:
                    valid_mask = pd.to_numeric(sub[ret_col], errors='coerce').notna()
                    sub_mature = sub[valid_mask]
                    rets = pd.to_numeric(sub_mature[ret_col], errors='coerce').dropna()
                else:
                    sub_mature = sub.iloc[0:0]
                    rets = pd.Series(dtype=float)
                    
                stops = np.where(sub_mature[stop_col].isna(), False, sub_mature[stop_col].values).astype(bool) if stop_col in sub_mature.columns else np.zeros(len(sub_mature), dtype=bool)
                
                rec[f'w{h}_mature_count'] = len(rets)
                rec[f'w{h}_mean_pct'] = round(float(rets.mean()), 3) if len(rets) > 0 else np.nan
                rec[f'w{h}_median_pct'] = round(float(rets.median()), 3) if len(rets) > 0 else np.nan
                rec[f'w{h}_p10_pct'] = round(float(np.quantile(rets, 0.10)), 3) if len(rets) >= 5 else np.nan
                rec[f'w{h}_p90_pct'] = round(float(np.quantile(rets, 0.90)), 3) if len(rets) >= 5 else np.nan
                rec[f'w{h}_cvar10_pct'] = round(compute_cvar(rets, 0.10), 3)
                rec[f'w{h}_stop8_rate_pct'] = round(100.0 * float(stops.mean()), 2) if len(stops) > 0 else np.nan

            records.append(rec)

    df_out = pd.DataFrame(records)
    OUT.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT / 'pullback_dry_diagnostic.csv', index=False)
    return df_out


def run_pullback_encoding_experiment(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Controlled Train-only comparison of pullback_v_is_dry encodings (symmetric, reward_only, ignored)."""
    if panel is None:
        panel = get_full_panel()
        
    train_panel = panel[panel.snapshot_date <= TRAIN_END].copy()
    from .evaluate import _folds, _prep_xy, _create_model
    
    rows = []
    for u_name in ('actionable', 'signal'):
        u_df = get_universe_panel(train_panel, u_name)
        weeks = sorted(u_df['snapshot_date'].unique())
        folds = _folds(weeks)
        if not folds:
            continue
            
        base_cols = [c for c in BASE_FEATURES if c in u_df.columns and c != 'pullback_v_is_dry'] + [c for c in TECH_FEATURES if c in u_df.columns]
        
        for enc_name in ('symmetric', 'reward_only', 'ignored'):
            mod_df = u_df.copy()
            if enc_name == 'symmetric':
                mod_df['pullback_v_is_dry'] = mod_df['pullback_v_is_dry'].map({1.0: 1.0, 0.0: -1.0}).fillna(0.0)
                feat_cols = base_cols + ['pullback_v_is_dry']
            elif enc_name == 'reward_only':
                mod_df['pullback_v_is_dry'] = mod_df['pullback_v_is_dry'].map({1.0: 1.0, 0.0: 0.0}).fillna(0.0)
                feat_cols = base_cols + ['pullback_v_is_dry']
            elif enc_name == 'ignored':
                feat_cols = base_cols
                
            for h in (1, 2, 4):
                label_col = f'w{h}_return_pct'
                stop_col = f'w{h}_stop8'
                
                oof_picks = []
                for tr_w, va_w in folds:
                    tr = mod_df[mod_df.snapshot_date.isin(tr_w) & mod_df[label_col].notna()]
                    va = mod_df[mod_df.snapshot_date.isin(va_w)]
                    if len(tr) < 20 or va.empty:
                        continue
                    xt, y, xs = _prep_xy(tr, va, feat_cols, label_col)
                    m = _create_model('ridge')
                    m.fit(xt, y)
                    va_pred = va.copy()
                    va_pred['score'] = m.predict(xs)
                    for _, g in va_pred.groupby('snapshot_date'):
                        top = g.sort_values('score', ascending=False).head(3)
                        oof_picks.append(top)
                        
                if oof_picks:
                    picks_df = pd.concat(oof_picks)
                    rets = pd.to_numeric(picks_df[label_col], errors='coerce').dropna()
                    stops = np.where(picks_df[stop_col].isna(), False, picks_df[stop_col].values).astype(bool) if stop_col in picks_df.columns else np.zeros(len(picks_df), dtype=bool)
                    
                    rows.append({
                        'universe': u_name,
                        'encoding': enc_name,
                        'horizon': f'W{h}',
                        'train_oof_picks': len(picks_df),
                        'mean_return_pct': float(rets.mean()) if len(rets) > 0 else np.nan,
                        'median_return_pct': float(rets.median()) if len(rets) > 0 else np.nan,
                        'cvar10_pct': compute_cvar(rets, 0.10),
                        'stop8_rate_pct': float(100.0 * stops.mean()) if len(stops) > 0 else np.nan,
                    })
                    
    df_exp = pd.DataFrame(rows)
    train_out_dir = OUT / 'train'
    train_out_dir.mkdir(parents=True, exist_ok=True)
    df_exp.to_csv(train_out_dir / 'pullback_encoding_experiment.csv', index=False)
    return df_exp


def run_lane_diagnostic(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Analyze historical unconditional lane outcome associations (Exploratory Hypothesis)."""
    if panel is None:
        panel = get_full_panel()
        
    from dashboard.skill_industry_eps_known import reasoned_item, is_review_universe
    
    rows = []
    for row_idx, row in panel.iterrows():
        if not is_review_universe(row):
            continue
        item = reasoned_item(row, row_idx)
        d = {
            'snapshot_date': str(row.get('snapshot_date')),
            'code': str(row.get('code')),
            'lane': item.lane,
            'entry_status': item.entry_status,
            'b0_eligible': row.get('b0_eligible', False),
            'w1_return_pct': row.get('w1_return_pct'),
            'w2_return_pct': row.get('w2_return_pct'),
            'w4_return_pct': row.get('w4_return_pct'),
            'w4_stop8': row.get('w4_stop8'),
        }
        rows.append(d)
        
    df_lanes = pd.DataFrame(rows)
    summary = []
    for lane, g in df_lanes.groupby('lane'):
        w4 = pd.to_numeric(g['w4_return_pct'], errors='coerce').dropna()
        stops = np.where(g['w4_stop8'].isna(), False, g['w4_stop8'].values).astype(bool) if 'w4_stop8' in g.columns else np.zeros(len(g), dtype=bool)
        summary.append({
            'lane': lane,
            'count': len(g),
            'w4_mature_count': len(w4),
            'w4_mean': float(w4.mean()) if len(w4) > 0 else np.nan,
            'w4_median': float(w4.median()) if len(w4) > 0 else np.nan,
            'w4_cvar10': compute_cvar(w4, 0.10),
            'w4_stop8_rate_pct': float(100.0 * stops.mean()) if len(stops) > 0 else np.nan,
            'diagnostic_note': 'Exploratory association only; not independent causal evidence.'
        })
    df_sum = pd.DataFrame(summary).sort_values('w4_median', ascending=False)
    df_sum.to_csv(OUT / 'lane_monotonicity_diagnostic.csv', index=False)
    return df_sum

