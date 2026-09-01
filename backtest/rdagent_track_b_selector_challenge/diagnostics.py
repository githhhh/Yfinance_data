from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .config import *
from .panel import get_full_panel, get_universe_panel


def _cvar(series: pd.Series, q: float = 0.10) -> float:
    vals = series.dropna().to_numpy(dtype=float)
    if len(vals) < 3:
        return np.nan
    cutoff = np.quantile(vals, q)
    tail = vals[vals <= cutoff]
    return float(np.mean(tail)) if len(tail) > 0 else np.nan


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
        # Categorize pullback_v_is_dry state
        # 1.0 -> 'True', 0.0 -> 'False', NaN -> 'Missing'
        dry_raw = u_df['pullback_v_is_dry']
        state = pd.Series('Missing', index=u_df.index)
        state[dry_raw == 1.0] = 'True'
        state[dry_raw == 0.0] = 'False'
        
        # Also compute encodings
        enc_sym = dry_raw.map({1.0: 1.0, 0.0: -1.0}).fillna(0.0)
        enc_reward = dry_raw.map({1.0: 1.0, 0.0: 0.0}).fillna(0.0)

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
                
                rets = pd.to_numeric(sub.get(ret_col), errors='coerce').dropna()
                stops = np.where(sub[stop_col].isna(), False, sub[stop_col].values).astype(bool) if stop_col in sub.columns else np.zeros(len(sub), dtype=bool)
                
                rec[f'w{h}_count'] = len(rets)
                rec[f'w{h}_mean_pct'] = round(float(rets.mean()), 3) if len(rets) > 0 else np.nan
                rec[f'w{h}_median_pct'] = round(float(rets.median()), 3) if len(rets) > 0 else np.nan
                rec[f'w{h}_p10_pct'] = round(float(np.quantile(rets, 0.10)), 3) if len(rets) >= 5 else np.nan
                rec[f'w{h}_p90_pct'] = round(float(np.quantile(rets, 0.90)), 3) if len(rets) >= 5 else np.nan
                rec[f'w{h}_cvar10_pct'] = round(_cvar(rets, 0.10), 3)
                rec[f'w{h}_stop8_rate_pct'] = round(100.0 * float(stops.mean()), 2) if len(stops) > 0 else np.nan
                
                # Profit20 before stop8 indicator if available
                if 'profit20_hit' in sub.columns:
                    p20 = np.where(sub['profit20_hit'].isna(), False, sub['profit20_hit'].values).astype(bool)
                    rec['profit20_rate_pct'] = round(100.0 * float(p20.mean()), 2)

            # Signal type breakdown
            if 'ibd_candidate_rule' in sub.columns:
                rules = sub['ibd_candidate_rule'].value_counts(normalize=True).to_dict()
                rec['top_signal_type'] = sub['ibd_candidate_rule'].mode().iloc[0] if not sub['ibd_candidate_rule'].empty else 'none'
                rec['ceiling_pct'] = round(100.0 * rules.get('ceiling', 0.0), 1)
                rec['ceiling_pullback_pct'] = round(100.0 * rules.get('ceiling_pullback', 0.0), 1)
                rec['pivot_pct'] = round(100.0 * rules.get('pivot', 0.0), 1)
                rec['ma10_touch_pct'] = round(100.0 * rules.get('ma10_touch_confirm', 0.0), 1)

            records.append(rec)

    df_out = pd.DataFrame(records)
    OUT.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT / 'pullback_dry_diagnostic.csv', index=False)
    return df_out


def run_lane_diagnostic(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Analyze whether B0 lanes have empirical monotonicity."""
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
            'w4_count': len(w4),
            'w4_mean': float(w4.mean()) if len(w4) > 0 else np.nan,
            'w4_median': float(w4.median()) if len(w4) > 0 else np.nan,
            'w4_cvar10': _cvar(w4, 0.10),
            'w4_stop8_rate_pct': float(100.0 * stops.mean()) if len(stops) > 0 else np.nan,
        })
    df_sum = pd.DataFrame(summary).sort_values('w4_median', ascending=False)
    df_sum.to_csv(OUT / 'lane_monotonicity_diagnostic.csv', index=False)
    return df_sum
