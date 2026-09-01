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


def run_pullback_dry_policy_experiment(
    panel: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    write_output: bool = True,
) -> pd.DataFrame:
    """Controlled Train-only comparison of B0 dry-up policies (symmetric, reward_only, ignored)
    across selector variants (distinct_1, pure_top3, max_2_per_ind) directly on production rule semantics
    using strictly mature portfolio-level weekly metrics."""
    if panel is None:
        panel = get_full_panel()
        
    train_panel = panel[panel.snapshot_date <= TRAIN_END].copy()
    from .b0_challengers import ALL_DRY_POLICIES, ALL_SELECTORS, select_b0_variant, challenger_id
    from .evaluate import compute_weekly_metrics
    
    # Run over each snapshot_date in Train
    snapshots = sorted(train_panel['snapshot_date'].unique())
    
    rows = []
    for dp in ALL_DRY_POLICIES:
        for sel in ALL_SELECTORS:
            cid = challenger_id(dp, sel)
            picks_list = []
            
            for s_date in snapshots:
                snap_pool = train_panel[train_panel.snapshot_date == s_date]
                if snap_pool.empty:
                    continue
                selected_cands = select_b0_variant(snap_pool, dry_policy=dp, selector=sel, limit=TOP_N)
                if not selected_cands:
                    continue
                sel_codes = [c.code for c in selected_cands]
                matched = snap_pool[snap_pool.code.isin(sel_codes)].copy()
                for rank_i, code in enumerate(sel_codes, 1):
                    matched.loc[matched.code == code, 'model_pick_order'] = rank_i
                matched['selector_id'] = cid
                matched['segment'] = 'train_rule_eval'
                picks_list.append(matched)
                
            if not picks_list:
                continue
                
            picks_df = pd.concat(picks_list, ignore_index=True)
            w_mets = compute_weekly_metrics(picks_df, cid, 'train_rule_eval')
            df_w_mets = pd.DataFrame(w_mets)
            
            for h in (1, 2, 4):
                h_str = f'W{h}'
                sub_w = df_w_mets[df_w_mets.horizon == h_str] if not df_w_mets.empty and 'horizon' in df_w_mets.columns else pd.DataFrame()
                
                if sub_w.empty:
                    rets = pd.Series(dtype=float)
                    stops = pd.Series(dtype=float)
                else:
                    rets = pd.to_numeric(sub_w['return_pct'], errors='coerce').dropna()
                    stops = pd.to_numeric(sub_w['stop_rate'], errors='coerce').dropna()
                
                rows.append({
                    'dry_policy': dp,
                    'selector': sel,
                    'challenger_id': cid,
                    'horizon': h_str,
                    'train_picks_count': len(picks_df),
                    'mature_weeks': len(rets),
                    'mean_return_pct': float(round(rets.mean(), 4)) if len(rets) > 0 else np.nan,
                    'median_return_pct': float(round(rets.median(), 4)) if len(rets) > 0 else np.nan,
                    'cvar10_pct': float(round(compute_cvar(rets, 0.10), 4)),
                    'stop8_rate_pct': float(round(100.0 * stops.mean(), 2)) if len(stops) > 0 else np.nan,
                })
                
    df_exp = pd.DataFrame(rows)
    if write_output:
        train_out_dir = output_dir or (OUT / 'train')
        train_out_dir.mkdir(parents=True, exist_ok=True)
        df_exp.to_csv(train_out_dir / 'pullback_dry_policy_experiment.csv', index=False)
        # Also save as pullback_encoding_experiment.csv for backwards compatibility
        df_exp.to_csv(train_out_dir / 'pullback_encoding_experiment.csv', index=False)
    return df_exp


# Backwards compatibility alias
run_pullback_encoding_experiment = run_pullback_dry_policy_experiment


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

